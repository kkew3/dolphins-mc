"""
Defines the video dataset class and related functions.
"""
import json
import os
import sys
from collections import deque
from operator import methodcaller

from filelock import FileLock
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
import torchvision.transforms as trans


# str.format template
# example: DEFAULT_DATASET_ROOTNAME_TMPL.format(8, (8, 0, 0))
DEFAULT_DATASET_ROOTNAME_TMPL = 'CH{0:0>2}-{1[0]:0>2}_{1[1]:0>2}_{1[2]:0>2}'

# npz file containing mean and std info of the dataset
NORMALIZATION_INFO_FILE = 'nml-stat.npz'



def get_normalization_stats(root):
    """
    Returns the normalization statistics (mean, std) in preprocessing step
    when loading the dataset. The normalization data presents in a ``npz``
    file ``$NORMALIZATION_INFO_FILE`` under the dataset root directory.

    Usage example::

        .. code-block::

            import torchvision.transforms as trans
            normalize = trans.Normalize(*get_normalization_stats(root))

    :param root: the root directory of the video dataset
    :raise IOError: if the ``$NORMALIZATION_INFO_FILE`` is not found under
           ``root``, which may due to spelling error in ``root`` or the file
           is absent as a matter of fact. For the latter case, compute the
           normalization statistics (using
           ``$PROJECT_HOME/bin/compute-perch-stat.py``), put the result file
           to the root directory, before calling this function.
    :return: the mean and std
    :rtype: Tuple[Tuple[float], Tuple[float]]
    """
    data = np.load(os.path.join(root, NORMALIZATION_INFO_FILE))
    mean = tuple(map(float, data['mean']))
    std = tuple(map(float, data['std']))
    return mean, std

def prepare_dataset_root(cam_channel, video_index):
    """
    A convenient function to get the root directory of dataset of which the
    name is named after DEFAULT_DATASET_ROOTNAME_TMPL and resides under
    PYTORCH_DATA_HOME.

    :param cam_channel: the camera channel ID
    :type cam_channel: int
    :param video_index: the video index, a 3-tuple of integers
    :type video_index: Tuple[int, int, int]
    :return: the root directory
    :rtype: str
    """
    root = os.path.join(os.environ['PYTORCH_DATA_HOME'],
                        DEFAULT_DATASET_ROOTNAME_TMPL.format(cam_channel,
                                                             video_index))
    if not os.path.isdir(root):
        raise ValueError('root "{}" is not an existing directory'
                         .format(root))
    return root


def get_h5filename_from_root(root):
    return os.path.join(root, os.path.basename(root) + '.h5')


class AbstractH5Dataset(Dataset):
    """
    An abstract class representing an HDF5-based dataset.
    """
    def __init__(self, root):
        self.root = root
        filename = get_h5filename_from_root(root)
        self.h5file = h5py.File(filename, 'r')

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self.h5file.close()


class VideoDataset(AbstractH5Dataset):
    """
    Represents an HDF5 video dataset. The ith index of the dataset returns the
    PIL image of the ith frame. When used with ``torch.utils.data.DataLoader``,
    at each batch, the dataloader iterator returns a batch of frame tensor.
    The dataset is unsupervised, thus no "labels"/"targets" will be returned.

    Example usage::

        .. code-block::

            import torchvision.transforms as trans
            from torch.utils.data import DataLoader
            with VideDataset(root, transform=trans.ToTensor()) as dset:
                dataloader = DataLoader(dset, shuffle=False, batch_size=2)
                for frames in dataloader:
                    pass
    """

    H5DATASET_NAME = 'frames'

    def __init__(self, root, transform=None):
        """
        :param root: the dataset root directory
        :param transform: the transformation to perform after loading the
               frames. A typical choice is
               ``torchvision.transforms.Totensor()`` followed by ``Normalize``.
        """
        super(VideoDataset, self).__init__(root)
        self.frame_data = self.h5file.get(type(self).H5DATASET_NAME)
        self.transform = transform
        self.access_lock = FileLock(os.path.join(root, os.path.basename(root) + '.access.lock'))

    def __len__(self):
        return self.frame_data.len()

    def __getitem__(self, index):
        """
        Error may be raised if the underlying HDF5 chunk has been corrupted.
        The error raising is maintained by HDF5's API.

        :param index: the index to load; ``index`` should not be a slice
        :type index: int
        """
        try:
            assert not isinstance(index, slice)
            with self.access_lock:
                # According to HDF Group, the HDF5 file can be accessed by only
                # one process at a time; otherwise undefined behavior will
                # occur
                frame = np.array(self.frame_data[index], dtype=np.uint8)
            frame = np.transpose(frame, (1, 2, 0))  # of dimension HWC
            if self.transform is not None:
                frame = self.transform(frame)
            return frame
        except IOError:
            sys.stderr.write('IOError raised when index={}\n'.format(index))
            raise

    @property
    def attrs(self):
        return self.h5file.require_group('/').attrs

    @property
    def shape(self):
        return self.frame_data.shape


class PairedVideoDataset(Dataset):
    """
    Represents a pairing of two HDF5-based video datasets, as built by
    ``${repo-root}/bin/build-video-dataset.py``. The dataset returns a pair of
    frame tensor when requesting the frame ID (indexed from zero).
    """

    def __init__(self, dataset1, dataset2):
        """
        :param dataset1: one ``VideoDataset`` instance representing one video
        :type dataset1: VideoDataset
        :param dataset2: another ``VideoDataset`` instance representing another
               video
        :type dataset2: VideoDataset
        """
        self.dataset_pair = dataset1, dataset2

    def __len__(self):
        return min(map(len, self.dataset_pair))

    def __getitem__(self, index):
        return tuple(ds[index] for ds in self.dataset_pair)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for ds in self.dataset_pair:
            ds.__exit__()


class VideoDatasetWriter(object):
    r"""
    Creates a video dataset from numpy array. The created dataset will have
    the following directory hierarchy::

        root/
        |- root.h5
        |  \- (Dataset) frames
        |     \- attrs
        \- (optional, not created by this class) nml-stat.npz

    where metadata can be accessed from ``frames.attrs``.
    """

    def __init__(self, root, shape, **meta):
        """
        :param root: the dataset root directory
        :param shape: the shape of a single frame (numpy array), without the
               prepending ``batch_size`` dimension. The shape should be
               arranged in (num_channels, height, width) manner
        :param meta: meta info of the dataset
        """
        filename = get_h5filename_from_root(root)
        self.shape = shape
        self.h5file = h5py.File(filename, 'w')
        self.frame_data = self.h5file.create_dataset(VideoDataset.H5DATASET_NAME,
                                                     (0,) + shape,
                                                     maxshape=(None,) + shape,
                                                     chunks=True,
                                                     compression='gzip',
                                                     fletcher32=True)
        for k, v in meta.items():
            self.h5file.require_group('/').attrs[k] = v

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def write_batch(self, data):
        """
        :param data: the batch of data to write; it will be asserted that
               ``data.shape[1:] == self.shape``
        :type data: numpy.ndarray
        """
        if not isinstance(data, np.ndarray):
            raise TypeError('Invalid data type: {}'.format(type(data)))
        if data.shape[1:] != self.shape:
            raise ValueError('Invalid data shape: {}; expected: {}'
                             .format(data.shape, (data.shape[0],)+self.shape))
        if data.shape[0] > 0:
            self.frame_data.resize(self.frame_data.len() + data.shape[0], axis=0)
            self.frame_data[-data.shape[0]:] = data

    def close(self):
        self.h5file.close()


class _SegmentWrapper(Dataset):
    def __init__(self, segment):
        self.segment = segment
    def __len__(self):
        return self.segment.len()
    def __getitem__(self, index):
        return self.segment[index]

class VideoSegmentDataset(AbstractH5Dataset):
    """
    Represents an HDF5 video segments dataset. Unlike ``VideoDataset`` which
    only holds one HDF5 Dataset instance, a ``VideoSegmentDataset`` holds K
    HDF5 Dataset instances, each corresponding to a video segment.
    """
    def __init__(self, root, transforms=None, segments=None):
        """
        :param root: the dataset root directory
        :type root: str
        :param transform: the transformation to perform after loading the
               frames. A typical choice is
               ``torchvision.transforms.Totensor()`` followed by ``Normalize``.
        :param segments: None to concatenate all segments as if they were a
               single video; otherwise, specify the name of the segment to
               read; or a list of names to concatenate
        :type segments: Optional[Union[str, Sequence[str]]]
        """
        super(VideoSegmentDataset, self).__init__(root)
        self.transforms = transforms
        if isinstance(segments, str):
            segments = [segments]

        def in_segments(ds_name):
            if segments is None:
                return True
            return ds_name in segments

        self.segment_data = ConcatDataset(list(map(_SegmentWrapper,
                                                   map(self.h5file.get,
                                                       filter(in_segments,
                                                              self.h5file)))))

    def __len__(self):
        return len(self.segment_data)

    def __getitem__(self, index):
        """
        Error may be raised if the underlying HDF5 chunk has been corrupted.
        The error raising is maintained by HDF5's API.

        :param index: the index to load; ``index`` should not be a slice
        :type index: int
        """
        assert not isinstance(index, slice)
        frame = np.array(self.frame_data[index], dtype=np.uint8)
        frame = np.transpose(frame, (1, 2, 0))  # of dimension HWC
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


class VideoSegmentDatasetWriter(object):
    def __init__(self, root, shape, **meta):
        """
        :param root: the dataset root directory
        :param shape: the shape of a single frame (numpy array), without the
               prepending ``batch_size`` dimension. The shape should be
               arranged in (num_channels, height, width) manner
        :param meta: meta info of the dataset
        """
        filename = get_h5filename_from_root(root)
        self.h5file = h5py.File(filename, 'w')
        self.shape = shape
        for k, v in meta.items():
            self.h5file.require_group('/').attrs[k] = v

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def write_batch(self, segment_name, data):
        """
        :param segment_name: the name of the video segment (HDF5 Dataset) to
               write to
        :param data: the batch of data to write; it will be asserted that
               ``data.shape[1:] == self.shape``
        :type data: numpy.ndarray
        """
        if not isinstance(data, np.ndarray):
            raise TypeError('Invalid data type: {}'.format(type(data)))
        if data.shape[1:] != self.shape:
            raise ValueError('Invalid data shape: {}; expected: {}'
                             .format(data.shape, (data.shape[0],)+self.shape))
        if segment_name not in self.h5file:
            self.h5file.create_dataset(segment_name,
                                       (0,) + self.shape,
                                       maxshape=(None,) + self.shape,
                                       chunks=True,
                                       compression='gzip',
                                       fletcher32=True)
        frame_data = self.h5file.get(segment_name)
        if data.shape[0] > 0:
            frame_data.resize(frame_data.len() + data.shape[0], axis=0)
            frame_data[-data.shape[0]:] = data

    def close(self):
        self.h5file.close()
