"""
Defines representing the video dataset class and related functions.

@deprecated Please use ``vhdata`` instead.
"""

import json
import multiprocessing
import os
import re
import hashlib
from collections import deque

from cachetools import LRUCache
# noinspection PyPackageRequirements
import hickle
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as trans
from PIL import Image

import utils


DATASET_NOT_FOUND = 0x01
DATASET_CORRUPTED = 0x02

DATASET_HASH_ALGORITHM = 'sha1'

# str.format template
# example: DEFAULT_DATASET_ROOTNAME_TMPL.format(8, (8, 0, 0))
DEFAULT_DATASET_ROOTNAME_TMPL = 'CH{0:0>2}-{1[0]:0>2}_{1[1]:0>2}_{1[2]:0>2}'

NORMALIZATION_INFO_FILE = 'nml-stat.npz'

def get_normalization_stats(root):
    """
    Returns the normalization statistics (mean, std) in preprocessing step
    when loading the dataset. The normalization data presents in a ``npz``
    file ``$NORMALIZATION_INFO_FILE`` under the dataset root directory.

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

def compute_file_integrity(filename):
    """
    Compute integrity ($DATASET_HASH_ALGORITHM) of a single file.

    :param filename: the filename
    :return: the checksum line, which is a tuple of form
             ``(base_filename, expected_hex)``
    """
    hashbuff = hashlib.new(DATASET_HASH_ALGORITHM)
    with open(filename, 'rb') as infile:
        for block in iter(lambda: infile.read(1024 * 1024), b''):
            hashbuff.update(block)
    expected_hex = hashbuff.hexdigest()
    return os.path.basename(filename), expected_hex

def check_file_integrity(checksum_line):
    """
    Check integrity ($DATASET_HASH_ALGORITHM) of a single file.

    :param checksum_line: a tuple of form ``(filename, expected_hex)``,
           where ``filename`` is the absolute path of the file to check, and
           ``expected_hex`` is the hex string of the expected SHA-1 hash
           of the block
    :return: 0 if the checking passes, and the bit-wise error signal (defined
             as ``DATASET_NOT_FOUND`` and ``DATASET_CORRUPTED``) otherwise
    :rtype: int
    """
    hashbuff = hashlib.new(DATASET_HASH_ALGORITHM)
    filename, expected_hex = checksum_line
    try:
        with open(filename, 'rb') as infile:
            for block in iter(lambda: infile.read(1024 * 1024), b''):
                hashbuff.update(block)
        if hashbuff.hexdigest() != expected_hex:
            return DATASET_CORRUPTED
        return 0
    except IOError:
        return DATASET_NOT_FOUND | DATASET_CORRUPTED

class VideoDataset(Dataset):
    """
    Represents a HDF5 block-based video dataset, as built by
    ``${repo-root}/bin/build-video-dataset.py``. The dataset returns the
    corresponding frame tensor when requesting the frame ID (indexed from zero).
    The dataset is not supervised.
    """
    BLOCKFILE_TMPL = 'B{}.h5'  # usage: BLOCKFILE_TMPL.format(block_id)
    BLOCKFILE_PAT = re.compile(r'^B(\d+)\.h5$')

    def __init__(self, root, max_block_cached=2, transform=None):
        """
        :param root: the dataset root directory
        :type root: str
        :param max_block_cached: maximum number of blocks to be cached in
               memory; must be at least 1
        :type max_block_cached: int
        :param transform: the transforms to perform after loading the frames
        """
        max_block_cached = int(max_block_cached)
        if max_block_cached < 1:
            raise ValueError('max_block_cached (got {}) must be at least 1'
                             .format(max_block_cached))
        video_name = os.path.basename(root)
        metafile = os.path.join(root, video_name + '.json')
        try:
            with open(metafile) as infile:
                self.metainfo = json.load(infile)
            self.root = root
            self._cached_blocks = LRUCache(max_block_cached)
        except IOError:
            raise ValueError('meta file "{}" not found under root "{}"'
                             .format(video_name + '.json', root))
        self.transform = transform
        self.check_integrity()

    def __len__(self):
        return self.metainfo['total_frames']

    def __getitem__(self, frame_id):
        """
        :param frame_id: the frame ID
        :type frame_id: int
        :return: the frame tensor, of dimension [1 x C x H x W]
        """
        if frame_id >= len(self) or frame_id < 0:
            raise IndexError()
        bid, local_frame_id = self.locate_block(frame_id)
        if bid not in self._cached_blocks:
            self._cached_blocks[bid] = self.load_block(
                    self.blockfile_name_from_id(bid))
        frame = self._cached_blocks[bid][local_frame_id]
        frame = Image.fromarray(frame)
        if self.transform is not None:
            frame = self.transform(frame)
        return frame

    def check_integrity(self, num_workers=4):
        video_name = os.path.basename(self.root)
        checksumfile = os.path.join(self.root, video_name + '.sha1')
        checksums = dict()
        with open(checksumfile) as infile:
            for line in infile:
                expected, block_filename = line.strip().split(' *')
                checksums[os.path.join(self.root, block_filename)] = expected
        if num_workers <= 1:
            # fail fast check
            for checksum_line in checksums.items():
                errno = check_file_integrity(checksum_line)
                if errno == DATASET_CORRUPTED | DATASET_NOT_FOUND:
                    raise RuntimeError('Dataset not found or corrupted')
                elif errno == DATASET_CORRUPTED:
                    raise RuntimeError('Dataset corrupted')
        else:
            # fail fast multiprocessing is too difficult to implement ...
            with utils.poolcontext(num_workers) as pool:
                errnos = pool.map(check_file_integrity, checksums.items())
                if DATASET_NOT_FOUND | DATASET_CORRUPTED in errnos:
                    raise RuntimeError('Dataset not found or corrupted')
                elif DATASET_CORRUPTED in errnos:
                    raise RuntimeError('Dataset corrupted')

    def locate_block(self, frame_id):
        """
        Locate the underlying frame tensor given the frame ID.

        :param frame_id: the frame ID (indexed from 0)
        :return: the (block ID, the local frame ID within the block)
        """
        bs = self.metainfo['block_size']
        return frame_id // bs, frame_id % bs

    def blockfile_name_from_id(self, bid):
        """
        :param bid: the block ID
        :return: the block filename
        """
        return os.path.join(self.root, type(self).BLOCKFILE_TMPL.format(bid))

    @staticmethod
    def load_block(block_file):
        with open(block_file) as infile:
            with utils.suppress_stdout():
                # data_block, numpy array of shape: [B x 3 x H x W]
                data_block = hickle.load(infile)
                # data_block, numpy array of shape: [B x H x W x 3]
                data_block = np.transpose(data_block, (0, 2, 3, 1))  # to HWC
                return data_block

class PairedVideoDataset(Dataset):
    """
    Represents a pairing of two HDF5 block-based video datasets, as built by
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

    def __getitem__(self, item):
        return tuple(ds[item] for ds in self.dataset_pair)


class InconsistentBlockSizeError(BaseException):
    def __init__(self, *args):
        """
        >>> print(InconsistentBlockSizeError(3, 2, 1))
        'Invalid write attempt (history block size): 3, 2, 1'
        """
        BaseException.__init__(self, 'Invalid write attempt (history block '
                               'size): {}'.format(', '.join(map(str, args))))

class VideoDatasetWriter(object):
    r"""
    Create video dataset from numpy arrays.
    The created dataset will have the following directory hierarchy::

        dataset_root/
        |- video_name_0/
        |  |- video_name_0.json
        |  |- video_name_0.sha1
        |  |- (optional, not created by this class) nml-stat.npz
        |  |- B0.h5
        |  |- B1.h5
        |  |- ...
        |  \- BM.h5
        |- ...
        |- video_name_x/
        |  |- ...
        \  \- BN.h5

    where the json file contains metainfo of the dataset; the sha1 file
    contains integrity info of the data blocks "Bx.h5"; and "Bx.h5" are data
    blocks in compressed HDF5 format. The data will be converted to numpy array
    before writing. The number of frames included in each data block should be
    consistent, except for the last block that's allowed to hold no larger than
    the block size.
    """
    def __init__(self, root, meta=None):
        """
        :param root: the root directory of the dataset
        :param meta: meta info of the dataset to create other than 'block_size'
               (number of frames for each block) and 'total_frames' (total
               number of frames of the underlying video)
        """
        self.root = root
        self.meta = dict() if meta is None else meta
        self.bid = 0  # the next block ID to use
        self.block_size_cache = deque([None, None], maxlen=2)
        self.total_frames = 0

    def write_block(self, data):
        """
        :param data: the data block to write, where ``data.shape[0]`` is
               treated as the number of frames in the block, i.e. the block
               size
        :type data: numpy.ndarray
        :raise InconsistentBlockSizeError: let current call of this function
               be the k-th invocation; if the block size of the (k-2)-th
               is different from the (k-1)-th's, this error will be raised
        """
        assert isinstance(data, np.ndarray)
        self._ensure_blocksize_consistency()
        tofile = self._get_current_blockfile()
        with open(tofile, 'w') as outfile:
            hickle.dump(data, outfile, compression='gzip')
        self.bid += 1
        self.block_size_cache.append(data.shape[0])
        self.total_frames += data.shape[0]

    def compute_checksum(self, num_workers=4):
        """
        Compute checksums.

        :param num_workers: number of processes to do file hashing, default
               to 4
        :return: the checksum lines
        :rtype: List[Tuple[str, str]]
        """
        block_files = map(lambda name: os.path.join(self.root, name),
                          filter(VideoDataset.BLOCKFILE_PAT.match,
                                 os.listdir(self.root)))
        if num_workers <= 1:
            checksum_lines = map(compute_file_integrity, block_files)
        else:
            with utils.poolcontext(num_workers) as pool:
                checksum_lines = pool.map(compute_file_integrity, block_files)
        return list(checksum_lines)

    def wrap_up(self, num_workers=4):
        """
        Compute integrities and write meta-info file.

        :param num_workers: number of processes to do file hashing, default 4
        :type num_workers: int
        """
        self._ensure_blocksize_consistency(finalizing=True)
        checksum_lines = self.compute_checksum()
        video_name = os.path.basename(self.root)
        checksumfile = os.path.join(self.root,
                                    '.'.join([video_name,
                                              DATASET_HASH_ALGORITHM]))
        # TODO

    @property
    def block_size(self):
        return self.block_size_cache[0]

    def _get_current_blockfile(self):
        return os.path.join(self.root,
                            VideoDataset.BLOCKFILE_TMPL.format(self.bid))

    def _ensure_blocksize_consistency(self, finalizing=False):
        """
        :param finalizing: True to indicate all writings have been completed
        :type finalizing: bool
        :raise InconsistentBlockSizeError: if inconsistent block size occurs
        """
        prev2_write, prev1_write = tuple(self.block_size_cache)
        if prev2_write is not None and prev1_write is not None:
            if not finalizing and prev2_write != prev1_write:
                raise InconsistentBlockSizeError(prev2_write, prev1_write)
            elif finalizing and prev2_write < prev1_write:
                raise InconsistentBlockSizeError(prev2_write, prev1_write)

def prepare_video_dataset(channel, index, rootdir=None, normalized=False,
                          additional_transforms=None):
    """
    Load video dataset from ``root``. This is a convenient function if the
    root directory of the target dataset is named after
    ``$DEFAULT_DATASET_ROOTNAME_TMPL``.

    :param channel: the channel, i.e. the two-digit number after "CH"; accept
           int or str. If int, it should be nonnegative and less than 100; if
           str, it should be convertible to int and with length 2. No argument
           type check will be made.
    :type channel: Union[int, str]
    :param index: the index string, may be a string of format
           ``$DEFAULT_DATASET_ROOTNAME_TMPL``,
           or a 3-tuple of string of format ``%2d``, or a 3-tuple of ints. The
           value requirement for the latter two cases is the same as in option
           ``channel``. No argument type check will be made.
    :type index: Union[str, Tuple[int, int, int], Tuple[str, str, str]]
    :param rootdir: the parent directory of the root directory, default to
           ``os.environ['PYTORCH_DATA_HOME']``
    :type rootdir: Optional[str]
    :param normalized: True to find ``$NORMALIZATION_INFO_FILE`` file under the
           dataset root directory, load normalization statistics from it, and
           normalize the loaded data after converted to tensor; default to
           False
    :param additional_transforms: additional transforms other than ToTensor
           (and Normalize if ``normalized``); should be a list of
           transformations
    :type additional_transforms: Sequence
    :return: the video dataset instance
    :rtype: VideoDataset
    """
    if isinstance(index, str):
        index = tuple(index.split('_'))
    rootname = DEFAULT_DATASET_ROOTNAME_TMPL.format(channel, index)
    if rootdir is None:
        rootdir = os.environ['PYTORCH_DATA_HOME']
    root = os.path.join(rootdir, rootname)
    transforms = [trans.ToTensor()]
    if normalized:
        mean, std = get_normalization_stats(root)
        mean = tuple(map(float, mean))
        std = tuple(map(float, std))
        transforms.append(trans.Normalize(mean=mean, std=std))
    if len(additional_transforms):
        transforms.extend(additional_transforms)
    dset = VideoDataset(root, transform=trans.Compose(transforms))
    return dset
