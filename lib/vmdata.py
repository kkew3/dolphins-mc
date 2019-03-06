"""
Video dataset based on memory mapped file and gzip. This implementation is
potentially much faster than ``vhdata`` as it allows concurrent access, and
memory-mapped file is inherently faster. However, since it consumes much larger
memory and disk when loading data.


Dataset structure
-----------------

Directory structure::

    .. code-block:: plain

        video_name/
        |- tmp/                           # temporarily cached uncompressed data
        |  |- data_batch_x
        |  |- data_batch_y
        |  |- ...
        |- data_batch_0.gz
        |- data_batch_1.gz
        |- ...
        |- data_batch_n.gz
        |- video_name.json                # metadata, see below
        |- video_name.sha1                # sha1 of data batches
        |- video_name.access0.lock        # mutex file lock for data_batch_0.gz
        | -video_name.access1.lock        # mutex file lock for data_batch_1.gz
        |- ...
        |- video_name.accessn.lock


Metadata
--------

Content of the JSON file::

    .. code-block:: json

        {
            "lens": [1000, 1000, ..., 216],       # lens[i] is len(data_batch_i)
            "resolution": [480, 704],             # height, width
            "channels" 3,                         # the number of color channels
            "dimension": "NHWC",                  # dimensions
            "dtype": "uint8",                     # data type of data batches

            "others": ...                         # supported by subclasses of
                                                  # ``VideoDataset``
        }
"""

import bisect
import gzip
import hashlib
import json
import operator as op
import os
import re
import shutil
import logging
from typing import Iterable, Iterator, List, Tuple, Union

import cv2
import numpy as np
from cachetools import LRUCache
from filelock import FileLock
from torch.utils.data import Dataset

import utils
from utils import loggername as _l

_cd = os.path.dirname(os.path.realpath(__file__))

HASH_ALGORITHM = 'sha1'
DATABATCH_FILENAME_PAT = re.compile(r'^data_batch_(\d+)$')
DATASETS_ROOT = os.path.join(os.path.normpath(os.environ['PYTORCH_DATA_HOME']),
                             'mmdatasets')

# npz file containing mean and std info of the dataset
NORMALIZATION_INFO_FILE = 'nml-stat.npz'
NORMALIZATION_INFO_FILE_BW = 'nml-stat_bw.npz'


def parse_checksum_file(filename):
    """
    Parse checksum file as created by ``/usr/bin/sha1sum``.

    :param filename: the checksum filename
    :type filename: str
    :return: a list ``l`` such that ``l[i]`` contains the expected hash of the
             i-th batch
    :rtype: List[str]
    """
    checksum_lines = []
    with open(filename) as infile:
        for i, line in enumerate(infile):
            expected_hex, f = line.strip().split(None, 1)
            if f.startswith('*'):
                # if startswith star, the file should be opened in binary;
                # however, this won't make any difference to our application
                # here
                f = f[1:]
            f = os.path.basename(f)
            matched = DATABATCH_FILENAME_PAT.match(f)
            if not matched:
                raise ValueError('file "{}" at line {} of "{}" does not '
                                 'conform to DATABATCH_FILENAME_PAT \'{}\''
                                 .format(f, i, filename,
                                         DATABATCH_FILENAME_PAT.pattern))
            batch_id = int(matched.group(1))
            checksum_lines.append((batch_id, expected_hex))
    checksum_lines.sort(key=lambda x: x[0])
    expected_hexes = list(map(lambda x: x[1], checksum_lines))
    return expected_hexes


def compute_file_integrity(filename):
    """
    Compute file integirty using HASH_ALGORITHM.

    :param filename: the filename to check integrity
    :type filename: str
    :return: the hex string of the file hash
    :rtype: str
    """
    hashbuf = hashlib.new(HASH_ALGORITHM)
    with open(filename, 'rb') as infile:
        for block in iter(lambda: infile.read(11048576), b''):
            hashbuf.update(block)
    return hashbuf.hexdigest()


def check_file_integrity(filename, expected_hex):
    """
    Check file integrity using HASH_ALGORITHM.

    :param filename: the filename to check integrity
    :type filename: str
    :param expected_hex: the expected hash hex
    :type expected_hex: str
    :return: True if the check passes
    :rtype: bool
    """
    actual_hex = compute_file_integrity(filename)
    return actual_hex == expected_hex


def accumulate(iterable, func=op.add):
    """
    Return running totals.

    Make an iterator that returns accumulated sums, or accumulated results of
    other binary functions (specified via the optional ``func`` argument). If
    ``func`` is supplied, it should be a function of two arguments. Elements
    of the input iterable may be any type that can be accepted as arguments
    to ``func``. (For example, with the default operation of addition, elements
    may be any addable type including Decimal or Fraction.) If the input
    ``iterable`` is empty, the output iterable will also be empty.

    Ported from Python 3.2.

    :param iterable: the iterable to accumulate
    :type iterable: Iterable
    :param func: function to reduce
    :type func: Callable
    :return: the running totals
    :rtype: Iterator

    >>> list(accumulate([1,2,3,4,5]))
    [1, 3, 6, 10, 15]
    >>> import operator
    >>> list(accumulate([1,2,3,4,5], operator.mul))
    [1, 2, 6, 24, 120]
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total


def compress_gzip(fromfile, tofile):
    """
    Compress a file into gzip file.

    :param fromfile: the file to compress
    :type fromfile: str
    :param tofile: the target gzip file
    :type tofile: str
    """
    with gzip.open(tofile, 'wb') as outfile:
        with open(fromfile, 'rb') as infile:
            shutil.copyfileobj(infile, outfile)


def extract_gzip(fromfile, tofile):
    """
    Extract file content from a gzip file to an uncompressed file.

    :param fromfile: source gzip file
    :type fromfile: str
    :param tofile: target file to write
    :type tofile: str
    """
    with gzip.open(fromfile, 'rb') as infile:
        with open(tofile, 'wb') as outfile:
            shutil.copyfileobj(infile, outfile)


def get_dset_filename_by_ext(root, ext):
    """
    Returns the filename of the file under root whose name is the same as root.

    :param root: the dataset root directory
    :type root: str
    :param ext: the extension name of the target file
    :type ext: str
    :return: the full filename of the target file
    :rtype: str
    """
    return os.path.join(root, os.path.basename(root) + ext)


class VideoDataset(Dataset):
    """
    Represents the video dataset (readonly).
    """

    def __init__(self, root, transform=None, max_mmap=1, max_gzcache=3):
        """
        :param root: the root directory of the dataset
        :type root: str
        :param transform: the transformations to be performed on loaded data
        :param max_mmap: the maximum number of memory map to keep
        :type max_mmap: int
        :param max_gzcache: the maximum number of extracted memory map files to
               keep on disk
        :type max_gzcache: int
        """
        self.root = root
        jsonfile = get_dset_filename_by_ext(root, '.json')
        with open(jsonfile) as infile:
            self.metainfo = json.load(infile)
        self.total_frames = np.sum(self.metainfo['lens'])
        self.lens_cumsum = list(accumulate(self.metainfo['lens']))
        shape = self.metainfo['resolution'] + [self.metainfo['channels']]
        self.frame_shape = tuple(shape)

        # if self.validated_batches[i] == 0, then batch i hasn't been validated
        self.validated_batches = [False] * len(self.metainfo['lens'])
        checksumfile = get_dset_filename_by_ext(root, '.' + HASH_ALGORITHM)
        self.expected_hexes = parse_checksum_file(checksumfile)

        self.root_tmp = os.path.join(root, 'tmp')
        if not os.path.isdir(self.root_tmp):
            os.mkdir(self.root_tmp)

        self.transform = transform

        max_mmap = max(1, max_mmap)
        self.mmap_cache = LRUCache(maxsize=max_mmap)
        self.gz_cache = LRUCache(maxsize=max(max_mmap, max_gzcache))
        self.max_gzcache = max(max_mmap, max_gzcache)

        # fine granularity lock for each data batch
        lockfile_tmpl = get_dset_filename_by_ext(root, '.access{}.lock')
        # note that I use the absolute to construct the file lock, so that the
        # lock will be shared by not only different processes, but also several
        # instances of this class, as long as they have been assigned the same
        # root
        self.access_locks = [FileLock(lockfile_tmpl.format(bid))
                             for bid in range(len(self.metainfo['lens']))]

        logger = logging.getLogger(_l(__name__, self, '__init__'))
        logger.info('Instantiated: root={}'.format(self.root))

    def __len__(self):
        """
        :return: the number of frames in the video
        :rtype: int
        """
        return self.total_frames

    def __getitem__(self, frame_id):
        """
        Returns a frame of dimension HWC upon the request of a frame ID.
        Note that when calling this method without using contiguous or nearly
        contiguous indices, the efficiency will be very low.

        :param frame_id: the frame index
        :return: the frame in numpy array of dimension HWC
        :rtype: np.ndarray
        """
        logger = logging.getLogger(_l(__name__, self, '__getitem__'))
        if frame_id < 0 or frame_id >= len(self):
            raise IndexError('Invalid index: {}'.format(frame_id))

        batch_id, rel_frame_id = self.locate_batch(frame_id)
        logger.debug('Waiting for lock ID {}'.format(batch_id))
        with self.access_locks[batch_id]:
            if batch_id not in self.mmap_cache:
                batchf = self.batch_filename_by_id(batch_id)
                if not os.path.isfile(batchf):
                    logger.info('Decompressing "{}"'.format(batchf))
                    extract_gzip(
                        self.batch_filename_by_id(batch_id, gzipped=True),
                        batchf)
                    assert os.path.isfile(batchf), \
                        '"{}" not found after decompressed' \
                            .format(batchf)
                if not self.validated_batches[batch_id]:
                    if not check_file_integrity(
                            batchf, self.expected_hexes[batch_id]):
                        logger.warning(
                            'File ingerity failed at "{}"; retrying'
                            .format(batchf))
                        # probably there's error with read last time; attempt
                        # to decompress again for once
                        os.remove(batchf)
                        extract_gzip(
                            self.batch_filename_by_id(batch_id, gzipped=True),
                            batchf)
                        assert os.path.isfile(batchf), \
                            '"{}" not found after decompressed' \
                                .format(batchf)
                        if not check_file_integrity(
                                batchf, self.expected_hexes[batch_id]):
                            logger.error('File integrity failed at "{}"; '
                                         'RuntimeError raised'
                                         .format(batchf))
                            raise RuntimeError('Data batch {} corrupted'
                                               .format(batch_id))
                    self.validated_batches[batch_id] = True
                    logger.info('File integrity check completed for batch {}'
                                .format(batch_id))

                # till here file "batchf" has been available
                self.gz_cache[batchf] = True

                shape = (self.metainfo['lens'][batch_id],) + self.frame_shape
                logger.debug('keys before mmap cache adjustment: {}'
                             .format(list(self.mmap_cache.keys())))
                self.mmap_cache[batch_id] = np.memmap(
                    str(batchf), mode='r', dtype=self.metainfo['dtype'],
                    shape=shape)
                logger.debug('keys after mmap cache adjustment: {}'
                             .format(list(self.mmap_cache.keys())))
        frame = np.copy(self.mmap_cache[batch_id][rel_frame_id])
        if self.transform is not None:
            frame = self.transform(frame)
        self.cleanup_unused_mmapfiles()
        return frame

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def cleanup_unused_mmapfiles(self):
        logger = logging.getLogger(
            _l(__name__, self, 'cleanup_unused_mmapfiles'))
        for filename in os.listdir(self.root_tmp):
            matched = DATABATCH_FILENAME_PAT.match(filename)
            if matched:
                batch_id = int(matched.group(1))
                with self.access_locks[batch_id]:
                    batchf = os.path.join(self.root_tmp, filename)
                    # Since len(self.gz_cache) >= len(self.mmap_cahce) and they
                    # are updated together, the latter must be a subset of the
                    # former.
                    if batchf not in self.gz_cache and \
                            len(os.listdir(self.root_tmp)) > self.max_gzcache:
                        try:
                            os.remove(batchf)
                        except OSError:
                            # due to concurrency, the file may have already been
                            # removed; due to the lock, however, no process will
                            # try to remove a file when another process is
                            # removing exactly the same file
                            pass
                        else:
                            logger.info('Decompressed batch "{}" removed'
                                        .format(batchf))

    def cleanup_all_mmapfiles(self):
        """
        Be sure to call this function only if there's no opened memory-mapped
        file. Usually this function is unnecessary unless the user want to save
        some disk space.
        """
        logger = logging.getLogger(_l(__name__, self, 'cleanup_all_mmapfiles'))
        if os.path.isdir(self.root_tmp):
            shutil.rmtree(self.root_tmp)
        if not os.path.isdir(self.root_tmp):
            os.mkdir(self.root_tmp)
        logger.info('All decompressed batches removed')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_mmap()

    def release_mmap(self):
        """
        Release all memory mapped dataset.
        """
        logger = logging.getLogger(_l(__name__, self, 'release_mmap'))
        keys = list(self.mmap_cache.keys())
        for k in keys:
            del self.mmap_cache[k]
        logger.info('All mmap released')

    def locate_batch(self, frame_id):
        """
        Locate the data batch the specified frame is stored.

        :param frame_id: the frame ID
        :type frame_id: int
        :return: the batch ID and the relative frame ID
        :rtype: Tuple[int, int]
        """
        batch_id = bisect.bisect_left(self.lens_cumsum, frame_id + 1)
        try:
            rel_frame_id = frame_id - self.lens_cumsum[batch_id]
        except:
            print('batch_id: {}'.format(batch_id))
            print('cumsum: {}'.format(self.lens_cumsum))
            print('frame_id+1: {}'.format(frame_id + 1))
            raise
        return batch_id, rel_frame_id

    def batch_filename_by_id(self, batch_id, gzipped=False):
        """
        Returns the data batch filename of the specified batch ID.

        :param batch_id: the batch ID
        :type batch_id: int
        :param gzipped: True to returns the gzipped file; else the extracted
               file
        :type gzipped: bool
        :return: the data batch filename
        :rtype: Path
        """
        if gzipped:
            return os.path.join(self.root, 'data_batch_{}.gz'.format(batch_id))
        else:
            return os.path.join(self.root_tmp,
                                'data_batch_{}'.format(batch_id))


def create_vdset(video_file, root, batch_size=1000, max_batches=None):
    r"""
    Create ``VideoDataset`` on disk from video.

    :param video_file: the video filename
    :type video_file: str
    :param root: the dataset root directory
    :type root: str
    :param batch_size: the number of frames to be stored in one memory mapped
           file; note that the size of the file will be
           :math:`\text{batch\_size} \times 3 \times w \times h` bytes
    :type batch_size: int
    :param max_batches: the maximum number of batches to write; ``None`` means
           as many as possible
    :type max_batches: Optional[int]
    """
    video_file = os.path.realpath(video_file)
    root = os.path.realpath(root)  # type: str
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(os.path.join(root, 'tmp')):
        os.mkdir(os.path.join(root, 'tmp'))

    channels = 3
    dimension = 'NHWC'
    dtype = 'uint8'
    lens = []
    checksum_lines = []
    with utils.capcontext(video_file) as cap:  # type: cv2.VideoCapture
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = height, width

        frames = list(utils.frameiter(cap, batch_size))
        while frames:
            lens.append(len(frames))
            frames = np.concatenate(frames, axis=0)
            batch_id = len(lens) - 1

            # create memory mapped file
            batch_filename = os.path.join(root, 'tmp',
                                          'data_batch_{}'.format(batch_id))
            with utils.memmapcontext(batch_filename, dtype,
                                     frames.shape, mode='w+') as mm:
                mm[:] = frames

            # compute hash
            actual_hex = compute_file_integrity(batch_filename)
            checksum_lines.append(
                (actual_hex, os.path.basename(batch_filename)))

            # compress
            batch_gzipfilename = os.path.join(root, 'data_batch_{}.gz'.format(
                batch_id))
            compress_gzip(batch_filename, batch_gzipfilename)

            # remove the uncompressed memory mapped file
            os.remove(batch_filename)

            # reset frame iterator and prepare for next round, if any
            if max_batches is not None and len(lens) == max_batches:
                break
            frames = list(utils.frameiter(cap, batch_size))

    if max_batches is not None:
        assert len(lens) == max_batches, 'len(lens) ({}) != max_batches ({})' \
            .format(len(lens), max_batches)

    checksumfile = get_dset_filename_by_ext(root, '.' + HASH_ALGORITHM)
    with open(checksumfile, 'w') as outfile:
        for row in checksum_lines:
            outfile.write('  '.join(row) + '\n')

    metainfo = {
        'lens': lens,
        'resolution': resolution,
        'channels': channels,
        'dimension': dimension,
        'dtype': dtype,
    }
    metafile = get_dset_filename_by_ext(root, '.json')
    with open(metafile, 'w') as outfile:
        json.dump(metainfo, outfile)


def get_normalization_stats(root: str, bw: bool = False,
                            tags: Tuple[Tuple[float], Tuple[float]] = (),
                            as_rgb: bool = False):
    """
    Returns the normalization statistics (mean, std) in preprocessing step
    when loading the dataset. The normalization data presents in a ``npz``
    file ``$NORMALIZATION_INFO_FILE`` under the dataset root directory.

    Usage example::

        .. code-block::

            import torchvision.transforms as trans
            normalize = trans.Normalize(*get_normalization_stats(root))

    :param root: the root directory of the video dataset
    :param bw: load the B&W mean/std instead
    :raise IOError: if the ``$NORMALIZATION_INFO_FILE`` is not found under
           ``root``, which may due to spelling error in ``root`` or the file
           is absent as a matter of fact. For the latter case, compute the
           normalization statistics (using
           ``$PROJECT_HOME/bin/compute-perch-stat.py``), put the result file
           to the root directory, before calling this function.
    :param tags: a list of additional tags; ``bw=True`` implies
           ``tags=['bw']``
    :param as_rgb: repeat mean and std into three times when ``bw`` is True or
           when ``tags`` contains 'bw'
    :return: the mean and std
    """
    if bw:
        tags = ['bw']
    name, ext = os.path.splitext(NORMALIZATION_INFO_FILE)
    filename = name + ''.join(['_{}'.format(t) for t in tags]) + ext
    data = np.load(os.path.join(root, filename))
    mean = tuple(map(float, data['mean']))
    std = tuple(map(float, data['std']))
    if 'bw' in tags and as_rgb:
        assert len(mean) == len(std) == 1, \
            f'len(mean)={len(mean)} len(std)={len(std)}'
        mean *= 3
        std *= 3
    return mean, std


def dataset_root(cam_channel: int,
                 video_index: Union[int, Tuple[int, int, int]],
                 day: Union[int, str] = None) -> str:
    """
    A convenient function to get the root directory under DATASETS_ROOT.

    :param cam_channel: the camera channel ID
    :param video_index: the video index, either a 3-tuple of integers, or
           a single integer which will be expanded to a 3-tuple as
           ``(video_index, 0, 0)``. The first element of the tuple represents
           the starting hour of the video
    :param day: the day, can be either a string represeting the full date,
           e.g. '2Feb13', or an integer that will be expanded to '{day}Feb13'
    :return: the root directory
    """
    if isinstance(video_index, int):
        video_index = video_index, 0, 0
    if isinstance(day, int):
        day = '{}Feb13'.format(day)
    # tmpl_nd.format(cam_channel, video_index)
    tmpl_nd = '[CH{0:>02}]{1[0]:>02}_{1[1]:>02}_{1[2]:>02}'
    # tmpl_d.format(cam_channel, video_index, video_index[0]+3, day)
    tmpl_d = os.path.join('{3}.{1[0]}to{2}', tmpl_nd)

    tokens = cam_channel, video_index
    if day:
        tokens += video_index[0] + 3, day
        tmpl = tmpl_d
    else:
        tmpl = tmpl_nd
    root = os.path.join(DATASETS_ROOT, tmpl.format(*tokens))

    if not os.path.isdir(root):
        raise ValueError('root "{}" is not an existing directory'
                         .format(root))
    return root


class LabelledVideoDataset(Dataset):
    """
    The labelled dataset based on ``VideoDataset`` (the unlablled).
    """

    def __init__(self, labels, vdset: VideoDataset):
        """
        The parameter of __init__ is the same as that of
        ``VideoDataset.__init__`` except for the additional positional argument
        ``labels``. The dataset labels can be provided via:

        - string containing filename: the i-th line of the file should be the
          label of the i-th datum in the dataset
        - an array-like object containing the labels: the type includes
          ``list``, ``torch.Tensor``, ``np.ndarray``. The element type must be
          a sort of integer (each takes at most 4 bytes).

        The number of labels must be the same as the size of the dataset.

        :param labels: the labels of the dataset
        :param vdset: the video dataset to label
        """
        if isinstance(labels, str):
            with open(labels) as infile:
                labels = list(map(int, map(str.strip, infile)))
        labels = np.array(labels, dtype=np.int64)
        if len(vdset) != len(labels):
            raise ValueError('len(labels) ({}) is different from len(dataset)'
                             ' ({})'.format(len(labels), len(vdset)))
        self.vdset = vdset
        self.labels = labels

    def __len__(self):
        return len(self.vdset)

    def __getitem__(self, item):
        return self.vdset[item], self.labels[item]


class GaussianMLENormalizedDataset(Dataset):
    """
    Remove most of the linear independence in the dataset using maximum
    likelihood estimation by minimizing Gaussian error. Inside the
    ``GaussianMLENormalizedDataset`` a ``VideoDataset`` instance serves as
    the backend.

    The pixels on the normalized frame will still be in range of [0, 255], of
    type ``numpy.uint8``.

    By default, the item returned by index ``i`` on
    ``GaussianMLENormalizedDataset`` is the normalized version
    of the item returned by index ``i + (window_size - 1) // 2`` on its
    ``VideoDataset`` backend.
    """

    def __init__(self, window_size: int, backend: VideoDataset,
                 transform=None, use_actual_index=False):
        """
        :param window_size: the window size of the Gaussian MLE range; should
               be an odd positive integer at least 3
        :param backend: the backend ``VideoDataset``
        :param transform: the transform to be applied, default to ``None``
        :param use_actual_index: True to index the frames by the actual
               index used on ``VideoDataset`` backend. Default to ``False``.
               Note that after setting to ``True``, all indices smaller than
               ``(window_size - 1) // 2`` and larger than or equal to
               ``len(self.backend) - (window_size - 1) // 2`` will lead to
               ``IndexError``
        """
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError('Illegal window_size value: {}'
                             .format(window_size))
        self.halfw = (window_size - 1) // 2
        self.use_actual_index = use_actual_index
        self.transform = transform
        self.backend = backend

    def __len__(self):
        return len(self.backend) - 2 * self.halfw + 1

    def __getitem__(self, index: int):
        if not self.use_actual_index:
            index += self.halfw
        bg = np.mean([self.backend[i]
                      for i in range(index - self.halfw,
                                     index + self.halfw + 1)],
                     axis=0)
        img = self.backend[index]
        img = np.array((img - bg + 255) / 2, dtype=np.uint8)
        if self.transform:
            img = self.transform(img)
        return img

    def __enter__(self):
        self.backend.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.__exit__(exc_type, exc_val, exc_tb)
