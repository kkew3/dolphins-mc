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
        |- tmp/                           # temporarily store uncompressed data
        |  |- data_batch_x
        |  |- data_batch_y
        |  |- ...
        |- data_batch_0.gz
        |- data_batch_1.gz
        |- ...
        |- data_batch_n.gz
        |- video_name.json                # metadata, see below
        |- video_name.sha1                # sha1 of data batches
        |- video_name.access.lock         # mutex file lock


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

import gzip
import os
import re
import json
import hashlib
import operator as op
from typing import Iterable, Callable, Iterator, List, Tuple
import bisect
import shutil

from filelock import FileLock
from cachetools import LRUCache
from pathlib2 import Path
import numpy as np
from torch.utils.data import Dataset
import cv2

import utils


HASH_ALGORITHM = 'sha1'
DATABATCH_FILENAME_PAT = re.compile(r'^data_batch_(\d+)$')


def parse_checksum_file(filename):
    """
    Parse checksum file as created by ``/usr/bin/sha1sum``.

    :param filename: the checksum filename
    :type filename: Path
    :return: a list ``l`` such that ``l[i]`` contains the expected hash of the
             i-th batch
    :rtype: List[str]
    """
    checksum_lines = []
    with filename.open() as infile:
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
    expected_hexes = map(lambda x: x[1], checksum_lines)
    return expected_hexes


def check_file_integrity(filename, expected_hex):
    """
    Check file integrity using HASH_ALGORITHM.

    :param filename: the filename to check integrity
    :type filename: Path
    :param expected_hex: the expected hash hex
    :type expected_hex: str
    :return: True if the check passes
    :rtype: bool
    """
    hashbuf = hashlib.new(HASH_ALGORITHM)
    with filename.open('rb') as infile:
        for block in iter(lambda: infile.read(11048576), b''):
            hashbuf.update(block)
    actual_hex = hashbuf.hexdigest()
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

def extract_gzip(fromfile, tofile):
    """
    :param fromfile: source gzip file
    :type fromfile: Path
    :param tofile: target file to write
    :type tofile: Path
    """
    with gzip.open(str(fromfile), 'rb') as infile:
        with tofile.open('wb') as outfile:
            shutil.copyfileobj(infile, outfile)

class VideoDataset(Dataset):
    """
    Represents the video dataset (readonly).
    """
    def __init__(self, root, transform=None, max_mmap=1):
        """
        :param root: the root directory of the dataset
        :type root: Union[str, Path]
        :param transform: the transformations to be performed on loaded data
        :param max_mmap: the maximum number of memory map to keep
        """
        self.root = Path(root).resolve()  # type: Path
        jsonfile = self.root / (self.root.stem + '.json')
        with jsonfile.open() as infile:
            self.metainfo = json.load(infile)
        self.total_frames = np.prod(self.metainfo['lens'])
        self.lens_cumsum = list(accumulate(self.metainfo['lens']))
        shape = self.metainfo['resolution'] + [self.metainfo['channels']]
        self.frame_shape = tuple(shape)

        # if self.validated_batches[i] == 0, then batch i hasn't been validated
        self.validated_batches = [False] * len(self.metainfo['lens'])
        checksumfile = self.root / (self.root.stem + '.' + HASH_ALGORITHM)
        self.expected_hexes = parse_checksum_file(checksumfile)

        self.root_tmp = self.root / 'tmp'  # type: Path
        if not self.root_tmp.is_dir():
            self.root_tmp.mkdir()

        self.transform = transform

        max_mmap = max(1, max_mmap)
        self.mmap_cache = LRUCache(maxsize=max_mmap)

        lockfile = self.root / (self.root.stem + '.access.lock')
        self.access_lock = FileLock(str(lockfile))

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
        batch_id, rel_frame_id = self.locate_batch(frame_id)
        with self.access_lock:
            if batch_id not in self.mmap_cache:
                batchf = self.batch_filename_by_id(batch_id)
                if not batchf.is_file():
                    extract_gzip(self.batch_filename_by_id(batch_id, gzipped=True),
                                 batchf)
                    assert batchf.is_file()
                if not self.validated_batches[batch_id]:
                    if not check_file_integrity(batchf, self.expected_hexes[batch_id]):
                        raise RuntimeError('Data batch {} corrupted'.format(batch_id))
                    self.validated_batches[batch_id] = True

                shape = (self.metainfo['lens'][batch_id],) + self.frame_shape
                self.mmap_cache[batch_id] = np.memmap(str(batchf), mode='r',
                                                      dtype=self.metainfo['dtype'],
                                                      shape=shape)
        return np.copy(self.mmap_cache[batch_id][rel_frame_id])

    def __del__(self):
        self.release_mmap()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_mmap()

    def release_mmap(self):
        """
        Release all memory mapped dataset.
        """
        for k in self.mmap_cache:
            del self.mmap_cache[k]

    def locate_batch(self, frame_id):
        """
        Locate the data batch the specified frame is stored.

        :param frame_id: the frame ID
        :type frame_id: int
        :return: the batch ID and the relative frame ID
        :rtype: Tuple[int, int]
        """
        batch_id = bisect.bisect_left(self.lens_cumsum, frame_id)
        rel_frame_id = frame_id - self.lens_cumsum[batch_id]
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
            return self.root / 'data_batch_{}.gz'.format(batch_id)
        else:
            return self.root_tmp / 'data_batch_{}'.format(batch_id)


def create_vdset(video_file, root):
    """
    Create ``VideoDataset`` on disk from video.

    :param video_file: the video filename
    :type video_file: Path
    :param root: the dataset root directory
    :type root: Path
    """
    pass
