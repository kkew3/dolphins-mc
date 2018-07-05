"""
Defines the class representing the video dataset
"""
import json
import os
import re
import hashlib

from cachetools import LRUCache
# noinspection PyPackageRequirements
import hickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import utils


class VideoDataset(Dataset):
    """
    Represents a HDF5 block-based video dataset, as built by
    "${repo-root}/bin/build-video-dataset.py". The dataset returns the
    corresponding frame tensor when requesting the frame ID (indexed from zero).
    The dataset is not supervised.
    """
    BLOCKFILE_TMPL = 'B{}.hkl'  # usage: BLOCKFILE_TMPL.format(block_id)
    BLOCKFILE_PAT = re.compile(r'^B(\d+)\.hkl$')

    def __init__(self, root, max_block_cached=2, transform=None):
        """
        :param root: the dataset root directory
        :param max_block_cached: maximum number of blocks to be cached in
               memory; must be at least 1
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

    def check_integrity(self):
        video_name = os.path.basename(self.root)
        checksumfile = os.path.join(self.root, video_name + '.sha1')
        checksums = dict()
        with open(checksumfile) as infile:
            for line in infile:
                expected, block_filename = line.strip().split(' *')
                checksums[os.path.join(self.root, block_filename)] = expected
        for filename in checksums:
            hashbuff = hashlib.sha1()
            try:
                with open(filename, 'rb') as infile:
                    for block in iter(lambda: infile.read(1024 * 1024), b''):
                        hashbuff.update(block)
                if hashbuff.hexdigest() != checksums[filename]:
                    raise RuntimeError('Dataset corrupted')
            except IOError:
                raise RuntimeError('Dataset not found or corrupted')

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
    "${repo-root}/bin/build-video-dataset.py". The dataset returns a pair of
    frame tensor when requesting the frame ID (indexed from zero).
    """
    def __init__(self, dataset1, dataset2):
        """
        :param dataset1: one `VideoDataset` instance representing one video
        :type dataset1: VideoDataset
        :param dataset2: another `VideoDataset` instance representing another
               video
        :type dataset2: VideoDataset
        """
        self.dataset_pair = dataset1, dataset2

    def __len__(self):
        return min(map(len, self.dataset_pair))

    def __getitem__(self, item):
        return tuple(ds[item] for ds in self.dataset_pair)
