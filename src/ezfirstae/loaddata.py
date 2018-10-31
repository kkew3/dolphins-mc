import numpy as np
from PIL import Image
import torch
from torchvision import transforms as trans
from typing import List, Sequence

import utils


class PreProcTransform(object):
    """
    The pre-processing transformation before the encoder.

    Steps:

        1. convert RGB to B&W by taking the maximum along the channel axis
        2. downsample by specified scale
        3. random crop a little bit such that the height and the width are
           both powers of the subsequent pooling scale
        4. normalize the image matrix
        5. optionally convert B&W back to RGB by repeating the image matrix
           three times along the channel axis
    """

    def __init__(self, normalize,
                 pool_scale: int = 1,
                 downsample_scale: int = 2,
                 to_rgb=False):
        """
        :param normalize: the normalization transform, i.e.
               ``torchvision.transforms.Normalize`` instance
        :param pool_scale: the overall scale of the pooling operations in
               subsequent encoder; the image will be cropped to (H', W') where
               H' and W' are the nearest positive integers to H and W that are
               the power of ``pool_scale``, so that ``unpool(pool(x))`` is of
               the same shape as ``x``
        :param downsample_scale: the scale to downsample the video frames
        :param to_rgb: if True, at the last step convert from B&W image to
               RGB image
        """
        self.T = trans.ToTensor()
        self.normalize = normalize
        self.pool_scale = pool_scale
        self.downsample_scale = downsample_scale
        self.to_rgb = to_rgb

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        toshape = tuple(x // self.downsample_scale for x in img.shape[:2])
        img = Image.fromarray(img.max(axis=2))
        if not hasattr(self, 'downsample'):
            self.downsample = trans.Resize(toshape)
        img = self.downsample(img)
        if not hasattr(self, 'crop'):
            h, w = img.height, img.width
            h = utils.inf_powerof(h, self.pool_scale)
            w = utils.inf_powerof(w, self.pool_scale)
            self.crop = trans.RandomCrop((h, w))
        img = self.crop(img)
        tensor = self.T(img)
        normalized = self.normalize(tensor)
        if self.to_rgb:
            normalized = normalized.repeat(3, *([1]*(len(normalized.shape)-1)))
        return normalized


def alternate_partition_dataset(indices: Sequence[int],
                                ratios: Sequence[int],
                                homo_period: int = 16,
                                alternating_period: int = 10) -> List[List[np.ndarray]]:
    """
    Partition the indices of a dataset into several partitions (say, in order to
    perform a train/test split). The partitions are arranged in an alternating
    order. The incontiguous regions of indices assigned to the same partition
    is denoted as belonging to the same class. Assume that we have three classes
    C1, C2 and C3, then the partition of indices appears like this::

        .. code-block::

            C1 | C2 | C3 | C1 | C2 | C3 | C1 | C2 | C3 | ...

    Steps to partition:

        1. Remove a minimum trailing part of the data source (no actual removal)
           such that the length of the remaining part is a power of
           ``homo_period``;
        2. Partition the remaining part into equal-length segments, each of
           length ``homo_period``;
        3. Group every ``alternating_period`` segments into groups, the last
           group may have less than ``alternating_period` segments;
        4. (Randomly) assign classes to each group. Let the number of segments
           in current group be ``N``, then the number of segments to be sampled
           to the i-th class will be ``round(ratios[i]/sum(ratios)*N)``.

    For the 4th step, assuming that grouped segments of indices are presented as
    ``[[[0,1],[2,3],[4,5],[6,7]], [[8,9],[10,11]]]``, i.e. two groups of
    ``alternating_period`` as 4 and ``homo_period`` as 2, assuming that the
    ``ratios`` is ``(5, 1)``, then 3 out of 4 segments in the 1st group will be
    sampled into trainset, and 1 out of 4 segments will end up into testset; for
    the 2nd group, all two segments will be sampled to trainset.

    :param indices: the indices to partition
    :param ratios: sequence of integers, the size of each class will be
           proportional to the fractions of these integers against the sum
    :param homo_period: the maximum length of contiguous frames such that
           they are sampled into the same class;
           the quantity should be identical to the ``batch_size`` when
           training
    :param alternating_period: the minimum number of contiguous groupings
           of frames as specified by ``homo_period`` such that at least one
           of frame of which is sampled to every class
    :return: K lists of arrays of indices such that the each array of indices in
             the ith list is one contiguous partition of the ith class, where K
             equals to the length of ``ratios``
    """

    ratios = np.array(ratios, dtype=np.float64)
    ratios = ratios / ratios.sum()

    disposable_indices = indices[:utils.inf_powerof(len(indices), homo_period)]
    segments = disposable_indices.reshape((-1, homo_period))

    # do partition on segments as per ratios
    segindices_pc = list([] for _ in range(len(ratios)))
    n = segments.shape[0]
    for i in range(0, n, alternating_period):
        group = segments[i:min(n, i + alternating_period)]
        gn = group.shape[0]
        # why to add 1e-6: so that np.round(0.5 + 1e-6) returns 1.0.
        # Try it: np.round(0.5) -> 0.0, which is incorrect in this context
        quota = np.round(ratios * gn + 1e-6).astype(np.int64)
        quota[-1] = gn - quota[:-1].sum()
        gind = np.arange(gn)
        lims = np.insert(np.cumsum(quota), 0, 0)
        for j, segindices in enumerate(segindices_pc):
            segindices.append(group[gind[lims[j]:lims[j + 1]]])
    return segindices_pc


def contiguous_partition_dataset(indices: Sequence[int],
                                 ratios: Sequence[int]) -> List[np.ndarray]:
    """
    Partition the indices of a dataset into K contiguous regions such that the
    length of the ith region is proportional to the fraction of the ith element
    of ``ratios`` in ``sum(ratios)``, where K equals to the length of
    ``ratios``.

    :param indices: the indices to partition
    :param ratios: sequence of integers, the size of each class will be
           proportional to the fractions of these integers against the sum
    :return: K arrays of indices
    """
    indices = np.array(indices, dtype=np.int64)
    ratios = np.array(ratios, dtype=np.float64)
    ratios = ratios / ratios.sum()
    quota = np.round(ratios * len(indices) + 1e-6).astype(np.int64)
    quota[-1] = len(indices) - quota[:-1].sum()
    lims = np.insert(np.cumsum(quota), 0, 0)
    partitions = []
    for j in range(len(ratios)):
        partitions.append(indices[lims[j]:lims[j+1]])
    return partitions
