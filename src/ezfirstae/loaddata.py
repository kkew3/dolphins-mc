import itertools
import operator as op
from functools import partial

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms as trans
from typing import Union, Tuple, List

import more_sampler
import utils


class PreProcTransform(object):
    """
    The pre-processing transformation before the encoder.
    """

    def __init__(self, normalize, pool_scale: int = 1, downsample_scale: int = 2):
        """
        :param normalize: the normalization transform, i.e.
               ``torchvision.transforms.Normalize`` instance
        :param pool_scale: the overall scale of the pooling operations in
               subsequent encoder; the image will be cropped to (H', W') where
               H' and W' are the nearest positive integers to H and W that are
               the power of ``pool_scale``, so that ``unpool(pool(x))`` is of
               the same shape as ``x``
        :param downsample_scale: the scale to downsample the video frames
        """
        self.T = trans.ToTensor()
        self.normalize = normalize
        self.pool_scale = pool_scale
        self.downsample_scale = downsample_scale

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
        return normalized


def rearrange_temporal_batch(data_batch: torch.Tensor, T: int) -> torch.Tensor:
    """
    Rearrange a hyper-batch of frames of shape (B*T, C, H, W) into
    (B, C, T, H, W) where:

        - B: the batch size
        - C: the number of channels
        - T: the temporal batch size
        - H: the height
        - W: the width

    :param data_batch: batch tensor to convert
    :param T: the temporal batch size
    :return: converted batch
    """
    assert len(data_batch.size()) == 4
    assert data_batch.size(0) % T == 0
    B = data_batch.size(0) // T
    data_batch = data_batch.view(B, T, *data_batch.shape[1:])
    data_batch = data_batch.transpose(1, 2).contiguous()
    return data_batch.detach()  # so that ``is_leaf`` is True


def split_inputs_targets(temporal_data_batch: torch.Tensor,
                         inputs_ind, targets_ind) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a termporal data batch of shape (B, C, T, H, W) into an input tensor
    of shape (B, C, Ti, H, W) and a target tensor of shape (B, C, To, H, W).

    :param temporal_data_batch: the batch tensor to split
    :param inputs_ind: the array-like indices along the temporal (T) axis to
           assign to the input tensor
    :param targets_ind: the array-like indices along the temporal (T) axis to
           assign to the target tensor
    :return: the input tensor and the target tensor
    """
    raise NotImplementedError()


class TrainValidTestAlternatingSamplers(object):
    """
    Initializer of a trainset sampler (, a validation set sampler) and a testset
    sampler from one data source.

    Steps to sample:

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
    the 2nd group, all two segments will be sampled to trainset. If not
    ``shuffled`` is True, then the trainset will be ``[0,1,2,3,4,5,8,9,10,11]``
    and the testset will be ``[6,7]``; otherwise, the assignment of each segment
    will remain the same except that the order is randomized, e.g. trainset as
    ``[2,3,10,11,8,9,0,1,4,5]``. Note that the order of frames in each segment
    always remains intact.


    How to get the samplers::

        .. code-block::

            # when `ratios` is a 2-tuple of integers
            samplers = TrainValidTestAlternatingSamplers(data_source, ratios=(5, 1))
            train_sampler = samplers.train_sampler
            test_sampler = samplers.test_sampler

            # when `ratios` is a 3-tuple of integers
            samplers = TrainValidTestAlternatingSamplers(data_source, ratios=(9, 1, 2))
            train_sampler = samplers.train_sampler
            validation_sampler = samplers.validation_sampler
            test_sampler = samplers.test_sampler


    The division of trainset (, validation set) and testset happens on
    instantiation. The sampling within each class of dataset happens every time
    on ``__getattr__`` of corresponding sampler name.
    """

    def __init__(self, data_source, homo_period: int = 16,
                 alternating_period: int = 10,
                 ratios: Union[Tuple[int, int, int], Tuple[int, int]] = (9, 1, 2),
                 shuffled: bool=False):
        """
        :param data_source: the handle to the data source; note that the handle
               itself won't be maintained in this sampler
        :param homo_period: the maximum length of contiguous frames such that
               they are sampled into the same class (train/validation/test);
               the quantity should be identical to the ``batch_size`` when
               training
        :param alternating_period: the minimum number of contiguous groupings
               of frames as specified by ``homo_period`` such that at least one
               of frame of which is sampled to every class
               (train/validation/test)
        :param ratios: either a 2-tuple or a 3-tuple of integers; when 2-tuple,
               the data source will be split to trainset and testset in ratio
               specified by ``ratios``; when 3-tuple, it will be split to
               trainset, validation set, and testset
        :param shuffled: True to do random assignment at the fourth step (see
               ``help(type(self))``); False to do sequential assignment
        """
        self.homo_period = homo_period
        self.alternating_period = alternating_period
        self.shuffled = shuffled
        ratios = np.array(ratios).astype(np.float64)
        self.ratios = ratios / ratios.sum()

        disposable_indices = np.arange(utils.inf_powerof(len(data_source), self.homo_period))
        segments = disposable_indices.reshape((-1, self.homo_period))
        self._segindices_pc = self.do_sample(segments, self.ratios)
        sampler_names = ['train_sampler', 'test_sampler']
        if len(ratios) == 3:
            sampler_names.insert(1, 'validation_sampler')
        self._samplername2index = {name: i for i, name in enumerate(sampler_names)}

    def __getattr__(self, item):
        try:
            super().__getattribute__(item)
        except AttributeError as e:
            try:
                return self._make_sampler(self._segindices_pc[self._samplername2index[item]])
            except KeyError:
                raise e

    def do_sample(self, segments: np.ndarray, ratios: np.ndarray) -> List[np.ndarray]:
        segindices_pc = list([] for _ in range(len(ratios)))
        n = segments.shape[0]
        for i in range(0, n, self.alternating_period):
            group = segments[i:min(n, i + self.alternating_period)]
            gn = group.shape[0]
            # why to add 1e-6: so that np.round(0.5 + 1e-6) returns 1.0.
            # Try it: np.round(0.5) -> 0.0, which is incorrect in this context
            quota = np.round(ratios * gn + 1e-6).astype(np.int64)
            quota[-1] = gn - quota[:-1].sum()
            gind = np.arange(gn)
            lims = np.insert(np.cumsum(quota), 0, 0)
            for j, segindices in enumerate(segindices_pc):
                segindices.append(group[gind[lims[j]:lims[j+1]]])
        segindices_pc = list(map(np.concatenate, segindices_pc))
        return segindices_pc

    def _make_sampler(self, segindices: np.ndarray) -> Sampler:
        segindices = np.copy(segindices)
        if self.shuffled:
            perm = np.random.permutation(segindices.shape[0])
            segindices = segindices[perm]
        indices = list(segindices.reshape(-1))
        return more_sampler.ListSampler(indices)


class SlidingWindowBatchSampler(Sampler):
    """
    Samples in a sliding window manner.
    """
    def __init__(self, indices, window_width: int,
                 shuffled: bool=False, batch_size: int=1,
                 drop_last: bool=False):
        """
        :param indices: array-like integer indices to sample; when presented as
               a list of arrays, no sample will span across more than one array
        :param window_width: the width of the window; if ``window_width`` is
               larger than the length of ``indices`` or the length of one of
               the sublists, then that list won't be sampled
        :param shuffled: whether to shuffle sampling, but the indices order
               within a batch is never shuffled
        :param batch_size: how many batches to yield upon each sampling
        :param drop_last: True to drop the remaining batches if the number of
               remaining batches is less than ``batch_size``

        Note on ``batch_size``
        ----------------------

        When ``batch_size = 2``, assuming that the two batch of indices are
        ``[1, 2, 3, 4]`` and ``[4, 5, 6, 7]``, then the yielded hyper-batch
        will be ``[1, 2, 3, 4, 4, 5, 6, 7]``.
        """
        indices = list(map(lambda x: x.astype(np.int64), np.array(indices)))
        if indices and not len(indices[0].shape):
            indices = [np.array(indices)]
        self.indices = indices  # a list of int64-arrays, or an empty list
        self.window_width = window_width
        self.shuffled = shuffled
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self):
        return sum(map(self._calc_sliding_distance, map(len, self.indices)))

    def __iter__(self):
        seglens = map(len, self.indices)
        slidedists = map(self._calc_sliding_distance, seglens)
        startindices = map(range, slidedists)
        segid_startindices = enumerate(startindices)
        segid_startindices = map(lambda x: utils.browadcast_value2list(*x),
                                 segid_startindices)
        segid_startindices = list(itertools.chain(*segid_startindices))
        perm = (np.random.permutation if self.shuffled else np.arange)(
                len(segid_startindices))
        _gi = partial(op.getitem, segid_startindices)
        for i in range(0, len(segid_startindices), self.batch_size):
            ind_tosample = perm[i:i+self.batch_size]
            if not (len(ind_tosample) < self.batch_size and self.drop_last):
                segid_startind_tosample = map(_gi, ind_tosample)
                sampled_batches = map(self._sample_batch_once, segid_startind_tosample)
                yield list(np.concatenate(list(sampled_batches)))

    def _calc_sliding_distance(self, length):
        return length - self.window_width + 1

    def _sample_batch_once(self, segid_startind):
        segid, startind = segid_startind
        return self.indices[segid][startind:startind+self.window_width]

