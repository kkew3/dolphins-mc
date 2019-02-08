import collections
import itertools
import operator as op
import random
from functools import partial

import numpy as np

import torch
from torch.utils.data.sampler import Sampler
from typing import List, Sequence, Dict

import utils


class DummySampler(Sampler):
    """
    For debug.
    """
    def __init__(self, *args, **kwargs):
        self.indices = range(100)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


class RandomSubsetSampler(Sampler):
    """
    Samples a random subset of indices of specified size from the data source.
    """

    def __init__(self, data_source, n, shuffle=False):
        """
        :param data_source: the dataset object
        :param n: subset size
        :param shuffle: False to sort the indices after sampling
        """
        self.data_source = data_source
        self.n = n
        self.shuffle = shuffle

    def __len__(self):
        return min(self.n, len(self.data_source))

    def __iter__(self):
        indices = torch.randperm(len(self.data_source)).tolist()[:self.n]
        if not self.shuffle:
            indices.sort()
        return iter(indices)


class ListSampler(Sampler):
    """
    A wrapper over list to make it a ``Sampler``.
    """
    def __init__(self, indices: List[int]):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


class SlidingWindowBatchSampler(Sampler):
    """
    Samples in a sliding window manner.
    """

    def __init__(self, indices, window_width: int,
                 shuffled: bool = False, batch_size: int = 1,
                 drop_last: bool = False):
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
        indices = [np.array(x, dtype=np.int64) for x in list(indices)]
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
            ind_tosample = perm[i:i + self.batch_size]
            if not (len(ind_tosample) < self.batch_size and self.drop_last):
                segid_startind_tosample = map(_gi, ind_tosample)
                sampled_batches = map(self._sample_batch_once, segid_startind_tosample)
                yield list(np.concatenate(list(sampled_batches)))

    def _calc_sliding_distance(self, length):
        return length - self.window_width + 1

    def _sample_batch_once(self, segid_startind):
        segid, startind = segid_startind
        return self.indices[segid][startind:startind + self.window_width]


class BalancedLabelSampler(torch.utils.data.Sampler):
    """
    This sampler implies shuffled=True.

    >>> class NumpyDataset(torch.utils.data.Dataset):
    ...     def __init__(self, arr):
    ...         self.data, self.labels = arr[:, 0], arr[:, 1]
    ...     def __len__(self):
    ...         return len(self.data)
    ...     def __getitem__(self, index):
    ...         return self.data[index], self.labels[index]
    >>> dataset = NumpyDataset(
    ...     np.stack((np.random.randint(5, 11, size=500),
    ...               np.repeat(np.arange(5), 100)), axis=1))
    >>> sam = BalancedLabelSampler(dataset)
    >>> loader = torch.utils.data.DataLoader(dataset, sampler=sam)
    >>> label_counts = collections.defaultdict(int)
    >>> labels = []
    >>> for i, (_, label) in enumerate(loader):
    ...     if i >= 100:
    ...         break
    ...     label_counts[label.item()] += 1
    >>> np.var(list(label_counts.values())) < 500
    True
    >>> loader = torch.utils.data.DataLoader(dataset, sampler=sam)
    >>> len([_ for _ in loader])
    500
    """

    # noinspection PyMissingConstructor
    def __init__(self, *args):
        """
        To sets of positional arguments are possible, either;

        :param indicies: range of indices to sample from the underlying
               dataset
        :type indicies: Sequence[int]
        :param labels: the labels of corresponding indicies
        :type labels: Sequence[int]

        or:

        :param dataset: dataset where each item consists of an input and a
               target label
        :type dataset: torch.utils.data.Dataset
        """
        if len(args) == 2:
            indices, labels = args
            if len(indices) != len(labels):
                raise ValueError('length of labels ({}) != number of '
                                 'indices ({})'
                                 .format(len(labels), len(indices)))
        else:
            dataset, = args
            indices = range(len(dataset))
            labels = [dataset[i][1] for i in range(len(dataset))]
        self._len_dataset = len(indices)

        ul2idx = collections.defaultdict(list)
        for i, l in zip(indices, labels):
            ul2idx[l].append(i)
        self.ul2idx = ul2idx  # type: Dict[int, List[int]]
        """uniq labels to indicies"""

    def __len__(self):
        return self._len_dataset

    def __iter__(self):
        cindex_lens = {k: len(indices) for k, indices in self.ul2idx.items()}
        shuffled_indices = {k: np.random.permutation(cindex_lens[k])
                            for k in self.ul2idx}
        to_visit_pointers = {k: 0 for k in cindex_lens}

        remaining_clists = [k for k in self.ul2idx
                            if to_visit_pointers[k] < cindex_lens[k]]
        while remaining_clists:
            next_cnt = random.choice(remaining_clists)
            idx = self.ul2idx[next_cnt][
                shuffled_indices[next_cnt][
                    to_visit_pointers[next_cnt]]]
            to_visit_pointers[next_cnt] += 1
            remaining_clists = [k for k in self.ul2idx
                                if to_visit_pointers[k] < cindex_lens[k]]
            yield idx
