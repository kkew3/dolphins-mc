import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import torchvision.transforms as trans
from typing import Callable, Tuple
import tensorboardX
from ddlogger import ddlogger

import salicae
import trainlib


class SlidingWindowBatchSampler(Sampler):
    """
    Batch sampler that yields a sequential sliding window of indices.

    >>> list(SlidingWindowBatchSampler(range(5), 3))
    [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    >>> list(SlidingWindowBatchSampler(range(5), 2))
    [[0, 1], [1, 2], [2, 3], [3, 4]]
    >>> list(SlidingWindowBatchSampler(range(5), 3, drop_margin=False))
    [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4]]
    >>> list(SlidingWindowBatchSampler(range(5), 3, drop_margin=False, left_biased=False))
    [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4]]
    >>> list(SlidingWindowBatchSampler(range(5), 2, drop_margin=False))
    [[0], [0, 1], [1, 2], [2, 3], [3, 4]]
    >>> list(SlidingWindowBatchSampler(range(5), 2, drop_margin=False, left_biased=False))
    [[0, 1], [1, 2], [2, 3], [3, 4], [4]]
    """

    def __init__(self, data_source, width, drop_margin=True, left_biased=True):
        r"""
        :param data_source: the data soure
        :type data_source: torch.utils.data.Dataset
        :param width: the window width, should be at least 1
        :type width: int
        :param drop_margin: True to drop batches on the two ends of the dataset
               of which the width is less than ``width``
        :param left_biased: True to make batch by extending the center index
               to left by one element, then extending on both sides
               :math:`\frac{w - 2}{2}` elements; make difference only if
               ``width`` is even and ``drop_margin`` is ``False``
        """
        super(SlidingWindowBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.width = max(1, width)
        self.N = len(self.data_source)
        self.drop_margin = drop_margin
        self.left_biased = left_biased

        # cached for later use
        self.even_width = (self.width % 2) == 0

        # to avoid external modifications of the attributes, since
        # `len(self.data_source)` and `self.even_width` has already been cached
        self.__setattr_lock = True

    def __repr__(self):
        to_show = ['width', 'drop_margin', 'left_biased']
        s = '{}({})'.format(type(self).__name__,
                            ', '.join(map(lambda x: '{0}={{{0}}}'.format(x),
                                          to_show)).format(**self.__dict__))
        return s

    def __len__(self):
        l = self.N
        if self.drop_margin:
            l = l - self.width + 1
        return l

    def __setattr__(self, key, value):
        if hasattr(self, '__setattr_lock') and self.__setattr_lock:
            raise ValueError('Attribute {} is readonly'.format(key))
        super(SlidingWindowBatchSampler, self).__setattr__(key, value)

    def __iter__(self):
        for center_index in range(self.N):
            if self.even_width:
                if self.left_biased:
                    center_indices = (center_index - 1, center_index)
                else:
                    center_indices = (center_index, center_index + 1)
                halfwidth = (self.width - 2) // 2
                left_end = max(0, center_indices[0] - halfwidth)
                right_end = min(self.N, center_indices[1] + halfwidth + 1)
            else:
                halfwidth = (self.width - 1) // 2
                left_end = max(0, center_index - halfwidth)
                right_end = min(self.N, center_index + halfwidth + 1)
            batch = range(left_end, right_end)
            if len(batch) < self.width and self.drop_margin:
                continue
            yield batch
        raise StopIteration()


# noinspection PyMissingConstructor
class BatchSlidingWindowBatchSampler(Sampler):
    """
    Unlike ``MovingWindowBatchSampler`` which samples at one time a moving
    window surrounding one frame, this sampler tries to samples a batch of
    center frames.

    >>> list(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 2))
    [[0, 1, 2, 3]]
    >>> len(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 2))
    1
    >>> list(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 2, drop_last=False))
    [[0, 1, 2, 3], [2, 3, 4]]
    >>> len(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 2, drop_last=False))
    2
    >>> list(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 3))
    [[0, 1, 2, 3, 4]]
    >>> len(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 3))
    1
    >>> list(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 4))
    []
    >>> len(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 4))
    0
    >>> list(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 4, drop_last=False))
    [[0, 1, 2, 3, 4]]
    >>> len(BatchSlidingWindowBatchSampler(SlidingWindowBatchSampler(range(5), 3), 4, drop_last=False))
    1
    """
    def __init__(self, sampler, batch_size, drop_last=True):
        """
        :param sampler: a ``MovingWindowBatchSampler`` instance
        :type sampler: SlidingWindowBatchSampler
        :param batch_size: the batch size
        :type batch_size: int
        :param drop_last: True to drop last batch if it does not fulfill
               ``batch_size``
        :type drop_last: bool
        """
        self.swsampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __repr__(self):
        to_show = ['swsampler', 'batch_size', 'drop_last']
        s = '{}({})'.format(type(self).__name__,
                            ', '.join(map(lambda x: '{0}={{{0}}}'.format(x),
                                          to_show)).format(**self.__dict__))
        return s

    def __len__(self):
        l = len(self.swsampler) // self.batch_size
        if not self.drop_last and len(self.swsampler) % self.batch_size > 0:
            l += 1
        return l

    def __iter__(self):
        bbatch = []
        for batch in self.swsampler:
            if len(bbatch) < self.batch_size:
                bbatch.append(batch)
            else:
                yield range(bbatch[0][0], bbatch[-1][-1] + 1)
                bbatch = [batch]
        if (0 < len(bbatch) < self.batch_size and not self.drop_last) or\
                (len(bbatch) == self.batch_size):
            yield range(bbatch[0][0], bbatch[-1][-1] + 1)
        raise StopIteration()

def reduce_sliding_window(sw, frames, reducef, cpol='last'):
    """
    Reduce a batch of frames (Tensor of dimension BCHW) in a sliding window
    pattern.

    :param sw: the sliding window sampler
    :type sw: SlidingWindowBatchSampler
    :param frames: the batch of frames
    :type frames: torch.Tensor
    :param reducef: the reduce function that accepts a batch of frames and
           returns a batch of frame where the batch size is one; i.e.
           [B x C x H x W] -> [1 x C x H x W]
    :type reducef: Callable
    :param cpol: policy to pick out the central frame from a batch of frames;
           available options: 'last' (the last frame), 'first' (the first
           frame), 'middle' (the middle frame, i.e. ``(frames.shape[0]-1)//2``)
    :type cpol: str
    :return: center frames of windows (arranged in BCHW manner), and the
             reduction results (arranged in BCHW manner)
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    sw = SlidingWindowBatchSampler(frames, sw.width, drop_margin=True)
    cind = {'first': 0, 'middle': (frames.shape[0] - 1) // 2, 'last': -1}[cpol]

    center_frames = []
    reduced_frames = []
    for indices in sw:
        window = frames[indices]
        center_frames.append(frames[[cind]])
        reduced_frames.append(reducef(window))
    center_frames = torch.cat(center_frames, dim=0)
    reduced_frames = torch.cat(reduced_frames, dim=0)
    return center_frames, reduced_frames

def sw_median(frames):
    return torch.median(frames, dim=0, keepdim=True)[0]


def train(net, dataset, device, batch_size, lasso_strength,
          statdir, savedir):
    """
    Train the saliency autoencoder.

    :param net: the saliency autoencoder network to train
    :type net: salicae.SaliencyCAE
    :param dataset: the dataset
    :type dataset: torch.utils.data.Dataset
    :param device: the device
    :type device: torch.device
    :param batch_size: the batch size of training
    :type batch_size: int
    :param lasso_strength: the Lasso regularization strength on saliency
    :type lasso_strength: float
    :param statdir: directory under which to store scalar statistics
    :type statdir: str
    :param savedir: the directory under which to store model states
    :type savedir: str
    """
    net = net.to(device)
    checkpoint_saver = trainlib.CheckpointSaver(net, 10, savedir)
    stat_saver = trainlib.StatSaver(10, statdir)
    optimizer = optim.Adam(net.parameters())
    sw = SlidingWindowBatchSampler(dataset, width=30, drop_margin=True)
    sam = BatchSlidingWindowBatchSampler(sw, batch_size=batch_size, drop_last=True)
    dl = DataLoader(dataset, num_workers=4, batch_sampler=sam)
    mse = nn.MSELoss(reduction='sum')

    l1_list = []
    l2_list = []
    sl_list = []
    with ddlogger() as ddlog:
        for bid, frames in enumerate(dl):
            inputs, bgs = reduce_sliding_window(sw, frames, sw_median)
            inputs, bgs = inputs.to(device), bgs.to(device)
            saliencies = net(inputs)
            reconstr = saliencies * inputs + (1 - saliencies) * bgs
            l1 = mse(reconstr, inputs)
            l2 = torch.mean(torch.norm(saliencies.view(saliencies.shape[0], -1), 1, 1))
            loss = l1 + lasso_strength * l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ddlog.update()
            l1_list.append(l1.detach().item())
            l2_list.append(lasso_strength * l2.detach().item())
            sl_list.append(loss.detach().item())
            checkpoint_saver(bid)
            stat_saver(bid, reconstr_loss=l1_list, lasso_loss=l2_list,
                       loss=sl_list)
