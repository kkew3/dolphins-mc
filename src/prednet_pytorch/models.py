"""
Modules and submodules of PredNet.
"""
import collections
import itertools
from functools import partial

import numpy as np
# noinspection PyPackageRequirements
import torch
# noinspection PyPackageRequirements
import torch.nn as nn
# noinspection PyPackageRequirements
from torch.nn import Parameter
# noinspection PyPackageRequirements
from torch.autograd import Variable
# noinspection PyPackageRequirements
import torch.nn.functional as F
#from torch.nn.modules.utils import _pair


class ConvLSTMCell(nn.Module):
    """
    The Convolutional LSTM cell.

    References:

    - Deep predictive coding networks for video prediction and unsupervised
      learning (https://arxiv.org/abs/1605.08104).
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=True):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride: controls the stride ofor the cross-correlation, a single
               number of a tuple
        :param padding: controls the amount of implicit zero-paddings on both
               sides for padding number of points for each dimension
        :param dilation: controls the spacing between the kernel points; also
               known as the atrous algorithm
        :param groups: controls the connections between inputs and outputs;
               `in_channels` and `out_channels` must both be divisible by
               `groups`
        :param bias: True to enable bias parameters
        """
        nn.Module.__init__(self)
        kernel_size = self._pair(kernel_size)
        stride = self._pair(stride)
        padding = self._pair(padding)
        dilation = self._pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.conv2d = partial(F.conv2d, stride=stride, padding=padding,
                              dilation=dilation, groups=groups)

        self.weight_ih = Parameter(torch.zeros(
                4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = Parameter(torch.zeros(
                4 * out_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias_ih = Parameter(torch.zero(4 * out_channels))
            self.bias_hh = Parameter(torch.zeros(4 * out_channels))
        else:
            # why to use `register_parameter` rather than `Parameter(None)`:
            # `register_parameter(name, None)` prevents from self.name being
            # returned by self.parameters(), while still creating attributes
            # `self.bias_ih` and `self.bias_hh`
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Xavier initialization.
        """
        fanin_i = np.prod(self.weight_ih.size()[1:])
        fanin_h = np.prod(self.weight_hh.size()[1:])

        self.weight_ih.data.normal_(std=1./np.sqrt(fanin_i))
        self.weight_hh.data.normal_(std=1./np.sqrt(fanin_h))
        if self.bias_ih is not None:
            self.bias_ih.data.normal_(std=1./np.sqrt(fanin_i))
        if self.bias_hh is not None:
            self.bias_hh.data.normal_(std=1./np.sqrt(fanin_h))

    def forward(self, inputs, hidden):
        """
        :param inputs: should be of size [B x Cx x H x W]
        :param hidden: tuple of hidden state and cell state, each should be of
               size [B x Ch x H x W]
        :return: the hidden state (next time step), and the tuple of the hidden
                 state (next time step) and the cell state (next time step)
        """
        h, c = hidden

        wx = self.conv2d(inputs, self.weight_ih, bias=self.bias_ih)  # type: Variable
        wh = self.conv2d(h, self.weight_hh, bias=self.bias_hh)  # type: Variable
        wx_wh = wx + wh  # type: Variable

        c_wxh = wx_wh.size(1) // 4
        assert not c_wxh % 4
        i = F.sigmoid(wx_wh[:, c_wxh])
        f = F.sigmoid(wx_wh[:, c_wxh : 2*c_wxh])
        g = F.tanh(wx_wh[:, 2*c_wxh : 3*c_wxh])
        o = F.sigmoid(wx_wh[:, 3*c_wxh])

        c_new = f * c + i * g
        h_new = o * F.tanh(c_new)
        assert h_new.size() == h.size(), \
            'h_new/h size mismatch: {}/{}'.format(h_new.size(), h.size())
        return h_new, (h_new, c_new)

    @staticmethod
    def _pair(x):
        # Ported torch.nn.modules.utils._pair here to perform python stubbing,
        # the stubbing will be performed only under context of `ConvLSTMCell`,
        # and that's why `_pair` is placed as a static method of the class.
        if isinstance(x, collections.Iterable):
            return x
        return tuple(itertools.repeat(x, 2))


class SatLU(nn.Module):
    """
    The saturating linear unit, with upper limit.
    Equation: $y = min(x, upper)$.
    """
    def __init__(self, upper):
        nn.Module.__init__(self)
        self.upper = upper

    def forward(self, x):
        # noinspection PyArgumentList
        return torch.clamp(x, max=self.upper)

    def __repr__(self):
        tmpl = '{} (\n  max: {}\n)'
        return tmpl.format(type(self).__name__, self.upper)


class PredNetForwardConv(nn.Module):
    """
    The $A_l$ block in PredNet. For convenience, this module serves as an
    identity mapping if `with_error` is set to False; otherwise, the input
    to `forward` method is regarded as the prediction error signal and will
    go through `conv`, `nn.MaxPool2d(2)` and `nn.ReLU()`.
    """

    def __init__(self, conv=None, with_error=True):
        """
        :param conv: the convolutional module, can be either a sequence of
               `nn.Conv2d` instances or a single `nn.Conv2d`; the last two
               dimensions of the input/output tensor, namely the height and the
               width, should not be changed by `conv`.
        :param with_error: True if the returned target is based on the error
               signal

        `conv` will be ignored if `with_error` is False; `conv` must not be None
        if `with_error` is True.
        """
        nn.Module.__init__(self)

        if with_error:
            if conv is None:
                raise ValueError('`conv` must not be None if `with_error`')
            # if `conv` is nn.Sequential or similar object, verify that all its
            # children modules are of type `nn.Conv2d`
            for child in conv.children():
                if not isinstance(child, nn.Conv2d):
                    raise ValueError('child of `conv` must be of type '
                                     '`torch.nn.Conv2d`')

        self.with_error = with_error
        if with_error:
            self.conv = conv
            self._postproc = nn.Sequential(nn.MaxPool2d(2), nn.ReLU())

    def forward(self, x):
        if self.with_error:
            y = self.conv(x)
            assert x.size()[-2:] == y.size()[-2:], \
                '`conv` changes the last two dimension at output tensor, from ' \
                '{} to {}; check the padding of `conv`'.format(x.size()[-2:],
                                                               y.size()[-2:])
            return self._postproc(y)
        return x
