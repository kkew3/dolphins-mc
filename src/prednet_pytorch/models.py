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
        :param stride: controls the stride for the cross-correlation, a single
               number of a tuple
        :param padding: controls the amount of implicit zero-paddings on both
               sides for padding number of points for each dimension
        :param dilation: controls the spacing between the kernel points; also
               known as the atrous algorithm
        :param groups: controls the connections between inputs and outputs;
               ``in_channels`` and ``out_channels`` must both be divisible by
               ``groups``
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
        :param inputs: should be of size
               :math:`[B \\times C_x \\times H \\times W]`
        :param hidden: tuple of hidden state and cell state, each should be of
               size :math:`[B \\times C_h \\times H \\times W]`
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
    Equation: :math:`y = \\min\\{x, upper\\}`.
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


class PredNetForwardError(nn.Module):
    """
    Module in *PredNet* that collects the prediction and target representation,
    and emits the error signal.
    """

    def __init__(self, target_as_input=False):
        """
        :param target_as_input: ``True`` if the target representation is the
               exact image in the input space. When this is the case, a
               ``SatLU`` of upper bound 1.0 will be applied to the prediction
               representation at the beginning of the ``forward`` method. The
               upper bound 1.0 is chosen since *PyTorch* normalizes pixel values
               to within range :math:`[0, 1]`
        """
        nn.Module.__init__(self)
        self.target_as_input = target_as_input
        if self.target_as_input:
            self.satlu = SatLU(upper=1.0)  # preprocessing of the *input* prediction
        self.relu = nn.ReLU()  # the activation function in error module

    def forward(self, predictions, targets):
        """
        :param predictions: the predicted representations. Variable of dimension
               :math:`[B \\times C_l \\times H_l \\times W_l]`, where
               :math:`C_l`, :math:`H_l` and :math:`W_l` are the number of
               channels, height and width of the :math:`l`-th representation
        :param targets: the target representations, Variable of dimension
               :math:`[B \\times C_l \\times H_l \\times W_l]`, where
               :math:`C_l`, :math:`H_l` and :math:`W_l` are the number of
               channels, height and width of the :math:`l`-th representation
        :return: the error signal, of dimension
                 :math:`[B \\times 2C_l \\times H_l \\times W_l]`
        """
        # sanity check (for debugging)
        assert predictions.size() == targets.size(), 'predictions/targets size ' \
                                                     'mismatch: {}/{}'.format(
                predictions.size(), targets.size())

        if self.target_as_input:
            predictions = self.satlu(predictions)  # type: Variable
        pos_err = self.relu(predictions - targets)
        neg_err = self.relu(targets - predictions)
        err = torch.cat((pos_err, neg_err), dim=1)  # concatenate on channels
        return err


class PredNetForwardTarget(nn.Module):
    """
    Module in *PredNet* that collects the error signal and emits the target
    representation for the next layer. This module is an aggregation of
    ``nn.Conv2d(...)``, ``nn.ReLU()`` and ``nn.MaxPool2d(2)``.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=True):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride: controls the stride for the cross-correlation, a single
               number of a tuple
        :param padding: controls the amount of implicit zero-paddings on both
               sides for padding number of points for each dimension
        :param dilation: controls the spacing between the kernel points; also
               known as the atrous algorithm
        :param groups: controls the connections between inputs and outputs;
               ``in_channels`` and ``out_channels`` must both be divisible by
               ``groups``
        :param bias: True to enable bias parameters
        """
        nn.Module.__init__(self)
        self.features = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, dilation=dilation,
                          groups=groups, bias=bias),
                nn.ReLU(),
                nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.features(x)


class PredNetPrediction(nn.Module):
    """
    Module in *PredNet* that accepts the output of the representation unit and
    emits the prediction of the target representation.
    """
    def __init__(self):
        nn.Module.__init__(self)
        pass
