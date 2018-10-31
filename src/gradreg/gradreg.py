import contextlib
import functools
import logging
from typing import Callable, Union, Optional, Iterable, Tuple

import more_sampler
import torch
import torch.nn as nn
from torch.autograd import Function, grad
from torch.utils.data import DataLoader


class GradRegularized(nn.Module):
    r"""
    Apply gradient regularization loss to a provided loss function. Denote the
    loss function (including the network itself), parameterized by
    :math:`\theta`, as :math:`L(x;\theta)`. Denote the regularization strength
    as :math:`\lambda`. Denote the regularization function, default to
    :math:`L_1` norm, as :math:`g`. Then the gradient regularized loss is
    given by:

        .. math::

            L_g(x;\theta) = L(x;\theta) + \lambda g\left(\frac{\partial}{\partial x}L(x;\theta)\right)

    Usage
    """
    def __init__(self, strength=0.0,
                 reg_method=functools.partial(torch.norm, p=1)):
        """
        :param strength: the gradient regularization strength
        :param reg_method: a function that takes as input the gradient with
               respect to ``inputs`` and returns a scalar gradient
               regularization loss; default to L1 norm
        """
        super().__init__()
        self.strength = strength
        self.reg_method = reg_method

    def forward(self, x, *args, f=None, **kwargs):
        """
        :param x: the inputs
        :param f: differentiable function of the signature
               ``f(x: torch.Tensor, *args, **kwargs) -> torch.Tensor`` where
               the return value should be a scalar
        """
        xrg = x.requires_grad
        x.requires_grad_()
        y = f(x, *args, **kwargs)
        gp = self.reg_method(grad(y, [x], create_graph=True)[0])
        x.requires_grad = xrg
        y = y + self.strength * gp
        return y
