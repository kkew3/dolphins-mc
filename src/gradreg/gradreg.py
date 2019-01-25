import contextlib
from types import SimpleNamespace
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.autograd import grad


@contextlib.contextmanager
def no_grad_params(m: Optional[nn.Module]=None,
                   params: Optional[Iterable[nn.Parameter]]=None):
    """
    Disable gradient calculation temporarily on a particular module or a set
    of parameters. This context manager is different from ``torch.no_grad``
    in that the latter disables all gradient calculation.

    :param m: module to disable gradient calculation
    :param params: the parameters of ``m`` to disable gradient

    When both ``m`` and ``params`` are specified, ``m`` will be ignored. When
    neither ``m`` nor ``params`` is specified, this context manager serves
    identical as ``torch.no_grad``.
    """
    disable_all = ((m, params) == (None, None))
    if not disable_all:
        if params is None:
            params = m.parameters()
        params = list(params)
        orig_grad_states = [p.requires_grad for p in params]
    else:
        torch.set_grad_enabled(False)
    try:
        yield
    finally:
        if not disable_all:
            for p, s in zip(params, orig_grad_states):
                p.requires_grad = s
        else:
            torch.set_grad_enabled(True)


@contextlib.contextmanager
def gradreg(inputs, strength=0.0, reg_method=None, train=True,
            normalize_wrt_batch=False):
    r"""
    Apply gradient regularization loss to a provided loss function. Denote the
    loss function (including the network itself), parameterized by
    :math:`\theta`, as :math:`L(x;\theta)`. Denote the regularization strength
    as :math:`\lambda`. Denote the regularization function, default to
    :math:`L_1` norm, as :math:`g`. Then the gradient regularized loss is
    given by:

        .. math::

            L_g(x;\theta) = L(x;\theta) + \lambda g\left(
                \frac{\partial}{\partial x}L(x;\theta)\right)

    This context manager returns a namespace. By the end of the ``with``
    block, exactly one attribute (of any name) needs to be set into the
    namespace. The value of the attribute corresponds to the value of
    :math:`L(x;\theta)`. The reason why to use a namespace is to retain
    reference to the same object in assignment. After the ``with`` block, the
    value of the attribute becomes :math:`L_g(x;\theta)`. Now backward can be
    done on it. The backward must not be done within the ``with`` block.

    Pseudocode usage::

        .. code-block:: python

            optimizer = SGD(chain(encoder.parameters(), decoder.parameters()))
            with gradreg(x) as ns:
                code = encoder(x)
                y = decoder(code)
                ns.loss = criterion(y, targets)
            optimizer.zero_grad()
            ns.loss.backward()
            optimizer.step()

    Note that ``ns.loss`` (or whatever name) must be a scalar. This means that
    if ``x`` is a batch of inputs, then ``ns.loss`` should be the reduced loss
    of the batch of losses, e.g. the average loss.

    :param inputs: the :math:`x` in the above equation, of shape (B, ...)
    :param strength: the regularization strength
    :param reg_method: a function that takes as input the gradient with
           respect to ``inputs`` and returns a scalar gradient
           regularization loss; default to L1 norm
    :param train: when set to ``False``, do not create graph for 2nd order
           derivative
    :param normalize_wrt_batch: ``True`` to normalize the gradient
           regularization by the batch size of ``inputs``; this argument will
           be ignored when ``reg_method`` is not ``None``

    Pseudocode usage when ``train=False``::

        .. code-block:: python

            with no_grad_params(encoder), no_grad_params(decoder):
                with gradreg(x, train=False) as ns:
                    code = encoder(x)
                    y = decoder(code)
                    ns.loss = criterion(y, targets)
                print(ns.loss.item())

    A real-life example to compute :math:`\frac{\partial}{\partial w,b}L_g(x;w,b)`
    where :math:`L_g(x;w,b)=w^Tx+b+\|\frac{(w^Tx+b)}{\partial x}\|_1. The
    correct answer should be :math:`x+\text{sign}(w)` for :math:`w` and 1.0
    for :math:`b`.

    >>> import torch
    >>> import torch.nn as nn
    >>> x = torch.rand(3)
    >>> net = nn.Linear(3, 1)
    >>> with gradreg(x, strength=1.0) as ns:
    ...     ns.loss = net(x)
    >>> ns.loss.backward()
    >>> assert x.grad is None
    >>> wg = net.weight.grad.view(-1)
    >>> bg = net.bias.grad.item()
    >>> sw = torch.sign(net.weight.data.view(-1))
    >>> wgt = x + sw
    >>> assert torch.all(wg == wgt)
    >>> assert bg == 1.0
    """
    xrg = inputs.requires_grad
    inputs.requires_grad_()
    ns = SimpleNamespace()
    try:
        yield ns
    except:
        raise
    else:
        if len(ns.__dict__) != 1:
            raise ValueError('Exactly one attribute needs to be assigned')
        yattr = next(iter(ns.__dict__))
        y = getattr(ns, yattr)
        if torch.numel(y) > 1:
            raise ValueError('Expected {} to be scalar, but of shape {}'
                             .format(yattr, y.size()))
        dx = grad(y, [inputs], create_graph=train)[0]
        if reg_method:
            gp = reg_method(dx)
        else:
            gp = torch.norm(dx, p=1)
            if normalize_wrt_batch:
                gp = gp / inputs.size(0)
        inputs.requires_grad = xrg
        setattr(ns, yattr, y + strength * gp)
    finally:
        inputs.requires_grad = xrg
