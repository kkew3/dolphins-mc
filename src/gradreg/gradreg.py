import contextlib
import functools
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.autograd import grad


@contextlib.contextmanager
def gradreg(inputs, strength=0.0, reg_method=functools.partial(torch.norm, p=1)):
    r"""
    Apply gradient regularization loss to a provided loss function. Denote the
    loss function (including the network itself), parameterized by
    :math:`\theta`, as :math:`L(x;\theta)`. Denote the regularization strength
    as :math:`\lambda`. Denote the regularization function, default to
    :math:`L_1` norm, as :math:`g`. Then the gradient regularized loss is
    given by:

        .. math::

            L_g(x;\theta) = L(x;\theta) + \lambda g\left(\frac{\partial}{\partial x}L(x;\theta)\right)

    This context manager returns a namespace. By the end of the ``with``
    block, exactly one attribute (of any name) needs to be set into the
    namespace. The value of the attribute corresponds to the value of
    :math:`L(x;\theta)`. The reason why to use a namespace is to retain
    reference to the same object in assignment. After the ``with`` block, the
    value of the attribute becomes :math:`L_g(x;\theta)`. Now backward can be
    done on it. The backward must not be done within the ``with`` block.

    Pseudocode usage:

        .. code-block:: python

            optimizer = SGD(chain(encoder.parameters(), decoder.parameters()))
            with gradreg(x) as ns:
                code = encoder(x)
                y = decoder(code)
                ns.loss = criterion(y, targets)
            optimizer.zero_grad()
            ns.loss.backward()
            optimizer.step()

    :param inputs: the :math:`x` in the above equation
    :param strength: the regularization strength
    :param reg_method: a function that takes as input the gradient with
           respect to ``inputs`` and returns a scalar gradient
           regularization loss; default to L1 norm

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
    >>> assert (wg == wgt).all()
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
        gp = reg_method(grad(y, [inputs], create_graph=True)[0])
        inputs.requires_grad = xrg
        setattr(ns, yattr, y + strength * gp)
    finally:
        inputs.requires_grad = xrg
