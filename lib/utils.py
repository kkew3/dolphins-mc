"""
Global library for other dedicated library or project/branch-specific codes.
"""
import itertools
import os
import sys
import torch
from contextlib import contextmanager

from typing import Callable, Sequence, Any, Iterable
import numpy as np
import cv2
import collections


@contextmanager
def capcontext(video_file):
    """
    Context manager that handles the release of video capture object.

    :param video_file: the video file path
    :type video_file: str
    """
    cap = cv2.VideoCapture(video_file)
    try:
        yield cap
    finally:
        cap.release()


@contextmanager
def videowritercontext(filename, fourcc, fps, wh):
    """
    :param filename: filename to write
    :param fourcc: fourcc
    :param fps: frame per second
    :param wh: width and height
    """
    writer = cv2.VideoWriter(filename, fourcc, fps, wh)
    try:
        yield writer
    finally:
        writer.release()


def frameiter(cap: cv2.VideoCapture, n: int = None, rgb: bool = True):
    """
    Yield frames in numpy array of shape (H, W, 3) where '3' stands for RGB, 'H'
    the height and 'W' the width. The number of frames is at most ``n``. If
    ``n`` is not specified, it's default to infinity.

    :param cap: the video capture object
    :param n: at most this number of frames are to be yielded; ``n`` should be
           a nonnegative integer
    :param rgb: True to returns array with color channel RGB, otherwise BGR
    :return: the frames in numpy array
    """
    if n is None:
        n = np.inf
    else:
        n = max(0, int(n))
    yielded_count = 0
    while cap.isOpened():
        if n == yielded_count:
            break
        s, f = cap.read()
        if not s:
            break
        yielded_count += 1
        if len(f.shape) == 3 and f.shape[2] == 3 and rgb:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        yield f


def aligned_enum(max_count):
    """
    Make string alignment to the index returned by `numerate` built-in.
    For example:

    >>> strings = [x for x in 'abcdefghijklmn']
    >>> len(strings)
    14
    >>> ind_aligner = aligned_enum(len(strings))
    >>> enum_strings = list(map(ind_aligner, enumerate(strings)))
    >>> enum_strings[:3] == [('00', 'a'), ('01', 'b'), ('02', 'c')]
    True
    >>> enum_strings[-3:] == [('11', 'l'), ('12', 'm'), ('13', 'n')]
    True

    :param max_count: alignment width
    :type max_count: int
    :return: function that align the index of each tuple yielded by `enumerate`
    """
    width = int(np.ceil(np.log10(max_count)))

    def _aligned_enum(t):
        return str(t[0]).rjust(width, '0'), t[1]

    return _aligned_enum


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout, sys.stdout = sys.stdout, devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextmanager
def memmapcontext(filename, dtype, shape, offset=0, mode='r'):
    """
    :param filename: the memory mapped filename
    :type filename: str
    :param dtype: data type
    :type dtype: Union[str, np.dtype]
    :param shape: data shape
    :type shape: Tuple[int]
    :param offset: offset in units of number of elements
    :param mode: the file open mode, choices: {'r', 'r+', 'w+', 'c'}; see
           https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html#numpy.memmap
    :return:
    """
    mm = np.memmap(filename, mode=mode, shape=shape, dtype=dtype,
                   offset=np.dtype(dtype).itemsize * offset)
    try:
        yield mm
    finally:
        del mm


def fcompose(funcs: Sequence[Callable]) -> Callable:
    """
    Compose any number of callable objects into one callable object, which is
    the functional composition of all the objects from left to right. Each
    callable object should expect one single positional argument.

    >>> def sum(values):
    ...     x, y = values
    ...     return x + y
    >>> def negate(value):
    ...     return -value
    >>> fcompose([sum, negate])((1, 2))
    -3

    :return: the functional composition
    """

    def _wrapped(arg):
        for f in funcs:
            arg = f(arg)
        return arg

    return _wrapped


def fstarcompose(funcs) -> Callable:
    """
    Same as ``fcompose``, except for doing star operation at the invocation of
    each callable objects.

    >>> def orig_sum(x, y):
    ...     return x, x + y
    >>> def product(x, y):
    ...     return x * y
    >>> fstarcompose([orig_sum, product])(2, 3)
    10

    :return: the functional composition
    """

    def _wrapped(*args):
        for f in funcs:
            args = f(*args)
        return args

    return _wrapped


def inf_powerof(num: int, pow: int) -> int:
    """
    Get the nearest number to ``num`` that is smaller than ``num`` and that is
    a power of ``pow``. "inf" stands for "infimum".
    """
    return num - num % pow


def browadcast_value2list(value: Any, iterable: Iterable) -> Iterable:
    """
    >>> list(browadcast_value2list(4, [2, 3, 5]))
    [(4, 2), (4, 3), (4, 5)]
    """
    return map(lambda x: (value, x), iterable)


def loggername(module_name, *args):
    """
    Get hierarchical logger name.

    Usage::

        .. code-block:: python

            loggername(__name__)
            loggername(__name__, self)
            loggername(__name__, 'function_name')
            loggername(__name__, self, 'method_name')
    """
    tokens = [module_name]
    if len(args) > 0:
        if isinstance(args[0], str):
            tokens.append(args[0])
        else:
            tokens.append(type(args[0]).__name__)
    if len(args) > 1:
        tokens.append(args[1])
    return '.'.join(tokens)


def jacobian(outputs: torch.Tensor, inputs: torch.Tensor,
             to_device: str = None) -> torch.Tensor:
    """
    Compute the Jacobian tensor. The returned tensor is placed on the same
    device as ``inputs``.

    :param outputs:
    :param inputs: ``inputs.requires_grad`` must be ``True``
    :param to_device: where to put the gradients; default to the same device
           as the inputs
    :return: a tensor of shape ``outputs.shape + inputs.shape``, such that
             for each ``coor``, ``inputs[coor]`` is the gradient of
             ``outputs[coor]``, where ``coor`` is a coordinate tuple of length
             ``len(outputs.size())``

    >>> M = torch.tensor([[0.0745, 0.4937],
    ...                   [0.7884, 0.7944]])
    >>> X = torch.tensor([[0.2161, 0.3782],
    ...                   [0.9080, 0.2498]], requires_grad=True)
    >>> Y = torch.mm(torch.t(X), torch.mm(M, X))
    >>> gYgX = jacobian(Y, X)
    >>> gYgX.size()
    torch.Size([2, 2, 2, 2])
    >>> expected = torch.tensor([
    ...        [[[1.1963, 0.0000],
    ...          [1.7197, 0.0000]],
    ...         [[0.1515, 0.7320],
    ...          [0.4966, 0.8280]]],
    ...        [[[0.2251, 0.4644],
    ...          [0.3852, 0.8917]],
    ...         [[0.0000, 0.3766],
    ...          [0.0000, 0.8818]]]])
    >>> # there might be some error in hand-written expected tensor ...
    >>> bool(torch.max(torch.abs(gYgX - expected)) < 1e-4)
    True
    >>> x = torch.tensor(0.2876, requires_grad=True)
    >>> y = torch.sigmoid(x * x)
    >>> x_ = x.detach()
    >>> expected = torch.sigmoid(x_ * x_) * (1 - torch.sigmoid(x_ * x_)) * 2 * x_
    >>> gygx = jacobian(y, x)
    >>> torch.allclose(expected, gygx)
    True
    """
    if not to_device:
        to_device = inputs.device

    if not len(outputs.size()):
        gradmaps = torch.empty_like(inputs, device=to_device)
        gygx, = torch.autograd.grad(outputs, inputs)
        gradmaps.copy_(gygx)
    else:
        gradmaps = torch.zeros(*(outputs.size() + inputs.size())).to(to_device)
        coors = list(itertools.product(*map(range, outputs.size())))
        n = len(coors)
        for i, c in enumerate(coors):
            gygx, = torch.autograd.grad(outputs[c], inputs, retain_graph=i < n - 1)
            gradmaps[c].copy_(gygx)

    return gradmaps
