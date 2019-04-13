"""
Global library for other dedicated library or project/branch-specific codes.
"""
import itertools
import collections
import os
import io
import re
import sys
import contextlib
import copy
import shutil
import zipfile
import tempfile
import logging
import typing

import numpy as np
import cv2
import torch

from loglib import loggername

T1 = typing.TypeVar('T1')
T2 = typing.TypeVar('T2')


@contextlib.contextmanager
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


@contextlib.contextmanager
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
    Yield frames in numpy array of shape (H, W, 3) where '3' stands for RGB,
    'H' the height and 'W' the width. The number of frames is at most ``n``.
    If ``n`` is not specified, it's default to infinity.

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
    :return: function that align the index of each tuple yielded by
             `enumerate`
    """
    width = int(np.ceil(np.log10(max_count)))

    def _aligned_enum(t):
        return str(t[0]).rjust(width, '0'), t[1]

    return _aligned_enum


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout, sys.stdout = sys.stdout, devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextlib.contextmanager
def memmapcontext(filename, dtype, shape, offset=0, mode='r'):
    """
    :param filename: the memory mapped filename
    :type filename: str
    :param dtype: data type
    :type dtype: typing.Union[str, np.dtype]
    :param shape: data shape
    :type shape: typing.Tuple[int]
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


def fcompose(funcs: typing.Sequence[typing.Callable], star=False) \
        -> typing.Callable:
    """
    Compose any number of callable objects into one callable object, which is
    the functional composition of all the objects from left to right. Each
    callable object should expect one single positional argument.

    :param funcs: a sequence of functions to serially evaluate
    :param star: set to True to allow the first function in ``funcs`` to
           accept any number of positional and/or keyword arguments, while
           other functions accepting one single positional argument as before;
           note the difference with ``fstarcompose``
    :return: the functional composition

    >>> def sum(values):
    ...     x, y = values
    ...     return x + y
    >>> def negate(value):
    ...     return -value
    >>> fcompose([sum, negate])((1, 2))
    -3
    """

    if not star:
        def _wrapped(arg):
            if not funcs:
                return None
            for f in funcs:
                arg = f(arg)
            return arg
    else:
        def _wrapped(*args, **kwargs):
            if not funcs:
                return None
            arg = funcs[0](*args, **kwargs)
            for f in funcs[1:]:
                arg = f(arg)
            return arg
    return _wrapped


def fstarcompose(funcs) -> typing.Callable:
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


def broadcast_value2list(value: T1, iterable: typing.Iterable[T2]) \
        -> typing.Iterator[typing.Tuple[T1, T2]]:
    """
    >>> list(broadcast_value2list(4, [2, 3, 5]))
    [(4, 2), (4, 3), (4, 5)]
    """
    return map(lambda x: (value, x), iterable)


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
            gygx, = torch.autograd.grad(outputs[c], inputs,
                                        retain_graph=i < n - 1)
            gradmaps[c].copy_(gygx)

    return gradmaps


def ituple_k(x: typing.Union[int, typing.Sequence[int]], k=2) -> \
        typing.Sequence[int]:
    """
    Expand an integer to a k-tuple of that integer if it's not already a
    k-tuple. Note that the *integer* type is not enforced -- it only appears
    in type hint. When expanding, the original ``x`` is shallow copied.
    If ``x`` is a list, it will be first coverted to a tuple.

    :param x: an integer or a k-tuple of integers
    :param k: the desired length of the returned tuple
    :return: the original tuple or the expanded tuple
    :raise ValueError: if ``x`` is a tuple but not of length ``k``

    >>> ituple_k(10)
    (10, 10)
    >>> ituple_k((1, 2))
    (1, 2)
    >>> ituple_k([1, 2])
    (1, 2)
    """
    if isinstance(x, (list, tuple)):
        x = tuple(x)
        if len(x) != k:
            raise ValueError('Expected len(x) to be {} but got {}'
                             .format(k, len(x)))
    else:
        _x = tuple(copy.copy(x) for _ in range(k))
        x = _x
    return x


# noinspection PyTypeChecker
def ituple_2(x: typing.Union[int, typing.Tuple[int, int]]) -> typing.Tuple[
    int, int]:
    """Shortcut to ``ituple_k(*, k=2)``."""
    return ituple_k(x, k=2)


# noinspection PyTypeChecker
def ituple_3(x: typing.Union[int, typing.Tuple[int, int, int]]) -> \
        typing.Tuple[int, int, int]:
    """Shortcut to ``ituple_k(*, k=3)``."""
    return ituple_k(x, k=3)


class _ValidIntDecider:
    def __init__(self, mode: str, arg):
        # (p)refix, (s)uffix, (n)one, (a)ll, (e)numeration
        if mode not in 'psnae':
            raise ValueError('Illegal mode, must be one of {}'
                             .format(list('psnae')))
        self.arg = arg
        self.mode = mode

    def __call__(self, x: int) -> bool:
        if self.mode == 'p':
            v = (x < self.arg)
        elif self.mode == 's':
            v = (x >= self.arg)
        elif self.mode == 'n':
            v = False
        elif self.mode == 'a':
            v = True
        else:
            v = (x in self.arg)
        return v


class _ValidIntSeqDecider:
    def __init__(self, ideciders):
        self.ideciders = ideciders

    def __call__(self, x: typing.Sequence[int]) -> bool:
        if len(self.ideciders) != len(x):
            raise ValueError('Expecting int sequence of length {} but got {}'
                             .format(len(self.ideciders), len(x)))
        for d, e in zip(self.ideciders, x):
            if not d(e):
                return False
        return True


def new_int_filter(pat: str) -> typing.Callable[[int], bool]:
    """
    Returns a function that decides if a (nonnegative) integer is in a set
    specified by ``irngpat``. Possible patterns of ``irngpat``:

        - ``COMMA_SEPARATED_LIST_OF_INTEGER``: matching integers that are in
          the enumeration of nonnegative integers
        - ``:``: matching all integers
        - ``INTEGER:``: matching all integers at least INTEGER
        - ``:INTEGER``: matching all integers less than INTEGER

    :param pat: matching integers
    :return: the decider function
    :raise ValueError: if ``pat`` is not one of the above patterns

    >>> f = new_int_filter(':')
    >>> all(map(f, range(100)))
    True
    >>> f = new_int_filter('0:')
    >>> all(map(f, range(100)))
    True
    >>> f = new_int_filter('44:')
    >>> any(map(f, range(0, 44))), all(map(f, range(44, 100)))
    (False, True)
    >>> f = new_int_filter(':44')
    >>> all(map(f, range(0, 44))), any(map(f, range(44, 100)))
    (True, False)
    >>> f = new_int_filter('1,2,3,5,8')
    >>> f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)
    (False, True, True, True, False, True, False, False, True)
    >>> any(map(f, range(9, 100)))
    False
    >>> import random
    >>> idomain = list(range(100))
    >>> gpolicies = 'psean'  # (p)refix, (s)uffix, (e)umeration, (a)ll, (n)one
    >>> for gpo in random.choices(gpolicies, k=100000):
    ...     if gpo in 'ps':
    ...         m = random.choice(idomain)
    ...         if gpo == 'p':
    ...             ans = idomain[:m]
    ...             pat = ':{}'.format(m)
    ...         else:
    ...             ans = idomain[m:]
    ...             pat = '{}:'.format(m)
    ...     elif gpo == 'e':
    ...         ans = random.choices(idomain, k=random.choice(range(1, 100)))
    ...         pat = ','.join(map(str, ans))
    ...     elif gpo == 'a':
    ...         ans = idomain
    ...         pat = ':'
    ...     else:
    ...         ans = []
    ...         pat = ''
    ...     f = new_int_filter(pat)
    ...     for x in ans:
    ...         assert f(x), 'pat={} FN_at={}'.format(pat, x)
    ...     for x in (set(idomain) - set(ans)):
    ...         assert not f(x), 'pat={} FP_at={}'.format(pat, x)
    """
    asp_patterns = [  # asp -- (a)xis (s)ub (p)attern
        ('s', re.compile(r'^(\d+):$')),
        ('p', re.compile(r'^:(\d+)$')),
        ('e', re.compile(r'^(\d+(,\d+)*)$')),
        ('a', re.compile(r'^:$')),
        ('n', re.compile(r'^$')),
    ]
    for name, asppat in asp_patterns:
        matched = asppat.match(pat)
        if matched:
            arg = None
            if name == 'e':
                arg = frozenset(map(int, matched.group(1).split(',')))
            elif name in 'na':
                pass
            else:
                arg = int(matched.group(1))
            return _ValidIntDecider(name, arg)
    raise ValueError('Illegal pattern `{}\''.format(pat))


def new_ituple_filter(pat: str) -> typing.Callable[
    [typing.Sequence[int]], bool]:
    """
    Adapting ``new_int_filter`` to multiple nonnegative integer scenario, such
    that the enumeration of integers must be enclosed by a pair of square
    brackets.

    :param pat: the pattern describing the decider function
    :return: the decider function
    :raise ValueError: if ``pat`` is illegal or ``tlen`` is not positive

    >>> f = new_ituple_filter('30:')
    >>> f((32,))
    True
    >>> f = new_ituple_filter('55:,65:')
    >>> f((57,59))
    False
    >>> f = new_ituple_filter('[82,46,2,57,56,39,16,8,89,56,43,72,36,96,75,'
    ...                       '93,53,57,9,16,59,41,19,31]')
    >>> import random
    >>> idomain = list(range(100))
    >>> ldomain = list(range(1, 3))
    >>> gpolicies = 'psean'  # (p)refix, (s)uffix, (e)umeration, (a)ll, (n)one
    >>> for gpo in random.choices(gpolicies, k=1000):
    ...    l = random.choice(ldomain)
    ...    anss = []
    ...    pats = []
    ...    for _ in range(l):
    ...        if gpo in 'ps':
    ...            m = random.choice(idomain)
    ...            if gpo == 'p':
    ...                ans = idomain[:m]
    ...                pat = ':{}'.format(m)
    ...            else:
    ...                ans = idomain[m:]
    ...                pat = '{}:'.format(m)
    ...        elif gpo == 'e':
    ...            ans = random.choices(idomain, k=random.choice(range(1, 100)))
    ...            pat = '[{}]'.format(','.join(map(str, ans)))
    ...        elif gpo == 'a':
    ...            ans = idomain
    ...            pat = ':'
    ...        else:
    ...            ans = []
    ...            pat = ''
    ...        anss.append(tuple(ans))
    ...        pats.append(pat)
    ...    pat = ','.join(pats)
    ...    f = new_ituple_filter(pat)
    ...    allcombs = set(itertools.combinations_with_replacement(idomain, l))
    ...    anscombs = set(itertools.product(*anss))
    ...    for x in anscombs:
    ...        assert f(x), 'pat={} FN_at={}'.format(pat, x)
    ...    for x in (set(allcombs) - set(anscombs)):
    ...        assert not f(x), 'pat={} FP_at={}'.format(pat, x)
    """
    axis_pattern = re.compile(
        r'^((?P<r>\d+|\d+:|:\d+|:|)|\[(?P<e>\d+(,\d+)*)\])(,|$)')

    patprefix = pat
    kfs = []
    while True:
        matched = axis_pattern.match(patprefix)
        if not matched:
            raise ValueError('Illegal pattern `{}\''.format(pat))
        for ky, apat in matched.groupdict().items():
            if apat is not None:
                break
        else:
            assert False
        kfs.append(new_int_filter(apat))
        try:
            comma_loc = patprefix.index(',', matched.end(ky))
        except ValueError:
            break
        else:
            patprefix = patprefix[comma_loc + 1:]

    return _ValidIntSeqDecider(tuple(kfs))


def parse_cmd_kwargs(cmd_words: typing.Sequence[str],
                     kv_delim='=') -> typing.Dict[str, str]:
    d = {}
    for w in cmd_words:
        key, value = w.split(kv_delim, maxsplit=1)
        d[key] = value
    return d


@contextlib.contextmanager
def suppress_numpy_warning(**kwargs):
    """
    Context manager that temporarily suppress some numpy warnings.
    :param kwargs: the keyword arguments to be passed to ``numpy.seterr``
    """
    old_errstate = np.seterr(**kwargs)
    try:
        yield
    finally:
        np.seterr(**old_errstate)


class ShapeMismatchError(ValueError):
    pass


def check_shape(array, *ref_shapes: typing.Sequence[typing.Optional[int]]):
    """
    Check shape of ``array` against a number of expected shapes. Fail the test
    if none of ``ref_shapes`` matches the shape of ``array``, or return the
    index of first match in ``ref_shapes``.

    :param array: any type with attribute ``shape`` that returns a sequence
           of ``int``s, or the ``shape`` itself if ``hasattr(array, 'shape')``
           fails
    :param ref_shapes: referential shape, where ``None`` denotes any int
    :return: the index of the first matched ``ref_shapes``
    :raise ShapeMismatchError: if none matches; note that
           ``ShapeMismatchError`` is a type of ``ValueError``
    """
    try:
        sh: typing.Sequence[int] = array.shape
    except AttributeError:
        sh: typing.Sequence[int] = array
    for i, ref in enumerate(ref_shapes):
        if len(sh) == len(ref) and all(x == y for x, y in zip(sh, ref)
                                       if y is not None):
            return i
    raise ShapeMismatchError('Unexpected shape {}; should be one of {}'
                             .format(sh, list(ref_shapes)))


ShapeHW = collections.namedtuple('ShapeHW', 'H W')
ShapeCHW = collections.namedtuple('ShapeCHW', 'C H W')
ShapeHWC = collections.namedtuple('ShapeHWC', 'H W C')
ShapeBHWC = collections.namedtuple('ShapeBHWC', 'B H W C')
ShapeBCHW = collections.namedtuple('ShapeBCHW', 'B C H W')
ShapeBTCHW = collections.namedtuple('ShapeBTCHW', 'B T C H W')
ShapeBHWD = collections.namedtuple('ShapeBHWD', 'B H W D')


class IncrementalNpzWriter:
    """
    Write data to npz file incrementally rather than compute all and write
    once, as in ``np.save``. This class can be used with ``contextlib.closing``
    to ensure closed after usage.
    """

    def __init__(self, tofile, mode: str = 'x'):
        """
        :param tofile: the ``npz`` file to write
        :param mode: must be one of {'x', 'w', 'a'}. See
               https://docs.python.org/3/library/zipfile.html for detail
        """
        self.tofile = zipfile.ZipFile(tofile, mode=mode,
                                      compression=zipfile.ZIP_DEFLATED)

    def write(self, key: str, data: typing.Union[np.ndarray, bytes],
              is_npy_data: bool = True) -> None:
        """
        :param key: the name of data to write
        :param data: the data
        :param is_npy_data: if ``True``, ".npz" will be appended to ``key``,
               and ``data`` will be serialized by ``np.save``;
               otherwise, ``key`` will be treated as is, and ``data`` will be
               treated as binary data
        :raise KeyError: if the transformed ``key`` (as per ``is_npy_data``)
               already exists in ``self.tofile``
        """
        if key in self.tofile.namelist():
            raise KeyError('Duplicate key "{}" already exists in "{}"'
                           .format(key, self.tofile.filename))
        self.update(key, data, is_npy_data=is_npy_data)

    def update(self, key: str, data: typing.Union[np.ndarray, bytes],
               is_npy_data: bool = True) -> None:
        """
        Same as ``self.write`` but overwrite existing data of name ``key``.

        :param key: the name of data to write
        :param data: the data
        :param is_npy_data: if ``True``, ".npz" will be appended to ``key``,
               and ``data`` will be serialized by ``np.save``;
               otherwise, ``key`` will be treated as is, and ``data`` will be
               treated as binary data
        """
        kwargs = {
            'mode': 'w',
            'force_zip64': True,
        }
        if is_npy_data:
            key += '.npy'
            with io.BytesIO() as cbuf:
                np.save(cbuf, data)
                cbuf.seek(0)
                with self.tofile.open(key, **kwargs) as outfile:
                    shutil.copyfileobj(cbuf, outfile)
        else:
            with self.tofile.open(key, **kwargs) as outfile:
                outfile.write(data)

    def close(self):
        if self.tofile is not None:
            self.tofile.close()
            self.tofile = None


class _TempMMap:
    def __init__(self, data_source, mmap_mode):
        self.cbuf = tempfile.NamedTemporaryFile(delete=False)
        try:
            with contextlib.closing(data_source):
                shutil.copyfileobj(data_source, self.cbuf)
        except:
            self.close()
            raise
        else:
            self.close(_delete=False)
        self.mmap_mode = mmap_mode

    def open(self):
        """
        :return: the memory-mapped array
        """
        return np.load(self.cbuf.name, mmap_mode=self.mmap_mode)

    def close(self, _delete=True):
        """
        Close and release the memory-mapped file.

        :param _delete: user should not modify this argument
        """
        logger = self._l(self.close.__name__)
        if self.cbuf is not None:
            self.cbuf.close()
        if _delete and self.cbuf is not None:
            try:
                os.remove(self.cbuf.name)
            except FileNotFoundError:
                self.cbuf = None
            except:
                logger.error('Error removing temp file "%s"', self.cbuf.name)
                raise
            else:
                self.cbuf = None

    def __enter__(self):
        return self.open()

    def __exit__(self, _1, _2, _3):
        self.close()

    @classmethod
    def _l(cls, method_name: str = None) -> logging.Logger:
        tokens = [__name__, cls.__name__]
        if method_name:
            tokens.append(method_name)
        return logging.getLogger(loggername(*tokens))


class NpzMMap:
    """
    Example usage::

        .. code-block::

            my_npzfile = ...
            with NpzMMap(my_npzfile) as zfile:
                with zfile.mmap(data_name) as data:
                    # do anything to memory-mapped ``data``
                    ...
    """

    def __init__(self, npzfile) -> None:
        """
        :param npzfile: anything representing an npz file that can be
               accepted by ``numpy.load``
        """
        self.npzfile = npzfile
        with np.load(self.npzfile) as zdata:
            self.npzkeys = set(zdata)
        self._zfile = zipfile.ZipFile(self.npzfile)

    def close(self):
        if self._zfile is not None:
            self._zfile.close()

    def mmap(self, key: str, mmap_mode: str = 'r'):
        """
        :param key: which entry in ``self.npzfile`` to memory-map.
        :param mmap_mode: see ``help(numpy.load)`` for detail; default to 'r'
        :return: memory-mapped file
        :raise KeyError: if ``key`` is not in ``keys()`` of ``self.npzfile``
        :raise ValueError: if ``mmap_mode`` is ``None`` or equivalent
        """
        if key not in self.npzkeys:
            raise KeyError('key "{}" not in npzfile "{}"'
                           .format(key, self.npzfile))
        if not mmap_mode:
            raise ValueError('mmap_mode must not be empty')
        if mmap_mode != 'r':
            raise NotImplementedError
        if key not in self._zfile.namelist():
            key += '.npy'
        assert key in self._zfile.namelist(), str(key)
        return _TempMMap(self._zfile.open(key), mmap_mode)

    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        self.close()
