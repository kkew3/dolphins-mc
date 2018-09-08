"""
Global library for other dedicated library or project/branch-specific codes.
"""

import os
import sys
from contextlib import contextmanager
from typing import Iterator
import multiprocessing

import numpy as np
import cv2


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


def frameiter(cap, n=None):
    """
    Yield frames in numpy array of shape (H, W, 3) where '3' stands for RGB, 'H'
    the height and 'W' the width. The number of frames is at most ``n``. If
    ``n`` is not specified, it's default to infinity.

    :param cap: the video capture object
    :type cap: cv2.VideoCapture
    :param n: at most this number of frames are to be yielded; ``n`` should be
           a nonnegative integer
    :type n: int
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
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        yield f


@contextmanager
def poolcontext(*args, **kwargs):
    """
    Use ``multiprocessing.Pool`` as a context manager. All arguments are passed
    directly to ``multiprocessing.Pool`` constructor. Usage:

    >>> with poolcontext(processes=1) as pool:
    ...     pass
    """
    pool = multiprocessing.Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.close()


def aligned_enum(max_count):
    """
    Make string alignment to the index returned by `numerate` built-in.
    For example:

    >>> strings = [x for x in 'abcdefghijklmn']
    >>> len(strings)
    14
    >>> ind_aligner = aligned_enum(len(strings))
    >>> enum_strings = map(ind_aligner, enumerate(strings))
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
    :param mode:
    :return:
    """
    mm = np.memmap(filename, mode=mode, shape=shape, dtype=dtype,
                   offset=np.dtype(dtype).itemsize * offset)
    try:
        yield mm
    finally:
        del mm
