"""
Global library for other dedicated library or project/branch-specific codes.
"""

import os
import sys
from contextlib import contextmanager
import multiprocessing

import numpy as np
import cv2
import torchvision.transforms as trans


@contextmanager
def capcontext(video_file):
    """
    Context manager that handles the release of video capture object.
    """
    cap = cv2.VideoCapture(video_file)
    yield cap
    cap.release()


class FrameIterator(object):
    """
    Yield frames in numpy array of dimension [H x W x 3] where '3' stands
    for RGB, 'H' the height and 'W' the width.
    """

    def __init__(self, cap, max_len=None):
        """
        :param cap: the video capture object
        :type cap: cv2.VideoCapture
        :param max_len: the maximum number of frames to read before stopped
        """
        if not cap.isOpened():
            raise ValueError('cap not opened')
        self.cap = cap
        self.max_len = np.inf if max_len is None else max_len
        self._read_count = 0

    def reset_counter(self):
        """
        Reset the read counter, so that another `self.max_len` frames can be
        yielded.
        """
        self._read_count = 0

    def next(self):
        if self._read_count < self.max_len:
            s, f = self.cap.read()
            if s:
                self._read_count += 1
                # BGR -> RGB
                return f[:,:,::-1]
        raise StopIteration()

    def __iter__(self):
        return self


@contextmanager
def poolcontext(*args, **kwargs):
    """
    Use `multiprocessing.Pool` as a context manager. All arguments are passed
    directly to `multiprocessing.Pool` constructor. Usage:

    >>> with poolcontext(processes=1) as pool:
    ...     pass
    """
    pool = multiprocessing.Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.terminate()


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


def clear_frametensor_channel(tensor, c=None):
    """
    Make a specific channel of a frame tensor to all-zero. This function is
    expected to be used as a Lambda transformation within
    ``torchvision.transforms.Compose``.

    :param tensor: the frame tensor of dimension [C x H x W] where C is the
           number of channels, H the height and W the width
    :param c: a list of channels to clear, default to []
    :type c: Sequence[int]
    :return: the original tensor if nothing changed, or a copy of the processed
             tensor
    """
    if not c:
        return tensor

    tensor = tensor.clone()
    for cid in c:
        tensor[cid].zero_()
    return tensor

