from contextlib import contextmanager
import multiprocessing
import numpy as np
import cv2


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
    Yield frames in numpy array of dimension [W x H x 3] where '3' stands
    for RGB, 'W' the width and 'H' the height.
    """

    def __init__(self, cap, max_len=None):
        """
        :param cap: the video capture object
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
    yield pool
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
    :return: function that align the index of each tuple yielded by `enumerate`
    """
    width = int(np.ceil(np.log10(max_count)))

    def _aligned_enum(t):
        return str(t[0]).rjust(width, '0'), t[1]
    return _aligned_enum
