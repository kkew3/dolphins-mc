from collections import defaultdict, deque
from functools import partial

import numpy as np

class GMLE(object):
    def __init__(self, window_size: int, diffth: float=np.inf):
        """
        :param window_size:
        :param diffth: when the absolute pixel difference is larger than this
               threshold, remains the original pixel rather than use the
               pixel difference
        """
        self.frames = defaultdict(partial(deque, maxlen=window_size))
        self.diffth = diffth

    def __call__(self, cl_frames):
        new_cl_frames = []
        for cl, f in cl_frames:
            if f is None:
                new_cl_frames.append((cl, f))
            else:
                self.frames[cl].append(f)
                if len(self.frames[cl]) < self.frames[cl].maxlen:
                    new_cl_frames.append((cl, np.zeros_like(f)))
                else:
                    d = f - np.mean(self.frames[cl], axis=0)
                    mask = (np.abs(d) > self.diffth)

                    newf = np.array(((f * mask + d * (~mask)) + 255) / 2,
                                    dtype=np.uint8)
                    new_cl_frames.append((cl, newf))
        return new_cl_frames

def frame_processor():
    return GMLE(101)
