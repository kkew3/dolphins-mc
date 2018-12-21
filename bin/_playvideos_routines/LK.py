import os
from collections import defaultdict, deque
from functools import partial

import numpy as np
import cv2

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

maxCorners = 40
blockSize = 7
colors = np.random.randint(0, 255, size=(maxCorners, 3))  # type: np.ndarray
shitomasi_eigval_lb = 2e-4


def goodFeaturesToTrack(f):
    q_eigval = np.percentile(cv2.cornerMinEigenVal(f, blockSize=blockSize), 100)
    # print('>>> mineigval:', mineigval)
    if q_eigval >= shitomasi_eigval_lb:
        return cv2.goodFeaturesToTrack(f, mask=None,
                                       maxCorners=maxCorners,
                                       qualityLevel=0.3,
                                       minDistance=14,
                                       blockSize=blockSize)
    else:
        return None


def calcOpticalFlowPyrLK(f0, f1, p0=None):
    if p0 is None:
        p0 = goodFeaturesToTrack(f0)
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(f0, f1, p0, None,
                                               winSize=(15, 15),
                                               maxLevel=5,
                                               criteria=(cv2.TERM_CRITERIA_EPS
                                                         | cv2.TERM_CRITERIA_COUNT,
                                                         10, 0.03))
        p0, p1 = p0[st == 1], p1[st == 1]
        return p0, p1
    else:
        return None


def get_patch(f: np.ndarray, center: np.ndarray, size: int):
    center = center.reshape(2).astype(np.int64)[::-1]
    augf = np.pad(f, [(size, size), (size, size)], 'mean')
    lb = center - (size - 1) // 2 + size
    rb = lb + size
    patch = augf[lb[0]:rb[0], lb[1]:rb[1]]
    assert np.prod(patch.shape), \
        'augf.shape={} center={} patch.shape={}' \
            .format(augf.shape, center, patch.shape)
    return patch


def save_patch(todir, name, patch_pair):
    for i, pa in enumerate(patch_pair):
        kw = {}
        if len(pa.shape) == 2:
            kw['cmap'] = 'gray'
        plt.imsave(os.path.join(todir, '{}_{}'.format(name, i)), pa, **kw)
        plt.close()


class MLE_ShiTomasi_LucasKanadePrym(object):
    def __init__(self, gen_patches=False, patch_size=64, patchdir=None):
        #self.frames = defaultdict(partial(deque, maxlen=100))
        self.lkpair = defaultdict(partial(deque, maxlen=2))
        self.gen_patches = gen_patches
        self.patch_size = patch_size
        self.patch_count = 0
        if gen_patches:
            patchdir = os.path.normpath(patchdir)
            os.makedirs(patchdir, exist_ok=True)
        self.patchdir = patchdir

    def __call__(self, cl_frames):
        new_cl_frames = []
        p0 = None
        for cl, f in cl_frames:
            if f is None:
                new_cl_frames.append((cl, f))
            else:
                self.lkpair[cl].append(f)
                if len(self.lkpair[cl]) < self.lkpair[cl].maxlen:
                    new_cl_frames.append((cl, np.zeros_like(f)))
                else:
                    try:
                        p0, p1 = calcOpticalFlowPyrLK(
                            *self.lkpair[cl])
                    except TypeError:
                        p0 = None
                        new_cl_frames.append((cl, f))
                    else:
                        if self.gen_patches:
                            for old, new in zip(p0, p1):
                                pch0 = get_patch(f, old, self.patch_size)
                                pch1 = get_patch(f, new, self.patch_size)
                                save_patch(self.patchdir, self.patch_count,
                                           (pch0, pch1))
                                self.patch_count += 1
                        mask = np.zeros_like(self.lkpair[cl][0])
                        for old, new, clr in zip(p0, p1, colors):
                            mask = cv2.line(mask,
                                            tuple(new.ravel()),
                                            tuple(old.ravel()),
                                            clr.tolist(), 2)
                        new_cl_frames.append((cl, cv2.add(f, mask)))
        return new_cl_frames

def frame_processor():
    return MLE_ShiTomasi_LucasKanadePrym()
