from collections import namedtuple
import random
from functools import partial
import math

import numpy as np
from typing import Tuple, List, Optional, Sequence

# the rectangular bound box
# element types: (int, int, int, int)
# x,y: the coordinate of the upper left corner
# width,height: the width/height of the rectangle
Rect = namedtuple('Rect', ('x', 'y', 'width', 'height'))
# the Harr features and weights
# element types: List[List[Rect]], List[List[float]]
HarrFeatures = namedtuple('HarrFeatures', ('features', 'weights'))


def rects2np(rects: Sequence[Rect]) -> np.ndarray:
    """
    Converts a sequence of ``Rect`` objects to a 2D numpy array with four rows,
    namely the X's, Y's, widths, and heights of the rectangles. The ``dtype`` of
    the returned arrays is ``np.int32``. The ``shape`` of the returned arrays is
    ``(4, len(rects))``.

    :param rects: the rectangles
    :return: the XYWH representation of the rectangles

    >>> r = [(1,2,3,4),(5,6,7,8),(9,10,11,12)]
    >>> a = np.array([[1,5,9],[2,6,10],[3,7,11],[4,8,12]])
    >>> (a == rects2np(r)).all()
    True
    """
    # noinspection PyTypeChecker
    return np.stack(tuple(map(partial(np.array, dtype=np.int32), zip(*rects))))


def xywh2ullr(rects_XYWH: np.ndarray):
    """
    Converts the XYWH representation of rectangles to ULLR representation, a 2D
    array of shape the same as ``rects_XYWH`` with the following rows::

        0. the X's of the upper left corner of the rectangles
        1. the Y's of the upper left corner of the rectangles
        2. the X's of the lower right corner of the rectangles
        3. the Y's of the lower right corner of the rectangles

    :param rects_XYWH: the XYWH of rectangles
    :return: the ULLR representation of the rectangles

    >>> a = np.array([[1,5,9],[2,6,10],[3,7,11],[4,8,12]])
    >>> b = np.array([[1,5,9],[2,6,10],[4,12,20],[6,14,22]])
    >>> c = xywh2ullr(a)
    >>> (b == c).all()
    True
    >>> c[1,2] = 0
    >>> a[1,2]
    10
    """
    r12 = rects_XYWH[[0, 1]]  # implicitly made a copy
    r3 = np.sum(rects_XYWH[[0, 2]], axis=0, keepdims=True)
    r4 = np.sum(rects_XYWH[[1, 3]], axis=0, keepdims=True)
    return np.concatenate((r12, r3, r4))


def repeat(n: int, a: np.ndarray, ascol=True) -> np.ndarray:
    """
    A shortcut function for ``np.repeat``. If ``a`` is a 1D array, repeat a as
    rows/columns of a new matrix. If ``a`` is a matrix, repeat each row of
    ``a`` in the same manner, such that the ith row of the result tensor is the
    ``repeat`` result of ``a[i]``.

    :param n: number of times to repeat
    :param a: the 1D/2D array to repeat
    :param ascol: True to repeat ``a`` or the rows of ``a`` as columns of the
           result matrix; False to repeat as rows
    :return: a matrix of shape (*a.shape, n) such that the i-th column
             equals to vector ``a``, if ``ascol`` is True; otherwise, the
             transpose of the last two axes of such matrix

    >>> a = np.array([1,2,3,4])
    >>> (np.array([[1]*3,[2]*3,[3]*3,[4]*3]) == repeat(3, a)).all()
    True
    >>> (np.array([[1,2,3,4]]*3) == repeat(3, a, ascol=False)).all()
    True
    >>> b = np.array([[1,2,3,4],[5,6,7,8]])
    >>> (np.array([[[1]*3,[2]*3,[3]*3,[4]*3],
    ...            [[5]*3,[6]*3,[7]*3,[8]*3]]) == repeat(3, b)).all()
    True
    >>> (np.array([[[1,2,3,4]]*3,
    ...            [[5,6,7,8]]*3]) == repeat(3, b, ascol=False)).all()
    True
    """
    m = np.repeat(a, n).reshape(a.shape + (n,))
    return m if ascol else m.transpose((1, 0) if len(a.shape) == 1 else (0, 2, 1))


def compute_feature(img_integral: np.ndarray, harr: HarrFeatures,
                    sample_boxes: List[Rect]) -> np.ndarray:
    """
    :param img_integral: the integrated image, of shape (height, width)
    :param harr: the Harr features and weights
    :param sample_boxes: the sample boxes
    :return: computed sample features
    """
    img_integral = img_integral.astype(np.float32)
    box_XYs = rects2np(sample_boxes)[[0, 1]]
    harrflens = list(map(len, harr.features))
    harrfmaxlen = max(harrflens)
    rbox_XYs = repeat(harrfmaxlen, box_XYs)
    rhfeatures = map(partial(repeat, len(sample_boxes), ascol=False),
                     map(xywh2ullr, map(rects2np, harr.features)))
    harrweights = map(np.array, harr.weights)
    sample_features = []
    for n, rhf, hw in zip(harrflens, rhfeatures, harrweights):
        xmins = rbox_XYs[0][:, :n] + rhf[0]
        xmaxs = rbox_XYs[0][:, :n] + rhf[2]
        ymins = rbox_XYs[1][:, :n] + rhf[1]
        ymaxs = rbox_XYs[1][:, :n] + rhf[3]
        sample_features.append(np.dot(img_integral[ymins, xmins]
                                      + img_integral[ymaxs, xmaxs]
                                      - img_integral[ymins, xmaxs]
                                      - img_integral[ymaxs, xmins], hw)
                               .astype(np.float32))
    sample_features = np.stack(sample_features)
    return sample_features


class CompressiveTracker(object):
    def __init__(self,
                 num_feature_rect_range: Tuple[int, int] = (2, 4),
                 num_features: int = 50,
                 radical_scope_positive: int = 4,
                 search_window_size: int = 25,
                 learning_rate: float = 0.85) -> None:
        """
        :param num_feature_rect_range: (min, max+1)
        :param num_features:
        :param radical_scope_positive:
        :param search_window_size:
        :param learning_rate:
        """
        self.num_feature_rect_range = num_feature_rect_range
        self.num_features = num_features
        self.radical_scope_positive = radical_scope_positive
        self.search_window_size = search_window_size
        self.lr = learning_rate

        self._pos_mean = np.zeros(self.num_features)
        self._neg_mean = np.zeros(self.num_features)
        self._pos_std = np.ones(self.num_features)
        self._neg_std = np.ones(self.num_features)

    def compute_haar(self, object_box: Rect, num_features: Optional[int] = None) -> HarrFeatures:
        """
        :param object_box: the object rectangle, with width no less than 3 and
               height no less than 3
        :param num_features: total number of features, default to
               ``self.num_features``
        :return: the features and features weight
        """
        features = []
        features_weight = []
        if num_features is None:
            num_features = self.num_features
        nums_rect = np.random.randint(*self.num_feature_rect_range,
                                      size=num_features)
        for n in range(nums_rect):
            rects = []
            weights = []
            for _ in range(n):
                x = random.randint(0, object_box.width - 3 - 1)
                y = random.randint(0, object_box.height - 3 - 1)
                w = random.randint(1, object_box.width - x - 2)
                h = random.randint(1, object_box.height - y - 2)
                rects.append(Rect(x, y, w, h))
                weight = 1.0 / math.sqrt(float(n))
                if random.random() < 0.5:
                    weight = -weight
                weights.append(weight)
            features.append(rects)
            features_weight.append(weights)
        return HarrFeatures(features, features_weight)

    def sample_rect(self, img: np.ndarray, object_box: Rect, inner_radius: float,
                    outer_radius: Optional[float] = None,
                    max_samples: Optional[int] = None) -> List[Rect]:
        """
        :param img: processing frame, of shape (height, width)
        :param object_box: recent object position
        :param inner_radius: inner sampling radius
        :param outer_radius: outer sampling radius
        :param max_samples: maximal number of sampled images
        :return: the rectangular coordinates of the sampled images
        """
        rowsz = img.shape[0] - object_box.height - 1
        colsz = img.shape[1] - object_box.width - 1
        inradsq = inner_radius * inner_radius
        if outer_radius is not None:
            outradsq = outer_radius * outer_radius
        minrow = max(0, int(object_box.y) - int(inner_radius))
        maxrow = min(rowsz - 1, object_box.y + int(inner_radius))
        mincol = max(0, object_box.x - int(inner_radius))
        maxcol = min(colsz - 1, object_box.x + int(inner_radius))
        prob = float(max_samples) / (maxrow - minrow + 1) / (maxcol - mincol + 1)

        mesh = np.meshgrid(np.arange(minrow, maxrow + 1),
                           np.arange(mincol, maxcol + 1))
        mesh = np.stack(mesh, axis=2).reshape((-1, 2))
        ulbox = np.array([object_box.y, object_box.x])
        meshdiff = mesh - ulbox
        distsqs = np.dot(meshdiff, meshdiff)
        if outer_radius is not None:
            mask = (np.random.rand(distsqs.shape[0]) < prob) & \
                   (distsqs < inradsq) & (distsqs >= outer_radius)
        else:
            mask = (distsqs < inner_radius)
        mesh = mesh[mask]

        sample_boxes = []
        for rc in mesh:
            y, x = rc
            sample_boxes.append(Rect(x, y, object_box.width, object_box.height))
        return sample_boxes
