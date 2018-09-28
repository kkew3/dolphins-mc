import numpy as np
import torch
from scipy.ndimage import filters
from typing import Sequence, Callable


class DeNormalize(object):
    """
    The inverse transformation of ``tochvision.transforms.Normalize``. As in
    ``tochvision.transforms.Normalize``, this operation modifies input tensor
    in place. The input tensor should be of shape (C, H, W), namely,
    (num_channels, height, width).

    :param mean: the mean used in ``Normalize``
    :param std: the std used in ``Normalize``
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float).view(-1, 1, 1)

    def __call__(self, tensor):
        tensor.mul_(self.std)
        tensor.add_(self.mean)
        return tensor

class ResetChannel(object):
    """
    Resets a specific input channel to 0.0.
    """
    def __init__(self, channel):
        """
        :param channel: the channel to reset
        :type channel: int
        """
        self.channel = channel

    def __call__(self, tensor):
        tensor = tensor.clone()
        tensor[self.channel].zero_()
        return tensor

class GaussianBlur(object):
    """
    Applicable only to Numpy arrays; thus it should be inserted before
    ``trans.ToTensor()``.
    """
    def __init__(self, std=1.5, tr_std=4.0):
        """
        :param std: the standard deviations
        :param tr_std: the upper limit of standard deviation beyond which the
               deviation is truncated
        """
        self.std = max(0.0, std)
        self.tr_std = max(self.std, tr_std)

    def __call__(self, img):
        """
        :param img: numpy array of dimension HWC
        :return: blurred image
        """
        return filters.gaussian_filter(img, (self.std, self.std, 0.0),
                                       truncate=self.tr_std)

class MedianBlur(object):
    """
    Applicable only to Numpy arrays; thus it should be inserted before
    ``trans.ToTensor()``.
    """
    def __init__(self, width=5):
        """
        :param width: the width of the sliding window.
        :type width: int
        """
        self.width = max(1, width)

    def __call__(self, img):
        """
        :param img: numpy array of dimension HWC
        :return: blurred image
        """
        return filters.median_filter(img, (self.width, self.width, 1))


def hwc2chw(tensor):
    """
    Transpose a numpy array from HWC to CHW.
    """
    return np.transpose(tensor, (2, 0, 1))

def chw2hwc(tensor):
    """
    Transpose a numpy array from CHW to HWC.
    """
    return np.transpose(tensor, (1, 2, 0))

def numpy_loader(dataloader):
    """
    Generator that converts pytorch tensor to numpy array on the fly.

    :param dataloader: a dataloader instance
    :type dataloader: torch.utils.data.DataLoader
    """
    for item in dataloader:
        if isinstance(item, tuple):
            yield tuple(x.numpy() for x in item)
        else:
            yield item.numpy()
