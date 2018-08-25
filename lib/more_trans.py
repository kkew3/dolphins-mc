from functools import partial

import numpy as np
import torchvision.transforms as trans


class DeNormalize(object):
    """
    The inverse transformation of ``tochvision.transforms.Normalize``.

    :param mean: the mean used in ``Normalize``
    :param std: the std used in ``Normalize``
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor * self.std + self.mean


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

class ResetChannelNumpy(object):
    """
    Same as ``ResetChannel`` but for numpy array.
    """
    def __init__(self, channel):
        self.channel = channel

    def __call__(self, tensor):
        tensor = np.copy(tensor)
        tensor[self.channel] = np.zeros(tensor.shape[1:])
        return tensor


class ToNumpy(object):
    """
    Converts from ``PIL.Image`` to numpy array, which involves normalizing all
    pixel values from range [0, 256) to [0.0, 1.0).
    """
    def __init__(self, dtype=np.float64):
        """
        :param dtype: the numpy data type, default to ``np.float64``
        """
        self.dtype = dtype

    def __call__(self, img):
        tensor = np.asarray(img, dtype=np.float64)
        tensor = tensor / 255.0
        return np.array(tensor, dtype=self.dtype)


class HWC2CHW(object):
    """
    Transposes numpy tensor from HWC to CHW.
    """
    def __call__(self, tensor):
        return np.transpose(tensor, (2, 0, 1))


class CHW2HWC(object):
    """
    Transoses numpy tensor from CHW to HWC.
    """
    def __call__(self, tensor):
        return np.transpose(tensor, (1, 2, 0))
