import numpy as np
import torch
import torchvision.transforms as trans
import utils
from PIL import Image
from scipy.ndimage import filters


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


class RGB2Gray(object):
    """
    Applicable only to PyTorch tensor.
    """

    def __init__(self, shape="chw"):
        """
        :param shape: either "chw" or "hwc"
        """
        self.cdim = shape.index('c')

    def __call__(self, tensor):
        gray_tensor, _ = tensor.max(self.cdim, keepdim=True)
        return gray_tensor


def hwc2chw(tensor):
    """
    Transpose a numpy array from HWC to CHW, or from BHWC to BCHW.
    """
    ndim2transpose = {
        3: (2, 0, 1),
        4: (0, 3, 1, 2),
    }
    try:
        return np.transpose(tensor, ndim2transpose[len(tensor.shape)])
    except KeyError:
        raise ValueError('Expecting tensor of three or four axes, but got {}'
                         .format(len(tensor.shape)))


def chw2hwc(tensor):
    """
    Transpose a numpy array from CHW to HWC, or from BCHW to BHWC.
    """
    ndim2transpose = {
        3: (1, 2, 0),
        4: (0, 2, 3, 1),
    }
    try:
        return np.transpose(tensor, ndim2transpose[len(tensor.shape)])
    except KeyError:
        raise ValueError('Expecting tensor of three or four axes, but got {}'
                         .format(len(tensor.shape)))


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


def rearrange_temporal_batch(data_batch: torch.Tensor, T: int) -> torch.Tensor:
    """
    Rearrange a hyper-batch of frames of shape (B*T, C, H, W) into
    (B, C, T, H, W) where:

        - B: the batch size
        - C: the number of channels
        - T: the temporal batch size
        - H: the height
        - W: the width

    :param data_batch: batch tensor to convert
    :param T: the temporal batch size
    :return: converted batch
    """
    assert len(data_batch.size()) == 4
    assert data_batch.size(0) % T == 0
    B = data_batch.size(0) // T
    data_batch = data_batch.view(B, T, *data_batch.shape[1:])
    data_batch = data_batch.transpose(1, 2).contiguous()
    return data_batch.detach()  # so that ``is_leaf`` is True


def clamp_tensor_to_image(tensor: torch.Tensor):
    return torch.clamp(tensor, 0.0, 1.0)


class BWCAEPreprocess(object):
    """
    The pre-processing transform before an autoregressive/autoencoding model
    of encode-edecoder architecture that accepts grayscale images as inputs.

    Steps:

        1. convert RGB to B&W by taking the maximum along the channel axis
        2. downsample by specified scale
        3. random crop a little bit such that the height and the width are
           both powers of the subsequent pooling scale
        4. normalize the image matrix
        5. optionally convert B&W back to RGB by repeating the image matrix
           three times along the channel axis
    """
    def __init__(self, normalize: trans.Normalize, pool_scale: int = 1,
                 downsample_scale: int = 1, to_rgb: bool = False):
        """
        :param normalize: the normalization transform
        :param pool_scale: the overall scale of the pooling operations in
               subsequent encoder; the image will be cropped to (H', W') where
               H' and W' are the nearest positive integers to H and W that are
               the power of ``pool_scale``, so that ``unpool(pool(x))`` is of
               the same shape as ``x``
        :param downsample_scale: the scale to downsample the video frames
        :param to_rgb: if True, at the last step convert from B&W image to
               RGB image
        """
        self.totensor = trans.ToTensor()
        self.normalize = normalize
        self.pool_scale = pool_scale
        self.downsample_scale = downsample_scale
        self.to_rgb = to_rgb

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """
        :param img: image of shape (H, W, C)
        :return: tensor of shape (1, H, W) if ``to_rgb`` is ``False``, else
                 of shape (3, H, W)
        """
        h, w = img.shape[:2]
        sh_after_ds = h // self.downsample_scale, w // self.downsample_scale
        gray = Image.fromarray(img.max(axis=2))
        try:
            _ = self.downsample
        except AttributeError:
            self.downsample = trans.Resize(sh_after_ds)
        gray = self.downsample(gray)
        try:
            _ = self.crop
        except AttributeError:
            gh, gw = gray.height, gray.width
            gh = utils.inf_powerof(gh, self.pool_scale)
            gw = utils.inf_powerof(gw, self.pool_scale)
            self.crop = trans.RandomCrop((gh, gw))
        gray = self.crop(gray)
        tensor = self.totensor(gray)
        tensor = self.normalize(tensor)
        assert tensor.size(0) == 1 and len(tensor.size()) == 3
        if self.to_rgb:
            tensor = tensor.repeat(3, 1, 1)
        return tensor
