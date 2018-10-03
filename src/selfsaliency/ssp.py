import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union


def make_init_grid(radius: int, input_size: Tuple[int, int]) -> torch.Tensor:
    """
    Make a square bounding box at the center of the input frame.

    :param radius: the size of the vision (square) bounding box
    :param input_size: the height and width of the entire input frame
    """
    H, W = input_size
    rows = np.linspace(-radius/(2*H), radius/(2*H), radius)
    cols = np.linspace(-radius/(2*W), radius/(2*W), radius)
    grid = np.meshgrid(rows, cols)
    grid = np.concatenate([x[..., np.newaxis] for x in grid], axis=2)
    return torch.from_numpy(grid).float()

class STGumbelSoftmaxEstimator(nn.Module):
    """
    Reference:
    https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f

    Inputs:

    - tensor of shape (..., num_classes)

    Outputs:

    - an one-hot tensor of shape (..., num_classes)
    """
    def __init__(self, temperature: float=1.0, eps: float=1e-20):
        super().__init__()

    def _sample(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        return -torch.log(-torch.log(U + eps) + eps)

    def _softmax_sample(self, logits):
        y = logits + self._sample(logits)
        return torch.softmax(y / self.temperature, dim=-1)

    def forward(self, logits):
        y = self._softmax_sample(logits)
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, y.shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*y.shape)
        return (y_hard - y).detach() + y


class SaliencyPredictor(nn.Module):
    def __init__(self, descriptor_size: int, in_channels: int,
                 kernel: Union[int, Tuple[int, int]]):
        """
        :param descriptor_size: number of saliency descriptors
        :param in_channels: the number of channels of the low-level latent
               representation of the input frames
        :param kernel: the kernel of the convolution, may be either an odd
               integer or a tuple of two odd integers
        """
        if isinstance(kernel, int):
            padding = ((kernel + 1) // 2,) * 2
            kernel = (kernel,) * 2
        else:
            padding = tuple((kernel[i] + 1) // 2 for i in range(2))
        self.kernels = torch.zeros(descriptor_size, )

        # initialize the Conv2d
        ...




class SelfSaliencyPredictor(nn.Module):
    def __init__(self, features: nn.Module,
                 input_size: Tuple[int, int, int, int],
                 hidden_size: int, descriptor_size: int,
                 matcher_kernel_size: Tuple[int, int, int],
                 central_radius: int, periphr_radius: int,
                 periphr_pool: nn.Module, num_lstm_layers: int=1):
        """
        :param features: the low-level feature extractor
        :param input_size: the size tuple of the inputs, of form (batch_size,
               num_channels, height, width)
        :param hidden_size: the number of LSTM units per LSTM layer
        :param descriptor_size: the number of discrete descriptor
        :param matcher_kernel_size: the size tuple of the convolutional kernel
               at matching stage, of form (num_channels, height, width)
        :param central_radius: the size of the central vision square bounding
               box; this value should be smaller than ``periphr_radius``
        :param periphr_radius: the size of the peripheral vision square
               bounding box; this value should be larger than
               ``central_radius``
        :param periphr_pool: the pooling module of the peripheral vision before
               ``features`` is applied
        :param num_lstm_layers: the number of LSTM layers, default to 1
        """
        super().__init__()
        self.features = features.eval()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.central_radius = central_radius
        self.periphr_radius = periphr_radius
        self.periphr_pool = periphr_pool

        for params in self.features.parameters():
            params.requires_grad = False
        _x = torch.rand(1, *self.input_size.shape[1:])
        latent_size = self.features(_x).view(-1).size(0)
        self.lstm = nn.LSTM(input_size=latent_size, hidden_size=hidden_size,
                            num_layers=num_lstm_layers)
        self.affine_regr = nn.Linear(hidden_size, 2)
        self.affine_desc = nn.Linear(hidden_size, descriptor_size)
        self.matcher_kernel = torch.zeros(np.prod(matcher_kernel_size), descriptor_size,
                                          dtype=torch.float32, requires_grad=True),
        self.matcher_bias = torch.zeros(input_size.shape[0],
                                        dtype=torch.float32, requires_grad=True)

        # initialize matcher_kernel
        nn.init.normal_(self.matcher_kernel.data)
        nn.init.zero_(self.matcher_bias.data)
