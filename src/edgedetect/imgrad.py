import pdb
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


def _make3x3kernel(kernel: np.ndarray) -> nn.Module:
    conv = nn.Conv2d(1, 1, 3, stride=1, padding=1)
    conv.weight.requires_grad = False
    conv.weight.data.copy_(torch.from_numpy(kernel).float())
    conv.bias.requires_grad = False
    conv.bias.data.zero_()
    return conv

def sobelkernels() -> Tuple[nn.Module, nn.Module]:
    """
    Returns two torch.nn.Conv2d modules representing respectively the
    horizontal Sobel kernel and the vertical Sobel kernel. The kernel size is
    3 x 3. The stride is 1. Both modules have been frozen. The modules apply
    only to B&W images of shape (B, 1, H, W).
    """
    kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)
    hconv = _make3x3kernel(kernel)    # horizontal
    vconv = _make3x3kernel(kernel.T)  # vertical
    return hconv, vconv


class EdgeDetector(nn.Module):
    r"""
    Returns edge map from B&W frames. The edge maps are not normalized, with
    range :math:`[0, \infty)`. The input pixel values should be in range of
    :math:`[0, 1]`.
    """
    def __init__(self):
        super().__init__()
        self.hconv, self.vconv = sobelkernels()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == 1
        hemap = self.hconv(x)
        vemap = self.vconv(x)
        emap = torch.sqrt(hemap.pow(2) + vemap.pow(2))
        return emap
