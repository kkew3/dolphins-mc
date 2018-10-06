import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as trans
from typing import Tuple

from edgedetect import EdgeDetector


class EzFirstCAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, attention: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.encoder(x)
        rec = self.decoder(code)
        attn = self.sigmoid(self.attention(code))
        return rec, attn


class CAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encoder(x)
        rec = self.decoder(code)
        return rec


def _reduce_loss(reduction: str, loss: torch.Tensor):
    sum = torch.sum(loss)
    if reduction == 'sum':
        return sum
    if reduction == 'elementwise_mean':
        return sum.div(loss.size(0))


class NonrigidPenalty(nn.Module):
    """
    Loss criterion that encourages large blocks of monochromatic regions.
    """
    def __init__(self, reduction='elementwise_mean'):
        """
        :param reduction: see ``help(torch.nn.MSELoss)``
        """
        super().__init__()
        self.edged = EdgeDetector()
        self.reduction = reduction

    def forward(self, attention):
        edgemap = self.edged(attention)
        edgemap = edgemap.view(edgemap.size(0), -1)
        return _reduce_loss(self.reduction, torch.norm(edgemap, 2, 1))


class DarknessPenalty(nn.Module):
    """
    Loss criterion that encourages more non-zero pixels, assuming the input is
    an one-channel attention map.
    """
    def __init__(self, normalize: trans.Normalize,
                 reduction='elementwise_mean'):
        """
        :param normalize: the ``trans.Normalize`` transformation used at
               __init__ of ``VideoDataset``
        :param reduction: see ``help(torch.nn.MSELoss)``
        """
        super().__init__()
        self.mean, self.std = normalize.mean[0], normalize.std[0]
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, attention):
        attention = attention.view(attention.size(0), -1)
        ulimit = (float(attention.shape[1]) - self.mean) / self.std
        return _reduce_loss(self.reduction, ulimit - torch.norm(attention, 1, 1))
