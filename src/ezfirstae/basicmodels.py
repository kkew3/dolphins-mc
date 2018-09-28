import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from edgedetect import EdgeDetector


class STCAEEncoder(nn.Module):
    """
    The ST-CAE encoder.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(32, 48, 3, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),

            nn.Conv3d(48, 48, 3, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2, stride=2),

            nn.Conv3d(48, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        code = self.features(x)
        return code


class STCAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, 2, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(16, 1, 2, stride=2),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, code):
        rec = self.upsample(code)
        return rec


class EzFirstCAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, attention: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.encoder(x)
        rec = self.decoder(code)
        attn = self.attention(code)
        return rec, attn


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
        super().__init__()
        self.edged = EdgeDetector()
        self.reduction = reduction

    def forward(self, attention):
        edgemap = self.edged(attention)
        edgemap = edgemap.view(edgemap.size(0), -1)
        return _reduce_loss(self.reduction, self.norm(edgemap, 2, 1))


class DarknessPenalty(nn.Module):
    """
    Loss criterion that encourages more non-zero pixels.
    """
    def __init__(self, reduction='elementwise_mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, attention):
        attention = attention.view(attention.size(0), -1)
        ulimit = float(attention.shape[1])
        return _reduce_loss(self.reduction, ulimit - self.norm(attention, 1, 1))
