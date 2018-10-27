from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as trans
from typing import Tuple

from edgedetect import EdgeDetector


SpatioTemporalInputShape = namedtuple('SpatioTemporalInputShape', tuple('BCTHW'))
SpatioInputShape = namedtuple('SpatioInputShape', tuple('BCHW'))


class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encoder(x)
        rec = self.decoder(code)
        return rec
