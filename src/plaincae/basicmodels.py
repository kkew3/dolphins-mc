from collections import namedtuple

import torch
import torch.nn as nn

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
