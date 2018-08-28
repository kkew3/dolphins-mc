from collections import deque

import torch
import torch.nn as nn


vgg_layers_configs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


class StoreMaxPoolIndices(nn.Module):
    """
    Store the ``nn.MaxPool2d`` indices to an FIFO cache.
    """
    def __init__(self, cache):
        super(StoreMaxPoolIndices, self).__init__()
        assert hasattr(cache, 'append')
        self.cache = cache

    def forward(self, poolresult):
        output, indices = poolresult
        self.cache.append(indices)
        return output

class RetrieveMaxPoolIndices(nn.Module):
    """
    Retrieve the previously stored ``nn.MaxPool2d`` indices from the FIFO
    cache.
    """
    def __init__(self, cache):
        super(RetrieveMaxPoolIndices, self).__init__()
        assert hasattr(cache, 'pop')
        self.cache = cache

    def forward(self, x):
        indices = self.cache.pop()
        return (x, indices)

class MaxUnpool2d(nn.Module):
    """
    Same as ``nn.MaxUnpool2d`` but takes as the first positional argument a
    2-tuple of inputs and indices rather than as the first two positional
    arguments the inputs and indices respectively.
    """
    def __init__(self, *args, **kwargs):
        super(MaxUnpool2d, self).__init__()
        self.upsample = nn.MaxUnpool2d(*args, **kwargs)

    def forward(self, input_indices, *args, **kwargs):
        x, indices = input_indices
        return self.upsample(x, indices, *args, **kwargs)

class SaliencyCAE(nn.Module):
    """
    VGG(bn)-based CAE with saliency output.
    """

    def __init__(self, image_channels=3, vgg_arch='D', batch_norm=True):
        """
        :param image_channels: number of channels of input images
        :type image_channels: int
        :param vgg_arch: the VGG architecture configuration key, one of {'A',
               'B', 'D', 'E'}
        :type vgg_arch: str
        :param batch_norm: True to enable batch normalization during both
               encoding and decoding; False to disable it during either phase
        :type batch_norm: bool
        """
        super(SaliencyCAE, self).__init__()
        assert vgg_arch in vgg_layers_configs
        self.layers_config = [image_channels] + vgg_layers_configs[vgg_arch]
        self.batch_norm = batch_norm
        self.fifocache = deque()

        self.sigmoid = nn.Sigmoid()
        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()

    def forward(self, x):
        code = self.encoder(x)
        rx = self.decoder(code)
        s = self.sigmoid(rx)
        return s

    def make_encoder(self):
        features = []
        in_channel = self.layers_config[0]
        for x in self.layers_config[1:]:
            if x == 'M':
                features.append(nn.MaxPool2d(2, stride=2, return_indices=True))
                features.append(StoreMaxPoolIndices(self.fifocache))
            else:
                features.append(nn.Conv2d(in_channel, x, 3, padding=1))
                if self.batch_norm:
                    features.append(nn.BatchNorm2d(x))
                features.append(nn.ReLU())
                in_channel = x
        return nn.Sequential(*features)

    def make_decoder(self):
        defeatures = []
        in_channel = self.layers_config[0]
        for x in self.layers_config[1:]:
            if x == 'M':
                defeatures.append(MaxUnpool2d(2, stride=2))
                defeatures.append(RetrieveMaxPoolIndices(self.fifocache))
            else:
                defeatures.append(nn.Conv2d(x, in_channel, 3, padding=1))
                if self.batch_norm:
                    defeatures.append(nn.BatchNorm2d(x))
                defeatures.append(nn.ReLU())
                in_channel = x
        defeatures = reversed(defeatures)
        return nn.Sequential(*defeatures)
