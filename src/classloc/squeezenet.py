from typing import Tuple
import logging

import torch
import torch.nn as nn
import torchvision.models

import trainlib


class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int, resolution: Tuple[int, int]):
        """
        :param num_classes: number of classes
        :param resolution: (height, width) of the input image
        """
        super().__init__()
        _inputs = torch.rand(1, 3, *resolution)
        self.features = torchvision.models.squeezenet1_1().features
        self.features.eval()
        with torch.no_grad():
            _outputs = self.features(_inputs)
        poolsize = _outputs.shape[2:]
        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Conv2d(512, num_classes, 1, stride=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(poolsize, stride=1),
        )

    def forward(self, x):
        h = self.features(x)
        c = self.classifier(h)
        return c.view(*c.shape[:2])
