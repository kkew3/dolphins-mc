import torch
import torch.nn as nn


pool_scale = 8
temporal_batch_size = 8


class STCAEEncoder(nn.Module):

    @staticmethod
    def c3d(ic, oc=None):
        return nn.Conv3d(ic, oc if oc else ic, 3, padding=1)

    def __init__(self):
        super().__init__()
        c3d = self.c3d
        bn = nn.BatchNorm3d
        relu = nn.ReLU(inplace=True)
        m3d = nn.MaxPool3d(2, stride=2)

        self.features = nn.Sequential(
            c3d(1, 64), bn(64), relu,
            c3d(64), bn(64), relu,
            m3d,
            c3d(64, 128), bn(128), relu,
            c3d(128), bn(128), relu,
            m3d,
            c3d(128, 256), bn(256), relu,
            c3d(256), bn(256), relu,
            m3d,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, inputs):
        return self.features(inputs)


class STCAEDecoder(nn.Module):
    @staticmethod
    def c3d(ic, oc=None):
        return nn.ConvTranspose3d(ic, oc if oc else ic, (1, 3, 3),
                                  padding=(0, 1, 1))

    @staticmethod
    def cp3d(ic, oc=None):
        return nn.ConvTranspose3d(ic, oc if oc else ic, (1, 2, 2),
                                  stride=(1, 2, 2))

    def __init__(self):
        super().__init__()
        c3d = self.c3d
        cp3d = self.cp3d
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm3d

        self.upsample = nn.Sequential(
            c3d(256), bn(256), relu,
            c3d(256, 128), bn(128), relu,
            c3d(128), bn(128), relu,
            cp3d(128, 64), bn(64), relu,
            c3d(64), bn(64), relu,
            cp3d(64, 32), bn(32), relu,
            c3d(32), bn(32), relu,
            cp3d(32, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, codes):
        assert codes.size(2) == 1
        return self.upsample(codes)
