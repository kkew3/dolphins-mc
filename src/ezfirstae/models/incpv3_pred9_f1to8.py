from collections import namedtuple

import torch
import torch.nn as nn
import torchvision.models


pool_scale = 8
temporal_batch_size = 8


_STInputShape = namedtuple('SpatioTemporalInputShape', tuple('BCTHW'))
_SInputShape = namedtuple('SpatioInputShape', tuple('BCHW'))


class PreFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        inception_net = torchvision.models.inception_v3(pretrained=True)
        self.prefeatures = nn.Sequential(
            inception_net.Conv2d_1a_3x3,
            inception_net.Conv2d_2a_3x3,
            inception_net.Conv2d_2b_3x3,
        )

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prefeatures(x)


class STCAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefeatures = PreFeatures()
        maxpool = nn.MaxPool3d(2, stride=2)
        relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            relu,

            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            relu,

            maxpool,

            nn.Conv3d(64, 96, 3, padding=1),
            nn.BatchNorm3d(96),
            relu,

            nn.Conv3d(96, 96, 3, padding=1),
            nn.BatchNorm3d(96),
            relu,

            maxpool,

            nn.Conv3d(96, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            relu,

            maxpool,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.size(1) == 3  # RGB = 3
        assert inputs.size(2) == temporal_batch_size
        assert all(inputs.size(i) % pool_scale == 0 for i in (3, 4))
        # perform pre-feature extraction
        sh = _STInputShape(*inputs.size())
        inputs = inputs.transpose(1, 2).reshape(sh.B*sh.T, sh.C, sh.H, sh.W)
        proc_inputs = self.prefeatures(inputs)
        proc_sh = _SInputShape(*proc_inputs.size())
        proc_inputs = proc_inputs.reshape(sh.B, sh.T, proc_sh.C, proc_sh.H, proc_sh.W)
        proc_inputs = proc_inputs.transpose(1, 2).contiguous()
        # standard ST forward pass
        codes = self.features(proc_inputs)
        return codes


class STCAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        relu = nn.ReLU(inplace=True)
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(128, 128, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            relu,

            nn.ConvTranspose3d(128, 128, (1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(128),
            relu,

            nn.ConvTranspose3d(128, 96, (1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(96),
            relu,

            nn.ConvTranspose3d(96, 96, (1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(96),
            relu,

            nn.ConvTranspose3d(96, 64, (1, 2, 2), stride=(1, 2, 2)),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        assert codes.size(2) == 1
        recs = self.upsample(codes)
        return recs


class STCAEDecoderAttn(nn.Module):
    def __init__(self):
        super().__init__()
        relu = nn.ReLU(inplace=True)
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(128, 128, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            relu,

            nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(64),
            relu,

            nn.ConvTranspose3d(64, 32, (1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            relu,

            nn.ConvTranspose3d(32, 16, (1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(16),
            relu,

            nn.ConvTranspose3d(16, 1, (1, 2, 2), stride=(1, 2, 2)),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        assert codes.size(2) == 1
        attns = self.upsample(codes)
        return attns
