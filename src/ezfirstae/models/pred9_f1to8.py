import torch
import torch.nn as nn


pool_scale = 8
temporal_batch_size = 8


class STCAEEncoder(nn.Module):
    """
    The ST-CAE encoder that expects frames over 8 time steps and outputs a
    group of feature maps over one time step.

        - Inputs: a batch of B&W frame segments of shape (B, 1, 8, H, W)
        - Outputs: a batch of feature tensors of shape (B, 64, 1, H', W')

    The height (H) and the width (W) should be the power of 8 so that the
    reconstruction from latent space to input space can be successful.
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.size(2) == temporal_batch_size
        assert all(inputs.size(i) % pool_scale == 0 for i in (3, 4))
        codes = self.features(inputs)
        return codes


class STCAEDecoder(nn.Module):
    """
    The ST-CAE decoder that expects a group of feature maps over one time step
    and outputs a reconstructed frame over one time step.

        - Inputs: a batch of feature tensors of shape (B, 64, 1, H', W')
        - Outputs: a batch of reconstructed frames of shape (B, 1, 1, H, W)
    """

    def __init__(self):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, (1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, (1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(16, 1, (1, 2, 2), stride=(1, 2, 2)),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        assert code.size(2) == 1
        rec = self.upsample(code)
        return rec
