import torch
import torch.nn as nn


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Yield3DCNN_Model(nn.Module):
    """A compact 3D CNN adapted from the patch-based yield-estimation baseline."""

    def __init__(self, in_shape, hidden_dims=(16, 32, 64, 128, 64, 32, 16, 8), out_dim=1, **kwargs):
        super().__init__()
        _, c, _, _ = in_shape

        channels = [c, *hidden_dims]
        self.blocks = nn.ModuleList([
            Conv3dBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])
        self.head = nn.Conv3d(hidden_dims[-1], out_dim, kernel_size=1)

    def forward(self, x, **kwargs):
        # [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        for block in self.blocks:
            x = block(x)
        # Collapse only the temporal axis and keep the spatial yield map.
        x = x.mean(dim=2, keepdim=True)
        x = self.head(x)
        return x.permute(0, 2, 1, 3, 4)
