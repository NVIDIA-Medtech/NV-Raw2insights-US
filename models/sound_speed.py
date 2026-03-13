from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

__all__ = ["SpatialProjector", "UNetSmall", "SoundSpeedDecoder"]


class SpatialProjector(nn.Module):
    """Pool (depth, lateral) latent to a square feature map via adaptive pooling + conv.

    Input:  [B, C, T_lat, n_rx]  — (depth, lateral) ≈ image space
    Output: [B, C, out_size, out_size]
    """

    def __init__(self, in_channels: int = 64, out_size: int = 32) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(out_size)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UNetSmall(nn.Module):
    """Compact U-Net used for sound-speed decoding."""

    def __init__(self, in_channels: int = 16, out_channels: int = 1, features: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for feat in features:
            self.downs.append(self._conv_block(in_channels, feat))
            in_channels = feat

        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.ups.append(self._conv_block(feat * 2, feat))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


class SoundSpeedDecoder(nn.Module):
    """Decode latent to sound speed map via tx compounding + spatial pooling + U-Net.

    Input:  [B, n_tx, n_rx, C, T_lat] (5D, n_tx>0) or [B, C, T_lat, n_rx] (4D)
    Output: [B, 1, out_size, out_size]
    """

    def __init__(self, in_channels: int = 64, out_size: int = 32, n_tx: int = 0, unet_features: Sequence[int] = (32, 64, 128), final_activation: str = "softplus") -> None:
        super().__init__()
        self.tx_proj = nn.Conv1d(n_tx, 1, 1) if n_tx else None  # learned tx combination
        self.projector = SpatialProjector(in_channels, out_size)
        self.unet = UNetSmall(in_channels=in_channels, out_channels=1, features=unet_features)
        self.final_act = nn.Softplus() if final_activation == "softplus" else nn.Sigmoid()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self.tx_proj is not None and latent.ndim == 5:
            B, nt, nr, C, T = latent.shape
            latent = self.tx_proj(latent.reshape(B, nt, -1)).squeeze(1)  # [B, nr*C*T]
            latent = latent.view(B, nr, C, T).permute(0, 2, 3, 1)       # [B, C, T, nr]
        return self.final_act(self.unet(self.projector(latent)))
