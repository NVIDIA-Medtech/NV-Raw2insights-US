# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn

__all__ = ["IQAutoencoder"]


class IQAutoencoder(nn.Module):
    """Deep 1-D convolutional auto-encoder for ultrasound IQ channel data."""

    def __init__(self, in_channels: int, *, n_features: int = 64, target_length: int = 1024) -> None:
        super().__init__()
        kw = dict(kernel_size=5, padding=2, stride=2, bias=False)

        self.n_features = n_features
        T = target_length
        # Deep encoder: 6 layers of downsampling (factor 64)

        layers = []
        c_in = in_channels
        c_out = n_features

        # 6 layers: T -> T/2 -> T/4 -> T/8 -> T/16 -> T/32 -> T/64
        for i in range(6):
            layers.extend([
                nn.Conv1d(c_in, c_out, **kw),
                nn.LayerNorm((c_out, T // (2**(i+1)))),
                nn.LeakyReLU(negative_slope=0.01)
            ])
            c_in = c_out
            # Optional: increase features with depth? Keeping constant for simplicity as per "Simplicity wins"

        self.encoder = nn.Sequential(*layers)

        # Decoder mirrors encoder
        dec_layers = []
        c_in = n_features
        c_out = n_features

        # 6 layers upsampling
        # We need output_padding=1 to match the stride=2 if input length is power of 2
        for i in range(5): # First 5 layers to n_features
            dec_layers.extend([
                nn.ConvTranspose1d(c_in, c_out, **kw, output_padding=1),
                nn.LayerNorm((c_out, T // (2**(5-i)))),
                nn.LeakyReLU(negative_slope=0.01)
            ])

        # Last layer to in_channels
        dec_layers.append(nn.ConvTranspose1d(c_in, in_channels, **kw, output_padding=1))

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)
