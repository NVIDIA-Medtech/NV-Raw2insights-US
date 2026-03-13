from __future__ import annotations

import torch
from torch import nn

__all__ = ["IQAutoencoder", "TransformerBlock", "sinusoid_pe"]


class TransformerBlock(nn.Module):
    """Lightweight transformer block for 1-D latent sequences."""

    def __init__(self, c: int, nhead: int = 8, dim_ff: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        dim_ff = dim_ff or 4 * c
        self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(c, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, c),
        )
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> attn expects [B, T, C]
        x_time = x.transpose(1, 2)
        attn_out, _ = self.attn(x_time, x_time, x_time)
        x_time = self.norm1(x_time + attn_out)
        ff_out = self.ff(x_time)
        x_time = self.norm2(x_time + ff_out)
        return x_time.transpose(1, 2)


def sinusoid_pe(T: int, C: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(T, device=device).float().unsqueeze(1)  # [T,1]
    i = torch.arange(C // 2, device=device).float().unsqueeze(0)  # [1,C/2]
    w = 1.0 / (10000 ** (2 * i / C))
    pe = torch.zeros(T, C, device=device)
    pe[:, 0::2] = torch.sin(pos * w)
    pe[:, 1::2] = torch.cos(pos * w)
    return pe.transpose(0, 1).unsqueeze(0)  # [1,C,T]


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
