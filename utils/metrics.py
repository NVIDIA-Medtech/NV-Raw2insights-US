# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for ultrasound IQ signal quality assessment."""

import torch


def snr(x: torch.Tensor, y: torch.Tensor, batch_dim=None) -> torch.Tensor:
    """Signal-to-noise ratio (dB)."""
    if x.shape != y.shape:
        raise ValueError("snr expects inputs with matching shapes")

    diff = x - y
    if batch_dim is None:
        reduce_dims = tuple(range(x.ndim))
    else:
        reduce_dims = tuple(i for i in range(x.ndim) if i != batch_dim)

    signal_power = x.abs().square().mean(dim=reduce_dims)
    noise_power = diff.abs().square().mean(dim=reduce_dims)

    eps = torch.finfo(signal_power.dtype).tiny
    return 10 * torch.log10(signal_power / noise_power.clamp_min(eps))


def complex_correlation(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Complex correlation coefficient between x and y along dim.

    Inputs can be [B, 2, T] (real/imag channels) or complex tensors.
    Returns real part of correlation (1.0 = perfect match).
    """
    def _to_complex(t):
        if t.is_complex():
            return t
        if t.ndim >= 2 and t.shape[1] == 2:
            return torch.complex(t[:, 0], t[:, 1])
        return t.type(torch.complex64) if t.dtype == torch.float32 else t

    x_c, y_c = _to_complex(x), _to_complex(y)
    xy = (x_c * y_c.conj()).sum(dim=dim)
    xx = x_c.abs().square().sum(dim=dim)
    yy = y_c.abs().square().sum(dim=dim)
    return (xy / (torch.sqrt(xx * yy) + eps)).real
