# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml


@dataclass
class RunPaths:
    run_dir: Path
    checkpoints: Path
    figures: Path
    logs: Path


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_run(
    phase: str,
    exp_name: Optional[str] = None,
    *,
    base_dir: Path | str = "experiments/runs",
) -> RunPaths:
    """Create a unified run directory layout.

    Layout: {base_dir}/{phase}/{exp_name}/{checkpoints,figures,logs}
    """
    if exp_name is None or exp_name == "":
        exp_name = os.environ.get("EXP_NAME", "default")

    base = Path(base_dir)
    run_dir = base / phase / exp_name
    checkpoints = _ensure_dir(run_dir / "checkpoints")
    figures = _ensure_dir(run_dir / "figures")
    logs = _ensure_dir(run_dir / "logs")
    _ensure_dir(run_dir)
    return RunPaths(run_dir=run_dir, checkpoints=checkpoints, figures=figures, logs=logs)


def save_checkpoint(state: dict[str, Any], run: RunPaths, *, name: str = "last.ckpt") -> Path:
    path = run.checkpoints / name
    torch.save(state, path)
    return path


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def save_config_snapshot(run: RunPaths, cfg: dict[str, Any], *, name: str = "config.yaml", overwrite: bool = False) -> Path:
    path = run.run_dir / name
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return path


def init_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize torch.distributed if available. Returns (rank, world_size, local_rank, device)."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1 and torch.distributed.is_available():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        print("Training without DDP")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        local_rank = 0
    return rank, world_size, local_rank, device


def is_main_process() -> bool:
    try:
        return torch.distributed.get_rank() == 0
    except Exception:
        return True


def barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def maybe_wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Wrap model with DDP when running under torchrun."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = model.to(device)
        # Use single-device process
        if device.type == "cuda":
            model = DDP(model, device_ids=[device.index] if device.index is not None else None, output_device=device.index)
        else:
            model = DDP(model)
    else:
        model = model.to(device)
    return model


def build_distributed_samplers(dataset, shuffle: bool = True):
    """Return a DistributedSampler if in distributed mode, else None."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler

        return DistributedSampler(dataset, shuffle=shuffle)
    return None


def set_epoch_for_sampler(data_loader, epoch: int) -> None:
    sampler = getattr(data_loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def rank_zero_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def pad_last(x: torch.Tensor, n: int) -> torch.Tensor:
    return F.pad(x, (0, max(0, n - x.shape[-1])))[..., :n]


def lr_schedule(step: int, total: int, lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * lr * (1 + math.cos(math.pi * t))


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model
