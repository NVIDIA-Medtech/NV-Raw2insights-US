# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import IQAutoencoder, SoundSpeedDecoder
from utils.training import (
    build_distributed_samplers,
    init_distributed,
    is_main_process,
    load_checkpoint,
    lr_schedule,
    maybe_wrap_ddp,
    rank_zero_print,
    save_checkpoint,
    save_config_snapshot,
    seed_all,
    set_epoch_for_sampler,
    setup_run,
    unwrap,
)

REPO = "nvidia/NV-Raw2Insights-US"
IQ_RMS = 0.6616


def encode_batch(encoder: torch.nn.Module, iq_real: torch.Tensor, iq_imag: torch.Tensor, nf: int, iq_rms: float) -> torch.Tensor:
    """Normalize by dataset RMS, flatten traces, encode, reshape to [B, tx, rx, nf, T_lat]."""
    iq_real, iq_imag = iq_real / iq_rms, iq_imag / iq_rms
    b, tx, rx, t = iq_real.shape
    traces = torch.stack([iq_real.flatten(1, 2), iq_imag.flatten(1, 2)], dim=2).reshape(-1, 2, t)
    latent = encoder(traces)
    return latent.view(b, tx, rx, nf, latent.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--phase1-ckpt", default="experiments/runs/phase1/default/checkpoints/best.ckpt")
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iq-rms", type=float, default=None, help="IQ RMS (auto-detected from HF Hub)")
    args = parser.parse_args()

    seed_all(23)
    torch.set_float32_matmul_precision("high")

    run = setup_run(phase="phase2", exp_name=args.exp_name)
    rank, _, _, device = init_distributed()
    seed_all(23 + rank)

    ckpt_path = run.checkpoints / "last.ckpt"
    if is_main_process():
        save_config_snapshot(run, vars(args))

    train_ds = load_dataset(REPO, split=args.train_split).with_format("torch")
    val_ds = load_dataset(REPO, split=args.val_split).with_format("torch")

    s0 = train_ds[0]
    iq_rms = args.iq_rms or IQ_RMS
    n_tx = int(s0["iq_real"].shape[0])
    n_samples = int(s0["iq_real"].shape[-1])
    out_size = int(np.asarray(s0["sound_speed_map"]).shape[-1])
    nf = args.n_features
    rank_zero_print(f"{len(train_ds)} train / {len(val_ds)} val, iq_rms={iq_rms}, n_tx={n_tx}, n_samples={n_samples}, out_size={out_size}")

    train_sampler = build_distributed_samplers(train_ds, shuffle=True)
    val_sampler = build_distributed_samplers(val_ds, shuffle=False)

    def collate(batch):
        return {
            "iq_real": torch.stack([s["iq_real"].float() for s in batch]),
            "iq_imag": torch.stack([s["iq_imag"].float() for s in batch]),
            "sound_speed_map": torch.stack([s["sound_speed_map"].float() for s in batch]),
        }

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, collate_fn=collate, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, collate_fn=collate, num_workers=0,
    )

    # Frozen phase-1 encoder
    phase1 = IQAutoencoder(in_channels=2, n_features=nf, target_length=n_samples)
    state = load_checkpoint(Path(args.phase1_ckpt))["model"]
    phase1.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()}, strict=False)
    phase1 = phase1.to(device).eval()
    for p in phase1.parameters():
        p.requires_grad = False

    decoder = maybe_wrap_ddp(SoundSpeedDecoder(in_channels=nf, out_size=out_size, n_tx=n_tx), device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)

    start_epoch, step, best_val = 0, 0, float("inf")
    if args.resume and ckpt_path.exists():
        state = load_checkpoint(ckpt_path)
        unwrap(decoder).load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, step = int(state.get("epoch", 0)), int(state.get("step", 0))
        best_val = float(state.get("best_val_loss", best_val))
        rank_zero_print(f"Resumed from {ckpt_path}")

    total_steps = max(1, args.n_epochs * max(1, len(train_loader)))

    def prep_target(y):
        return (y.unsqueeze(1) if y.ndim == 3 else y) / 1000.0

    for epoch in range(start_epoch, args.n_epochs):
        set_epoch_for_sampler(train_loader, epoch)
        decoder.train()
        progress = tqdm(train_loader, desc=f"phase2 {epoch}", disable=not is_main_process())

        for batch in progress:
            iq_real = batch["iq_real"].to(device, non_blocking=True)
            iq_imag = batch["iq_imag"].to(device, non_blocking=True)
            y = prep_target(batch["sound_speed_map"].to(device, non_blocking=True))

            with torch.no_grad():
                latent = encode_batch(phase1.encoder, iq_real, iq_imag, nf, iq_rms)

            optimizer.param_groups[0]["lr"] = lr_schedule(step, total_steps, args.lr, args.warmup_steps)
            optimizer.zero_grad()
            pred = decoder(latent)
            loss = F.mse_loss(pred, y)
            l1 = F.l1_loss(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            if is_main_process():
                progress.set_postfix(loss=f"{loss.item():.4f}", l1=f"{l1.item() * 1000:.1f}m/s")

        if (epoch + 1) % args.save_every:
            continue

        decoder.eval()
        val_loss = val_l1 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                iq_real = batch["iq_real"].to(device, non_blocking=True)
                iq_imag = batch["iq_imag"].to(device, non_blocking=True)
                y = prep_target(batch["sound_speed_map"].to(device, non_blocking=True))
                latent = encode_batch(phase1.encoder, iq_real, iq_imag, nf, iq_rms)
                pred = decoder(latent)
                val_loss += F.mse_loss(pred, y).item()
                val_l1 += F.l1_loss(pred, y).item()

        metrics = torch.tensor([val_loss, val_l1, len(val_loader)], device=device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(metrics)
        n = max(1.0, metrics[2].item())

        if is_main_process():
            avg_loss, avg_l1 = metrics[0].item() / n, metrics[1].item() / n
            ckpt = {"model": unwrap(decoder).state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "step": step, "best_val_loss": best_val,
                    "phase1_ckpt": str(args.phase1_ckpt)}
            save_checkpoint(ckpt, run, name="last.ckpt")
            if avg_loss < best_val:
                best_val = avg_loss
                ckpt["best_val_loss"] = best_val
                save_checkpoint(ckpt, run, name="best.ckpt")
            print(f"Epoch {epoch} | val_loss={avg_loss:.4f} val_l1={avg_l1 * 1000:.1f} m/s")


if __name__ == "__main__":
    main()
