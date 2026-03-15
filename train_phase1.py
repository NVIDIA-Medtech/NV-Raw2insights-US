# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


from models import IQAutoencoder
from utils.metrics import complex_correlation
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


def flatten_traces(iq_real: torch.Tensor, iq_imag: torch.Tensor, iq_rms: float) -> torch.Tensor:
    """Normalize by dataset RMS, reshape [B, TX, RX, T] -> [B*TX*RX, 2, T]."""
    iq_real, iq_imag = iq_real / iq_rms, iq_imag / iq_rms
    b, _, _, t = iq_real.shape
    return torch.stack([iq_real.flatten(1, 2), iq_imag.flatten(1, 2)], dim=2).reshape(-1, 2, t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iq-rms", type=float, default=None, help="IQ RMS (auto-detected from HF Hub)")
    args = parser.parse_args()

    seed_all(42)
    torch.set_float32_matmul_precision("high")

    run = setup_run(phase="phase1", exp_name=args.exp_name)
    rank, _, _, device = init_distributed()
    seed_all(42 + rank)

    ckpt_path = run.checkpoints / "last.ckpt"
    if is_main_process():
        save_config_snapshot(run, vars(args))

    train_ds = load_dataset(REPO, split=args.train_split).with_format("torch")
    val_ds = load_dataset(REPO, split=args.val_split).with_format("torch")

    iq_rms = args.iq_rms or IQ_RMS
    n_samples = int(train_ds[0]["iq_real"].shape[-1])
    rank_zero_print(f"{len(train_ds)} train / {len(val_ds)} val samples, iq_rms={iq_rms}, n_samples={n_samples}")

    train_sampler = build_distributed_samplers(train_ds, shuffle=True)
    val_sampler = build_distributed_samplers(val_ds, shuffle=False)

    def collate(batch):
        return {
            "iq_real": torch.stack([s["iq_real"].float() for s in batch]),
            "iq_imag": torch.stack([s["iq_imag"].float() for s in batch]),
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

    model = maybe_wrap_ddp(
        IQAutoencoder(in_channels=2, n_features=args.n_features, target_length=n_samples), device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch, step, best_val = 0, 0, float("inf")
    if args.resume and ckpt_path.exists():
        state = load_checkpoint(ckpt_path)
        unwrap(model).load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, step = int(state.get("epoch", 0)), int(state.get("step", 0))
        best_val = float(state.get("best_val_loss", best_val))
        rank_zero_print(f"Resumed from {ckpt_path}")

    total_steps = max(1, args.n_epochs * max(1, len(train_loader)))

    for epoch in range(start_epoch, args.n_epochs):
        set_epoch_for_sampler(train_loader, epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"phase1 {epoch}", disable=not is_main_process())

        for batch in progress:
            x = flatten_traces(
                batch["iq_real"].to(device, non_blocking=True),
                batch["iq_imag"].to(device, non_blocking=True),
                iq_rms,
            )
            optimizer.param_groups[0]["lr"] = lr_schedule(step, total_steps, args.lr, args.warmup_steps)
            optimizer.zero_grad()
            # Chunk traces to fit in GPU memory (180x180 = 32k traces per sample)
            n_traces, chunk = x.shape[0], 4096
            mse_sum = cc_sum = 0.0
            for i in range(0, n_traces, chunk):
                xi = x[i : i + chunk]
                ri = model(xi)
                mse_i = F.mse_loss(ri, xi, reduction="sum")
                cc_i = complex_correlation(ri, xi).sum()
                (mse_i / x.numel() + 0.1 * (1.0 - cc_i / n_traces)).backward()
                mse_sum += mse_i.item()
                cc_sum += cc_i.item()
            optimizer.step()
            step += 1
            if is_main_process():
                progress.set_postfix(loss=f"{mse_sum / x.numel():.4f}", cc=f"{cc_sum / n_traces:.4f}")

        if (epoch + 1) % args.save_every:
            continue

        model.eval()
        val_loss = val_mse = val_cc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = flatten_traces(
                    batch["iq_real"].to(device, non_blocking=True),
                    batch["iq_imag"].to(device, non_blocking=True),
                    iq_rms,
                )
                chunks = [model(x[i : i + 4096]) for i in range(0, x.shape[0], 4096)]
                recon = torch.cat(chunks)
                mse = F.mse_loss(recon, x)
                cc = complex_correlation(recon, x).mean()
                val_loss += (mse + 0.1 * (1.0 - cc)).item()
                val_mse += mse.item()
                val_cc += cc.item()

        metrics = torch.tensor([val_loss, val_mse, val_cc, len(val_loader)], device=device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(metrics)
        n = max(1.0, metrics[3].item())

        if is_main_process():
            avg_loss, avg_mse, avg_cc = metrics[0].item() / n, metrics[1].item() / n, metrics[2].item() / n
            ckpt = {"model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "step": step, "best_val_loss": best_val}
            save_checkpoint(ckpt, run, name="last.ckpt")
            if avg_loss < best_val:
                best_val = avg_loss
                ckpt["best_val_loss"] = best_val
                save_checkpoint(ckpt, run, name="best.ckpt")
            print(f"Epoch {epoch} | val_loss={avg_loss:.4f} val_mse={avg_mse:.4f} val_cc={avg_cc:.4f}")


if __name__ == "__main__":
    main()
