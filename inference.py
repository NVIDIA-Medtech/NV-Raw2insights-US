"""Inference: stream HF validation data, predict sound speed, plot results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from models import IQAutoencoder, SoundSpeedDecoder

REPO = "nvidia/NV-Raw2Insights-US"
IQ_RMS = 0.6616


def load_weights(filename, device):
    path = hf_hub_download(REPO, filename, repo_type="model")
    return torch.load(path, map_location=device, weights_only=True)["model"]


def main():
    p = argparse.ArgumentParser(description="Sound speed inference on HF validation data")
    p.add_argument("--split", default="validation")
    p.add_argument("--n-infer", type=int, default=5)
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--iq-rms", type=float, default=IQ_RMS)
    p.add_argument("--output-dir", default="inference_results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Probe dataset for shapes
    ds = load_dataset(REPO, split=args.split, streaming=True)
    s0 = next(iter(ds))
    iq0 = np.asarray(s0["iq_real"])
    n_tx, n_rx, n_samples = iq0.shape
    out_size = np.asarray(s0["sound_speed_map"]).shape[-1]
    nf = args.n_features

    # Load models
    encoder = IQAutoencoder(in_channels=2, n_features=nf, target_length=n_samples)
    encoder.load_state_dict(load_weights("phase1.pt", device))
    encoder = encoder.to(device).eval()

    decoder = SoundSpeedDecoder(in_channels=nf, out_size=out_size, n_tx=n_tx)
    decoder.load_state_dict(load_weights("phase2.pt", device))
    decoder = decoder.to(device).eval()

    # Stream samples and predict
    ds = load_dataset(REPO, split=args.split, streaming=True).with_format("torch")
    for idx, sample in enumerate(ds):
        if idx >= args.n_infer:
            break

        iq_r = sample["iq_real"].float().unsqueeze(0).to(device) / args.iq_rms
        iq_i = sample["iq_imag"].float().unsqueeze(0).to(device) / args.iq_rms
        gt = np.asarray(sample["sound_speed_map"]).squeeze()
        bmode = np.asarray(sample["bmode_focused"]).squeeze()

        with torch.no_grad():
            b, tx, rx, t = iq_r.shape
            traces = torch.stack([iq_r.flatten(1, 2), iq_i.flatten(1, 2)], dim=2).reshape(-1, 2, t)
            latent = encoder.encoder(traces).view(b, tx, rx, nf, -1)
            pred = decoder(latent)[0, 0].cpu().numpy() * 1000

        mae = np.abs(pred - gt).mean()
        print(f"[{idx}] MAE = {mae:.1f} m/s")

        # Plot: focused B-mode | GT sound speed | predicted sound speed
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        bmode_db = 20 * np.log10(np.maximum(bmode, 1e-6))
        bmode_db -= bmode_db.max()
        axes[0].imshow(bmode_db, cmap="gray", vmin=-60, vmax=0, aspect="auto")
        axes[0].set_title("Focused B-mode")

        vmin, vmax = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        im1 = axes[1].imshow(gt, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto")
        axes[1].set_title("GT sound speed")

        im2 = axes[2].imshow(pred, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto")
        axes[2].set_title(f"Predicted (MAE={mae:.1f} m/s)")

        for ax in axes:
            ax.axis("off")
        fig.colorbar(im2, ax=axes[2], label="m/s", shrink=0.8)

        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{idx:03d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {min(args.n_infer, idx + 1)} figures to {out_dir}/")


if __name__ == "__main__":
    main()
