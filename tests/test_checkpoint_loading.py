# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for the public training checkpoint format."""

import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from utils.training import load_checkpoint, save_checkpoint, setup_run


class UnsupportedMetadata:
    """An inert custom object that must never be allowlisted by the loader."""


class CheckpointLoadingTests(unittest.TestCase):
    def test_training_state_round_trip_and_resume(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 1,
            "step": 1,
            "best_val_loss": 0.25,
            "phase1_ckpt": "phase1/best.ckpt",
        }
        with TemporaryDirectory() as directory:
            run = setup_run("phase2", base_dir=directory)
            checkpoint = save_checkpoint(state, run)
            loaded = load_checkpoint(str(checkpoint))

        restored = torch.nn.Linear(2, 1)
        restored.load_state_dict(loaded["model"])
        restored_optimizer = torch.optim.AdamW(restored.parameters())
        restored_optimizer.load_state_dict(loaded["optimizer"])
        for key in ("epoch", "step", "best_val_loss", "phase1_ckpt"):
            self.assertEqual(loaded[key], state[key])
        for value in loaded["model"].values():
            self.assertEqual(value.device.type, "cpu")
        # A resumed optimizer must produce the same next update.
        for network, optim in ((model, optimizer), (restored, restored_optimizer)):
            network(torch.ones(1, 2)).sum().backward()
            optim.step()
        for actual, expected in zip(restored.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_custom_objects_rejected_even_if_unsafe_default_requested(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "unsupported.ckpt"
            torch.save({"model": {}, "metadata": UnsupportedMetadata()}, checkpoint)
            # Explicit weights_only=True must win over this PyTorch override.
            with patch.dict(os.environ, {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                                         "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "0"}):
                with self.assertRaises(pickle.UnpicklingError):
                    load_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
