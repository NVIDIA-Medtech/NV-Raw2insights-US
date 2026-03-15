# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the NV-Raw2Insights-US dataset from HuggingFace Hub.

Run once before training -- subsequent load_dataset() calls use the cache.

    uv run python prepare.py
"""

from datasets import load_dataset

REPO = "nvidia/NV-Raw2Insights-US"

print(f"Downloading {REPO}...")
ds = load_dataset(REPO)
for split in ds:
    d = ds[split]
    print(f"  {split}: {len(d)} samples, columns={d.column_names}")
print("Done. Cached for training.")
