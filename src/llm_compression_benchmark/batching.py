from __future__ import annotations

from typing import Any

import torch


def collate_fixed_blocks(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate already-tokenized fixed-length causal-LM blocks."""
    if not features:
        raise ValueError("Cannot collate an empty feature list")

    keys = {"input_ids", "attention_mask", "labels"} & set(features[0].keys())
    batch: dict[str, Any] = {}
    for key in keys:
        batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
    if "input_ids" not in batch:
        raise KeyError("Each feature must contain input_ids")
    if "attention_mask" not in batch:
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    if "labels" not in batch:
        batch["labels"] = batch["input_ids"].clone()
    return batch
