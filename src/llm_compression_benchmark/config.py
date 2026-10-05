from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "compression-benchmark", "seed": 42, "output_dir": "outputs"},
    "model": {"trust_remote_code": False},
    "methods": {
        "baseline": True,
        "quantization": True,
        "pruning": True,
        "distillation": True,
        "quantization_distillation": True,
    },
    "quantization": {"bits": 8, "backend": "auto"},
    "pruning": {"amount": 0.10, "sweep_amounts": []},
    "distillation": {
        "temperature": 2.0,
        "alpha_kl": 0.70,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_steps": 100,
        "warmup_steps": 10,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "fp16": True,
    },
    "data": {
        "dataset_name": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "text_column": "text",
        "train_split": "train",
        "eval_split": "validation",
        "block_size": 256,
        "max_train_samples": 4096,
        "max_eval_samples": 1024,
    },
    "benchmark": {
        "device": "auto",
        "eval_batch_size": 4,
        "eval_max_batches": 64,
        "generation_batch_size": 1,
        "prompt": "Language models can be compressed by",
        "max_new_tokens": 32,
        "warmup_runs": 10,
        "timed_runs": 30,
        "save_models": False,
    },
    "plot": {"size_metric": "model_size_mb", "performance_metric": "accuracy"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user_cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    model = cfg.get("model", {})
    if not model.get("teacher"):
        raise ValueError("model.teacher is required")
    distill_enabled = cfg["methods"].get("distillation") or cfg["methods"].get("quantization_distillation")
    if distill_enabled and not model.get("student"):
        raise ValueError("model.student is required when distillation is enabled")
    bits = int(cfg["quantization"].get("bits", 8))
    if bits not in {4, 8}:
        raise ValueError("quantization.bits must be 4 or 8")
    amount = float(cfg["pruning"].get("amount", 0.1))
    if not 0.0 <= amount < 1.0:
        raise ValueError("pruning.amount must be in [0, 1)")
    sweep = cfg["pruning"].get("sweep_amounts") or []
    if not isinstance(sweep, list):
        raise ValueError("pruning.sweep_amounts must be a list")
    for value in sweep:
        parsed = float(value)
        if not 0.0 <= parsed < 1.0:
            raise ValueError("every pruning.sweep_amounts value must be in [0, 1)")
    alpha = float(cfg["distillation"].get("alpha_kl", 0.7))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("distillation.alpha_kl must be in [0, 1]")
    if int(cfg["data"].get("block_size", 0)) < 2:
        raise ValueError("data.block_size must be >= 2")
    if int(cfg["benchmark"].get("warmup_runs", -1)) < 0:
        raise ValueError("benchmark.warmup_runs must be >= 0")
    if int(cfg["benchmark"].get("timed_runs", 0)) <= 0:
        raise ValueError("benchmark.timed_runs must be > 0")
    if int(cfg["benchmark"].get("eval_batch_size", 0)) <= 0:
        raise ValueError("benchmark.eval_batch_size must be > 0")
    if int(cfg["benchmark"].get("generation_batch_size", 0)) <= 0:
        raise ValueError("benchmark.generation_batch_size must be > 0")
