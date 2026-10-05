from __future__ import annotations

import gc
import json
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024.0**2)


def tensor_storage_bytes(model: torch.nn.Module) -> int:
    total = 0
    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if not torch.is_tensor(tensor):
            continue
        try:
            ptr = tensor.untyped_storage().data_ptr()
            storage_nbytes = tensor.untyped_storage().nbytes()
        except Exception:
            ptr = tensor.data_ptr()
            storage_nbytes = tensor.numel() * tensor.element_size()
        if ptr not in seen:
            seen.add(ptr)
            total += int(storage_nbytes)
    return total


def effective_nonzero_bytes(model: torch.nn.Module) -> int:
    """Theoretical lower bound ignoring sparse index/metadata overhead."""
    total = 0
    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            continue
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        if tensor.is_floating_point() or tensor.is_complex() or tensor.dtype in {
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            nnz = int(torch.count_nonzero(tensor.detach()).item())
            total += nnz * tensor.element_size()
        else:
            total += tensor.numel() * tensor.element_size()
    return total


def model_sparsity(model: torch.nn.Module) -> float:
    zeros = 0
    total = 0
    for p in model.parameters():
        if p.numel() == 0:
            continue
        total += p.numel()
        zeros += int(torch.count_nonzero(p.detach() == 0).item())
    return zeros / total if total else 0.0


def directory_size_bytes(path: str | Path) -> int:
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def serialized_model_size_bytes(model: torch.nn.Module) -> int:
    """Measure an on-disk model artifact, with a torch state_dict fallback."""
    with tempfile.TemporaryDirectory(prefix="llmcb-size-") as td:
        root = Path(td)
        if hasattr(model, "save_pretrained"):
            try:
                model.save_pretrained(root, safe_serialization=True)
                size = directory_size_bytes(root)
                if size > 0:
                    return size
            except Exception:
                pass
        fallback = root / "state_dict.pt"
        torch.save(model.state_dict(), fallback)
        return fallback.stat().st_size


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def collect_environment() -> dict[str, Any]:
    import datasets
    import transformers

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
    try:
        import bitsandbytes as bnb

        env["bitsandbytes"] = getattr(bnb, "__version__", "unknown")
    except Exception:
        env["bitsandbytes"] = None
    return env
