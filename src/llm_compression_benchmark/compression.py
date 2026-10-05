from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import prune
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


@dataclass
class QuantizedLoad:
    model: nn.Module
    backend: str
    device: str


def load_fp_model(model_id_or_path: str, *, device: str, trust_remote_code: bool = False) -> nn.Module:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )
    model.eval()
    return model.to(device)


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def load_quantized_model(
    model_id_or_path: str,
    *,
    bits: int,
    backend: str,
    device: str,
    trust_remote_code: bool = False,
) -> QuantizedLoad:
    requested = backend.lower()
    if requested == "bitsandbytes" and not _bnb_available():
        raise RuntimeError("bitsandbytes backend requested but bitsandbytes is not installed")
    use_bnb = requested == "bitsandbytes" or (
        requested == "auto" and device.startswith("cuda") and _bnb_available()
    )

    if use_bnb:
        qconfig = BitsAndBytesConfig(load_in_8bit=bits == 8, load_in_4bit=bits == 4)
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            quantization_config=qconfig,
            device_map={"": 0} if device == "cuda" else "auto",
        )
        model.eval()
        return QuantizedLoad(model=model, backend=f"bitsandbytes-int{bits}", device=device)

    if requested not in {"auto", "torch_dynamic"}:
        raise ValueError(f"Unknown quantization backend: {backend}")
    if bits != 8:
        raise RuntimeError("CPU torch_dynamic fallback supports int8 only; use bitsandbytes for 4-bit")
    if device != "cpu":
        warnings.warn(
            "bitsandbytes is unavailable; falling back to CPU dynamic int8. "
            "Latency comparisons across different devices are not scientifically comparable.",
            stacklevel=2,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32,
    ).eval().to("cpu")
    try:
        quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    except AttributeError:
        quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quantized.eval()
    return QuantizedLoad(model=quantized, backend="torch-dynamic-int8", device="cpu")


def apply_global_magnitude_pruning(model: nn.Module, amount: float) -> nn.Module:
    targets: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
            continue
        if isinstance(module, nn.Embedding):
            continue
        if isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D":
            targets.append((module, "weight"))

    if not targets:
        raise RuntimeError("No prunable 2-D dense weights found")

    prune.global_unstructured(
        targets,
        pruning_method=prune.L1Unstructured,
        amount=float(amount),
    )
    for module, name in targets:
        prune.remove(module, name)
    return model
