from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from .batching import collate_fixed_blocks
from .utils import (
    bytes_to_mb,
    effective_nonzero_bytes,
    model_sparsity,
    serialized_model_size_bytes,
    tensor_storage_bytes,
)


@dataclass(frozen=True)
class QualityResult:
    """Quality metrics measured on the common causal-LM evaluation corpus."""

    perplexity: float
    token_accuracy: float
    eval_tokens: int
    mean_loss: float


@dataclass(frozen=True)
class GenerationResult:
    """Runtime metrics for deterministic autoregressive generation.

    ``latency_ms`` is kept as the canonical/backward-compatible median latency.
    Additional fields expose the distribution so experiments do not rely on one
    point estimate. CUDA memory fields are zero on non-CUDA devices.
    """

    latency_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_p95_ms: float
    latency_min_ms: float
    latency_max_ms: float
    throughput_tok_s: float
    vram_peak_mb: float
    vram_resident_mb: float
    vram_generation_delta_mb: float
    vram_peak_reserved_mb: float
    generated_tokens: int
    timed_runs: int
    latency_samples_ms: tuple[float, ...]


def _model_input_device(model: torch.nn.Module) -> torch.device:
    """Return the device on which input_ids should be placed.

    This handles ordinary PyTorch models as well as Hugging Face models loaded
    with Accelerate/device_map (including bitsandbytes quantized models).
    """
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        # Prefer a real CUDA/MPS/CPU execution device and skip offload markers.
        for value in device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, torch.device):
                return value
            if isinstance(value, str) and value not in {"disk", "meta"}:
                try:
                    return torch.device(value)
                except (RuntimeError, ValueError):
                    pass

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return bytes_to_mb(torch.cuda.memory_allocated(device))


def _cuda_peak_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return bytes_to_mb(torch.cuda.max_memory_allocated(device))


def _cuda_peak_reserved_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return bytes_to_mb(torch.cuda.max_memory_reserved(device))


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile without adding a NumPy dependency here."""
    if not values:
        return float("nan")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def evaluate_quality(
    model: torch.nn.Module,
    eval_dataset: Any,
    *,
    batch_size: int,
    max_batches: int,
) -> QualityResult:
    """Compute perplexity and next-token top-1 accuracy.

    Both metrics use the same shifted causal-LM targets. Loss is aggregated by
    the number of valid target tokens rather than averaging batch means, which
    keeps perplexity correct when the last batch is smaller.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_batches <= 0:
        raise ValueError("max_batches must be > 0")

    loader = DataLoader(
        eval_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_fixed_blocks,
        drop_last=False,
    )
    if len(loader) == 0:
        raise ValueError("Evaluation DataLoader is empty")

    device = _model_input_device(model)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= int(max_batches):
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            logits = outputs.logits

            # Causal LM predicts token t from positions < t, hence the shift.
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            valid = attention_mask[:, 1:].bool() & shift_labels.ne(-100)
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue

            # HF causal-LM loss is already the mean over valid shifted tokens.
            # Weight each batch by token count so the corpus mean is exact.
            batch_loss = float(outputs.loss.detach().float().item())
            total_loss += batch_loss * n_valid
            total_tokens += n_valid

            predictions = shift_logits.argmax(dim=-1)
            total_correct += int((predictions.eq(shift_labels) & valid).sum().item())

    if total_tokens == 0:
        raise RuntimeError("No valid evaluation tokens were produced")

    mean_loss = total_loss / total_tokens
    # exp can overflow for pathological models; report +inf rather than crash.
    perplexity = math.exp(mean_loss) if mean_loss < 709.0 else float("inf")
    accuracy = total_correct / total_tokens
    return QualityResult(
        perplexity=float(perplexity),
        token_accuracy=float(accuracy),
        eval_tokens=int(total_tokens),
        mean_loss=float(mean_loss),
    )


def benchmark_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
    warmup_runs: int,
    timed_runs: int,
) -> GenerationResult:
    """Benchmark deterministic generation with synchronized wall-clock timing.

    Latency statistics are computed over synchronized end-to-end greedy
    generation calls. Throughput is actual generated continuation tokens divided
    by total timed wall-clock time. CUDA peak allocation includes the operational
    resident model plus generation buffers; ``vram_generation_delta_mb`` isolates
    the extra allocation above the post-warmup resident footprint.
    """
    if not prompt:
        raise ValueError("prompt must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if timed_runs <= 0:
        raise ValueError("timed_runs must be > 0")

    device = _model_input_device(model)
    model.eval()

    encoded = tokenizer(
        [prompt] * int(batch_size),
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in encoded.items()
    }
    prompt_length = int(inputs["input_ids"].shape[1])

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = int(eos_token_id)

    with torch.inference_mode():
        for _ in range(int(warmup_runs)):
            model.generate(**inputs, **generation_kwargs)
            _sync_if_cuda(device)

        _sync_if_cuda(device)
        vram_resident_mb = _cuda_allocated_mb(device)
        _reset_cuda_peak(device)

        latencies_s: list[float] = []
        generated_token_counts: list[int] = []

        for _ in range(int(timed_runs)):
            _sync_if_cuda(device)
            start = time.perf_counter()
            output_ids = model.generate(**inputs, **generation_kwargs)
            _sync_if_cuda(device)
            elapsed = time.perf_counter() - start

            latencies_s.append(elapsed)
            # All sequences in a generated tensor have equal padded width. This
            # measures produced continuation token slots, the standard throughput
            # quantity for batched autoregressive decoding.
            new_per_sequence = max(0, int(output_ids.shape[1]) - prompt_length)
            generated_token_counts.append(new_per_sequence * int(output_ids.shape[0]))

    total_time = sum(latencies_s)
    total_generated = sum(generated_token_counts)
    throughput = total_generated / total_time if total_time > 0 else float("nan")
    latency_samples_ms = [value * 1000.0 for value in latencies_s]
    latency_mean_ms = statistics.fmean(latency_samples_ms)
    latency_std_ms = statistics.pstdev(latency_samples_ms) if len(latency_samples_ms) > 1 else 0.0
    latency_median_ms = statistics.median(latency_samples_ms)
    latency_p95_ms = _percentile(latency_samples_ms, 0.95)
    vram_peak_mb = _cuda_peak_mb(device)

    return GenerationResult(
        latency_ms=float(latency_median_ms),
        latency_mean_ms=float(latency_mean_ms),
        latency_std_ms=float(latency_std_ms),
        latency_p95_ms=float(latency_p95_ms),
        latency_min_ms=float(min(latency_samples_ms)),
        latency_max_ms=float(max(latency_samples_ms)),
        throughput_tok_s=float(throughput),
        vram_peak_mb=float(vram_peak_mb),
        vram_resident_mb=float(vram_resident_mb),
        vram_generation_delta_mb=float(max(0.0, vram_peak_mb - vram_resident_mb)),
        vram_peak_reserved_mb=float(_cuda_peak_reserved_mb(device)),
        generated_tokens=int(total_generated),
        timed_runs=int(timed_runs),
        latency_samples_ms=tuple(float(x) for x in latency_samples_ms),
    )


def collect_model_metrics(model: torch.nn.Module) -> dict[str, float | int]:
    """Collect deployability/storage metrics for one model variant."""
    parameter_count = sum(int(p.numel()) for p in model.parameters())

    # Keep each measurement independent: serialized size is what users actually
    # store; tensor storage reflects live tensors; effective size is only a
    # theoretical lower bound for sparse models and ignores index metadata.
    model_size = bytes_to_mb(serialized_model_size_bytes(model))
    tensor_size = bytes_to_mb(tensor_storage_bytes(model))

    try:
        effective_size = bytes_to_mb(effective_nonzero_bytes(model))
    except (RuntimeError, NotImplementedError, TypeError):
        # Some third-party quantized tensor classes do not implement count_nonzero.
        effective_size = tensor_size

    try:
        sparsity = float(model_sparsity(model))
    except (RuntimeError, NotImplementedError, TypeError):
        sparsity = 0.0

    return {
        "parameters_m": parameter_count / 1_000_000.0,
        "model_size_mb": float(model_size),
        "tensor_storage_mb": float(tensor_size),
        "effective_size_mb": float(effective_size),
        "sparsity_pct": float(sparsity * 100.0),
    }
