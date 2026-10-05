from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from .analysis import add_pareto_flags
from .compression import apply_global_magnitude_pruning, load_fp_model, load_quantized_model
from .data import prepare_lm_dataset
from .distill import distill_student
from .evaluate import benchmark_generation, collect_model_metrics, evaluate_quality
from .hf_compat import canonical_model_id
from .plotting import create_plots
from .utils import cleanup, collect_environment, resolve_device, save_json, set_seed


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "variant"


def _evaluate_variant(
    method: str,
    model: torch.nn.Module,
    tokenizer: Any,
    eval_dataset: Any,
    cfg: dict[str, Any],
    *,
    backend: str,
    device_label: str,
    output_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality = evaluate_quality(
        model,
        eval_dataset,
        batch_size=int(cfg["benchmark"]["eval_batch_size"]),
        max_batches=int(cfg["benchmark"]["eval_max_batches"]),
    )
    runtime = benchmark_generation(
        model,
        tokenizer,
        prompt=str(cfg["benchmark"]["prompt"]),
        batch_size=int(cfg["benchmark"]["generation_batch_size"]),
        max_new_tokens=int(cfg["benchmark"]["max_new_tokens"]),
        warmup_runs=int(cfg["benchmark"]["warmup_runs"]),
        timed_runs=int(cfg["benchmark"]["timed_runs"]),
    )

    samples_rel = Path("runtime_samples") / f"{_safe_filename(method)}.json"
    save_json(
        {
            "method": method,
            "backend": backend,
            "device": device_label,
            **asdict(runtime),
        },
        output_dir / samples_rel,
    )

    row = {
        "experiment": cfg["experiment"]["name"],
        "seed": int(cfg["experiment"]["seed"]),
        "method": method,
        "backend": backend,
        "device": device_label,
        **collect_model_metrics(model),
        # Backward-compatible canonical latency: median of timed runs.
        "latency_ms": runtime.latency_ms,
        "latency_mean_ms": runtime.latency_mean_ms,
        "latency_std_ms": runtime.latency_std_ms,
        "latency_p95_ms": runtime.latency_p95_ms,
        "latency_min_ms": runtime.latency_min_ms,
        "latency_max_ms": runtime.latency_max_ms,
        "timed_runs": runtime.timed_runs,
        "throughput_tok_s": runtime.throughput_tok_s,
        "vram_peak_mb": runtime.vram_peak_mb,
        "vram_resident_mb": runtime.vram_resident_mb,
        "vram_generation_delta_mb": runtime.vram_generation_delta_mb,
        "vram_peak_reserved_mb": runtime.vram_peak_reserved_mb,
        "perplexity": quality.perplexity,
        "accuracy": quality.token_accuracy,
        "token_accuracy": quality.token_accuracy,
        "eval_tokens": quality.eval_tokens,
        "runtime_samples_file": samples_rel.as_posix(),
    }
    if extra:
        row.update(extra)
    return row


def _persist_model_if_requested(model: torch.nn.Module, output_dir: Path, name: str, enabled: bool) -> None:
    if not enabled:
        return
    target = output_dir / "models" / name
    target.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(target, safe_serialization=True)
    else:
        torch.save(model.state_dict(), target / "state_dict.pt")


def _pruning_amounts(cfg: dict[str, Any]) -> list[float]:
    sweep = cfg["pruning"].get("sweep_amounts")
    if sweep:
        return [float(value) for value in sweep]
    return [float(cfg["pruning"]["amount"])]



def run_experiment(cfg: dict[str, Any], config_path: str | Path | None = None) -> pd.DataFrame:
    set_seed(int(cfg["experiment"]["seed"]))
    device = resolve_device(str(cfg["benchmark"]["device"]))
    outdir = Path(cfg["experiment"]["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    if config_path:
        with (outdir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    save_json(collect_environment(), outdir / "environment.json")

    teacher_id = canonical_model_id(str(cfg["model"]["teacher"]))
    student_raw = str(cfg["model"].get("student", ""))
    student_id = canonical_model_id(student_raw) if student_raw else ""
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(teacher_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_cfg = cfg["data"]
    eval_dataset = prepare_lm_dataset(
        tokenizer,
        dataset_name=data_cfg["dataset_name"],
        dataset_config=data_cfg.get("dataset_config"),
        split=data_cfg["eval_split"],
        text_column=data_cfg["text_column"],
        block_size=int(data_cfg["block_size"]),
        max_samples=data_cfg.get("max_eval_samples"),
    )

    need_train = cfg["methods"].get("distillation") or cfg["methods"].get("quantization_distillation")
    train_dataset = None
    if need_train:
        train_dataset = prepare_lm_dataset(
            tokenizer,
            dataset_name=data_cfg["dataset_name"],
            dataset_config=data_cfg.get("dataset_config"),
            split=data_cfg["train_split"],
            text_column=data_cfg["text_column"],
            block_size=int(data_cfg["block_size"]),
            max_samples=data_cfg.get("max_train_samples"),
        )

    rows: list[dict[str, Any]] = []
    save_models = bool(cfg["benchmark"].get("save_models", False))

    if cfg["methods"].get("baseline"):
        model = load_fp_model(teacher_id, device=device, trust_remote_code=trust_remote_code)
        rows.append(
            _evaluate_variant(
                "baseline",
                model,
                tokenizer,
                eval_dataset,
                cfg,
                backend="fp",
                device_label=device,
                output_dir=outdir,
            )
        )
        _persist_model_if_requested(model, outdir, "baseline", save_models)
        del model
        cleanup()

    if cfg["methods"].get("quantization"):
        q = load_quantized_model(
            teacher_id,
            bits=int(cfg["quantization"]["bits"]),
            backend=str(cfg["quantization"]["backend"]),
            device=device,
            trust_remote_code=trust_remote_code,
        )
        rows.append(
            _evaluate_variant(
                "quantization",
                q.model,
                tokenizer,
                eval_dataset,
                cfg,
                backend=q.backend,
                device_label=q.device,
                output_dir=outdir,
            )
        )
        _persist_model_if_requested(q.model, outdir, "quantization", save_models)
        del q
        cleanup()

    if cfg["methods"].get("pruning"):
        pruning_amounts = _pruning_amounts(cfg)
        for amount in pruning_amounts:
            model = load_fp_model(teacher_id, device=device, trust_remote_code=trust_remote_code)
            model = apply_global_magnitude_pruning(model, amount)
            method = "pruning" if len(pruning_amounts) == 1 else f"pruning@{amount:.0%}"
            rows.append(
                _evaluate_variant(
                    method,
                    model,
                    tokenizer,
                    eval_dataset,
                    cfg,
                    backend=f"global-l1-{amount:.2f}",
                    device_label=device,
                    output_dir=outdir,
                    extra={"pruning_amount": amount},
                )
            )
            _persist_model_if_requested(model, outdir, _safe_filename(method), save_models)
            del model
            cleanup()

    distilled_model: torch.nn.Module | None = None
    distill_extra: dict[str, Any] = {}
    distilled_checkpoint: tempfile.TemporaryDirectory[str] | None = None

    if need_train:
        dcfg = cfg["distillation"]
        result = distill_student(
            teacher_id,
            student_id,
            train_dataset,
            device=device,
            trust_remote_code=trust_remote_code,
            temperature=float(dcfg["temperature"]),
            alpha_kl=float(dcfg["alpha_kl"]),
            learning_rate=float(dcfg["learning_rate"]),
            batch_size=int(dcfg["batch_size"]),
            gradient_accumulation_steps=int(dcfg["gradient_accumulation_steps"]),
            max_steps=int(dcfg["max_steps"]),
            warmup_steps=int(dcfg["warmup_steps"]),
            weight_decay=float(dcfg["weight_decay"]),
            max_grad_norm=float(dcfg["max_grad_norm"]),
            fp16=bool(dcfg["fp16"]),
        )
        distilled_model = result.student
        distill_extra = {"distillation_loss": result.final_loss, "distillation_steps": result.steps}
        # Drop the result wrapper immediately; otherwise it keeps a second live
        # Python reference to the student and prevents deterministic GPU cleanup.
        del result

        if cfg["methods"].get("quantization_distillation"):
            distilled_checkpoint = tempfile.TemporaryDirectory(prefix="llmcb-distilled-")
            distilled_model.save_pretrained(distilled_checkpoint.name, safe_serialization=True)

    if cfg["methods"].get("distillation") and distilled_model is not None:
        rows.append(
            _evaluate_variant(
                "distillation",
                distilled_model,
                tokenizer,
                eval_dataset,
                cfg,
                backend="offline-logit-kd",
                device_label=device,
                output_dir=outdir,
                extra=distill_extra,
            )
        )
        _persist_model_if_requested(distilled_model, outdir, "distillation", save_models)

    # Critical lifecycle rule: the FP student must be gone before loading the
    # quantized student. Otherwise vram_peak_mb measures both models at once.
    if distilled_model is not None:
        del distilled_model
        distilled_model = None
        cleanup()

    if cfg["methods"].get("quantization_distillation"):
        if distilled_checkpoint is None:
            raise RuntimeError("Quantization+distillation requires a distilled checkpoint")
        qd = load_quantized_model(
            distilled_checkpoint.name,
            bits=int(cfg["quantization"]["bits"]),
            backend=str(cfg["quantization"]["backend"]),
            device=device,
            trust_remote_code=trust_remote_code,
        )
        rows.append(
            _evaluate_variant(
                "quantization+distillation",
                qd.model,
                tokenizer,
                eval_dataset,
                cfg,
                backend=f"{qd.backend}+offline-logit-kd",
                device_label=qd.device,
                output_dir=outdir,
                extra=distill_extra,
            )
        )
        _persist_model_if_requested(qd.model, outdir, "quantization_distillation", save_models)
        del qd
        cleanup()

    if distilled_checkpoint is not None:
        distilled_checkpoint.cleanup()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No benchmark methods were enabled")

    baseline_rows = df[df["method"] == "baseline"]
    if not baseline_rows.empty:
        baseline_size = float(baseline_rows.iloc[0]["model_size_mb"])
        baseline_acc = float(baseline_rows.iloc[0]["token_accuracy"])
        baseline_ppl = float(baseline_rows.iloc[0]["perplexity"])
        df["compression_ratio"] = baseline_size / df["model_size_mb"]
        df["accuracy_retention"] = df["token_accuracy"] / baseline_acc if baseline_acc else float("nan")
        df["perplexity_ratio"] = df["perplexity"] / baseline_ppl if baseline_ppl else float("nan")
    else:
        df["compression_ratio"] = float("nan")
        df["accuracy_retention"] = float("nan")
        df["perplexity_ratio"] = float("nan")

    df = add_pareto_flags(df)
    df.to_csv(outdir / "results.csv", index=False)
    with (outdir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=str)

    create_plots(
        outdir / "results.csv",
        outdir / "plots",
        size_metric=str(cfg["plot"]["size_metric"]),
        performance_metric=str(cfg["plot"]["performance_metric"]),
    )
    return df
