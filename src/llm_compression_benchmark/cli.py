from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from .analysis import summarize_replicates
from .benchmark import run_experiment
from .config import load_config
from .plotting import create_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-compression-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a compression benchmark experiment")
    run.add_argument("--config", required=True, help="YAML experiment config")

    replicate = sub.add_parser("replicate", help="Run the same experiment with independent seeds")
    replicate.add_argument("--config", required=True, help="YAML experiment config")
    replicate.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    replicate.add_argument(
        "--output-dir",
        default=None,
        help="Root directory for per-seed runs and aggregate files",
    )

    plot = sub.add_parser("plot", help="Regenerate plots from an existing results.csv")
    plot.add_argument("--results", required=True, help="Path to results.csv")
    plot.add_argument("--output-dir", default="outputs/plots", help="Directory for PNG files")
    plot.add_argument("--size-metric", default="model_size_mb")
    plot.add_argument("--performance-metric", default="accuracy")

    aggregate = sub.add_parser("aggregate", help="Combine results from multiple model experiments")
    aggregate.add_argument("--results", nargs="+", required=True, help="Two or more results.csv files")
    aggregate.add_argument("--output", default="outputs/suite/results.csv")
    aggregate.add_argument("--plot-dir", default="outputs/suite/plots")
    aggregate.add_argument("--size-metric", default="model_size_mb")
    aggregate.add_argument("--performance-metric", default="accuracy")
    return parser


def _print_run_table(df: pd.DataFrame) -> None:
    cols = [
        "method",
        "model_size_mb",
        "compression_ratio",
        "vram_resident_mb",
        "vram_peak_mb",
        "latency_ms",
        "latency_std_ms",
        "latency_p95_ms",
        "throughput_tok_s",
        "accuracy",
        "perplexity",
        "sparsity_pct",
        "pareto_size_accuracy",
    ]
    existing = [col for col in cols if col in df.columns]
    print(df[existing].to_string(index=False))


def _run_replicates(config_path: str, seeds: list[int], output_dir: str | None) -> None:
    base_cfg = load_config(config_path)
    base_output = Path(base_cfg["experiment"]["output_dir"])
    root = Path(output_dir) if output_dir else base_output.parent / f"{base_output.name}-replicates"
    root.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for seed in seeds:
        cfg = deepcopy(base_cfg)
        cfg["experiment"]["seed"] = int(seed)
        cfg["experiment"]["output_dir"] = str(root / f"seed-{seed}")
        df = run_experiment(cfg, config_path=config_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined_path = root / "replicates.csv"
    combined.to_csv(combined_path, index=False)

    summary = summarize_replicates(combined)
    summary_path = root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    create_plots(
        summary_path,
        root / "plots",
        size_metric=str(base_cfg["plot"]["size_metric"]),
        performance_metric=str(base_cfg["plot"]["performance_metric"]),
    )

    print("\nAcross-seed summary:")
    summary_cols = [
        "method",
        "replicate_count",
        "model_size_mb",
        "accuracy",
        "accuracy_across_seed_std",
        "perplexity",
        "perplexity_across_seed_std",
        "latency_ms",
        "latency_ms_across_seed_std",
        "throughput_tok_s",
        "throughput_tok_s_across_seed_std",
    ]
    existing = [col for col in summary_cols if col in summary.columns]
    print(summary[existing].to_string(index=False))
    print(f"\nSaved replicate rows to {combined_path.resolve()}")
    print(f"Saved across-seed summary to {summary_path.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        cfg = load_config(args.config)
        df = run_experiment(cfg, config_path=args.config)
        _print_run_table(df)
        print(f"\nSaved results to {Path(cfg['experiment']['output_dir']).resolve()}")
    elif args.command == "replicate":
        _run_replicates(args.config, args.seeds, args.output_dir)
    elif args.command == "plot":
        paths = create_plots(
            args.results,
            args.output_dir,
            size_metric=args.size_metric,
            performance_metric=args.performance_metric,
        )
        for path in paths:
            print(path)
    elif args.command == "aggregate":
        frames = [pd.read_csv(path) for path in args.results]
        combined = pd.concat(frames, ignore_index=True)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output, index=False)
        paths = create_plots(
            output,
            args.plot_dir,
            size_metric=args.size_metric,
            performance_metric=args.performance_metric,
        )
        print(f"Combined results: {output}")
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
