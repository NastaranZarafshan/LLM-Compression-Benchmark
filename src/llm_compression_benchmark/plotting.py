from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _label_for_row(row: pd.Series, multiple_experiments: bool) -> str:
    label = str(row["method"])
    if multiple_experiments:
        label = f"{row['experiment']} / {label}"
    return label


def _scatter_with_labels(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    output: Path,
    *,
    yerr: str | None = None,
    draw_pareto: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    multiple_experiments = "experiment" in df.columns and df["experiment"].nunique() > 1

    for _, row in df.iterrows():
        if yerr and yerr in df.columns:
            ax.errorbar(row[x], row[y], yerr=row[yerr], fmt="o", capsize=3)
        else:
            ax.scatter(row[x], row[y], s=70)
        ax.annotate(
            _label_for_row(row, multiple_experiments),
            (row[x], row[y]),
            xytext=(6, 5),
            textcoords="offset points",
        )

    if draw_pareto and not multiple_experiments and "pareto_size_accuracy" in df.columns:
        frontier = df[df["pareto_size_accuracy"].astype(bool)].sort_values(x)
        if len(frontier) >= 2:
            ax.plot(frontier[x], frontier[y], linestyle="--", linewidth=1.2, label="Pareto frontier")
            ax.legend()

    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def create_plots(
    results_csv: str | Path,
    output_dir: str | Path,
    *,
    size_metric: str = "model_size_mb",
    performance_metric: str = "accuracy",
) -> list[Path]:
    df = pd.read_csv(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"method", size_metric, performance_metric, "perplexity", "latency_ms", "throughput_tok_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV is missing columns: {sorted(missing)}")

    outputs: list[Path] = []
    p = output_dir / "performance_vs_model_size.png"
    draw_pareto = size_metric == "model_size_mb" and performance_metric in {"accuracy", "token_accuracy"}
    _scatter_with_labels(
        df,
        size_metric,
        performance_metric,
        "Performance vs Model Size",
        p,
        draw_pareto=draw_pareto,
    )
    outputs.append(p)

    p = output_dir / "perplexity_vs_model_size.png"
    _scatter_with_labels(df, size_metric, "perplexity", "Perplexity vs Model Size (lower is better)", p)
    outputs.append(p)

    p = output_dir / "latency_vs_model_size.png"
    if "latency_mean_ms" in df.columns:
        if "latency_mean_ms_across_seed_std" in df.columns:
            latency_yerr = "latency_mean_ms_across_seed_std"
            latency_title = "Latency vs Model Size (across-seed mean ± std; lower is better)"
        elif "latency_std_ms" in df.columns:
            latency_yerr = "latency_std_ms"
            latency_title = "Latency vs Model Size (timed-run mean ± std; lower is better)"
        else:
            latency_yerr = None
            latency_title = "Latency vs Model Size (mean; lower is better)"
        _scatter_with_labels(
            df,
            size_metric,
            "latency_mean_ms",
            latency_title,
            p,
            yerr=latency_yerr,
        )
    else:
        _scatter_with_labels(df, size_metric, "latency_ms", "Latency vs Model Size (lower is better)", p)
    outputs.append(p)

    p = output_dir / "throughput_vs_model_size.png"
    _scatter_with_labels(df, size_metric, "throughput_tok_s", "Throughput vs Model Size (higher is better)", p)
    outputs.append(p)
    return outputs
