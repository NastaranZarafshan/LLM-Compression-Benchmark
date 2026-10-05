from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def add_pareto_flags(
    df: pd.DataFrame,
    *,
    size_col: str = "model_size_mb",
    quality_col: str = "accuracy",
    output_col: str = "pareto_size_accuracy",
) -> pd.DataFrame:
    """Mark points that are non-dominated for smaller size / higher quality."""
    flags: list[bool] = []
    for i, row_i in df.iterrows():
        size_i = float(row_i[size_col])
        quality_i = float(row_i[quality_col])
        if not math.isfinite(size_i) or not math.isfinite(quality_i):
            flags.append(False)
            continue

        dominated = False
        for j, row_j in df.iterrows():
            if i == j:
                continue
            size_j = float(row_j[size_col])
            quality_j = float(row_j[quality_col])
            if not math.isfinite(size_j) or not math.isfinite(quality_j):
                continue
            no_worse = size_j <= size_i and quality_j >= quality_i
            strictly_better = size_j < size_i or quality_j > quality_i
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)

    result = df.copy()
    result[output_col] = flags
    return result


def summarize_replicates(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str] = ("method", "backend", "device"),
) -> pd.DataFrame:
    """Aggregate independent experiment seeds without mixing them with timing samples.

    Numeric result columns are reported as their across-seed mean under the
    original column name plus ``<metric>_across_seed_std``. The original names
    keep plotting/backward compatibility while the explicit suffix makes the
    source of variability unambiguous.
    """
    group_cols = tuple(col for col in group_cols if col in df.columns)
    if not group_cols:
        raise ValueError("No replicate grouping columns are present")
    if "seed" not in df.columns:
        raise ValueError("Replicate results must contain a seed column")

    excluded_numeric = {"seed", "pareto_size_accuracy"}
    numeric_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in excluded_numeric
    ]

    records: list[dict[str, object]] = []
    for keys, group in df.groupby(list(group_cols), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record: dict[str, object] = dict(zip(group_cols, keys))
        seeds = sorted({int(value) for value in group["seed"].tolist()})
        record["replicate_count"] = len(seeds)
        record["seeds"] = ",".join(str(seed) for seed in seeds)

        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            record[col] = float(values.mean())
            record[f"{col}_across_seed_std"] = float(values.std(ddof=0))
        records.append(record)

    summary = pd.DataFrame(records)
    if {"model_size_mb", "accuracy"}.issubset(summary.columns):
        summary = add_pareto_flags(summary)
    return summary
