import pandas as pd
import pytest

from llm_compression_benchmark.analysis import add_pareto_flags, summarize_replicates


def test_pareto_flags_size_accuracy():
    df = pd.DataFrame(
        [
            {"method": "baseline", "model_size_mb": 100.0, "accuracy": 0.90},
            {"method": "small", "model_size_mb": 50.0, "accuracy": 0.88},
            {"method": "dominated", "model_size_mb": 100.0, "accuracy": 0.80},
        ]
    )
    out = add_pareto_flags(df)
    flags = dict(zip(out["method"], out["pareto_size_accuracy"]))
    assert flags == {"baseline": True, "small": True, "dominated": False}


def test_summarize_replicates_separates_across_seed_std():
    df = pd.DataFrame(
        [
            {
                "method": "baseline",
                "backend": "fp",
                "device": "cuda",
                "seed": 42,
                "model_size_mb": 100.0,
                "accuracy": 0.90,
                "latency_ms": 10.0,
            },
            {
                "method": "baseline",
                "backend": "fp",
                "device": "cuda",
                "seed": 43,
                "model_size_mb": 100.0,
                "accuracy": 0.88,
                "latency_ms": 12.0,
            },
        ]
    )
    out = summarize_replicates(df)
    row = out.iloc[0]
    assert row["replicate_count"] == 2
    assert row["accuracy"] == pytest.approx(0.89)
    assert row["accuracy_across_seed_std"] == pytest.approx(0.01)
    assert row["latency_ms"] == pytest.approx(11.0)
    assert row["latency_ms_across_seed_std"] == pytest.approx(1.0)
