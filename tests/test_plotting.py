import pandas as pd

from llm_compression_benchmark.plotting import create_plots


def test_create_plots(tmp_path):
    csv = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "method": "baseline",
                "model_size_mb": 100,
                "accuracy": 0.2,
                "token_accuracy": 0.2,
                "perplexity": 20,
                "latency_ms": 10,
                "throughput_tok_s": 100,
            },
            {
                "method": "quantization",
                "model_size_mb": 30,
                "accuracy": 0.19,
                "token_accuracy": 0.19,
                "perplexity": 21,
                "latency_ms": 7,
                "throughput_tok_s": 140,
            },
        ]
    ).to_csv(csv, index=False)
    outputs = create_plots(csv, tmp_path / "plots")
    assert len(outputs) == 4
    assert all(p.exists() for p in outputs)
