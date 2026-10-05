"""Regression test for the public evaluation API imported by benchmark.py."""


def test_evaluate_exports_expected_api():
    # Import inside the test so a missing/overwritten function fails exactly the
    # way the CLI import failed in the reported regression.
    from llm_compression_benchmark import evaluate

    assert callable(evaluate.evaluate_quality)
    assert callable(evaluate.benchmark_generation)
    assert callable(evaluate.collect_model_metrics)
