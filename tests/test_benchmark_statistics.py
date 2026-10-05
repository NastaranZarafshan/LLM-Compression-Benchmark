import pytest

from llm_compression_benchmark.evaluate import _percentile


def test_percentile_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 4.0
    assert _percentile(values, 0.5) == 2.5
    assert _percentile(values, 0.95) == pytest.approx(3.85)
