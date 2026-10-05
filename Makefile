.PHONY: install test lint quick run gpt2 suite plot

install:
	pip install -e ".[quant,dev]"

test:
	pytest -q

lint:
	ruff check src tests

quick:
	llm-compression-benchmark run --config configs/quick.yaml

run:
	llm-compression-benchmark run --config configs/pythia.yaml

gpt2:
	llm-compression-benchmark run --config configs/gpt2.yaml

suite:
	llm-compression-benchmark aggregate --results outputs/pythia-160m/results.csv outputs/gpt2/results.csv --output outputs/suite/results.csv --plot-dir outputs/suite/plots

plot:
	llm-compression-benchmark plot --results outputs/pythia-160m/results.csv --output-dir outputs/pythia-160m/plots
