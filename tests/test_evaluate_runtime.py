from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from llm_compression_benchmark.evaluate import (
    benchmark_generation,
    collect_model_metrics,
    evaluate_quality,
)


class ToyLM(nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.vocab_size), -20.0, device=input_ids.device)
        # At position t, predict token (input[t] + 1) % vocab for the next target.
        pred = (input_ids + 1) % self.vocab_size
        logits.scatter_(2, pred.unsqueeze(-1), 20.0)
        logits = logits + self.anchor * 0.0
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.vocab_size),
                labels[:, 1:].reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
        out = input_ids
        for _ in range(max_new_tokens):
            nxt = ((out[:, -1] + 1) % self.vocab_size).unsqueeze(-1)
            out = torch.cat([out, nxt], dim=1)
        return out


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def __call__(self, texts, return_tensors=None, padding=True):
        rows = [[1, 2, 3] for _ in texts]
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def test_quality_generation_and_model_metrics_smoke():
    model = ToyLM()
    dataset = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": [1, 2, 3, 4]},
        {"input_ids": [2, 3, 4, 5], "attention_mask": [1, 1, 1, 1], "labels": [2, 3, 4, 5]},
    ]

    quality = evaluate_quality(model, dataset, batch_size=2, max_batches=1)
    assert quality.token_accuracy == 1.0
    assert quality.eval_tokens == 6
    assert quality.perplexity >= 1.0

    runtime = benchmark_generation(
        model,
        ToyTokenizer(),
        prompt="toy",
        batch_size=2,
        max_new_tokens=3,
        warmup_runs=1,
        timed_runs=2,
    )
    assert runtime.latency_ms >= 0.0
    assert runtime.latency_mean_ms >= 0.0
    assert runtime.latency_std_ms >= 0.0
    assert runtime.latency_p95_ms >= runtime.latency_min_ms
    assert runtime.latency_max_ms >= runtime.latency_p95_ms
    assert len(runtime.latency_samples_ms) == 2
    assert runtime.timed_runs == 2
    assert runtime.throughput_tok_s > 0.0
    assert runtime.generated_tokens == 12
    assert runtime.vram_peak_mb == 0.0
    assert runtime.vram_resident_mb == 0.0
    assert runtime.vram_generation_delta_mb == 0.0

    metrics = collect_model_metrics(model)
    assert metrics["model_size_mb"] > 0.0
    assert metrics["parameters_m"] > 0.0
    assert metrics["sparsity_pct"] == 100.0  # the only toy parameter is zero
