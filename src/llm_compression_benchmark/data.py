from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from .batching import collate_fixed_blocks
from .hf_compat import canonical_dataset_id


def prepare_lm_dataset(
    tokenizer: PreTrainedTokenizerBase,
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    block_size: int,
    max_samples: int | None,
) -> Dataset:
    resolved_dataset_name = canonical_dataset_id(dataset_name)
    ds = load_dataset(resolved_dataset_name, dataset_config, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    if text_column not in ds.column_names:
        raise ValueError(f"text column {text_column!r} not found. Columns: {ds.column_names}")

    remove_columns = list(ds.column_names)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [x if isinstance(x, str) else str(x) for x in batch[text_column]]
        return tokenizer(texts, add_special_tokens=False)

    tokenized = ds.map(tokenize, batched=True, remove_columns=remove_columns, desc=f"Tokenizing {split}")

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {k: [] for k in concatenated}
        result = {
            k: [values[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, values in concatenated.items()
        }
        result["labels"] = [x.copy() for x in result["input_ids"]]
        return result

    grouped = tokenized.map(group_texts, batched=True, desc=f"Grouping {split}")
    if len(grouped) == 0:
        raise ValueError("Dataset became empty after token grouping; lower block_size or increase samples")
    return grouped
