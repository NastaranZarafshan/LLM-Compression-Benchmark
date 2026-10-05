from __future__ import annotations


# Newer huggingface_hub releases require canonical namespace/name repository IDs
# when datasets resolves Hub files through hf:// URIs. Keep aliases here so old
# experiment configs remain reproducible instead of failing before data loading.
LEGACY_DATASET_ALIASES: dict[str, str] = {
    "wikitext": "Salesforce/wikitext",
}

LEGACY_MODEL_ALIASES: dict[str, str] = {
    "gpt2": "openai-community/gpt2",
}


def canonical_dataset_id(dataset_id: str) -> str:
    """Return a current namespaced Hub dataset ID for known legacy aliases."""
    value = str(dataset_id).strip()
    return LEGACY_DATASET_ALIASES.get(value, value)


def canonical_model_id(model_id: str) -> str:
    """Return a current namespaced Hub model ID for known legacy aliases."""
    value = str(model_id).strip()
    return LEGACY_MODEL_ALIASES.get(value, value)
