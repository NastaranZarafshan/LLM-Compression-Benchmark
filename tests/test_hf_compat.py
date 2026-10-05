from llm_compression_benchmark.hf_compat import canonical_dataset_id, canonical_model_id


def test_wikitext_legacy_id_is_namespaced():
    assert canonical_dataset_id("wikitext") == "Salesforce/wikitext"
    assert canonical_dataset_id("Salesforce/wikitext") == "Salesforce/wikitext"


def test_gpt2_legacy_id_is_namespaced():
    assert canonical_model_id("gpt2") == "openai-community/gpt2"
    assert canonical_model_id("EleutherAI/pythia-70m") == "EleutherAI/pythia-70m"
