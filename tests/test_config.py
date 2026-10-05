from llm_compression_benchmark.config import DEFAULTS, validate_config


def test_defaults_validate():
    cfg = {
        **DEFAULTS,
        "model": {"teacher": "teacher", "student": "student", "trust_remote_code": False},
    }
    validate_config(cfg)


def test_default_dataset_id_is_namespaced():
    assert DEFAULTS["data"]["dataset_name"] == "Salesforce/wikitext"


def test_pruning_sweep_validation():
    cfg = {
        **DEFAULTS,
        "model": {"teacher": "teacher", "student": "student", "trust_remote_code": False},
        "pruning": {"amount": 0.1, "sweep_amounts": [0.05, 0.1, 0.2, 0.3]},
    }
    validate_config(cfg)
