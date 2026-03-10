"""Tests for crasis.spec — CrasisSpec, BuildRequest, BuildConfig."""

import pytest
from pydantic import ValidationError

from crasis.spec import (
    BuildConfig,
    BuildRequest,
    CrasisSpec,
    TaskType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BINARY_SPEC_DATA = {
    "name": "refund-detector",
    "description": "Detects refund requests in customer support emails.",
    "task": {
        "type": "binary_classification",
        "trigger": "customer explicitly asks for a refund or money back",
        "ignore": "general complaints without a refund request",
    },
    "quality": {"min_accuracy": 0.93},
    "training": {"volume": 3000},
}

MULTICLASS_SPEC_DATA = {
    "name": "support-router",
    "description": "Routes support tickets to the correct team.",
    "task": {
        "type": "multiclass",
        "trigger": "any customer support message",
        "ignore": "spam",
        "classes": ["billing", "technical", "returns", "general"],
    },
    "quality": {"min_accuracy": 0.88, "min_f1": 0.85},
    "training": {"volume": 5000},
}


# ---------------------------------------------------------------------------
# CrasisSpec — basic construction
# ---------------------------------------------------------------------------


def test_binary_spec_builds():
    spec = CrasisSpec.model_validate(BINARY_SPEC_DATA)
    assert spec.name == "refund-detector"
    assert spec.task.type == TaskType.binary_classification


def test_multiclass_spec_builds():
    spec = CrasisSpec.model_validate(MULTICLASS_SPEC_DATA)
    assert spec.task.type == TaskType.multiclass
    assert spec.task.classes == ["billing", "technical", "returns", "general"]


def test_defaults_applied():
    spec = CrasisSpec.model_validate(BINARY_SPEC_DATA)
    assert spec.constraints.max_model_size_mb == 27
    assert spec.constraints.max_inference_ms == 100
    assert spec.constraints.connectivity == "none"
    assert spec.constraints.target_hardware == "cpu_only"
    assert spec.training.augmentation is True


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["refund-detector", "spam-filter", "a1b2", "x"])
def test_valid_kebab_names(name):
    data = {**BINARY_SPEC_DATA, "name": name}
    spec = CrasisSpec.model_validate(data)
    assert spec.name == name


@pytest.mark.parametrize("name", ["Refund", "refund_detector", "refund detector", "-start", "end-", ""])
def test_invalid_names_rejected(name):
    data = {**BINARY_SPEC_DATA, "name": name}
    with pytest.raises(ValidationError):
        CrasisSpec.model_validate(data)


# ---------------------------------------------------------------------------
# Multiclass — classes required
# ---------------------------------------------------------------------------


def test_multiclass_requires_classes():
    data = {
        **MULTICLASS_SPEC_DATA,
        "task": {
            "type": "multiclass",
            "trigger": "any message",
            "ignore": "spam",
            # classes omitted
        },
    }
    with pytest.raises(ValidationError):
        CrasisSpec.model_validate(data)


# ---------------------------------------------------------------------------
# label_names property
# ---------------------------------------------------------------------------


def test_binary_label_names():
    spec = CrasisSpec.model_validate(BINARY_SPEC_DATA)
    assert spec.label_names == ["negative", "positive"]


def test_multiclass_label_names():
    spec = CrasisSpec.model_validate(MULTICLASS_SPEC_DATA)
    assert spec.label_names == ["billing", "technical", "returns", "general"]


def test_num_labels_binary():
    spec = CrasisSpec.model_validate(BINARY_SPEC_DATA)
    assert spec.num_labels == 2


def test_num_labels_multiclass():
    spec = CrasisSpec.model_validate(MULTICLASS_SPEC_DATA)
    assert spec.num_labels == 4


# ---------------------------------------------------------------------------
# spec_hash
# ---------------------------------------------------------------------------


def test_spec_hash_length():
    spec = CrasisSpec.model_validate(BINARY_SPEC_DATA)
    assert len(spec.spec_hash()) == 16


def test_spec_hash_is_hex():
    spec = CrasisSpec.model_validate(BINARY_SPEC_DATA)
    int(spec.spec_hash(), 16)  # raises ValueError if not valid hex


def test_spec_hash_deterministic():
    spec = CrasisSpec.model_validate(BINARY_SPEC_DATA)
    assert spec.spec_hash() == spec.spec_hash()


def test_spec_hash_changes_on_mutation():
    spec_a = CrasisSpec.model_validate(BINARY_SPEC_DATA)
    data_b = {**BINARY_SPEC_DATA, "training": {"volume": 9999}}
    spec_b = CrasisSpec.model_validate(data_b)
    assert spec_a.spec_hash() != spec_b.spec_hash()


# ---------------------------------------------------------------------------
# BuildRequest
# ---------------------------------------------------------------------------


def test_build_request_wraps_spec():
    req = BuildRequest.model_validate({"spec": BINARY_SPEC_DATA})
    assert req.spec.name == "refund-detector"
    assert req.build.dry_run is False


def test_build_request_from_json():
    req = BuildRequest.from_json({"spec": BINARY_SPEC_DATA})
    assert isinstance(req.spec, CrasisSpec)


def test_build_request_from_yaml(tmp_path):
    import yaml

    # Bare spec YAML (no 'spec' wrapper)
    yaml_file = tmp_path / "spec.yaml"
    yaml_file.write_text(yaml.dump(BINARY_SPEC_DATA), encoding="utf-8")
    req = BuildRequest.from_yaml(yaml_file)
    assert req.spec.name == "refund-detector"


def test_build_request_from_yaml_with_wrapper(tmp_path):
    import yaml

    # Full BuildRequest YAML with 'spec' key
    data = {"spec": BINARY_SPEC_DATA, "build": {"dry_run": True}}
    yaml_file = tmp_path / "build.yaml"
    yaml_file.write_text(yaml.dump(data), encoding="utf-8")
    req = BuildRequest.from_yaml(yaml_file)
    assert req.build.dry_run is True


# ---------------------------------------------------------------------------
# BuildConfig — no api_key field
# ---------------------------------------------------------------------------


def test_build_config_has_no_api_key_field():
    """BuildConfig must never expose an api_key field (platform injects it server-side)."""
    assert not hasattr(BuildConfig(), "api_key")
    assert "api_key" not in BuildConfig.model_fields


# ---------------------------------------------------------------------------
# CrasisSpec.from_yaml convenience
# ---------------------------------------------------------------------------


def test_crasis_spec_from_yaml(tmp_path):
    import yaml

    yaml_file = tmp_path / "spec.yaml"
    yaml_file.write_text(yaml.dump(BINARY_SPEC_DATA), encoding="utf-8")
    spec = CrasisSpec.from_yaml(yaml_file)
    assert spec.name == "refund-detector"
