"""Tests for crasis.train — distillation pipeline."""

import json
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crasis.spec import CrasisSpec
from crasis.train import (
    TrainResult,
    _check_quality_gate,
    _detect_device,
    _load_jsonl,
    _make_compute_metrics,
    _save_train_result,
    _select_base_model,
    train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BINARY_SPEC = CrasisSpec.model_validate(
    {
        "name": "refund-detector",
        "description": "Detects refund requests.",
        "task": {
            "type": "binary_classification",
            "trigger": "customer asks for refund",
            "ignore": "general complaints",
        },
        "quality": {"min_accuracy": 0.90, "min_f1": 0.88},
        "training": {"volume": 100},
    }
)

MULTICLASS_SPEC = CrasisSpec.model_validate(
    {
        "name": "support-router",
        "description": "Routes support tickets.",
        "task": {
            "type": "multiclass",
            "trigger": "any support message",
            "ignore": "spam",
            "classes": ["billing", "technical", "returns"],
        },
        "quality": {"min_accuracy": 0.85},
        "training": {"volume": 200},
    }
)


def _make_jsonl(tmp_path: Path, spec: CrasisSpec, n: int = 100) -> Path:
    """Write a minimal JSONL dataset for the given spec."""
    out = tmp_path / f"{spec.name}_train.jsonl"
    labels = spec.label_names
    lines = []
    for i in range(n):
        label = labels[i % len(labels)]
        lines.append(
            json.dumps(
                {
                    "text": f"example number {i} for testing purposes",
                    "label": label,
                    "label_id": labels.index(label),
                }
            )
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _make_train_result(model_path: Path, passed: bool = True) -> TrainResult:
    return TrainResult(
        model_path=model_path,
        base_model="prajjwal1/bert-tiny",
        eval_accuracy=0.95 if passed else 0.70,
        eval_f1=0.94 if passed else 0.65,
        training_duration_s=42.0,
        num_train_samples=90,
        num_eval_samples=10,
        passed_quality_gate=passed,
    )


# ---------------------------------------------------------------------------
# _select_base_model
# ---------------------------------------------------------------------------


def test_select_tiny_for_tight_budget():
    assert _select_base_model(17) == "google/bert_uncased_L-2_H-128_A-2"


def test_select_tiny_when_budget_between_tiny_and_mini():
    # 19MB — bert-mini is 43MB, too large; bert-tiny (17MB) fits
    assert _select_base_model(19) == "google/bert_uncased_L-2_H-128_A-2"


def test_select_tiny_for_27mb_budget():
    # 27MB — bert-mini is 43MB, too large; bert-tiny (17MB) fits
    assert _select_base_model(27) == "google/bert_uncased_L-2_H-128_A-2"


def test_select_mini_for_medium_budget():
    assert _select_base_model(43) == "google/bert_uncased_L-4_H-256_A-4"


def test_select_small_for_larger_budget():
    assert _select_base_model(110) == "google/bert_uncased_L-4_H-512_A-8"


def test_select_medium_for_large_budget():
    assert _select_base_model(158) == "google/bert_uncased_L-8_H-512_A-8"


def test_select_picks_largest_fitting_model():
    # 50MB budget — bert-mini (43MB) fits, bert-small (110MB) does not
    assert _select_base_model(50) == "google/bert_uncased_L-4_H-256_A-4"


# ---------------------------------------------------------------------------
# _detect_device
# ---------------------------------------------------------------------------


def test_detect_device_returns_string():
    device = _detect_device()
    assert device in ("cpu", "cuda", "mps")


def test_detect_cuda_when_available():
    with patch("crasis.train.torch.cuda.is_available", return_value=True):
        with patch("crasis.train.torch.cuda.get_device_name", return_value="RTX 4060"):
            assert _detect_device() == "cuda"


def test_detect_cpu_when_no_gpu():
    with patch("crasis.train.torch.cuda.is_available", return_value=False):
        with patch("crasis.train.torch.backends.mps.is_available", return_value=False):
            assert _detect_device() == "cpu"


# ---------------------------------------------------------------------------
# _load_jsonl
# ---------------------------------------------------------------------------


def test_load_jsonl_returns_list(tmp_path):
    path = _make_jsonl(tmp_path, BINARY_SPEC, 20)
    examples = _load_jsonl(path)
    assert isinstance(examples, list)
    assert len(examples) == 20


def test_load_jsonl_each_example_has_required_keys(tmp_path):
    path = _make_jsonl(tmp_path, BINARY_SPEC, 10)
    examples = _load_jsonl(path)
    for ex in examples:
        assert "text" in ex
        assert "label" in ex
        assert "label_id" in ex


def test_load_jsonl_shuffles(tmp_path):
    """Shuffled order should differ from original (probabilistic — uses fixed seed)."""
    path = _make_jsonl(tmp_path, BINARY_SPEC, 100)
    original_texts = [json.loads(l)["text"] for l in path.read_text().splitlines() if l.strip()]
    random.seed(0)
    loaded = _load_jsonl(path)
    loaded_texts = [ex["text"] for ex in loaded]
    # With 100 items, probability of identical order after shuffle is negligible
    assert original_texts != loaded_texts


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "train.jsonl"
    lines = [
        json.dumps({"text": "a", "label": "positive", "label_id": 1}),
        "",
        json.dumps({"text": "b", "label": "negative", "label_id": 0}),
        "   ",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    assert len(_load_jsonl(path)) == 2


# ---------------------------------------------------------------------------
# _check_quality_gate
# ---------------------------------------------------------------------------


def test_quality_gate_passes_when_above_thresholds():
    assert _check_quality_gate(BINARY_SPEC, accuracy=0.95, f1=0.94) is True


def test_quality_gate_fails_on_low_accuracy():
    assert _check_quality_gate(BINARY_SPEC, accuracy=0.80, f1=0.94) is False


def test_quality_gate_fails_on_low_f1():
    assert _check_quality_gate(BINARY_SPEC, accuracy=0.95, f1=0.70) is False


def test_quality_gate_fails_when_both_low():
    assert _check_quality_gate(BINARY_SPEC, accuracy=0.80, f1=0.70) is False


def test_quality_gate_passes_without_min_f1():
    # MULTICLASS_SPEC has no min_f1
    assert _check_quality_gate(MULTICLASS_SPEC, accuracy=0.90, f1=0.0) is True


def test_quality_gate_fails_at_exact_threshold():
    # min_accuracy=0.90 — value must be >= threshold, not strictly >
    assert _check_quality_gate(BINARY_SPEC, accuracy=0.90, f1=0.94) is True


def test_quality_gate_fails_just_below_threshold():
    assert _check_quality_gate(BINARY_SPEC, accuracy=0.8999, f1=0.94) is False


# ---------------------------------------------------------------------------
# _make_compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_accuracy_and_f1():
    import numpy as np

    compute = _make_compute_metrics(BINARY_SPEC)
    logits = np.array([[2.0, 0.5], [0.1, 3.0], [1.5, 0.2]])
    labels = np.array([0, 1, 0])
    result = compute((logits, labels))
    assert "accuracy" in result
    assert "f1" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0


def test_compute_metrics_perfect_predictions():
    import numpy as np

    compute = _make_compute_metrics(BINARY_SPEC)
    logits = np.array([[3.0, 0.0], [0.0, 3.0]])
    labels = np.array([0, 1])
    result = compute((logits, labels))
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0


# ---------------------------------------------------------------------------
# _save_train_result
# ---------------------------------------------------------------------------


def test_save_train_result_creates_json(tmp_path):
    result = _make_train_result(tmp_path)
    _save_train_result(result, tmp_path)
    meta_path = tmp_path / "train_result.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["eval_accuracy"] == result.eval_accuracy
    assert meta["eval_f1"] == result.eval_f1
    assert meta["base_model"] == result.base_model
    assert meta["passed_quality_gate"] == result.passed_quality_gate
    assert meta["num_train_samples"] == result.num_train_samples
    assert meta["num_eval_samples"] == result.num_eval_samples
    assert meta["training_duration_s"] == result.training_duration_s


# ---------------------------------------------------------------------------
# train() — full pipeline (mocked Trainer + model downloads)
# ---------------------------------------------------------------------------


def _make_mock_trainer(accuracy: float = 0.95, f1: float = 0.94):
    trainer = MagicMock()
    trainer.evaluate.return_value = {"eval_accuracy": accuracy, "eval_f1": f1}
    return trainer


def _patch_train_deps(accuracy=0.95, f1=0.94):
    """Patch all heavy dependencies so train() runs without GPU or downloads."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    mock_tokenizer.save_pretrained = MagicMock()

    mock_model = MagicMock()
    mock_trainer = _make_mock_trainer(accuracy, f1)
    mock_training_args = MagicMock()

    patches = [
        patch("crasis.train.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch(
            "crasis.train.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
        patch("crasis.train.TrainingArguments", return_value=mock_training_args),
        patch("crasis.train.Trainer", return_value=mock_trainer),
        patch("crasis.train._tokenize", side_effect=lambda examples, tok: MagicMock()),
    ]
    return patches


def _apply_patches(patches):
    """Enter a list of patch context managers and return them."""
    entered = [p.__enter__() for p in patches]
    return patches, entered


def _exit_patches(patches):
    for p in patches:
        p.__exit__(None, None, None)


@pytest.fixture()
def mock_train_env():
    """Fixture that patches all heavy train() dependencies."""
    patches = _patch_train_deps()
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


@pytest.fixture()
def mock_train_env_failing():
    """Fixture where eval metrics fall below quality thresholds."""
    patches = _patch_train_deps(accuracy=0.70, f1=0.60)
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def test_train_raises_if_data_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        train(BINARY_SPEC, tmp_path / "nonexistent.jsonl", tmp_path)


def test_train_returns_train_result(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    assert isinstance(result, TrainResult)


def test_train_result_model_path_exists(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    assert result.model_path.exists()


def test_train_result_under_spec_name_subdir(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    assert result.model_path.name == BINARY_SPEC.name


def test_train_selects_correct_base_model(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    assert result.base_model == _select_base_model(BINARY_SPEC.constraints.max_model_size_mb)


def test_train_records_sample_counts(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 100)
    result = train(BINARY_SPEC, data, tmp_path)
    assert result.num_train_samples + result.num_eval_samples == 100


def test_train_90_10_split(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 100)
    result = train(BINARY_SPEC, data, tmp_path)
    assert result.num_train_samples == 90
    assert result.num_eval_samples == 10


def test_train_passes_quality_gate_when_metrics_good(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    assert result.passed_quality_gate is True


def test_train_fails_quality_gate_when_metrics_poor(tmp_path, mock_train_env_failing):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    assert result.passed_quality_gate is False


def test_train_writes_label_map_json(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    label_map_path = result.model_path / "label_map.json"
    assert label_map_path.exists()
    label_map = json.loads(label_map_path.read_text())
    assert "label2id" in label_map
    assert "id2label" in label_map


def test_train_label_map_correct_for_binary(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    label_map = json.loads((result.model_path / "label_map.json").read_text())
    assert label_map["label2id"]["negative"] == 0
    assert label_map["label2id"]["positive"] == 1


def test_train_label_map_correct_for_multiclass(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, MULTICLASS_SPEC, 60)
    result = train(MULTICLASS_SPEC, data, tmp_path)
    label_map = json.loads((result.model_path / "label_map.json").read_text())
    assert set(label_map["label2id"].keys()) == {"billing", "technical", "returns"}


def test_train_writes_train_result_json(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    assert (result.model_path / "train_result.json").exists()


def test_train_records_duration(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    result = train(BINARY_SPEC, data, tmp_path)
    assert result.training_duration_s >= 0.0


def test_train_device_override(tmp_path, mock_train_env):
    data = _make_jsonl(tmp_path, BINARY_SPEC, 50)
    # Should not raise even with an explicit device override
    result = train(BINARY_SPEC, data, tmp_path, device="cpu")
    assert isinstance(result, TrainResult)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def test_stratified_split_preserves_label_balance(tmp_path, mock_train_env):
    """
    Stratified 90/10 split must preserve label proportions within 2pp in each set.

    With a perfectly balanced 200-example dataset (100 positive, 100 negative),
    a stratified split should yield ~90 train and ~10 eval per class — i.e.,
    each split stays within 2pp of the overall 50/50 distribution.
    """
    # Write 200 examples with exactly 50/50 balance
    out = tmp_path / "train.jsonl"
    labels = BINARY_SPEC.label_names  # ["negative", "positive"]
    lines = []
    for i in range(200):
        label = labels[i % 2]  # strict alternation -> exactly 50/50
        lines.append(
            json.dumps({"text": f"example {i}", "label": label, "label_id": labels.index(label)})
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Patch _tokenize to capture what split was actually passed to the Trainer
    captured: dict = {}
    original_tokenize = __import__("crasis.train", fromlist=["_tokenize"])._tokenize

    def capturing_tokenize(examples, tokenizer):
        label_list = [ex["label"] for ex in examples]
        # record by call order: first call = train, second = eval
        key = "train_labels" if "train_labels" not in captured else "eval_labels"
        captured[key] = label_list
        return MagicMock()

    with patch("crasis.train._tokenize", side_effect=capturing_tokenize):
        train(BINARY_SPEC, out, tmp_path)

    assert "train_labels" in captured, "train split was not captured"
    assert "eval_labels" in captured, "eval split was not captured"

    def label_ratio(labels: list[str], target: str) -> float:
        return labels.count(target) / len(labels)

    for label in BINARY_SPEC.label_names:
        train_ratio = label_ratio(captured["train_labels"], label)
        eval_ratio = label_ratio(captured["eval_labels"], label)
        assert abs(train_ratio - 0.5) < 0.02, (
            f"Train label '{label}' ratio {train_ratio:.3f} is more than 2pp from 0.5"
        )
        assert abs(eval_ratio - 0.5) < 0.02, (
            f"Eval label '{label}' ratio {eval_ratio:.3f} is more than 2pp from 0.5"
        )
