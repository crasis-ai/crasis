"""
Tests for crasis.export — ONNX export and packaging.

Named test_eval.py to match CLAUDE.md's test file listing.
The export pipeline is what enforces the quality + size gates before a
specialist can be deployed, so these tests are critical.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crasis.export import (
    ExportResult,
    _benchmark_latency_dummy,
    _copy_label_map,
    _copy_tokenizer,
    _write_meta,
    export,
)
from crasis.spec import CrasisSpec


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
        "constraints": {"max_model_size_mb": 27, "max_inference_ms": 100},
        "quality": {"min_accuracy": 0.93},
        "training": {"volume": 100},
    }
)


def _make_model_dir(tmp_path: Path, spec: CrasisSpec) -> Path:
    """Create a minimal fake trained model directory."""
    model_dir = tmp_path / spec.name
    model_dir.mkdir()

    # label_map.json
    label2id = {name: i for i, name in enumerate(spec.label_names)}
    id2label = {str(i): name for name, i in label2id.items()}
    (model_dir / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}), encoding="utf-8"
    )

    # Minimal tokenizer files
    (model_dir / "tokenizer_config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model_dir / "vocab.txt").write_text("dummy vocab\n", encoding="utf-8")

    return model_dir


def _make_fake_onnx(path: Path, size_bytes: int = 1024 * 1024 * 10) -> None:
    """Write a fake ONNX file of the given size."""
    path.write_bytes(b"\x00" * size_bytes)


# ---------------------------------------------------------------------------
# _write_meta
# ---------------------------------------------------------------------------


def test_write_meta_creates_file(tmp_path):
    _write_meta(BINARY_SPEC, tmp_path, size_mb=18.5, latency_ms=42.0)
    assert (tmp_path / "crasis_meta.json").exists()


def test_write_meta_content(tmp_path):
    _write_meta(BINARY_SPEC, tmp_path, size_mb=18.5, latency_ms=42.0)
    meta = json.loads((tmp_path / "crasis_meta.json").read_text())

    assert meta["name"] == BINARY_SPEC.name
    assert meta["description"] == BINARY_SPEC.description
    assert meta["task_type"] == "binary_classification"
    assert meta["label_names"] == ["negative", "positive"]
    assert meta["num_labels"] == 2
    assert meta["model_size_mb"] == 18.5
    assert meta["benchmark_latency_ms"] == 42.0
    assert meta["max_model_size_mb"] == 27
    assert meta["max_inference_ms"] == 100


def test_write_meta_within_constraints_true(tmp_path):
    _write_meta(BINARY_SPEC, tmp_path, size_mb=20.0, latency_ms=80.0)
    meta = json.loads((tmp_path / "crasis_meta.json").read_text())
    assert meta["within_size_constraint"] is True
    assert meta["within_latency_constraint"] is True


def test_write_meta_within_constraints_false(tmp_path):
    _write_meta(BINARY_SPEC, tmp_path, size_mb=50.0, latency_ms=200.0)
    meta = json.loads((tmp_path / "crasis_meta.json").read_text())
    assert meta["within_size_constraint"] is False
    assert meta["within_latency_constraint"] is False


def test_write_meta_latency_none(tmp_path):
    _write_meta(BINARY_SPEC, tmp_path, size_mb=18.5, latency_ms=None)
    meta = json.loads((tmp_path / "crasis_meta.json").read_text())
    assert meta["benchmark_latency_ms"] is None
    assert meta["within_latency_constraint"] is None


def test_write_meta_includes_spec_hash(tmp_path):
    _write_meta(BINARY_SPEC, tmp_path, size_mb=18.5, latency_ms=42.0)
    meta = json.loads((tmp_path / "crasis_meta.json").read_text())
    assert meta["spec_hash"] == BINARY_SPEC.spec_hash()
    assert len(meta["spec_hash"]) == 16


# ---------------------------------------------------------------------------
# _copy_tokenizer
# ---------------------------------------------------------------------------


def test_copy_tokenizer_creates_subdir(tmp_path):
    model_dir = _make_model_dir(tmp_path, BINARY_SPEC)
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    _copy_tokenizer(model_dir, pkg_dir)
    assert (pkg_dir / "tokenizer").is_dir()


def test_copy_tokenizer_copies_present_files(tmp_path):
    model_dir = _make_model_dir(tmp_path, BINARY_SPEC)
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    _copy_tokenizer(model_dir, pkg_dir)
    assert (pkg_dir / "tokenizer" / "tokenizer_config.json").exists()
    assert (pkg_dir / "tokenizer" / "vocab.txt").exists()


def test_copy_tokenizer_skips_missing_files(tmp_path):
    model_dir = _make_model_dir(tmp_path, BINARY_SPEC)
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    _copy_tokenizer(model_dir, pkg_dir)
    # merges.txt is not in our fake model dir — should not appear
    assert not (pkg_dir / "tokenizer" / "merges.txt").exists()


def test_copy_tokenizer_uses_tokenizer_subdir_if_present(tmp_path):
    model_dir = _make_model_dir(tmp_path, BINARY_SPEC)
    tok_subdir = model_dir / "tokenizer"
    tok_subdir.mkdir()
    (tok_subdir / "vocab.txt").write_text("subdir vocab\n", encoding="utf-8")

    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    _copy_tokenizer(model_dir, pkg_dir)
    content = (pkg_dir / "tokenizer" / "vocab.txt").read_text()
    assert content == "subdir vocab\n"


# ---------------------------------------------------------------------------
# _copy_label_map
# ---------------------------------------------------------------------------


def test_copy_label_map_copies_file(tmp_path):
    model_dir = _make_model_dir(tmp_path, BINARY_SPEC)
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    _copy_label_map(model_dir, pkg_dir)
    assert (pkg_dir / "label_map.json").exists()


def test_copy_label_map_content_preserved(tmp_path):
    model_dir = _make_model_dir(tmp_path, BINARY_SPEC)
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    _copy_label_map(model_dir, pkg_dir)
    data = json.loads((pkg_dir / "label_map.json").read_text())
    assert "label2id" in data
    assert "id2label" in data


def test_copy_label_map_missing_does_not_raise(tmp_path):
    model_dir = tmp_path / "empty_model"
    model_dir.mkdir()
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    # Should log warning but not raise
    _copy_label_map(model_dir, pkg_dir)


# ---------------------------------------------------------------------------
# ExportResult dataclass
# ---------------------------------------------------------------------------


def test_export_result_fields():
    r = ExportResult(
        onnx_path=Path("/tmp/model.onnx"),
        package_dir=Path("/tmp/pkg"),
        model_size_mb=18.5,
        within_size_constraint=True,
        benchmark_latency_ms=42.0,
    )
    assert r.model_size_mb == 18.5
    assert r.within_size_constraint is True
    assert r.benchmark_latency_ms == 42.0


def test_export_result_latency_optional():
    r = ExportResult(
        onnx_path=Path("/tmp/model.onnx"),
        package_dir=Path("/tmp/pkg"),
        model_size_mb=18.5,
        within_size_constraint=True,
    )
    assert r.benchmark_latency_ms is None


# ---------------------------------------------------------------------------
# export() — full pipeline (mocked ONNX conversion)
# ---------------------------------------------------------------------------


def _make_mock_ort_model(pkg_dir: Path, spec: CrasisSpec, size_bytes: int = 10 * 1024 * 1024):
    """Return a mock ORTModel whose save_pretrained writes a fake ONNX file."""

    def fake_save_pretrained(out_dir):
        onnx_out = Path(out_dir) / "model.onnx"
        onnx_out.write_bytes(b"\x00" * size_bytes)

    mock = MagicMock()
    mock.save_pretrained.side_effect = fake_save_pretrained
    return mock


@pytest.fixture()
def model_dir(tmp_path):
    return _make_model_dir(tmp_path, BINARY_SPEC)


class _ExportPatcher:
    """Context manager that patches both ONNX conversion and quantization."""

    def __init__(self, size_bytes):
        self._size_bytes = size_bytes
        self._patches = []

    def __enter__(self):
        def fake_quantize(onnx_path, pkg_dir, spec):
            return onnx_path

        size_bytes = self._size_bytes

        def fake_convert(model_path, pkg_dir, spec):
            onnx_path = pkg_dir / f"{spec.name}.onnx"
            onnx_path.write_bytes(b"\x00" * size_bytes)
            return onnx_path

        p1 = patch("crasis.export._convert_to_onnx", side_effect=fake_convert)
        p2 = patch("crasis.export._quantize", side_effect=fake_quantize)
        self._patches = [p1, p2]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *args):
        for p in self._patches:
            p.stop()


def _patch_export(model_dir, size_bytes=10 * 1024 * 1024):
    return _ExportPatcher(size_bytes)


def test_export_raises_if_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        export(BINARY_SPEC, tmp_path / "nonexistent", tmp_path)


def test_export_returns_export_result(tmp_path, model_dir):
    with _patch_export(model_dir):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert isinstance(result, ExportResult)


def test_export_onnx_path_has_spec_name(tmp_path, model_dir):
    with _patch_export(model_dir):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert result.onnx_path.stem == BINARY_SPEC.name
    assert result.onnx_path.suffix == ".onnx"


def test_export_package_dir_named_with_onnx_suffix(tmp_path, model_dir):
    with _patch_export(model_dir):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert result.package_dir.name == f"{BINARY_SPEC.name}-onnx"


def test_export_onnx_file_exists(tmp_path, model_dir):
    with _patch_export(model_dir):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert result.onnx_path.exists()


def test_export_within_size_constraint_true(tmp_path, model_dir):
    # 10MB is under the 27MB limit
    with _patch_export(model_dir, size_bytes=10 * 1024 * 1024):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert result.within_size_constraint is True


def test_export_within_size_constraint_false(tmp_path, model_dir):
    # 30MB exceeds the 27MB limit
    with _patch_export(model_dir, size_bytes=30 * 1024 * 1024):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert result.within_size_constraint is False


def test_export_model_size_mb_accurate(tmp_path, model_dir):
    size_bytes = 15 * 1024 * 1024
    with _patch_export(model_dir, size_bytes=size_bytes):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert abs(result.model_size_mb - 15.0) < 0.1


def test_export_writes_label_map(tmp_path, model_dir):
    with _patch_export(model_dir):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert (result.package_dir / "label_map.json").exists()


def test_export_writes_crasis_meta(tmp_path, model_dir):
    with _patch_export(model_dir):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert (result.package_dir / "crasis_meta.json").exists()


def test_export_crasis_meta_correct(tmp_path, model_dir):
    with _patch_export(model_dir):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    meta = json.loads((result.package_dir / "crasis_meta.json").read_text())
    assert meta["name"] == BINARY_SPEC.name
    assert meta["label_names"] == ["negative", "positive"]
    assert meta["spec_hash"] == BINARY_SPEC.spec_hash()


def test_export_no_benchmark_when_disabled(tmp_path, model_dir):
    with _patch_export(model_dir):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert result.benchmark_latency_ms is None


def test_export_tokenizer_dir_created(tmp_path, model_dir):
    with _patch_export(model_dir):
        result = export(BINARY_SPEC, model_dir, tmp_path, benchmark=False)
    assert (result.package_dir / "tokenizer").is_dir()


# ---------------------------------------------------------------------------
# _benchmark_latency_dummy — runs a real ONNX session on a tiny model
# ---------------------------------------------------------------------------


def test_benchmark_latency_dummy_returns_positive_ms(tmp_path):
    """Build a minimal real ONNX model and verify the benchmark returns a positive latency."""
    import onnx
    from onnx import TensorProto, helper

    # Build a trivial ONNX graph: identity op on [1, 128] int64 input
    X = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 128])
    Y = helper.make_tensor_value_info("output", TensorProto.INT64, [1, 128])
    node = helper.make_node("Identity", inputs=["input_ids"], outputs=["output"])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    onnx_path = tmp_path / "tiny.onnx"
    onnx.save(model, str(onnx_path))

    latency = _benchmark_latency_dummy(onnx_path, n_runs=10)
    assert latency > 0.0
    assert latency < 1000.0  # sanity: identity op should be <1s


# ---------------------------------------------------------------------------
# crasis.eval — eval_on_holdout
# ---------------------------------------------------------------------------


def _make_holdout_jsonl(tmp_path: Path, spec: CrasisSpec, n: int = 20) -> Path:
    """Write a minimal holdout JSONL for the given spec."""
    out = tmp_path / f"{spec.name}_holdout.jsonl"
    labels = spec.label_names
    lines = []
    for i in range(n):
        label = labels[i % len(labels)]
        lines.append(
            json.dumps(
                {
                    "text": f"holdout example {i}",
                    "label": label,
                    "label_id": labels.index(label),
                }
            )
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _make_mock_specialist(spec: CrasisSpec, correct: bool = True):
    """
    Return a mock Specialist that either always predicts correctly
    or always predicts label_id=0 (useful for testing gap detection).
    """
    from unittest.mock import MagicMock

    specialist = MagicMock()
    specialist.name = spec.name
    label_names = spec.label_names

    def mock_classify(text):
        # Extract label_id from the text ("holdout example {i}")
        i = int(text.split()[-1])
        true_label_id = i % len(label_names)
        pred_label_id = true_label_id if correct else 0
        return {
            "label": label_names[pred_label_id],
            "label_id": pred_label_id,
            "confidence": 0.95,
            "latency_ms": 1.0,
        }

    specialist.classify.side_effect = mock_classify
    return specialist


def test_eval_on_holdout_returns_holdout_result(tmp_path):
    from crasis.eval import eval_on_holdout, HoldoutResult

    holdout_path = _make_holdout_jsonl(tmp_path, BINARY_SPEC, n=20)
    specialist = _make_mock_specialist(BINARY_SPEC, correct=True)
    result = eval_on_holdout(specialist, holdout_path, BINARY_SPEC)
    assert isinstance(result, HoldoutResult)


def test_eval_on_holdout_perfect_predictions(tmp_path):
    from crasis.eval import eval_on_holdout

    holdout_path = _make_holdout_jsonl(tmp_path, BINARY_SPEC, n=20)
    specialist = _make_mock_specialist(BINARY_SPEC, correct=True)
    result = eval_on_holdout(specialist, holdout_path, BINARY_SPEC)
    assert result.accuracy == 1.0
    assert result.f1_macro == 1.0


def test_eval_on_holdout_num_samples(tmp_path):
    from crasis.eval import eval_on_holdout

    holdout_path = _make_holdout_jsonl(tmp_path, BINARY_SPEC, n=30)
    specialist = _make_mock_specialist(BINARY_SPEC, correct=True)
    result = eval_on_holdout(specialist, holdout_path, BINARY_SPEC)
    assert result.num_samples == 30


def test_eval_on_holdout_raises_if_file_missing(tmp_path):
    from crasis.eval import eval_on_holdout

    specialist = _make_mock_specialist(BINARY_SPEC, correct=True)
    with pytest.raises(FileNotFoundError):
        eval_on_holdout(specialist, tmp_path / "nonexistent.jsonl", BINARY_SPEC)


def test_eval_on_holdout_gap_not_flagged_when_close(tmp_path):
    """No gap flag when holdout accuracy is within 5pp of synthetic."""
    from crasis.eval import eval_on_holdout

    holdout_path = _make_holdout_jsonl(tmp_path, BINARY_SPEC, n=20)
    specialist = _make_mock_specialist(BINARY_SPEC, correct=True)
    # synthetic=1.0, holdout=1.0 → gap=0.0, no flag
    result = eval_on_holdout(specialist, holdout_path, BINARY_SPEC, synthetic_accuracy=1.0)
    assert result.gap_flagged is False
    assert result.synthetic_accuracy == 1.0


def test_eval_on_holdout_gap_flagged_when_large(tmp_path):
    """Gap flag fires when holdout accuracy drops >5pp below synthetic."""
    from crasis.eval import eval_on_holdout

    # All-zeros predictor on balanced data: accuracy=0.5
    holdout_path = _make_holdout_jsonl(tmp_path, BINARY_SPEC, n=20)
    specialist = _make_mock_specialist(BINARY_SPEC, correct=False)
    # synthetic=0.99, holdout≈0.5 → gap≈0.49, must flag
    result = eval_on_holdout(
        specialist, holdout_path, BINARY_SPEC, synthetic_accuracy=0.99
    )
    assert result.gap_flagged is True


def test_eval_on_holdout_detects_gap(tmp_path):
    """
    Integration test: train on easy patterns, evaluate on deliberately hard holdout.

    Uses a mock specialist that always predicts 'negative' (label_id=0).
    On a balanced 50/50 dataset this yields ~50% accuracy.
    Combined with a synthetic_accuracy of 0.99 this produces a >5pp gap.
    The result must have gap_flagged=True and accuracy < synthetic_accuracy.
    """
    from crasis.eval import eval_on_holdout

    holdout_path = _make_holdout_jsonl(tmp_path, BINARY_SPEC, n=40)
    specialist = _make_mock_specialist(BINARY_SPEC, correct=False)
    synthetic_acc = 0.99

    result = eval_on_holdout(
        specialist, holdout_path, BINARY_SPEC, synthetic_accuracy=synthetic_acc
    )

    assert result.gap_flagged is True
    assert result.accuracy < synthetic_acc
    assert result.synthetic_accuracy == synthetic_acc
    assert result.num_samples == 40
    assert result.classification_report != ""
