"""Tests for crasis.deploy — local inference wrapper (Specialist)."""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from crasis.deploy import Specialist, _softmax


# ---------------------------------------------------------------------------
# Helpers — build a real minimal ONNX package
# ---------------------------------------------------------------------------


def _make_onnx_model(num_labels: int = 2) -> onnx.ModelProto:
    """
    Build a minimal valid ONNX sequence-classifier graph.

    Architecture: input_ids [1,128] -> Cast(float) -> Flatten -> Gemm -> logits [1,num_labels]
    Accepts input_ids and attention_mask, outputs logits [1, num_labels].
    """
    feat = 128  # max_length
    W = np.zeros((num_labels, feat), dtype=np.float32)
    b = np.zeros(num_labels, dtype=np.float32)

    W_init = numpy_helper.from_array(W, name="W")
    b_init = numpy_helper.from_array(b, name="b")

    X = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, feat])
    attn = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, feat])
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, num_labels])

    cast = helper.make_node("Cast", inputs=["input_ids"], outputs=["x_float"], to=TensorProto.FLOAT)
    flatten = helper.make_node("Flatten", inputs=["x_float"], outputs=["x_flat"], axis=1)
    gemm = helper.make_node("Gemm", inputs=["x_flat", "W", "b"], outputs=["logits"], transB=1)

    graph = helper.make_graph(
        [cast, flatten, gemm],
        "classifier",
        [X, attn],
        [logits_out],
        initializer=[W_init, b_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _make_package(
    tmp_path: Path,
    name: str = "refund-detector",
    label_names: list[str] | None = None,
    with_meta: bool = True,
) -> Path:
    """
    Write a minimal but fully valid Specialist package directory.

    Contains:
      - <name>.onnx       real ONNX model (2-class by default)
      - label_map.json
      - tokenizer/        minimal tokenizer files
      - crasis_meta.json  (optional)
    """
    label_names = label_names or ["negative", "positive"]
    pkg = tmp_path / name
    pkg.mkdir()

    # ONNX model
    model = _make_onnx_model(num_labels=len(label_names))
    onnx.save(model, str(pkg / f"{name}.onnx"))

    # label_map.json
    label2id = {n: i for i, n in enumerate(label_names)}
    id2label = {str(i): n for n, i in label2id.items()}
    (pkg / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}), encoding="utf-8"
    )

    # Minimal tokenizer (bert-base-uncased vocab subset)
    tok_dir = pkg / "tokenizer"
    tok_dir.mkdir()
    # tokenizer_config.json — enough for AutoTokenizer to recognise as bert
    (tok_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_type": "bert", "do_lower_case": True}), encoding="utf-8"
    )

    if with_meta:
        meta = {
            "name": name,
            "description": "Test specialist",
            "spec_hash": "abc123",
            "label_names": label_names,
        }
        (pkg / "crasis_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    return pkg


# ---------------------------------------------------------------------------
# _softmax
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    result = _softmax(x)
    assert abs(result.sum() - 1.0) < 1e-6


def test_softmax_max_input_gets_highest_prob():
    x = np.array([0.0, 10.0, 1.0])
    result = _softmax(x)
    assert result.argmax() == 1


def test_softmax_uniform_input():
    x = np.array([1.0, 1.0, 1.0])
    result = _softmax(x)
    np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)


def test_softmax_numerical_stability():
    # Large values should not produce NaN or inf
    x = np.array([1000.0, 1001.0, 999.0])
    result = _softmax(x)
    assert np.all(np.isfinite(result))
    assert abs(result.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Specialist._find_onnx
# ---------------------------------------------------------------------------


def test_find_onnx_by_name(tmp_path):
    pkg = tmp_path / "my-model"
    pkg.mkdir()
    (pkg / "my-model.onnx").write_bytes(b"\x00")
    path = Specialist._find_onnx(pkg)
    assert path.name == "my-model.onnx"


def test_find_onnx_falls_back_to_model_onnx(tmp_path):
    pkg = tmp_path / "my-model"
    pkg.mkdir()
    (pkg / "model.onnx").write_bytes(b"\x00")
    path = Specialist._find_onnx(pkg)
    assert path.name == "model.onnx"


def test_find_onnx_falls_back_to_any_onnx(tmp_path):
    pkg = tmp_path / "my-model"
    pkg.mkdir()
    (pkg / "other.onnx").write_bytes(b"\x00")
    path = Specialist._find_onnx(pkg)
    assert path.suffix == ".onnx"


def test_find_onnx_raises_when_missing(tmp_path):
    pkg = tmp_path / "empty"
    pkg.mkdir()
    with pytest.raises(FileNotFoundError):
        Specialist._find_onnx(pkg)


# ---------------------------------------------------------------------------
# Specialist.load
# ---------------------------------------------------------------------------


def test_load_raises_if_package_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        Specialist.load(tmp_path / "nonexistent")


def test_load_raises_if_label_map_missing(tmp_path):
    from unittest.mock import MagicMock, patch

    pkg = tmp_path / "no-labels"
    pkg.mkdir()
    model = _make_onnx_model()
    onnx.save(model, str(pkg / "no-labels.onnx"))

    mock_tok = MagicMock()
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        with pytest.raises(FileNotFoundError):
            Specialist.load(pkg)


def test_load_succeeds_with_valid_package(tmp_path):
    pkg = _make_package(tmp_path, "refund-detector")
    # Load with a known tokenizer so we don't need full vocab
    from unittest.mock import patch, MagicMock

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": np.zeros((1, 128), dtype=np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
    }
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        s = Specialist.load(pkg)

    assert s.name == "refund-detector"
    assert s.label_names == ["negative", "positive"]


def test_load_reads_name_from_crasis_meta(tmp_path):
    pkg = _make_package(tmp_path, "sentiment-gate", with_meta=True)
    from unittest.mock import patch, MagicMock

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": np.zeros((1, 128), dtype=np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
    }
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        s = Specialist.load(pkg)

    assert s.name == "sentiment-gate"


def test_load_falls_back_to_dir_name_without_meta(tmp_path):
    pkg = _make_package(tmp_path, "spam-filter", with_meta=False)
    from unittest.mock import patch, MagicMock

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": np.zeros((1, 128), dtype=np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
    }
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        s = Specialist.load(pkg)

    assert s.name == "spam-filter"


# ---------------------------------------------------------------------------
# Specialist fixture (mock tokenizer, real ONNX)
# ---------------------------------------------------------------------------


@pytest.fixture()
def specialist(tmp_path):
    """A loaded Specialist with a real ONNX model and mocked tokenizer."""
    from unittest.mock import MagicMock, patch

    pkg = _make_package(tmp_path, "refund-detector")

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": np.zeros((1, 128), dtype=np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
    }
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        s = Specialist.load(pkg)
    return s


@pytest.fixture()
def multiclass_specialist(tmp_path):
    """A loaded 4-class Specialist with a real ONNX model and mocked tokenizer."""
    from unittest.mock import MagicMock, patch

    labels = ["billing", "technical", "returns", "general"]
    pkg = _make_package(tmp_path, "support-router", label_names=labels)

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": np.zeros((1, 128), dtype=np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
    }
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        s = Specialist.load(pkg)
    return s


# ---------------------------------------------------------------------------
# Specialist.classify
# ---------------------------------------------------------------------------


def test_classify_returns_dict(specialist):
    result = specialist.classify("I want a refund")
    assert isinstance(result, dict)


def test_classify_has_required_keys(specialist):
    result = specialist.classify("I want a refund")
    assert "label" in result
    assert "label_id" in result
    assert "confidence" in result
    assert "latency_ms" in result


def test_classify_label_is_valid(specialist):
    result = specialist.classify("I want a refund")
    assert result["label"] in specialist.label_names


def test_classify_label_id_matches_label(specialist):
    result = specialist.classify("I want a refund")
    assert specialist.label_names[result["label_id"]] == result["label"]


def test_classify_confidence_in_range(specialist):
    result = specialist.classify("I want a refund")
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_latency_positive(specialist):
    result = specialist.classify("any text")
    assert result["latency_ms"] > 0.0


def test_classify_confidence_rounded(specialist):
    result = specialist.classify("test")
    # Should be rounded to 4 decimal places
    assert result["confidence"] == round(result["confidence"], 4)


def test_classify_latency_rounded(specialist):
    result = specialist.classify("test")
    assert result["latency_ms"] == round(result["latency_ms"], 2)


def test_classify_multiclass_label_valid(multiclass_specialist):
    result = multiclass_specialist.classify("My invoice is wrong")
    assert result["label"] in multiclass_specialist.label_names


def test_classify_multiclass_four_classes(multiclass_specialist):
    assert len(multiclass_specialist.label_names) == 4


# ---------------------------------------------------------------------------
# Specialist.classify_batch
# ---------------------------------------------------------------------------


def test_classify_batch_returns_list(specialist):
    results = specialist.classify_batch(["text one", "text two", "text three"])
    assert isinstance(results, list)
    assert len(results) == 3


def test_classify_batch_order_preserved(specialist):
    texts = ["first", "second", "third"]
    results = specialist.classify_batch(texts)
    # Each result must be a valid classification dict
    for r in results:
        assert "label" in r
        assert "label_id" in r


def test_classify_batch_empty_input(specialist):
    assert specialist.classify_batch([]) == []


def test_classify_batch_single(specialist):
    results = specialist.classify_batch(["just one"])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Specialist.__repr__
# ---------------------------------------------------------------------------


def test_repr_contains_name(specialist):
    assert "refund-detector" in repr(specialist)


def test_repr_contains_classes(specialist):
    r = repr(specialist)
    assert "negative" in r
    assert "positive" in r


# ---------------------------------------------------------------------------
# Inference is local — no network calls
# ---------------------------------------------------------------------------


def test_classify_makes_no_network_calls(specialist):
    """classify() must never make outbound requests. Verify by patching socket."""
    import socket
    from unittest.mock import patch

    original_connect = socket.socket.connect

    def fail_connect(self, *args, **kwargs):
        raise AssertionError(f"Network call attempted during inference: {args}")

    with patch.object(socket.socket, "connect", fail_connect):
        # Should complete without triggering fail_connect
        result = specialist.classify("test input")

    assert "label" in result
