"""Tests for crasis.tools — CrasisToolkit."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from crasis.tools import CrasisToolkit
from crasis.deploy import Specialist


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_deploy.py pattern)
# ---------------------------------------------------------------------------


def _make_onnx_model(num_labels: int = 2) -> onnx.ModelProto:
    feat = 128
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
        [cast, flatten, gemm], "classifier", [X, attn], [logits_out],
        initializer=[W_init, b_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _make_package(tmp_path: Path, name: str, label_names: list[str] | None = None) -> Path:
    label_names = label_names or ["negative", "positive"]
    pkg = tmp_path / name
    pkg.mkdir()
    model = _make_onnx_model(num_labels=len(label_names))
    onnx.save(model, str(pkg / f"{name}.onnx"))
    label2id = {n: i for i, n in enumerate(label_names)}
    id2label = {str(i): n for n, i in label2id.items()}
    (pkg / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}), encoding="utf-8"
    )
    tok_dir = pkg / "tokenizer"
    tok_dir.mkdir()
    (tok_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_type": "bert", "do_lower_case": True}), encoding="utf-8"
    )
    meta = {"name": name, "description": "Test", "spec_hash": "abc", "label_names": label_names}
    (pkg / "crasis_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return pkg


def _mock_tok():
    tok = MagicMock()
    tok.return_value = {
        "input_ids": np.zeros((1, 128), dtype=np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
    }
    return tok


@pytest.fixture()
def binary_specialist(tmp_path):
    pkg = _make_package(tmp_path, "sentiment-gate")
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=_mock_tok()):
        return Specialist.load(pkg)


@pytest.fixture()
def multiclass_specialist(tmp_path):
    labels = ["billing", "technical", "returns", "general"]
    pkg = _make_package(tmp_path, "support-router", label_names=labels)
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=_mock_tok()):
        return Specialist.load(pkg)


@pytest.fixture()
def toolkit(binary_specialist, multiclass_specialist):
    return CrasisToolkit.from_specialists(binary_specialist, multiclass_specialist)


# ---------------------------------------------------------------------------
# CrasisToolkit construction
# ---------------------------------------------------------------------------


def test_from_specialists_loads_both(binary_specialist, multiclass_specialist):
    tk = CrasisToolkit.from_specialists(binary_specialist, multiclass_specialist)
    assert set(tk.specialists()) == {"sentiment-gate", "support-router"}


def test_from_dir_discovers_packages(tmp_path):
    _make_package(tmp_path, "spam-filter")
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=_mock_tok()):
        tk = CrasisToolkit.from_dir(tmp_path)
    assert "spam-filter" in tk.specialists()


def test_from_dir_skips_non_onnx_dirs(tmp_path):
    (tmp_path / "not-a-specialist").mkdir()
    _make_package(tmp_path, "spam-filter")
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=_mock_tok()):
        tk = CrasisToolkit.from_dir(tmp_path)
    assert "not-a-specialist" not in tk.specialists()


def test_from_dir_raises_if_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        CrasisToolkit.from_dir(tmp_path)


def test_repr_contains_specialist_names(toolkit):
    r = repr(toolkit)
    assert "sentiment-gate" in r
    assert "support-router" in r


# ---------------------------------------------------------------------------
# get_specialist
# ---------------------------------------------------------------------------


def test_get_specialist_returns_instance(toolkit, binary_specialist):
    s = toolkit.get_specialist("sentiment-gate")
    assert s.name == "sentiment-gate"


def test_get_specialist_raises_on_unknown(toolkit):
    with pytest.raises(KeyError, match="not found"):
        toolkit.get_specialist("nonexistent")


def test_get_specialist_error_lists_available(toolkit):
    with pytest.raises(KeyError, match="sentiment-gate"):
        toolkit.get_specialist("nonexistent")


# ---------------------------------------------------------------------------
# classify (direct dispatch)
# ---------------------------------------------------------------------------


def test_classify_returns_dict(toolkit):
    result = toolkit.classify("sentiment-gate", "I want a refund")
    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result


def test_classify_raises_on_unknown_specialist(toolkit):
    with pytest.raises(KeyError):
        toolkit.classify("unknown-specialist", "text")


def test_classify_label_in_valid_set(toolkit):
    result = toolkit.classify("sentiment-gate", "test text")
    assert result["label"] in ["negative", "positive"]


# ---------------------------------------------------------------------------
# openai_tools
# ---------------------------------------------------------------------------


def test_openai_tools_returns_list(toolkit):
    tools = toolkit.openai_tools()
    assert isinstance(tools, list)
    assert len(tools) == 2


def test_openai_tool_has_type_function(toolkit):
    for tool in toolkit.openai_tools():
        assert tool["type"] == "function"


def test_openai_tool_names_prefixed_classify(toolkit):
    names = {t["function"]["name"] for t in toolkit.openai_tools()}
    assert all(n.startswith("classify_") for n in names)


def test_openai_tool_has_text_parameter(toolkit):
    for tool in toolkit.openai_tools():
        props = tool["function"]["parameters"]["properties"]
        assert "text" in props


def test_openai_tool_text_is_required(toolkit):
    for tool in toolkit.openai_tools():
        assert "text" in tool["function"]["parameters"]["required"]


def test_openai_tool_description_mentions_specialist(toolkit):
    tool = next(t for t in toolkit.openai_tools() if "sentiment_gate" in t["function"]["name"])
    assert "sentiment-gate" in tool["function"]["description"]


# ---------------------------------------------------------------------------
# handle_tool_call (OpenAI)
# ---------------------------------------------------------------------------


def test_handle_tool_call_returns_json_string(toolkit):
    tool_call = MagicMock()
    tool_call.function.name = "classify_sentiment_gate"
    tool_call.function.arguments = json.dumps({"text": "I want a refund"})
    result = toolkit.handle_tool_call(tool_call)
    parsed = json.loads(result)
    assert "label" in parsed


def test_handle_tool_call_raises_on_unknown_tool(toolkit):
    tool_call = MagicMock()
    tool_call.function.name = "classify_nonexistent_model"
    tool_call.function.arguments = json.dumps({"text": "text"})
    with pytest.raises(KeyError):
        toolkit.handle_tool_call(tool_call)


def test_openai_tool_message_structure(toolkit):
    tool_call = MagicMock()
    tool_call.id = "call_abc123"
    tool_call.function.name = "classify_sentiment_gate"
    tool_call.function.arguments = json.dumps({"text": "test"})
    result_json = toolkit.handle_tool_call(tool_call)
    msg = toolkit.openai_tool_message(tool_call, result_json)
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_abc123"
    assert msg["content"] == result_json


# ---------------------------------------------------------------------------
# anthropic_tools
# ---------------------------------------------------------------------------


def test_anthropic_tools_returns_list(toolkit):
    tools = toolkit.anthropic_tools()
    assert isinstance(tools, list)
    assert len(tools) == 2


def test_anthropic_tool_has_input_schema(toolkit):
    for tool in toolkit.anthropic_tools():
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"


def test_anthropic_tool_name_prefixed_classify(toolkit):
    for tool in toolkit.anthropic_tools():
        assert tool["name"].startswith("classify_")


def test_anthropic_tool_text_is_required(toolkit):
    for tool in toolkit.anthropic_tools():
        assert "text" in tool["input_schema"]["required"]


# ---------------------------------------------------------------------------
# handle_tool_use (Anthropic)
# ---------------------------------------------------------------------------


def test_handle_tool_use_returns_tool_result_dict(toolkit):
    block = MagicMock()
    block.name = "classify_sentiment_gate"
    block.id = "toolu_abc"
    block.input = {"text": "I hate this product"}
    result = toolkit.handle_tool_use(block)
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_abc"


def test_handle_tool_use_content_is_json(toolkit):
    block = MagicMock()
    block.name = "classify_sentiment_gate"
    block.id = "toolu_xyz"
    block.input = {"text": "test"}
    result = toolkit.handle_tool_use(block)
    parsed = json.loads(result["content"])
    assert "label" in parsed


# ---------------------------------------------------------------------------
# gemini_tools
# ---------------------------------------------------------------------------


def test_gemini_tools_raises_without_sdk(toolkit):
    with patch.dict("sys.modules", {"google.generativeai": None, "google.generativeai.types": None}):
        with pytest.raises(ImportError, match="google-generativeai"):
            toolkit.gemini_tools()


# ---------------------------------------------------------------------------
# _dispatch internals
# ---------------------------------------------------------------------------


def test_dispatch_hyphen_to_underscore_mapping(toolkit):
    result_json = toolkit._dispatch("classify_sentiment_gate", {"text": "hello"})
    assert json.loads(result_json)["label"] in ["negative", "positive"]


def test_dispatch_raises_on_unknown_tool(toolkit):
    with pytest.raises(KeyError):
        toolkit._dispatch("classify_nonexistent", {"text": "text"})


def test_dispatch_empty_text_still_classifies(toolkit):
    result_json = toolkit._dispatch("classify_sentiment_gate", {"text": ""})
    parsed = json.loads(result_json)
    assert "label" in parsed


# ---------------------------------------------------------------------------
# Tool name helpers
# ---------------------------------------------------------------------------


def test_tool_name_replaces_hyphens(binary_specialist):
    name = CrasisToolkit._tool_name(binary_specialist)
    assert name == "classify_sentiment_gate"
    assert "-" not in name


def test_tool_description_mentions_labels(binary_specialist):
    desc = CrasisToolkit._tool_description(binary_specialist)
    assert "negative" in desc
    assert "positive" in desc


def test_tool_description_no_api_calls(binary_specialist):
    desc = CrasisToolkit._tool_description(binary_specialist)
    assert "No API" in desc or "no API" in desc or "No cloud" in desc
