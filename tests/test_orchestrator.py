"""Tests for crasis.orchestrator — CrasisOrchestrator and OrchestratorResult."""

import json
import time
from dataclasses import fields
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from crasis.deploy import Specialist
from crasis.orchestrator import CrasisOrchestrator, OrchestratorResult
from crasis.tools import CrasisToolkit


# ---------------------------------------------------------------------------
# Helpers
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


def _make_package(tmp_path, name, label_names=None):
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
def toolkit(tmp_path):
    pkg = _make_package(tmp_path, "sentiment-gate")
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=_mock_tok()):
        s = Specialist.load(pkg)
    return CrasisToolkit.from_specialists(s)


# ---------------------------------------------------------------------------
# OrchestratorResult
# ---------------------------------------------------------------------------


def test_orchestrator_result_fields():
    field_names = {f.name for f in fields(OrchestratorResult)}
    assert "response" in field_names
    assert "tool_calls" in field_names
    assert "total_latency_ms" in field_names
    assert "frontier_model" in field_names
    assert "provider" in field_names


def test_orchestrator_result_defaults():
    r = OrchestratorResult(response="hello")
    assert r.tool_calls == []
    assert r.total_latency_ms == 0.0
    assert r.frontier_model == ""
    assert r.provider == ""


def test_orchestrator_result_tool_calls_mutable_default():
    # Each instance must get its own list (not shared default)
    r1 = OrchestratorResult(response="a")
    r2 = OrchestratorResult(response="b")
    r1.tool_calls.append(("tool", "input", {}))
    assert len(r2.tool_calls) == 0


# ---------------------------------------------------------------------------
# CrasisOrchestrator construction
# ---------------------------------------------------------------------------


def test_orchestrator_invalid_provider(toolkit):
    with pytest.raises(ValueError, match="Unknown provider"):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="azure", model="gpt-4")  # type: ignore[arg-type]
        orch.run("test")


def test_orchestrator_stores_model(toolkit):
    orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
    assert orch._model == "claude-opus-4-5"


def test_orchestrator_stores_provider(toolkit):
    orch = CrasisOrchestrator(toolkit=toolkit, provider="openai", model="gpt-4o")
    assert orch._provider == "openai"


def test_orchestrator_default_system_prompt(toolkit):
    orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
    assert orch._system_prompt is not None
    assert len(orch._system_prompt) > 0


def test_orchestrator_custom_system_prompt(toolkit):
    orch = CrasisOrchestrator(
        toolkit=toolkit, provider="anthropic", model="claude-opus-4-5",
        system_prompt="Custom prompt."
    )
    assert orch._system_prompt == "Custom prompt."


def test_orchestrator_default_max_iterations(toolkit):
    orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
    assert orch._max_iterations == 10


# ---------------------------------------------------------------------------
# Anthropic loop — mocked SDK
# ---------------------------------------------------------------------------


def _make_anthropic_response(stop_reason: str, content: list) -> MagicMock:
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content
    return response


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(name: str, input_: dict, block_id: str = "toolu_1") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = block_id
    block.input = input_
    return block


def test_anthropic_loop_raises_without_sdk(toolkit):
    with patch.dict("sys.modules", {"anthropic": None}):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
        with pytest.raises(ImportError, match="crasis\\[agents\\]"):
            orch.run("test")


def test_anthropic_loop_returns_result_on_end_turn(toolkit):
    text_block = _make_text_block("The sentiment is negative.")
    final_response = _make_anthropic_response("end_turn", [text_block])

    mock_client = MagicMock()
    mock_client.messages.create.return_value = final_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
        result = orch.run("Is this customer angry?")

    assert isinstance(result, OrchestratorResult)
    assert result.response == "The sentiment is negative."
    assert result.provider == "anthropic"
    assert result.frontier_model == "claude-opus-4-5"


def test_anthropic_loop_total_latency_nonzero(toolkit):
    text_block = _make_text_block("done")
    final_response = _make_anthropic_response("end_turn", [text_block])

    mock_client = MagicMock()
    mock_client.messages.create.return_value = final_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
        result = orch.run("test")

    assert result.total_latency_ms > 0.0


def test_anthropic_loop_executes_tool_call(toolkit):
    tool_block = _make_tool_use_block("classify_sentiment_gate", {"text": "I want a refund"})
    tool_response = _make_anthropic_response("tool_use", [tool_block])
    text_block = _make_text_block("The customer is angry.")
    final_response = _make_anthropic_response("end_turn", [text_block])

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_response, final_response]

    with patch("anthropic.Anthropic", return_value=mock_client):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
        result = orch.run("Is this customer angry?")

    assert len(result.tool_calls) == 1
    tool_name, input_text, tool_result = result.tool_calls[0]
    assert tool_name == "classify_sentiment_gate"
    assert input_text == "I want a refund"
    assert "label" in tool_result


def test_anthropic_loop_max_iterations_guard(toolkit):
    # Always returns tool_use — should hit max_iterations
    tool_block = _make_tool_use_block("classify_sentiment_gate", {"text": "test"})
    infinite_tool_response = _make_anthropic_response("tool_use", [tool_block])

    mock_client = MagicMock()
    mock_client.messages.create.return_value = infinite_tool_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        orch = CrasisOrchestrator(
            toolkit=toolkit, provider="anthropic", model="claude-opus-4-5", max_iterations=3
        )
        result = orch.run("loop forever")

    assert result.response == "[max_iterations reached]"
    assert mock_client.messages.create.call_count == 3


def test_anthropic_loop_uses_api_key(toolkit):
    text_block = _make_text_block("done")
    final_response = _make_anthropic_response("end_turn", [text_block])

    mock_client = MagicMock()
    mock_client.messages.create.return_value = final_response

    with patch("anthropic.Anthropic", return_value=mock_client) as mock_cls:
        orch = CrasisOrchestrator(
            toolkit=toolkit, provider="anthropic", model="claude-opus-4-5", api_key="sk-ant-test"
        )
        orch.run("test")

    mock_cls.assert_called_once_with(api_key="sk-ant-test")


# ---------------------------------------------------------------------------
# OpenAI loop — mocked SDK
# ---------------------------------------------------------------------------


def _make_openai_response(finish_reason: str, content: str | None = None, tool_calls=None) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    response.choices = [choice]
    return response


def test_openai_loop_raises_without_sdk(toolkit):
    with patch.dict("sys.modules", {"openai": None}):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="openai", model="gpt-4o")
        with pytest.raises(ImportError, match="crasis\\[agents\\]"):
            orch.run("test")


def test_openai_loop_returns_result_on_stop(toolkit):
    final_response = _make_openai_response("stop", content="The text is negative.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = final_response

    with patch("openai.OpenAI", return_value=mock_client):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="openai", model="gpt-4o")
        result = orch.run("classify this")

    assert result.response == "The text is negative."
    assert result.provider == "openai"
    assert result.frontier_model == "gpt-4o"


def test_openai_loop_executes_tool_call(toolkit):
    tool_call = MagicMock()
    tool_call.id = "call_abc"
    tool_call.function.name = "classify_sentiment_gate"
    tool_call.function.arguments = json.dumps({"text": "awful product"})

    tool_response = _make_openai_response("tool_calls", tool_calls=[tool_call])
    final_response = _make_openai_response("stop", content="Customer is unhappy.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [tool_response, final_response]

    with patch("openai.OpenAI", return_value=mock_client):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="openai", model="gpt-4o")
        result = orch.run("rate the sentiment")

    assert len(result.tool_calls) == 1
    tool_name, input_text, tool_result = result.tool_calls[0]
    assert tool_name == "classify_sentiment_gate"
    assert input_text == "awful product"
    assert "label" in tool_result


def test_openai_loop_max_iterations_guard(toolkit):
    tool_call = MagicMock()
    tool_call.id = "call_loop"
    tool_call.function.name = "classify_sentiment_gate"
    tool_call.function.arguments = json.dumps({"text": "test"})

    infinite_response = _make_openai_response("tool_calls", tool_calls=[tool_call])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = infinite_response

    with patch("openai.OpenAI", return_value=mock_client):
        orch = CrasisOrchestrator(
            toolkit=toolkit, provider="openai", model="gpt-4o", max_iterations=2
        )
        result = orch.run("loop")

    assert result.response == "[max_iterations reached]"
    assert mock_client.chat.completions.create.call_count == 2


def test_openai_loop_uses_api_key(toolkit):
    final_response = _make_openai_response("stop", content="done")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = final_response

    with patch("openai.OpenAI", return_value=mock_client) as mock_cls:
        orch = CrasisOrchestrator(
            toolkit=toolkit, provider="openai", model="gpt-4o", api_key="sk-openai-test"
        )
        orch.run("test")

    mock_cls.assert_called_once_with(api_key="sk-openai-test")


# ---------------------------------------------------------------------------
# Gemini loop — mocked SDK
# ---------------------------------------------------------------------------


def test_gemini_loop_raises_without_sdk(toolkit):
    with patch.dict("sys.modules", {"google.generativeai": None, "google": None}):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="gemini", model="gemini-pro")
        with pytest.raises(ImportError, match="crasis\\[agents\\]"):
            orch.run("test")


def test_gemini_loop_returns_result_on_no_function_calls(toolkit):
    mock_response = MagicMock()
    mock_response.text = "The sentiment is neutral."
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(spec=[])]))]
    # part has no function_call attribute (spec=[]) so hasattr check fails

    mock_chat = MagicMock()
    mock_chat.send_message.return_value = mock_response

    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_chat

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model

    with patch.dict("sys.modules", {"google.generativeai": mock_genai, "google.generativeai.types": MagicMock()}):
        orch = CrasisOrchestrator(toolkit=toolkit, provider="gemini", model="gemini-pro")
        result = orch.run("classify this")

    assert result.response == "The sentiment is neutral."
    assert result.provider == "gemini"
    assert result.frontier_model == "gemini-pro"


# ---------------------------------------------------------------------------
# run() — provider dispatch
# ---------------------------------------------------------------------------


def test_run_dispatches_to_anthropic(toolkit):
    with patch.object(CrasisOrchestrator, "_run_anthropic_loop") as mock_loop:
        mock_loop.return_value = OrchestratorResult(response="ok")
        orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
        orch.run("test")
    mock_loop.assert_called_once_with("test")


def test_run_dispatches_to_openai(toolkit):
    with patch.object(CrasisOrchestrator, "_run_openai_loop") as mock_loop:
        mock_loop.return_value = OrchestratorResult(response="ok")
        orch = CrasisOrchestrator(toolkit=toolkit, provider="openai", model="gpt-4o")
        orch.run("test")
    mock_loop.assert_called_once_with("test")


def test_run_dispatches_to_gemini(toolkit):
    with patch.object(CrasisOrchestrator, "_run_gemini_loop") as mock_loop:
        mock_loop.return_value = OrchestratorResult(response="ok")
        orch = CrasisOrchestrator(toolkit=toolkit, provider="gemini", model="gemini-pro")
        orch.run("test")
    mock_loop.assert_called_once_with("test")


def test_run_sets_frontier_model_on_result(toolkit):
    with patch.object(CrasisOrchestrator, "_run_anthropic_loop") as mock_loop:
        mock_loop.return_value = OrchestratorResult(response="ok")
        orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
        result = orch.run("test")
    assert result.frontier_model == "claude-opus-4-5"


def test_run_sets_provider_on_result(toolkit):
    with patch.object(CrasisOrchestrator, "_run_openai_loop") as mock_loop:
        mock_loop.return_value = OrchestratorResult(response="ok")
        orch = CrasisOrchestrator(toolkit=toolkit, provider="openai", model="gpt-4o")
        result = orch.run("test")
    assert result.provider == "openai"


def test_run_sets_total_latency_ms(toolkit):
    with patch.object(CrasisOrchestrator, "_run_anthropic_loop") as mock_loop:
        mock_loop.return_value = OrchestratorResult(response="ok")
        orch = CrasisOrchestrator(toolkit=toolkit, provider="anthropic", model="claude-opus-4-5")
        result = orch.run("test")
    assert result.total_latency_ms > 0.0


# ---------------------------------------------------------------------------
# Public API export
# ---------------------------------------------------------------------------


def test_orchestrator_exported_from_crasis():
    from crasis import CrasisOrchestrator as CO, OrchestratorResult as OR
    assert CO is CrasisOrchestrator
    assert OR is OrchestratorResult
