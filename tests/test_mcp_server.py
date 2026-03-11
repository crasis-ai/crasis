"""Tests for crasis.mcp_server — MCP server tool definitions and handlers.

Tests are split into two tiers:
  - Tier 1 (no optional deps): generate_claude_desktop_config, empty-toolkit path,
    CLI _pull_specialist_sync integration via mock, and ImportError guards.
  - Tier 2 (requires mcp): handler correctness tests — skipped if `mcp` is not installed.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from crasis.deploy import Specialist
from crasis.tools import CrasisToolkit
from crasis.mcp_server import generate_claude_desktop_config

# Detect optional deps once at module load
try:
    import mcp  # noqa: F401
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

requires_mcp = pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp package not installed")


# ---------------------------------------------------------------------------
# Shared helpers
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
def toolkit(binary_specialist):
    return CrasisToolkit.from_specialists(binary_specialist)


# ---------------------------------------------------------------------------
# generate_claude_desktop_config — no optional deps required
# ---------------------------------------------------------------------------


def test_desktop_config_command_key(tmp_path):
    cfg = generate_claude_desktop_config(tmp_path)
    assert "crasis" in cfg
    assert cfg["crasis"]["command"] == "crasis"


def test_desktop_config_args_contains_mcp(tmp_path):
    cfg = generate_claude_desktop_config(tmp_path)
    assert "mcp" in cfg["crasis"]["args"]


def test_desktop_config_args_contains_models_dir(tmp_path):
    cfg = generate_claude_desktop_config(tmp_path)
    args = cfg["crasis"]["args"]
    assert "--models-dir" in args
    assert str(tmp_path.absolute()) in args


def test_desktop_config_no_env_without_api_key(tmp_path):
    cfg = generate_claude_desktop_config(tmp_path)
    assert "env" not in cfg["crasis"]


def test_desktop_config_env_with_api_key(tmp_path):
    cfg = generate_claude_desktop_config(tmp_path, api_key="sk-test")
    assert cfg["crasis"]["env"]["OPENROUTER_API_KEY"] == "sk-test"


def test_desktop_config_custom_executable(tmp_path):
    cfg = generate_claude_desktop_config(tmp_path, crasis_executable="/usr/local/bin/crasis")
    assert cfg["crasis"]["command"] == "/usr/local/bin/crasis"


def test_desktop_config_absolute_path(tmp_path):
    cfg = generate_claude_desktop_config(tmp_path / "specialists")
    args = cfg["crasis"]["args"]
    models_arg = args[args.index("--models-dir") + 1]
    assert Path(models_arg).is_absolute()


# ---------------------------------------------------------------------------
# run_server — ImportError without mcp package
# ---------------------------------------------------------------------------


def test_run_server_raises_without_mcp(tmp_path):
    from crasis.mcp_server import run_server

    with patch.dict("sys.modules", {"mcp": None, "mcp.server": None, "mcp.server.stdio": None, "mcp.types": None}):
        with pytest.raises((ImportError, TypeError)):
            asyncio.run(run_server(models_dir=tmp_path))


# ---------------------------------------------------------------------------
# _handle_list_specialists — requires mcp
# ---------------------------------------------------------------------------


@requires_mcp
def test_list_specialists_returns_text_content(toolkit):
    from crasis.mcp_server import _handle_list_specialists
    result = _handle_list_specialists(toolkit)
    assert len(result) == 1
    assert result[0].type == "text"


@requires_mcp
def test_list_specialists_json_parseable(toolkit):
    from crasis.mcp_server import _handle_list_specialists
    result = _handle_list_specialists(toolkit)
    data = json.loads(result[0].text)
    assert isinstance(data, list)


@requires_mcp
def test_list_specialists_contains_name(toolkit):
    from crasis.mcp_server import _handle_list_specialists
    result = _handle_list_specialists(toolkit)
    data = json.loads(result[0].text)
    names = [entry["name"] for entry in data]
    assert "sentiment-gate" in names


@requires_mcp
def test_list_specialists_contains_labels(toolkit):
    from crasis.mcp_server import _handle_list_specialists
    result = _handle_list_specialists(toolkit)
    data = json.loads(result[0].text)
    entry = next(e for e in data if e["name"] == "sentiment-gate")
    assert "negative" in entry["labels"]
    assert "positive" in entry["labels"]


@requires_mcp
def test_list_specialists_contains_tool_name(toolkit):
    from crasis.mcp_server import _handle_list_specialists
    result = _handle_list_specialists(toolkit)
    data = json.loads(result[0].text)
    entry = next(e for e in data if e["name"] == "sentiment-gate")
    assert entry["tool_name"] == "classify_sentiment_gate"


@requires_mcp
def test_list_specialists_empty_toolkit():
    from crasis.mcp_server import _handle_list_specialists
    empty_toolkit = CrasisToolkit({})
    result = _handle_list_specialists(empty_toolkit)
    data = json.loads(result[0].text)
    assert data == []


# ---------------------------------------------------------------------------
# _handle_classify — requires mcp
# ---------------------------------------------------------------------------


@requires_mcp
def test_handle_classify_returns_text_content(toolkit):
    from crasis.mcp_server import _handle_classify
    result = asyncio.run(_handle_classify(toolkit, "classify_sentiment_gate", {"text": "I want a refund"}))
    assert len(result) == 1
    assert result[0].type == "text"


@requires_mcp
def test_handle_classify_text_contains_label(toolkit):
    from crasis.mcp_server import _handle_classify
    result = asyncio.run(_handle_classify(toolkit, "classify_sentiment_gate", {"text": "great product"}))
    assert "label:" in result[0].text


@requires_mcp
def test_handle_classify_text_contains_confidence(toolkit):
    from crasis.mcp_server import _handle_classify
    result = asyncio.run(_handle_classify(toolkit, "classify_sentiment_gate", {"text": "test"}))
    assert "confidence:" in result[0].text


@requires_mcp
def test_handle_classify_text_contains_latency(toolkit):
    from crasis.mcp_server import _handle_classify
    result = asyncio.run(_handle_classify(toolkit, "classify_sentiment_gate", {"text": "test"}))
    assert "latency_ms:" in result[0].text


@requires_mcp
def test_handle_classify_unknown_tool_returns_error(toolkit):
    from crasis.mcp_server import _handle_classify
    result = asyncio.run(_handle_classify(toolkit, "classify_nonexistent", {"text": "test"}))
    assert "Error" in result[0].text


@requires_mcp
def test_handle_classify_empty_text_returns_error(toolkit):
    from crasis.mcp_server import _handle_classify
    result = asyncio.run(_handle_classify(toolkit, "classify_sentiment_gate", {}))
    assert "Error" in result[0].text


# ---------------------------------------------------------------------------
# _handle_pull — requires mcp
# ---------------------------------------------------------------------------


@requires_mcp
def test_handle_pull_missing_name_returns_error():
    from crasis.mcp_server import _handle_pull
    server = MagicMock()
    result = asyncio.run(_handle_pull(Path("/tmp"), "_tool", {}, server))
    assert "Error" in result[0].text
    assert "name" in result[0].text


@requires_mcp
def test_handle_pull_calls_pull_specialist_sync(tmp_path):
    from crasis.mcp_server import _handle_pull

    server = MagicMock()
    server.request_context.session.send_log_message = AsyncMock()

    pulled_path = tmp_path / "spam-filter"
    pulled_path.mkdir()

    with patch("crasis.cli._pull_specialist_sync", return_value=pulled_path) as mock_pull:
        asyncio.run(_handle_pull(tmp_path, "_tool", {"name": "spam-filter"}, server))

    mock_pull.assert_called_once()


@requires_mcp
def test_handle_pull_error_from_sync_surfaces_in_result(tmp_path):
    from crasis.mcp_server import _handle_pull

    server = MagicMock()
    server.request_context.session.send_log_message = AsyncMock()

    with patch(
        "crasis.cli._pull_specialist_sync",
        side_effect=RuntimeError("specialist 'bad-name' not found"),
    ):
        result = asyncio.run(_handle_pull(tmp_path, "_tool", {"name": "bad-name"}, server))

    assert "ERROR" in result[0].text or "not found" in result[0].text.lower()


@requires_mcp
def test_handle_pull_restart_message_in_result(tmp_path):
    from crasis.mcp_server import _handle_pull

    server = MagicMock()
    server.request_context.session.send_log_message = AsyncMock()

    pulled_path = tmp_path / "spam-filter"
    pulled_path.mkdir()

    with patch("crasis.cli._pull_specialist_sync", return_value=pulled_path):
        result = asyncio.run(_handle_pull(tmp_path, "_tool", {"name": "spam-filter"}, server))

    assert "Restart" in result[0].text or "restart" in result[0].text


# ---------------------------------------------------------------------------
# _handle_build — error cases only (no train deps or real pipeline needed)
# ---------------------------------------------------------------------------


@requires_mcp
def test_handle_build_missing_spec_yaml_returns_error(tmp_path):
    from crasis.mcp_server import _handle_build
    server = MagicMock()
    result = asyncio.run(_handle_build(
        models_dir=tmp_path,
        data_dir=tmp_path,
        api_key="sk-test",
        arguments={},
        server=server,
    ))
    assert "Error" in result[0].text
    assert "spec_yaml" in result[0].text


@requires_mcp
def test_handle_build_missing_api_key_returns_error(tmp_path):
    from crasis.mcp_server import _handle_build
    server = MagicMock()
    result = asyncio.run(_handle_build(
        models_dir=tmp_path,
        data_dir=tmp_path,
        api_key=None,
        arguments={"spec_yaml": "name: test"},
        server=server,
    ))
    assert "Error" in result[0].text
    assert "OPENROUTER_API_KEY" in result[0].text
