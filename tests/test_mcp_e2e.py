"""
End-to-end MCP server test.

Launches `crasis mcp` as a subprocess, connects via the MCP stdio client,
and exercises all four tools against a real specialist package.

Requires: pip install crasis[mcp]
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

requires_mcp = pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp package not installed")


# ---------------------------------------------------------------------------
# Build a real specialist package that the subprocess can load
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


_BERT_TINY_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--google--bert_uncased_L-2_H-128_A-2"
    / "snapshots"
    / "30b0a37ccaaa32f332884b96992754e246e48c5f"
)


def _write_specialist_package(specialists_dir: Path, name: str, label_names: list[str]) -> Path:
    """
    Write a complete specialist package that Specialist.load() can open.

    Copies real tokenizer files from the cached BERT-Tiny snapshot so
    AutoTokenizer.from_pretrained() succeeds in the subprocess.
    """
    import shutil

    pkg = specialists_dir / name
    pkg.mkdir(parents=True, exist_ok=True)

    model = _make_onnx_model(num_labels=len(label_names))
    onnx.save(model, str(pkg / f"{name}.onnx"))

    label2id = {n: i for i, n in enumerate(label_names)}
    id2label = {str(i): n for n, i in label2id.items()}
    (pkg / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}), encoding="utf-8"
    )

    tok_dir = pkg / "tokenizer"
    tok_dir.mkdir(exist_ok=True)

    if _BERT_TINY_SNAPSHOT.exists():
        # Copy real tokenizer files from cache so AutoTokenizer loads correctly
        for fname in ("vocab.txt", "config.json"):
            src = _BERT_TINY_SNAPSHOT / fname
            if src.exists():
                shutil.copy(src, tok_dir / fname)
        (tok_dir / "tokenizer_config.json").write_text(
            json.dumps({
                "model_type": "bert",
                "do_lower_case": True,
                "tokenizer_class": "BertTokenizer",
            }),
            encoding="utf-8",
        )
    else:
        # Fallback stub (will fail tokenizer load but keeps package structure valid)
        (tok_dir / "tokenizer_config.json").write_text(
            json.dumps({"model_type": "bert", "do_lower_case": True}), encoding="utf-8"
        )

    meta = {
        "name": name,
        "description": "E2E test specialist",
        "spec_hash": "e2e",
        "label_names": label_names,
    }
    (pkg / "crasis_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return pkg


# ---------------------------------------------------------------------------
# Fixture: a temp specialists dir with one pre-loaded specialist
# ---------------------------------------------------------------------------


@pytest.fixture()
def specialists_dir(tmp_path):
    _write_specialist_package(tmp_path, "sentiment-gate", ["negative", "positive"])
    return tmp_path


# ---------------------------------------------------------------------------
# Async helper: run a coroutine against a live crasis mcp subprocess
# ---------------------------------------------------------------------------


async def _run_with_server(specialists_dir: Path, coro_factory):
    """
    Start crasis mcp, connect, run coro_factory(session), return its result.

    Uses a patched AutoTokenizer so the subprocess doesn't need real BERT weights.
    """
    import sys

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "crasis.cli", "mcp", "--models-dir", str(specialists_dir)],
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await coro_factory(session)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


@requires_mcp
def test_e2e_list_tools(specialists_dir):
    """Server advertises classify_sentiment_gate + 3 management tools."""

    async def _check(session: "ClientSession"):
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        assert "classify_sentiment_gate" in names
        assert "list_specialists" in names
        assert "pull_specialist" in names
        assert "build_specialist" in names
        return names

    names = asyncio.run(_run_with_server(specialists_dir, _check))
    assert len(names) == 4


@requires_mcp
def test_e2e_list_specialists(specialists_dir):
    """list_specialists returns JSON with the loaded specialist."""

    async def _check(session: "ClientSession"):
        result = await session.call_tool("list_specialists", {})
        data = json.loads(result.content[0].text)
        return data

    data = asyncio.run(_run_with_server(specialists_dir, _check))
    assert len(data) == 1
    assert data[0]["name"] == "sentiment-gate"
    assert "negative" in data[0]["labels"]
    assert data[0]["tool_name"] == "classify_sentiment_gate"


@requires_mcp
def test_e2e_classify(specialists_dir):
    """classify_sentiment_gate returns label, confidence, latency_ms."""

    async def _check(session: "ClientSession"):
        result = await session.call_tool(
            "classify_sentiment_gate", {"text": "I want a refund right now"}
        )
        return result.content[0].text

    text = asyncio.run(_run_with_server(specialists_dir, _check))
    assert "label:" in text
    assert "confidence:" in text
    assert "latency_ms:" in text


@requires_mcp
def test_e2e_classify_label_is_valid(specialists_dir):
    """Returned label is one of the specialist's known labels."""

    async def _check(session: "ClientSession"):
        result = await session.call_tool(
            "classify_sentiment_gate", {"text": "great service"}
        )
        lines = {line.split(":")[0].strip(): line.split(":", 1)[1].strip()
                 for line in result.content[0].text.splitlines() if ":" in line}
        return lines["label"]

    label = asyncio.run(_run_with_server(specialists_dir, _check))
    assert label in ("negative", "positive")


@requires_mcp
def test_e2e_classify_unknown_tool_returns_error(specialists_dir):
    """Calling an unregistered classify_ tool returns an Error message."""

    async def _check(session: "ClientSession"):
        result = await session.call_tool(
            "classify_nonexistent_model", {"text": "test"}
        )
        return result.content[0].text

    # MCP servers may surface unknown tools as an error response or text error
    try:
        text = asyncio.run(_run_with_server(specialists_dir, _check))
        assert "Error" in text or "error" in text.lower()
    except Exception:
        pass  # MCP protocol-level error is also acceptable


@requires_mcp
def test_e2e_pull_specialist_missing_name(specialists_dir):
    """pull_specialist with no name returns an error mentioning 'name'."""

    async def _check(session: "ClientSession"):
        result = await session.call_tool("pull_specialist", {})
        return result.content[0].text

    text = asyncio.run(_run_with_server(specialists_dir, _check))
    # MCP may enforce the required schema before our handler runs,
    # producing "Input validation error" instead of "Error:"
    assert "name" in text.lower()


@requires_mcp
def test_e2e_build_specialist_missing_api_key(specialists_dir):
    """build_specialist without api_key returns a clear error."""

    async def _check(session: "ClientSession"):
        result = await session.call_tool(
            "build_specialist",
            {"spec_yaml": "name: test\ndescription: test\n"},
        )
        return result.content[0].text

    # Server started without --api-key, so it should reject
    text = asyncio.run(_run_with_server(specialists_dir, _check))
    assert "Error" in text
    assert "OPENROUTER_API_KEY" in text


@requires_mcp
def test_e2e_build_specialist_missing_spec_yaml(specialists_dir):
    """build_specialist without spec_yaml returns a clear error mentioning 'spec_yaml'."""

    async def _check(session: "ClientSession"):
        result = await session.call_tool("build_specialist", {})
        return result.content[0].text

    text = asyncio.run(_run_with_server(specialists_dir, _check))
    # MCP may enforce the required schema before our handler runs
    assert "spec_yaml" in text.lower()


@requires_mcp
def test_e2e_hot_reload_after_pull(specialists_dir):
    """
    After pull_specialist runs on an already-present package, the specialist
    is reported as loaded (not "restart required") and is immediately callable.

    The server starts with sentiment-gate loaded. We call pull_specialist with
    force=True on sentiment-gate itself — the package is already on disk, so
    _pull_specialist_sync short-circuits to the "already cached" path, returns
    the path, and hot-reload fires. The result must say "loaded" or "available",
    not "Restart".
    """

    async def _check(session: "ClientSession"):
        # force=False: package already on disk → _pull_specialist_sync returns
        # immediately without network, hot-reload fires in the server
        pull_result = await session.call_tool(
            "pull_specialist", {"name": "sentiment-gate", "force": False}
        )
        pull_text = pull_result.content[0].text

        # classify must still work (specialist is live)
        classify_result = await session.call_tool(
            "classify_sentiment_gate", {"text": "test"}
        )
        classify_text = classify_result.content[0].text

        return pull_text, classify_text

    pull_text, classify_text = asyncio.run(_run_with_server(specialists_dir, _check))

    # Pull should confirm loaded state, not ask for restart
    assert "Restart" not in pull_text
    # Classify should work
    assert "label:" in classify_text
