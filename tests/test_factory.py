"""Tests for crasis.factory — synthetic data generation pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crasis.factory import (
    _BinaryPromptBuilder,
    _MulticlassPromptBuilder,
    _count_existing,
    _generate_batch,
    _make_client,
    _parse_batch_response,
    _prompt_builder_for,
    generate,
)
from crasis.spec import CrasisSpec, TaskType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BINARY_SPEC = CrasisSpec.model_validate(
    {
        "name": "refund-detector",
        "description": "Detects refund requests in customer support emails.",
        "task": {
            "type": "binary_classification",
            "trigger": "customer explicitly asks for a refund",
            "ignore": "general complaints without a refund request",
        },
        "quality": {"min_accuracy": 0.93},
        "training": {"volume": 100},
    }
)

MULTICLASS_SPEC = CrasisSpec.model_validate(
    {
        "name": "support-router",
        "description": "Routes support tickets to the correct team.",
        "task": {
            "type": "multiclass",
            "trigger": "any customer support message",
            "ignore": "spam",
            "classes": ["billing", "technical", "returns", "general"],
        },
        "quality": {"min_accuracy": 0.88},
        "training": {"volume": 200},
    }
)


def _make_mock_response(content: str) -> MagicMock:
    """Build a mock that looks like an OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _binary_batch_json(n: int = 4) -> str:
    examples = []
    for i in range(n):
        label = "positive" if i % 2 == 0 else "negative"
        examples.append({"text": f"example text {i}", "label": label})
    return json.dumps(examples)


def _multiclass_batch_json(n: int = 4, classes=None) -> str:
    classes = classes or ["billing", "technical", "returns", "general"]
    examples = [{"text": f"example {i}", "label": classes[i % len(classes)]} for i in range(n)]
    return json.dumps(examples)


# ---------------------------------------------------------------------------
# _make_client
# ---------------------------------------------------------------------------


def test_make_client_points_at_openrouter():
    client = _make_client("test-key")
    assert "openrouter" in str(client.base_url).lower()


def test_make_client_uses_provided_key():
    client = _make_client("sk-or-test-123")
    assert client.api_key == "sk-or-test-123"


# ---------------------------------------------------------------------------
# _count_existing
# ---------------------------------------------------------------------------


def test_count_existing_empty_file(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("", encoding="utf-8")
    assert _count_existing(p) == 0


def test_count_existing_counts_lines(tmp_path):
    p = tmp_path / "train.jsonl"
    lines = [json.dumps({"text": f"t{i}", "label": "positive", "label_id": 1}) for i in range(7)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _count_existing(p) == 7


def test_count_existing_missing_file(tmp_path):
    assert _count_existing(tmp_path / "nonexistent.jsonl") == 0


def test_count_existing_ignores_blank_lines(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"text":"a","label":"positive","label_id":1}\n\n{"text":"b","label":"negative","label_id":0}\n', encoding="utf-8")
    assert _count_existing(p) == 2


# ---------------------------------------------------------------------------
# _parse_batch_response
# ---------------------------------------------------------------------------


def test_parse_valid_binary_response():
    raw = _binary_batch_json(4)
    results = _parse_batch_response(raw, BINARY_SPEC)
    assert len(results) == 4
    for r in results:
        assert "text" in r
        assert r["label"] in ("positive", "negative")
        assert r["label_id"] in (0, 1)


def test_parse_strips_markdown_fences():
    raw = "```json\n" + _binary_batch_json(2) + "\n```"
    results = _parse_batch_response(raw, BINARY_SPEC)
    assert len(results) == 2


def test_parse_strips_fences_no_language_tag():
    raw = "```\n" + _binary_batch_json(2) + "\n```"
    results = _parse_batch_response(raw, BINARY_SPEC)
    assert len(results) == 2


def test_parse_invalid_json_returns_empty():
    results = _parse_batch_response("this is not json", BINARY_SPEC)
    assert results == []


def test_parse_wrong_json_type_returns_empty():
    results = _parse_batch_response('{"text": "x", "label": "positive"}', BINARY_SPEC)
    assert results == []


def test_parse_filters_invalid_labels():
    raw = json.dumps([
        {"text": "valid", "label": "positive"},
        {"text": "bad label", "label": "unknown"},
        {"text": "empty label", "label": ""},
    ])
    results = _parse_batch_response(raw, BINARY_SPEC)
    assert len(results) == 1
    assert results[0]["label"] == "positive"


def test_parse_filters_empty_text():
    raw = json.dumps([
        {"text": "", "label": "positive"},
        {"text": "   ", "label": "negative"},
        {"text": "good text", "label": "positive"},
    ])
    results = _parse_batch_response(raw, BINARY_SPEC)
    assert len(results) == 1


def test_parse_label_id_correct_for_binary():
    raw = json.dumps([
        {"text": "refund me", "label": "positive"},
        {"text": "just browsing", "label": "negative"},
    ])
    results = _parse_batch_response(raw, BINARY_SPEC)
    assert results[0]["label_id"] == BINARY_SPEC.label_names.index("positive")
    assert results[1]["label_id"] == BINARY_SPEC.label_names.index("negative")


def test_parse_multiclass_response():
    raw = _multiclass_batch_json(4)
    results = _parse_batch_response(raw, MULTICLASS_SPEC)
    assert len(results) == 4
    for r in results:
        assert r["label"] in MULTICLASS_SPEC.label_names


# ---------------------------------------------------------------------------
# enforce_distillable_text is always sent
# ---------------------------------------------------------------------------


def test_generate_batch_sends_enforce_distillable_text():
    """enforce_distillable_text: True must be in every API call. Non-negotiable."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(_binary_batch_json(4))

    builder = _BinaryPromptBuilder(BINARY_SPEC)
    _generate_batch(mock_client, BINARY_SPEC, builder, 4)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"]["provider"]["enforce_distillable_text"] is True


def test_generate_batch_uses_openrouter_model():
    """Must use the configured generator model, never a direct provider model."""
    from crasis.factory import _GENERATOR_MODELS
    from crasis.spec import TaskType

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(_binary_batch_json(4))

    builder = _BinaryPromptBuilder(BINARY_SPEC)
    _generate_batch(mock_client, BINARY_SPEC, builder, 4)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == _GENERATOR_MODELS[TaskType.binary_classification]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def test_binary_prompt_builder_returns_two_strings():
    system, user = _BinaryPromptBuilder(BINARY_SPEC).build(10)
    assert isinstance(system, str) and len(system) > 0
    assert isinstance(user, str) and len(user) > 0


def test_binary_prompt_mentions_trigger():
    _, user = _BinaryPromptBuilder(BINARY_SPEC).build(10)
    assert BINARY_SPEC.task.trigger in user


def test_binary_prompt_mentions_ignore():
    _, user = _BinaryPromptBuilder(BINARY_SPEC).build(10)
    assert BINARY_SPEC.task.ignore in user


def test_binary_prompt_requests_correct_count():
    _, user = _BinaryPromptBuilder(BINARY_SPEC).build(20)
    assert "20" in user


def test_multiclass_prompt_mentions_all_classes():
    _, user = _MulticlassPromptBuilder(MULTICLASS_SPEC).build(20)
    for cls in MULTICLASS_SPEC.task.classes:
        assert cls in user


def test_prompt_builder_for_binary():
    builder = _prompt_builder_for(BINARY_SPEC)
    assert isinstance(builder, _BinaryPromptBuilder)


def test_prompt_builder_for_multiclass():
    builder = _prompt_builder_for(MULTICLASS_SPEC)
    assert isinstance(builder, _MulticlassPromptBuilder)


def test_prompt_builder_for_unsupported_raises():
    extraction_spec = CrasisSpec.model_validate(
        {
            "name": "meeting-parser",
            "description": "Extracts meeting details.",
            "task": {
                "type": "extraction",
                "trigger": "meeting details",
                "ignore": "non-meeting content",
            },
            "quality": {"min_accuracy": 0.90},
            "training": {"volume": 500},
        }
    )
    with pytest.raises(NotImplementedError):
        _prompt_builder_for(extraction_spec)


# ---------------------------------------------------------------------------
# generate() — full pipeline (mocked OpenRouter)
# ---------------------------------------------------------------------------


def _patch_generate_batch(spec, batch_json_fn):
    """Context manager that patches _generate_batch to return fake examples."""

    def fake_batch(client, spec_, builder, n):
        raw = batch_json_fn(n)
        return _parse_batch_response(raw, spec_)

    return patch("crasis.factory._generate_batch", side_effect=fake_batch)


def test_generate_creates_output_file(tmp_path):
    with _patch_generate_batch(BINARY_SPEC, _binary_batch_json):
        out = generate(BINARY_SPEC, tmp_path, api_key="fake-key", count=10)
    assert out.exists()
    assert out.suffix == ".jsonl"


def test_generate_writes_correct_count(tmp_path):
    with _patch_generate_batch(BINARY_SPEC, _binary_batch_json):
        out = generate(BINARY_SPEC, tmp_path, api_key="fake-key", count=20)
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 20


def test_generate_output_is_valid_jsonl(tmp_path):
    with _patch_generate_batch(BINARY_SPEC, _binary_batch_json):
        out = generate(BINARY_SPEC, tmp_path, api_key="fake-key", count=10)
    for line in out.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            assert "text" in record
            assert "label" in record
            assert "label_id" in record


def test_generate_output_under_spec_name_subdir(tmp_path):
    with _patch_generate_batch(BINARY_SPEC, _binary_batch_json):
        out = generate(BINARY_SPEC, tmp_path, api_key="fake-key", count=10)
    assert out.parent.name == BINARY_SPEC.name


def test_generate_resumes_from_existing(tmp_path):
    # Pre-populate 5 examples
    out_dir = tmp_path / BINARY_SPEC.name
    out_dir.mkdir()
    out_path = out_dir / "train.jsonl"
    existing = [
        json.dumps({"text": f"pre{i}", "label": "positive", "label_id": 1})
        for i in range(5)
    ]
    out_path.write_text("\n".join(existing) + "\n", encoding="utf-8")

    with _patch_generate_batch(BINARY_SPEC, _binary_batch_json):
        out = generate(BINARY_SPEC, tmp_path, api_key="fake-key", count=10, resume=True)

    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 10  # 5 existing + 5 new


def test_generate_no_resume_overwrites(tmp_path):
    out_dir = tmp_path / BINARY_SPEC.name
    out_dir.mkdir()
    out_path = out_dir / "train.jsonl"
    out_path.write_text(
        "\n".join(json.dumps({"text": f"old{i}", "label": "positive", "label_id": 1}) for i in range(50)) + "\n",
        encoding="utf-8",
    )

    with _patch_generate_batch(BINARY_SPEC, _binary_batch_json):
        out = generate(BINARY_SPEC, tmp_path, api_key="fake-key", count=10, resume=False)

    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 10


def test_generate_skips_when_already_complete(tmp_path):
    out_dir = tmp_path / BINARY_SPEC.name
    out_dir.mkdir()
    out_path = out_dir / "train.jsonl"
    out_path.write_text(
        "\n".join(json.dumps({"text": f"t{i}", "label": "positive", "label_id": 1}) for i in range(100)) + "\n",
        encoding="utf-8",
    )

    with patch("crasis.factory._generate_batch") as mock_batch:
        generate(BINARY_SPEC, tmp_path, api_key="fake-key", count=100, resume=True)
        mock_batch.assert_not_called()


def test_generate_retries_on_batch_failure(tmp_path):
    call_count = {"n": 0}

    def flaky_batch(client, spec_, builder, n):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated API error")
        return _parse_batch_response(_binary_batch_json(n), spec_)

    with patch("crasis.factory._generate_batch", side_effect=flaky_batch):
        with patch("crasis.factory.time.sleep"):  # don't actually sleep
            out = generate(BINARY_SPEC, tmp_path, api_key="fake-key", count=10)

    assert call_count["n"] >= 2
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 10
