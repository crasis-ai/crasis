"""
crasis.factory — Synthetic training data generation via OpenRouter.

ALL data generation routes through OpenRouter with enforce_distillable_text: true.
This is the legal and compliance layer. It is never bypassed.

Do not call Anthropic, OpenAI, or Google APIs directly from this module.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from crasis.spec import CrasisSpec, TaskType

logger = logging.getLogger(__name__)

# Generator models per task type, selected by benchmarking parse reliability
# and latency against the distillable endpoint (scripts/benchmark_generators.py).
#
# Verified distillable via enforce_distillable_text: true — models that appear
# in OpenRouter's distillable filter but return 404 at call time are excluded.
#
# Results (3 calls x 10 examples, 2025-03-05):
#   meta-llama/llama-3.3-70b-instruct      binary=100%/8.2s  multiclass=100%/8.8s
#   deepseek/deepseek-chat-v3.1            binary=100%/4.1s  multiclass=67%/8.3s
#   mistralai/mistral-small-3.2-24b-instruct binary=100%/15s multiclass=100%/7.0s
#   qwen/qwen3-32b                         binary=100%/37s   multiclass=100%/46s  (too slow)
#   google/gemma-3-27b-it                  404 — not actually distillable
#   google/gemini-2.0-flash-lite-001       404 — not actually distillable
#   openai/gpt-4.1-nano                    404 — not actually distillable
_GENERATOR_MODELS: dict[TaskType, str] = {
    TaskType.binary_classification: "deepseek/deepseek-chat-v3.1",
    TaskType.multiclass:            "meta-llama/llama-3.3-70b-instruct",
    TaskType.extraction:            "meta-llama/llama-3.3-70b-instruct",
    TaskType.sequence:              "meta-llama/llama-3.3-70b-instruct",
}

# Examples requested per API call. Higher = fewer calls, larger responses.
_BATCH_SIZE = 50

# Concurrent API calls. OpenRouter handles parallel requests well.
_WORKERS = 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(
    spec: CrasisSpec,
    output_dir: str | Path,
    api_key: str,
    count: int | None = None,
    resume: bool = True,
) -> Path:
    """
    Generate synthetic training data for a specialist from a validated spec.

    Calls OpenRouter exclusively, with enforce_distillable_text: true.
    Outputs a JSONL file at output_dir/<spec.name>/train.jsonl.

    Args:
        spec: Validated CrasisSpec instance.
        output_dir: Root data directory. Subdirectory is created automatically.
        api_key: OpenRouter API key.
        count: Override for sample count. Defaults to spec.training.volume.
        resume: If True and a partial output file exists, resume from where
                generation left off rather than starting over.

    Returns:
        Path to the generated JSONL file.

    Raises:
        RuntimeError: If the OpenRouter API returns unexpected responses.
    """
    target_count = count or spec.training.volume
    out_dir = Path(output_dir) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    # Resume logic: count existing valid lines; truncate if not resuming
    if not resume and out_path.exists():
        out_path.unlink()
    existing = _count_existing(out_path) if resume and out_path.exists() else 0
    if existing >= target_count:
        logger.info("Data already complete: %d samples at %s", existing, out_path)
        return out_path

    remaining = target_count - existing
    logger.info(
        "Generating %d samples for '%s' (have %d, need %d total)",
        remaining,
        spec.name,
        existing,
        target_count,
    )

    client = _make_client(api_key)
    prompt_builder = _prompt_builder_for(spec)

    # Build the list of batch sizes to dispatch. Each worker gets one batch job.
    batch_sizes: list[int] = []
    todo = remaining
    while todo > 0:
        batch_sizes.append(min(_BATCH_SIZE, todo))
        todo -= batch_sizes[-1]

    write_lock = threading.Lock()
    generated = 0

    def _fetch(batch_n: int) -> list[dict]:
        """Fetch one batch with retry, returning examples."""
        while True:
            try:
                return _generate_batch(client, spec, prompt_builder, batch_n)
            except Exception as exc:
                logger.warning("Batch failed, retrying after 5s: %s", exc)
                time.sleep(5)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Generating training data"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("generate", total=remaining)

        with out_path.open("a", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
                futures = {pool.submit(_fetch, n): n for n in batch_sizes}
                for future in as_completed(futures):
                    examples = future.result()
                    with write_lock:
                        for ex in examples:
                            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        fh.flush()
                    generated += len(examples)
                    progress.advance(task, len(examples))

    total = existing + generated
    logger.info("Generation complete: %d samples → %s", total, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_client(api_key: str) -> OpenAI:
    """Create an OpenAI-compatible client pointed at OpenRouter."""
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _generate_batch(
    client: OpenAI,
    spec: CrasisSpec,
    prompt_builder: "_PromptBuilder",
    n: int,
) -> list[dict]:
    """
    Request a batch of labeled examples from OpenRouter.

    Uses enforce_distillable_text: true — this is non-negotiable.
    Model is selected per task type from _GENERATOR_MODELS.
    """
    system_prompt, user_prompt = prompt_builder.build(n)
    model = _GENERATOR_MODELS[spec.task.type]

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={
            "provider": {
                "enforce_distillable_text": True,
            }
        },
    )

    if not response.choices:
        raise RuntimeError(
            "OpenRouter returned empty choices — likely a rate limit or provider error"
        )
    raw = response.choices[0].message.content or ""
    return _parse_batch_response(raw, spec)


def _parse_batch_response(raw: str, spec: CrasisSpec) -> list[dict]:
    """
    Extract labeled examples from the model's response.

    Expects the model to return a JSON array of objects with 'text' and 'label'.
    Gracefully handles partial or malformed responses.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = (
            "\n".join(lines[1:-1])
            if lines[-1].strip() == "```"
            else "\n".join(lines[1:])
        )

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse batch response as JSON: %s", exc)
        return []

    valid = []
    valid_labels = set(spec.label_names)

    for item in parsed:
        if not isinstance(item, dict):
            continue
        text_val = item.get("text", "").strip()
        label_val = str(item.get("label", "")).strip().lower()

        if not text_val or label_val not in valid_labels:
            continue

        valid.append(
            {
                "text": text_val,
                "label": label_val,
                "label_id": spec.label_names.index(label_val),
            }
        )

    return valid


def _count_existing(path: Path) -> int:
    """Count valid JSONL lines in an existing output file."""
    count = 0
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


class _PromptBuilder:
    """Base class for spec-specific prompt construction."""

    def __init__(self, spec: CrasisSpec) -> None:
        self.spec = spec

    def build(self, n: int) -> tuple[str, str]:
        raise NotImplementedError


class _BinaryPromptBuilder(_PromptBuilder):
    """Prompt builder for binary classification tasks."""

    def build(self, n: int) -> tuple[str, str]:
        spec = self.spec
        half = n // 2
        positive_n = half
        negative_n = n - half  # handles odd n

        system = (
            "You are a training data generator for a text classification specialist model. "
            "You produce realistic, diverse, labeled text examples in JSON format. "
            "Your output must be valid JSON only — no explanation, no markdown, no preamble."
        )

        user = f"""Generate {n} labeled text examples for a binary text classifier.

Specialist description: {spec.description}

POSITIVE examples (label: "positive"):
- Definition: {spec.task.trigger}
- Generate {positive_n} positive examples

NEGATIVE examples (label: "negative"):
- Definition: What does NOT trigger this classifier
{f'- Explicitly exclude: {spec.task.ignore}' if spec.task.ignore else ''}
- Generate {negative_n} negative examples

Requirements:
- Vary length: mix short (1 sentence) and longer (3-5 sentences) examples
- Vary tone: formal, casual, frustrated, polite
- Make negatives realistic — they should look plausible but genuinely not match the trigger
- No duplicate or near-duplicate examples
- Each example should be a standalone message or text snippet (not a title or fragment)

Return ONLY a JSON array, no other text:
[
  {{"text": "example text here", "label": "positive"}},
  {{"text": "example text here", "label": "negative"}}
]

Generate exactly {n} examples now:"""

        return system, user


class _MulticlassPromptBuilder(_PromptBuilder):
    """Prompt builder for multiclass tasks."""

    def build(self, n: int) -> tuple[str, str]:
        spec = self.spec
        classes = spec.task.classes or []
        per_class = max(1, n // len(classes))

        class_descriptions = "\n".join(
            f'- "{cls}": {n_} examples'
            for cls, n_ in zip(classes, [per_class] * len(classes))
        )

        system = (
            "You are a training data generator for a text classification specialist model. "
            "Output valid JSON only — no explanation, no markdown."
        )

        user = f"""Generate {n} labeled text examples for a multiclass classifier.

Specialist description: {spec.description}
Trigger: {spec.task.trigger}

Classes and target counts:
{class_descriptions}

Return ONLY a JSON array:
[{{"text": "...", "label": "<class_name>"}}, ...]

Generate exactly {n} examples now:"""

        return system, user


def _prompt_builder_for(spec: CrasisSpec) -> _PromptBuilder:
    """Return the appropriate prompt builder for this spec's task type."""
    if spec.task.type == TaskType.binary_classification:
        return _BinaryPromptBuilder(spec)
    if spec.task.type == TaskType.multiclass:
        return _MulticlassPromptBuilder(spec)
    raise NotImplementedError(
        f"Data generation for task type '{spec.task.type}' is not yet implemented. "
        "Binary and multiclass are supported in v1."
    )
