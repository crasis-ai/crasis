"""
scripts/benchmark_generators.py

Benchmark candidate data-generation models for JSON parse reliability,
output diversity, and cost per example.

Usage:
    source /home/aoaustin/trainonce-pyenv/bin/activate
    python scripts/benchmark_generators.py --api-key $OPENROUTER_API_KEY

    # Test a specific subset of models:
    python scripts/benchmark_generators.py --api-key $OPENROUTER_API_KEY \\
        --models google/gemma-3-27b-it mistralai/mistral-small-3.2-24b-instruct

    # Change batch size or call count:
    python scripts/benchmark_generators.py --api-key $OPENROUTER_API_KEY \\
        --calls 3 --batch-size 10

Results are printed as a ranked table. The winner for each task type is
the model with the highest parse success rate at the lowest cost per
successfully parsed example.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from crasis.factory import (
    _make_client,
    _parse_batch_response,
    _BinaryPromptBuilder,
    _MulticlassPromptBuilder,
)
from crasis.spec import CrasisSpec

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate models to benchmark (edit this list freely)
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    # Current default
    "meta-llama/llama-3.3-70b-instruct",
    # Cheaper candidates
    "google/gemma-3-27b-it",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3-32b",
    "google/gemini-2.0-flash-lite-001",
    # Mid-tier
    "deepseek/deepseek-chat-v3.1",
    "openai/gpt-4.1-nano",
]

# ---------------------------------------------------------------------------
# Benchmark fixtures — one spec per task type
# ---------------------------------------------------------------------------

BINARY_SPEC_DATA = {
    "crasis_spec": "v1",
    "name": "whatsapp-triage",
    "description": (
        "Classifies incoming WhatsApp messages as requiring an immediate human "
        "response versus routine or automated messages that can be handled later."
    ),
    "task": {
        "type": "binary_classification",
        "trigger": (
            "Message requires urgent human attention — customer is angry, requesting "
            "action, has a time-sensitive problem, or has escalated."
        ),
        "ignore": (
            "Automated notifications, delivery receipts, greetings without requests, "
            "spam, subscription confirmations."
        ),
    },
    "quality": {"min_accuracy": 0.93},
    "training": {"volume": 3000},
}

MULTICLASS_SPEC_DATA = {
    "crasis_spec": "v1",
    "name": "email-urgency",
    "description": (
        "Classifies incoming emails by urgency level so that critical messages are "
        "surfaced immediately and low-priority noise stays buried."
    ),
    "task": {
        "type": "multiclass",
        "trigger": (
            "Any email that can be meaningfully categorized by how quickly it demands "
            "attention or action from the recipient."
        ),
        "ignore": "Encrypted or garbled content, empty messages.",
        "classes": ["critical", "high", "normal", "low"],
    },
    "quality": {"min_accuracy": 0.91},
    "training": {"volume": 5000},
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    model_id: str
    task_type: str
    calls: int = 0
    successful_parses: int = 0
    total_examples_returned: int = 0
    total_examples_requested: int = 0
    errors: list[str] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    # Populated from OpenRouter usage headers / response if available
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0

    @property
    def parse_success_rate(self) -> float:
        return self.successful_parses / self.calls if self.calls else 0.0

    @property
    def example_yield_rate(self) -> float:
        """Fraction of requested examples actually returned and valid."""
        return (
            self.total_examples_returned / self.total_examples_requested
            if self.total_examples_requested
            else 0.0
        )

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def _run_one_call(
    client,
    model_id: str,
    spec: CrasisSpec,
    builder,
    batch_size: int,
    result: ModelResult,
) -> None:
    """Make one batch call, record parse success and latency."""
    system_prompt, user_prompt = builder.build(batch_size)

    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_id,
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
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result.calls += 1
        result.errors.append(str(exc)[:120])
        result.latencies_ms.append(elapsed_ms)
        return

    elapsed_ms = (time.perf_counter() - t0) * 1000
    result.calls += 1
    result.latencies_ms.append(elapsed_ms)

    if response.usage:
        result.prompt_tokens_used += response.usage.prompt_tokens or 0
        result.completion_tokens_used += response.usage.completion_tokens or 0

    raw = (response.choices[0].message.content or "") if response.choices else ""
    if not raw:
        result.errors.append("empty response")
        return

    examples = _parse_batch_response(raw, spec)
    result.total_examples_requested += batch_size
    result.total_examples_returned += len(examples)

    if examples:
        result.successful_parses += 1
    else:
        # Capture first 200 chars of raw response to diagnose failures
        result.errors.append(f"parse_fail: {raw[:200]!r}")


def benchmark_model(
    model_id: str,
    api_key: str,
    n_calls: int,
    batch_size: int,
) -> list[ModelResult]:
    """
    Run n_calls batches for each task type fixture against model_id.
    Returns one ModelResult per task type.
    """
    client = _make_client(api_key)
    results = []

    fixtures = [
        ("binary", BINARY_SPEC_DATA),
        ("multiclass", MULTICLASS_SPEC_DATA),
    ]

    for task_label, spec_data in fixtures:
        spec = CrasisSpec.model_validate(spec_data)
        builder = (
            _BinaryPromptBuilder(spec)
            if task_label == "binary"
            else _MulticlassPromptBuilder(spec)
        )
        result = ModelResult(model_id=model_id, task_type=task_label)

        print(f"  [{task_label}] {model_id} ...", end="", flush=True)
        for i in range(n_calls):
            _run_one_call(client, model_id, spec, builder, batch_size, result)
            print(".", end="", flush=True)

        print(
            f" parse={result.parse_success_rate:.0%}"
            f" yield={result.example_yield_rate:.0%}"
            f" avg={result.avg_latency_ms:.0f}ms"
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(all_results: list[ModelResult]) -> None:
    header = (
        f"{'Model':<55} {'Task':<12} {'Parse%':>7} {'Yield%':>7} "
        f"{'AvgMs':>7} {'Errors':>6}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    # Sort: task type, then parse success desc, then latency asc
    sorted_results = sorted(
        all_results,
        key=lambda r: (r.task_type, -r.parse_success_rate, r.avg_latency_ms),
    )

    current_task = None
    for r in sorted_results:
        if r.task_type != current_task:
            if current_task is not None:
                print()
            current_task = r.task_type

        model_short = r.model_id
        print(
            f"{model_short:<55} {r.task_type:<12} "
            f"{r.parse_success_rate:>6.0%}  "
            f"{r.example_yield_rate:>6.0%}  "
            f"{r.avg_latency_ms:>6.0f}  "
            f"{len(r.errors):>5}"
        )

    print(sep)
    print()

    # Highlight winner per task type
    by_task: dict[str, list[ModelResult]] = {}
    for r in all_results:
        by_task.setdefault(r.task_type, []).append(r)

    print("Recommended defaults:")
    for task_type, results in sorted(by_task.items()):
        # Best = highest parse rate, then lowest latency
        best = max(results, key=lambda r: (r.parse_success_rate, -r.avg_latency_ms))
        print(f"  {task_type:<14} -> {best.model_id}  (parse={best.parse_success_rate:.0%}, avg={best.avg_latency_ms:.0f}ms)")
    print()

    # Print any error details
    errors_seen = [(r, e) for r in all_results for e in r.errors]
    if errors_seen:
        print("Error details (first per model):")
        seen_models = set()
        for r, e in errors_seen:
            key = (r.model_id, r.task_type)
            if key not in seen_models:
                seen_models.add(key)
                print(f"  {r.model_id} [{r.task_type}]: {e}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark distillable generator models")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="MODEL_ID",
        help="Model IDs to benchmark (default: hardcoded candidate list)",
    )
    parser.add_argument(
        "--calls",
        type=int,
        default=5,
        help="Number of API calls per model per task type (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Examples requested per call (default: 10)",
    )
    args = parser.parse_args()

    print(f"Benchmarking {len(args.models)} model(s), {args.calls} calls x {args.batch_size} examples each")
    print(f"Task types: binary_classification, multiclass")
    print(f"Total API calls: {len(args.models) * 2 * args.calls}")
    print()

    all_results: list[ModelResult] = []
    for model_id in args.models:
        print(f"Model: {model_id}")
        results = benchmark_model(model_id, args.api_key, args.calls, args.batch_size)
        all_results.extend(results)

    _print_table(all_results)


if __name__ == "__main__":
    main()
