"""
Measure how many frontier model calls Crasis eliminates on a classification-heavy workload.

Baseline: every message goes directly to the frontier model for classification.
Crasis-first: run the local specialist first; only escalate to the frontier model
              if specialist confidence < threshold.

Usage:
    python scripts/benchmark_agent_calls.py \
        --messages ./data/benchmark_messages.jsonl \
        --specialist ./models/email-urgency-onnx \
        --model claude-haiku \
        --threshold 0.70

Output:
    Baseline:      100 frontier calls, 12,400 tokens, ~$0.062, 47.3s
    Crasis-first:  18 frontier calls, 2,232 tokens, ~$0.011, 4.8s
    Reduction:     82% fewer calls, 82% fewer tokens, 82% cost reduction
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Cost per 1M tokens (input) for common models used as lightweight classifiers
# Update these as pricing changes
MODEL_INPUT_COST_PER_1M = {
    "claude-haiku": 0.25,
    "claude-haiku-4-5": 0.25,
    "gpt-4o-mini": 0.15,
    "gpt-4o-mini-2024-07-18": 0.15,
}

# Approximate tokens per classification request (system prompt + message + response)
TOKENS_PER_CLASSIFICATION = 124


@dataclass
class BenchmarkResult:
    frontier_calls: int
    specialist_calls: int
    total_tokens: int
    estimated_cost_usd: float
    total_latency_s: float
    escalations: list[dict] = field(default_factory=list)


def load_messages(path: Path) -> list[str]:
    """Load messages from a JSONL file. Each line must have a 'text' field."""
    messages = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" not in obj:
                    log.warning("Line %d missing 'text' field, skipping", line_num)
                    continue
                messages.append(obj["text"])
            except json.JSONDecodeError as e:
                log.warning("Line %d invalid JSON: %s", line_num, e)
    return messages


def estimate_cost(tokens: int, model: str) -> float:
    """Estimate cost in USD for a given token count and model."""
    cost_per_1m = MODEL_INPUT_COST_PER_1M.get(model, 0.25)
    return (tokens / 1_000_000) * cost_per_1m


def run_baseline(
    messages: list[str],
    model: str,
    api_key: Optional[str],
    provider: str,
    dry_run: bool,
) -> BenchmarkResult:
    """Send every message directly to the frontier model."""
    log.info("Running baseline: %d messages → frontier model (%s)", len(messages), model)

    if dry_run:
        log.info("Dry run — simulating frontier model latency (150ms/call)")
        total_tokens = len(messages) * TOKENS_PER_CLASSIFICATION
        total_latency = len(messages) * 0.150
        return BenchmarkResult(
            frontier_calls=len(messages),
            specialist_calls=0,
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost(total_tokens, model),
            total_latency_s=total_latency,
        )

    if provider == "anthropic":
        result = _baseline_anthropic(messages, model, api_key)
    elif provider == "openai":
        result = _baseline_openai(messages, model, api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return result


def _baseline_anthropic(messages: list[str], model: str, api_key: Optional[str]) -> BenchmarkResult:
    try:
        import anthropic
    except ImportError:
        log.error("pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    total_tokens = 0
    start = time.perf_counter()

    for msg in messages:
        response = client.messages.create(
            model=model,
            max_tokens=32,
            messages=[{
                "role": "user",
                "content": f"Classify this message with a single word label. Message: {msg}"
            }]
        )
        total_tokens += response.usage.input_tokens + response.usage.output_tokens

    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        frontier_calls=len(messages),
        specialist_calls=0,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost(total_tokens, model),
        total_latency_s=elapsed,
    )


def _baseline_openai(messages: list[str], model: str, api_key: Optional[str]) -> BenchmarkResult:
    try:
        import openai
    except ImportError:
        log.error("pip install openai")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    total_tokens = 0
    start = time.perf_counter()

    for msg in messages:
        response = client.chat.completions.create(
            model=model,
            max_tokens=32,
            messages=[{
                "role": "user",
                "content": f"Classify this message with a single word label. Message: {msg}"
            }]
        )
        total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens

    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        frontier_calls=len(messages),
        specialist_calls=0,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost(total_tokens, model),
        total_latency_s=elapsed,
    )


def run_crasis_first(
    messages: list[str],
    specialist_path: str,
    model: str,
    api_key: Optional[str],
    provider: str,
    threshold: float,
    dry_run: bool,
) -> BenchmarkResult:
    """Run local specialist first; escalate to frontier only when confidence < threshold."""
    from crasis import Specialist

    log.info(
        "Running Crasis-first: %d messages, specialist at %s, threshold %.2f",
        len(messages), specialist_path, threshold
    )

    specialist = Specialist.load(specialist_path)
    frontier_calls = 0
    specialist_calls = 0
    total_tokens = 0
    escalations = []
    start = time.perf_counter()

    for msg in messages:
        spec_result = specialist.classify(msg)
        specialist_calls += 1

        if spec_result["confidence"] >= threshold:
            # Specialist confident — no frontier call needed
            continue

        # Low confidence — escalate
        escalations.append({
            "text": msg[:80],
            "specialist_label": spec_result["label"],
            "specialist_confidence": spec_result["confidence"],
        })

        if dry_run:
            total_tokens += TOKENS_PER_CLASSIFICATION
            frontier_calls += 1
            continue

        if provider == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model,
                    max_tokens=32,
                    messages=[{
                        "role": "user",
                        "content": f"Classify this message with a single word label. Message: {msg}"
                    }]
                )
                total_tokens += response.usage.input_tokens + response.usage.output_tokens
            except ImportError:
                log.error("pip install anthropic")
                sys.exit(1)
        elif provider == "openai":
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=32,
                    messages=[{
                        "role": "user",
                        "content": f"Classify this message with a single word label. Message: {msg}"
                    }]
                )
                total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
            except ImportError:
                log.error("pip install openai")
                sys.exit(1)

        frontier_calls += 1

    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        frontier_calls=frontier_calls,
        specialist_calls=specialist_calls,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost(total_tokens, model),
        total_latency_s=elapsed,
        escalations=escalations,
    )


def print_comparison(
    baseline: BenchmarkResult,
    crasis: BenchmarkResult,
    n: int,
    model: str,
    threshold: float,
) -> None:
    def pct_reduction(a: float, b: float) -> str:
        if a == 0:
            return "N/A"
        return f"{(a - b) / a * 100:.0f}%"

    print()
    print(f"Messages:      {n}")
    print(f"Model:         {model}")
    print(f"Threshold:     {threshold:.2f} (escalate if specialist confidence < this)")
    print()
    print(f"{'':20s} {'Baseline':>12s}  {'Crasis-first':>12s}  {'Reduction':>10s}")
    print(f"{'':20s} {'--------':>12s}  {'------------':>12s}  {'---------':>10s}")
    print(
        f"{'Frontier calls':20s} {baseline.frontier_calls:>12d}  "
        f"{crasis.frontier_calls:>12d}  "
        f"{pct_reduction(baseline.frontier_calls, crasis.frontier_calls):>10s}"
    )
    print(
        f"{'Tokens':20s} {baseline.total_tokens:>12,d}  "
        f"{crasis.total_tokens:>12,d}  "
        f"{pct_reduction(baseline.total_tokens, crasis.total_tokens):>10s}"
    )
    print(
        f"{'Est. cost (USD)':20s} {'${:.4f}'.format(baseline.estimated_cost_usd):>12s}  "
        f"{'${:.4f}'.format(crasis.estimated_cost_usd):>12s}  "
        f"{pct_reduction(baseline.estimated_cost_usd, crasis.estimated_cost_usd):>10s}"
    )
    print(
        f"{'Latency (s)':20s} {baseline.total_latency_s:>12.2f}  "
        f"{crasis.total_latency_s:>12.2f}  "
        f"{pct_reduction(baseline.total_latency_s, crasis.total_latency_s):>10s}"
    )
    print()

    if crasis.escalations:
        print(f"Escalated {len(crasis.escalations)} messages (confidence < {threshold:.2f}):")
        for e in crasis.escalations[:5]:
            print(f"  [{e['specialist_label']} @ {e['specialist_confidence']:.2f}] {e['text']}")
        if len(crasis.escalations) > 5:
            print(f"  ... and {len(crasis.escalations) - 5} more")
    else:
        print("No escalations — specialist was confident on all messages.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure frontier model call reduction from Crasis specialist-first routing"
    )
    parser.add_argument(
        "--messages",
        required=True,
        help="Path to JSONL file with messages (each line: {\"text\": \"...\"})",
    )
    parser.add_argument(
        "--specialist",
        required=True,
        help="Path to loaded ONNX specialist directory (e.g. ./models/email-urgency-onnx)",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku",
        help="Frontier model to use for baseline and escalations (default: claude-haiku)",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="API provider (default: anthropic)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Escalate to frontier model if specialist confidence < threshold (default: 0.70)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate frontier model calls (no real API calls). Uses estimated token counts.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write raw results to a JSON file",
    )

    args = parser.parse_args()

    messages_path = Path(args.messages)
    if not messages_path.exists():
        log.error("Messages file not found: %s", messages_path)
        sys.exit(1)

    messages = load_messages(messages_path)
    if not messages:
        log.error("No messages loaded from %s", messages_path)
        sys.exit(1)

    log.info("Loaded %d messages", len(messages))

    baseline = run_baseline(
        messages=messages,
        model=args.model,
        api_key=args.api_key,
        provider=args.provider,
        dry_run=args.dry_run,
    )

    crasis = run_crasis_first(
        messages=messages,
        specialist_path=args.specialist,
        model=args.model,
        api_key=args.api_key,
        provider=args.provider,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )

    print_comparison(
        baseline=baseline,
        crasis=crasis,
        n=len(messages),
        model=args.model,
        threshold=args.threshold,
    )

    if args.output_json:
        output = {
            "messages": len(messages),
            "model": args.model,
            "threshold": args.threshold,
            "dry_run": args.dry_run,
            "baseline": {
                "frontier_calls": baseline.frontier_calls,
                "total_tokens": baseline.total_tokens,
                "estimated_cost_usd": baseline.estimated_cost_usd,
                "total_latency_s": baseline.total_latency_s,
            },
            "crasis_first": {
                "frontier_calls": crasis.frontier_calls,
                "specialist_calls": crasis.specialist_calls,
                "total_tokens": crasis.total_tokens,
                "estimated_cost_usd": crasis.estimated_cost_usd,
                "total_latency_s": crasis.total_latency_s,
                "escalations": len(crasis.escalations),
            },
            "reduction": {
                "frontier_calls_pct": round(
                    (baseline.frontier_calls - crasis.frontier_calls)
                    / baseline.frontier_calls * 100, 1
                ) if baseline.frontier_calls else 0,
                "tokens_pct": round(
                    (baseline.total_tokens - crasis.total_tokens)
                    / baseline.total_tokens * 100, 1
                ) if baseline.total_tokens else 0,
                "cost_pct": round(
                    (baseline.estimated_cost_usd - crasis.estimated_cost_usd)
                    / baseline.estimated_cost_usd * 100, 1
                ) if baseline.estimated_cost_usd else 0,
                "latency_pct": round(
                    (baseline.total_latency_s - crasis.total_latency_s)
                    / baseline.total_latency_s * 100, 1
                ) if baseline.total_latency_s else 0,
            },
        }
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        log.info("Results written to %s", args.output_json)


if __name__ == "__main__":
    main()
