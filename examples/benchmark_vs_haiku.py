"""
scripts/benchmark_vs_haiku.py — Crasis vs Haiku classification benchmark.

Measures what fraction of agent classification calls Crasis handles locally
at various confidence thresholds, and whether escalations to Haiku are justified.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/benchmark_vs_haiku.py \\
        --specialist whatsapp-triage \\
        --holdout data/whatsapp-triage/holdout.jsonl \\
        --models-dir ./models

Output:
    Threshold table (stdout)
    benchmark_results.json (full per-example results)

Holdout file format (JSONL):
    {"text": "hey are you coming tonight?", "label": "low_priority"}
    {"text": "URGENT: server is down", "label": "high_priority"}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import anthropic

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Example:
    text: str
    ground_truth: str


@dataclass
class ExampleResult:
    text: str
    ground_truth: str
    crasis_label: str
    crasis_confidence: float
    crasis_latency_ms: float
    haiku_label: str | None = None         # None if not escalated
    haiku_latency_ms: float | None = None
    haiku_tokens_used: int | None = None
    haiku_cost_usd: float | None = None


@dataclass
class ThresholdResult:
    threshold: float
    total: int
    local_count: int
    escalated_count: int
    local_pct: float
    local_accuracy: float        # Crasis accuracy on locally-handled examples
    escalation_accuracy: float   # How often Crasis was RIGHT to escalate
    overall_accuracy: float      # Combined accuracy across all examples
    total_haiku_cost_usd: float
    cost_per_1000_usd: float     # Projected cost per 1000 classifications


# ---------------------------------------------------------------------------
# Haiku classifier
# ---------------------------------------------------------------------------

# Haiku pricing as of March 2026
HAIKU_INPUT_COST_PER_TOKEN  = 0.80  / 1_000_000   # $0.80 per 1M input tokens
HAIKU_OUTPUT_COST_PER_TOKEN = 4.00  / 1_000_000   # $4.00 per 1M output tokens


def classify_with_haiku(
    client: anthropic.Anthropic,
    text: str,
    label_names: list[str],
) -> tuple[str, float, int, float]:
    """
    Classify text using Claude Haiku.

    Returns:
        (label, latency_ms, tokens_used, cost_usd)
    """
    labels_str = " | ".join(label_names)
    system = (
        f"You are a text classifier. "
        f"Classify the input as exactly one of: {labels_str}. "
        f"Respond with only the label, nothing else."
    )

    t0 = time.perf_counter()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    label = response.content[0].text.strip().lower()

    # Snap to nearest valid label (Haiku occasionally adds punctuation)
    if label not in label_names:
        for valid in label_names:
            if valid in label:
                label = valid
                break
        else:
            label = label_names[0]  # fallback

    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    tokens_used   = input_tokens + output_tokens
    cost_usd      = (
        input_tokens  * HAIKU_INPUT_COST_PER_TOKEN +
        output_tokens * HAIKU_OUTPUT_COST_PER_TOKEN
    )

    return label, latency_ms, tokens_used, cost_usd


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    specialist_name: str,
    holdout_path: Path,
    models_dir: Path,
    thresholds: list[float],
    haiku_sample_limit: int | None,
) -> tuple[list[ExampleResult], list[ThresholdResult]]:
    """
    Run the full benchmark and return per-example results and threshold table.
    """
    from crasis.deploy import Specialist

    # Load specialist
    specialist_dir = models_dir / specialist_name
    if not specialist_dir.exists():
        raise FileNotFoundError(
            f"Specialist package not found: {specialist_dir}\n"
            "Run `crasis pull {specialist_name}` or `crasis build` first."
        )
    specialist = Specialist.load(specialist_dir)
    label_names = specialist.label_names
    logger.info("Loaded specialist: %s  labels: %s", specialist_name, label_names)

    # Load holdout set
    examples: list[Example] = []
    with holdout_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(Example(text=obj["text"], ground_truth=obj["label"]))
    logger.info("Loaded %d holdout examples", len(examples))

    # Anthropic client (only used for escalated examples)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
    haiku_client = anthropic.Anthropic(api_key=api_key)

    # --- Run Crasis on all examples first ---
    logger.info("Running Crasis inference on all %d examples...", len(examples))
    crasis_results: list[dict] = []
    for ex in examples:
        result = specialist.classify(ex.text)
        crasis_results.append(result)

    # --- Determine which examples need Haiku at the most aggressive threshold ---
    # We only call Haiku once per example (at the lowest threshold that would
    # escalate it), then reuse the result across all threshold calculations.
    lowest_threshold = min(thresholds)
    needs_haiku = {
        i for i, r in enumerate(crasis_results)
        if r["confidence"] < lowest_threshold
    }

    if haiku_sample_limit is not None and len(needs_haiku) > haiku_sample_limit:
        logger.warning(
            "Capping Haiku calls at %d (of %d escalations at threshold %.2f)",
            haiku_sample_limit, len(needs_haiku), lowest_threshold,
        )
        needs_haiku = set(list(needs_haiku)[:haiku_sample_limit])

    # --- Call Haiku for escalated examples ---
    haiku_results: dict[int, tuple[str, float, int, float]] = {}
    if needs_haiku:
        logger.info("Calling Haiku for %d escalated examples...", len(needs_haiku))
        for i in sorted(needs_haiku):
            ex = examples[i]
            label, latency_ms, tokens, cost = classify_with_haiku(
                haiku_client, ex.text, label_names
            )
            haiku_results[i] = (label, latency_ms, tokens, cost)
            if (i + 1) % 10 == 0:
                logger.info("  Haiku progress: %d / %d", i + 1, len(needs_haiku))
            time.sleep(0.05)  # gentle rate limiting

    # --- Build per-example results ---
    example_results: list[ExampleResult] = []
    for i, (ex, cr) in enumerate(zip(examples, crasis_results)):
        er = ExampleResult(
            text=ex.text,
            ground_truth=ex.ground_truth,
            crasis_label=cr["label"],
            crasis_confidence=cr["confidence"],
            crasis_latency_ms=cr["latency_ms"],
        )
        if i in haiku_results:
            er.haiku_label, er.haiku_latency_ms, er.haiku_tokens_used, er.haiku_cost_usd = (
                haiku_results[i]
            )
        example_results.append(er)

    # --- Compute threshold table ---
    threshold_results: list[ThresholdResult] = []
    for threshold in sorted(thresholds):
        local     = [r for r in example_results if r.crasis_confidence >= threshold]
        escalated = [r for r in example_results if r.crasis_confidence <  threshold]

        # Accuracy: local examples where Crasis was correct
        local_correct = sum(
            1 for r in local if r.crasis_label == r.ground_truth
        )
        local_accuracy = local_correct / len(local) if local else 0.0

        # Escalation accuracy: how often Crasis was RIGHT to be uncertain
        # (i.e. Crasis label != ground truth on escalated examples)
        escalation_justified = sum(
            1 for r in escalated
            if r.haiku_label is not None and r.crasis_label != r.ground_truth
        )
        escalated_with_haiku = sum(
            1 for r in escalated if r.haiku_label is not None
        )
        escalation_accuracy = (
            escalation_justified / escalated_with_haiku
            if escalated_with_haiku else 0.0
        )

        # Overall accuracy: Crasis on local + Haiku on escalated
        overall_correct = local_correct + sum(
            1 for r in escalated
            if r.haiku_label is not None and r.haiku_label == r.ground_truth
        )
        overall_denominator = len(local) + escalated_with_haiku
        overall_accuracy = overall_correct / overall_denominator if overall_denominator else 0.0

        # Cost
        total_haiku_cost = sum(
            r.haiku_cost_usd for r in escalated if r.haiku_cost_usd is not None
        )
        cost_per_1000 = (total_haiku_cost / len(example_results)) * 1000 if example_results else 0.0

        threshold_results.append(ThresholdResult(
            threshold=threshold,
            total=len(example_results),
            local_count=len(local),
            escalated_count=len(escalated),
            local_pct=len(local) / len(example_results) * 100 if example_results else 0.0,
            local_accuracy=local_accuracy,
            escalation_accuracy=escalation_accuracy,
            overall_accuracy=overall_accuracy,
            total_haiku_cost_usd=total_haiku_cost,
            cost_per_1000_usd=cost_per_1000,
        ))

    return example_results, threshold_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    specialist_name: str,
    threshold_results: list[ThresholdResult],
    training_cost_usd: float,
) -> None:
    """Print the benchmark report to stdout."""

    print(f"\n{'=' * 72}")
    print(f"  Crasis vs Haiku — {specialist_name}")
    print(f"  Training cost: ${training_cost_usd:.4f}")
    print(f"{'=' * 72}\n")

    header = (
        f"{'Threshold':>10}  "
        f"{'Local %':>8}  "
        f"{'Local Acc':>10}  "
        f"{'Esc. Justified':>15}  "
        f"{'Overall Acc':>12}  "
        f"{'Cost/1k':>10}  "
        f"{'Breakeven':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in threshold_results:
        breakeven = (
            int(training_cost_usd / (r.cost_per_1000_usd / 1000))
            if r.cost_per_1000_usd > 0
            else 0
        )
        breakeven_str = f"{breakeven:,}" if breakeven > 0 else "∞ (free)"

        print(
            f"{r.threshold:>10.2f}  "
            f"{r.local_pct:>7.1f}%  "
            f"{r.local_accuracy:>9.1%}  "
            f"{r.escalation_accuracy:>14.1%}  "
            f"{r.overall_accuracy:>11.1%}  "
            f"${r.cost_per_1000_usd:>9.4f}  "
            f"{breakeven_str:>12}"
        )

    print(f"\n{'=' * 72}")

    # Pull out the 0.85 row for the summary callout
    row_85 = next((r for r in threshold_results if abs(r.threshold - 0.85) < 0.001), None)
    if row_85:
        print(f"\nAt threshold 0.85:")
        print(f"  {row_85.local_pct:.0f}% of classifications handled locally")
        print(f"  Overall accuracy: {row_85.overall_accuracy:.1%}")
        print(f"  Cost per 1,000 calls: ${row_85.cost_per_1000_usd:.4f}")
        breakeven = (
            int(training_cost_usd / (row_85.cost_per_1000_usd / 1000))
            if row_85.cost_per_1000_usd > 0
            else 0
        )
        if breakeven > 0:
            print(f"  Breakeven vs pure Haiku: {breakeven:,} classifications")
        else:
            print(f"  Breakeven: immediate (all calls handled locally)")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Crasis specialist vs Claude Haiku on a holdout set."
    )
    parser.add_argument(
        "--specialist",
        default="whatsapp-triage",
        help="Specialist name (default: whatsapp-triage)",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        required=True,
        help="Path to holdout JSONL file ({text, label} per line)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
        help="Directory containing specialist packages (default: ./models)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.75, 0.80, 0.85, 0.90],
        help="Confidence thresholds to evaluate (default: 0.75 0.80 0.85 0.90)",
    )
    parser.add_argument(
        "--training-cost",
        type=float,
        default=0.138051,
        help="Actual training cost in USD for breakeven calculation (default: 0.138051)",
    )
    parser.add_argument(
        "--haiku-limit",
        type=int,
        default=None,
        help="Cap Haiku API calls (useful for cost control during testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output path for full results JSON (default: benchmark_results.json)",
    )
    args = parser.parse_args()

    example_results, threshold_results = run_benchmark(
        specialist_name=args.specialist,
        holdout_path=args.holdout,
        models_dir=args.models_dir,
        thresholds=args.thresholds,
        haiku_sample_limit=args.haiku_limit,
    )

    print_report(args.specialist, threshold_results, args.training_cost)

    # Write full results to JSON
    output = {
        "specialist": args.specialist,
        "training_cost_usd": args.training_cost,
        "threshold_results": [asdict(r) for r in threshold_results],
        "examples": [asdict(r) for r in example_results],
    }
    args.output.write_text(json.dumps(output, indent=2))
    logger.info("Full results written to %s", args.output)


if __name__ == "__main__":
    main()
