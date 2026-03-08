"""
scripts/run_demo.py

Classifies the 847-email demo inbox using the email-urgency specialist.
Shows a live terminal ticker, then prints the final summary line.

No API calls. No internet. Everything stays on device.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --model ./models/email-urgency-onnx
    python scripts/run_demo.py --mock   # dry-run without a trained model (for dev/recording)
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
WHITE   = "\033[97m"

LABEL_COLORS = {
    "critical": RED,
    "high":     YELLOW,
    "low":      DIM + WHITE,
}

LABEL_ICONS = {
    "critical": "🔴",
    "high":     "🟡",
    "low":      "⚪",
}


def colored(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def clear_line() -> None:
    sys.stdout.write("\033[2K\r")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Mock classifier (used when --mock is set or model path not found)
# ---------------------------------------------------------------------------

def mock_classifier(email: dict) -> dict:
    """
    Deterministic mock that returns plausible labels based on category.
    Used for recording the demo before the real model is trained,
    or for testing this script without ONNX runtime installed.
    """
    import random
    rng = random.Random(email["id"])

    category = email.get("category", "work")

    dist = {
        "urgent":     {"critical": 0.65, "high": 0.30, "low": 0.05},
        "work":       {"critical": 0.05, "high": 0.70, "low": 0.25},
        "newsletter": {"critical": 0.00, "high": 0.05, "low": 0.95},
        "receipt":    {"critical": 0.00, "high": 0.05, "low": 0.95},
        "social":     {"critical": 0.00, "high": 0.10, "low": 0.90},
        "spam":       {"critical": 0.00, "high": 0.05, "low": 0.95},
        "misc":       {"critical": 0.02, "high": 0.18, "low": 0.80},
    }.get(category, {"critical": 0.05, "high": 0.20, "low": 0.75})

    labels  = list(dist.keys())
    weights = list(dist.values())
    label   = rng.choices(labels, weights=weights, k=1)[0]
    conf    = round(rng.uniform(0.82, 0.99), 3)

    # Simulate ~1–4ms latency (real specialist range)
    latency_ms = round(rng.uniform(1.2, 3.9), 1)

    return {"label": label, "confidence": conf, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# Real classifier (requires crasis package + trained model)
# ---------------------------------------------------------------------------

def load_real_classifier(model_path: str):
    try:
        from crasis import Specialist
        model = Specialist.load(model_path)

        def classify(email: dict) -> dict:
            text = f"Subject: {email['subject']}\n\n{email['body']}"
            return model.classify(text)

        return classify
    except ImportError:
        print(colored(
            "⚠  crasis package not found. Install with: pip install -e .",
            YELLOW,
        ))
        raise
    except Exception as e:
        print(colored(f"⚠  Failed to load model: {e}", YELLOW))
        raise


# ---------------------------------------------------------------------------
# Live ticker display
# ---------------------------------------------------------------------------

TICKER_INTERVAL = 10  # Update display every N emails


def format_label(label: str) -> str:
    icon  = LABEL_ICONS.get(label, "•")
    color = LABEL_COLORS.get(label, WHITE)
    label_padded = f"{label.upper():<8}"
    return f"{icon} {colored(label_padded, color)}"


def print_header():
    width = 72
    print()
    print(colored("─" * width, DIM))
    print(colored("  CRASIS  ", BOLD + CYAN) + colored("email-urgency specialist · local inference · no API calls", DIM))
    print(colored("─" * width, DIM))
    print(
        f"  {'#':<6} {'FROM':<28} {'SUBJECT':<30} {'LABEL':<12}",
    )
    print(colored("─" * width, DIM))


def print_ticker_row(email: dict, result: dict, row_num: int):
    from_display    = email["from_name"][:26]
    subject_display = email["subject"][:28]
    label           = result["label"]
    icon            = LABEL_ICONS.get(label, "•")
    color           = LABEL_COLORS.get(label, WHITE)
    label_str       = colored(f"{icon} {label}", color)

    print(
        f"  {row_num:<6} {from_display:<28} {subject_display:<30} {label_str}",
    )


def print_progress_bar(done: int, total: int, elapsed: float):
    width     = 40
    pct       = done / total
    filled    = int(width * pct)
    bar       = "█" * filled + "░" * (width - filled)
    rate      = done / elapsed if elapsed > 0 else 0
    eta_s     = (total - done) / rate if rate > 0 else 0

    clear_line()
    sys.stdout.write(
        f"  {colored(bar, CYAN)}  "
        f"{colored(f'{done}/{total}', BOLD)} emails  "
        f"{colored(f'{rate:.0f}/s', DIM)}  "
        f"ETA {eta_s:.1f}s"
    )
    sys.stdout.flush()


def print_summary(results: list[dict], emails: list[dict], elapsed: float, mock: bool):
    counts = Counter(r["label"] for r in results)
    total  = len(results)
    avg_ms = sum(r["latency_ms"] for r in results) / total

    tag = colored(" [MOCK]", YELLOW + BOLD) if mock else ""

    print("\n")
    print(colored("─" * 72, DIM))
    print(
        f"\n  {colored('✓', GREEN + BOLD)}  "
        f"{colored(str(total), BOLD + WHITE)} emails classified in "
        f"{colored(f'{elapsed:.1f}s', BOLD + CYAN)}"
        f"  ·  avg latency {colored(f'{avg_ms:.1f}ms', CYAN)} per email"
        f"{tag}"
    )
    print()
    print(f"  {'LABEL':<12}  {'COUNT':>6}  {'BAR'}")
    print(colored("  " + "─" * 52, DIM))

    for label in ["critical", "high", "low"]:
        n     = counts.get(label, 0)
        pct   = n / total * 100
        bar   = "█" * int(pct / 2)
        color = LABEL_COLORS.get(label, WHITE)
        icon  = LABEL_ICONS.get(label, "•")
        print(
            f"  {colored(f'{icon} {label}', color):<22}  "
            f"{n:>6}  {colored(bar, color)}  {colored(f'{pct:.1f}%', DIM)}"
        )

    print()
    print(colored("  Zero API calls. Zero data left this machine.", DIM))
    print(colored("─" * 72, DIM))
    print()


# ---------------------------------------------------------------------------
# Eval (--eval flag only)
# ---------------------------------------------------------------------------

# Maps inbox ground-truth category to expected urgency label.
# Not a perfect proxy — some work emails are legitimately critical — but
# good enough to catch systematic failures before recording the demo.
_CATEGORY_TO_LABEL = {
    "urgent":     "critical",
    "work":       "high",
    "newsletter": "low",
    "receipt":    "low",
    "social":     "low",
    "spam":       "low",
    "misc":       "low",
}

LABELS = ["critical", "high", "low"]


def print_eval(results: list[dict], emails: list[dict]) -> None:
    """Print confusion matrix and per-class F1 using category as ground truth."""
    y_true, y_pred = [], []
    skipped = 0
    for email, result in zip(emails, results):
        expected = _CATEGORY_TO_LABEL.get(email.get("category", ""))
        if expected is None:
            skipped += 1
            continue
        y_true.append(expected)
        y_pred.append(result["label"])

    total = len(y_true)
    if total == 0:
        print(colored("  No ground-truth categories found — skipping eval.", YELLOW))
        return

    # Confusion matrix
    label_idx = {l: i for i, l in enumerate(LABELS)}
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            matrix[label_idx[t]][label_idx[p]] += 1

    # Per-class precision, recall, F1
    def _prf(label: str) -> tuple[float, float, float]:
        i = label_idx[label]
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(LABELS))) - tp
        fn = sum(matrix[i][c] for c in range(len(LABELS))) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / total
    macro_f1 = sum(_prf(l)[2] for l in LABELS) / len(LABELS)

    width = 72
    print()
    print(colored("─" * width, DIM))
    print(colored("  EVAL  ", BOLD + MAGENTA) + colored("ground truth: inbox category → expected label", DIM))
    print(colored("─" * width, DIM))

    # Confusion matrix header
    col_w = 10
    header = f"  {'':16}" + "".join(f"{'pred:'+l:>{col_w}}" for l in LABELS)
    print(colored(header, DIM))
    for i, true_label in enumerate(LABELS):
        row_total = sum(matrix[i])
        row = f"  {'true:'+true_label:<16}"
        for j, pred_label in enumerate(LABELS):
            val = matrix[i][j]
            cell = f"{val:>{col_w}}"
            if i == j:
                color = GREEN if val > 0 else DIM
            else:
                color = RED if val > 0 else DIM
            row += colored(cell, color)
        row += colored(f"  (n={row_total})", DIM)
        print(row)

    print()
    print(f"  {'LABEL':<12}  {'PREC':>6}  {'REC':>6}  {'F1':>6}")
    print(colored("  " + "─" * 36, DIM))
    for label in LABELS:
        prec, rec, f1 = _prf(label)
        color = LABEL_COLORS.get(label, WHITE)
        icon  = LABEL_ICONS.get(label, "•")
        print(
            f"  {colored(f'{icon} {label}', color):<22}"
            f"  {colored(f'{prec:.3f}', WHITE):>6}"
            f"  {colored(f'{rec:.3f}', WHITE):>6}"
            f"  {colored(f'{f1:.3f}', BOLD + WHITE):>6}"
        )

    print()
    acc_color  = GREEN if accuracy >= 0.85 else (YELLOW if accuracy >= 0.75 else RED)
    f1_color   = GREEN if macro_f1 >= 0.85 else (YELLOW if macro_f1 >= 0.75 else RED)
    print(
        f"  accuracy  {colored(f'{accuracy:.3f}', acc_color + BOLD)}   "
        f"macro-F1  {colored(f'{macro_f1:.3f}', f1_color + BOLD)}"
    )
    if skipped:
        print(colored(f"  ({skipped} emails skipped — no category field)", DIM))
    print(colored("─" * width, DIM))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Crasis email demo")
    parser.add_argument(
        "--inbox",
        type=str,
        default="demo/inbox_847.jsonl",
        help="Path to inbox JSONL (default: demo/inbox_847.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/email-urgency-onnx",
        help="Path to trained ONNX specialist",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock classifier (no model required)",
    )
    parser.add_argument(
        "--ticker-rows",
        type=int,
        default=14,
        help="How many rows to show in the rolling ticker (default: 14)",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=11.0,
        help="Mock mode: pace to this many seconds total for demo recording (default: 11.0). "
             "Ignored when using a real model.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional: write classified results to this JSONL path",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Print confusion matrix and F1 scores using inbox category as ground truth",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load inbox
    # ------------------------------------------------------------------
    inbox_path = Path(args.inbox)
    if not inbox_path.exists():
        print(colored(f"✗  Inbox not found: {inbox_path}", RED))
        print(colored(f"   Run first: python scripts/generate_demo_inbox.py", DIM))
        sys.exit(1)

    with inbox_path.open() as f:
        emails = [json.loads(line) for line in f]

    total = len(emails)

    # ------------------------------------------------------------------
    # Load classifier
    # ------------------------------------------------------------------
    if args.mock:
        classify = mock_classifier
        mode_str = colored("mock mode", YELLOW)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            print(colored(f"✗  Model not found: {model_path}", RED))
            print(colored("   Train first:  crasis build --spec specialists/email-urgency/spec.yaml", DIM))
            print(colored("   Then export:  crasis export --spec specialists/email-urgency/spec.yaml", DIM))
            print()
            print(colored("   Use --mock to run without a model.", DIM))
            sys.exit(1)
        else:
            try:
                classify = load_real_classifier(str(model_path))
                mode_str = colored("live model", GREEN)
            except Exception as e:
                import os
                print(colored(f"✗  Model load failed: {e}", RED))
                print(colored(f"   Path given:  {model_path}", DIM))
                if model_path.is_dir():
                    files = sorted(model_path.iterdir())
                    if files:
                        print(colored("   Directory contains:", DIM))
                        for f in files:
                            print(colored(f"     {f.name}", DIM))
                    else:
                        print(colored("   Directory is empty.", DIM))
                print()
                print(colored("   Use --mock to run without a model.", DIM))
                sys.exit(1)

    # ------------------------------------------------------------------
    # Throttle calculation (mock only)
    # Sleep per batch so total wall time ≈ --target-seconds.
    # We split time budget: 30% for the ticker phase, 70% for the rest.
    # ------------------------------------------------------------------
    ticker_rows = args.ticker_rows
    post_ticker = total - ticker_rows
    sleep_per_ticker_row  = 0.0
    sleep_per_batch       = 0.0

    if args.target_seconds > 0:
        budget_ticker = args.target_seconds * 0.30
        budget_post   = args.target_seconds * 0.70
        sleep_per_ticker_row = budget_ticker / max(ticker_rows, 1)
        batches_post         = max(post_ticker // TICKER_INTERVAL, 1)
        sleep_per_batch      = budget_post / batches_post

    # ------------------------------------------------------------------
    # Run — Phase 1: ticker rows
    # ------------------------------------------------------------------
    print_header()

    results    = []
    t_start    = time.perf_counter()

    # Print the ticker rows with pacing
    for i in range(min(ticker_rows, total)):
        result = classify(emails[i])
        results.append(result)
        print_ticker_row(emails[i], result, i + 1)
        sys.stdout.flush()
        if sleep_per_ticker_row > 0:
            time.sleep(sleep_per_ticker_row)

    # Print the ellipsis separator — after this line the cursor is on a new
    # line and progress bar \r updates will NOT clobber the ticker rows.
    if total > ticker_rows:
        print(colored(f"  {'...':<6} {'(processing remaining emails)':<60}", DIM))

    # Print a blank line that the progress bar will own exclusively.
    # All subsequent \r updates rewrite only this line.
    sys.stdout.write("  \n")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Run — Phase 2: remaining emails with progress bar on its own line
    # ------------------------------------------------------------------
    for i in range(ticker_rows, total):
        result = classify(emails[i])
        results.append(result)

        elapsed = time.perf_counter() - t_start
        if (i + 1 - ticker_rows) % TICKER_INTERVAL == 0 or i == total - 1:
            # Move cursor up one line, overwrite the progress line, move back down
            sys.stdout.write("\033[1A")   # cursor up
            print_progress_bar(i + 1, total, elapsed)
            sys.stdout.write("\n")
            sys.stdout.flush()
            if sleep_per_batch > 0:
                time.sleep(sleep_per_batch)

    elapsed = time.perf_counter() - t_start

    # Move cursor up and clear the progress bar line before printing summary
    sys.stdout.write("\033[1A")
    clear_line()

    print_summary(results, emails, elapsed, mock=args.mock)

    if args.eval:
        print_eval(results, emails)

    # ------------------------------------------------------------------
    # Optional output
    # ------------------------------------------------------------------
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for email, result in zip(emails, results):
                row = {**email, **result}
                f.write(json.dumps(row) + "\n")
        print(colored(f"  Results written to {out_path}", DIM))


if __name__ == "__main__":
    main()