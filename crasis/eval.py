"""
crasis.eval — Evaluation harness and quality gate enforcement.

Two evaluation modes:
  1. evaluate()        — runs the model on a JSONL eval set, enforces quality gates
  2. eval_on_holdout() — runs the ONNX specialist on a hand-authored holdout JSONL,
                         reports accuracy/F1 and flags the synthetic-vs-real gap

The second mode is the honest number. Use it to sanity-check that high synthetic
accuracy reflects genuine skill, not synthetic distribution memorization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from crasis.spec import CrasisSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """
    Outcome of a model evaluation pass.

    Attributes:
        accuracy: Accuracy on the evaluated dataset.
        f1_macro: Macro-averaged F1 on the evaluated dataset.
        classification_report: Full sklearn classification report string.
        passed_quality_gate: Whether the model met spec.quality thresholds.
        num_samples: Number of examples evaluated.
        label_names: Ordered list of class label strings.
    """

    accuracy: float
    f1_macro: float
    classification_report: str
    passed_quality_gate: bool
    num_samples: int
    label_names: list[str]


@dataclass
class HoldoutResult:
    """
    Outcome of a holdout evaluation pass.

    Attributes:
        accuracy: Accuracy on the holdout dataset.
        f1_macro: Macro-averaged F1 on the holdout dataset.
        classification_report: Full sklearn classification report string.
        num_samples: Number of examples evaluated.
        label_names: Ordered list of class label strings.
        synthetic_accuracy: Synthetic eval accuracy for gap comparison (optional).
        gap_flagged: True if holdout accuracy is >5pp below synthetic_accuracy.
    """

    accuracy: float
    f1_macro: float
    classification_report: str
    num_samples: int
    label_names: list[str]
    synthetic_accuracy: float | None = None
    gap_flagged: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    spec: CrasisSpec,
    model_path: str | Path,
    eval_data_path: str | Path | None = None,
) -> EvalResult:
    """
    Evaluate a trained PyTorch model against spec quality gates.

    Loads the model from model_path, tokenizes eval_data_path (or reads
    train_result.json for cached metrics), computes accuracy and F1, and
    checks against spec.quality thresholds.

    Args:
        spec: Validated CrasisSpec instance.
        model_path: Path to the trained model directory (output of crasis train).
        eval_data_path: Optional path to eval JSONL. If None, reads cached
                        metrics from train_result.json in model_path.

    Returns:
        EvalResult with metrics and quality gate status.

    Raises:
        FileNotFoundError: If model_path or eval_data_path does not exist.
        ValueError: If no eval data and no cached train_result.json.
    """
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    import numpy as np

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Fast path: use cached metrics from training if no eval data provided
    if eval_data_path is None:
        # Check model_path directly, then fall back to the sibling PyTorch dir
        # (ONNX packages are named <name>-onnx; PyTorch dir is named <name>)
        cached = model_path / "train_result.json"
        if not cached.exists():
            pytorch_sibling = model_path.parent / model_path.name.removesuffix("-onnx")
            cached = pytorch_sibling / "train_result.json"
        if not cached.exists():
            raise ValueError(
                f"No eval_data_path provided and no train_result.json in {model_path} "
                f"or its PyTorch sibling. Pass --eval-data or re-run crasis train."
            )
        meta = json.loads(cached.read_text(encoding="utf-8"))
        accuracy = meta["eval_accuracy"]
        f1 = meta["eval_f1"]
        passed = _check_gates(spec, accuracy, f1)
        return EvalResult(
            accuracy=accuracy,
            f1_macro=f1,
            classification_report="(cached from training run — no eval data provided)",
            passed_quality_gate=passed,
            num_samples=meta.get("num_eval_samples", 0),
            label_names=spec.label_names,
        )

    eval_data_path = Path(eval_data_path)
    if not eval_data_path.exists():
        raise FileNotFoundError(f"Eval data not found: {eval_data_path}")

    examples = _load_jsonl(eval_data_path)
    texts = [ex["text"] for ex in examples]
    true_ids = [ex["label_id"] for ex in examples]

    # Load PyTorch model for evaluation
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    pred_ids: list[int] = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length",
            )
            logits = model(**enc).logits
            pred_ids.append(int(torch.argmax(logits, dim=-1).item()))

    accuracy = float(accuracy_score(true_ids, pred_ids))
    f1 = float(f1_score(true_ids, pred_ids, average="macro", zero_division=0))
    report = classification_report(
        true_ids,
        pred_ids,
        target_names=spec.label_names,
        zero_division=0,
    )

    passed = _check_gates(spec, accuracy, f1)

    return EvalResult(
        accuracy=accuracy,
        f1_macro=f1,
        classification_report=report,
        passed_quality_gate=passed,
        num_samples=len(examples),
        label_names=spec.label_names,
    )


def eval_on_holdout(
    specialist,  # crasis.deploy.Specialist
    holdout_path: str | Path,
    spec: CrasisSpec,
    synthetic_accuracy: float | None = None,
    gap_threshold: float = 0.05,
) -> HoldoutResult:
    """
    Evaluate a deployed ONNX specialist on a hand-authored holdout JSONL.

    This is the honest metric. The specialist must already be loaded via
    Specialist.load(). The holdout file should contain real-world or
    carefully hand-authored examples that were NOT generated by factory.py.

    A gap is flagged when holdout accuracy drops more than gap_threshold below
    synthetic_accuracy — this indicates the model is memorizing synthetic
    distribution patterns rather than learning genuine semantics.

    Args:
        specialist: A loaded Specialist instance (crasis.deploy.Specialist).
        holdout_path: Path to holdout JSONL with 'text' and 'label' fields.
        spec: Validated CrasisSpec instance.
        synthetic_accuracy: Optional synthetic eval accuracy for gap comparison.
                            If provided, a gap warning is emitted if the holdout
                            accuracy is more than gap_threshold below it.
        gap_threshold: Fraction below which a gap is flagged (default 0.05 = 5pp).

    Returns:
        HoldoutResult with accuracy, F1, report, and gap flag.

    Raises:
        FileNotFoundError: If holdout_path does not exist.
    """
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    holdout_path = Path(holdout_path)
    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout data not found: {holdout_path}")

    examples = _load_jsonl(holdout_path)
    label_names = spec.label_names
    label2id = {name: i for i, name in enumerate(label_names)}

    true_ids: list[int] = []
    pred_ids: list[int] = []

    for ex in examples:
        text = ex["text"]
        true_label = ex["label"]
        true_id = ex.get("label_id", label2id.get(true_label, -1))
        true_ids.append(true_id)

        result = specialist.classify(text)
        pred_ids.append(result["label_id"])

    accuracy = float(accuracy_score(true_ids, pred_ids))
    f1 = float(f1_score(true_ids, pred_ids, average="macro", zero_division=0))
    report = classification_report(
        true_ids,
        pred_ids,
        target_names=label_names,
        zero_division=0,
    )

    gap_flagged = False
    if synthetic_accuracy is not None:
        gap = synthetic_accuracy - accuracy
        if gap > gap_threshold:
            gap_flagged = True
            logger.warning(
                "Holdout gap flagged: synthetic=%.4f holdout=%.4f gap=%.4f (threshold=%.2f). "
                "Model may be memorizing synthetic distribution patterns.",
                synthetic_accuracy,
                accuracy,
                gap,
                gap_threshold,
            )
        else:
            logger.info(
                "Holdout gap OK: synthetic=%.4f holdout=%.4f gap=%.4f",
                synthetic_accuracy,
                accuracy,
                gap,
            )

    logger.info(
        "Holdout eval complete: accuracy=%.4f f1=%.4f n=%d",
        accuracy,
        f1,
        len(examples),
    )

    return HoldoutResult(
        accuracy=accuracy,
        f1_macro=f1,
        classification_report=report,
        num_samples=len(examples),
        label_names=label_names,
        synthetic_accuracy=synthetic_accuracy,
        gap_flagged=gap_flagged,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts. Does not shuffle."""
    examples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _check_gates(spec: CrasisSpec, accuracy: float, f1: float) -> bool:
    """Check quality gate thresholds. Returns True if all gates pass."""
    passed = True

    if accuracy < spec.quality.min_accuracy:
        logger.warning(
            "Quality gate FAILED: accuracy %.4f < required %.4f",
            accuracy,
            spec.quality.min_accuracy,
        )
        passed = False

    if spec.quality.min_f1 is not None and f1 < spec.quality.min_f1:
        logger.warning(
            "Quality gate FAILED: F1 %.4f < required %.4f",
            f1,
            spec.quality.min_f1,
        )
        passed = False

    if passed:
        logger.info("Quality gate PASSED")

    return passed
