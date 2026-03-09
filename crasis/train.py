"""
crasis.train — Specialist model distillation pipeline.

Default target: BERT-Tiny (~4.3MB ONNX after int8 quantization, <1ms CPU inference).
Only goes larger if spec.constraints.max_model_size_mb explicitly requires it.

Architecture selection (float32 ONNX size budget → model):
  ≤ 17MB  → google/bert_uncased_L-2_H-128_A-2  (BERT-Tiny,   4.4M params, ~4.3MB quantized)
  ≤ 43MB  → google/bert_uncased_L-4_H-256_A-4  (BERT-Mini,   11M params,  ~10.8MB quantized)
  ≤ 110MB → google/bert_uncased_L-4_H-512_A-8  (BERT-Small,  28.7M params)
  ≤ 158MB → google/bert_uncased_L-8_H-512_A-8  (BERT-Medium, 41.4M params)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from crasis.spec import CrasisSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture selection table
# ---------------------------------------------------------------------------

# Each entry: (actual_onnx_size_mb, model_id)
# All base models are from the official Google BERT family (L=layers, H=hidden, A=heads).
# Ordered smallest to largest. _select_base_model picks the largest model
# whose ONNX artifact fits within the spec's max_model_size_mb budget.
_ARCH_TABLE = [
    (17, "google/bert_uncased_L-2_H-128_A-2"),  # BERT-Tiny,   4.4M params
    (43, "google/bert_uncased_L-4_H-256_A-4"),  # BERT-Mini,   11M params
    (110, "google/bert_uncased_L-4_H-512_A-8"),  # BERT-Small,  28.7M params
    (158, "google/bert_uncased_L-8_H-512_A-8"),  # BERT-Medium, 41.4M params
]


def _select_base_model(max_mb: int) -> str:
    """Select the largest model whose ONNX size fits within max_mb."""
    selected = _ARCH_TABLE[0][1]  # always fits: bert-tiny is 17MB
    for size_mb, model_id in _ARCH_TABLE:
        if size_mb <= max_mb:
            selected = model_id
    return selected


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """
    Outcome of a completed training run.

    Attributes:
        model_path: Directory containing the saved model and tokenizer.
        base_model: HuggingFace model ID used as the starting point.
        eval_accuracy: Accuracy on the held-out eval split.
        eval_f1: Macro F1 on the held-out eval split.
        training_duration_s: Wall-clock training time in seconds.
        num_train_samples: Number of examples used for training.
        num_eval_samples: Number of examples used for evaluation.
        passed_quality_gate: Whether the model met spec.quality thresholds.
    """

    model_path: Path
    base_model: str
    eval_accuracy: float
    eval_f1: float
    training_duration_s: float
    num_train_samples: int
    num_eval_samples: int
    passed_quality_gate: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    spec: CrasisSpec,
    data_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
) -> TrainResult:
    """
    Train a specialist model from a validated spec and generated JSONL dataset.

    Args:
        spec: Validated CrasisSpec instance.
        data_path: Path to JSONL training data (output of crasis.factory.generate).
        output_dir: Root model output directory. Model saved to output_dir/<spec.name>/.
        device: Force a specific device ('cpu', 'cuda', 'mps'). Auto-detected if None.

    Returns:
        TrainResult with model path, eval metrics, and training metadata.

    Raises:
        FileNotFoundError: If data_path does not exist.
        QualityGateError: If the trained model fails spec.quality thresholds.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    model_out = Path(output_dir) / spec.name
    model_out.mkdir(parents=True, exist_ok=True)

    resolved_device = device or _detect_device()
    base_model = _select_base_model(spec.constraints.max_model_size_mb)
    logger.info("Base model: %s | Device: %s", base_model, resolved_device)

    # Load data
    examples = _load_jsonl(data_path)
    logger.info("Loaded %d examples from %s", len(examples), data_path)

    # Build label mappings
    label2id = {name: i for i, name in enumerate(spec.label_names)}
    id2label = {i: name for name, i in label2id.items()}

    # Stratified 90/10 split — preserves label distribution in both sets
    from sklearn.model_selection import train_test_split

    labels_for_split = [ex["label"] for ex in examples]
    train_examples, eval_examples = train_test_split(
        examples,
        test_size=0.1,
        random_state=42,
        stratify=labels_for_split,
    )
    logger.info("Train: %d | Eval: %d", len(train_examples), len(eval_examples))

    # Tokenise
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_dataset = _tokenize(train_examples, tokenizer)
    eval_dataset = _tokenize(eval_examples, tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=spec.num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Training arguments — tuned for RTX 4060 / CPU fallback
    use_gpu = resolved_device.startswith("cuda")
    training_args = TrainingArguments(
        output_dir=str(model_out / "checkpoints"),
        num_train_epochs=10,
        per_device_train_batch_size=32 if use_gpu else 16,
        per_device_eval_batch_size=64 if use_gpu else 32,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        logging_steps=50,
        fp16=use_gpu,
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(spec),
    )

    t0 = time.time()
    trainer.train()
    duration = time.time() - t0
    logger.info("Training complete in %.1fs", duration)

    # Final evaluation
    eval_results = trainer.evaluate()
    accuracy = eval_results.get("eval_accuracy", 0.0)
    f1 = eval_results.get("eval_f1", 0.0)
    logger.info("Eval accuracy: %.4f | F1: %.4f", accuracy, f1)

    # Save model and tokenizer
    trainer.save_model(str(model_out))
    tokenizer.save_pretrained(str(model_out))

    # Persist label map alongside model
    label_map = {"label2id": label2id, "id2label": id2label}
    (model_out / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )

    # Quality gate check
    passed = _check_quality_gate(spec, accuracy, f1)

    result = TrainResult(
        model_path=model_out,
        base_model=base_model,
        eval_accuracy=accuracy,
        eval_f1=f1,
        training_duration_s=duration,
        num_train_samples=len(train_examples),
        num_eval_samples=len(eval_examples),
        passed_quality_gate=passed,
    )

    _save_train_result(result, model_out)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("CUDA device: %s", name)
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS")
        return "mps"
    logger.info("No GPU found, using CPU")
    return "cpu"


def _load_jsonl(path: Path) -> list[dict]:
    """Load and shuffle JSONL training data."""
    import random

    examples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    random.shuffle(examples)
    return examples


def _tokenize(examples: list[dict], tokenizer) -> Dataset:
    """Tokenize a list of examples into a HuggingFace Dataset."""
    ds = Dataset.from_list(
        [{"text": ex["text"], "labels": ex["label_id"]} for ex in examples]
    )

    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding=False,
        )

    return ds.map(_tok, batched=True, remove_columns=["text"])


def _make_compute_metrics(spec: CrasisSpec):
    """Return a metrics function for the Trainer."""
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "f1": f1}

    return compute_metrics


def _check_quality_gate(spec: CrasisSpec, accuracy: float, f1: float) -> bool:
    """
    Check whether eval metrics meet the spec's quality requirements.

    Logs a warning rather than raising — the export step enforces the hard stop.
    """
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
        logger.info("Quality gate PASSED ✓")

    return passed


def _save_train_result(result: TrainResult, model_out: Path) -> None:
    """Persist training metadata to model directory."""
    meta = {
        "base_model": result.base_model,
        "eval_accuracy": result.eval_accuracy,
        "eval_f1": result.eval_f1,
        "training_duration_s": result.training_duration_s,
        "num_train_samples": result.num_train_samples,
        "num_eval_samples": result.num_eval_samples,
        "passed_quality_gate": result.passed_quality_gate,
    }
    (model_out / "train_result.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
