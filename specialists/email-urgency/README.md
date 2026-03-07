# email-urgency

Classifies incoming emails into four urgency levels — `critical`, `high`, `normal`, `low` — so that messages demanding immediate attention are surfaced first and low-priority noise stays buried.

Runs locally in under 100ms per message, no network required after installation.

---

## Classes

| Label | Meaning |
|-------|---------|
| `critical` | Requires action within hours — system outages, legal deadlines, executive escalations, security incidents |
| `high` | Requires action today — client requests, time-sensitive decisions, unblocking requests from teammates |
| `normal` | Requires action this week — standard project updates, meeting requests, routine follow-ups |
| `low` | No action required soon — newsletters, automated reports, FYI threads, marketing |

---

## Performance

Trained on BERT-Mini (`google/bert_uncased_L-4_H-256_A-4`) with 3,671 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 86.76% | **65.63%** |
| Macro F1 | 0.8703 | **0.6533** |
| Samples | 408 | 32 |

| Metric | Value |
|--------|-------|
| Model size | 10.84MB (limit: 43MB) |
| Avg latency (CPU) | 2.79 ms |
| Training samples | 3,671 |

Synthetic-real gap: **21.1pp.** The four-class urgency boundary (`critical` / `high` / `normal` / `low`) is genuinely hard to learn from synthetic data alone — real emails rarely announce their urgency as clearly as generated examples do. Per-class holdout F1 is roughly even across all four classes (~0.62–0.75), so no single class is broken; the whole task is underfit. Not recommended for production use without a hybrid or real-data training strategy. Holdout fixture: [`tests/fixtures/email-urgency.jsonl`](../../tests/fixtures/email-urgency.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/email-urgency-onnx")

result = model.classify(
    "The production database is down. All services affected. Need immediate response."
)
# {"label": "critical", "confidence": 0.97, "latency_ms": 42}

result = model.classify(
    "Can you review the Q3 proposal before EOD? Client is waiting."
)
# {"label": "high", "confidence": 0.91, "latency_ms": 38}

result = model.classify(
    "Here are the notes from Tuesday's sync."
)
# {"label": "normal", "confidence": 0.88, "latency_ms": 39}

result = model.classify(
    "Your monthly usage report is ready to view."
)
# {"label": "low", "confidence": 0.94, "latency_ms": 37}
```

Batch classification:

```python
emails = [email.body for email in inbox]
results = model.classify_batch(emails)

critical = [e for e, r in zip(inbox, results) if r["label"] == "critical"]
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.78` — emails classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
