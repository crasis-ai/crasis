# meeting-parser

Detects whether a message or email contains a meeting request, scheduling proposal, or calendar invite that requires the recipient to accept, decline, or propose an alternative time.

Runs locally in under 100ms per message, no network required after installation.

---

## Performance

Trained on BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`) with 5,332 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 99.83% | **86.67%** |
| Macro F1 | 0.9983 | **0.8667** |
| Samples | 593 | 30 |

| Metric | Value |
|--------|-------|
| Model size | 4.3MB (limit: 35MB) |
| Avg latency (CPU) | 0.55 ms |
| Training samples | 5,332 |

Synthetic-real gap: **13.2pp.** The main failure mode on holdout is soft scheduling language — "let's find time this week", "I'll send a Calendly link" — which synthetic data tends to represent with more explicit time/date markers than real messages use. Meeting confirmation receipts and past-meeting references are distinguished correctly. For production use, the high-confidence positive predictions (≥0.90) are reliable; borderline cases should be reviewed. Holdout fixture: [`tests/fixtures/meeting-parser.jsonl`](../../tests/fixtures/meeting-parser.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/meeting-parser-onnx")

result = model.classify(
    "Can we jump on a call tomorrow at 3pm to walk through the proposal?"
)
# {"label": "positive", "confidence": 0.99, "latency_ms": 39}

result = model.classify(
    "Here are the notes from yesterday's call. Let me know if I missed anything."
)
# {"label": "negative", "confidence": 0.98, "latency_ms": 37}
```

Batch classification:

```python
emails = [e.body for e in inbox]
results = model.classify_batch(emails)

scheduling_needed = [e for e, r in zip(inbox, results) if r["label"] == "positive"]
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.80` — messages classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
