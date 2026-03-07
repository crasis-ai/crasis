# availability-handler

Detects messages where someone is communicating their availability or unavailability — so scheduling assistants and calendar tools can act on the signal without reading every message manually.

Runs locally in under 100ms per message, no network required after installation.

---

## Performance

Trained on BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`) with 3,519 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 98.72% | **93.33%** |
| Macro F1 | 0.9872 | **0.9333** |
| Samples | 392 | 30 |

| Metric | Value |
|--------|-------|
| Model size | 4.3MB (limit: 22MB) |
| Avg latency (CPU) | 0.62 ms |
| Training samples | 3,519 |

Synthetic-real gap: **5.4pp.** Performance is solid but watch the boundary between availability statements and scheduling requests — "please hold while I check my calendar" and "I'll circle back" read similarly to genuine availability windows in some contexts. Out-of-office auto-replies without a return date are correctly classified as negative. Suitable for production use with standard confidence monitoring. Holdout fixture: [`tests/fixtures/availability-handler.jsonl`](../../tests/fixtures/availability-handler.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/availability-handler-onnx")

result = model.classify(
    "I'm free Thursday after 2pm or any time Friday. Let me know what works."
)
# {"label": "positive", "confidence": 0.97, "latency_ms": 38}

result = model.classify(
    "Can we find a time to sync this week? What does your calendar look like?"
)
# {"label": "negative", "confidence": 0.95, "latency_ms": 36}
```

Batch classification:

```python
messages = [msg.body for msg in thread]
results = model.classify_batch(messages)

availability_signals = [m for m, r in zip(thread, results) if r["label"] == "positive"]
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.80` — messages classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
