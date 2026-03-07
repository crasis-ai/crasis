# whatsapp-triage

Classifies incoming WhatsApp messages as requiring an immediate human response versus routine or automated messages that can be handled later or ignored.

Runs locally in under 100ms per message, no network required after installation.

---

## Performance

Trained on BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`) with 2,746 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 98.04% | **96.67%** |
| Macro F1 | 0.9804 | **0.9666** |
| Samples | 306 | 30 |

| Metric | Value |
|--------|-------|
| Model size | 4.3MB (limit: 19MB) |
| Avg latency (CPU) | 0.56 ms |
| Training samples | 2,746 |

Synthetic-real gap: **1.4pp.** One of the two best-performing specialists on holdout eval. The urgent/non-urgent boundary is crisp and well-represented by synthetic data — automated notifications and transactional messages are distinct enough from genuine escalations that the model transfers cleanly to real text. Suitable for production use. Holdout fixture: [`tests/fixtures/whatsapp-triage.jsonl`](../../tests/fixtures/whatsapp-triage.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/whatsapp-triage-onnx")

result = model.classify(
    "My delivery was supposed to arrive yesterday and nobody can tell me where it is. I need this resolved NOW."
)
# {"label": "positive", "confidence": 0.98, "latency_ms": 40}

result = model.classify(
    "Your order #8821 has been dispatched and will arrive within 2–3 business days."
)
# {"label": "negative", "confidence": 0.99, "latency_ms": 38}
```

Batch classification:

```python
messages = [m.text for m in whatsapp_queue]
results = model.classify_batch(messages)

urgent = [m for m, r in zip(whatsapp_queue, results) if r["label"] == "positive"]
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.80` — messages classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
