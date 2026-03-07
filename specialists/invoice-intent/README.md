# invoice-intent

Detects whether an incoming message contains an invoice-related action request — payment confirmation, dispute, request for a copy, or overdue notice follow-up — so finance teams can prioritize their queue automatically.

Runs locally in under 100ms per message, no network required after installation.

---

## Performance

Trained on BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`) with 4,517 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 99.60% | **96.67%** |
| Macro F1 | 0.9960 | **0.9666** |
| Samples | 502 | 30 |

| Metric | Value |
|--------|-------|
| Model size | 4.3MB (limit: 27MB) |
| Avg latency (CPU) | 0.55 ms |
| Training samples | 4,517 |

Synthetic-real gap: **2.9pp.** One of the two best-performing specialists on holdout eval. Invoice-specific language (`invoice #`, payment confirmation, dispute, reissue) is concrete and consistent enough that synthetic data transfers well to real messages. The spec boundary — invoice action requests versus general billing questions and pricing inquiries — holds cleanly in practice. Suitable for production use. Holdout fixture: [`tests/fixtures/invoice-intent.jsonl`](../../tests/fixtures/invoice-intent.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/invoice-intent-onnx")

result = model.classify(
    "Hi, I need to dispute a charge on invoice #4821. The amount doesn't match what we agreed on."
)
# {"label": "positive", "confidence": 0.98, "latency_ms": 40}

result = model.classify(
    "Can you send me your pricing sheet for the enterprise plan?"
)
# {"label": "negative", "confidence": 0.97, "latency_ms": 38}
```

Batch classification:

```python
tickets = [t.body for t in support_queue]
results = model.classify_batch(tickets)

invoice_queue = [t for t, r in zip(support_queue, results) if r["label"] == "positive"]
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.80` — messages classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
