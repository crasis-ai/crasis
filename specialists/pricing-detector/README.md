# pricing-detector

Detects messages that contain a pricing inquiry, quote request, or question about cost — enabling sales teams to route or prioritize revenue-generating conversations automatically.

Runs locally in under 100ms per message, no network required after installation.

---

## Performance

Trained on BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`) with 3,511 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 99.74% | **93.33%** |
| Macro F1 | 0.9974 | **0.9330** |
| Samples | 391 | 30 |

| Metric | Value |
|--------|-------|
| Model size | 4.3MB (limit: 22MB) |
| Avg latency (CPU) | 0.57 ms |
| Training samples | 3,511 |

Synthetic-real gap: **6.4pp.** Performance degrades slightly on edge cases — billing complaints that mention price in passing, and indirect phrasing like "do you have a free tier" — which the synthetic generator tends to keep more distinct. Real-world holdout accuracy of 93.3% is still strong enough for production routing. Monitor confidence scores on messages that mention billing without a direct pricing question. Holdout fixture: [`tests/fixtures/pricing-detector.jsonl`](../../tests/fixtures/pricing-detector.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/pricing-detector-onnx")

result = model.classify(
    "What's your pricing for the team plan? Do you offer annual discounts?"
)
# {"label": "positive", "confidence": 0.99, "latency_ms": 40}

result = model.classify(
    "I noticed a charge on my account that I didn't expect. Can you explain it?"
)
# {"label": "negative", "confidence": 0.96, "latency_ms": 38}
```

Batch classification:

```python
messages = [m.body for m in contact_form_submissions]
results = model.classify_batch(messages)

sales_leads = [m for m, r in zip(contact_form_submissions, results) if r["label"] == "positive"]
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.80` — messages classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
