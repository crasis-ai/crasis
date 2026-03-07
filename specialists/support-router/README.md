# support-router

Routes incoming customer support messages to the correct team so that tickets reach the right people without manual triage or keyword-based routing rules that fail on ambiguous phrasing.

Runs locally in under 100ms per message, no network required after installation.

---

## Classes

| Label | Meaning |
|-------|---------|
| `billing` | Invoices, charges, refunds, subscription changes, payment failures |
| `technical` | Bugs, errors, performance issues, integration failures, feature behavior questions |
| `returns` | Return requests, exchanges, damaged goods, warranty claims |
| `account` | Login issues, password resets, profile changes, account access and permissions |
| `general` | Everything else — product questions, feedback, partnership inquiries |

---

## Performance

Trained on BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`) with 4,317 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 96.25% | **84.00%** |
| Macro F1 | 0.9621 | **0.8336** |
| Samples | 480 | 25 |

| Metric | Value |
|--------|-------|
| Model size | 4.3MB (limit: 35MB) |
| Avg latency (CPU) | 0.57 ms |
| Training samples | 4,317 |

Synthetic-real gap: **12.3pp.** Five-class routing with overlapping categories (`billing` vs `account`, `technical` vs `general`) is a hard boundary for a small model. Real support tickets are frequently ambiguous in ways that synthetic examples are not — a real message about a login problem after a payment failure can reasonably belong to `billing`, `technical`, or `account`. For production use, review misroutes on the `general` bucket and consider a confidence threshold below which tickets are held for human assignment. Holdout fixture: [`tests/fixtures/support-router.jsonl`](../../tests/fixtures/support-router.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/support-router-onnx")

result = model.classify(
    "I was charged twice for my subscription this month. Can you refund one of the payments?"
)
# {"label": "billing", "confidence": 0.97, "latency_ms": 42}

result = model.classify(
    "The API is returning a 500 error whenever I try to upload files larger than 10MB."
)
# {"label": "technical", "confidence": 0.96, "latency_ms": 40}

result = model.classify(
    "I can't log in — it says my account doesn't exist but I've been a customer for two years."
)
# {"label": "account", "confidence": 0.95, "latency_ms": 41}
```

Batch classification:

```python
tickets = [t.body for t in incoming_queue]
results = model.classify_batch(tickets)

for ticket, result in zip(incoming_queue, results):
    route_to_team(ticket, team=result["label"])
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.78` — tickets classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
