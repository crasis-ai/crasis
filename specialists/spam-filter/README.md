# spam-filter

Classifies incoming messages as spam or legitimate so that inboxes, support queues, and contact forms stay clean without relying on blocklists or regex rules that break on novel phrasing.

Runs locally in under 100ms per message, no network required after installation.

---

## Performance

Trained on BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`) with 4,500 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 99.20% | **70.00%** |
| Macro F1 | 0.9920 | **0.6970** |
| Samples | 500 | 50 |

| Metric | Value |
|--------|-------|
| Model size | 10.86MB (limit: 27MB) |
| Avg latency (CPU) | 2.67 ms |
| Training samples | 4,500 |

Synthetic-real gap: **29.2pp** — the largest gap of all ten specialists. Spam is a hard task: legitimate cold outreach, transactional messages, and thoughtfully worded spam sit very close together in the feature space. The synthetic generator produces spam that is more stereotypically spammy than real-world examples, inflating synthetic accuracy. Not recommended for production use without a hybrid or real-data training strategy. Holdout fixture: [`tests/fixtures/spam-filter.jsonl`](../../tests/fixtures/spam-filter.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/spam-filter-onnx")

result = model.classify(
    "Congratulations! You've been selected for a $500 gift card. Click here to claim your prize now!"
)
# {"label": "positive", "confidence": 0.99, "latency_ms": 44}

result = model.classify(
    "Hi, I saw your post about the API and wanted to ask about enterprise pricing."
)
# {"label": "negative", "confidence": 0.97, "latency_ms": 41}
```

Batch classification:

```python
messages = [msg.body for msg in inbox]
results = model.classify_batch(messages)

clean = [m for m, r in zip(inbox, results) if r["label"] == "negative"]
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.85` — messages classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
