# sentiment-gate

Flags negative or hostile customer messages so they can be prioritized for human review before an automated response makes things worse.

Runs locally in under 100ms per message, no network required after installation.

---

## Performance

Trained on BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`) with 2,649 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 99.32% | **80.00%** |
| Macro F1 | 0.9932 | **0.8000** |
| Samples | 295 | 50 |

| Metric | Value |
|--------|-------|
| Model size | 4.3MB (limit: 19MB) |
| Avg latency (CPU) | 0.56 ms |
| Training samples | 2,649 |

Synthetic-real gap: **19.3pp.** The model is over-triggering positive on neutral messages that don't match the synthetic negative examples it learned. Precision on `negative` is 1.00 but recall is 0.67 — it never misses an angry message but cries wolf on calm ones. Consider a hybrid training strategy or a larger base model for production use. Holdout fixture: [`tests/fixtures/sentiment-gate.jsonl`](../../tests/fixtures/sentiment-gate.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/sentiment-gate-onnx")

result = model.classify(
    "This is absolutely unacceptable. I've been waiting three weeks and nobody has bothered to respond."
)
# {"label": "positive", "confidence": 0.98, "latency_ms": 41}

result = model.classify(
    "Could you let me know when my order will ship? No rush, just checking in."
)
# {"label": "negative", "confidence": 0.96, "latency_ms": 39}
```

Batch classification:

```python
messages = [msg.body for msg in support_queue]
results = model.classify_batch(messages)

flagged = [m for m, r in zip(support_queue, results) if r["label"] == "positive"]
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.82` — messages classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
