# social-classifier

Classifies social media mentions and comments as requiring a brand response versus organic content that does not need action — keeping community managers focused on conversations that matter.

Runs locally in under 100ms per message, no network required after installation.

---

## Performance

Trained on BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`) with 2,691 synthetic examples.

| Metric | Synthetic eval | Holdout (hand-authored) |
|--------|---------------|------------------------|
| Accuracy | 99.67% | **86.67%** |
| Macro F1 | 0.9967 | **0.8661** |
| Samples | 299 | 30 |

| Metric | Value |
|--------|-------|
| Model size | 4.3MB (limit: 19MB) |
| Avg latency (CPU) | 0.56 ms |
| Training samples | 2,691 |

Synthetic-real gap: **13.0pp.** The task boundary is inherently fuzzy: organic brand mentions, third-party comparisons, and reshares without commentary sit close to direct questions and complaints in the feature space. Synthetic data keeps these cleaner than real social media text. Use with confidence-threshold filtering in production — low-confidence predictions (below 0.80) should be reviewed rather than acted on automatically. Holdout fixture: [`tests/fixtures/social-classifier.jsonl`](../../tests/fixtures/social-classifier.jsonl).

---

## Usage

```python
from crasis import Specialist

model = Specialist.load("models/social-classifier-onnx")

result = model.classify(
    "@yourbrand your checkout is broken and I've been trying to complete my order for 20 minutes"
)
# {"label": "positive", "confidence": 0.98, "latency_ms": 41}

result = model.classify(
    "Just discovered yourbrand through a friend — looks interesting"
)
# {"label": "negative", "confidence": 0.97, "latency_ms": 39}
```

Batch classification:

```python
mentions = [m.text for m in social_feed]
results = model.classify_batch(mentions)

needs_response = [m for m, r in zip(social_feed, results) if r["label"] == "positive"]
```

---

## Spec

See [`spec.yaml`](spec.yaml) for the full training configuration.

Key settings:
- `confidence_threshold: 0.80` — messages classified below this confidence are logged for potential retraining
- `log_low_confidence: true` — low-confidence inputs are saved to disk for review
- `augmentation: true` — text augmentation was applied during training to improve robustness
