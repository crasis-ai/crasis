# Contributing to Crasis

The ten pre-built specialists are the beginning. The goal is 100 community specialists by end of 2026 — one for every recurring task nobody should be paying tokens for anymore.

If you've trained a specialist for something not covered here, we want it in this repo.

---

## What Makes a Good Specialist

A specialist is worth contributing if it meets three criteria:

**It solves a recurring, narrow problem.** Not "understand this document" — that's a generalist task. "Does this message contain a pricing question?" — that's a specialist task. The narrower the better. If you can write the trigger condition in one sentence, it's probably a good specialist.

**It outperforms a frontier model on cost and latency, not just accuracy.** A specialist that's 94% accurate and runs in 40ms locally beats a frontier model that's 97% accurate and costs $0.005 per call — for every use case where you're making the same classification hundreds of times a day.

**It has clean provenance.** All training data must be generated through OpenRouter with `enforce_distillable_text: true`. No exceptions. This is the legal layer. Don't submit specialists trained on data from direct Anthropic, OpenAI, or Google API calls.

---

## The Quality Bar

To be merged, a specialist must pass:

| Requirement | Threshold |
|-------------|-----------|
| Accuracy | ≥ 93% on held-out eval set |
| F1 score | ≥ 91% (required for imbalanced classes) |
| Model size | Within `max_model_size_mb` from spec |
| Inference latency | Within `max_inference_ms` from spec (CPU) |
| Held-out test examples | ≥ 100, included in PR |
| Spec completeness | All required fields present and validated |

These thresholds exist because specialists get pulled and deployed without further evaluation. If it's in the repo, people will trust it.

---

## Step-by-Step: Contributing a Specialist

### 1. Write the spec

Create `specialists/<your-specialist-name>/spec.yaml`. Use an existing specialist spec as a reference — `specialists/spam-filter/spec.yaml` is a good starting point for binary classification.

The spec is the contract. Take your time on `task.trigger` and `task.ignore` — these drive your training data quality more than anything else. Be specific about what you want and explicit about what you don't.

```yaml
crasis_spec: v1
name: your-specialist-name        # kebab-case, becomes the model filename
description: >
  One or two sentences in plain English. What does this model detect?
  Write it for someone who's never seen your code.

task:
  type: binary_classification     # binary_classification | multiclass | extraction
  trigger: >
    Describe positive examples specifically. What words, patterns, or
    intent marks something as a positive case?
  ignore: >
    Describe what looks similar but isn't. Edge cases the model should
    classify negative. This is where most data quality problems start.

constraints:
  max_model_size_mb: 27           # drives architecture selection
  max_inference_ms: 100
  connectivity: none
  target_hardware: cpu_only

quality:
  min_accuracy: 0.93
  min_f1: 0.91
  eval_on:
    - name_your_hard_cases        # what slices do you want tested?

training:
  strategy: synthetic
  volume: 5000                    # 3k minimum, 10k for multiclass
  augmentation: false
```

Note: do not include a `generator:` field — it was removed from the spec format and is now a constant in the pipeline.

Validate your spec before generating data:

```python
from crasis.spec import BuildRequest
req = BuildRequest.from_yaml("specialists/your-specialist-name/spec.yaml")
print(req)  # will raise ValidationError if something is wrong
```

### 2. Generate training data

```bash
crasis generate --spec specialists/your-specialist-name/spec.yaml --count 5000
```

This calls OpenRouter with `enforce_distillable_text: true` and writes labeled examples to `data/your-specialist-name/train.jsonl`. Takes roughly 10 minutes for 5,000 samples.

Spot-check the output before training. Read 50 examples. If the positive and negative examples aren't clearly distinct, improve the spec's `trigger` and `ignore` clauses and regenerate. Bad data at this stage produces a bad model — the training pipeline can't fix label noise.

### 3. Train

```bash
crasis train --spec specialists/your-specialist-name/spec.yaml \
             --data ./data/your-specialist-name/train.jsonl
```

The pipeline will select the smallest architecture that fits your size budget, train on your GPU (or CPU if no GPU is available), and enforce the quality gate from your spec. Training fails with a clear error if the model doesn't meet `min_accuracy` or `min_f1`.

Training time on an RTX 4060 (measured):
- Binary classifier, 4,500 samples: ~7 seconds
- Multiclass (4 classes), 3,700 samples: ~16 seconds

Training is fast. The bottleneck is data generation (~10 minutes for 5,000 samples), not training.

### 4. Export

```bash
crasis export --spec specialists/your-specialist-name/spec.yaml \
              --model ./models/your-specialist-name
```

Exports to ONNX with dynamic int8 quantization. The exported package lands in `models/your-specialist-name-onnx/`. Verify the size is within your spec's `max_model_size_mb` limit.

### 5. Evaluate

Run the exported model against your held-out test set. You need at least 100 examples that were **not** part of training — ideally real examples from the actual domain, not just more synthetic data.

```python
from crasis import Specialist

model = Specialist.load("./models/your-specialist-name-onnx")

# Verify size
import os
onnx_path = "models/your-specialist-name-onnx/your-specialist-name.onnx"
size_mb = os.path.getsize(onnx_path) / 1_000_000
print(f"Model size: {size_mb:.1f}MB")

# Verify latency
result = model.classify("your test input here")
print(result)
# → {"label": "...", "confidence": 0.97, "latency_ms": 38}

# Evaluate on held-out set
correct = 0
for text, expected_label in your_test_set:
    result = model.classify(text)
    if result["label"] == expected_label:
        correct += 1

accuracy = correct / len(your_test_set)
print(f"Held-out accuracy: {accuracy:.1%}")  # must be >= 0.93
```

Save your eval results as `specialists/your-specialist-name/eval_results.json`:

```json
{
  "accuracy": 0.947,
  "f1": 0.943,
  "eval_set_size": 200,
  "model_size_mb": 11.2,
  "avg_latency_ms": 38,
  "trained_on": "2026-03-05",
  "training_samples": 5000,
  "base_model": "google/bert_uncased_L-2_H-128_A-2"
}
```

### 6. Write a README

Create `specialists/your-specialist-name/README.md`. Two sections:

- **What it detects** — plain English, two to three sentences. Who would use this and why.
- **Example inference** — copy-pasteable Python showing `Specialist.load()` and two or three `classify()` calls with real outputs.

Look at `specialists/spam-filter/spec.yaml` as a reference — it's the most complete example currently in the repo.

### 7. Open a pull request

Your PR should include:

```
specialists/your-specialist-name/
├── spec.yaml           ← validated spec
├── eval_results.json   ← accuracy, F1, size, latency
└── README.md           ← what it does + example inference
```

**Do not include** the trained model weights, ONNX files, or training data in the PR. Models are distributed separately. Training data is regenerable from the spec.

In the PR description, include:
- One sentence on what problem this solves
- The accuracy and F1 on your held-out set
- Any edge cases you explicitly tested and how the model handled them
- What the failure modes are — where does the model struggle?

That last one matters. Honest evaluation of failure modes is more useful to adopters than an inflated headline number.

---

## What Gets Rejected

**Missing held-out evaluation.** Eval on the training split doesn't count. 100 examples minimum, ideally more, ideally real-world examples.

**Training data with unclear provenance.** If you can't confirm `enforce_distillable_text: true` was used, the specialist can't be merged.

**Specialists that duplicate existing ones without clear improvement.** A spam filter that's 94% accurate when the existing one is 98% is not an improvement. If you've trained a better version of an existing specialist, open an issue first to discuss.

**Overly broad task definitions.** "Classify whether this email needs a response" is too broad — it depends on who you are and what your inbox looks like. "Classify whether this email contains an explicit yes/no question addressed to the recipient" is a specialist task.

**Models above the size limit.** The size constraints exist for a reason. If your task genuinely requires a larger model, update the spec's `max_model_size_mb` with a justification and the trade-off will be evaluated in review.

---

## Getting Help

If your accuracy isn't hitting the bar, the most common causes in order of likelihood:

1. **`task.ignore` is underspecified** — the model doesn't know what the hard negatives look like. Add more detail about edge cases to exclude.
2. **Training volume is too low** — multiclass tasks especially need more data. Try 8k–10k samples.
3. **The task is underspecified overall** — if you can't describe the trigger in two sentences, the generator can't produce good training data from it. Sharpen the spec first.

Open an issue with your spec and eval results and we'll help debug it.

---

## License

By contributing a specialist, you agree that your contribution is licensed under MIT, the same license as this repository. The training data, spec, and eval results are all MIT. The trained model weights, once distributed, carry the same license.

