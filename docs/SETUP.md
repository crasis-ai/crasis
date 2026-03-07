# Setup — Run Your First Specialist

This guide takes you from zero to a trained, local sentiment-gate specialist.

**Prerequisites:** Python 3.11+, pip, an RTX 4060 (or any CUDA GPU), OpenRouter API key.

---

## 1. Install

```bash
git clone https://github.com/your-org/crasis.git
cd crasis
pip install -e .
```

Installs the `crasis` CLI and all dependencies (~2GB for PyTorch + transformers).

---

## 2. Set your OpenRouter API key

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Or add it to your shell profile. The `crasis` CLI reads this env var automatically.

---

## 3. Generate training data

```bash
crasis generate \
  --spec specialists/sentiment-gate/spec.yaml \
  --output ./data
```

This calls OpenRouter with `enforce_distillable_text: true`, generating 3,000 labeled
examples of angry vs non-angry customer messages.

**Time:** ~15–20 minutes  
**Cost:** ~$1.50 in OpenRouter credits (at Llama 3.1 70B pricing)

Output: `./data/sentiment-gate/train.jsonl`

---

## 4. Train the specialist

```bash
crasis train \
  --spec specialists/sentiment-gate/spec.yaml \
  --data ./data/sentiment-gate/train.jsonl \
  --output ./models
```

Trains a BERT-mini classifier on your GPU. 5 epochs, batch size 32.

**Time:** ~8–12 minutes on RTX 4060  
**GPU:** ~2GB VRAM  

Output: `./models/sentiment-gate/` (PyTorch weights + tokenizer)

---

## 5. Export to ONNX

```bash
crasis export \
  --spec specialists/sentiment-gate/spec.yaml \
  --model ./models/sentiment-gate \
  --output ./models
```

Converts to ONNX. This is the deployable artifact.

Output: `./models/sentiment-gate/sentiment-gate.onnx` (~17MB)

---

## 6. Classify text

```bash
# Single input
crasis classify --model ./models/sentiment-gate-onnx "I want my money back RIGHT NOW"

# Multiple inputs
crasis classify --model ./models/sentiment-gate-onnx \
  "I want my money back RIGHT NOW" \
  "The delivery was a bit late, could you check?" \
  "YOU PEOPLE ARE ABSOLUTELY USELESS!!!" \
  "I'm disappointed with the quality"

# From a file
crasis classify --model ./models/sentiment-gate-onnx --file test_inputs.txt
```

Expected output:
```
Specialist: sentiment-gate
┌──────────────────────────────────────────────┬──────────┬────────────┬─────────┐
│ Text                                          │ Label    │ Confidence │ Latency │
├──────────────────────────────────────────────┼──────────┼────────────┼─────────┤
│ I want my money back RIGHT NOW                │ positive │ 0.967      │ 42.1ms  │
│ The delivery was a bit late, could you check? │ negative │ 0.954      │ 38.7ms  │
│ YOU PEOPLE ARE ABSOLUTELY USELESS!!!          │ positive │ 0.991      │ 39.2ms  │
│ I'm disappointed with the quality             │ negative │ 0.923      │ 37.8ms  │
└──────────────────────────────────────────────┴──────────┴────────────┴─────────┘
```

---

## Or: full pipeline in one command

```bash
crasis build --spec specialists/sentiment-gate/spec.yaml
```

Runs all steps automatically: generate → train → export.

---

## Python API

```python
from crasis import Specialist

model = Specialist.load("./models/sentiment-gate-onnx")

result = model.classify("I WANT MY MONEY BACK THIS IS RIDICULOUS")
# → {"label": "positive", "label_id": 1, "confidence": 0.97, "latency_ms": 42}

result = model.classify("The delivery took longer than expected")
# → {"label": "negative", "label_id": 0, "confidence": 0.95, "latency_ms": 38}
```

---

## Troubleshooting

**`OPENROUTER_API_KEY` not set**
```
Error: Missing option '--api-key'. Set OPENROUTER_API_KEY environment variable.
```
Solution: `export OPENROUTER_API_KEY=sk-or-v1-...`

**CUDA out of memory**
```
torch.cuda.OutOfMemoryError
```
Solution: `crasis train --device cpu` — slower (~45min) but works on any machine.

**Quality gate failed**
```
QualityGateError: accuracy 0.89 < required 0.92
```
Solution: Re-run `crasis generate` with `--no-resume` to get fresh data, then retrain.
This is rare with 3,000 samples but can happen with unlucky batches.
