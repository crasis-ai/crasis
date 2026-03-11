# Setup — Run Your First Specialist

This guide takes you from zero to a trained, local sentiment-gate specialist.

**Prerequisites:** Python 3.11+, pip, an OpenRouter API key (for training only).

---

## Option A: Use a pre-built specialist (no training required)

If you just want to run a specialist locally, skip all the way to step 4.
No GPU, no API key, no PyTorch needed.

```bash
pip install crasis
crasis pull sentiment-gate
crasis classify --model ~/.crasis/specialists/sentiment-gate "I want my money back RIGHT NOW"
```

---

## Option B: Build your own specialist

### 1. Install

```bash
git clone https://github.com/crasis-ai/crasis.git
cd crasis
pip install -e ".[train]"
```

The base install (`pip install crasis`) is inference-only (~15MB + transformers).
The `[train]` group adds PyTorch, datasets, and the ONNX export toolchain.

---

### 2. Set your OpenRouter API key

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Or add it to a `.env` file in the repo root (gitignored). The key is only needed
at training time — inference never touches it.

---

### 3. Generate training data

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

### 4. Train the specialist

```bash
crasis train \
  --spec specialists/sentiment-gate/spec.yaml \
  --data ./data/sentiment-gate/train.jsonl \
  --output ./models
```

Trains a BERT-Tiny classifier on your GPU. 5 epochs, batch size 32.

**Time:** ~8–12 minutes on RTX 4060
**GPU:** ~2GB VRAM

Output: `./models/sentiment-gate/` (PyTorch weights + tokenizer)

---

### 5. Export to ONNX

```bash
crasis export \
  --spec specialists/sentiment-gate/spec.yaml \
  --model ./models/sentiment-gate \
  --output ./models
```

Converts to ONNX. This is the deployable artifact.

Output: `./models/sentiment-gate-onnx/sentiment-gate.onnx` (~4.3MB)

---

### Or: full pipeline in one command

```bash
crasis build --spec specialists/sentiment-gate/spec.yaml
```

Runs all steps automatically: generate → train → export.

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
│ I want my money back RIGHT NOW                │ positive │ 0.967      │ 0.6ms   │
│ The delivery was a bit late, could you check? │ negative │ 0.954      │ 0.6ms   │
│ YOU PEOPLE ARE ABSOLUTELY USELESS!!!          │ positive │ 0.991      │ 0.6ms   │
│ I'm disappointed with the quality             │ negative │ 0.923      │ 0.6ms   │
└──────────────────────────────────────────────┴──────────┴────────────┴─────────┘
```

---

## Python API

```python
from crasis import Specialist

model = Specialist.load("./models/sentiment-gate-onnx")

result = model.classify("I WANT MY MONEY BACK THIS IS RIDICULOUS")
# → {"label": "positive", "label_id": 1, "confidence": 0.97, "latency_ms": 0.6}

result = model.classify("The delivery took longer than expected")
# → {"label": "negative", "label_id": 0, "confidence": 0.95, "latency_ms": 0.6}
```

---

## Retraining with real data

Once you've collected real-world examples — emails that were mislabeled, spam that slipped through, messages that were borderline — feed them back in with `crasis mix`:

```bash
# Format your examples as JSONL
# {"text": "Win a free iPhone click here", "label": "positive"}
# {"text": "Your invoice #4821 is attached", "label": "negative"}

crasis mix \
  --spec specialists/sentiment-gate/spec.yaml \
  --real-data ./my-examples.jsonl
```

Valid label names are printed at the start of the run. `crasis mix` validates every row, merges your real examples with the existing synthetic data (oversampled 3x by default), retrains, and exports a new ONNX to a timestamped directory. The original model is never overwritten.

Options:

```bash
# Tune how heavily real data is weighted vs synthetic
crasis mix --spec ... --real-data ... --real-weight 5

# Point at a specific synthetic dataset instead of auto-discovery
crasis mix --spec ... --real-data ... \
           --synthetic-data ./data/sentiment-gate/train.jsonl

# Force CPU training
crasis mix --spec ... --real-data ... --device cpu
```

If no synthetic data is found, `crasis mix` trains on real data only — useful if you've accumulated enough real examples to stand alone.

---

## Use with agents

Once you have a local specialist, plug it into any agent framework as a tool. The frontier model calls specialists for classification decisions instead of burning tokens:

```python
from crasis.tools import CrasisToolkit

toolkit = CrasisToolkit.from_dir("./models")

# Drop-in tools for Anthropic, OpenAI, or Gemini
toolkit.anthropic_tools()   # pass as tools= to messages.create
toolkit.openai_tools()      # pass as tools= to chat.completions.create
toolkit.gemini_tools()      # pass to GenerativeModel
```

Or use the MCP server to expose specialists as native tools in Claude Desktop and Claude Code:

```bash
crasis mcp --models-dir ./models
```

See [docs/AGENTIC.md](AGENTIC.md) for the full agent integration guide.

---

## Troubleshooting

**`OPENROUTER_API_KEY` not set**
```
Error: Missing option '--api-key'. Set OPENROUTER_API_KEY environment variable.
```
Solution: `export OPENROUTER_API_KEY=sk-or-v1-...`

**Train dependencies not installed**
```
Train dependencies are not installed.
  Run: pip install crasis[train]
```
Solution: `pip install crasis[train]` — required for generate, train, export, and build.

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
