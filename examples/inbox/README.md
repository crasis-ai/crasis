# examples/inbox

The email triage demo. Classifies 847 synthetic emails locally using the `email-urgency` specialist — no API calls, no data leaving the machine.

---

## Quick start

```bash
# 1. Generate the synthetic inbox (one-time)
python examples/inbox/generate_demo_inbox.py

# 2. Run the demo (requires trained model)
python examples/inbox/run_demo.py

# 3. Or run in mock mode without a model
python examples/inbox/run_demo.py --mock
```

---

## Files

| File | Purpose |
|---|---|
| `generate_demo_inbox.py` | Generates `demo/inbox_847.jsonl` — 847 synthetic emails with realistic senders, subjects, and bodies. No real PII. Deterministic with `--seed`. |
| `run_demo.py` | Classifies the inbox using the `email-urgency` specialist. Shows a live ticker and progress bar, prints the final timing summary. |

The generated inbox is written to `demo/inbox_847.jsonl` by default. The `demo/` directory is gitignored — run the generator to produce it locally.

---

## Training the model first

This demo requires the `email-urgency` specialist to be trained and exported. If you haven't done that yet:

```bash
crasis build --spec specialists/email-urgency/spec.yaml
```

That runs the full pipeline: generate training data → train → eval → export ONNX. Takes ~30 minutes on an RTX 4060. When it finishes, `run_demo.py` will automatically find the model.

---

## Options

### `generate_demo_inbox.py`

```
--seed INT        Random seed for reproducibility (default: 42)
--out PATH        Output path (default: demo/inbox_847.jsonl)
--no-category     Strip the ground-truth category field from output
```

### `run_demo.py`

```
--inbox PATH        Path to inbox JSONL (default: demo/inbox_847.jsonl)
--model PATH        Path to trained ONNX specialist (default: ./models/email-urgency-onnx)
--mock              Run with a deterministic mock classifier — no model required
--target-seconds N  Mock mode only: pace the run to N seconds for recording (default: 11.0)
--ticker-rows N     Number of emails to show in the live ticker (default: 14)
--out PATH          Write classified results to this JSONL path
```

---

## Recording the demo GIF

```bash
# Pace to ~11s so the progress bar is visible during recording
python examples/inbox/run_demo.py --mock --target-seconds 11

# Once the real model is trained — drop --mock entirely
python examples/inbox/run_demo.py
```

The output is designed for terminal recording. Recommended setup: 72-character wide terminal, dark background, font size 14+. The final line reads:

```
✓  847 emails classified in 11.2s  ·  avg latency 2.6ms per email
```

---

## Inbox distribution

The synthetic inbox mirrors a realistic personal/work account:

| Category | Count |
|---|---|
| Newsletter | 210 |
| Work | 185 |
| Receipt/transactional | 130 |
| Urgent | 120 |
| Social notification | 95 |
| Spam | 75 |
| Misc | 32 |
| **Total** | **847** |

All senders, names, and domains are fictional. Safe to use in any public recording.