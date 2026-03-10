# Changelog

All notable changes to this project will be documented in this file.

## [1.1.2] - 2026-03-09

### Added
- `crasis pull <name>` — download pre-built specialists from the GitHub release registry, no training required
- `crasis mix` — retrain a specialist by mixing real-world JSONL examples with synthetic training data; validates labels, oversamples real data (configurable `--real-weight`), exports to a timestamped ONNX package
- Ten pre-built specialists published as release assets — all `crasis pull` commands now work end-to-end

### Changed
- Default install (`pip install crasis`) is now inference-only — ~15MB instead of ~2GB. PyTorch and training dependencies moved to `pip install crasis[train]`
- `crasis generate`, `train`, `eval`, `export`, and `build` now print a clear install hint if `[train]` deps are missing, rather than a confusing ImportError

### Removed
- `TelemetrySpec` and `telemetry:` block removed from spec format and all specialist specs — telemetry is antithetical to the local-first value proposition

---

## [1.0.0] - 2026-03-07

### Added
- Crasis spec v1 format — plain English task definitions compile to training contracts (`spec.yaml`)
- Synthetic data generation via OpenRouter with `enforce_distillable_text: true` compliance layer
- BERT-Tiny / BERT-Mini / BERT-Small distillation pipeline on consumer GPU (RTX 4060)
- ONNX export with int8 quantization — CPU inference, no GPU required at runtime
- `Specialist.load()` — single-call inference, <1ms median latency on laptop CPU
- Ten pre-built specialists covering email, WhatsApp, support, and social workflows
- Quality gate enforcement — training fails hard if holdout thresholds are not met
- Stratified train/eval split
- Hand-authored holdout evaluation — real-world accuracy documented for all specialists in SCORECARD.md
- `crasis generate / train / eval / export / classify / build / pull / mix` CLI
- `enforce_distillable_text: true` enforced on all data generation — legal compliance layer

### Architecture
- ONNX as the universal deployment target (not PyTorch-only, not GGUF)
- OpenRouter compliance layer — all training data from distillable model providers
- Spec hash as sole cache key — deterministic, reproducible builds
- No internet connection required at inference time
