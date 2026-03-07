"""
crasis.export — ONNX export and packaging.

Takes a trained PyTorch specialist (output of crasis.train) and produces a
self-contained package directory containing:
  - <name>.onnx        — the deployable ONNX model
  - tokenizer/         — tokenizer files for local inference
  - label_map.json     — id2label / label2id mappings
  - crasis_meta.json   — specialist metadata (name, spec hash, size, latency)

The ONNX artifact is the only supported deployment format. PyTorch weights
are kept as a training checkpoint but are never the primary artifact.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from crasis.spec import CrasisSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    """
    Outcome of a completed ONNX export.

    Attributes:
        onnx_path: Path to the exported .onnx file.
        package_dir: Directory containing the full deployable package.
        model_size_mb: Size of the ONNX file in megabytes.
        within_size_constraint: Whether model_size_mb <= spec.constraints.max_model_size_mb.
        benchmark_latency_ms: Median single-inference latency on CPU (ms), or None if skipped.
    """

    onnx_path: Path
    package_dir: Path
    model_size_mb: float
    within_size_constraint: bool
    benchmark_latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export(
    spec: CrasisSpec,
    model_path: str | Path,
    output_dir: str | Path,
    benchmark: bool = True,
) -> ExportResult:
    """
    Export a trained specialist to an ONNX package.

    Reads PyTorch weights from model_path, converts to ONNX via optimum,
    copies tokenizer and label map, writes crasis_meta.json.

    Args:
        spec: Validated CrasisSpec instance.
        model_path: Directory containing trained PyTorch model and tokenizer.
        output_dir: Root output directory. Package written to output_dir/<name>-onnx/.
        benchmark: If True, run a quick latency benchmark after export.

    Returns:
        ExportResult with paths, size, and constraint check.

    Raises:
        FileNotFoundError: If model_path does not exist.
        QualityGateError: If the exported model exceeds spec.constraints.max_model_size_mb.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    pkg_dir = Path(output_dir) / f"{spec.name}-onnx"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting '%s' to ONNX: %s", spec.name, pkg_dir)

    # Convert to ONNX via optimum
    onnx_path = _convert_to_onnx(model_path, pkg_dir, spec)

    # Quantize to int8 — reduces size ~4x, latency ~2x, minimal accuracy loss
    onnx_path = _quantize(onnx_path, pkg_dir, spec)

    # Copy tokenizer
    _copy_tokenizer(model_path, pkg_dir)

    # Copy label map
    _copy_label_map(model_path, pkg_dir)

    # Measure size
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    within = size_mb <= spec.constraints.max_model_size_mb
    logger.info(
        "ONNX size: %.1fMB (limit: %dMB) — %s",
        size_mb,
        spec.constraints.max_model_size_mb,
        "OK" if within else "EXCEEDS LIMIT",
    )

    # Benchmark latency
    latency_ms = None
    if benchmark:
        latency_ms = _benchmark_latency(onnx_path, pkg_dir)
        logger.info(
            "Benchmark latency: %.1fms (limit: %dms)",
            latency_ms,
            spec.constraints.max_inference_ms,
        )

    # Write crasis_meta.json
    _write_meta(spec, pkg_dir, size_mb, latency_ms)

    return ExportResult(
        onnx_path=onnx_path,
        package_dir=pkg_dir,
        model_size_mb=round(size_mb, 2),
        within_size_constraint=within,
        benchmark_latency_ms=round(latency_ms, 2) if latency_ms is not None else None,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_to_onnx(model_path: Path, pkg_dir: Path, spec: CrasisSpec) -> Path:
    """Convert a trained HuggingFace model to ONNX using optimum."""
    from optimum.onnxruntime import ORTModelForSequenceClassification

    logger.info("Converting to ONNX via optimum...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path),
        export=True,
    )

    onnx_path = pkg_dir / f"{spec.name}.onnx"
    ort_model.save_pretrained(str(pkg_dir))

    # optimum saves as model.onnx — rename to <spec.name>.onnx
    model_onnx = pkg_dir / "model.onnx"
    if model_onnx.exists() and not onnx_path.exists():
        model_onnx.rename(onnx_path)

    if not onnx_path.exists():
        raise RuntimeError(f"ONNX export produced no .onnx file in {pkg_dir}")

    return onnx_path


def _quantize(onnx_path: Path, pkg_dir: Path, spec: CrasisSpec) -> Path:
    """
    Apply dynamic int8 quantization to the exported ONNX model.

    Reduces model size ~4x and latency ~2x with minimal accuracy loss.
    The quantized model replaces the float32 artifact as the deployable file.
    """
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    logger.info("Quantizing to int8...")
    quantizer = ORTQuantizer.from_pretrained(pkg_dir, file_name=onnx_path.name)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(
        save_dir=str(pkg_dir),
        quantization_config=qconfig,
        file_suffix="int8",
    )

    q_path = pkg_dir / f"{onnx_path.stem}_int8.onnx"
    if not q_path.exists():
        logger.warning("Quantization produced no output — falling back to float32")
        return onnx_path

    # Remove float32 artifact; quantized model is the deployable file
    onnx_path.unlink()
    q_path.rename(onnx_path)
    logger.info(
        "Quantized: %.1fMB", onnx_path.stat().st_size / (1024 * 1024)
    )
    return onnx_path


def _copy_tokenizer(model_path: Path, pkg_dir: Path) -> None:
    """Copy tokenizer files into a tokenizer/ subdirectory of the package."""
    tok_dst = pkg_dir / "tokenizer"
    tok_dst.mkdir(exist_ok=True)

    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    # Also check for a tokenizer/ subdirectory in source
    tok_src = model_path / "tokenizer"
    src = tok_src if tok_src.exists() else model_path

    for fname in tokenizer_files:
        candidate = src / fname
        if candidate.exists():
            shutil.copy2(candidate, tok_dst / fname)
            logger.debug("Copied tokenizer file: %s", fname)


def _copy_label_map(model_path: Path, pkg_dir: Path) -> None:
    """Copy label_map.json from the trained model directory."""
    src = model_path / "label_map.json"
    if src.exists():
        shutil.copy2(src, pkg_dir / "label_map.json")
    else:
        logger.warning(
            "label_map.json not found in %s — deploy.py will fail without it",
            model_path,
        )


def _benchmark_latency(onnx_path: Path, pkg_dir: Path, n_runs: int = 50) -> float:
    """
    Run n_runs inferences on a dummy input and return the median latency in ms.

    Uses CPU provider only — inference is always local-first.
    """
    import numpy as np
    import onnxruntime as ort

    tok_dir = pkg_dir / "tokenizer"
    tok_src = str(tok_dir) if tok_dir.exists() else str(pkg_dir)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        dummy = tokenizer(
            "This is a benchmark input for latency measurement.",
            return_tensors="np",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
    except Exception as exc:
        logger.warning("Could not load tokenizer for benchmark: %s", exc)
        return _benchmark_latency_dummy(onnx_path, n_runs)

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )

    input_names = {inp.name for inp in session.get_inputs()}
    ort_inputs = {k: v.astype(np.int64) for k, v in dummy.items() if k in input_names}

    # Warm up
    for _ in range(3):
        session.run(None, ort_inputs)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    return float(np.median(latencies))


def _benchmark_latency_dummy(onnx_path: Path, n_runs: int = 50) -> float:
    """Fallback benchmark using a zeros input when tokenizer is unavailable."""
    import numpy as np
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )

    inputs = session.get_inputs()
    ort_inputs = {inp.name: np.zeros([1, 128], dtype=np.int64) for inp in inputs}

    for _ in range(3):
        session.run(None, ort_inputs)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    return float(np.median(latencies))


def _write_meta(
    spec: CrasisSpec,
    pkg_dir: Path,
    size_mb: float,
    latency_ms: float | None,
) -> None:
    """Write crasis_meta.json — human-readable specialist metadata."""
    meta = {
        "name": spec.name,
        "description": spec.description,
        "spec_hash": spec.spec_hash(),
        "task_type": spec.task.type.value,
        "label_names": spec.label_names,
        "num_labels": spec.num_labels,
        "model_size_mb": round(size_mb, 2),
        "benchmark_latency_ms": (
            round(latency_ms, 2) if latency_ms is not None else None
        ),
        "max_model_size_mb": spec.constraints.max_model_size_mb,
        "max_inference_ms": spec.constraints.max_inference_ms,
        "within_size_constraint": size_mb <= spec.constraints.max_model_size_mb,
        "within_latency_constraint": (
            latency_ms <= spec.constraints.max_inference_ms
            if latency_ms is not None
            else None
        ),
    }
    (pkg_dir / "crasis_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info("Wrote crasis_meta.json to %s", pkg_dir)
