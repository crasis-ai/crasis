"""
crasis.deploy — Local inference wrapper.

The Specialist class is the public inference API. Load once, classify forever.
No internet connection required at inference time — ever.

Usage:
    from crasis import Specialist

    model = Specialist.load("./models/sentiment-gate/")
    result = model.classify("This is absolutely unacceptable!")
    # → {"label": "angry", "label_id": 1, "confidence": 0.97, "latency_ms": 38}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Softmax for confidence scores
def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class Specialist:
    """
    A loaded, ready-to-classify specialist model.

    Wraps an ONNX model and tokenizer for zero-dependency local inference.
    All computation is local. No API calls are made after __init__.

    Attributes:
        name: Specialist name from crasis_meta.json.
        label_names: List of class names in label_id order.
    """

    def __init__(
        self,
        session,  # onnxruntime.InferenceSession
        tokenizer,  # transformers.PreTrainedTokenizer
        label_names: list[str],
        name: str,
    ) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self.label_names = label_names
        self.name = name

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, package_dir: str | Path) -> "Specialist":
        """
        Load a specialist from an exported package directory.

        The directory must contain:
          - A .onnx model file (or model.onnx)
          - A tokenizer/ subdirectory (or tokenizer files in root)
          - label_map.json

        Args:
            package_dir: Path to the exported specialist package.

        Returns:
            A loaded Specialist ready to classify.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        import onnxruntime as ort
        from transformers import AutoTokenizer

        pkg = Path(package_dir)
        if not pkg.exists():
            raise FileNotFoundError(f"Package directory not found: {pkg}")

        # Find ONNX file
        onnx_path = cls._find_onnx(pkg)
        logger.info("Loading ONNX model from %s", onnx_path)

        # ONNX session — CPU provider by default (inference is local-first)
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 4
        sess_opts.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # Tokenizer — prefer tokenizer/ subdirectory, fall back to root
        tok_dir = pkg / "tokenizer"
        tok_src = str(tok_dir) if tok_dir.exists() else str(pkg)
        tokenizer = AutoTokenizer.from_pretrained(tok_src)

        # Label map
        label_map_path = pkg / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"label_map.json not found in {pkg}")
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        id2label: dict[str, str] = label_map["id2label"]
        label_names = [id2label[str(i)] for i in range(len(id2label))]

        # Specialist name
        name = pkg.name
        meta_path = pkg / "crasis_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            name = meta.get("name", name)

        logger.info(
            "Loaded specialist '%s' with %d classes: %s",
            name,
            len(label_names),
            label_names,
        )

        return cls(
            session=session, tokenizer=tokenizer, label_names=label_names, name=name
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, text: str) -> dict:
        """
        Classify a single text input.

        Args:
            text: The input text to classify.

        Returns:
            A dict with:
              - label: str — predicted class name
              - label_id: int — predicted class index
              - confidence: float — softmax probability of predicted class
              - latency_ms: float — inference time in milliseconds

        Example:
            >>> model.classify("I want a refund NOW")
            {"label": "positive", "label_id": 1, "confidence": 0.97, "latency_ms": 38}
        """
        t0 = time.perf_counter()

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128,
            padding="max_length",
        )

        # Run ONNX inference
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        # Add token_type_ids if the model expects it
        if "token_type_ids" in [inp.name for inp in self._session.get_inputs()]:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        logits = self._session.run(None, ort_inputs)[0][0]

        latency_ms = (time.perf_counter() - t0) * 1000

        probs = _softmax(logits)
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])

        return {
            "label": self.label_names[label_id],
            "label_id": label_id,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
        }

    def classify_batch(self, texts: list[str]) -> list[dict]:
        """
        Classify a list of texts.

        This is a convenience wrapper; each text is classified independently.
        For high-throughput use cases, batched ONNX inference can be added here.

        Args:
            texts: List of input strings.

        Returns:
            List of classification result dicts in the same order as inputs.
        """
        return [self.classify(t) for t in texts]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Specialist(name={self.name!r}, " f"classes={self.label_names!r})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_onnx(pkg: Path) -> Path:
        """Locate the ONNX model file within a package directory."""
        # Prefer <name>.onnx, then model.onnx, then any .onnx
        for candidate in [pkg / f"{pkg.name}.onnx", pkg / "model.onnx"]:
            if candidate.exists():
                return candidate

        onnx_files = list(pkg.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]

        raise FileNotFoundError(
            f"No .onnx file found in {pkg}. "
            "Run `crasis export` to generate the ONNX artifact."
        )
