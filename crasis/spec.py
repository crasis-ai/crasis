# crasis/spec.py

import logging
from pydantic import BaseModel, AnyHttpUrl, field_validator, model_validator
from enum import Enum
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Supported task types for specialist models."""

    binary_classification = "binary_classification"
    multiclass = "multiclass"
    extraction = "extraction"
    sequence = "sequence"


class TaskSpec(BaseModel):
    """Defines what the specialist classifies and what constitutes a positive example."""

    type: TaskType
    trigger: str
    ignore: str
    classes: list[str] | None = None
    class_descriptions: dict[str, str] | None = None

    @model_validator(mode="after")
    def classes_required_for_multiclass(self) -> "TaskSpec":
        if self.type == TaskType.multiclass and not self.classes:
            raise ValueError("classes is required for multiclass tasks")
        return self


class ConstraintsSpec(BaseModel):
    """Hardware and deployment constraints that drive architecture selection."""

    max_model_size_mb: int = 27
    max_inference_ms: int = 100
    connectivity: Literal["none", "optional", "required"] = "none"
    target_hardware: Literal["cpu_only", "gpu_optional", "gpu_required"] = "cpu_only"


class QualitySpec(BaseModel):
    """Minimum quality thresholds. Training fails if these are not met."""

    min_accuracy: float
    min_f1: float | None = None
    eval_on: list[str] = []


class TrainingSpec(BaseModel):
    """Controls data generation strategy and training volume."""

    strategy: Literal["synthetic", "hybrid", "real_data"] = "synthetic"
    volume: int
    augmentation: bool = True


class TelemetrySpec(BaseModel):
    """Confidence monitoring and retrain trigger configuration."""

    enabled: bool = True
    confidence_threshold: float = 0.80
    log_low_confidence: bool = True


class CrasisSpec(BaseModel):
    """
    The complete spec for a Crasis specialist. This is the source of truth for
    data generation, architecture selection, quality gates, and telemetry.
    """

    crasis_spec: Literal["v1"] = "v1"
    name: str  # validated: kebab-case only
    description: str
    task: TaskSpec
    constraints: ConstraintsSpec = ConstraintsSpec()
    quality: QualitySpec
    training: TrainingSpec
    telemetry: TelemetrySpec = TelemetrySpec()

    @field_validator("name")
    def name_must_be_kebab(cls, v):
        import re

        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", v):
            raise ValueError("name must be kebab-case (e.g. refund-detector)")
        return v

    @property
    def label_names(self) -> list[str]:
        """Ordered list of class label strings for this specialist."""
        if self.task.type == TaskType.binary_classification:
            return ["negative", "positive"]
        if self.task.type == TaskType.multiclass:
            return self.task.classes or []
        # extraction / sequence — no fixed label set
        return []

    @property
    def num_labels(self) -> int:
        """Number of output classes."""
        return len(self.label_names)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CrasisSpec":
        """Load a bare CrasisSpec from a YAML file (no BuildRequest wrapper)."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        # Strip build wrapper if present (e.g. when passed a full BuildRequest YAML)
        if "spec" in data:
            data = data["spec"]
        return cls.model_validate(data)

    def spec_hash(self) -> str:
        """
        Returns a 16-char SHA-256 hex digest of the canonical sorted JSON representation.

        This hash is the cache key for generated training data. Any change to spec
        fields invalidates the cache and triggers a full pipeline rebuild.
        """
        import hashlib
        import json

        canonical = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class BuildConfig(BaseModel):
    """
    Pipeline execution options for a build. Contains no secrets — the hosted
    platform injects the OpenRouter key server-side before calling factory.py.
    """

    dry_run: bool = False
    notify_webhook: AnyHttpUrl | None = None


class BuildRequest(BaseModel):
    """
    The wire format for the hosted pipeline. Wraps a CrasisSpec with optional
    BuildConfig. YAML files are deserialized into this model via from_yaml().
    """

    spec: CrasisSpec
    build: BuildConfig = BuildConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "BuildRequest":
        """Load from a local YAML spec file. Wraps bare spec in a BuildRequest."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        # Bare spec YAML (no 'build' key) is valid — wrap it
        if "spec" not in data:
            data = {"spec": data}
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, data: dict) -> "BuildRequest":
        """Deserialize a BuildRequest from a raw dict (e.g. parsed JSON POST body)."""
        return cls.model_validate(data)
