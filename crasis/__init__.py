"""
Crasis — distill frontier model intelligence into tiny, local, task-specific specialists.

Public API:
    from crasis import Specialist          # local inference
    from crasis.spec import CrasisSpec     # spec validation
    from crasis.factory import generate    # data generation
    from crasis.train import train         # distillation
    from crasis.export import export       # ONNX packaging
    from crasis.tools import CrasisToolkit            # LLM tool definitions
    from crasis.orchestrator import CrasisOrchestrator, OrchestratorResult  # agentic loop
"""

from crasis.deploy import Specialist
from crasis.spec import CrasisSpec, BuildRequest, BuildConfig
from crasis.tools import CrasisToolkit
from crasis.orchestrator import CrasisOrchestrator, OrchestratorResult

__all__ = [
    "Specialist",
    "CrasisSpec",
    "BuildRequest",
    "BuildConfig",
    "CrasisToolkit",
    "CrasisOrchestrator",
    "OrchestratorResult",
]
