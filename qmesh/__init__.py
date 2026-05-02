"""qmesh — a 2026-native quantum operating layer.

Modality-agnostic IR, AI-augmented compiler, FT-mode scheduling, and
reproducibility-as-runtime — across gate-based, neutral-atom analog,
photonic-CV, and pulse-level backends.

See ARCHITECTURE.md for the design and PLAN.md for the roadmap.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Neuraparse"
__email__ = "open-source@neuraparse.com"
__license__ = "Apache-2.0"
__url__ = "https://github.com/neuraparse/qmesh"

from qmesh.api import diff, replay, submit
from qmesh.backends.base import Backend, Capabilities, RunResult
from qmesh.ir import Bit, Function, Modality, Module, Op, Qubit, Region
from qmesh.ir.builder import circuit
from qmesh.provenance.manifest import Manifest

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    "Module",
    "Function",
    "Region",
    "Op",
    "Modality",
    "Qubit",
    "Bit",
    "circuit",
    "Backend",
    "Capabilities",
    "RunResult",
    "Manifest",
    "submit",
    "replay",
    "diff",
]
