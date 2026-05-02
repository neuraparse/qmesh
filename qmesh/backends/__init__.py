"""qmesh.backends — adapters from qmesh.ir to vendor SDKs / simulators.

Each backend declares Capabilities. The router uses these to decide which
backend a circuit can run on and how it should score.
"""

from __future__ import annotations

from qmesh.backends.base import Backend, Capabilities, RunResult, ControlLevel
from qmesh.backends.registry import register, all_backends, get
from qmesh.backends import statevec_sim  # noqa: F401  (registers itself)

# optional vendor backends — register when their package is importable
try:
    from qmesh.backends import aer_sim  # noqa: F401
except ImportError:
    pass

try:
    from qmesh.backends import stim_sim  # noqa: F401
except ImportError:
    pass

try:
    from qmesh.backends import pulser_sim  # noqa: F401
except ImportError:
    pass

try:
    from qmesh.backends import sf_sim  # noqa: F401
except ImportError:
    pass

# OpenPulse backend (Phase 3δ) self-registers; gracefully α-degrades when
# qiskit.pulse is missing — no try/except needed.
from qmesh.backends import openpulse_sim  # noqa: F401

__all__ = [
    "Backend",
    "Capabilities",
    "RunResult",
    "ControlLevel",
    "register",
    "all_backends",
    "get",
]
