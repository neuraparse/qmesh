"""qmesh.backends.base — abstract Backend + Capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from qmesh.ir.module import Module
from qmesh.ir.types import Modality


class ControlLevel(str, Enum):
    NONE = "none"
    IF = "if"        # if_test only (current Qiskit Runtime, Braket)
    LOOP = "loop"    # if + while/for
    FULL = "full"    # full classical control flow (Quantinuum, Q#)


@dataclass(slots=True)
class Capabilities:
    name: str
    vendor: str
    modalities: set[Modality]
    qubit_count: int
    native_gates: set[str] = field(default_factory=set)
    measurement_feedforward: bool = False
    feedforward_latency_ns: int | None = None
    classical_control: ControlLevel = ControlLevel.NONE
    pulse_access: bool = False
    cost_per_shot_usd: float | None = None
    queue_depth: int | None = None
    fidelity_2q_typical: float = 0.99
    is_simulator: bool = False
    notes: str = ""


@dataclass(slots=True)
class RunResult:
    counts: dict[str, int]
    shots: int
    wall_seconds: float
    qpu_seconds: float | None = None
    cost_usd: float | None = None
    raw: Any | None = None  # vendor-native result blob, kept for the manifest
    backend_metadata: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Abstract qmesh backend.

    Concrete backends:
        - inspect a Module's modalities and ops in `accepts(...)`
        - lower the IR to their native form in `run(...)`
        - return a RunResult that the provenance layer captures
    """

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    def accepts(self, module: Module) -> tuple[bool, str]:
        """Return (ok, reason). Default: check modality coverage only."""
        used: set[Modality] = set()
        for f in module.functions:
            for op in f.body.ops:
                used.add(op.modality)
        unsupported = used - self.capabilities.modalities - {Modality.CLASSICAL}
        if unsupported:
            return False, f"unsupported modalities: {[m.value for m in unsupported]}"
        return True, "ok"

    @abstractmethod
    def run(self, module: Module, shots: int = 1024, **kwargs: Any) -> RunResult: ...

    def estimate_cost(self, module: Module, shots: int = 1024) -> float:
        if self.capabilities.cost_per_shot_usd is None:
            return 0.0
        return self.capabilities.cost_per_shot_usd * shots
