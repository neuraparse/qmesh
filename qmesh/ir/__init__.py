"""qmesh.ir — modality-agnostic typed IR.

The IR is shaped after MLIR/QIR conventions but keeps a Pythonic builder API.
It distinguishes four execution modalities (GATE, RYDBERG, CV, PULSE) and
provides shared module/function/region/op containers.

Canonicalisation is deterministic so two semantically-equal modules hash to
the same value — a precondition for reproducibility.
"""

from __future__ import annotations

from qmesh.ir.module import Function, Module, Region
from qmesh.ir.ops import (
    BarrierOp,
    ChannelOp,
    ClassicalOp,
    CVOp,
    DelayOp,
    GateOp,
    MeasureOp,
    Op,
    PulseOp,
    QECOp,
    ResetOp,
    RydbergOp,
)
from qmesh.ir.types import (
    Atom,
    Bit,
    Channel,
    ClassicalFloat,
    ClassicalInt,
    Modality,
    QType,
    Qubit,
    Qumode,
)

__all__ = [
    "Modality",
    "QType",
    "Qubit",
    "Qumode",
    "Atom",
    "Bit",
    "ClassicalInt",
    "ClassicalFloat",
    "Channel",
    "Op",
    "GateOp",
    "MeasureOp",
    "ResetOp",
    "DelayOp",
    "BarrierOp",
    "PulseOp",
    "RydbergOp",
    "CVOp",
    "ChannelOp",
    "ClassicalOp",
    "QECOp",
    "Module",
    "Function",
    "Region",
]
