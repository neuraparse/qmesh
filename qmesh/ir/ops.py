"""qmesh.ir.ops — IR operation classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass(slots=True)
class Op:
    """Base IR operation.

    Subclasses pin their `modality` and the kinds of operands they accept.
    A stable `digest()` over operands + params + name supports deterministic
    hashing of IR modules.
    """

    name: str
    operands: tuple[QType, ...] = field(default_factory=tuple)
    params: tuple[float, ...] = field(default_factory=tuple)
    modality: Modality = Modality.GATE
    attrs: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> bytes:
        from hashlib import sha256

        h = sha256()
        h.update(self.name.encode())
        h.update(self.modality.value.encode())
        for op in self.operands:
            h.update(repr(op).encode())
        for p in self.params:
            h.update(repr(p).encode())
        for k in sorted(self.attrs):
            h.update(k.encode())
            h.update(repr(self.attrs[k]).encode())
        return h.digest()


@dataclass(slots=True)
class GateOp(Op):
    modality: Modality = Modality.GATE


@dataclass(slots=True)
class MeasureOp(Op):
    """Mid-circuit or terminal measurement. `target` ∈ {qubit, qumode, atom}.

    Result is stored in `Bit` (or a wider classical register for CV homodyne).
    """

    name: str = "measure"


@dataclass(slots=True)
class ResetOp(Op):
    name: str = "reset"


@dataclass(slots=True)
class DelayOp(Op):
    """Idle delay; param[0] is duration in nanoseconds."""

    name: str = "delay"


@dataclass(slots=True)
class BarrierOp(Op):
    """Compiler barrier. May be entanglement-aware in scheduler context."""

    name: str = "barrier"


@dataclass(slots=True)
class PulseOp(Op):
    """Pulse-level op: amplitude, phase, duration, channel.

    Backends without pulse access reject this at lower-time.
    """

    modality: Modality = Modality.PULSE


@dataclass(slots=True)
class RydbergOp(Op):
    """Neutral-atom analog op: global laser, blockade window, atom-set.

    See Pulser / Bloqade semantics.
    """

    modality: Modality = Modality.RYDBERG


@dataclass(slots=True)
class CVOp(Op):
    """Photonic continuous-variable op.

    Names follow Strawberry Fields: Squeeze, Displace, BSgate, Sgate,
    MeasureHomodyne, MeasureFock, etc.
    """

    modality: Modality = Modality.CV


@dataclass(slots=True)
class ChannelOp(Op):
    """First-class cross-modality entanglement / data carrier.

    Models photonic interconnect (IonQ Apr 2026), entanglement-swap nodes,
    photon-mediated transfer between modules — and, in qmesh, the explicit
    boundary between modality-coherent IR blocks. A ChannelOp inserted
    between a GateOp run and a RydbergOp run is what tells the scheduler
    to lower the program into two QPUPrimitive nodes joined by a small
    ClassicalTask "channel" stub that forwards data (e.g. a measurement
    outcome → a Pulser pulse amplitude).

    Fields:
        source_modality   — modality of the upstream block (Modality)
        target_modality   — modality of the downstream block (Modality)
        kind              — channel kind, free-form string. Phase 3β α
                            handles {"gate->rydberg", "gate->cv",
                            "rydberg->cv"}; the rest are documented as
                            Phase 3γ.
        payload           — small JSON-serialisable dict describing what
                            classical data is forwarded. Typical keys:
                              {"forward": "p_excited",
                               "into_param": "amp_rad_per_us",
                               "scale": 6.28, "default": 0.0}
                            The scheduler reads this to build the bridge
                            ClassicalTask.

    Compat: legacy callers can still pass a `Channel` typed value as an
    operand; if `source_modality`/`target_modality` are unset the op will
    fall back to that operand's `src` / `dst`.
    """

    name: str = "channel"
    modality: Modality = Modality.CLASSICAL  # the channel itself is a classical bridge
    source_modality: Modality | None = None
    target_modality: Modality | None = None
    kind: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # If src/dst weren't given but a Channel operand carries them, pull them up.
        if self.source_modality is None or self.target_modality is None:
            for v in self.operands:
                if isinstance(v, Channel):
                    if self.source_modality is None:
                        self.source_modality = v.src
                    if self.target_modality is None:
                        self.target_modality = v.dst
                    break
        # Default `kind` from src/dst if not user-supplied.
        if not self.kind and self.source_modality and self.target_modality:
            self.kind = f"{self.source_modality.value}->{self.target_modality.value}"

    def digest(self) -> bytes:
        from hashlib import sha256
        h = sha256()
        h.update(Op.digest(self))
        h.update((self.source_modality.value if self.source_modality else "").encode())
        h.update((self.target_modality.value if self.target_modality else "").encode())
        h.update(self.kind.encode())
        for k in sorted(self.payload):
            h.update(k.encode())
            h.update(repr(self.payload[k]).encode())
        return h.digest()


@dataclass(slots=True)
class ClassicalOp(Op):
    """Classical arithmetic / control flow / branching."""

    modality: Modality = Modality.CLASSICAL


@dataclass(slots=True)
class QECOp(Op):
    """QEC primitive: syndrome extraction, decoder hand-off, code switch.

    Lowered by `qmesh.ftmode` when FT mode is enabled; otherwise emitted as
    a no-op or rejected per backend capability.
    """

    name: str = "qec"


__all__ = [
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
]
