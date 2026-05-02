"""qmesh.frontends.sf — Strawberry Fields ↔ qmesh.ir.

Xanadu's Strawberry Fields programs operate on `Qumode`s with continuous-
variable gates: Sgate (squeezing), Dgate (displacement), BSgate (beam
splitter), Rgate (rotation), Pgate, Vgate, etc., plus MeasureFock and
MeasureHomodyne.

We translate:
    SF Program → qmesh.ir Module with CVOps
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qmesh.ir.builder import circuit
from qmesh.ir.module import Module

if TYPE_CHECKING:
    import strawberryfields as sf  # noqa: F401


def from_sf(program, *, name: str = "sf_program") -> Module:
    """Translate a Strawberry Fields Program into qmesh.ir."""
    n_qumodes = program.num_subsystems
    # Each Fock measurement produces one classical int; we allocate one bit
    # per measurement (over-allocation is harmless).
    n_bits = sum(1 for cmd in program.circuit if "Measure" in type(cmd.op).__name__)

    with circuit(name, n_qubits=0, n_bits=n_bits, n_qumodes=n_qumodes) as c:
        meas_idx = 0
        for cmd in program.circuit:
            op_name = type(cmd.op).__name__
            mode_idx = [r.ind for r in cmd.reg]
            params = tuple(float(p) if not hasattr(p, "evaluate") else float(p.evaluate())
                           for p in cmd.op.p)
            if op_name.startswith("Measure"):
                # Measurement on each qumode in the op
                for mi in mode_idx:
                    c._region.append(_cv_measure_op(c.qumodes[mi], meas_idx, kind=op_name))
                    meas_idx += 1
            else:
                c.cv(op_name, mode_idx, **{f"p{i}": v for i, v in enumerate(params)})
    return c.module


def _cv_measure_op(qumode, bit_idx: int, *, kind: str):
    from qmesh.ir.ops import MeasureOp
    from qmesh.ir.types import Bit, Modality
    return MeasureOp(
        operands=(qumode, Bit(name=f"c{bit_idx}", index=bit_idx)),
        modality=Modality.CV,
        attrs={"basis": kind},
    )


__all__ = ["from_sf"]
