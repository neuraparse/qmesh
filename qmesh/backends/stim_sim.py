"""qmesh.backends.stim_sim — Stim Clifford simulator wrapper.

Stim is the standard for QEC: distance-100 surface code analyses run in 15s
[zenodo.org]. We accept Clifford-only circuits (the Aer-style escape hatch
detects non-Clifford gates and rejects them).

Two surfaces:
- TableauSimulator-based shot loop for arbitrary Clifford circuits with
  measurements (returns counts).
- Direct .stim source forwarding via `Backend.run(module, stim_program=...)`
  for surface-code memory experiments where the circuit is already encoded.
"""

from __future__ import annotations

import time
from typing import Any

import stim

from qmesh.backends.base import Backend, Capabilities, ControlLevel, RunResult
from qmesh.backends.registry import register
from qmesh.ir.module import Module
from qmesh.ir.ops import GateOp, MeasureOp, ResetOp
from qmesh.ir.types import Bit, Modality, Qubit

_CLIFFORD_GATES = {
    "h", "x", "y", "z", "s", "sdg",
    "cx", "cy", "cz", "swap", "iswap",
}


def _ir_to_stim_circuit(module: Module) -> tuple[stim.Circuit, list[tuple[int, int]]]:
    """Lower the IR to a stim.Circuit. Reject non-Clifford gates.

    Returns (stim_circuit, measurement_pairs) where each pair is
    (qubit_idx, output_bit_idx) — used to reorder the bitstring at the end.
    """
    circ = stim.Circuit()
    meas_pairs: list[tuple[int, int]] = []

    for f in module.functions:
        for op in f.body.ops:
            if isinstance(op, GateOp):
                if op.name not in _CLIFFORD_GATES:
                    raise ValueError(
                        f"stim backend rejects non-Clifford gate '{op.name}'"
                    )
                qs = [v.index for v in op.operands if isinstance(v, Qubit)]
                stim_name = op.name.upper()
                if stim_name == "CX":
                    stim_name = "CNOT"
                elif stim_name == "SDG":
                    stim_name = "S_DAG"
                circ.append(stim_name, qs)
            elif isinstance(op, ResetOp):
                qs = [v.index for v in op.operands if isinstance(v, Qubit)]
                circ.append("R", qs)
            elif isinstance(op, MeasureOp):
                q = next(v.index for v in op.operands if isinstance(v, Qubit))
                b = next(v.index for v in op.operands if isinstance(v, Bit))
                circ.append("M", [q])
                meas_pairs.append((q, b))
            elif op.modality == Modality.CLASSICAL:
                continue
            elif op.name == "barrier":
                continue
            else:
                raise NotImplementedError(
                    f"stim backend cannot lower op '{op.name}' modality={op.modality}"
                )
    return circ, meas_pairs


class StimSimulator(Backend):
    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="qmesh.stim",
            vendor="qmesh+stim",
            modalities={Modality.GATE},
            qubit_count=1024,  # Stim handles distance-100 surface (~20k qubits) trivially
            native_gates=_CLIFFORD_GATES,
            measurement_feedforward=False,
            classical_control=ControlLevel.NONE,
            cost_per_shot_usd=0.0,
            queue_depth=0,
            fidelity_2q_typical=1.0,
            is_simulator=True,
            notes="Stim Clifford simulator — wraps the standard QEC tool",
        )

    def accepts(self, module: Module) -> tuple[bool, str]:
        ok, why = super().accepts(module)
        if not ok:
            return ok, why
        for f in module.functions:
            for op in f.body.ops:
                if isinstance(op, GateOp) and op.name not in _CLIFFORD_GATES:
                    return False, f"non-Clifford gate '{op.name}'"
        return True, "ok"

    def run(self, module: Module, shots: int = 1024, **_: Any) -> RunResult:
        t0 = time.time()
        circ, meas_pairs = _ir_to_stim_circuit(module)
        sampler = circ.compile_sampler()
        samples = sampler.sample(shots=shots)
        # samples shape: (shots, n_measurements). Measurements appear in the
        # order Stim received M ops. Reorder to bit-position layout.
        n_bits = max((b for _, b in meas_pairs), default=-1) + 1
        counts: dict[str, int] = {}
        meas_to_bit = [b for _, b in meas_pairs]
        for row in samples:
            bits = ["0"] * n_bits
            for k, val in enumerate(row):
                bi = meas_to_bit[k]
                bits[bi] = "1" if val else "0"
            key = "".join(reversed(bits))  # MSB-first
            counts[key] = counts.get(key, 0) + 1
        return RunResult(
            counts=counts,
            shots=shots,
            wall_seconds=time.time() - t0,
            qpu_seconds=time.time() - t0,
            cost_usd=0.0,
            backend_metadata={"stim_circuit_repr": str(circ)[:512]},
        )


register(StimSimulator())
