"""qmesh.frontends.cirq — Cirq ↔ qmesh.ir.

Phase-1: gate-modality only. Cirq's moment structure is flattened. Common
gates are mapped to canonical names; unknown native gates are passed through
verbatim so backends can accept them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qmesh.ir.builder import circuit
from qmesh.ir.module import Module

if TYPE_CHECKING:
    import cirq  # noqa: F401


_NAME_MAP = {
    "H": "h", "X": "x", "Y": "y", "Z": "z", "S": "s", "T": "t",
    "CNOT": "cx", "CX": "cx", "CZ": "cz", "SWAP": "swap",
    "Rx": "rx", "Ry": "ry", "Rz": "rz",
    "rx": "rx", "ry": "ry", "rz": "rz",
    "TOFFOLI": "ccx", "CCX": "ccx",
}


def _qubit_index(q: Any) -> int:
    """Best-effort linear index for cirq qubits (LineQubit / NamedQubit)."""
    import cirq

    if isinstance(q, cirq.LineQubit):
        return q.x
    if isinstance(q, cirq.GridQubit):
        # row-major: row * 100 + col is fine if grids stay small
        return q.row * 100 + q.col
    if isinstance(q, cirq.NamedQubit):
        # extract trailing integer if any
        s = q.name
        digits = ""
        for ch in reversed(s):
            if ch.isdigit():
                digits = ch + digits
            else:
                break
        return int(digits) if digits else 0
    raise NotImplementedError(f"qmesh.cirq: unsupported qubit type {type(q).__name__}")


def from_cirq(cirq_circuit: "cirq.Circuit") -> Module:
    import cirq

    qubits = sorted(cirq_circuit.all_qubits(), key=_qubit_index)
    qubit_map = {q: i for i, q in enumerate(qubits)}
    n_qubits = len(qubits)

    # collect measurement keys → output bit positions
    meas_keys: dict[str, int] = {}
    for moment in cirq_circuit:
        for op in moment.operations:
            if isinstance(op.gate, cirq.MeasurementGate):
                k = op.gate.key
                if k not in meas_keys:
                    meas_keys[k] = len(meas_keys)
    n_bits = len(meas_keys) if meas_keys else n_qubits

    with circuit("cirq", n_qubits=n_qubits, n_bits=n_bits) as c:
        for moment in cirq_circuit:
            for op in moment.operations:
                qs = [qubit_map[q] for q in op.qubits]
                gate = op.gate
                if gate is None:
                    continue
                gname = type(gate).__name__

                if isinstance(gate, cirq.MeasurementGate):
                    bit = meas_keys[gate.key]
                    for qi in qs:
                        c.measure(qi, bit)
                    continue

                if isinstance(gate, cirq.HPowGate) and gate.exponent == 1:
                    c._gate("h", qs)
                    continue
                if isinstance(gate, cirq.XPowGate) and gate.exponent == 1:
                    c._gate("x", qs)
                    continue
                if isinstance(gate, cirq.YPowGate) and gate.exponent == 1:
                    c._gate("y", qs)
                    continue
                if isinstance(gate, cirq.ZPowGate) and gate.exponent == 1:
                    c._gate("z", qs)
                    continue
                if isinstance(gate, cirq.CNotPowGate) and gate.exponent == 1:
                    c._gate("cx", qs)
                    continue
                if isinstance(gate, cirq.CZPowGate) and gate.exponent == 1:
                    c._gate("cz", qs)
                    continue
                if isinstance(gate, cirq.SwapPowGate) and gate.exponent == 1:
                    c._gate("swap", qs)
                    continue
                if isinstance(gate, (cirq.Rx, cirq.Ry, cirq.Rz)):
                    name = type(gate).__name__.lower()
                    rads = float(gate._rads)  # type: ignore[attr-defined]
                    c._gate(name, qs, (rads,))
                    continue

                # Fallback: use canonicalised name + numeric params
                canonical = _NAME_MAP.get(gname, gname.lower())
                params = ()
                if hasattr(gate, "_rads"):
                    params = (float(gate._rads),)  # type: ignore[attr-defined]
                elif hasattr(gate, "exponent") and getattr(gate, "exponent") != 1:
                    params = (float(gate.exponent),)
                c._gate(canonical, qs, params)
    return c.module
