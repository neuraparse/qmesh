"""qmesh.frontends.pennylane — PennyLane ↔ qmesh.ir.

Phase 6β SDK plugin. Imports PennyLane lazily so qmesh stays installable on
boxes without it. Two public entries:

    from_pennylane_tape(tape) -> Module
    from_qnode(qnode, *args, **kwargs) -> Module

α-honest scope:
  * Common 1-qubit and 2-qubit gates: H, X, Y, Z, S, T, RX, RY, RZ, CNOT/CX,
    CZ, SWAP, Toffoli (γ — translated to a 3-qubit named gate "ccx").
  * Mid-circuit measurements / classical control are out of α scope.
  * `qml.expval` / `qml.var` observables are not lowered (they are observable
    *post-processing*, not gates) — the Module captures only the unitary
    circuit. Callers that care about the observable should attach it manually
    via `module.metadata`.

Why ship this even if PennyLane isn't on the box: the lazy-import shape
matches the rest of qmesh.frontends (qiskit, pulser, sf), and once the user
runs `pip install pennylane` the converter works without further wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qmesh.ir.builder import circuit
from qmesh.ir.module import Module

if TYPE_CHECKING:  # pragma: no cover
    import pennylane as qml  # noqa: F401


# PennyLane operation name → (qmesh gate, n_params)
_PL_GATE_MAP: dict[str, tuple[str, int]] = {
    "Hadamard": ("h", 0),
    "PauliX": ("x", 0), "PauliY": ("y", 0), "PauliZ": ("z", 0),
    "S": ("s", 0), "T": ("t", 0),
    "RX": ("rx", 1), "RY": ("ry", 1), "RZ": ("rz", 1),
    "PhaseShift": ("rz", 1),       # PhaseShift acts as Rz up to global phase.
    "CNOT": ("cx", 0),
    "CZ": ("cz", 0),
    "SWAP": ("swap", 0),
    "Toffoli": ("ccx", 0),
    "Identity": ("id", 0),
}


def _pl_available() -> bool:
    try:
        import pennylane  # noqa: F401
        return True
    except ImportError:
        return False


def from_pennylane_tape(tape: Any, *, name: str = "pennylane_tape") -> Module:
    """Lower a `pennylane.tape.QuantumScript` (or `QuantumTape`) to qmesh IR.

    Parameters
    ----------
    tape :
        A PennyLane tape. The function reads ``tape.operations`` and
        ``tape.measurements`` — exactly what PennyLane's QNode produces
        internally during `qnode.construct()`.
    name :
        Module name (used in ledger filenames).
    """
    if not _pl_available():
        raise RuntimeError(
            "pennylane not installed; `pip install pennylane>=0.36` to enable."
        )
    import pennylane as qml  # noqa: F401  (probe only; consumed below)

    # PennyLane wires can be arbitrary labels; map them to dense ints.
    wire_set: list[Any] = []
    for op in tape.operations:
        for w in op.wires.tolist():
            if w not in wire_set:
                wire_set.append(w)
    for m in tape.measurements:
        try:
            ws = m.wires.tolist()
        except Exception:
            ws = []
        for w in ws:
            if w not in wire_set:
                wire_set.append(w)

    n_qubits = len(wire_set) or 1
    wire_to_idx = {w: i for i, w in enumerate(wire_set)}

    # Estimate a classical-bit register big enough for measurements.
    n_meas = sum(1 for m in tape.measurements if m.return_type.value == "sample"
                 or m.return_type.value == "counts" or m.return_type.value == "probs")
    n_bits = max(n_qubits, n_meas)

    with circuit(name, n_qubits=n_qubits, n_bits=n_bits) as c:
        for op in tape.operations:
            qmesh_entry = _PL_GATE_MAP.get(op.name)
            wires = [wire_to_idx[w] for w in op.wires.tolist()]
            if qmesh_entry is None:
                # Generic escape hatch — emit the gate by its PL name.
                params = tuple(float(p) for p in (op.parameters or ()))
                c._gate(op.name.lower(), wires, params)
                continue
            qmesh_name, n_params = qmesh_entry
            if n_params == 0:
                c._gate(qmesh_name, wires)
            else:
                params = tuple(float(p) for p in op.parameters[:n_params])
                c._gate(qmesh_name, wires, params)
        # Measurements: lower `qml.sample` / `qml.counts` to per-wire measure.
        for m in tape.measurements:
            try:
                wires = [wire_to_idx[w] for w in m.wires.tolist()]
            except Exception:
                wires = []
            for i, w in enumerate(wires):
                c.measure(w, i)
    return c.module


def from_qnode(qnode: Any, *args: Any, **kwargs: Any) -> Module:
    """Convenience: build the QNode's tape with the given args and lower.

    Equivalent to::

        qnode.construct(args, kwargs)
        return from_pennylane_tape(qnode.tape)
    """
    if not _pl_available():
        raise RuntimeError(
            "pennylane not installed; `pip install pennylane>=0.36` to enable."
        )
    qnode.construct(args, kwargs)
    return from_pennylane_tape(qnode.tape, name=getattr(qnode, "func", None).__name__
                               if getattr(qnode, "func", None) else "qnode")


__all__ = ["from_pennylane_tape", "from_qnode"]
