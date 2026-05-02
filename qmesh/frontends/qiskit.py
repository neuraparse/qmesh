"""qmesh.frontends.qiskit — Qiskit ↔ qmesh.ir.

Phase-0 minimal converter. Lazy import so qmesh works without qiskit installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qmesh.ir.builder import circuit
from qmesh.ir.module import Module

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


_GATE_MAP = {
    "h": ("h", 0), "x": ("x", 0), "y": ("y", 0), "z": ("z", 0),
    "s": ("s", 0), "t": ("t", 0),
    "rx": ("rx", 1), "ry": ("ry", 1), "rz": ("rz", 1),
    "cx": ("cx", 0), "cz": ("cz", 0), "swap": ("swap", 0),
}


def from_qiskit(qc: "QuantumCircuit") -> Module:
    n_qubits = qc.num_qubits
    n_bits = qc.num_clbits
    with circuit(qc.name or "qc", n_qubits=n_qubits, n_bits=n_bits) as c:
        for ci in qc.data:
            instr = ci.operation
            qargs = ci.qubits
            cargs = ci.clbits
            name = instr.name
            qubit_indices = [qc.find_bit(q).index for q in qargs]
            if name == "measure":
                bit_index = qc.find_bit(cargs[0]).index
                c.measure(qubit_indices[0], bit_index)
                continue
            if name == "barrier":
                c.barrier()
                continue
            if name in _GATE_MAP:
                qmesh_name, n_params = _GATE_MAP[name]
                if n_params == 0:
                    c._gate(qmesh_name, qubit_indices)
                else:
                    params = tuple(float(p) for p in instr.params[:n_params])
                    c._gate(qmesh_name, qubit_indices, params)
                continue
            # generic native escape hatch
            c._gate(name, qubit_indices, tuple(float(p) for p in instr.params))
    return c.module
