"""qmesh.frontends — adapters into qmesh.ir from vendor SDKs / source languages.

Phase 0 stub. Each adapter will live in its own module and import the vendor
package lazily. See `qiskit.py`, `qasm3.py`, `pulser.py`, `sf.py` for the planned
list. Contracts:

    qmesh.frontends.qiskit.from_qiskit(qc: QuantumCircuit) -> Module
    qmesh.frontends.qasm3.parse(text: str) -> Module
    qmesh.frontends.pulser.from_sequence(seq) -> Module
    qmesh.frontends.sf.from_program(prog) -> Module
"""

from __future__ import annotations
