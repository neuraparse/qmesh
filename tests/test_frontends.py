"""Cross-frontend semantic equivalence + QASM3 round-trip tests."""

from __future__ import annotations

import pytest

from qmesh.frontends.qasm3 import emit, parse


BELL_QASM = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""


def test_qasm3_round_trip_hash_stable():
    a = parse(BELL_QASM)
    b = parse(emit(a))
    assert a.hash() == b.hash()


def test_qasm3_emit_contains_keywords():
    a = parse(BELL_QASM)
    text = emit(a)
    assert "OPENQASM 3.0" in text
    assert "qubit[2]" in text
    assert "h q[0]" in text
    assert "cx q[0], q[1]" in text
    assert "measure" in text


def test_cross_frontend_qiskit_cirq_semantic_equivalence():
    qiskit = pytest.importorskip("qiskit")
    cirq = pytest.importorskip("cirq")
    from qmesh.frontends.qiskit import from_qiskit
    from qmesh.frontends.cirq import from_cirq

    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1); qc.measure(0, 0); qc.measure(1, 1)

    qubits = cirq.LineQubit.range(2)
    cc = cirq.Circuit([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(qubits[0], key="m0"),
        cirq.measure(qubits[1], key="m1"),
    ])

    assert from_qiskit(qc).semantic_hash() == from_cirq(cc).semantic_hash()


def test_pennylane_plugin_lowers_a_stub_tape():
    """Run the PennyLane plugin against a hand-rolled stub tape — no real
    PennyLane installed → from_pennylane_tape raises a clean RuntimeError."""
    from qmesh.frontends.pennylane import from_pennylane_tape, _pl_available

    if _pl_available():
        # If PennyLane is on the box, run a real Bell tape.
        import pennylane as qml
        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.sample(wires=[0, 1])
        m = from_pennylane_tape(tape, name="bell-pl")
        assert m.functions[0].name == "bell-pl"
        # Must be semantically equivalent to a hand-built Bell module.
        from qmesh import circuit
        with circuit("bell-ref", n_qubits=2, n_bits=2) as c:
            c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
        assert m.semantic_hash() == c.module.semantic_hash()
    else:
        with pytest.raises(RuntimeError, match="pennylane not installed"):
            from_pennylane_tape(object())


def test_qbraid_plugin_routes_through_qasm3():
    """qBraid plugin lowers to QASM 3 then through qmesh.frontends.qasm3.parse.
    Without qbraid installed → clean RuntimeError. With qbraid installed,
    we feed it an OpenQASM 3 string program (qBraid's identity transpile)."""
    from qmesh.frontends.qbraid import from_qbraid, _qbraid_available

    if _qbraid_available():
        # Identity transpile: pass already-QASM3 source.
        m = from_qbraid(BELL_QASM, name="bell-qbraid")
        assert m.functions[0].name == "bell-qbraid"
        # Round-trip the result against a fresh QASM parse.
        assert parse(BELL_QASM).semantic_hash() == m.semantic_hash()
    else:
        with pytest.raises(RuntimeError, match="qbraid not installed"):
            from_qbraid(BELL_QASM)


def test_qasm3_parse_with_rotations():
    src = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q; bit[2] c;
rx(0.5) q[0];
ry(0.3) q[1];
cx q[0], q[1];
c[0] = measure q[0]; c[1] = measure q[1];
"""
    m = parse(src)
    # round-trip
    assert parse(emit(m)).semantic_hash() == m.semantic_hash()
