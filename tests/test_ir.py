"""IR canonical-hashing and structural-vs-full hash tests."""

from __future__ import annotations

import qmesh


def _bell():
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    return c.module


def test_hash_deterministic():
    a = _bell().hash()
    b = _bell().hash()
    assert a == b


def test_hash_changes_with_structure():
    base = _bell().hash()
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0)  # one measurement missing
    assert c.module.hash() != base


def test_semantic_hash_ignores_function_name():
    a = _bell().semantic_hash()
    with qmesh.circuit("totally_different_name", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    assert c.module.semantic_hash() == a
    # but full hash differs because function name differs
    assert c.module.hash() != _bell().hash()


def test_to_text_round_trip_stable():
    text1 = _bell().to_text()
    text2 = _bell().to_text()
    assert text1 == text2
