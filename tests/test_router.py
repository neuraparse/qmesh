"""Router scoring + decision tests."""

from __future__ import annotations

import pytest

import qmesh
from qmesh.router import Objective, choose, profile


def test_profile_counts_gates_correctly():
    with qmesh.circuit("c", n_qubits=3, n_bits=3) as c:
        c.h(0); c.h(1); c.h(2)
        c.cx(0, 1); c.cx(1, 2)
        c.measure(0, 0); c.measure(1, 1); c.measure(2, 2)
    p = profile(c.module)
    assert p.n_qubits == 3
    assert p.one_q_gates == 3
    assert p.two_q_gates == 2
    assert not p.has_mcm


def test_router_excludes_too_small_simulators():
    with qmesh.circuit("big", n_qubits=25, n_bits=25) as c:
        for i in range(25): c.h(i)
        for i in range(25): c.measure(i, i)
    bk, why = choose(c.module)
    # statevec caps at 20 qubits
    rejected_names = [name for name, _ in why["rejected"]]
    assert "qmesh.statevec" in rejected_names


def test_router_excludes_stim_for_non_clifford():
    pytest.importorskip("stim")
    with qmesh.circuit("t", n_qubits=1, n_bits=1) as c:
        c.h(0); c.t(0); c.measure(0, 0)
    bk, why = choose(c.module)
    rejected_names = [name for name, _ in why["rejected"]]
    assert "qmesh.stim" in rejected_names
    assert bk.capabilities.name != "qmesh.stim"


def test_router_prefers_higher_fidelity_under_equal_cost():
    pytest.importorskip("qiskit_aer")
    with qmesh.circuit("c", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    bk, why = choose(c.module, Objective(min_fidelity=0.999))
    # noisy aer has fidelity 0.997 < min, must be filtered
    rejected_names = [name for name, _ in why["rejected"]]
    assert "qmesh.aer.noisy" in rejected_names
