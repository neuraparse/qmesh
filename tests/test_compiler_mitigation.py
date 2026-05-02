"""Compiler pass + mitigation integration tests."""

from __future__ import annotations

import pytest

import qmesh
from qmesh.compiler import PassContext


def test_tket_remove_redundancies_drops_double_h():
    pytest.importorskip("pytket")
    pytest.importorskip("qiskit")
    from qmesh.compiler.tket import tket_pass

    with qmesh.circuit("redundant", n_qubits=1, n_bits=1) as c:
        c.h(0); c.h(0); c.measure(0, 0)
    ctx = PassContext(backend_name="qmesh.aer")
    optimised = tket_pass("RemoveRedundancies")(c.module, ctx)
    # post-optimisation: the two H gates should be gone; only measurement remains
    n_gates = sum(
        1 for f in optimised.functions for op in f.body.ops
        if hasattr(op, "name") and op.name == "h"
    )
    assert n_gates == 0
    assert ctx.log[0]["pass"] == "tket.RemoveRedundancies"


def test_zne_improves_estimate_on_noisy_aer():
    pytest.importorskip("mitiq")
    pytest.importorskip("qiskit_aer")
    from qmesh.mitigation.zne import zne_estimate

    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)

    def even_parity(counts):
        total = sum(counts.values())
        return sum(v for k, v in counts.items() if k.count("1") % 2 == 0) / total

    res = zne_estimate(c.module, backend="qmesh.aer.noisy",
                      observable=even_parity, scale_factors=(1, 3, 5), shots=4096)
    # mitigated should not be worse than unmitigated by more than statistical noise
    assert res["mitigated"] >= res["unmitigated"] - 0.05
    assert "raw" in res
