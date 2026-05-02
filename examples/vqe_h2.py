"""VQE on H2 (toy demo).

A textbook H2 Hamiltonian at fixed bond length expressed in 4-Pauli form:

    H = -1.0523732 I
        + 0.39793742 IZ
        - 0.39793742 ZI
        - 0.0112801 ZZ
        + 0.18093119 XX

ansatz: hardware-efficient — Ry rotations + linear entangler + Ry rotations.
This file performs a tiny scipy.optimize.minimize loop with the qmesh runtime
under the hood, exercising the auto-generated reproducibility manifest per
shot batch.
"""

from __future__ import annotations

import math

import numpy as np
from rich import print

import qmesh

H_TERMS = [
    ("II", -1.0523732),
    ("IZ", 0.39793742),
    ("ZI", -0.39793742),
    ("ZZ", -0.0112801),
    ("XX", 0.18093119),
]


def _ansatz(theta: list[float]) -> qmesh.Module:
    with qmesh.circuit("h2_ansatz", n_qubits=2, n_bits=2) as c:
        c.ry(theta[0], 0)
        c.ry(theta[1], 1)
        c.cx(0, 1)
        c.ry(theta[2], 0)
        c.ry(theta[3], 1)
    return c.module


def _measure_in(basis: str, prepared: qmesh.Module) -> qmesh.Module:
    """Append basis-rotation gates + measurement to the prepared state."""
    # Re-build a circuit that is the prepared circuit + rotation
    from qmesh.ir.builder import circuit
    with circuit("h2_meas", n_qubits=2, n_bits=2) as c:
        # mirror prepared ops
        for f in prepared.functions:
            for op in f.body.ops:
                from qmesh.ir.ops import GateOp
                if isinstance(op, GateOp):
                    qubit_idx = [v.index for v in op.operands]
                    c._gate(op.name, qubit_idx, op.params)
        # rotate basis
        for i, b in enumerate(reversed(basis)):  # basis is MSB-first
            if b == "X":
                c.h(i)
            elif b == "Y":
                c.sdg(i); c.h(i)
            # Z = no rotation
        c.measure(0, 0); c.measure(1, 1)
    return c.module


def _expectation(counts: dict[str, int], basis: str) -> float:
    """Pauli-string expectation from measurement counts (Z-basis after rotation).

    For a string like "ZZ" (or the rotated basis) the eigenvalue is +1 if the
    parity over qubits where the basis is non-I is even, else -1.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    val = 0.0
    for bitstring, n in counts.items():
        sign = 1
        for i, b in enumerate(reversed(basis)):  # bitstring MSB-first
            bit = int(bitstring[len(bitstring) - 1 - i])
            if b == "I":
                continue
            if bit == 1:
                sign *= -1
        val += sign * n / total
    return val


def energy(theta: list[float], backend: str = "qmesh.aer", shots: int = 4096) -> float:
    """Compute <H> for the given angles."""
    state = _ansatz(theta)
    e = 0.0
    for basis, coeff in H_TERMS:
        if basis == "II":
            e += coeff
            continue
        meas_module = _measure_in(basis, state)
        result, _ = qmesh.submit(
            meas_module, backend=backend, shots=shots, sign=False, ledger_dir="ledger/vqe",
        )
        e += coeff * _expectation(result.counts, basis)
    return e


def main() -> None:
    from scipy.optimize import minimize

    print("[bold]VQE on H2[/]  (qmesh.aer ideal simulator)")
    print(f"target ground state energy: -1.857  (from FCI)")

    theta0 = [0.1, 0.1, 0.1, 0.1]
    history: list[float] = []

    def objective(theta):
        e = energy(list(theta), backend="qmesh.aer", shots=2048)
        history.append(e)
        if len(history) % 5 == 0:
            print(f"  iter {len(history)}: E = {e:.5f}")
        return e

    res = minimize(objective, theta0, method="COBYLA", options={"maxiter": 30})
    print()
    print(f"[bold]Optimised E:[/] {res.fun:.5f}")
    print(f"theta:        {[round(t, 4) for t in res.x]}")
    print(f"iterations:   {len(history)}")
    print()
    print(f"[dim]Manifests for every shot batch in ledger/vqe/[/]")


if __name__ == "__main__":
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("[yellow]scipy not installed; skipping VQE optimization.[/]")
        raise SystemExit(0)
    main()
