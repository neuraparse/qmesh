"""GHZ state scaling demo.

Builds N-qubit GHZ states for N in {3, 6, 12, 18}, lets the router pick the
best backend (the small ones run on Stim — Clifford-only path; the larger
ones go to Aer) and prints scaling.

Run:
    PYTHONPATH=. python3 examples/ghz_scaling.py
"""

from __future__ import annotations

from rich import print
from rich.table import Table

import qmesh
from qmesh.router import Objective, choose, profile


def build_ghz(n: int) -> qmesh.Module:
    with qmesh.circuit(f"ghz{n}", n_qubits=n, n_bits=n) as c:
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)
        for i in range(n):
            c.measure(i, i)
    return c.module


def main() -> None:
    table = Table(title="qmesh GHZ scaling demo")
    table.add_column("N", justify="right")
    table.add_column("backend", style="cyan")
    table.add_column("2Q gates", justify="right")
    table.add_column("est. fid", justify="right")
    table.add_column("p(GHZ)", justify="right")
    table.add_column("wall (s)", justify="right")

    for n in (3, 6, 12, 18, 25):
        module = build_ghz(n)
        bk, why = choose(module, Objective(min_fidelity=0.0))
        prof = profile(module)
        result, _ = qmesh.submit(module, backend=bk.capabilities.name, shots=4000)
        good = "0" * n
        bad = "1" * n
        p_ghz = (result.counts.get(good, 0) + result.counts.get(bad, 0)) / result.shots
        table.add_row(
            str(n),
            bk.capabilities.name,
            str(prof.two_q_gates),
            f"{why['breakdown']['estimated_fidelity']:.4f}",
            f"{p_ghz:.4f}",
            f"{result.wall_seconds:.3f}",
        )
    print(table)


if __name__ == "__main__":
    main()
