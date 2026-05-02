"""qmesh quick benchmark: compare backends on identical circuits.

Run:
    PYTHONPATH=. python3 scripts/benchmark.py
"""

from __future__ import annotations

import time

from rich import print
from rich.table import Table

import qmesh


def build_ghz(n: int) -> qmesh.Module:
    with qmesh.circuit(f"ghz{n}", n_qubits=n, n_bits=n) as c:
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)
        for i in range(n):
            c.measure(i, i)
    return c.module


def build_rcs(n: int, depth: int) -> qmesh.Module:
    """Random-Clifford-style circuit: alternating layers of H/S/CX. Stays Clifford
    so all four backends can run it (good for apples-to-apples timing)."""
    import random
    rng = random.Random(42)
    with qmesh.circuit(f"rcs{n}d{depth}", n_qubits=n, n_bits=n) as c:
        for _ in range(depth):
            for q in range(n):
                op = rng.choice(["h", "s", "x", "z"])
                c._gate(op, [q])
            for a in range(0, n - 1, 2):
                c.cx(a, a + 1)
            for a in range(1, n - 1, 2):
                c.cx(a, a + 1)
        for q in range(n):
            c.measure(q, q)
    return c.module


def main() -> None:
    targets = [
        ("GHZ-12", build_ghz(12)),
        ("GHZ-18", build_ghz(18)),
        ("RCS-10×8", build_rcs(10, 8)),
        ("RCS-15×10", build_rcs(15, 10)),
    ]
    backends = ["qmesh.statevec", "qmesh.aer", "qmesh.aer.noisy", "qmesh.stim"]
    shots = 4096

    table = Table(title=f"qmesh backend benchmark ({shots} shots)")
    table.add_column("circuit")
    for b in backends:
        table.add_column(b, justify="right")

    for cname, module in targets:
        row = [cname]
        for bname in backends:
            try:
                t0 = time.time()
                qmesh.submit(module, backend=bname, shots=shots, sign=False)
                row.append(f"{(time.time() - t0) * 1000:.0f} ms")
            except Exception as e:  # noqa: BLE001
                row.append(f"[red]{type(e).__name__}[/]")
        table.add_row(*row)
    print(table)


if __name__ == "__main__":
    main()
