"""QFT demo: build an N-qubit Quantum Fourier Transform and show that the
qmesh router routes it to a non-Clifford-capable backend (rx/ry/rz, etc.).
"""

from __future__ import annotations

import math

from rich import print

import qmesh
from qmesh.router import Objective, choose


def qft(n: int) -> qmesh.Module:
    with qmesh.circuit(f"qft{n}", n_qubits=n, n_bits=n) as c:
        for i in range(n):
            c.h(i)
            for k in range(i + 1, n):
                theta = math.pi / (2 ** (k - i))
                # controlled-phase via cx + rz + cx + rz pattern not portable;
                # use a single rz parametrised and treat it as native
                c._gate("cp", [k, i], (theta,))
        # bit-reversal swaps
        for i in range(n // 2):
            c.swap(i, n - 1 - i)
        for i in range(n):
            c.measure(i, i)
    return c.module


def main() -> None:
    n = 5
    module = qft(n)
    bk, why = choose(module)
    print(f"router chose [bold cyan]{bk.capabilities.name}[/] for {n}-qubit QFT")
    print("rejected:", why["rejected"])
    print("ranked:", why["all_scored"])
    result, manifest = qmesh.submit(module, backend=bk.capabilities.name, shots=4096)
    print(f"counts: {len(result.counts)} unique outcomes "
          f"(uniform-ish for QFT|0...0>)")
    # show top 4
    top = sorted(result.counts.items(), key=lambda kv: -kv[1])[:4]
    for k, v in top:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
