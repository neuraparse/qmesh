"""hello_bell_crossbackend — Bell state, qmesh end-to-end.

Builds a Bell-state IR module, submits it via the router (which currently picks
the lowest-cost simulator), and prints a signed provenance manifest.

Run:
    pip install -e .
    python examples/hello_bell_crossbackend.py
"""

from __future__ import annotations

from pathlib import Path

from rich import print
from rich.panel import Panel

import qmesh
from qmesh.router import Objective, choose


def main() -> None:
    # 1. Build a Bell-state circuit in qmesh.ir
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0)
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)
    module = c.module

    print(Panel.fit(module.to_text(), title="qmesh IR (canonical text form)"))
    print(f"[bold]IR hash:[/] {module.hash()}")

    # 2. Pick a backend via the router
    backend, why = choose(module, Objective(budget_usd=1.0, min_fidelity=0.9))
    print(f"[bold]Router chose:[/] {backend.capabilities.name} "
          f"({backend.capabilities.vendor})")
    print(f"[dim]  ranked: {why['all_scored']}[/]")

    # 3. Submit. submit() emits a signed manifest into ./ledger/.
    result, manifest = qmesh.submit(
        module, backend=backend.capabilities.name, shots=4096,
        ledger_dir=Path("ledger"),
    )

    print(Panel.fit(
        "\n".join(f"  {k}: {v}" for k, v in sorted(result.counts.items())),
        title=f"counts ({result.shots} shots, {result.wall_seconds:.3f}s)",
    ))

    print(Panel.fit(
        manifest.to_json(),
        title=f"manifest @ {manifest.hash()[:16]}",
    ))


if __name__ == "__main__":
    main()
