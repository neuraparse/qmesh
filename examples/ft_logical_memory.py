"""End-to-end FT-mode demo: logical memory + threshold sweep + resource estimate.

Demonstrates:
  - SurfaceCode primitive
  - FTConfig with PyMatching decoder
  - memory_experiment() emits a signed manifest with the full FT block
  - threshold_sweep() across (d, p) — extracts Λ
  - estimate() — Microsoft-RE-shaped resource sizing for a 1k-T-gate workload

Run:
    PYTHONPATH=. python3 examples/ft_logical_memory.py
"""

from __future__ import annotations

import json
from rich import print
from rich.panel import Panel
from rich.table import Table

import qmesh
from qmesh.ftmode import (
    FTConfig,
    SurfaceCode,
    estimate,
    memory_experiment,
    threshold_sweep,
)


def demo_memory() -> None:
    print(Panel.fit(
        "[bold cyan]demo 1:[/] surface-code memory at d=5, p_phys=1e-3",
        title="qmesh.ftmode",
    ))
    cfg = FTConfig(distance=5, rounds=8, physical_error_rate=1e-3, decoder="pymatching")
    result, manifest = memory_experiment(ftconfig=cfg, shots=30_000)

    table = Table(show_header=False)
    table.add_row("code",        f"{result.code_name}")
    table.add_row("decoder",     f"{result.decoder_name}")
    table.add_row("rounds",      f"{result.rounds}")
    table.add_row("p_phys",      f"{cfg.physical_error_rate:.0e}")
    table.add_row("shots",       f"{result.shots}")
    table.add_row("logical err / shot", f"[bold]{result.logical_error_rate:.6f}[/]")
    table.add_row("wall (total)", f"{result.wall_seconds:.3f}s")
    table.add_row("manifest",    f"{result.manifest_path}")
    print(table)


def demo_threshold() -> None:
    print()
    print(Panel.fit(
        "[bold cyan]demo 2:[/] Λ-curve sweep across (d, p)",
        title="qmesh.ftmode.threshold_sweep",
    ))
    template = FTConfig(distance=3, rounds=8, physical_error_rate=1e-3,
                       decoder="pymatching")
    results = threshold_sweep(
        ftconfig_template=template,
        distances=[3, 5, 7, 9],
        physical_error_rates=[5e-4, 1e-3, 3e-3, 1e-2],
        shots=8_000,
    )

    distances = sorted({r.distance for r in results})
    table = Table(title="logical error rate per shot")
    table.add_column("d", justify="right")
    table.add_column("p=5e-4", justify="right")
    table.add_column("p=1e-3", justify="right")
    table.add_column("p=3e-3", justify="right")
    table.add_column("p=1e-2", justify="right")

    # threshold_sweep iterates p inside d (in the order passed in); preserve.
    for d in distances:
        cells: list[str] = [str(d)]
        for r in (rr for rr in results if rr.distance == d):
            cells.append(f"{r.logical_error_rate:.5f}")
        while len(cells) < 5:
            cells.append("—")
        table.add_row(*cells)
    print(table)


def demo_estimate() -> None:
    print()
    print(Panel.fit(
        "[bold cyan]demo 3:[/] resource estimate — 1k T-gate workload",
        title="qmesh.ftmode.estimate (Microsoft-RE-shaped)",
    ))
    # Build a logical workload with 1000 T-gates
    with qmesh.circuit("workload", n_qubits=4, n_bits=4) as c:
        for _ in range(250):
            for q in range(4):
                c.t(q)

    table = Table(title="resource estimate vs distance (p=1e-3, target p_L=1e-9)")
    table.add_column("d", justify="right")
    table.add_column("physical qubits", justify="right")
    table.add_column("T-states", justify="right")
    table.add_column("cycles", justify="right")
    table.add_column("wall (s)", justify="right")
    table.add_column("est p_L", justify="right")
    table.add_column("hits target?", justify="center")

    for d in (5, 7, 9, 11, 13, 15):
        re = estimate(
            c.module,
            code=SurfaceCode(distance=d, rounds=8),
            physical_error_rate=1e-3,
            target_logical_error_rate=1e-9,
        )
        ok = "✓" if re.estimated_logical_error_rate <= re.target_logical_error_rate else "✗"
        table.add_row(
            str(d),
            f"{re.physical_qubits:,}",
            f"{re.T_states_required:,}",
            f"{re.cycles:,}",
            f"{re.wall_seconds:.3f}",
            f"{re.estimated_logical_error_rate:.2e}",
            ok,
        )
    print(table)


def main() -> None:
    demo_memory()
    demo_threshold()
    demo_estimate()


if __name__ == "__main__":
    main()
