"""End-to-end FT-mode demo: logical-gate lattice surgery + signed manifest.

Demonstrates:
  - lower a logical IR Module (H + S + measure on 1 logical qubit) into a
    Stim lattice-surgery circuit via qmesh.ftmode.lattice_surgery,
  - run the resulting circuit through the FT-mode runner with the
    PyMatching MWPM decoder,
  - report the FTResult numbers + signed manifest path.

Run:
    PYTHONPATH=. python3 examples/ft_lattice_surgery.py
"""

from __future__ import annotations

from rich import print
from rich.panel import Panel
from rich.table import Table

import qmesh
from qmesh.ftmode import FTConfig, promote_and_run
from qmesh.provenance.manifest import ManifestSigner


def demo_single_logical() -> None:
    print(Panel.fit(
        "[bold cyan]demo:[/] lattice surgery — H + S on 1 logical qubit, d=3",
        title="qmesh.ftmode.lattice_surgery",
    ))
    with qmesh.circuit("logical_HS", n_qubits=1, n_bits=1) as c:
        c.h(0)
        c.s(0)
        c.measure(0, 0)

    cfg = FTConfig(
        distance=3, rounds=4,
        physical_error_rate=1e-3,
        decoder="pymatching",
    )
    result, manifest = promote_and_run(
        c.module, ftconfig=cfg, shots=10_000,
    )

    table = Table(show_header=False)
    table.add_row("logical-IR ops",     "h(0) → s(0) → measure(0,0)")
    table.add_row("execution path",     manifest.ftmode["execution_path"])
    table.add_row("code",               result.code_name)
    table.add_row("decoder",            result.decoder_name)
    table.add_row("rounds (per patch)", f"{result.rounds}")
    table.add_row("p_phys",             f"{cfg.physical_error_rate:.0e}")
    table.add_row("shots",              f"{result.shots}")
    table.add_row("logical err / shot", f"[bold]{result.logical_error_rate:.6f}[/]")
    table.add_row("wall (total)",       f"{result.wall_seconds:.3f}s")
    table.add_row("sample wall",        f"{result.sample_seconds:.3f}s")
    table.add_row("decode wall",        f"{result.decode_seconds:.3f}s")
    table.add_row("manifest signed",    f"{ManifestSigner.verify(manifest)}")
    table.add_row("manifest path",      f"{result.manifest_path}")
    print(table)

    ls = manifest.ftmode["lattice_surgery"]
    sub = Table(title="lattice surgery lowering")
    sub.add_column("field")
    sub.add_column("value")
    sub.add_row("logical qubits",   f"{ls['logical_qubits']}")
    sub.add_row("stim qubits",      f"{ls['stim_qubits']}")
    sub.add_row("stim detectors",   f"{ls['stim_detectors']}")
    sub.add_row("stim observables", f"{ls['stim_observables']}")
    sub.add_row("init basis",       f"{ls['initial_basis']}")
    sub.add_row("final basis",      f"{ls['final_basis']}")
    sub.add_row(
        "ops_lowered",
        ", ".join(
            (e["name"] if e["kind"] == "1q" else
             ("cx" if e["kind"] == "2q" else "measure"))
            for e in ls["ops_lowered"]
        ),
    )
    print(sub)
    print(f"\n[dim]{ls['notes']}[/]")


def demo_two_logical_cx() -> None:
    print()
    print(Panel.fit(
        "[bold cyan]demo:[/] lattice surgery — logical CX between 2 patches",
        title="qmesh.ftmode.lattice_surgery (CX merge/split)",
    ))
    with qmesh.circuit("logical_CX", n_qubits=2, n_bits=2) as c:
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)

    cfg = FTConfig(
        distance=3, rounds=4,
        physical_error_rate=1e-3,
        decoder="pymatching",
    )
    result, manifest = promote_and_run(
        c.module, ftconfig=cfg, shots=4_000,
    )

    table = Table(show_header=False)
    table.add_row("logical-IR ops",     "cx(0,1) → measure(0,0) → measure(1,1)")
    table.add_row("code",               result.code_name)
    table.add_row("logical qubits",     f"{manifest.ftmode['lattice_surgery']['logical_qubits']}")
    table.add_row("stim qubits",        f"{manifest.ftmode['lattice_surgery']['stim_qubits']}")
    table.add_row("stim detectors",     f"{manifest.ftmode['lattice_surgery']['stim_detectors']}")
    table.add_row("shots",              f"{result.shots}")
    table.add_row("logical err / shot", f"[bold]{result.logical_error_rate:.6f}[/]")
    table.add_row("wall (total)",       f"{result.wall_seconds:.3f}s")
    table.add_row("manifest signed",    f"{ManifestSigner.verify(manifest)}")
    print(table)


def main() -> None:
    demo_single_logical()
    demo_two_logical_cx()


if __name__ == "__main__":
    main()
