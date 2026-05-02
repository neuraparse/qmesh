"""End-to-end Phase 2γ FT-mode demo: cultivation-T injection in lattice surgery.

Demonstrates:
  - lower a logical IR Module (H + T + measure on 1 logical qubit) into a
    Stim lattice-surgery circuit via qmesh.ftmode.lattice_surgery,
  - the lowering reserves a cultivation block per T gate; the actual
    cultivation circuit (Gidney & Shutty 2409.17595) is upstream-research
    code referenced from cultivation.py — reserved here, swapped in Phase 6,
  - run end-to-end, print the resulting manifest's
    ftmode.lattice_surgery.t_injection_blocks + cultivation cost from the
    resource estimate.

Run:
    PYTHONPATH=. python3 examples/ft_t_injection.py
"""

from __future__ import annotations

from rich import print
from rich.panel import Panel
from rich.table import Table

import qmesh
from qmesh.ftmode import FTConfig, promote_and_run
from qmesh.provenance.manifest import ManifestSigner


def main() -> None:
    print(Panel.fit(
        "[bold cyan]Phase 2γ demo:[/] cultivation-T injection in lattice surgery — "
        "H + T + measure on 1 logical qubit, d=3",
        title="qmesh.ftmode.lattice_surgery (T-injection)",
    ))

    with qmesh.circuit("logical_HT", n_qubits=1, n_bits=1) as c:
        c.h(0)
        c.t(0)
        c.measure(0, 0)

    cfg = FTConfig(
        distance=3, rounds=4,
        physical_error_rate=1e-3,
        decoder="pymatching",
    )
    result, manifest = promote_and_run(
        c.module, ftconfig=cfg, shots=2_000,
    )

    # Top-level result
    head = Table(show_header=False)
    head.add_row("logical-IR ops",     "h(0) → t(0) → measure(0,0)")
    head.add_row("execution path",     manifest.ftmode["execution_path"])
    head.add_row("code",               result.code_name)
    head.add_row("decoder",            result.decoder_name)
    head.add_row("rounds (per patch)", f"{result.rounds}")
    head.add_row("p_phys",             f"{cfg.physical_error_rate:.0e}")
    head.add_row("shots",              f"{result.shots}")
    head.add_row("logical err / shot", f"[bold]{result.logical_error_rate:.6f}[/]")
    head.add_row("wall (total)",       f"{result.wall_seconds:.3f}s")
    head.add_row("manifest signed",    f"{ManifestSigner.verify(manifest)}")
    head.add_row("manifest path",      f"{result.manifest_path}")
    print(head)

    # Cultivation T-injection block
    ls = manifest.ftmode["lattice_surgery"]
    blocks = ls["t_injection_blocks"]
    print()
    print(Panel.fit(
        f"[bold green]ftmode.lattice_surgery.t_injection_blocks[/] "
        f"({len(blocks)} block(s))",
        title="cultivation reservation",
    ))
    if not blocks:
        print("[red]no T injections detected[/]")
    for blk in blocks:
        t = Table(show_header=False)
        t.add_row("block_index",                  f"{blk['block_index']}")
        t.add_row("logical_qubit",                f"{blk['logical_qubit']}")
        t.add_row("n_T_required",                 f"{blk['n_T_required']}")
        t.add_row("cultivation_factory",          f"{blk['cultivation_factory']}")
        t.add_row("cultivation_cycles_per_T",     f"{blk['cultivation_cycles_per_T']}")
        t.add_row("cultivation_cycles_block",     f"{blk['cultivation_cycles_block']}")
        t.add_row("stim_marker",                  f"{blk['stim_marker']}")
        print(t)
    print()
    summary = Table(show_header=False)
    summary.add_row("n_T_total",                  f"{ls['n_T_total']}")
    summary.add_row("cultivation_factory",        f"{ls['cultivation_factory']}")
    summary.add_row("cultivation_cycles_total",   f"{ls['cultivation_cycles_total']}")
    print(Panel(summary, title="lattice_surgery cultivation summary"))

    # Resource estimate's cultivation cost
    re = manifest.ftmode["resource_estimate"]
    re_table = Table(show_header=False)
    re_table.add_row("T_states_required",          f"{re['T_states_required']}")
    re_table.add_row("cycles (incl. cultivation)", f"{re['cycles']}")
    re_table.add_row("physical_qubits",            f"{re['physical_qubits']}")
    re_table.add_row("estimated_logical_error",    f"{re['estimated_logical_error_rate']:.3e}")
    re_table.add_row("wall_seconds (estimate)",    f"{re['wall_seconds']:.3e}")
    print()
    print(Panel(re_table, title="resource estimate (Microsoft RE-shaped)"))

    cul = manifest.ftmode.get("cultivation")
    if cul:
        cul_t = Table(show_header=False)
        for k, v in cul.items():
            cul_t.add_row(k, str(v))
        print()
        print(Panel(cul_t, title="ftmode.cultivation (factory parameters)"))

    print()
    print(Panel.fit(
        "[dim]γ-honest framing: the lowering reserves cultivation rounds in "
        "the manifest and the resource estimator counts them. The actual "
        "cultivation circuit (Gidney & Shutty 2409.17595) is upstream-research "
        "code — Phase 6 will swap it in.[/]",
        title="α/γ note",
    ))


if __name__ == "__main__":
    main()
