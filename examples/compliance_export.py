"""Phase 6 demo: regulated-industry audit pack + Metriq export.

Pipeline:
    1. Run three different qmesh experiments — Bell, GHZ, FT-memory.
    2. Build a compliance pack (.tar.gz) over the resulting ledger.
    3. Verify the pack offline (no private key, no network).
    4. Export the same ledger as a Metriq submission (dry-run).

Run:
    PYTHONPATH=. python3 examples/compliance_export.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from rich import print
from rich.panel import Panel
from rich.table import Table

import qmesh
from qmesh.compliance import build_pack, verify_pack
from qmesh.metriq import export_run_dir, submit_to_metriq


def _bell(ledger_dir: Path) -> Path:
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    _, m = qmesh.submit(c.module, backend="qmesh.statevec",
                        shots=512, ledger_dir=ledger_dir)
    return ledger_dir / m.submitted_at[:10] / f"{m.hash()[:16]}.json"


def _ghz(ledger_dir: Path) -> Path:
    with qmesh.circuit("ghz", n_qubits=4, n_bits=4) as c:
        c.h(0); c.cx(0, 1); c.cx(1, 2); c.cx(2, 3)
        for q in range(4):
            c.measure(q, q)
    _, m = qmesh.submit(c.module, backend="qmesh.statevec",
                        shots=1024, ledger_dir=ledger_dir)
    return ledger_dir / m.submitted_at[:10] / f"{m.hash()[:16]}.json"


def _ft_memory(ledger_dir: Path) -> Path:
    try:
        import stim  # noqa: F401
        import pymatching  # noqa: F401
    except ImportError:
        print("[yellow]stim+pymatching not installed; skipping FT memory[/]")
        return None  # type: ignore[return-value]
    from qmesh.ftmode import FTConfig, memory_experiment

    cfg = FTConfig(distance=3, rounds=4, physical_error_rate=1e-3,
                   decoder="pymatching")
    res, m = memory_experiment(ftconfig=cfg, shots=1500,
                               ledger_dir=ledger_dir, seed=2026)
    return res.manifest_path


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="qmesh_compliance_"))
    print(Panel.fit(f"[bold]workdir:[/] {work}", title="Phase 6 demo"))

    ledger = work / "ledger"
    paths = [
        _bell(ledger),
        _ghz(ledger),
    ]
    ft_path = _ft_memory(ledger)
    if ft_path is not None:
        paths.append(ft_path)

    runs_table = Table(title="qmesh runs (signed manifests)")
    runs_table.add_column("kind", style="cyan")
    runs_table.add_column("manifest")
    runs_table.add_column("hash (16)")
    for p, kind in zip(paths, ["bell", "ghz", "ft-memory"]):
        if p is None:
            continue
        runs_table.add_row(kind, str(p), p.stem)
    print(runs_table)

    # ------------------------------------------------------------------ pack
    archive = work / "compliance_pack.tar.gz"
    out = build_pack(ledger, out_path=archive)
    print(Panel.fit(
        f"[bold]archive:[/] {out}\n"
        f"[bold]size:[/]    {out.stat().st_size / 1024:.1f} KB",
        title="compliance-pack built",
    ))

    # ---------------------------------------------------------------- verify
    r = verify_pack(out)
    verify_table = Table(title="compliance-verify (offline)")
    verify_table.add_column("field", style="cyan")
    verify_table.add_column("value", justify="right")
    verify_table.add_row(
        "chain_valid",
        "[green]✓[/]" if r.chain_valid else "[red]✗[/]",
    )
    verify_table.add_row("n_manifests", str(r.n_manifests))
    verify_table.add_row("n_signatures_ok", str(r.n_signatures_ok))
    verify_table.add_row("broken_at",
                         "—" if r.broken_at is None else str(r.broken_at))
    verify_table.add_row("errors", str(len(r.errors)))
    print(verify_table)
    if r.notes:
        print(Panel.fit("\n".join(f"  • {n}" for n in r.notes),
                        title="notes"))

    # ---------------------------------------------------------------- metriq
    sub = export_run_dir(ledger, name="qmesh phase6 demo",
                         tags=["qmesh", "phase6", "demo"])
    sent = submit_to_metriq(sub, dry_run=True)

    metriq_table = Table(title="metriq submission (dry-run)")
    metriq_table.add_column("field", style="cyan")
    metriq_table.add_column("value")
    metriq_table.add_row("name", sub["name"])
    metriq_table.add_row("n_results", str(len(sub["results"])))
    metriq_table.add_row("tags", ", ".join(sub["tags"]))
    metriq_table.add_row("dry_run", str(sent["dry_run"]))
    metriq_table.add_row("endpoint", sent["endpoint"])
    print(metriq_table)

    metrics_table = Table(title="metriq metric rows (first 10)")
    metrics_table.add_column("metric", style="cyan")
    metrics_table.add_column("value", justify="right")
    metrics_table.add_column("unit")
    metrics_table.add_column("samples", justify="right")
    metrics_table.add_column("manifest_hash (12)")
    for row in sub["results"][:10]:
        metrics_table.add_row(
            row["metric_name"],
            f"{row['metric_value']:.4g}",
            row["metric_unit"],
            str(row["sample_size"]),
            row["evidence"]["manifest_hash"][:12],
        )
    print(metrics_table)

    print(Panel.fit(
        f"[bold]demo complete[/]\n"
        f"chain_valid     = {r.chain_valid}\n"
        f"n_manifests     = {r.n_manifests}\n"
        f"n_signatures_ok = {r.n_signatures_ok}\n"
        f"metriq_results  = {len(sub['results'])}",
        title="phase 6: regulated-industry audit pack + metriq export",
    ))

    # cleanup — keep the archive in /tmp for inspection
    print(f"[dim]artifacts left in {work}; remove with `rm -rf {work}`[/]")


if __name__ == "__main__":
    main()
