"""qmesh CLI."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print
from rich.panel import Panel
from rich.table import Table

from qmesh import __version__
from qmesh.api import diff as diff_cmd
from qmesh.api import replay as replay_cmd
from qmesh.api import submit as submit_cmd
from qmesh.backends.registry import all_backends, get
from qmesh.router import Objective, choose, profile

app = typer.Typer(help="qmesh — 2026-native quantum operating layer.")


@app.command()
def version() -> None:
    """Print qmesh version."""
    print(f"qmesh [bold cyan]{__version__}[/]")


@app.command()
def backends() -> None:
    """List registered backends and their capabilities."""
    table = Table(title="qmesh backends")
    table.add_column("name", style="cyan")
    table.add_column("vendor")
    table.add_column("modalities")
    table.add_column("qubits", justify="right")
    table.add_column("sim?", justify="center")
    table.add_column("fid 2Q", justify="right")
    table.add_column("$/shot", justify="right")
    table.add_column("notes", overflow="fold", max_width=40)
    for bk in all_backends().values():
        c = bk.capabilities
        table.add_row(
            c.name, c.vendor,
            ",".join(m.value for m in c.modalities),
            str(c.qubit_count),
            "✓" if c.is_simulator else "",
            f"{c.fidelity_2q_typical:.4f}",
            f"{c.cost_per_shot_usd:.5f}" if c.cost_per_shot_usd is not None else "—",
            c.notes,
        )
    print(table)


@app.command()
def submit(
    qasm_file: Path = typer.Argument(..., help="Path to OpenQASM 3 source file."),
    backend: str = typer.Option("auto", help="Backend name or 'auto' to let router pick."),
    shots: int = typer.Option(1024),
    budget: float = typer.Option(1.0),
    min_fidelity: float = typer.Option(0.0),
    ledger_dir: Path = typer.Option("ledger"),
) -> None:
    """Parse a QASM 3 file, route, run, and emit a signed manifest."""
    from qmesh.frontends.qasm3 import parse as qasm_parse

    text = qasm_file.read_text()
    module = qasm_parse(text)
    print(f"[dim]Parsed {qasm_file.name}: hash={module.hash()[:16]}[/]")

    if backend == "auto":
        bk, why = choose(module, Objective(budget_usd=budget, min_fidelity=min_fidelity), shots=shots)
        print(Panel.fit(json.dumps(why, indent=2, default=str), title="router decision"))
        backend = bk.capabilities.name
    else:
        get(backend)  # raise if missing

    result, manifest = submit_cmd(module, backend=backend, shots=shots, ledger_dir=ledger_dir)
    print(Panel.fit(
        "\n".join(f"  {k}: {v}" for k, v in sorted(result.counts.items())[:20]),
        title=f"counts ({result.shots} shots, {result.wall_seconds:.3f}s)",
    ))
    print(f"[bold]manifest:[/] {ledger_dir}/{manifest.submitted_at[:10]}/{manifest.hash()[:16]}.json")


@app.command()
def check(manifest: Path) -> None:
    """Verify a manifest's signature."""
    ok, msg = replay_cmd(manifest)
    icon = "[green]✓[/]" if ok else "[red]✗[/]"
    print(f"{icon} {msg}")
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def diff(a: Path, b: Path) -> None:
    """Diff two manifests."""
    print(json.dumps(diff_cmd(a, b), indent=2, default=str))


@app.command()
def inspect(qasm_file: Path) -> None:
    """Parse a QASM 3 file and show its qmesh.ir + circuit profile."""
    from qmesh.frontends.qasm3 import parse as qasm_parse

    module = qasm_parse(qasm_file.read_text())
    print(Panel.fit(module.to_text(), title=f"qmesh IR (hash {module.hash()[:16]})"))
    prof = profile(module)
    table = Table(title="circuit profile")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("n_qubits", str(prof.n_qubits))
    table.add_row("depth", str(prof.depth))
    table.add_row("1Q gates", str(prof.one_q_gates))
    table.add_row("2Q gates", str(prof.two_q_gates))
    table.add_row("MCM?", "yes" if prof.has_mcm else "no")
    table.add_row("gate set", ", ".join(sorted(prof.gate_set)))
    print(table)


@app.command()
def emit(qasm_file: Path) -> None:
    """Parse a QASM file and re-emit it (useful for round-trip testing)."""
    from qmesh.frontends.qasm3 import emit as qasm_emit
    from qmesh.frontends.qasm3 import parse as qasm_parse

    module = qasm_parse(qasm_file.read_text())
    print(qasm_emit(module))


@app.command()
def dag_status(run_dir: Path = typer.Argument(...)) -> None:
    """Show the status of a DAG run: nodes, durations, manifest hashes."""
    manifest_file = run_dir / "dag_run_manifest.json"
    if not manifest_file.exists():
        print(f"[red]no manifest at {manifest_file}[/]")
        raise typer.Exit(code=1)
    data = json.loads(manifest_file.read_text())
    table = Table(title=f"DAG run {data['execution']['run_id']}")
    table.add_column("node id", style="cyan")
    table.add_column("kind")
    table.add_column("ok")
    table.add_column("dur (s)", justify="right")
    table.add_column("skipped", justify="center")
    table.add_column("manifest")
    for r in data["execution"]["node_results"]:
        table.add_row(
            r["id"][:18], r["kind"], "✓" if r["ok"] else "✗",
            f"{r['duration_s']:.3f}",
            "↻" if r.get("skipped") else "",
            (r["manifest_hash"] or "")[:18],
        )
    print(table)
    lineage = data.get("lineage")
    if lineage:
        print(Panel.fit(
            f"parent_run_id        {lineage.get('parent_run_id')}\n"
            f"parent_manifest_hash {lineage.get('parent_manifest_hash')}\n"
            f"parent_signature_alg {lineage.get('parent_signature_alg')}\n"
            f"n_skipped_resume     {lineage.get('n_skipped_resume')}",
            title="lineage",
        ))
    print(f"[dim]aggregate manifest signature: {data.get('signature', {}).get('alg', 'unsigned')}[/]")


@app.command()
def dag_resume_info(run_dir: Path = typer.Argument(...)) -> None:
    """Inspect a run dir and report resume coverage from its node_results/.

    This does not need the original DAG object; it just looks at which
    NodeResult snapshots are persisted and which are missing/failed.
    """
    results_dir = run_dir / "node_results"
    if not results_dir.exists():
        print(f"[red]no node_results/ at {run_dir} — nothing to resume[/]")
        raise typer.Exit(code=1)

    manifest_file = run_dir / "dag_run_manifest.json"
    if not manifest_file.exists():
        print(f"[red]no manifest at {manifest_file}[/]")
        raise typer.Exit(code=1)
    data = json.loads(manifest_file.read_text())
    declared = {r["id"] for r in data["execution"]["node_results"]}

    table = Table(title=f"resume info for {run_dir}")
    table.add_column("node id", style="cyan")
    table.add_column("snapshot")
    table.add_column("ok")
    table.add_column("on resume")
    succeeded = 0
    will_rerun = 0
    for nid in sorted(declared):
        snap = results_dir / f"{nid}.json"
        if not snap.exists():
            table.add_row(nid[:18], "—", "—", "[yellow]rerun[/]")
            will_rerun += 1
            continue
        snap_data = json.loads(snap.read_text())
        if snap_data.get("ok"):
            table.add_row(nid[:18], "✓", "✓", "[green]skip[/]")
            succeeded += 1
        else:
            table.add_row(nid[:18], "✓", "[red]✗[/]", "[yellow]rerun[/]")
            will_rerun += 1
    print(table)
    print(f"[bold]parent run id:[/] {data['execution']['run_id']}")
    print(f"[bold]would skip:[/]   {succeeded}")
    print(f"[bold]would rerun:[/]  {will_rerun}")


@app.command()
def ai_draft(
    prompt: str = typer.Argument(..., help="Natural-language description of the circuit."),
    backend: str = typer.Option("auto", help="Backend to validate against."),
    iterations: int = typer.Option(2, help="Max LLM re-prompt iterations."),
    show_qasm: bool = typer.Option(False, help="Emit the QASM 3 source."),
) -> None:
    """Draft a circuit from a natural-language prompt (intent + LLM + QCoder validation)."""
    from qmesh.ai import draft
    from qmesh.frontends.qasm3 import emit as qasm_emit

    result = draft(prompt, max_iterations=iterations)
    print(Panel.fit(
        f"[bold]description:[/]   {result.description}\n"
        f"[bold]provider:[/]      {result.provider}\n"
        f"[bold]model:[/]         {result.model}\n"
        f"[bold]intent:[/]        {result.intent_pattern or '—'}\n"
        f"[bold]iterations:[/]   {result.iterations}\n"
        f"[bold]validation:[/]   {result.validation}\n"
        f"[bold]notes:[/]\n  - " + "\n  - ".join(result.notes),
        title=f"qmesh.ai.draft  →  Module @{result.module.hash()[:12]}",
    ))
    if show_qasm:
        try:
            print(qasm_emit(result.module))
        except NotImplementedError as e:
            print(f"[yellow](cannot emit QASM: {e})[/]")


@app.command()
def ai_train_decoder(
    distance: int = typer.Option(3, help="Surface code distance (odd ≥ 3)."),
    rounds: int = typer.Option(4, help="Syndrome rounds."),
    p_phys: float = typer.Option(1e-3, help="Physical error rate."),
    n_train: int = typer.Option(20_000, help="Training samples."),
    epochs: int = typer.Option(6),
    hidden_dim: int = typer.Option(128),
    weights_dir: Path = typer.Option(Path("ai_weights")),
) -> None:
    """Train a neural decoder for the given surface-code parameters."""
    from qmesh.ai import train_neural_decoder
    from qmesh.ftmode import SurfaceCode

    code = SurfaceCode(distance=distance, rounds=rounds)
    decoder, tr = train_neural_decoder(
        code, physical_error_rate=p_phys, n_train=n_train,
        epochs=epochs, hidden_dim=hidden_dim, weights_dir=weights_dir,
    )
    print(Panel.fit(
        f"[bold]code:[/]            {code.metadata().name}\n"
        f"[bold]p_phys:[/]          {p_phys:.0e}\n"
        f"[bold]train samples:[/]  {tr.train_samples}\n"
        f"[bold]val samples:[/]    {tr.val_samples}\n"
        f"[bold]epochs:[/]         {tr.epochs}\n"
        f"[bold]val accuracy:[/]   {tr.accuracy:.4f}\n"
        f"[bold]final loss:[/]     {tr.final_loss:.4f}\n"
        f"[bold]wall:[/]           {tr.duration_seconds:.2f}s\n"
        f"[bold]weights:[/]        {tr.weights_path}\n"
        f"[bold]sha256:[/]         {tr.weights_sha256}",
        title="qmesh.ai.train_neural_decoder",
    ))


@app.command("hpc-status")
def hpc_status(
    job_id: str = typer.Argument(..., help="Scheduler-assigned job ID."),
    scheduler: str = typer.Option(
        "slurm", help="Scheduler kind: slurm | pbs | mock."
    ),
) -> None:
    """Show raw scheduler status for an HPC-dispatched qmesh job."""
    from qmesh.scheduler.hpc import (
        MockHPCConnector,
        PBSConnector,
        SLURMConnector,
    )

    cls = {
        "slurm": SLURMConnector,
        "pbs": PBSConnector,
        "mock": MockHPCConnector,
    }.get(scheduler)
    if cls is None:
        print(f"[red]unknown scheduler:[/] {scheduler}")
        raise typer.Exit(code=1)
    conn = cls()
    try:
        out = conn.status(job_id)
    except FileNotFoundError as e:
        print(f"[red]{scheduler} client not on PATH:[/] {e}")
        raise typer.Exit(code=2) from e
    print(Panel.fit(
        out or f"(no output — job {job_id} likely complete or unknown)",
        title=f"{scheduler} status: {job_id}",
    ))


@app.command("ai-eval")
def ai_eval(
    suite: str = typer.Option("quanbench", help="Benchmark suite to run."),
    provider: str = typer.Option("mock", help="LLM provider: mock | auto."),
    output: Path = typer.Option(None, help="Optional path to write the JSON report."),
) -> None:
    """Run the QuanBench-shaped evaluation harness on the copilot."""
    from qmesh.ai.quanbench import run_quanbench

    if suite != "quanbench":
        print(f"[red]unknown suite {suite!r}; only 'quanbench' is built in[/]")
        raise typer.Exit(code=1)
    report = run_quanbench(provider=provider, output_path=output)

    table = Table(title=f"qmesh.ai-eval ({suite}, provider={provider})")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("n_prompts", str(report["n_prompts"]))
    table.add_row("success_rate", f"{report['success_rate']:.3f}")
    table.add_row("n_compiled", str(report["n_compiled"]))
    table.add_row("n_simulator_passed", str(report["n_simulator_passed"]))
    table.add_row("n_intent_recognised", str(report["n_intent_recognised"]))
    table.add_row("mean_iterations", f"{report['mean_iterations']:.2f}")
    table.add_row("total wall (s)", f"{report['wall_time_seconds']['total']:.2f}")
    print(table)

    pat = Table(title="per-pattern")
    pat.add_column("pattern", style="cyan")
    pat.add_column("n", justify="right")
    pat.add_column("passed", justify="right")
    pat.add_column("compiled", justify="right")
    pat.add_column("sim_passed", justify="right")
    for k, slot in report["per_pattern"].items():
        pat.add_row(k, str(slot["n"]), str(slot["passed"]),
                    str(slot["compiled"]), str(slot["sim_passed"]))
    print(pat)

    prov = Table(title="per-provider")
    prov.add_column("provider", style="cyan")
    prov.add_column("n", justify="right")
    prov.add_column("passed", justify="right")
    prov.add_column("mean_iter", justify="right")
    for k, slot in report["per_provider"].items():
        prov.add_row(k, str(slot["n"]), str(slot["passed"]),
                     f"{slot['mean_iter']:.2f}")
    print(prov)

    if output is not None:
        print(f"[dim]report written to {output}[/]")


@app.command("metriq-export")
def metriq_export(
    target: Path = typer.Argument(
        ..., help="Path to a single manifest .json or a run/ledger directory."
    ),
    output: Path = typer.Option(
        None, help="Optional path to write the JSON; otherwise printed."
    ),
    name: str = typer.Option(None, help="Override Submission.name."),
    repo: str = typer.Option(None, help="Submission.repo URL."),
    commit: str = typer.Option(None, help="Submission.commit SHA."),
    tags: str = typer.Option(None, help="Comma-separated tags."),
) -> None:
    """Export a manifest (or whole ledger dir) as a Metriq submission JSON."""
    from qmesh.metriq import export_manifest, export_run_dir

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    if target.is_dir():
        sub = export_run_dir(target, name=name, repo=repo, commit=commit, tags=tag_list)
    else:
        sub = export_manifest(target, name=name, repo=repo, commit=commit, tags=tag_list)
    payload = json.dumps(sub, indent=2, default=str)
    if output is not None:
        output.write_text(payload)
        print(f"[bold]wrote[/] {output}")
    else:
        print(payload)
    print(Panel.fit(
        f"[bold]submission name:[/] {sub['name']}\n"
        f"[bold]results:[/]         {len(sub['results'])}\n"
        f"[bold]tags:[/]            {', '.join(sub['tags'])}",
        title="qmesh metriq-export",
    ))


@app.command("compliance-pack")
def compliance_pack(
    ledger: Path = typer.Option(..., help="Ledger directory to bundle."),
    out: Path = typer.Option(..., help="Destination archive path (.tar.gz)."),
    since: str = typer.Option(None, help="Lower-bound ISO date (YYYY-MM-DD)."),
    until: str = typer.Option(None, help="Upper-bound ISO date (YYYY-MM-DD)."),
) -> None:
    """Build a tamper-evident compliance pack from a ledger directory."""
    from qmesh.compliance import build_pack

    out_path = build_pack(ledger, since=since, until=until, out_path=out)
    size_kb = out_path.stat().st_size / 1024
    print(Panel.fit(
        f"[bold]archive:[/]    {out_path}\n"
        f"[bold]size:[/]       {size_kb:.1f} KB\n"
        f"[bold]since:[/]      {since or '—'}\n"
        f"[bold]until:[/]      {until or '—'}",
        title="qmesh compliance-pack",
    ))


@app.command("compliance-verify")
def compliance_verify(
    archive: Path = typer.Argument(..., help="Path to compliance pack .tar.gz."),
) -> None:
    """Verify a compliance pack offline (no private key, no network)."""
    from qmesh.compliance import verify_pack

    r = verify_pack(archive)
    table = Table(title=f"compliance-verify: {archive.name}")
    table.add_column("field", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("chain_valid", "[green]✓[/]" if r.chain_valid else "[red]✗[/]")
    table.add_row("n_manifests", str(r.n_manifests))
    table.add_row("n_signatures_ok", str(r.n_signatures_ok))
    table.add_row("broken_at", str(r.broken_at) if r.broken_at is not None else "—")
    print(table)
    if r.notes:
        print(Panel.fit("\n".join(f"  • {n}" for n in r.notes), title="notes"))
    if r.errors:
        print(Panel.fit("\n".join(f"  • {e}" for e in r.errors), title="errors"))
    raise typer.Exit(code=0 if r.chain_valid else 1)


if __name__ == "__main__":
    app()
