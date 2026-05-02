"""Phase 3γ demo: reverse-arrow ChannelOps in one Module, end-to-end.

The Phase 3γ centrepiece. We build a single qmesh.ir Module that contains:

    RydbergOps (Pasqal-style ground-rydberg readout on 2 atoms)
        │
        ▼   ChannelOp(rydberg → gate)   payload: p_excited → ry(theta) scale
        │
    GateOps (a parametric ry rotation conditioned on the Rydberg readout)
        │
        ▼   ChannelOp(gate → cv)        payload: parity_even → squeezing r
        │
    CVOps (single-mode squeeze + homodyne readout)

Then we lower it through `qmesh.scheduler.channelop_lowering` into a hybrid
DAG and execute it with `qmesh.scheduler.execute()`. Each modality block
runs on its native backend (qmesh.pulser / qmesh.aer / qmesh.sf.gaussian),
each QPUPrimitive emits its own signed manifest, and the aggregate DAG
manifest references all of them — "ONE program, three modalities,
*reverse-arrow* data flowing across two ChannelOps, signed end-to-end."

Run:
    PYTHONPATH=. python3 examples/cross_modality_reverse.py
"""

from __future__ import annotations

from rich import print
from rich.panel import Panel
from rich.table import Table

import qmesh
from qmesh.ir.builder import circuit
from qmesh.ir.ops import ChannelOp, CVOp, MeasureOp, RydbergOp
from qmesh.ir.types import Atom, Bit, Modality
from qmesh.scheduler import execute
from qmesh.scheduler.channelop_lowering import lower_module_to_dag


def build_one_program() -> qmesh.Module:
    """The headline IR: one Module, three modalities, two reverse-arrow
    ChannelOps."""
    with circuit(
        "tri_modal_reverse", n_qubits=1, n_bits=5, n_atoms=2, n_qumodes=1,
    ) as c:
        # Pin atom positions for the Pasqal-style drive (5 µm → blockade)
        positioned = (
            Atom(name="a0", index=0, position=(0.0, 0.0, 0.0)),
            Atom(name="a1", index=1, position=(5.0, 0.0, 0.0)),
        )
        c.atoms = positioned
        c.module.functions[0].inputs = c.atoms + c.qubits + c.qumodes

        # ---------- rydberg block: Pasqal global drive + readout ----------
        c._region.append(RydbergOp(
            name="pulse", operands=tuple(positioned),
            # 2π Rabi for a partial-excitation readout on 2 atoms in blockade
            params=(6.28, 0.0, 0.0, 1000.0),
            modality=Modality.RYDBERG,
            attrs={"channel": "rydberg", "addressing": "Global"},
        ))
        for i, atom in enumerate(positioned):
            c._region.append(MeasureOp(
                operands=(atom, Bit(name=f"c{i}", index=i)),
                modality=Modality.RYDBERG,
                attrs={"basis": "ground-rydberg"},
            ))

        # ---------- channel: rydberg → gate (REVERSE arrow) ----------
        c._region.append(ChannelOp(
            source_modality=Modality.RYDBERG, target_modality=Modality.GATE,
            kind="rydberg->gate",
            payload={
                "forward": "p_excited",   # mean atom-excitation probability
                "into_param": "theta_scale",
                "scale": 1.0,
                "default": 0.0,
            },
        ))

        # ---------- gate block: ry conditioned on Rydberg readout ----------
        # Baseline ry angle = π. The channel scales it by p_excited so a
        # fully-excited Rydberg readout becomes a full π rotation, half
        # gives π/2, etc.
        c.ry(3.14159, 0)
        c.measure(0, 2)

        # ---------- channel: gate → cv ----------
        c._region.append(ChannelOp(
            source_modality=Modality.GATE, target_modality=Modality.CV,
            kind="gate->cv",
            payload={
                "forward": "parity_even",  # parity of the gate readout
                "into_param": "r",         # CV squeezing parameter
                "scale": 0.5,
                "default": 0.2,
            },
        ))

        # ---------- cv block: single-mode squeeze + homodyne ----------
        c._region.append(CVOp(
            name="Sgate", operands=(c.qumodes[0],),
            params=(0.0,),  # patched by the gate→cv channel
            modality=Modality.CV,
        ))
        c._region.append(MeasureOp(
            operands=(c.qumodes[0], Bit(name="c3", index=3)),
            modality=Modality.CV,
            attrs={"basis": "MeasureFock"},
        ))
    return c.module


def main() -> None:
    print(Panel.fit(
        "[bold cyan]ONE qmesh.ir Module — three modalities — REVERSE arrows[/]\n"
        "RydbergOps (Pasqal readout) [dim]→ ChannelOp(rydberg→gate) →[/] "
        "GateOps (ry conditioned)\n"
        "                            [dim]→ ChannelOp(gate→cv) →[/] "
        "CVOps (squeeze + Fock)\n\n"
        "[dim]Lowered to a DAG via qmesh.scheduler.channelop_lowering, executed[/]\n"
        "[dim]on qmesh.pulser + qmesh.aer + qmesh.sf.gaussian, every block signed.[/]",
        title="qmesh Phase 3γ — reverse-arrow ChannelOps + pulse bridges",
    ))

    module = build_one_program()
    print(f"\n[bold]source module hash:[/] {module.hash()}")

    dag = lower_module_to_dag(module, shots=64, sign=True)

    layout = Table(title="lowered DAG (Phase 3γ)")
    layout.add_column("node id", style="cyan")
    layout.add_column("kind")
    layout.add_column("backend / fn")
    layout.add_column("depends on", overflow="fold")
    for n in dag.toposort():
        backend_or_fn = (
            getattr(n, "backend", None)
            or (n.fn.__name__ if hasattr(n, "fn") and n.fn else "—")
        )
        layout.add_row(
            n.id, n.kind(), str(backend_or_fn),
            ", ".join(n.depends_on) if n.depends_on else "—",
        )
    print(layout)

    run = execute(dag, ledger_dir="ledger/cross_modality_reverse", sign=True)

    out = Table(title="per-node manifest hashes")
    out.add_column("node id", style="cyan")
    out.add_column("kind")
    out.add_column("ok?")
    out.add_column("manifest hash")
    out.add_column("metadata", overflow="fold")
    for nid, nr in run.results.items():
        meta_str = ", ".join(f"{k}={v}" for k, v in (nr.metadata or {}).items()
                             if k in ("backend", "wall_seconds"))
        out.add_row(
            nid, nr.kind, "[green]ok[/]" if nr.ok else "[red]fail[/]",
            (nr.manifest_hash[:24] if nr.manifest_hash else "—"),
            meta_str or "—",
        )
    print(out)

    ryd = run.results.get("qpu_0_rydberg", None)
    gate = run.results.get("qpu_1_gate", None)
    cv = run.results.get("qpu_2_cv", None)
    ch1 = run.context.get("channel_1_rydberg_to_gate", {}) or {}
    ch2 = run.context.get("channel_2_gate_to_cv", {}) or {}

    print()
    if ryd and ryd.ok:
        print(f"[bold]Rydberg counts (top4):[/] "
              f"{dict(sorted(ryd.output['counts'].items(), key=lambda kv: -kv[1])[:4])}")
    print(f"[bold]ryd→gate channel  :[/] p_excited={ch1.get('stat', 0):.4f}  "
          f"per-atom={ch1.get('p_excited_per_atom', [])}\n"
          f"                    [dim]→[/] "
          f"theta_scale = {ch1.get('theta_scale', 0):.4f} "
          f"(ry baseline π is *scaled* by this)")
    if gate and gate.ok:
        print(f"[bold]Gate counts       :[/] "
              f"{dict(sorted(gate.output['counts'].items(), key=lambda kv: -kv[1])[:4])}")
    print(f"[bold]gate→cv channel   :[/] parity_even={ch2.get('stat', 0):.4f}  "
          f"[dim]→[/] squeezing r = {ch2.get('r', 0):.4f}")
    if cv and cv.ok:
        print(f"[bold]CV counts (top4)  :[/] "
              f"{dict(sorted(cv.output['counts'].items(), key=lambda kv: -kv[1])[:4])}")

    print()
    print(Panel.fit(
        f"[bold green]aggregate DAG manifest:[/] {run.manifest_hash}\n"
        f"[dim]wall:[/] {run.duration_seconds:.2f}s   "
        f"[dim]nodes:[/] {len(run.results)} "
        f"({sum(1 for r in run.results.values() if r.kind == 'qpu')} QPU, "
        f"{sum(1 for r in run.results.values() if r.kind == 'classical')} classical)\n"
        f"[dim]signed manifest path:[/] {run.manifest_path}",
        title="Reverse arrows complete: ONE program, three modalities, signed",
    ))


if __name__ == "__main__":
    main()
