"""Cross-modality ChannelOp demo: ONE Module, three modalities, signed e2e.

The Phase 3β α centrepiece. We build a single qmesh.ir Module that contains:

    GateOps (Bell prep on qubits)
        │
        ▼   ChannelOp(gate → rydberg)   payload: parity_even × 6.28 → Rabi amp
        │
    RydbergOps (Pasqal-style 2π drive over a 3-atom register)
        │
        ▼   ChannelOp(rydberg → cv)     payload: p_excited × 0.5 → squeezing r
        │
    CVOps (two-mode squeezed + beam-splitter readout)

then lower it through `qmesh.scheduler.channelop_lowering` into a hybrid DAG
and execute it with `qmesh.scheduler.execute()`. Each modality block runs on
its native backend (qmesh.aer / qmesh.pulser / qmesh.sf.gaussian), each
QPUPrimitive emits its own signed manifest, and the aggregate DAG manifest
references all of them — "ONE program, three modalities, signed end-to-end."

Run:
    PYTHONPATH=. python3 examples/cross_modality_channelop.py
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
    """The headline IR: one Module, three modalities, two ChannelOps."""
    with circuit(
        "tri_modal", n_qubits=2, n_bits=6, n_atoms=3, n_qumodes=2,
    ) as c:
        # ---------- gate block: Bell prep ----------
        c.h(0)
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)

        # ---------- channel: gate → rydberg ----------
        c._region.append(ChannelOp(
            source_modality=Modality.GATE,
            target_modality=Modality.RYDBERG,
            kind="gate->rydberg",
            payload={
                "forward": "parity_even",       # Bell parity
                "into_param": "amp_rad_per_us", # patches RydbergOp.params[0]
                "scale": 6.28,                  # 2π Rabi for a perfect Bell
                "default": 3.14,
            },
        ))

        # Pin atom positions for the Pasqal-style drive (5 µm spacing → blockade)
        positioned = (
            Atom(name="a0", index=0, position=(0.0, 0.0, 0.0)),
            Atom(name="a1", index=1, position=(5.0, 0.0, 0.0)),
            Atom(name="a2", index=2, position=(10.0, 0.0, 0.0)),
        )
        c.atoms = positioned
        c.module.functions[0].inputs = c.qubits + c.atoms + c.qumodes

        # ---------- rydberg block: Pasqal-shaped global drive ----------
        c._region.append(RydbergOp(
            name="pulse",
            operands=tuple(positioned),
            # params: (amp [rad/µs], detuning, phase, duration_ns)
            # amp will be patched by the channel task at run time.
            params=(0.0, 0.0, 0.0, 1000.0),
            modality=Modality.RYDBERG,
            attrs={"channel": "rydberg", "addressing": "Global"},
        ))
        for i, atom in enumerate(positioned):
            c._region.append(MeasureOp(
                operands=(atom, Bit(name=f"c{2 + i}", index=2 + i)),
                modality=Modality.RYDBERG,
                attrs={"basis": "ground-rydberg"},
            ))

        # ---------- channel: rydberg → cv ----------
        c._region.append(ChannelOp(
            source_modality=Modality.RYDBERG,
            target_modality=Modality.CV,
            kind="rydberg->cv",
            payload={
                "forward": "p_excited",   # fraction of shots with any '1'
                "into_param": "r",        # patches CVOp.params[0] (squeezing)
                "scale": 0.5,
                "default": 0.2,
            },
        ))

        # ---------- cv block: two-mode squeezed + beam splitter ----------
        c._region.append(CVOp(
            name="Sgate", operands=(c.qumodes[0],),
            params=(0.0,),  # patched by channel task
            modality=Modality.CV,
        ))
        c._region.append(CVOp(
            name="Sgate", operands=(c.qumodes[1],),
            params=(0.4,),
            modality=Modality.CV,
        ))
        c._region.append(CVOp(
            name="BSgate", operands=(c.qumodes[0], c.qumodes[1]),
            params=(0.785, 0.0),
            modality=Modality.CV,
        ))
        for i, qm in enumerate(c.qumodes):
            c._region.append(MeasureOp(
                operands=(qm, Bit(name=f"c{5 + i}", index=5 + i)),
                modality=Modality.CV,
                attrs={"basis": "MeasureFock"},
            ))
    return c.module


def main() -> None:
    print(Panel.fit(
        "[bold cyan]ONE qmesh.ir Module — three modalities — signed end-to-end[/]\n"
        "GateOps (Bell prep) [dim]→ ChannelOp(gate→rydberg) →[/] RydbergOps "
        "(Pasqal drive)\n"
        "                     [dim]→ ChannelOp(rydberg→cv) →[/] CVOps "
        "(two-mode BS readout)\n\n"
        "[dim]Lowered to a DAG via qmesh.scheduler.channelop_lowering, executed[/]\n"
        "[dim]on qmesh.aer + qmesh.pulser + qmesh.sf.gaussian, each block signed.[/]",
        title="qmesh Phase 3β α — first-class ChannelOp",
    ))

    module = build_one_program()
    print(f"\n[bold]source module hash:[/] {module.hash()}")

    dag = lower_module_to_dag(module, shots=64, sign=True)

    layout = Table(title="lowered DAG (Phase 3β α)")
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

    run = execute(dag, ledger_dir="ledger/cross_modality_channelop", sign=True)

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
            nid, nr.kind, "[green]✓[/]" if nr.ok else "[red]✗[/]",
            (nr.manifest_hash[:24] if nr.manifest_hash else "—"),
            meta_str or "—",
        )
    print(out)

    bell = run.results["qpu_0_gate"].output
    ryd = run.results["qpu_1_rydberg"].output
    cv = run.results["qpu_2_cv"].output
    ch1 = run.context.get("channel_1_gate_to_rydberg", {})
    ch2 = run.context.get("channel_2_rydberg_to_cv", {})

    print()
    print(f"[bold]Bell counts        :[/] {dict(sorted(bell['counts'].items()))}")
    print(f"[bold]gate→ryd channel   :[/] parity_even={ch1.get('stat', 0):.4f}  "
          f"→ Rabi amp = {ch1.get('amp_rad_per_us', 0):.4f} rad/µs")
    print(f"[bold]Rydberg counts (top4):[/] "
          f"{dict(sorted(ryd['counts'].items(), key=lambda kv: -kv[1])[:4])}")
    print(f"[bold]ryd→cv channel     :[/] p_excited={ch2.get('stat', 0):.4f}  "
          f"→ squeezing r = {ch2.get('r', 0):.4f}")
    print(f"[bold]CV counts (top4)   :[/] "
          f"{dict(sorted(cv['counts'].items(), key=lambda kv: -kv[1])[:4])}")

    print()
    print(Panel.fit(
        f"[bold green]aggregate DAG manifest:[/] {run.manifest_hash}\n"
        f"[dim]wall:[/] {run.duration_seconds:.2f}s   "
        f"[dim]nodes:[/] {len(run.results)} "
        f"({sum(1 for r in run.results.values() if r.kind == 'qpu')} QPU, "
        f"{sum(1 for r in run.results.values() if r.kind == 'classical')} classical)\n"
        f"[dim]signed manifest path:[/] {run.manifest_path}",
        title="ONE program, three modalities, signed end-to-end",
    ))


if __name__ == "__main__":
    main()
