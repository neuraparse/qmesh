"""Cross-modality DAG: gate-prep → classical bridge → Rydberg drive → CV readout.

This is qmesh's central thesis executed end-to-end as a single graph:

   ┌──────────────────────┐    ┌─────────────────────┐
   │ QPU: prep Bell       │ →  │ Classical: derive   │ →
   │ (gate, qmesh.aer)    │    │ Rydberg ω from p₁   │
   └──────────────────────┘    └─────────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────┐
                              │ QPU: Rydberg drive      │ →
                              │ (rydberg, qmesh.pulser) │
                              └─────────────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────┐
                              │ Classical: derive CV    │ →
                              │ squeezing from Rydberg  │
                              └─────────────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────┐
                              │ QPU: photonic readout   │
                              │ (cv, qmesh.sf.gaussian) │
                              └─────────────────────────┘

Each step is a real run on a real backend; the next step's parameters are
*derived* from the previous step's measurement counts via Python — exactly
the pattern that hybrid HPC+QC programs use today (RIKEN/NERSC/ORNL/JSC).
The `ChannelOp` IR primitive will, in Phase 4β, replace the classical
bridges with first-class entanglement carriers; for now the DAG plumbs
results through the Python context.

Run:
    PYTHONPATH=. python3 examples/dag_cross_modality.py
"""

from __future__ import annotations

from rich import print
from rich.panel import Panel

import qmesh
from qmesh.scheduler import DAG, ClassicalTask, QPUPrimitive, execute


def gate_module() -> qmesh.Module:
    with qmesh.circuit("bell_prep", n_qubits=2, n_bits=2) as c:
        c.h(0)
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)
    return c.module


def derive_rabi(ctx: dict) -> dict:
    """Use the Bell-state outcome statistics to set a Rydberg Rabi amplitude.

    Faithful but artificial: ω = 2π × (1 − parity_violation), so a perfect
    Bell state targets a 2π rotation, an imperfect prep softens it. This
    is the kind of feedback loop we want the DAG to express end-to-end.
    """
    counts = ctx["bell_prep_node"]["counts"]
    total = sum(counts.values())
    parity = sum(v for k, v in counts.items() if k.count("1") % 2 == 0) / total
    omega = 6.28318 * parity                # rad/µs
    return {"omega_rad_per_us": omega, "parity": parity}


def rydberg_factory(ctx: dict) -> qmesh.Module:
    """Build the Rydberg sequence using ω from the upstream classical task."""
    import pulser
    from pulser import Pulse, Register, Sequence
    from pulser.devices import MockDevice

    from qmesh.frontends.pulser import from_pulser

    omega = ctx["derive_rabi_node"]["omega_rad_per_us"]
    reg = Register({"q0": (0, 0), "q1": (5, 0), "q2": (10, 0)})
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg", "rydberg_global")
    seq.add(Pulse.ConstantPulse(1000, omega, 0, 0), "rydberg")
    seq.measure("ground-rydberg")
    return from_pulser(seq)


def derive_squeezing(ctx: dict) -> dict:
    """Map Rydberg measurement statistics to a CV squeezing parameter."""
    counts = ctx["rydberg_node"]["counts"]
    total = sum(counts.values())
    p_excited = (
        sum(v for k, v in counts.items() if k.count("1") >= 1) / total
        if total else 0.0
    )
    # squeezing ∈ [0, 0.6]: more excitation → more squeezing
    r = 0.6 * p_excited
    return {"squeezing_r": r, "p_excited": p_excited}


def cv_factory(ctx: dict) -> qmesh.Module:
    import strawberryfields as sf
    from strawberryfields import ops

    from qmesh.frontends.sf import from_sf

    r = ctx["derive_squeezing_node"]["squeezing_r"]
    prog = sf.Program(2)
    with prog.context as q:
        ops.Sgate(r) | q[0]
        ops.Sgate(r) | q[1]
        ops.BSgate(0.785, 0) | (q[0], q[1])
        ops.MeasureFock() | q[0]
        ops.MeasureFock() | q[1]
    return from_sf(prog)


def main() -> None:
    print(Panel.fit(
        "[bold cyan]cross-modality DAG[/]\n"
        "gate (Aer) → classical bridge → Rydberg (Pulser) →\n"
        "classical bridge → photonic CV (Strawberry Fields).\n\n"
        "[dim]Each backend's measurements drive the next module's parameters.[/]",
        title="qmesh Phase-4α",
    ))

    dag = DAG()
    bell = dag.add(QPUPrimitive(
        id="bell_prep_node", name="prep Bell state",
        module=gate_module(), backend="qmesh.aer", shots=512, sign=False,
    ))
    rabi = dag.add(ClassicalTask(
        id="derive_rabi_node", name="derive Rydberg Rabi from Bell parity",
        fn=derive_rabi, depends_on=[bell.id],
    ))
    rydberg = dag.add(QPUPrimitive(
        id="rydberg_node", name="Rydberg drive",
        module_factory=rydberg_factory, backend="qmesh.pulser",
        shots=64, sign=False, depends_on=[rabi.id],
    ))
    sqz = dag.add(ClassicalTask(
        id="derive_squeezing_node", name="derive CV squeezing from Rydberg",
        fn=derive_squeezing, depends_on=[rydberg.id],
    ))
    cv = dag.add(QPUPrimitive(
        id="cv_node", name="photonic readout",
        module_factory=cv_factory, backend="qmesh.sf.gaussian",
        shots=64, sign=False, depends_on=[sqz.id],
    ))

    run = execute(dag, ledger_dir="ledger/dag_cross_modality")
    print()
    print(f"[bold]bell parity:[/]    {run.context[rabi.id]['parity']:.4f}")
    print(f"[bold]Rabi ω:[/]         {run.context[rabi.id]['omega_rad_per_us']:.4f} rad/µs")
    print(f"[bold]Rydberg p_exc:[/]  {run.context[sqz.id]['p_excited']:.4f}")
    print(f"[bold]CV r:[/]           {run.context[sqz.id]['squeezing_r']:.4f}")
    print()
    print(f"[bold]bell counts:[/]    {dict(sorted(run.context[bell.id]['counts'].items()))}")
    print(f"[bold]Rydberg counts:[/] {dict(sorted(run.context[rydberg.id]['counts'].items())[:4])}")
    print(f"[bold]CV counts:[/]      {dict(sorted(run.context[cv.id]['counts'].items(), key=lambda kv: -kv[1])[:4])}")
    print()
    print(f"[dim]wall: {run.duration_seconds:.2f}s, "
          f"5 nodes, 3 modalities, 1 signed manifest at {run.manifest_path}[/]")


if __name__ == "__main__":
    main()
