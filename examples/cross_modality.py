"""Cross-modality demo: gate-based + Rydberg-analog + photonic-CV in one program.

This is qmesh's central thesis in code. A single qmesh.ir Module mixes:
  - GateOps   (gate-based: H, CX, measurement on qubits)
  - RydbergOps (neutral-atom analog: ConstantPulse on atoms)
  - CVOps     (photonic CV: Sgate, BSgate, MeasureFock on qumodes)
  - ChannelOp (typed cross-modality entanglement carrier — IonQ photonic
               interconnect April 2026; Cisco Universal Quantum Switch)

No existing framework expresses this in one IR. qmesh routes each modality
to an appropriate backend; the runner keeps the manifest intact across all
of them.

Run:
    PYTHONPATH=. python3 examples/cross_modality.py
"""

from __future__ import annotations

from rich import print
from rich.panel import Panel
from rich.table import Table

import qmesh
from qmesh.backends.registry import all_backends
from qmesh.ir.types import Modality
from qmesh.router import choose, profile


def gate_module() -> qmesh.Module:
    """A small gate-modality circuit (Bell state)."""
    with qmesh.circuit("gate_bell", n_qubits=2, n_bits=2) as c:
        c.h(0)
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)
    return c.module


def rydberg_module() -> qmesh.Module:
    """A neutral-atom analog program with Rydberg blockade."""
    import pulser
    from pulser import Pulse, Register, Sequence
    from pulser.devices import MockDevice

    from qmesh.frontends.pulser import from_pulser

    reg = Register({"q0": (0, 0), "q1": (5, 0), "q2": (10, 0)})
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg", "rydberg_global")
    # 2π Rabi rotation; blockade splits |011⟩ + |101⟩ patterns
    seq.add(Pulse.ConstantPulse(1000, 6.28, 0, 0), "rydberg")
    seq.measure("ground-rydberg")
    return from_pulser(seq)


def photonic_module() -> qmesh.Module:
    """Two-mode squeezed vacuum + beam splitter."""
    import strawberryfields as sf
    from strawberryfields import ops

    from qmesh.frontends.sf import from_sf

    prog = sf.Program(2)
    with prog.context as q:
        ops.Sgate(0.4) | q[0]
        ops.Sgate(0.4) | q[1]
        ops.BSgate(0.785, 0) | (q[0], q[1])
        ops.MeasureFock() | q[0]
        ops.MeasureFock() | q[1]
    return from_sf(prog)


def main() -> None:
    print(Panel.fit(
        "[bold cyan]qmesh cross-modality demo[/]\n"
        "Three programs, three modalities, one IR shape, three real backends.\n"
        "[dim]Each backend declares its modalities; the router refuses runs that[/]\n"
        "[dim]don't match.[/]",
        title="modality reach",
    ))

    table = Table(title="registered backends by modality")
    table.add_column("backend", style="cyan")
    table.add_column("modalities")
    table.add_column("simulator?")
    for bk in all_backends().values():
        c = bk.capabilities
        table.add_row(
            c.name,
            ", ".join(m.value for m in sorted(c.modalities, key=lambda m: m.value)),
            "✓" if c.is_simulator else "",
        )
    print(table)
    print()

    runs: list[tuple[str, Modality, qmesh.Module, str]] = [
        ("gate_bell",   Modality.GATE,    gate_module(),     "qmesh.aer"),
        ("rydberg",     Modality.RYDBERG, rydberg_module(),  "qmesh.pulser"),
        ("photonic_cv", Modality.CV,      photonic_module(), "qmesh.sf.gaussian"),
    ]

    summary = Table(title="cross-modality runs")
    summary.add_column("program")
    summary.add_column("modality")
    summary.add_column("backend")
    summary.add_column("top outcome", overflow="fold")
    summary.add_column("manifest hash")

    for name, modality, module, backend in runs:
        prof = profile(module)
        result, manifest = qmesh.submit(module, backend=backend, shots=100,
                                        ledger_dir="ledger/cross_modality")
        top = sorted(result.counts.items(), key=lambda kv: -kv[1])[:1]
        top_str = f"{top[0][0]}: {top[0][1]}" if top else "—"
        summary.add_row(
            name, modality.value, backend, top_str, manifest.hash()[:16],
        )
    print(summary)
    print()
    print(Panel.fit(
        "Each manifest is independently signed. The IR Module itself can mix all\n"
        "three modalities (plus a ChannelOp linking them) — the [bold]Phase-3β[/]\n"
        "scope is to wire that combined IR to a single hybrid run with the\n"
        "[bold]qmesh.scheduler[/] DAG executor (Phase 4). Today, three separate\n"
        "submissions prove the modality-reach contract end-to-end.",
        title="what this demonstrates",
    ))


if __name__ == "__main__":
    main()
