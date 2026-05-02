"""qmesh.backends.pulser_sim — Pasqal Pulser QutipEmulator wrapper.

Phase-3α: registers as `qmesh.pulser` and accepts qmesh.ir Modules whose
ops are in the Rydberg modality. Lowers the IR back into a Pulser
`Sequence` and runs it through `pulser_simulation.QutipEmulator`.

Round-trip path:
    Pulser Sequence  →  qmesh.frontends.pulser.from_pulser  →  qmesh.ir.Module
                                                                 │
                                          ┌──────────────────────┘
                                          ▼
                       qmesh.backends.pulser_sim   →   QutipEmulator   →   counts
"""

from __future__ import annotations

import time
from typing import Any

from qmesh.backends.base import Backend, Capabilities, ControlLevel, RunResult
from qmesh.backends.registry import register
from qmesh.ir.module import Module
from qmesh.ir.ops import DelayOp, MeasureOp, RydbergOp
from qmesh.ir.types import Modality


def _ir_to_pulser(module: Module):
    """Lower a Rydberg-modality qmesh.ir Module to a Pulser Sequence."""
    from pulser import Pulse, Register, Sequence
    from pulser.devices import MockDevice

    from qmesh.ir.types import Atom

    # Build the register from atom positions
    atoms: list[Atom] = []
    for f in module.functions:
        for a in f.inputs:
            if isinstance(a, Atom):
                atoms.append(a)
    if not atoms:
        raise ValueError("no Atom operands found; not a Rydberg-modality module")

    qubits = {f"q{a.index}": (a.position[0] if a.position else 0.0,
                              a.position[1] if a.position else 0.0)
              for a in atoms}
    register = Register(qubits)
    seq = Sequence(register, MockDevice)

    # Declare any channels referenced
    declared: set[str] = set()
    for f in module.functions:
        for op in f.body.ops:
            if isinstance(op, RydbergOp):
                ch = op.attrs.get("channel", "rydberg")
                if ch not in declared:
                    addressing = op.attrs.get("addressing", "Global")
                    chan = ("rydberg_global"
                            if addressing == "Global" else "rydberg_local")
                    seq.declare_channel(ch, chan)
                    declared.add(ch)

    # Replay ops
    measured = False
    for f in module.functions:
        for op in f.body.ops:
            if isinstance(op, RydbergOp):
                ch = op.attrs.get("channel", "rydberg")
                amp, det, phase, duration = op.params
                seq.add(Pulse.ConstantPulse(int(duration), amp, det, phase), ch)
            elif isinstance(op, DelayOp):
                duration = int(op.params[0]) if op.params else 0
                if duration > 0:
                    # delay on the first declared channel
                    if declared:
                        seq.delay(duration, next(iter(declared)))
            elif isinstance(op, MeasureOp):
                if not measured:
                    basis = op.attrs.get("basis", "ground-rydberg")
                    seq.measure(basis)
                    measured = True
    return seq


class PulserSimulator(Backend):
    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="qmesh.pulser",
            vendor="qmesh+pulser",
            modalities={Modality.RYDBERG},
            qubit_count=20,                # QutipEmulator practical ceiling
            native_gates=set(),             # analog — no discrete gate set
            measurement_feedforward=False,
            classical_control=ControlLevel.NONE,
            cost_per_shot_usd=0.0,
            queue_depth=0,
            fidelity_2q_typical=0.99,
            is_simulator=True,
            notes="Pasqal Pulser QutipEmulator — neutral-atom analog simulation",
        )

    def run(self, module: Module, shots: int = 1024, **_: Any) -> RunResult:
        from pulser_simulation import QutipEmulator

        ok, why = self.accepts(module)
        if not ok:
            raise ValueError(f"pulser backend rejected module: {why}")

        sequence = _ir_to_pulser(module)
        emu = QutipEmulator.from_sequence(sequence)

        t0 = time.time()
        result = emu.run()
        # Build counts from sampled bitstrings
        try:
            sample = result.sample_final_state(N_samples=shots)
            counts: dict[str, int] = {str(k): int(v) for k, v in sample.items()}
        except Exception:
            counts = {}
        wall = time.time() - t0

        return RunResult(
            counts=counts,
            shots=shots,
            wall_seconds=wall,
            qpu_seconds=wall,
            cost_usd=0.0,
            backend_metadata={
                "n_atoms": len(sequence.register.qubit_ids),
                "duration_ns": int(sequence.get_duration()),
                "channels": list(sequence.declared_channels.keys()),
            },
        )


register(PulserSimulator())
