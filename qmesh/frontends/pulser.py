"""qmesh.frontends.pulser — Pulser ↔ qmesh.ir.

Pasqal's Pulser describes neutral-atom programs as `Sequence` objects on a
`Register` of atoms, with `Pulse` and delay events on declared channels
(rydberg_global, rydberg_local, ramen, etc.). Each pulse parametrises an
analog drive over the Rydberg blockade.

We translate:
    Pulser Sequence → qmesh.ir Module with RydbergOps + DelayOps + MeasureOps
    (atom positions captured as `Atom.position`)

This is the first frontend that exercises the **Rydberg modality** of qmesh
and proves the modality-agnostic IR is real.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qmesh.ir.builder import circuit
from qmesh.ir.module import Module

if TYPE_CHECKING:
    from pulser import Sequence  # noqa: F401


def from_pulser(sequence: "Sequence") -> Module:
    """Translate a (concrete, non-parametrised) Pulser Sequence into qmesh.ir."""
    if sequence.is_parametrized():
        raise NotImplementedError(
            "qmesh.pulser.from_pulser: parametrised Pulser sequences are "
            "Phase-3β work. Resolve with sequence.build(...) first."
        )

    register = sequence.register
    atom_ids = list(register.qubit_ids)
    n_atoms = len(atom_ids)
    n_bits = n_atoms  # measure → 1 bit per atom (ground-rydberg readout)

    with circuit("pulser_seq", n_qubits=0, n_bits=n_bits, n_atoms=n_atoms) as c:
        # Set atom positions from the Register
        coords = register.qubits
        new_atoms = []
        for i, aid in enumerate(atom_ids):
            pos = coords[aid]
            try:
                pos_t = (float(pos[0]), float(pos[1]),
                         float(pos[2]) if len(pos) > 2 else 0.0)
            except Exception:
                pos_t = (0.0, 0.0, 0.0)
            from qmesh.ir.types import Atom
            new_atoms.append(Atom(name=f"a{i}", index=i, position=pos_t))
        c.atoms = tuple(new_atoms)
        # Replace the Function input list with the typed atoms
        c.module.functions[0].inputs = tuple(new_atoms)

        # Channels
        channels = list(sequence.declared_channels.keys())

        # Walk the schedule
        for ch_name in channels:
            ch_sched = sequence._schedule[ch_name]
            for slot in ch_sched:
                kind = getattr(slot, "type", None)
                ti = getattr(slot, "ti", 0)
                tf = getattr(slot, "tf", 0)
                duration = max(0, int(tf) - max(0, int(ti)))

                # `slot.type` is a Pulse for actual pulses, "delay" string for delays,
                # "target" / "detuning_map" for setup ops we skip.
                if isinstance(kind, str) and kind == "delay":
                    if duration > 0:
                        # delay applies to all atoms in the global channel
                        c.delay(duration, q=None) if False else c._region.append(
                            _delay_op(new_atoms, duration_ns=duration)
                        )
                    continue

                # Pulse object
                from pulser.pulse import Pulse
                if isinstance(kind, Pulse):
                    amp = _waveform_average(kind.amplitude)
                    det = _waveform_average(kind.detuning)
                    phase = float(kind.phase)
                    addressing = sequence.declared_channels[ch_name].addressing
                    rb_op = _rydberg_pulse_op(
                        atoms=new_atoms,
                        ch_name=ch_name,
                        addressing=addressing,
                        amp_rad_per_us=amp,
                        det_rad_per_us=det,
                        phase=phase,
                        duration_ns=duration,
                    )
                    c._region.append(rb_op)

        # Measurement basis (Pulser's seq.measure)
        if sequence._measurement is not None:
            for i in range(n_atoms):
                c.measure(0, i) if False else c._region.append(_atom_measure_op(new_atoms[i], i))
    return c.module


def _waveform_average(wf) -> float:
    """Best-effort scalar amplitude/detuning for a Pulser waveform."""
    try:
        # Constant case
        return float(wf._value)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        # Generic: average of the sampled samples
        s = wf.samples  # numpy array of duration samples
        if len(s):
            return float(s.mean())
    except Exception:
        pass
    return 0.0


def _rydberg_pulse_op(
    *, atoms, ch_name: str, addressing: str,
    amp_rad_per_us: float, det_rad_per_us: float, phase: float, duration_ns: int,
):
    """Construct a RydbergOp."""
    from qmesh.ir.ops import RydbergOp
    from qmesh.ir.types import Modality
    return RydbergOp(
        name="pulse",
        operands=tuple(atoms),
        params=(amp_rad_per_us, det_rad_per_us, phase, float(duration_ns)),
        modality=Modality.RYDBERG,
        attrs={"channel": ch_name, "addressing": addressing},
    )


def _delay_op(atoms, duration_ns: int):
    from qmesh.ir.ops import DelayOp
    from qmesh.ir.types import Modality
    return DelayOp(
        name="delay",
        operands=tuple(atoms),
        params=(float(duration_ns),),
        modality=Modality.RYDBERG,
    )


def _atom_measure_op(atom, bit_index: int):
    from qmesh.ir.ops import MeasureOp
    from qmesh.ir.types import Bit, Modality
    return MeasureOp(
        operands=(atom, Bit(name=f"c{bit_index}", index=bit_index)),
        modality=Modality.RYDBERG,
        attrs={"basis": "ground-rydberg"},
    )


__all__ = ["from_pulser"]
