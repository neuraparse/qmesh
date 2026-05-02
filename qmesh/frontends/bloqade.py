"""qmesh.frontends.bloqade — Bloqade ↔ qmesh.ir.

QuEra's Bloqade describes neutral-atom programs in two flavours:

    1. **Hamiltonian / analog** — a `Register` of atoms together with global
       (and increasingly local) drives parametrised by piecewise-linear
       (or piecewise-constant) Rabi amplitude / detuning / phase
       waveforms. This is the QuEra Aquila execution model and it lines
       up almost 1:1 with Pulser. The official Python entry-point is the
       Julia-based `bloqade` (and the newer pure-Python `bloqade-python`
       / `bloqade-analog`); both expose programs that can be lowered to
       a sequence of (atom_positions, durations, amplitudes, detunings,
       phases) tuples per drive segment.

    2. **Digital / gate** — `bloqade-digital` exposes atom-native gate
       sequences. Out of scope for Phase 3β α; we keep the door open via
       a `kind="digital"` branch that raises NotImplementedError with a
       clear pointer.

We translate (analog flavour):

    Bloqade program → qmesh.ir Module with RydbergOps + DelayOps + MeasureOps
    (atom positions captured as `Atom.position`, drive segments as one
    RydbergOp per (channel, segment) pair).

This mirrors qmesh.frontends.pulser exactly so a downstream backend can
trivially round-trip into a Pulser Sequence (the QuEra Aquila simulator
is not yet Python-installable on every host, so the qmesh.pulser backend
doubles as the run target during Phase 3β).

α-quality: real Bloqade installs duck-type into the same shape. If
Bloqade is not installed (today, on this host, May 2026), tests use the
`BloqadeProgram` shim defined below — the same shape a real Bloqade
program lowers to via `prog.to_segments()`. Tests gate on
`pytest.importorskip("bloqade")` per the Phase 3β α contract.

The shim is intentionally minimal: a `Register` of atom positions, a
`drives: list[DriveSegment]` schedule, and a `measurement_basis` flag.
That is the smallest faithful subset that an Aquila run needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from qmesh.ir.builder import circuit
from qmesh.ir.module import Module
from qmesh.ir.ops import DelayOp, MeasureOp, RydbergOp
from qmesh.ir.types import Atom, Bit, Modality

if TYPE_CHECKING:
    pass  # real bloqade types resolved lazily inside from_bloqade


# ---------- α-quality shim ----------

@dataclass(slots=True)
class BloqadeAtom:
    """Atom site. Position in micrometres."""
    position: tuple[float, float, float]


@dataclass(slots=True)
class BloqadeRegister:
    """Lattice of neutral atoms. Mirrors `bloqade.atom_arrangement.AtomArrangement`."""
    atoms: list[BloqadeAtom] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @classmethod
    def from_positions(
        cls, positions: list[tuple[float, float] | tuple[float, float, float]]
    ) -> BloqadeRegister:
        atoms = []
        for p in positions:
            if len(p) == 2:
                atoms.append(BloqadeAtom(position=(float(p[0]), float(p[1]), 0.0)))
            else:
                atoms.append(BloqadeAtom(
                    position=(float(p[0]), float(p[1]), float(p[2]))))
        return cls(atoms=atoms)


@dataclass(slots=True)
class DriveSegment:
    """A single piecewise-constant drive segment.

    Mirrors the per-segment record that Bloqade's lowering pipeline emits
    when it hands an Aquila program off to the QuEra hardware schedule.
    """
    duration_ns: int
    amplitude_rad_per_us: float = 0.0
    detuning_rad_per_us: float = 0.0
    phase: float = 0.0
    channel: str = "rydberg_global"
    addressing: str = "Global"  # | "Local"


@dataclass(slots=True)
class BloqadeProgram:
    """α-quality shim of a Bloqade analog program.

    A real `bloqade.HamiltonianProgram` (or `bloqade-analog` `Routine`)
    lowers to the same triple — `(register, drives, measurement_basis)` —
    via its public `.to_segments()` / `.compile()` API. Tests can build
    one of these directly.
    """
    register: BloqadeRegister
    drives: list[DriveSegment] = field(default_factory=list)
    measurement_basis: str | None = "ground-rydberg"


# ---------- public translator ----------

def from_bloqade(prog: Any, *, name: str = "bloqade_program") -> Module:
    """Translate a Bloqade analog program into a qmesh.ir Module.

    Accepts:
        - a `BloqadeProgram` shim (this file's dataclass)
        - a real `bloqade` HamiltonianProgram / Routine (duck-typed: must
          expose `.register` with iterable atom positions and `.drives`
          with DriveSegment-shaped records, OR a `.to_segments()` method
          returning that pair)

    Returns a Module whose Function has Atoms as inputs and Bits as
    outputs (one per atom, when measurement_basis is set).
    """
    register, drives, measurement_basis = _normalise(prog)
    n_atoms = len(register.atoms)
    n_bits = n_atoms if measurement_basis else 0

    with circuit(name, n_qubits=0, n_bits=n_bits, n_atoms=n_atoms) as c:
        # Replace the auto-generated Atoms with positioned ones.
        new_atoms = tuple(
            Atom(name=f"a{i}", index=i, position=a.position)
            for i, a in enumerate(register.atoms)
        )
        c.atoms = new_atoms
        c.module.functions[0].inputs = new_atoms

        # Emit one RydbergOp per drive segment (or DelayOp for zero-amp idle).
        for seg in drives:
            duration = max(0, int(seg.duration_ns))
            if duration == 0:
                continue
            if seg.amplitude_rad_per_us == 0.0 and seg.detuning_rad_per_us == 0.0:
                # idle window — use DelayOp on the Rydberg modality so
                # backends that care about it can re-emit the gap.
                c._region.append(DelayOp(
                    name="delay",
                    operands=new_atoms,
                    params=(float(duration),),
                    modality=Modality.RYDBERG,
                ))
                continue
            c._region.append(RydbergOp(
                name="pulse",
                operands=new_atoms,
                params=(
                    float(seg.amplitude_rad_per_us),
                    float(seg.detuning_rad_per_us),
                    float(seg.phase),
                    float(duration),
                ),
                modality=Modality.RYDBERG,
                attrs={"channel": seg.channel, "addressing": seg.addressing,
                       "frontend": "bloqade"},
            ))

        if measurement_basis:
            for i, atom in enumerate(new_atoms):
                c._region.append(MeasureOp(
                    operands=(atom, Bit(name=f"c{i}", index=i)),
                    modality=Modality.RYDBERG,
                    attrs={"basis": measurement_basis, "frontend": "bloqade"},
                ))

    c.module.metadata["frontend"] = "bloqade"
    return c.module


# ---------- duck-type normaliser ----------

def _normalise(prog: Any) -> tuple[BloqadeRegister, list[DriveSegment], str | None]:
    """Coerce a Bloqade-shaped object into our (register, drives, basis) triple."""
    # 1) Native shim
    if isinstance(prog, BloqadeProgram):
        return prog.register, list(prog.drives), prog.measurement_basis

    # 2) Real bloqade duck: prefer .to_segments() if available
    to_seg = getattr(prog, "to_segments", None)
    if callable(to_seg):
        register, drives, basis = to_seg()
        return _coerce_register(register), [_coerce_drive(d) for d in drives], basis

    # 3) Fall back to attribute access
    register = _coerce_register(getattr(prog, "register"))
    raw_drives = list(getattr(prog, "drives", []))
    drives = [_coerce_drive(d) for d in raw_drives]
    basis = getattr(prog, "measurement_basis", "ground-rydberg")
    return register, drives, basis


def _coerce_register(r: Any) -> BloqadeRegister:
    if isinstance(r, BloqadeRegister):
        return r
    # Most "real" bloqade Registers expose `.positions` or are iterable.
    positions = getattr(r, "positions", None)
    if positions is None:
        positions = list(r)
    return BloqadeRegister.from_positions([tuple(p) for p in positions])


def _coerce_drive(d: Any) -> DriveSegment:
    if isinstance(d, DriveSegment):
        return d
    return DriveSegment(
        duration_ns=int(getattr(d, "duration_ns", getattr(d, "duration", 0))),
        amplitude_rad_per_us=float(getattr(d, "amplitude_rad_per_us",
                                           getattr(d, "amplitude", 0.0))),
        detuning_rad_per_us=float(getattr(d, "detuning_rad_per_us",
                                          getattr(d, "detuning", 0.0))),
        phase=float(getattr(d, "phase", 0.0)),
        channel=str(getattr(d, "channel", "rydberg_global")),
        addressing=str(getattr(d, "addressing", "Global")),
    )


__all__ = [
    "from_bloqade",
    "BloqadeProgram",
    "BloqadeRegister",
    "BloqadeAtom",
    "DriveSegment",
]
