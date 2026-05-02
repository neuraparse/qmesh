"""qmesh.ir.types — typed value & modality system.

Modalities are first-class. A `RydbergOp` is not a fake `GateOp`; backends that
don't speak Rydberg refuse it at lower-time, not at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


class Modality(str, Enum):
    GATE = "gate"
    RYDBERG = "rydberg"
    CV = "cv"
    PULSE = "pulse"
    CLASSICAL = "classical"


@dataclass(frozen=True, slots=True)
class QType:
    """Base for typed IR values."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Qubit(QType):
    """Discrete two-level qubit (gate modality)."""

    index: int

    def __str__(self) -> str:
        return f"q{self.index}"


@dataclass(frozen=True, slots=True)
class Qumode(QType):
    """Continuous-variable qumode (photonic CV modality)."""

    index: int
    cutoff: int = 10

    def __str__(self) -> str:
        return f"m{self.index}[{self.cutoff}]"


@dataclass(frozen=True, slots=True)
class Atom(QType):
    """Neutral-atom site (Rydberg modality). Position can be set per-backend."""

    index: int
    position: tuple[float, float, float] | None = None  # μm

    def __str__(self) -> str:
        return f"a{self.index}"


@dataclass(frozen=True, slots=True)
class Bit(QType):
    """Single classical bit (output of measurements)."""

    index: int

    def __str__(self) -> str:
        return f"c{self.index}"


@dataclass(frozen=True, slots=True)
class ClassicalInt(QType):
    """Classical integer, runtime-mutable."""

    name: str
    width: int = 32


@dataclass(frozen=True, slots=True)
class ClassicalFloat(QType):
    """Classical float."""

    name: str


T = TypeVar("T", bound=QType)
U = TypeVar("U", bound=QType)


@dataclass(frozen=True, slots=True)
class Channel(QType, Generic[T, U]):
    """A typed cross-modality channel.

    Carries entanglement / state from one modality to another. Used to model
    photonic interconnects (IonQ April 2026, Cisco Universal Quantum Switch),
    photon-mediated transfer between modules, etc.
    """

    name: str
    src: Modality
    dst: Modality

    def __str__(self) -> str:
        return f"{self.name}: {self.src.value}→{self.dst.value}"
