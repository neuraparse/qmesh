"""qmesh.ir.builder — friendly Python DSL for emitting IR.

Authoring API:

    from qmesh import circuit
    with circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0)
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)
    module = c.module
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from qmesh.ir.module import Function, Module, Region
from qmesh.ir.ops import BarrierOp, CVOp, DelayOp, GateOp, MeasureOp, ResetOp, RydbergOp
from qmesh.ir.types import Atom, Bit, Modality, Qubit, Qumode


class CircuitBuilder:
    """Modality-aware circuit builder.

    Gate ops use canonical names (h, x, y, z, s, t, cx, cz, swap, rx, ry, rz,
    ccx, ...). Names that don't match a canonical entry pass through verbatim
    so backends can accept native gates.
    """

    GATE_NAMES_1Q = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "id"}
    GATE_NAMES_2Q = {"cx", "cz", "cy", "swap", "iswap", "ecr"}
    GATE_NAMES_3Q = {"ccx", "cswap"}
    GATE_NAMES_PARAM_1Q = {"rx", "ry", "rz", "p", "u3"}
    GATE_NAMES_PARAM_2Q = {"rxx", "ryy", "rzz", "cp", "crx", "cry", "crz"}

    def __init__(self, name: str, n_qubits: int = 0, n_bits: int = 0,
                 n_qumodes: int = 0, n_atoms: int = 0) -> None:
        self.name = name
        self.qubits = tuple(Qubit(name=f"q{i}", index=i) for i in range(n_qubits))
        self.bits = tuple(Bit(name=f"c{i}", index=i) for i in range(n_bits))
        self.qumodes = tuple(Qumode(name=f"m{i}", index=i) for i in range(n_qumodes))
        self.atoms = tuple(Atom(name=f"a{i}", index=i) for i in range(n_atoms))
        self._region = Region(label=name)
        self.module = Module()
        self.module.add(Function(
            name=name,
            inputs=self.qubits + self.qumodes + self.atoms,
            outputs=self.bits,
            body=self._region,
        ))

    # ---------- gate-modality methods ----------

    def _gate(self, name: str, qubits: list[int], params: tuple[float, ...] = ()) -> None:
        operands = tuple(self.qubits[i] for i in qubits)
        self._region.append(GateOp(name=name, operands=operands, params=params))

    def h(self, q: int) -> None: self._gate("h", [q])
    def x(self, q: int) -> None: self._gate("x", [q])
    def y(self, q: int) -> None: self._gate("y", [q])
    def z(self, q: int) -> None: self._gate("z", [q])
    def s(self, q: int) -> None: self._gate("s", [q])
    def t(self, q: int) -> None: self._gate("t", [q])
    def rx(self, theta: float, q: int) -> None: self._gate("rx", [q], (theta,))
    def ry(self, theta: float, q: int) -> None: self._gate("ry", [q], (theta,))
    def rz(self, theta: float, q: int) -> None: self._gate("rz", [q], (theta,))

    def cx(self, ctrl: int, tgt: int) -> None: self._gate("cx", [ctrl, tgt])
    def cz(self, ctrl: int, tgt: int) -> None: self._gate("cz", [ctrl, tgt])
    def swap(self, a: int, b: int) -> None: self._gate("swap", [a, b])
    def ccx(self, c1: int, c2: int, tgt: int) -> None: self._gate("ccx", [c1, c2, tgt])
    def rzz(self, theta: float, a: int, b: int) -> None: self._gate("rzz", [a, b], (theta,))

    # native escape hatch: c.gate("ms", [0,1], theta=0.5)
    def gate(self, name: str, qubits: list[int], **params: float) -> None:
        self._gate(name, qubits, tuple(params.values()))

    # ---------- measurement / reset / barriers ----------

    def measure(self, q: int, c: int) -> None:
        self._region.append(MeasureOp(
            operands=(self.qubits[q], self.bits[c]),
            modality=Modality.GATE,
        ))

    def reset(self, q: int) -> None:
        self._region.append(ResetOp(operands=(self.qubits[q],), modality=Modality.GATE))

    def barrier(self) -> None:
        self._region.append(BarrierOp(operands=self.qubits, modality=Modality.GATE))

    def delay(self, ns: float, q: int | None = None) -> None:
        ops = (self.qubits[q],) if q is not None else self.qubits
        self._region.append(DelayOp(operands=ops, params=(ns,), modality=Modality.GATE))

    # ---------- rydberg / CV escape hatches ----------

    def rydberg(self, name: str, atom_indices: list[int], **params: float) -> None:
        self._region.append(RydbergOp(
            name=name,
            operands=tuple(self.atoms[i] for i in atom_indices),
            params=tuple(params.values()),
        ))

    def cv(self, name: str, mode_indices: list[int], **params: float) -> None:
        self._region.append(CVOp(
            name=name,
            operands=tuple(self.qumodes[i] for i in mode_indices),
            params=tuple(params.values()),
        ))


@contextmanager
def circuit(name: str, n_qubits: int = 0, n_bits: int = 0,
            n_qumodes: int = 0, n_atoms: int = 0) -> Iterator[CircuitBuilder]:
    builder = CircuitBuilder(name, n_qubits, n_bits, n_qumodes, n_atoms)
    yield builder


__all__ = ["circuit", "CircuitBuilder"]
