"""qmesh.backends.statevec_sim — minimal NumPy state-vector simulator.

Reference implementation that runs without any external SDK so the scaffold
"hello-world" works on a fresh box. Real deployments swap in CUDA-Q,
Qiskit Aer, or vendor backends.

Supports the canonical 1Q/2Q gate set + measurement. Modality.GATE only.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from qmesh.backends.base import Backend, Capabilities, ControlLevel, RunResult
from qmesh.backends.registry import register
from qmesh.ir.module import Module
from qmesh.ir.ops import GateOp, MeasureOp, ResetOp
from qmesh.ir.types import Modality, Qubit


def _kron_chain(matrices: list[np.ndarray]) -> np.ndarray:
    out = matrices[0]
    for m in matrices[1:]:
        out = np.kron(out, m)
    return out


_I = np.eye(2, dtype=np.complex128)
_H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
                    dtype=np.complex128)


_1Q_GATES: dict[str, np.ndarray] = {
    "h": _H, "x": _X, "y": _Y, "z": _Z, "s": _S, "t": _T, "id": _I,
}


class StateVectorSimulator(Backend):
    """Tiny CPU state-vector simulator. Up to ~12 qubits comfortably."""

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="qmesh.statevec",
            vendor="qmesh",
            modalities={Modality.GATE},
            qubit_count=20,
            native_gates={"h", "x", "y", "z", "s", "t", "rx", "ry", "rz",
                          "cx", "cz", "swap", "cp", "p"},
            measurement_feedforward=False,
            classical_control=ControlLevel.NONE,
            cost_per_shot_usd=0.0,
            queue_depth=0,
            fidelity_2q_typical=1.0,
            is_simulator=True,
            notes="reference NumPy simulator — used by the hello-world scaffold",
        )

    def run(self, module: Module, shots: int = 1024, **_: Any) -> RunResult:
        ok, why = self.accepts(module)
        if not ok:
            raise ValueError(f"simulator rejected module: {why}")

        t0 = time.time()
        # Find unique qubit indices
        qubits: set[int] = set()
        bits: set[int] = set()
        for f in module.functions:
            for op in f.body.ops:
                for v in op.operands:
                    if isinstance(v, Qubit):
                        qubits.add(v.index)
                if isinstance(op, MeasureOp):
                    # second operand is Bit
                    bits.add(op.operands[1].index)
        n = max(qubits) + 1 if qubits else 0
        if n == 0:
            return RunResult(counts={}, shots=shots, wall_seconds=time.time() - t0)
        if n > self.capabilities.qubit_count:
            raise ValueError(f"too many qubits ({n} > {self.capabilities.qubit_count})")

        # Apply unitary chain to the |0…0> state
        psi = np.zeros(2**n, dtype=np.complex128)
        psi[0] = 1.0
        meas_pairs: list[tuple[int, int]] = []  # (qubit_idx, bit_idx)

        for f in module.functions:
            for op in f.body.ops:
                if isinstance(op, MeasureOp):
                    qi = op.operands[0].index
                    bi = op.operands[1].index
                    meas_pairs.append((qi, bi))
                elif isinstance(op, ResetOp):
                    qi = op.operands[0].index
                    psi = self._reset(psi, qi, n)
                elif isinstance(op, GateOp):
                    psi = self._apply_gate(psi, op, n)
                # ignore barriers / delays for ideal simulation

        # Sample shots
        probs = np.abs(psi) ** 2
        probs /= probs.sum()
        rng = np.random.default_rng()
        samples = rng.choice(2**n, size=shots, p=probs)

        n_bits = max(bits) + 1 if bits else 0
        counts: dict[str, int] = {}
        for s in samples:
            bitstring = ["0"] * n_bits
            for qi, bi in meas_pairs:
                bit = (int(s) >> qi) & 1
                bitstring[bi] = str(bit)
            key = "".join(reversed(bitstring))  # MSB-first like Qiskit
            counts[key] = counts.get(key, 0) + 1
        return RunResult(
            counts=counts,
            shots=shots,
            wall_seconds=time.time() - t0,
            qpu_seconds=time.time() - t0,
            cost_usd=0.0,
            backend_metadata={"n_qubits": n, "n_bits": n_bits},
        )

    # ---- gate kernels ----

    def _apply_gate(self, psi: np.ndarray, op: GateOp, n: int) -> np.ndarray:
        name = op.name
        ops_list = op.operands
        params = op.params
        if name in _1Q_GATES:
            return self._apply_1q(psi, _1Q_GATES[name], ops_list[0].index, n)
        if name == "rx":
            return self._apply_1q(psi, _rx(params[0]), ops_list[0].index, n)
        if name == "ry":
            return self._apply_1q(psi, _ry(params[0]), ops_list[0].index, n)
        if name == "rz":
            return self._apply_1q(psi, _rz(params[0]), ops_list[0].index, n)
        if name == "cx":
            return self._apply_cx(psi, ops_list[0].index, ops_list[1].index, n)
        if name == "cz":
            return self._apply_cz(psi, ops_list[0].index, ops_list[1].index, n)
        if name == "swap":
            return self._apply_swap(psi, ops_list[0].index, ops_list[1].index, n)
        if name == "cp":
            theta = params[0]
            return self._apply_cp(psi, theta, ops_list[0].index, ops_list[1].index, n)
        if name == "p":
            theta = params[0]
            return self._apply_1q(psi, np.array([[1, 0], [0, np.exp(1j * theta)]],
                                                dtype=np.complex128), ops_list[0].index, n)
        raise NotImplementedError(f"gate '{name}' not implemented in qmesh.statevec")

    @staticmethod
    def _apply_1q(psi: np.ndarray, mat: np.ndarray, q: int, n: int) -> np.ndarray:
        psi = psi.reshape([2] * n)
        psi = np.tensordot(mat, psi, axes=([1], [n - 1 - q]))
        psi = np.moveaxis(psi, 0, n - 1 - q)
        return psi.reshape(-1)

    @staticmethod
    def _apply_cx(psi: np.ndarray, c: int, t: int, n: int) -> np.ndarray:
        out = psi.copy()
        for i in range(2**n):
            if (i >> c) & 1:
                j = i ^ (1 << t)
                if j > i:
                    out[i], out[j] = psi[j], psi[i]
        return out

    @staticmethod
    def _apply_cz(psi: np.ndarray, c: int, t: int, n: int) -> np.ndarray:
        out = psi.copy()
        for i in range(2**n):
            if ((i >> c) & 1) and ((i >> t) & 1):
                out[i] = -out[i]
        return out

    @staticmethod
    def _apply_cp(psi: np.ndarray, theta: float, c: int, t: int, n: int) -> np.ndarray:
        out = psi.copy()
        phase = np.exp(1j * theta)
        for i in range(2**n):
            if ((i >> c) & 1) and ((i >> t) & 1):
                out[i] = out[i] * phase
        return out

    @staticmethod
    def _apply_swap(psi: np.ndarray, a: int, b: int, n: int) -> np.ndarray:
        if a == b:
            return psi
        out = psi.copy()
        for i in range(2**n):
            ba = (i >> a) & 1
            bb = (i >> b) & 1
            if ba != bb:
                j = i ^ (1 << a) ^ (1 << b)
                if j > i:
                    out[i], out[j] = psi[j], psi[i]
        return out

    @staticmethod
    def _reset(psi: np.ndarray, q: int, n: int) -> np.ndarray:
        # collapse to |0> on qubit q (deterministic ideal reset)
        out = np.zeros_like(psi)
        for i in range(2**n):
            if not ((i >> q) & 1):
                out[i] = psi[i]
        norm = np.linalg.norm(out)
        if norm > 0:
            out /= norm
        else:
            out[0] = 1.0
        return out


# self-register on import
register(StateVectorSimulator())
