"""qmesh.backends.aer_sim — Qiskit Aer wrapper.

Two registered profiles:
- `qmesh.aer`           — ideal Aer state-vector
- `qmesh.aer.noisy`     — Aer with a configurable noise model

Aer is the reference noisy/SV simulator and supports calibration ingestion
via `NoiseModel.from_backend(backend)` when a real backend is available.
We expose a small synthetic Heron-like noise model out of the box so users
can compare noisy vs ideal without an IBM account.
"""

from __future__ import annotations

import time
from typing import Any

from qmesh.backends.base import Backend, Capabilities, ControlLevel, RunResult
from qmesh.backends.registry import register
from qmesh.ir.module import Module
from qmesh.ir.types import Modality


def _ir_to_qiskit(module: Module):
    """Lower a qmesh Module to a Qiskit QuantumCircuit (gate modality only)."""
    from qiskit import QuantumCircuit

    from qmesh.ir.ops import BarrierOp, DelayOp, GateOp, MeasureOp, ResetOp
    from qmesh.ir.types import Bit, Qubit

    n_qubits = 0
    n_bits = 0
    for f in module.functions:
        for op in f.body.ops:
            for v in op.operands:
                if isinstance(v, Qubit):
                    n_qubits = max(n_qubits, v.index + 1)
                elif isinstance(v, Bit):
                    n_bits = max(n_bits, v.index + 1)

    qc = QuantumCircuit(n_qubits, n_bits)
    for f in module.functions:
        for op in f.body.ops:
            qubit_idx = [v.index for v in op.operands if isinstance(v, Qubit)]
            if isinstance(op, MeasureOp):
                bit_idx = next(v.index for v in op.operands if isinstance(v, Bit))
                qc.measure(qubit_idx[0], bit_idx)
            elif isinstance(op, ResetOp):
                qc.reset(qubit_idx[0])
            elif isinstance(op, BarrierOp):
                qc.barrier()
            elif isinstance(op, DelayOp):
                ns = op.params[0] if op.params else 0
                if qubit_idx:
                    for q in qubit_idx:
                        qc.delay(int(ns), q, unit="ns")
                else:
                    for q in range(n_qubits):
                        qc.delay(int(ns), q, unit="ns")
            elif isinstance(op, GateOp):
                fn = getattr(qc, op.name, None)
                if fn is None:
                    raise NotImplementedError(
                        f"Aer backend cannot lower gate '{op.name}'"
                    )
                if op.params:
                    fn(*op.params, *qubit_idx)
                else:
                    fn(*qubit_idx)
    return qc


def _heron_like_noise_model():
    """Small synthetic Heron r2-shaped depolarising noise model.

    Calibration ingestion (`NoiseModel.from_backend`) replaces this when the
    user attaches a real backend. Until then, this lets noisy vs ideal
    comparisons work locally.
    """
    from qiskit_aer.noise import (
        NoiseModel,
        ReadoutError,
        depolarizing_error,
        thermal_relaxation_error,
    )

    nm = NoiseModel()

    # Single-qubit gate error: ~3e-4 depolarising
    nm.add_all_qubit_quantum_error(depolarizing_error(3e-4, 1),
                                   ["x", "y", "z", "h", "s", "sdg", "t", "tdg",
                                    "rx", "ry", "rz", "p"])
    # Two-qubit gate error: ~3e-3 depolarising (Heron r2 typical)
    nm.add_all_qubit_quantum_error(depolarizing_error(3e-3, 2),
                                   ["cx", "cz", "swap", "ecr"])

    # Thermal relaxation: T1=200µs, T2=120µs, gate time 60ns/600ns
    t1, t2 = 200e3, 120e3  # ns
    nm.add_all_qubit_quantum_error(thermal_relaxation_error(t1, t2, 60),
                                   ["x", "y", "z", "h", "s", "sdg", "t", "tdg",
                                    "rx", "ry", "rz", "p"])
    # 2-qubit thermal relaxation = single-qubit channel ⊗ single-qubit channel
    tr_2q = thermal_relaxation_error(t1, t2, 600).tensor(
        thermal_relaxation_error(t1, t2, 600)
    )
    nm.add_all_qubit_quantum_error(tr_2q, ["cx", "cz", "swap", "ecr"])

    # Readout error: 1.5% asymmetric (Heron typical)
    ro = ReadoutError([[0.985, 0.015], [0.020, 0.980]])
    nm.add_all_qubit_readout_error(ro)
    return nm


class AerSimulator(Backend):
    def __init__(self, *, noisy: bool = False, name: str | None = None) -> None:
        self._noisy = noisy
        self._name = name or ("qmesh.aer.noisy" if noisy else "qmesh.aer")

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self._name,
            vendor="qmesh+qiskit-aer",
            modalities={Modality.GATE},
            qubit_count=29,  # comfortable on 80GB RAM, c128
            native_gates={"h", "x", "y", "z", "s", "sdg", "t", "tdg",
                          "rx", "ry", "rz", "cx", "cz", "swap", "ecr",
                          "ccx", "cswap"},
            measurement_feedforward=True,  # Aer supports if_test in dynamic mode
            classical_control=ControlLevel.IF,
            cost_per_shot_usd=0.0,
            queue_depth=0,
            fidelity_2q_typical=0.997 if self._noisy else 1.0,
            is_simulator=True,
            notes=("Aer with synthetic Heron-like noise model"
                   if self._noisy else "ideal Aer state-vector"),
        )

    def run(self, module: Module, shots: int = 1024, **kwargs: Any) -> RunResult:
        from qiskit_aer import AerSimulator as _Aer

        ok, why = self.accepts(module)
        if not ok:
            raise ValueError(f"aer rejected module: {why}")

        qc = _ir_to_qiskit(module)
        if self._noisy:
            sim = _Aer(noise_model=_heron_like_noise_model())
        else:
            sim = _Aer()

        t0 = time.time()
        result = sim.run(qc, shots=shots).result()
        wall = time.time() - t0

        try:
            counts_raw = result.get_counts()
        except Exception:
            counts_raw = {}

        # Aer returns space-separated multi-register strings. Normalise to a
        # single bitstring (MSB first).
        counts: dict[str, int] = {}
        for k, v in counts_raw.items():
            key = k.replace(" ", "")
            counts[key] = counts.get(key, 0) + v

        return RunResult(
            counts=counts,
            shots=shots,
            wall_seconds=wall,
            qpu_seconds=wall,
            cost_usd=0.0,
            backend_metadata={"noisy": self._noisy, "n_qubits": qc.num_qubits},
        )


# self-register
register(AerSimulator(noisy=False))
register(AerSimulator(noisy=True))
