"""qmesh.ftmode.estimate — resource estimation.

Estimates physical-qubit count, T-state count, syndrome cycles, and wall
time for a logical workload. Output is shaped to be Microsoft Azure
Resource Estimator-compatible so users can cross-check.

Phase-2α model: rule-based, parameterised by code distance, gate counts,
and a small set of architecture constants. Phase-2β work: replace the
constants with calibration-driven estimates per backend.

Reference numbers used for sanity (May 2026):
- Gidney 2025 (arXiv 2505.15917): RSA-2048 in <1M physical qubits, <1 week
  with magic-state cultivation.
- Surface-code physical = (d² + (d² - 1)) per logical qubit (rotated).
- Cultivation overhead: Gidney 2024 (arXiv 2409.17595) — ~1 cultivation
  patch per logical T at 4×10⁻¹¹ logical error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from qmesh.ftmode.codes import Code
from qmesh.ir.module import Module
from qmesh.ir.ops import GateOp


@dataclass(slots=True)
class ResourceEstimate:
    """Microsoft-Resource-Estimator-shaped output."""

    physical_qubits: int
    logical_qubits: int
    code_distance: int
    code_name: str
    T_states_required: int
    cycles: int                                   # total syndrome rounds
    wall_seconds: float                            # estimated, not measured
    physical_error_rate: float
    target_logical_error_rate: float
    estimated_logical_error_rate: float
    breakdown: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "physical_qubits": self.physical_qubits,
            "logical_qubits": self.logical_qubits,
            "code": {
                "name": self.code_name,
                "distance": self.code_distance,
            },
            "T_states_required": self.T_states_required,
            "cycles": self.cycles,
            "wall_seconds": self.wall_seconds,
            "physical_error_rate": self.physical_error_rate,
            "target_logical_error_rate": self.target_logical_error_rate,
            "estimated_logical_error_rate": self.estimated_logical_error_rate,
            "breakdown": self.breakdown,
        }


# Architecture constants (tuned to give the right shape; not a calibration)
_SYNDROME_CYCLE_TIME_NS = 1_000.0      # 1 µs per round (typical superconducting)
_T_GATE_CULTIVATION_CYCLES = 50         # Gidney cultivation cost (very rough)
_T_GATE_OVERHEAD_PHYSICAL = 100         # extra qubits for cultivation pool


def _count_t_gates(module: Module) -> int:
    n = 0
    for f in module.functions:
        for op in f.body.ops:
            if isinstance(op, GateOp) and op.name in ("t", "tdg"):
                n += 1
    return n


def _logical_error_per_round_below_threshold(d: int, p: float, p_th: float) -> float:
    """Heuristic logical error per round, surface code, p < p_th.

    Standard exponential-suppression model:
        p_L ≈ A * (p / p_th) ** ((d + 1) / 2)
    with A = 0.1 (typical surface-code prefactor).
    """
    if p >= p_th:
        return 1.0  # at or above threshold there's no suppression
    return 0.1 * (p / p_th) ** ((d + 1) / 2)


def estimate(
    module: Module,
    *,
    code: Code,
    physical_error_rate: float = 1e-3,
    target_logical_error_rate: float = 1e-9,
    surface_code_threshold: float = 1e-2,
) -> ResourceEstimate:
    """Estimate resources for executing `module` under FT mode using `code`."""
    md = code.metadata()
    n_logical = max(md.logical_qubits, 1)

    t_gates = _count_t_gates(module)
    cycles = md.rounds + t_gates * _T_GATE_CULTIVATION_CYCLES
    p_per_round = _logical_error_per_round_below_threshold(
        md.distance, physical_error_rate, surface_code_threshold
    )
    estimated = 1 - (1 - p_per_round) ** cycles

    overhead_T = t_gates * _T_GATE_OVERHEAD_PHYSICAL
    physical = md.physical_qubits * n_logical + overhead_T
    wall = cycles * _SYNDROME_CYCLE_TIME_NS / 1e9  # seconds

    return ResourceEstimate(
        physical_qubits=physical,
        logical_qubits=n_logical,
        code_distance=md.distance,
        code_name=md.name,
        T_states_required=t_gates,
        cycles=cycles,
        wall_seconds=wall,
        physical_error_rate=physical_error_rate,
        target_logical_error_rate=target_logical_error_rate,
        estimated_logical_error_rate=estimated,
        breakdown={
            "t_gate_cultivation_cycles_each": _T_GATE_CULTIVATION_CYCLES,
            "t_gate_physical_overhead_each": _T_GATE_OVERHEAD_PHYSICAL,
            "syndrome_cycle_time_ns": _SYNDROME_CYCLE_TIME_NS,
            "p_per_round": p_per_round,
            "code_metadata": md.to_dict(),
            "below_threshold": physical_error_rate < surface_code_threshold,
        },
    )


def estimate_for_distances(
    module: Module,
    *,
    distances: Iterable[int] = (3, 5, 7, 9, 11),
    rounds: int = 8,
    physical_error_rate: float = 1e-3,
) -> list[ResourceEstimate]:
    """Sweep across distances; useful for choosing the smallest d that hits target."""
    from qmesh.ftmode.codes import SurfaceCode
    out = []
    for d in distances:
        out.append(estimate(
            module,
            code=SurfaceCode(distance=d, rounds=rounds),
            physical_error_rate=physical_error_rate,
        ))
    return out
