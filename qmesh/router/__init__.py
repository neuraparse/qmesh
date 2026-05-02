"""qmesh.router — backend selection across (cost, fidelity, queue, MCM-latency).

Phase 1: real scoring across registered backends. Live calibration ingestion
is wired through `Backend.capabilities` (`fidelity_2q_typical`, `queue_depth`,
`feedforward_latency_ns`). Per-circuit estimated fidelity uses a simple
2Q-gate-count × per-gate-fidelity model — accurate enough to rank backends.

The scoring algorithm is deliberately small and explainable; it lives in pure
Python so users can inspect why a backend was chosen. The richer MQT-Predictor
ML model can later replace this scorer behind the same interface.

Phase 2 will add: live calibration URL fetch, queue-depth polling,
mid-circuit-measurement weight, native-gate-match bonus, and a learned cost
model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

from qmesh.backends.base import Backend
from qmesh.backends.registry import all_backends
from qmesh.ir.module import Module
from qmesh.ir.ops import GateOp, MeasureOp


@dataclass(slots=True)
class Objective:
    mode: str = "max_fidelity_under_budget"   # | "min_cost_above_fidelity"
    budget_usd: float = 1.0
    min_fidelity: float = 0.0
    max_wall_seconds: float = 600.0
    require_simulator: bool = False
    require_hardware: bool = False
    require_modalities: set[str] | None = None  # subset of {gate,rydberg,cv,pulse}
    weights: dict[str, float] = field(default_factory=lambda: {
        "fidelity": 1.0,
        "cost": 0.5,
        "queue": 0.2,
        "mcm_penalty": 0.3,
        "native_match": 0.2,
    })


@dataclass(slots=True)
class CircuitProfile:
    n_qubits: int
    depth: int
    one_q_gates: int
    two_q_gates: int
    has_mcm: bool
    gate_set: set[str]


def profile(module: Module) -> CircuitProfile:
    """Cheap structural profile of a circuit, used by the scorer."""
    qubits: set[int] = set()
    one_q = two_q = 0
    has_mcm = False
    gate_set: set[str] = set()
    seen_measure = False

    for f in module.functions:
        for op in f.body.ops:
            for v in op.operands:
                from qmesh.ir.types import Qubit
                if isinstance(v, Qubit):
                    qubits.add(v.index)
            if isinstance(op, MeasureOp):
                seen_measure = True
            elif isinstance(op, GateOp):
                gate_set.add(op.name)
                # mid-circuit measurement: a Gate that comes after a Measure
                if seen_measure:
                    has_mcm = True
                from qmesh.ir.types import Qubit
                qs = [v for v in op.operands if isinstance(v, Qubit)]
                if len(qs) == 1:
                    one_q += 1
                elif len(qs) >= 2:
                    two_q += 1
    return CircuitProfile(
        n_qubits=len(qubits),
        depth=one_q + two_q,
        one_q_gates=one_q,
        two_q_gates=two_q,
        has_mcm=has_mcm,
        gate_set=gate_set,
    )


def score(backend: Backend, prof: CircuitProfile, objective: Objective,
          shots: int = 1024) -> tuple[float, dict[str, float]]:
    """Score a backend for this circuit. Higher = better. Returns (score, breakdown)."""
    cap = backend.capabilities

    # Estimate fidelity: each 2Q gate contributes (fid_2q − 1); 1Q ~10× better
    f_2q = cap.fidelity_2q_typical
    f_1q = 1 - (1 - f_2q) / 10
    estimated_fidelity = (f_1q ** prof.one_q_gates) * (f_2q ** prof.two_q_gates)

    cost = cap.cost_per_shot_usd or 0.0
    total_cost = cost * shots
    cost_score = 1 - min(1.0, total_cost / max(objective.budget_usd, 1e-9))

    # Queue: simulators always 0, hardware should publish queue depth
    queue = cap.queue_depth or 0
    queue_score = 1 / (1 + queue)

    mcm_penalty = 0.0
    if prof.has_mcm and not cap.measurement_feedforward:
        mcm_penalty = 1.0  # cannot run; will be filtered
    elif prof.has_mcm and cap.feedforward_latency_ns:
        # 1µs ~ acceptable; 100µs ~ painful
        mcm_penalty = min(1.0, cap.feedforward_latency_ns / 100_000)

    native_match = len(prof.gate_set & cap.native_gates) / max(len(prof.gate_set), 1)

    w = objective.weights
    total = (
        w["fidelity"] * estimated_fidelity
        + w["cost"] * cost_score
        + w["queue"] * queue_score
        + w["native_match"] * native_match
        - w["mcm_penalty"] * mcm_penalty
    )
    return total, {
        "estimated_fidelity": estimated_fidelity,
        "cost_score": cost_score,
        "queue_score": queue_score,
        "native_match": native_match,
        "mcm_penalty": mcm_penalty,
        "total": total,
    }


def choose(module: Module, objective: Objective | None = None,
           shots: int = 1024,
           candidates: Iterable[Backend] | None = None) -> tuple[Backend, dict]:
    """Pick the highest-scoring backend that accepts the module.

    Returns (backend, breakdown). The breakdown is also embedded in the
    provenance manifest so the user can audit *why* this backend was chosen.
    """
    objective = objective or Objective()
    prof = profile(module)
    candidates = list(candidates) if candidates else list(all_backends().values())
    rows: list[tuple[float, Backend, dict]] = []
    rejected: list[tuple[Backend, str]] = []
    for bk in candidates:
        cap = bk.capabilities
        if objective.require_simulator and not cap.is_simulator:
            rejected.append((bk, "not a simulator"))
            continue
        if objective.require_hardware and cap.is_simulator:
            rejected.append((bk, "is a simulator"))
            continue
        ok, why = bk.accepts(module)
        if not ok:
            rejected.append((bk, why))
            continue
        if cap.qubit_count < prof.n_qubits:
            rejected.append((bk, f"qubit_count {cap.qubit_count} < {prof.n_qubits}"))
            continue
        s, breakdown = score(bk, prof, objective, shots=shots)
        if breakdown["estimated_fidelity"] < objective.min_fidelity:
            rejected.append((bk, f"fid {breakdown['estimated_fidelity']:.4f} < min"))
            continue
        if breakdown["mcm_penalty"] >= 1.0:
            rejected.append((bk, "circuit needs MCM, backend doesn't support"))
            continue
        rows.append((s, bk, breakdown))

    if not rows:
        raise RuntimeError(
            "no backend matched objective. Rejected:\n  "
            + "\n  ".join(f"{b.capabilities.name}: {why}" for b, why in rejected)
        )
    rows.sort(key=lambda t: -t[0])
    best_score, best_bk, best_breakdown = rows[0]
    return best_bk, {
        "chosen": best_bk.capabilities.name,
        "score": best_score,
        "breakdown": best_breakdown,
        "all_scored": [(b.capabilities.name, s) for s, b, _ in rows],
        "rejected": [(b.capabilities.name, why) for b, why in rejected],
        "circuit_profile": asdict(prof),
    }
