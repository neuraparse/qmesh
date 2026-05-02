"""qmesh.backends.openpulse_sim — pulse-level simulator skeleton (Phase 3δ).

Many real superconducting hardware vendors (IBM, Rigetti, OQC, IQM) accept
*pulse-level* schedules in addition to gate-level circuits. qmesh's gate IR
already records gate-level operations; this backend lowers a gate program to
an OpenPulse-shaped schedule and returns the schedule plus per-shot counts
*if* a pulse simulator is installed.

Operating modes:

  * **β** when ``qiskit.pulse`` is importable (Qiskit 1.x with the optional
    ``qiskit-experiments`` / ``qiskit-dynamics`` install). The backend builds
    a Schedule, optionally simulates with qiskit-dynamics, and returns
    counts.

  * **α-degraded** when ``qiskit.pulse`` is unavailable (the common case on
    Qiskit ≥ 2.x, where pulse was removed). The backend still runs: it
    produces a *schedule descriptor* (a dict of pulse durations and
    amplitudes per gate) and synthesises counts via a fast-path Aer
    fallback so the Module-level smoke tests still pass. The descriptor is
    stamped into ``RunResult.backend_metadata['schedule_descriptor']`` so
    downstream tooling / hardware vendors that ingest dict-shaped pulse
    schedules can use it directly.

α-honest scope:
  * Gate-to-pulse calibration is a single-channel Gaussian envelope per gate
    with deterministic durations (40 ns single-qubit, 200 ns two-qubit) and
    amplitudes drawn from a small lookup. This is enough to get auditors a
    schedule that is *shape-checkable*; it is **not** a calibrated DRAG +
    cross-resonance schedule. Real calibration is vendor-specific and
    requires hardware bring-up data.
"""

from __future__ import annotations

import time
from typing import Any

from qmesh.backends.base import Backend, Capabilities, ControlLevel, RunResult
from qmesh.backends.registry import register
from qmesh.ir.module import Module
from qmesh.ir.types import Modality

# Default pulse calibrations (ns durations, [0..1] amplitudes).
_DEFAULT_PULSE_CALIBRATION: dict[str, dict[str, float]] = {
    "h":   {"duration_ns": 40.0,  "amplitude": 0.30,  "sigma_ns": 8.0},
    "x":   {"duration_ns": 40.0,  "amplitude": 0.30,  "sigma_ns": 8.0},
    "y":   {"duration_ns": 40.0,  "amplitude": 0.30,  "sigma_ns": 8.0},
    "z":   {"duration_ns": 0.0,   "amplitude": 0.0,   "sigma_ns": 0.0},   # virtual Z
    "s":   {"duration_ns": 0.0,   "amplitude": 0.0,   "sigma_ns": 0.0},
    "sdg": {"duration_ns": 0.0,   "amplitude": 0.0,   "sigma_ns": 0.0},
    "t":   {"duration_ns": 0.0,   "amplitude": 0.0,   "sigma_ns": 0.0},
    "tdg": {"duration_ns": 0.0,   "amplitude": 0.0,   "sigma_ns": 0.0},
    "rx":  {"duration_ns": 40.0,  "amplitude": 0.30,  "sigma_ns": 8.0},
    "ry":  {"duration_ns": 40.0,  "amplitude": 0.30,  "sigma_ns": 8.0},
    "rz":  {"duration_ns": 0.0,   "amplitude": 0.0,   "sigma_ns": 0.0},
    "cx":  {"duration_ns": 200.0, "amplitude": 0.40,  "sigma_ns": 32.0},
    "cz":  {"duration_ns": 200.0, "amplitude": 0.40,  "sigma_ns": 32.0},
    "swap": {"duration_ns": 600.0, "amplitude": 0.40, "sigma_ns": 32.0},
    "ecr": {"duration_ns": 200.0, "amplitude": 0.45,  "sigma_ns": 32.0},
}


def _pulse_module_available() -> bool:
    """Return True iff `qiskit.pulse` is importable (Qiskit 1.x)."""
    try:
        import qiskit.pulse  # noqa: F401
        return True
    except ImportError:
        return False


def _build_schedule_descriptor(module: Module) -> dict[str, Any]:
    """Produce a deterministic pulse-schedule descriptor from the IR.

    This is a *vendor-neutral* shape: list of timestamped pulse events with
    durations, amplitudes, sigma, and qubit assignments. It is what we
    persist to the manifest in the α-degraded path; β code can consume it
    to drive qiskit.pulse.Schedule construction or vendor-specific
    schedule formats.
    """
    from qmesh.ir.ops import GateOp, MeasureOp, ResetOp
    from qmesh.ir.types import Qubit

    events: list[dict[str, Any]] = []
    cursor_ns_per_q: dict[int, float] = {}
    total_ns = 0.0
    qubits_used: set[int] = set()

    for f in module.functions:
        for op in f.body.ops:
            qubit_idx = [v.index for v in op.operands if isinstance(v, Qubit)]
            for q in qubit_idx:
                qubits_used.add(q)
            if isinstance(op, GateOp):
                cal = _DEFAULT_PULSE_CALIBRATION.get(op.name)
                if cal is None:
                    cal = {"duration_ns": 100.0, "amplitude": 0.30, "sigma_ns": 16.0}
                start = max((cursor_ns_per_q.get(q, 0.0) for q in qubit_idx),
                            default=0.0)
                end = start + cal["duration_ns"]
                for q in qubit_idx:
                    cursor_ns_per_q[q] = end
                total_ns = max(total_ns, end)
                events.append({
                    "kind": "gate_pulse",
                    "name": op.name,
                    "qubits": list(qubit_idx),
                    "params": list(op.params),
                    "start_ns": start,
                    "duration_ns": cal["duration_ns"],
                    "amplitude": cal["amplitude"],
                    "sigma_ns": cal["sigma_ns"],
                    "is_virtual_z": cal["duration_ns"] == 0.0,
                })
            elif isinstance(op, MeasureOp):
                start = cursor_ns_per_q.get(qubit_idx[0], 0.0)
                end = start + 1500.0   # ~1.5 µs typical readout
                cursor_ns_per_q[qubit_idx[0]] = end
                total_ns = max(total_ns, end)
                events.append({
                    "kind": "measure",
                    "qubits": list(qubit_idx),
                    "start_ns": start,
                    "duration_ns": 1500.0,
                })
            elif isinstance(op, ResetOp):
                start = cursor_ns_per_q.get(qubit_idx[0], 0.0)
                end = start + 800.0
                cursor_ns_per_q[qubit_idx[0]] = end
                total_ns = max(total_ns, end)
                events.append({
                    "kind": "reset",
                    "qubits": list(qubit_idx),
                    "start_ns": start,
                    "duration_ns": 800.0,
                })

    return {
        "schedule_format": "qmesh.openpulse.v0",
        "events": events,
        "total_duration_ns": total_ns,
        "qubits_used": sorted(qubits_used),
        "n_events": len(events),
    }


def _build_qiskit_pulse_schedule(module: Module):
    """Construct a `qiskit.pulse.Schedule` from the IR. β path.

    Requires `qiskit.pulse`. Used only when ``_pulse_module_available()``.
    """
    import qiskit.pulse as pulse  # type: ignore[import-not-found]

    desc = _build_schedule_descriptor(module)
    schedule = pulse.Schedule()
    for ev in desc["events"]:
        if ev["kind"] != "gate_pulse" or ev.get("is_virtual_z"):
            continue
        for q in ev["qubits"]:
            chan = pulse.DriveChannel(q)
            gauss = pulse.Gaussian(
                duration=int(ev["duration_ns"]),
                amp=ev["amplitude"],
                sigma=max(ev["sigma_ns"], 1.0),
            )
            schedule = schedule.insert(int(ev["start_ns"]), pulse.Play(gauss, chan))
    return schedule


class OpenPulseSimulator(Backend):
    """Pulse-level backend skeleton.

    Configuration:
      * ``name`` — registry name, default ``"qmesh.openpulse"``.
      * ``fallback`` — when ``qiskit.pulse`` is missing, route gate-level
        execution through ``qmesh.aer`` (when available) or
        ``qmesh.statevec``. Default True.
    """

    def __init__(
        self,
        *,
        name: str = "qmesh.openpulse",
        fallback: bool = True,
    ) -> None:
        self._name = name
        self._fallback = fallback
        self._has_pulse = _pulse_module_available()

    @property
    def capabilities(self) -> Capabilities:
        notes = (
            "OpenPulse-shaped schedules (β when qiskit.pulse importable; "
            "α-fallback to gate sim otherwise). "
            f"qiskit.pulse available: {self._has_pulse}."
        )
        return Capabilities(
            name=self._name,
            vendor="qmesh+openpulse",
            modalities={Modality.GATE},
            qubit_count=20,
            native_gates=set(_DEFAULT_PULSE_CALIBRATION),
            measurement_feedforward=False,
            classical_control=ControlLevel.NONE,
            pulse_access=True,
            cost_per_shot_usd=0.0,
            queue_depth=0,
            fidelity_2q_typical=0.99 if self._has_pulse else 1.0,
            is_simulator=True,
            notes=notes,
        )

    def run(self, module: Module, shots: int = 1024, **kwargs: Any) -> RunResult:
        ok, why = self.accepts(module)
        if not ok:
            raise ValueError(f"openpulse rejected module: {why}")

        descriptor = _build_schedule_descriptor(module)
        meta: dict[str, Any] = {
            "schedule_descriptor": descriptor,
            "qiskit_pulse_available": self._has_pulse,
            "execution_path": "schedule_only",
        }

        t0 = time.time()
        if self._has_pulse:
            try:
                sched = _build_qiskit_pulse_schedule(module)
                meta["qiskit_schedule_duration"] = int(sched.duration)
                meta["qiskit_schedule_n_instructions"] = len(sched.instructions)
                meta["execution_path"] = "qiskit.pulse.Schedule"
            except Exception as e:
                meta["execution_path"] = "schedule_build_failed"
                meta["schedule_build_error"] = str(e)

        # Counts: route through an existing gate-level simulator so the
        # backend remains useful (returns valid counts) even when no pulse
        # simulator is installed. We deliberately do *not* attach
        # qiskit-dynamics here — that would balloon the optional-dep matrix
        # and is the right place for a Phase-3β / vendor-collab item.
        counts: dict[str, int] = {}
        if self._fallback:
            counts, sub_meta = self._fallback_counts(module, shots)
            meta["fallback_backend"] = sub_meta.get("backend_name", "unknown")
        wall = time.time() - t0

        return RunResult(
            counts=counts,
            shots=shots,
            wall_seconds=wall,
            qpu_seconds=wall,
            cost_usd=0.0,
            backend_metadata=meta,
        )

    def _fallback_counts(
        self, module: Module, shots: int,
    ) -> tuple[dict[str, int], dict[str, Any]]:
        """Run gate-level execution on the best installed simulator."""
        from qmesh.backends.registry import all_backends, get
        # Prefer the lightest, most-likely-installed simulator.
        for cand in ("qmesh.statevec", "qmesh.aer", "qmesh.stim"):
            if cand in all_backends():
                try:
                    sub = get(cand).run(module, shots=shots)
                    return sub.counts, {"backend_name": cand}
                except Exception:
                    continue
        return {}, {"backend_name": "none"}


# self-register
register(OpenPulseSimulator())
