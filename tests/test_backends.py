"""Backend correctness tests."""

from __future__ import annotations

import pytest

import qmesh


def _bell():
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    return c.module


def _ghz(n: int):
    with qmesh.circuit(f"ghz{n}", n_qubits=n, n_bits=n) as c:
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)
        for i in range(n):
            c.measure(i, i)
    return c.module


@pytest.mark.parametrize("backend", ["qmesh.statevec", "qmesh.aer", "qmesh.stim"])
def test_bell_state_50_50(backend, tmp_path):
    pytest.importorskip("qiskit_aer") if backend == "qmesh.aer" else None
    pytest.importorskip("stim") if backend == "qmesh.stim" else None
    r, _ = qmesh.submit(_bell(), backend=backend, shots=8192, ledger_dir=tmp_path)
    p_even = (r.counts.get("00", 0) + r.counts.get("11", 0)) / r.shots
    assert p_even > 0.97, f"{backend} Bell parity too low: {r.counts}"


def test_aer_noisy_introduces_some_errors(tmp_path):
    pytest.importorskip("qiskit_aer")
    r, _ = qmesh.submit(_bell(), backend="qmesh.aer.noisy", shots=8192, ledger_dir=tmp_path)
    p_odd = (r.counts.get("01", 0) + r.counts.get("10", 0)) / r.shots
    # Heron-like noise should land somewhere between 1% and 8% odd-parity
    assert 0.005 < p_odd < 0.10, f"unexpected noisy parity: {p_odd}"


def test_stim_rejects_non_clifford(tmp_path):
    pytest.importorskip("stim")
    with qmesh.circuit("t", n_qubits=1, n_bits=1) as c:
        c.h(0); c.t(0); c.measure(0, 0)
    with pytest.raises(ValueError):
        qmesh.submit(c.module, backend="qmesh.stim", shots=10, ledger_dir=tmp_path)


def test_ghz_scaling_correctness(tmp_path):
    pytest.importorskip("qiskit_aer")
    for n in (4, 8, 12):
        r, _ = qmesh.submit(_ghz(n), backend="qmesh.aer", shots=4096, ledger_dir=tmp_path)
        good = "0" * n
        bad = "1" * n
        p = (r.counts.get(good, 0) + r.counts.get(bad, 0)) / r.shots
        assert p > 0.97, f"GHZ {n}-qubit parity {p}"


# ---------- Phase 3δ: OpenPulse backend skeleton -----------------------------

def test_openpulse_backend_registered():
    """OpenPulse backend self-registers and reports pulse_access=True."""
    from qmesh.backends import all_backends
    backends = all_backends()
    assert "qmesh.openpulse" in backends
    bk = backends["qmesh.openpulse"]
    caps = bk.capabilities
    assert caps.pulse_access is True
    assert caps.is_simulator is True


def test_openpulse_emits_schedule_descriptor(tmp_path):
    """Running a Bell circuit through qmesh.openpulse must produce a valid
    schedule descriptor with the right number of events (h + cx + 2*measure)
    and counts via the gate-level fallback path. Without qiskit.pulse the
    execution_path stays at 'schedule_only'; with qiskit.pulse it switches
    to 'qiskit.pulse.Schedule'."""
    r, m = qmesh.submit(_bell(), backend="qmesh.openpulse", shots=512,
                        ledger_dir=tmp_path)
    meta = m.execution.get("backend_metadata") or {}
    desc = meta.get("schedule_descriptor")
    assert desc is not None and desc["schedule_format"] == "qmesh.openpulse.v0"
    # Bell: h(0) + cx(0,1) + measure(0,0) + measure(1,1) → 4 events
    assert desc["n_events"] == 4
    kinds = [ev["kind"] for ev in desc["events"]]
    assert kinds.count("gate_pulse") == 2
    assert kinds.count("measure") == 2
    # Schedule has positive total duration (h: 40ns + cx: 200ns + meas: 1500ns).
    assert desc["total_duration_ns"] > 0
    # Bell parity comes through the fallback: still ~50/50 on 00 vs 11.
    p_even = (r.counts.get("00", 0) + r.counts.get("11", 0)) / r.shots
    assert p_even > 0.9, f"openpulse fallback Bell parity too low: {r.counts}"


def test_openpulse_qiskit_pulse_status_reported_truthfully():
    """The capabilities + run() metadata report whether qiskit.pulse is
    actually importable. This is what auditors read to know if they're
    looking at β (real schedule) or α (descriptor-only)."""
    import qmesh
    from qmesh.backends import get
    from qmesh.backends.openpulse_sim import _pulse_module_available

    bk = get("qmesh.openpulse")
    expected = _pulse_module_available()
    notes = bk.capabilities.notes
    assert (f"qiskit.pulse available: {expected}") in notes
