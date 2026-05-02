"""Manifest signing/verification + replay/diff tests."""

from __future__ import annotations

import json

import qmesh
from qmesh.api import diff, replay
from qmesh.provenance.manifest import Manifest, ManifestSigner


def _bell():
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    return c.module


def test_manifest_signature_round_trips(tmp_path):
    _, m = qmesh.submit(_bell(), backend="qmesh.statevec", shots=128, ledger_dir=tmp_path)
    assert m.signature is not None
    assert m.signature["alg"] == "ed25519"
    assert ManifestSigner.verify(m)


def test_manifest_tampering_breaks_verification(tmp_path):
    _, m = qmesh.submit(_bell(), backend="qmesh.statevec", shots=128, ledger_dir=tmp_path)
    m.execution["shots"] = 99999  # tamper
    assert not ManifestSigner.verify(m)


def test_replay_accepts_valid_manifest(tmp_path):
    _, m = qmesh.submit(_bell(), backend="qmesh.statevec", shots=128, ledger_dir=tmp_path)
    out_dir = tmp_path / m.submitted_at[:10]
    files = list(out_dir.glob("*.json"))
    assert len(files) == 1
    ok, msg = replay(files[0])
    assert ok, msg


def test_diff_two_manifests(tmp_path):
    _, m1 = qmesh.submit(_bell(), backend="qmesh.statevec", shots=128, ledger_dir=tmp_path)
    _, m2 = qmesh.submit(_bell(), backend="qmesh.statevec", shots=256, ledger_dir=tmp_path)

    out_dir = tmp_path / m1.submitted_at[:10]
    files = sorted(out_dir.glob("*.json"))
    d = diff(files[0], files[1])
    # at minimum execution.shots should differ
    assert "execution" in d
