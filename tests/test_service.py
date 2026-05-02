"""FastAPI service-mode tests."""

from __future__ import annotations

import pytest


def _client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from qmesh.service.app import app
    return TestClient(app)


def test_health():
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_backends_listing():
    c = _client()
    r = c.get("/backends")
    assert r.status_code == 200
    body = r.json()
    assert any(b["name"] == "qmesh.statevec" for b in body)


def test_submit_round_trip():
    c = _client()
    qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q; bit[2] c;
h q[0]; cx q[0], q[1];
c[0] = measure q[0]; c[1] = measure q[1];
"""
    r = c.post("/submit", json={"qasm": qasm, "shots": 1024})
    assert r.status_code == 200
    data = r.json()
    assert data["shots"] == 1024
    assert data["chosen_backend"] in ("qmesh.statevec", "qmesh.aer", "qmesh.stim")

    # verify with the 16-char on-disk prefix
    v = c.get(f"/manifests/{data['manifest_hash'][:16]}/verify")
    assert v.status_code == 200
    assert v.json()["verified"] is True

    # verify with the full 64-char hash that /submit actually returns —
    # this is the typical caller flow. The lookup must accept it too.
    full = c.get(f"/manifests/{data['manifest_hash']}/verify")
    assert full.status_code == 200, (
        f"GET /manifests/<full-hash>/verify returned {full.status_code}; "
        f"the lookup should accept the full hash that /submit emits."
    )
    assert full.json()["verified"] is True

    # GET the manifest body with the full hash works too.
    body = c.get(f"/manifests/{data['manifest_hash']}")
    assert body.status_code == 200
    assert body.json()["qmesh_version"]
