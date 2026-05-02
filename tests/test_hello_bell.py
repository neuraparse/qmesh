"""Smoke test: Bell state runs end-to-end through qmesh on the bundled simulator."""

from __future__ import annotations

import qmesh
from qmesh.provenance.manifest import ManifestSigner


def test_bell_runs_and_manifest_is_signed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0)
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)
    module = c.module

    result, manifest = qmesh.submit(
        module, backend="qmesh.statevec", shots=2048,
        ledger_dir=tmp_path,
    )

    # In the |00> + |11> Bell basis we expect ~50/50 between "00" and "11"
    total = sum(result.counts.values())
    assert total == 2048
    p00 = result.counts.get("00", 0) / total
    p11 = result.counts.get("11", 0) / total
    assert p00 + p11 > 0.95, f"unexpected counts: {result.counts}"

    # Determinism: same module hashes identically
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c2:
        c2.h(0); c2.cx(0, 1); c2.measure(0, 0); c2.measure(1, 1)
    assert module.hash() == c2.module.hash()

    # Manifest is signed and round-trips verification
    assert manifest.signature is not None
    assert ManifestSigner.verify(manifest)
