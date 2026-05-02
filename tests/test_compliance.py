"""Compliance pack + Metriq exporter + audit-grade FT records (Phase 6)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import qmesh
from qmesh.compliance import (
    Certificate,
    ComplianceVerification,
    TrustResult,
    TrustStore,
    attach_cert_chain_to_manifest,
    build_pack,
    generate_root_keypair,
    issue_certificate,
    verify_certificate_signature,
    verify_chain,
    verify_pack,
)
from qmesh.metriq import (
    export_manifest,
    export_run_dir,
    submit_to_metriq,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _bell_run(ledger_dir: Path):
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    _, m = qmesh.submit(
        c.module, backend="qmesh.statevec", shots=128, ledger_dir=ledger_dir,
    )
    return m


def _ghz_run(ledger_dir: Path):
    with qmesh.circuit("ghz", n_qubits=3, n_bits=3) as c:
        c.h(0); c.cx(0, 1); c.cx(1, 2)
        c.measure(0, 0); c.measure(1, 1); c.measure(2, 2)
    _, m = qmesh.submit(
        c.module, backend="qmesh.statevec", shots=256, ledger_dir=ledger_dir,
    )
    return m


def _find_manifest_path(ledger_dir: Path, manifest) -> Path:
    out_dir = ledger_dir / manifest.submitted_at[:10]
    matches = list(out_dir.glob(f"{manifest.hash()[:16]}.json"))
    assert matches, f"could not locate manifest under {out_dir}"
    return matches[0]


# --------------------------------------------------------------------------- #
# Metriq export                                                               #
# --------------------------------------------------------------------------- #


def test_metriq_export_translates_a_signed_manifest(tmp_path):
    m = _bell_run(tmp_path)
    p = _find_manifest_path(tmp_path, m)

    sub = export_manifest(p, name="bell-α", tags=["bell", "qmesh"])
    assert "name" in sub
    assert "results" in sub
    assert isinstance(sub["results"], list) and sub["results"], (
        "metriq export must contain at least one result row"
    )
    row = sub["results"][0]
    for k in ("task_id", "method_name", "metric_name", "metric_value",
              "metric_unit", "sample_size", "evaluatedAt", "evidence"):
        assert k in row, f"missing key {k!r} in result row"

    # Evidence should reproduce the manifest hash exactly.
    assert row["evidence"]["manifest_hash"] == m.hash(), (
        "metriq evidence.manifest_hash must equal manifest.hash()"
    )
    assert row["evidence"]["signature_alg"] == "ed25519"
    assert row["evidence"]["signature_value"] == m.signature["value"]


def test_metriq_dry_run_submit_returns_payload_without_network(tmp_path, monkeypatch):
    """submit_to_metriq with dry_run=True must not even try to load `requests`."""
    m = _bell_run(tmp_path)
    p = _find_manifest_path(tmp_path, m)
    sub = export_manifest(p)

    # Sabotage `requests` via sys.modules so any accidental import would fail.
    import sys
    sentinel = object()
    monkeypatch.setitem(sys.modules, "requests", sentinel)  # not callable as .post

    out = submit_to_metriq(sub, endpoint="https://metriq.info/api/x", dry_run=True)
    assert out["dry_run"] is True
    assert out["payload"] == sub
    assert out["n_results"] == len(sub["results"])
    assert out["endpoint"].startswith("https://")


def test_export_run_dir_walks_multiple_manifests(tmp_path):
    _bell_run(tmp_path)
    _ghz_run(tmp_path)

    sub = export_run_dir(tmp_path)
    # Bell + GHZ each give at least one row → at least 2 results
    assert len(sub["results"]) >= 2
    task_ids = {r["task_id"] for r in sub["results"]}
    assert len(task_ids) >= 2


# --------------------------------------------------------------------------- #
# Compliance pack                                                             #
# --------------------------------------------------------------------------- #


def test_compliance_pack_round_trips(tmp_path):
    # Build a tmp ledger with 3 signed Bell manifests
    for _ in range(3):
        _bell_run(tmp_path)

    out_arc = tmp_path / "pack.tar.gz"
    arc = build_pack(tmp_path, out_path=out_arc)
    assert arc.exists() and arc.stat().st_size > 0

    r: ComplianceVerification = verify_pack(arc)
    assert r.chain_valid is True
    assert r.n_manifests == 3
    assert r.n_signatures_ok == 3
    assert r.broken_at is None
    assert not r.errors


def test_compliance_pack_detects_tamper(tmp_path):
    for _ in range(3):
        _bell_run(tmp_path)

    out_arc = tmp_path / "pack.tar.gz"
    arc = build_pack(tmp_path, out_path=out_arc)

    # Re-write the archive with one manifest's payload mutated.
    import io
    import tarfile

    edited = tmp_path / "tampered.tar.gz"
    with tarfile.open(arc, "r:gz") as src, tarfile.open(edited, "w:gz") as dst:
        for member in src.getmembers():
            f = src.extractfile(member)
            data = f.read() if f is not None else b""
            if member.name.startswith("manifests/") and member.name.endswith(".json") \
                    and "001" in member.name:
                # Tamper a substantive field in the manifest body.
                body = json.loads(data)
                body.setdefault("execution", {})["shots"] = 99999
                data = json.dumps(body, sort_keys=True, indent=2).encode()
                member.size = len(data)
            dst.addfile(member, io.BytesIO(data))

    r = verify_pack(edited)
    assert r.chain_valid is False
    assert r.broken_at is not None
    assert r.broken_at >= 1
    assert any("manifest_hash mismatch" in e or "this_hash mismatch" in e
               for e in r.errors)


def test_compliance_pack_skips_unsigned_with_warning(tmp_path):
    """Unsigned manifests are bundled but counted as not-signature-ok with a note."""
    # Build 1 signed + 1 unsigned manifest
    _bell_run(tmp_path)

    with qmesh.circuit("bell2", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    _, m_unsigned = qmesh.submit(
        c.module, backend="qmesh.statevec", shots=64,
        ledger_dir=tmp_path, sign=False,
    )

    out_arc = tmp_path / "mixed.tar.gz"
    build_pack(tmp_path, out_path=out_arc)
    r = verify_pack(out_arc)
    assert r.n_manifests == 2
    # The unsigned one cannot contribute to signature audit
    assert r.n_signatures_ok == 1
    assert any("unsigned" in n for n in r.notes)
    # Chain itself is still valid (unsigned doesn't break the chain)
    assert r.chain_valid is True


def test_compliance_verify_offline_no_keys_required(tmp_path, monkeypatch):
    """verify_pack must work with private key file unreadable — only public
    keys embedded in manifests are needed."""
    _bell_run(tmp_path)
    _bell_run(tmp_path)

    out_arc = tmp_path / "pack.tar.gz"
    build_pack(tmp_path, out_path=out_arc)

    # Sabotage the user's private key location: point HOME at empty dir,
    # and make the standard signing key path a non-existent location.
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    r = verify_pack(out_arc)
    assert r.chain_valid is True
    assert r.n_signatures_ok == 2


# --------------------------------------------------------------------------- #
# Audit-grade FT execution records                                            #
# --------------------------------------------------------------------------- #


def test_ft_decoder_audit_records_weights_sha256(tmp_path):
    """Train a NeuralDecoder, run memory_experiment with it, and assert
    the manifest captures the weights' sha256 inside ftmode.decoder_audit."""
    pytest.importorskip("torch")
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    from qmesh.ai import train_neural_decoder
    from qmesh.ftmode import FTConfig, SurfaceCode, memory_experiment

    code = SurfaceCode(distance=3, rounds=4)
    weights_dir = tmp_path / "weights"
    decoder, tr = train_neural_decoder(
        code,
        physical_error_rate=1e-3,
        n_train=200,
        n_val=80,
        epochs=1,
        hidden_dim=16,
        weights_dir=weights_dir,
    )
    assert Path(tr.weights_path).exists()
    assert decoder.weights_path is not None

    # FTConfig.build_decoder dispatches via DecoderChoice; we route the
    # trained NeuralDecoder in by monkeypatching the class-level method.
    cfg = FTConfig(distance=3, rounds=4, physical_error_rate=1e-3)
    orig_build = FTConfig.build_decoder
    FTConfig.build_decoder = lambda self: decoder  # type: ignore[assignment]
    try:
        res, manifest = memory_experiment(
            ftconfig=cfg, shots=200, ledger_dir=tmp_path / "ft_ledger", seed=1,
        )
    finally:
        FTConfig.build_decoder = orig_build  # type: ignore[assignment]
    audit = manifest.ftmode["decoder_audit"]
    assert audit["weights_sha256"], (
        f"decoder_audit.weights_sha256 unset on neural decoder run: {audit!r}"
    )
    # Independently re-hash the file and compare.
    import hashlib
    h = hashlib.sha256(Path(tr.weights_path).read_bytes()).hexdigest()
    assert audit["weights_sha256"] == h
    assert audit.get("weights_size_bytes", 0) > 0


def test_ft_cultivation_chain_links_two_runs(tmp_path):
    """Call memory_experiment twice; the second with chain_to=first.manifest_path.
    Assert the second's cultivation_audit.cumulative_ft_chain == first.hash()."""
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    from qmesh.ftmode import FTConfig, memory_experiment

    cfg = FTConfig(distance=3, rounds=4, physical_error_rate=1e-3,
                   decoder="pymatching")
    res1, m1 = memory_experiment(
        ftconfig=cfg, shots=200, ledger_dir=tmp_path / "ft_ledger", seed=1,
    )
    assert res1.manifest_path is not None
    res2, m2 = memory_experiment(
        ftconfig=cfg, shots=200, ledger_dir=tmp_path / "ft_ledger", seed=2,
        chain_to=res1.manifest_path,
    )

    audit1 = m1.ftmode["cultivation_audit"]
    audit2 = m2.ftmode["cultivation_audit"]
    assert audit1["cumulative_ft_chain"] is None, (
        "first FT run has no parent — chain slot must be null"
    )
    assert audit2["cumulative_ft_chain"] == m1.hash(), (
        f"second run's cumulative_ft_chain={audit2['cumulative_ft_chain']!r} "
        f"does not equal first.hash()={m1.hash()!r}"
    )
    assert audit2["chain_to_path"] == str(res1.manifest_path)
    # Both audit blocks carry a signing timestamp + a snapshot of the
    # cultivation params.
    assert "audit_signed_at" in audit2
    assert audit2["snapshot"]["factory"] == "in_place_cultivation_v0"


# --------------------------------------------------------------------------- #
# Cert-chain trust model (Phase 6β)                                           #
# --------------------------------------------------------------------------- #


def _signed_bell_with_cert_chain(ledger_dir: Path, signer):
    """Run a Bell circuit, then re-sign the on-disk manifest using `signer`
    so the test can pin a specific cert chain into submitter.cert_chain."""
    with qmesh.circuit("bell-cert", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    _, m = qmesh.submit(
        c.module, backend="qmesh.statevec", shots=64,
        ledger_dir=ledger_dir, sign=False,
    )
    p = _find_manifest_path(ledger_dir, m)
    # Re-sign the on-disk file with the cert-bearing signer.
    m_data = json.loads(p.read_text())
    m_data["signature"] = None
    m_data["submitter"] = {
        k: v for k, v in m_data.get("submitter", {}).items()
        if k not in ("public_key", "cert_chain")
    }
    # Use the dataclass to round-trip through sign().
    from qmesh.provenance.manifest import Manifest
    rehydrated = Manifest(
        qmesh_version=m_data["qmesh_version"],
        submitted_at=m_data["submitted_at"],
        submitter=m_data["submitter"],
        circuit=m_data["circuit"],
        frontend=m_data["frontend"],
        compiler=m_data["compiler"],
        backend=m_data["backend"],
        execution=m_data["execution"],
        ftmode=m_data.get("ftmode"),
        mitigation=m_data.get("mitigation"),
        lineage=m_data.get("lineage"),
    )
    signer.sign(rehydrated)
    p.unlink()
    rehydrated.write(p)
    return rehydrated, p


def test_certificate_issuance_and_self_verification(tmp_path):
    """A cert issued by a root key must self-verify under
    verify_certificate_signature without any chain walk."""
    root_pem, root_label = generate_root_keypair()
    user_pem, user_label = generate_root_keypair()

    cert = issue_certificate(
        subject_pubkey=user_label,
        subject_name="alice@team",
        issuer_private_key_pem=root_pem,
        issuer_name="qmesh-corp-ca",
    )
    assert cert.signature, "issued cert must carry a signature"
    assert cert.subject_pubkey == user_label
    assert cert.issuer_pubkey == root_label
    assert verify_certificate_signature(cert) is True

    # Tampering with subject_name invalidates the signature.
    bad = Certificate.from_dict({**cert.to_dict(), "subject_name": "mallory@team"})
    assert verify_certificate_signature(bad) is False


def test_verify_chain_one_level(tmp_path):
    """Leaf cert directly issued by the root → chain valid."""
    root_pem, root_label = generate_root_keypair()
    _, leaf_label = generate_root_keypair()

    leaf_cert = issue_certificate(
        subject_pubkey=leaf_label,
        subject_name="alice",
        issuer_private_key_pem=root_pem,
        issuer_name="qmesh-corp-ca",
    )

    store = TrustStore()
    store.trust(root_label, "qmesh-corp-ca")
    r = verify_chain(leaf_label, [leaf_cert], store)
    assert r.valid, r.reason
    assert len(r.chain) == 1
    assert r.chain[0].subject_pubkey == leaf_label


def test_verify_chain_intermediate_ca(tmp_path):
    """leaf → intermediate → root: chain walks two levels."""
    root_pem, root_label = generate_root_keypair()
    int_pem, int_label = generate_root_keypair()
    _, leaf_label = generate_root_keypair()

    int_cert = issue_certificate(
        subject_pubkey=int_label,
        subject_name="qmesh-eu-ca",
        issuer_private_key_pem=root_pem,
        issuer_name="qmesh-root-ca",
    )
    leaf_cert = issue_certificate(
        subject_pubkey=leaf_label,
        subject_name="alice",
        issuer_private_key_pem=int_pem,
        issuer_name="qmesh-eu-ca",
    )

    store = TrustStore()
    store.trust(root_label, "qmesh-root-ca")
    r = verify_chain(leaf_label, [leaf_cert, int_cert], store)
    assert r.valid, r.reason
    assert len(r.chain) == 2


def test_verify_chain_rejects_unrooted_leaf(tmp_path):
    """An unknown root → not trusted, with a clear reason."""
    root_pem, root_label = generate_root_keypair()
    _, leaf_label = generate_root_keypair()

    leaf_cert = issue_certificate(
        subject_pubkey=leaf_label,
        subject_name="alice",
        issuer_private_key_pem=root_pem,
        issuer_name="rogue-ca",
    )
    empty_store = TrustStore()
    r = verify_chain(leaf_label, [leaf_cert], empty_store)
    assert r.valid is False
    assert "trust store is empty" in r.reason

    store = TrustStore()
    store.trust("ed25519:00" * 16, "some-other-root")
    r2 = verify_chain(leaf_label, [leaf_cert], store)
    assert r2.valid is False


def test_verify_chain_detects_cycle(tmp_path):
    """Two certs that point at each other → cycle detected, not infinite loop."""
    a_pem, a_label = generate_root_keypair()
    b_pem, b_label = generate_root_keypair()
    cert_a_by_b = issue_certificate(
        subject_pubkey=a_label, subject_name="A",
        issuer_private_key_pem=b_pem, issuer_name="B",
    )
    cert_b_by_a = issue_certificate(
        subject_pubkey=b_label, subject_name="B",
        issuer_private_key_pem=a_pem, issuer_name="A",
    )
    store = TrustStore()
    store.trust("ed25519:00" * 16, "unrelated-root")
    r = verify_chain(a_label, [cert_a_by_b, cert_b_by_a], store)
    assert r.valid is False
    assert "cycle" in r.reason.lower()


def test_verify_chain_validity_window(tmp_path):
    """A cert whose validity window doesn't cover the manifest instant fails."""
    root_pem, root_label = generate_root_keypair()
    _, leaf_label = generate_root_keypair()
    leaf_cert = issue_certificate(
        subject_pubkey=leaf_label,
        subject_name="alice",
        issuer_private_key_pem=root_pem,
        issuer_name="ca",
        not_before="2030-01-01T00:00:00Z",
        not_after="2031-01-01T00:00:00Z",
    )
    store = TrustStore()
    store.trust(root_label, "ca")

    # Within window — ok.
    r_ok = verify_chain(leaf_label, [leaf_cert], store, instant="2030-06-01T00:00:00Z")
    assert r_ok.valid is True
    # Before window — fail.
    r_bad = verify_chain(leaf_label, [leaf_cert], store, instant="2026-05-02T00:00:00Z")
    assert r_bad.valid is False
    assert "not valid at" in r_bad.reason


def test_verify_pack_with_trust_store(tmp_path):
    """Sign two manifests with a cert-chained ed25519 key, build a pack, then
    verify_pack(trust_roots=...) reports trust_chain_valid=True."""
    from qmesh.provenance.manifest import ManifestSigner

    root_pem, root_label = generate_root_keypair()
    user_priv_path = tmp_path / "user_id_ed25519"
    user_pem, user_label = generate_root_keypair()
    user_priv_path.write_bytes(user_pem)

    # Issue a cert: root signs over user pubkey.
    user_cert = issue_certificate(
        subject_pubkey=user_label,
        subject_name="alice",
        issuer_private_key_pem=root_pem,
        issuer_name="qmesh-root-ca",
    )

    signer = ManifestSigner(
        private_key_path=user_priv_path,
        cert_chain=[user_cert.to_dict()],
    )
    # The signer's cert chain expects user_pem to match user_label — and it
    # does, because we wrote the same PEM at user_priv_path.
    for _ in range(2):
        _signed_bell_with_cert_chain(tmp_path, signer)

    out = tmp_path / "trusted.tar.gz"
    build_pack(tmp_path, out_path=out)

    store = TrustStore()
    store.trust(root_label, "qmesh-root-ca")
    r = verify_pack(out, trust_roots=store)
    assert r.chain_valid is True
    assert r.n_signatures_ok == 2
    assert r.trust_chain_valid is True
    assert r.n_trusted == 2
    assert not r.untrusted_signers


def test_verify_pack_trust_audit_skipped_without_store(tmp_path):
    """Without trust_roots, trust_chain_valid stays None — back-compat with α."""
    _bell_run(tmp_path)
    out = tmp_path / "p.tar.gz"
    build_pack(tmp_path, out_path=out)
    r = verify_pack(out)  # no trust_roots
    assert r.trust_chain_valid is None
    assert r.n_trusted == 0


def test_verify_pack_flags_untrusted_signer(tmp_path):
    """A signed manifest whose cert chain doesn't reach a configured root
    is flagged in `untrusted_signers` and trust_chain_valid is False."""
    _bell_run(tmp_path)   # signed with the *default* host key — no cert chain.
    out = tmp_path / "untrusted.tar.gz"
    build_pack(tmp_path, out_path=out)

    _, root_label = generate_root_keypair()
    store = TrustStore()
    store.trust(root_label, "totally-different-ca")
    r = verify_pack(out, trust_roots=store)
    assert r.chain_valid is True            # internal chain still ok
    assert r.n_signatures_ok == 1            # signature itself verifies
    assert r.trust_chain_valid is False
    assert r.n_trusted == 0
    assert len(r.untrusted_signers) == 1
