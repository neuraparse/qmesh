"""qmesh.compliance — tamper-evident archives over signed-manifest run-windows.

Public API:
    build_pack(ledger_dir, since=None, until=None, out_path=...) -> Path
    verify_pack(archive_path) -> ComplianceVerification
    ComplianceVerification (dataclass)

Schema overview:
    archive (.tar.gz):
        manifests/<n>_<hash>.json   — original manifest bytes (canonical JSON)
        signatures.json             — {n -> {alg, value, public_key}}
        chain.json                  — list[ChainEntry], one per included manifest
        README.txt                  — human-readable verify steps

Each ChainEntry:
    {
        "index"           : 0..N-1,
        "manifest_path"   : "manifests/000_<hash>.json",
        "manifest_hash"   : sha256(manifest minus signature),
        "signature_alg"   : "ed25519" | "unsigned" | ...,
        "signature_value" : hex,
        "prev_hash"       : prior chain entry's `this_hash` (or "GENESIS"),
        "this_hash"       : sha256(prev_hash || manifest_hash || signature_value),
    }

Verification re-walks the chain, recomputes each entry's `this_hash`, and
re-verifies every ed25519 signature against the embedded public key — no
private key needed, no network needed.
"""

from __future__ import annotations

from qmesh.compliance.pack import (
    ComplianceVerification,
    build_pack,
    verify_pack,
)
from qmesh.compliance.trust import (
    Certificate,
    TrustResult,
    TrustStore,
    attach_cert_chain_to_manifest,
    generate_root_keypair,
    issue_certificate,
    verify_certificate_signature,
    verify_chain,
)

__all__ = [
    "ComplianceVerification",
    "build_pack",
    "verify_pack",
    # cert-chain trust model (Phase 6β)
    "Certificate",
    "TrustResult",
    "TrustStore",
    "attach_cert_chain_to_manifest",
    "generate_root_keypair",
    "issue_certificate",
    "verify_certificate_signature",
    "verify_chain",
]
