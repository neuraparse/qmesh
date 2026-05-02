"""qmesh.provenance.manifest — provenance-by-default.

Every execution emits a Manifest:
    {
        circuit { ir_hash_sha256, ir_cbor_path },
        frontend { name, version },
        compiler { passes [{ name, version, model_hash? }], input_hash, output_hash },
        ftmode { code, decoder, cultivation { ... } } | null,
        mitigation { stack [{ name, ... }] } | null,
        backend { vendor, device, calibration_snapshot_hash, queue_position },
        execution { shots, wall_seconds, qpu_seconds, cost_usd, raw_counts_path,
                    post_processed_path },
        signature { alg, value }
    }

Manifests are deterministic; the same canonical JSON encoding hashes to the
same value across machines, so they can be content-addressed and shared.

Signing uses ed25519 (cryptography library) — supports a per-host signing key
or a user/team-supplied key. Trust model is "users sign their own claims and
publish their public key"; verification is local and stateless.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field, asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


def _canonical_json(obj: Any) -> bytes:
    """Stable JSON serialisation (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


@dataclass(slots=True)
class Manifest:
    qmesh_version: str
    submitted_at: str
    submitter: dict[str, str]
    circuit: dict[str, str]
    frontend: dict[str, str]
    compiler: dict[str, Any]
    backend: dict[str, Any]
    execution: dict[str, Any]
    ftmode: dict[str, Any] | None = None
    mitigation: dict[str, Any] | None = None
    # Lineage block: present when this manifest descends from a parent run
    # (e.g. a resumed DAG). Carries parent_run_id, parent_manifest_hash,
    # parent_signature_value, n_skipped_resume — first-class so Metriq export
    # and audit-chain verification do not need ad-hoc dict-key parsing.
    lineage: dict[str, Any] | None = None
    signature: dict[str, str] | None = None

    @classmethod
    def new(cls, *, qmesh_version: str, ir_hash: str, ir_path: str | None,
            frontend: dict[str, str], backend: dict[str, Any]) -> Manifest:
        return cls(
            qmesh_version=qmesh_version,
            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            submitter={
                "host": socket.gethostname(),
                "user": os.getenv("USER", "unknown"),
                "platform": platform.platform(),
            },
            circuit={
                "ir_hash_sha256": ir_hash,
                "ir_cbor_path": ir_path or "",
            },
            frontend=frontend,
            compiler={"passes": [], "input_hash": ir_hash, "output_hash": ir_hash},
            backend=backend,
            execution={},
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("ftmode") is None:
            d.pop("ftmode", None)
        if d.get("mitigation") is None:
            d.pop("mitigation", None)
        if d.get("lineage") is None:
            d.pop("lineage", None)
        if d.get("signature") is None:
            d.pop("signature", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def hash(self) -> str:
        d = self.to_dict()
        d.pop("signature", None)
        return sha256(_canonical_json(d)).hexdigest()

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path


@dataclass(slots=True)
class ManifestSigner:
    """ed25519 signer. Generates per-host key if not supplied.

    Public key is embedded in the manifest's submitter field; verifiers use it
    along with the canonical-JSON signature.

    When ``cert_chain`` is supplied (Phase 6β trust model), it is embedded
    in ``submitter.cert_chain`` *before* the canonical body is signed. That
    way the chain is part of the integrity guarantee — flipping a cert
    invalidates the signature.
    """

    private_key_path: Path = field(default=Path.home() / ".qmesh" / "id_ed25519")
    cert_chain: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not _HAS_CRYPTO:
            return
        if not self.private_key_path.exists():
            self.private_key_path.parent.mkdir(parents=True, exist_ok=True)
            key = Ed25519PrivateKey.generate()
            self.private_key_path.write_bytes(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
            self.private_key_path.chmod(0o600)

    def _load(self) -> Ed25519PrivateKey:
        return serialization.load_pem_private_key(  # type: ignore[return-value]
            self.private_key_path.read_bytes(), password=None
        )

    def public_key_hex(self) -> str:
        if not _HAS_CRYPTO:
            return "ed25519:unsigned"
        key = self._load()
        pub = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return f"ed25519:{pub.hex()}"

    def sign(self, manifest: Manifest) -> Manifest:
        manifest.submitter["public_key"] = self.public_key_hex()
        if self.cert_chain:
            # Phase 6β: bind the cert chain into the body so any post-hoc
            # cert swap invalidates the signature.
            manifest.submitter["cert_chain"] = list(self.cert_chain)
        if not _HAS_CRYPTO:
            manifest.signature = {"alg": "unsigned", "value": ""}
            return manifest
        key = self._load()
        body = self._canonical_body(manifest)
        sig = key.sign(body)
        manifest.signature = {"alg": "ed25519", "value": sig.hex()}
        return manifest

    @staticmethod
    def verify(manifest: Manifest) -> bool:
        if not _HAS_CRYPTO:
            return manifest.signature is not None and manifest.signature.get("alg") == "unsigned"
        if not manifest.signature or manifest.signature["alg"] != "ed25519":
            return False
        pub_hex = manifest.submitter.get("public_key", "").removeprefix("ed25519:")
        if not pub_hex:
            return False
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
        body = ManifestSigner._canonical_body(manifest)
        try:
            pub.verify(bytes.fromhex(manifest.signature["value"]), body)
            return True
        except Exception:
            return False

    @staticmethod
    def _canonical_body(manifest: Manifest) -> bytes:
        d = manifest.to_dict()
        d.pop("signature", None)
        return _canonical_json(d)
