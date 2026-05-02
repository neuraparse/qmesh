"""qmesh.compliance.trust — cert-chain trust model on top of ed25519 sigs.

Phase 6β. The Phase 6α pack already verifies that each manifest carries a
signature whose embedded public key validates against the canonical body.
That answers "is this body the same body the signer signed?" — but it does
*not* answer "do I trust the signer?".

This module adds a thin certificate model that fills exactly that gap:

  * A :class:`Certificate` is an ed25519 signature by an *issuer* key over a
    *subject* public key + metadata (subject name, validity window, serial).
  * Issuers themselves can be subjects of higher-level certs, giving a chain
    leaf → ... → root.
  * A :class:`TrustStore` holds a configured set of root public keys (CAs).
    A leaf is "trusted" iff a chain of valid certs leads from the leaf
    pubkey to a root in the store.

Manifest integration:
  * ``submitter.cert_chain`` may carry a list of cert dicts. When present,
    :func:`qmesh.compliance.verify_pack` will walk it against an optional
    ``trust_roots: TrustStore`` argument and report ``trust_chain_valid``
    + ``untrusted_signers`` alongside the existing chain audit.

α-honest scope:
  * No revocation list yet (β); ``not_before`` / ``not_after`` are honoured
    against the manifest's ``submitted_at`` if available.
  * No name-constraint extensions; the ``subject_name`` field is informational.
  * Roots are pinned by hex-encoded raw pubkey (``ed25519:<hex>``).
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Iterable

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


_ED25519_PREFIX = "ed25519:"


def _strip_prefix(pubkey_label: str) -> str:
    return pubkey_label.removeprefix(_ED25519_PREFIX)


def _canonical_cert_body(d: dict[str, Any]) -> bytes:
    """Cert canonical body excludes the signature itself."""
    body = {k: v for k, v in d.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()


# --------------------------------------------------------------------------- #
# Certificate                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Certificate:
    """A signed assertion that ``issuer`` vouches for ``subject``'s pubkey.

    Layout (canonical JSON-serialisable):

        {
            "subject_pubkey" : "ed25519:<hex>",
            "subject_name"   : "alice@team",
            "issuer_pubkey"  : "ed25519:<hex>",
            "issuer_name"    : "qmesh-corp-ca",
            "not_before"     : "2026-01-01T00:00:00Z",
            "not_after"      : "2027-01-01T00:00:00Z",
            "serial"         : "<hex>",
            "signature"      : "<hex>",     # ed25519 by issuer over body
        }
    """

    subject_pubkey: str
    subject_name: str
    issuer_pubkey: str
    issuer_name: str
    not_before: str
    not_after: str
    serial: str
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_pubkey": self.subject_pubkey,
            "subject_name": self.subject_name,
            "issuer_pubkey": self.issuer_pubkey,
            "issuer_name": self.issuer_name,
            "not_before": self.not_before,
            "not_after": self.not_after,
            "serial": self.serial,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Certificate:
        return cls(
            subject_pubkey=str(d.get("subject_pubkey", "")),
            subject_name=str(d.get("subject_name", "")),
            issuer_pubkey=str(d.get("issuer_pubkey", "")),
            issuer_name=str(d.get("issuer_name", "")),
            not_before=str(d.get("not_before", "")),
            not_after=str(d.get("not_after", "")),
            serial=str(d.get("serial", "")),
            signature=str(d.get("signature", "")),
        )

    def fingerprint(self) -> str:
        """sha256 over the canonical body — handy for logs / dedup."""
        return sha256(_canonical_cert_body(self.to_dict())).hexdigest()

    def is_self_signed(self) -> bool:
        return self.subject_pubkey == self.issuer_pubkey

    def covers_instant(self, iso_ts: str) -> bool:
        """Return True if ``iso_ts`` falls within [not_before, not_after]."""
        if not iso_ts:
            return True   # caller didn't tell us; don't fail closed
        try:
            return self.not_before <= iso_ts <= self.not_after
        except Exception:
            return False


# --------------------------------------------------------------------------- #
# Issuance                                                                    #
# --------------------------------------------------------------------------- #


def _load_signer(private_key_pem: bytes) -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(   # type: ignore[return-value]
        private_key_pem, password=None,
    )


def issue_certificate(
    *,
    subject_pubkey: str,
    subject_name: str,
    issuer_private_key_pem: bytes,
    issuer_name: str,
    not_before: str | None = None,
    not_after: str | None = None,
    serial: str | None = None,
    validity_seconds: int = 365 * 24 * 3600,
) -> Certificate:
    """Issue (sign) a Certificate.

    The issuer is identified by the *PEM bytes* of an ed25519 private key —
    we deliberately don't take a file path so test fixtures and key-rotation
    scripts can pass freshly-generated keys without touching disk.

    Raises :class:`RuntimeError` if the cryptography library is unavailable.
    """
    if not _HAS_CRYPTO:
        raise RuntimeError(
            "qmesh.compliance.trust requires the `cryptography` package; "
            "install it with `pip install cryptography`."
        )

    issuer_priv = _load_signer(issuer_private_key_pem)
    issuer_pub_raw = issuer_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    issuer_pubkey = f"{_ED25519_PREFIX}{issuer_pub_raw.hex()}"

    now = time.time()
    nb = not_before or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    na = not_after or time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + validity_seconds),
    )
    ser = serial or secrets.token_hex(16)

    cert = Certificate(
        subject_pubkey=subject_pubkey,
        subject_name=subject_name,
        issuer_pubkey=issuer_pubkey,
        issuer_name=issuer_name,
        not_before=nb,
        not_after=na,
        serial=ser,
    )
    body = _canonical_cert_body(cert.to_dict())
    cert.signature = issuer_priv.sign(body).hex()
    return cert


def verify_certificate_signature(cert: Certificate) -> bool:
    """Re-verify the cert's ed25519 signature using the issuer pubkey embedded
    in the cert. Does not check trust path — only "is this signature good?"."""
    if not _HAS_CRYPTO:
        return False
    if not cert.signature or not cert.issuer_pubkey:
        return False
    issuer_hex = _strip_prefix(cert.issuer_pubkey)
    try:
        issuer_pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(issuer_hex))
    except Exception:
        return False
    body = _canonical_cert_body(cert.to_dict())
    try:
        issuer_pub.verify(bytes.fromhex(cert.signature), body)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Trust store + chain walk                                                    #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TrustStore:
    """Set of trusted root pubkeys (CAs).

    Roots are addressed by their full label — e.g. ``"ed25519:<hex>"``.
    """
    roots: dict[str, str] = field(default_factory=dict)

    def trust(self, pubkey: str, name: str) -> None:
        self.roots[pubkey] = name

    def is_root(self, pubkey: str) -> bool:
        return pubkey in self.roots

    def __bool__(self) -> bool:
        return bool(self.roots)


@dataclass(slots=True)
class TrustResult:
    valid: bool
    chain: list[Certificate] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "chain_len": len(self.chain),
            "chain": [c.to_dict() for c in self.chain],
        }


def verify_chain(
    leaf_pubkey: str,
    certs: Iterable[Certificate | dict[str, Any]],
    trust_store: TrustStore,
    *,
    instant: str | None = None,
) -> TrustResult:
    """Walk the cert graph from ``leaf_pubkey`` toward a trusted root.

    Algorithm:
      1. Build a map ``subject_pubkey -> Certificate`` from ``certs``.
      2. Start at ``leaf_pubkey``. If it's already a root, return valid.
      3. Otherwise, find the cert whose ``subject_pubkey == current``;
         verify its signature; check ``not_before`` / ``not_after`` against
         ``instant`` if provided; advance ``current = cert.issuer_pubkey``.
      4. Stop when ``current`` is a root, when no cert covers ``current``
         (fail), or when a cycle is detected (fail).
    """
    cert_objs: list[Certificate] = []
    for c in certs:
        cert_objs.append(c if isinstance(c, Certificate) else Certificate.from_dict(c))

    by_subject: dict[str, Certificate] = {}
    for c in cert_objs:
        by_subject.setdefault(c.subject_pubkey, c)

    if not trust_store:
        return TrustResult(False, [], "trust store is empty (no roots configured)")

    if trust_store.is_root(leaf_pubkey):
        # Direct trust: leaf is itself a root. Nothing to walk.
        return TrustResult(True, [], "leaf is a configured trust root")

    visited: set[str] = set()
    walked: list[Certificate] = []
    current = leaf_pubkey
    while True:
        if current in visited:
            return TrustResult(False, walked, f"cycle detected at {current}")
        visited.add(current)

        cert = by_subject.get(current)
        if cert is None:
            return TrustResult(
                False, walked,
                f"no certificate for subject {current!r} (chain incomplete)",
            )

        if not verify_certificate_signature(cert):
            return TrustResult(
                False, walked,
                f"certificate signature failed for subject {current!r}",
            )

        if instant and not cert.covers_instant(instant):
            return TrustResult(
                False, walked,
                f"certificate for {cert.subject_name!r} not valid at {instant} "
                f"(window {cert.not_before}..{cert.not_after})",
            )

        walked.append(cert)

        if trust_store.is_root(cert.issuer_pubkey):
            return TrustResult(True, walked, "chain reaches a configured root")

        # Step up.
        if cert.is_self_signed():
            # Self-signed but not in trust store → not trusted.
            return TrustResult(
                False, walked,
                f"self-signed cert for {cert.subject_name!r} not in trust store",
            )
        current = cert.issuer_pubkey


# --------------------------------------------------------------------------- #
# Conveniences for callers                                                    #
# --------------------------------------------------------------------------- #


def generate_root_keypair() -> tuple[bytes, str]:
    """Generate an ed25519 keypair for a trust root.

    Returns:
        (private_key_pem_bytes, public_key_label).
    """
    if not _HAS_CRYPTO:
        raise RuntimeError("cryptography library not available")
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    label = f"{_ED25519_PREFIX}{pub_raw.hex()}"
    return pem, label


def attach_cert_chain_to_manifest(
    manifest_data: dict[str, Any],
    chain: Iterable[Certificate | dict[str, Any]],
) -> None:
    """In-place: insert ``submitter.cert_chain`` so the cert chain is bundled
    with every manifest write. Idempotent — repeated calls overwrite."""
    submitter = manifest_data.setdefault("submitter", {})
    if not isinstance(submitter, dict):
        return
    submitter["cert_chain"] = [
        (c if isinstance(c, dict) else c.to_dict()) for c in chain
    ]


__all__ = [
    "Certificate",
    "TrustStore",
    "TrustResult",
    "issue_certificate",
    "verify_certificate_signature",
    "verify_chain",
    "generate_root_keypair",
    "attach_cert_chain_to_manifest",
]
