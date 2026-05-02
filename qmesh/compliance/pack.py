"""qmesh.compliance.pack — build & verify tamper-evident manifest archives.

A compliance pack bundles an immutable run-window of signed qmesh manifests
into a single tar.gz with a chained-hash audit trail. Auditors can re-verify
the pack offline using only public keys embedded in the manifests.

Design choices (α-honest):

* **Skip-on-unsigned, don't fail.** Unsigned manifests are recorded in the
  chain with `signature_alg: "unsigned"` and a warning entry — they break
  signature-counting but do *not* break chain integrity. Compliance teams
  who require strict signing can reject any pack with `n_signatures_ok <
  n_manifests`.
* **Canonical JSON drives hashes.** The same `_canonical_json` used by the
  Manifest class is used here, so a pack built from on-disk JSON and a pack
  built from in-memory `Manifest` objects hash identically.
* **Chain step.** `this_hash = sha256(prev_hash || manifest_hash ||
  signature_value)` — git-style. `prev_hash` for entry 0 is the literal
  string "GENESIS".
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from qmesh import __version__
from qmesh.provenance.manifest import _canonical_json

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


logger = logging.getLogger(__name__)


GENESIS = "GENESIS"


# --------------------------------------------------------------------------- #
# Verification result                                                         #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ComplianceVerification:
    chain_valid: bool
    n_manifests: int
    n_signatures_ok: int
    broken_at: int | None = None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Phase 6β — populated only when verify_pack() is given a TrustStore.
    # `trust_chain_valid` is True iff every signed manifest's signer pubkey
    # chains to a configured root via embedded cert chain. When the caller
    # passes no trust store, this stays None (audit was not requested).
    trust_chain_valid: bool | None = None
    n_trusted: int = 0
    untrusted_signers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_valid": self.chain_valid,
            "n_manifests": self.n_manifests,
            "n_signatures_ok": self.n_signatures_ok,
            "broken_at": self.broken_at,
            "errors": list(self.errors),
            "notes": list(self.notes),
            "trust_chain_valid": self.trust_chain_valid,
            "n_trusted": self.n_trusted,
            "untrusted_signers": list(self.untrusted_signers),
        }


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _iso_to_date(s: str | None) -> str | None:
    if not s:
        return None
    return s[:10]


def _within_window(
    submitted_at: str | None,
    since: str | None,
    until: str | None,
) -> bool:
    if submitted_at is None:
        return True
    d = _iso_to_date(submitted_at) or ""
    if since and d < since:
        return False
    if until and d > until:
        return False
    return True


def _manifest_payload_hash(data: dict[str, Any]) -> str:
    """Compute the manifest hash the way Manifest.hash() does — minus signature."""
    d = {k: v for k, v in data.items() if k != "signature"}
    return sha256(_canonical_json(d)).hexdigest()


def _chain_step(prev_hash: str, manifest_hash: str, signature_value: str) -> str:
    body = (prev_hash + manifest_hash + (signature_value or "")).encode()
    return sha256(body).hexdigest()


def _gather_manifests(
    ledger_dir: Path,
    *,
    since: str | None,
    until: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    if not ledger_dir.exists():
        return out
    for p in sorted(ledger_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            logger.warning("compliance.pack: skip unreadable %s (%s)", p, e)
            continue
        # Heuristic: a real qmesh manifest carries circuit + qmesh_version.
        if "qmesh_version" not in data or "circuit" not in data:
            continue
        if not _within_window(data.get("submitted_at"), since, until):
            continue
        out.append((p, data))
    # Sort by submitted_at primarily, hash secondarily (deterministic).
    out.sort(key=lambda t: (
        t[1].get("submitted_at", ""),
        _manifest_payload_hash(t[1]),
    ))
    return out


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #


_README_TEMPLATE = """\
qmesh compliance pack
=====================

This archive bundles {n} signed qmesh manifests in a tamper-evident chain.

Layout
------
- manifests/<index>_<hash>.json : the original signed manifest bytes
- signatures.json               : map index -> {{alg, value, public_key}}
- chain.json                    : ordered chain of (manifest_hash, prev_hash, this_hash)

Verify offline (Python):

    from qmesh.compliance import verify_pack
    r = verify_pack("{archive_name}")
    assert r.chain_valid and r.n_signatures_ok == r.n_manifests

Verify offline (CLI):

    qmesh compliance-verify {archive_name}

Chain rule
----------
    this_hash[0] = sha256("GENESIS" || manifest_hash[0] || signature_value[0])
    this_hash[i] = sha256(this_hash[i-1] || manifest_hash[i] || signature_value[i])

Tamper detection
----------------
Editing any manifest changes its `manifest_hash`, which changes its
`this_hash`, which changes every subsequent `this_hash` in the chain.
verify_pack reports `broken_at` = the first index where the recomputed
`this_hash` no longer matches the stored value.

Generated by qmesh {qmesh_version} at {created_at}.
"""


def build_pack(
    ledger_dir: str | Path,
    *,
    since: str | None = None,
    until: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Build a compliance pack archive.

    Args:
        ledger_dir: directory containing one or more qmesh manifest JSON files
            (recursive walk; non-manifest .json files are skipped).
        since: ISO date 'YYYY-MM-DD'; manifests with `submitted_at` lexicographically
            earlier than this are excluded. None = no lower bound.
        until: same shape, inclusive upper bound. None = no upper bound.
        out_path: destination archive path (.tar.gz). Defaults to
            `ledger_dir/compliance_pack_<window>.tar.gz`.

    Returns:
        Path to the written archive.
    """
    ledger = Path(ledger_dir)
    items = _gather_manifests(ledger, since=since, until=until)

    if out_path is None:
        win = f"{since or 'beginning'}_{until or 'now'}"
        out_path = ledger / f"compliance_pack_{win}.tar.gz"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    chain: list[dict[str, Any]] = []
    sigs: dict[str, dict[str, str]] = {}
    notes: list[str] = []

    prev_hash = GENESIS
    with tarfile.open(out, "w:gz") as tar:
        for i, (orig_path, data) in enumerate(items):
            mhash = _manifest_payload_hash(data)
            sig = data.get("signature") or {}
            sig_alg = sig.get("alg", "unsigned")
            sig_val = sig.get("value", "")
            pubkey = (data.get("submitter") or {}).get("public_key", "")
            if sig_alg in ("unsigned", ""):
                notes.append(
                    f"index {i} ({orig_path.name}) is unsigned — "
                    f"included but cannot contribute to signature audit"
                )

            arc_name = f"manifests/{i:03d}_{mhash[:16]}.json"
            # Re-emit canonical JSON so the archive is byte-stable across hosts.
            payload = json.dumps(
                data, sort_keys=True, indent=2, default=str,
            ).encode()
            ti = tarfile.TarInfo(name=arc_name)
            ti.size = len(payload)
            ti.mtime = int(time.time())
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(payload))

            this_hash = _chain_step(prev_hash, mhash, sig_val)
            entry = {
                "index": i,
                "manifest_path": arc_name,
                "manifest_hash": mhash,
                "signature_alg": sig_alg,
                "signature_value": sig_val,
                "prev_hash": prev_hash,
                "this_hash": this_hash,
            }
            chain.append(entry)
            sigs[str(i)] = {
                "alg": sig_alg,
                "value": sig_val,
                "public_key": pubkey,
            }
            prev_hash = this_hash

        # signatures.json
        sigs_payload = json.dumps(sigs, sort_keys=True, indent=2).encode()
        ti = tarfile.TarInfo("signatures.json")
        ti.size = len(sigs_payload)
        ti.mtime = int(time.time())
        ti.mode = 0o644
        tar.addfile(ti, io.BytesIO(sigs_payload))

        # chain.json
        chain_doc = {
            "qmesh_version": __version__,
            "n_manifests": len(items),
            "since": since,
            "until": until,
            "notes": notes,
            "entries": chain,
        }
        chain_payload = json.dumps(chain_doc, sort_keys=True, indent=2).encode()
        ti = tarfile.TarInfo("chain.json")
        ti.size = len(chain_payload)
        ti.mtime = int(time.time())
        ti.mode = 0o644
        tar.addfile(ti, io.BytesIO(chain_payload))

        # README.txt
        readme_payload = _README_TEMPLATE.format(
            n=len(items),
            archive_name=out.name,
            qmesh_version=__version__,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ).encode()
        ti = tarfile.TarInfo("README.txt")
        ti.size = len(readme_payload)
        ti.mtime = int(time.time())
        ti.mode = 0o644
        tar.addfile(ti, io.BytesIO(readme_payload))

    return out


# --------------------------------------------------------------------------- #
# Verify                                                                      #
# --------------------------------------------------------------------------- #


def _verify_signature_against_pubkey(
    manifest_data: dict[str, Any],
) -> bool:
    """Re-verify ed25519 signature using the public key embedded in the manifest.
    Returns False on any failure (including missing key/library)."""
    if not _HAS_CRYPTO:
        return False
    sig = manifest_data.get("signature") or {}
    if sig.get("alg") != "ed25519":
        return False
    pub_hex = ((manifest_data.get("submitter") or {})
               .get("public_key", "")
               .removeprefix("ed25519:"))
    if not pub_hex:
        return False
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
    except Exception:
        return False

    body = {k: v for k, v in manifest_data.items() if k != "signature"}
    try:
        pub.verify(
            bytes.fromhex(sig["value"]),
            _canonical_json(body),
        )
        return True
    except Exception:
        return False


def verify_pack(
    archive_path: str | Path,
    *,
    trust_roots: "Any | None" = None,
) -> ComplianceVerification:
    """Re-walk the chain in an archive and re-verify every signature.

    No private key is needed — only the public key embedded in each
    manifest's `submitter.public_key` field. Operates fully offline.

    Args:
        archive_path: path to the ``.tar.gz`` produced by :func:`build_pack`.
        trust_roots: optional :class:`qmesh.compliance.trust.TrustStore`. When
            supplied, every signed manifest's signer pubkey is chained
            (via ``submitter.cert_chain`` embedded in the manifest body)
            up to a configured root. The result populates ``trust_chain_valid``,
            ``n_trusted`` and ``untrusted_signers``. When ``None`` the
            trust audit is skipped and those fields stay at defaults.
    """
    from qmesh.compliance.trust import TrustStore, verify_chain as _verify_chain
    arc = Path(archive_path)
    errors: list[str] = []
    notes: list[str] = []

    if not arc.exists():
        return ComplianceVerification(
            chain_valid=False, n_manifests=0, n_signatures_ok=0,
            broken_at=0, errors=[f"archive not found: {arc}"],
        )

    try:
        tar = tarfile.open(arc, "r:gz")
    except Exception as e:
        return ComplianceVerification(
            chain_valid=False, n_manifests=0, n_signatures_ok=0,
            broken_at=0, errors=[f"cannot open archive: {e}"],
        )

    with tar:
        # Read chain.json
        try:
            chain_member = tar.getmember("chain.json")
            chain_doc = json.loads(tar.extractfile(chain_member).read())
        except Exception as e:
            return ComplianceVerification(
                chain_valid=False, n_manifests=0, n_signatures_ok=0,
                broken_at=0, errors=[f"missing/unreadable chain.json: {e}"],
            )
        notes.extend(chain_doc.get("notes") or [])
        entries = chain_doc.get("entries", [])
        n_manifests = len(entries)

        prev_hash = GENESIS
        broken_at: int | None = None
        n_sig_ok = 0
        n_trusted = 0
        untrusted_signers: list[str] = []

        for i, entry in enumerate(entries):
            idx = entry.get("index", i)
            mpath = entry.get("manifest_path")
            stored_mhash = entry.get("manifest_hash")
            stored_sigval = entry.get("signature_value", "")
            stored_this = entry.get("this_hash")
            stored_prev = entry.get("prev_hash")

            # Check declared prev_hash matches what we just computed.
            if stored_prev != prev_hash and broken_at is None:
                broken_at = i
                errors.append(
                    f"chain step {i}: prev_hash declared "
                    f"{stored_prev!r} but expected {prev_hash!r}"
                )

            # Fetch manifest bytes and re-derive its hash.
            try:
                m_member = tar.getmember(mpath)
                m_bytes = tar.extractfile(m_member).read()
                m_data = json.loads(m_bytes)
            except Exception as e:
                if broken_at is None:
                    broken_at = i
                errors.append(f"chain step {i}: cannot read {mpath}: {e}")
                # We still need to keep the chain walking so subsequent
                # entries' `prev_hash` checks compose with the broken state.
                fresh_mhash = ""
                m_data = None
            else:
                fresh_mhash = _manifest_payload_hash(m_data)

            if fresh_mhash != stored_mhash:
                if broken_at is None:
                    broken_at = i
                errors.append(
                    f"chain step {i}: manifest_hash mismatch "
                    f"(stored {stored_mhash}, recomputed {fresh_mhash})"
                )

            # Re-verify ed25519 signature using the embedded public key.
            sig_alg = entry.get("signature_alg", "unsigned")
            if m_data is not None and sig_alg == "ed25519":
                if _verify_signature_against_pubkey(m_data):
                    n_sig_ok += 1
                else:
                    errors.append(
                        f"chain step {i}: signature failed re-verification"
                    )

                # Optional cert-chain trust audit (Phase 6β).
                if trust_roots is not None:
                    submitter = (m_data.get("submitter") or {})
                    signer_pub = submitter.get("public_key", "")
                    cert_chain = submitter.get("cert_chain") or []
                    submitted_at = m_data.get("submitted_at")
                    tr = _verify_chain(
                        signer_pub, cert_chain, trust_roots,
                        instant=submitted_at,
                    )
                    if tr.valid:
                        n_trusted += 1
                    else:
                        untrusted_signers.append(
                            f"index {i} ({signer_pub[:24]}…): {tr.reason}"
                        )

            # Recompute this_hash and compare.
            recomputed = _chain_step(prev_hash, fresh_mhash, stored_sigval)
            if recomputed != stored_this:
                if broken_at is None:
                    broken_at = i
                errors.append(
                    f"chain step {i}: this_hash mismatch "
                    f"(stored {stored_this}, recomputed {recomputed})"
                )
            prev_hash = stored_this  # follow the stored chain forward

    chain_valid = broken_at is None
    if trust_roots is not None:
        # Trusted iff every signed manifest chained to a root.
        trust_chain_valid: bool | None = (
            n_sig_ok > 0 and n_trusted == n_sig_ok and not untrusted_signers
        )
    else:
        trust_chain_valid = None
    return ComplianceVerification(
        chain_valid=chain_valid,
        n_manifests=n_manifests,
        n_signatures_ok=n_sig_ok,
        broken_at=broken_at,
        errors=errors,
        notes=notes,
        trust_chain_valid=trust_chain_valid,
        n_trusted=n_trusted,
        untrusted_signers=untrusted_signers,
    )
