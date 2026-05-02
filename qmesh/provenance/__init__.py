"""qmesh.provenance — signed manifest + replay ledger.

The manifest is the central artifact. Every run emits one. Replay any past run
from a manifest; diff results across versions; export Metriq-compatible rows.
"""

from __future__ import annotations

from qmesh.provenance.manifest import Manifest, ManifestSigner

__all__ = ["Manifest", "ManifestSigner"]
