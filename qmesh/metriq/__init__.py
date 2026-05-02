"""qmesh.metriq — translate qmesh manifests into Metriq benchmark-result rows.

Metriq (Unitary Foundation, https://metriq.info) catalogs reproducible
quantum-benchmark submissions. qmesh manifests already carry the canonical
fields a Metriq submission needs (circuit hash, backend identity, signed
evidence) — the exporter just remaps them to Metriq's submission/result
schema.

Public API:
    export_manifest(manifest_path)  -> dict   # Metriq submission JSON
    export_run_dir(run_dir)         -> dict   # walks a ledger dir
    submit_to_metriq(submission, endpoint, dry_run=True) -> dict

`dry_run=True` (default for tests) returns the payload without making a
network call. Live submission imports `requests` lazily.
"""

from __future__ import annotations

from qmesh.metriq.exporter import (
    export_manifest,
    export_run_dir,
    submit_to_metriq,
    metriq_submission_from_manifest,
)

__all__ = [
    "export_manifest",
    "export_run_dir",
    "submit_to_metriq",
    "metriq_submission_from_manifest",
]
