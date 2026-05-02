"""qmesh.api — top-level submit / replay / diff entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qmesh import __version__
from qmesh.backends.base import RunResult
from qmesh.backends.registry import get as get_backend
from qmesh.ir.module import Module
from qmesh.provenance.manifest import Manifest, ManifestSigner


def submit(
    module: Module,
    backend: str = "qmesh.statevec",
    shots: int = 1024,
    ledger_dir: Path | str = "ledger",
    sign: bool = True,
    **backend_kwargs: Any,
) -> tuple[RunResult, Manifest]:
    """Run `module` on `backend`, emit a signed provenance manifest.

    Returns (result, manifest). Manifest is also written to disk under
    `ledger_dir/<date>/<hash>.json`.
    """
    bk = get_backend(backend)
    ok, why = bk.accepts(module)
    if not ok:
        raise ValueError(f"backend {backend!r} rejected module: {why}")

    ir_hash = module.hash()
    manifest = Manifest.new(
        qmesh_version=__version__,
        ir_hash=ir_hash,
        ir_path=None,
        frontend={"name": "qmesh.builder", "version": __version__},
        backend={
            "name": bk.capabilities.name,
            "vendor": bk.capabilities.vendor,
            "is_simulator": bk.capabilities.is_simulator,
            "qubit_count": bk.capabilities.qubit_count,
            "fidelity_2q_typical": bk.capabilities.fidelity_2q_typical,
        },
    )

    result = bk.run(module, shots=shots, **backend_kwargs)

    manifest.execution = {
        "shots": result.shots,
        "wall_seconds": result.wall_seconds,
        "qpu_seconds": result.qpu_seconds,
        "cost_usd": result.cost_usd,
        "counts": result.counts,
        "backend_metadata": result.backend_metadata,
    }
    if sign:
        manifest = ManifestSigner().sign(manifest)

    out_dir = Path(ledger_dir) / manifest.submitted_at[:10]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest.hash()[:16]}.json"
    out_path.write_text(manifest.to_json())
    return result, manifest


def replay(manifest_path: str | Path) -> tuple[bool, str]:
    """Load a manifest, verify its signature, and report drift status.

    Real replay requires re-emitting the IR — left for Phase 1.
    """
    p = Path(manifest_path)
    data = json.loads(p.read_text())
    m = Manifest(**{k: v for k, v in data.items() if k in Manifest.__slots__})
    ok = ManifestSigner.verify(m)
    return ok, ("signature ok" if ok else "signature failed")


def diff(a_path: str | Path, b_path: str | Path) -> dict[str, Any]:
    """Compare two manifests; return field-level diff. Phase-1 stub."""
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    out: dict[str, Any] = {}
    for k in set(a) | set(b):
        if a.get(k) != b.get(k):
            out[k] = {"a": a.get(k), "b": b.get(k)}
    return out
