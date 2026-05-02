"""qmesh.service.app — FastAPI app factory."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from qmesh import __version__
from qmesh.api import submit as submit_cmd
from qmesh.backends.registry import all_backends
from qmesh.frontends.qasm3 import parse as qasm_parse
from qmesh.provenance.manifest import ManifestSigner
from qmesh.router import Objective, choose


LEDGER_DIR = Path(os.getenv("QMESH_LEDGER_DIR", "ledger"))


class SubmitRequest(BaseModel):
    qasm: str
    backend: str = "auto"
    shots: int = 1024
    budget_usd: float = 1.0
    min_fidelity: float = 0.0


class SubmitResponse(BaseModel):
    counts: dict[str, int]
    shots: int
    wall_seconds: float
    chosen_backend: str
    manifest_hash: str
    manifest_path: str


def create_app() -> FastAPI:
    app = FastAPI(title="qmesh", version=__version__,
                  description="2026-native quantum operating layer")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__,
                "backends_registered": len(all_backends())}

    @app.get("/backends")
    def list_backends() -> list[dict]:
        out = []
        for bk in all_backends().values():
            c = bk.capabilities
            out.append({
                "name": c.name, "vendor": c.vendor,
                "modalities": [m.value for m in c.modalities],
                "qubit_count": c.qubit_count,
                "is_simulator": c.is_simulator,
                "fidelity_2q_typical": c.fidelity_2q_typical,
                "cost_per_shot_usd": c.cost_per_shot_usd,
                "notes": c.notes,
            })
        return out

    @app.post("/submit", response_model=SubmitResponse)
    def submit_endpoint(req: SubmitRequest) -> SubmitResponse:
        try:
            module = qasm_parse(req.qasm)
        except Exception as e:
            raise HTTPException(400, f"failed to parse QASM: {e}") from e

        if req.backend == "auto":
            try:
                bk, _ = choose(module,
                               Objective(budget_usd=req.budget_usd, min_fidelity=req.min_fidelity),
                               shots=req.shots)
                backend_name = bk.capabilities.name
            except RuntimeError as e:
                raise HTTPException(400, f"router could not choose: {e}") from e
        else:
            backend_name = req.backend

        try:
            result, manifest = submit_cmd(module, backend=backend_name,
                                          shots=req.shots, ledger_dir=LEDGER_DIR)
        except Exception as e:
            raise HTTPException(500, f"submission failed: {e}") from e

        manifest_path = (LEDGER_DIR / manifest.submitted_at[:10] /
                         f"{manifest.hash()[:16]}.json")
        return SubmitResponse(
            counts=result.counts, shots=result.shots,
            wall_seconds=result.wall_seconds,
            chosen_backend=backend_name,
            manifest_hash=manifest.hash(),
            manifest_path=str(manifest_path),
        )

    @app.get("/manifests/{partial_hash}")
    def get_manifest(partial_hash: str) -> dict:
        for p in LEDGER_DIR.rglob("*.json"):
            if p.stem.startswith(partial_hash):
                return json.loads(p.read_text())
        raise HTTPException(404, f"no manifest with hash starting '{partial_hash}'")

    @app.get("/manifests/{partial_hash}/verify")
    def verify_manifest(partial_hash: str) -> dict:
        from qmesh.provenance.manifest import Manifest
        for p in LEDGER_DIR.rglob("*.json"):
            if p.stem.startswith(partial_hash):
                data = json.loads(p.read_text())
                m = Manifest(**{k: v for k, v in data.items() if k in Manifest.__slots__})
                ok = ManifestSigner.verify(m)
                return {"verified": ok, "path": str(p)}
        raise HTTPException(404, f"no manifest with hash starting '{partial_hash}'")

    @app.get("/dag_runs/{run_id}")
    def get_dag_run(run_id: str) -> dict:
        """Return the aggregate manifest of a DAG run."""
        for p in (LEDGER_DIR.parent / "dag").rglob(f"*{run_id}*/dag_run_manifest.json"):
            return json.loads(p.read_text())
        for p in LEDGER_DIR.rglob(f"*{run_id}*/dag_run_manifest.json"):
            return json.loads(p.read_text())
        raise HTTPException(404, f"no DAG run with id {run_id!r}")

    return app


app = create_app()
