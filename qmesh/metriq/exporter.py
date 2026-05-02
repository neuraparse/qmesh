"""qmesh.metriq.exporter — manifest -> Metriq submission JSON.

Schema target (Metriq alpha, May 2026):

    Submission:
        name        : str           # short title
        repo        : str | null    # source repo URL
        commit      : str | null    # git commit SHA (if known)
        tags        : list[str]
        results     : list[Result]

    Result:
        task_id        : str        # circuit / experiment identifier
        method_name    : str        # qmesh + backend identity
        metric_name    : str        # e.g. "logical_error_rate", "fidelity"
        metric_value   : float
        metric_unit    : str
        sample_size    : int        # shots
        evaluatedAt    : str        # ISO-8601 UTC
        evidence       : dict       # manifest hash + signature + qmesh_version

Notes on α-honesty
------------------
* The real Metriq endpoint may evolve. We pick the field names that match
  the public ``metriq-client`` schema (snake_case for results, camelCase
  for ``evaluatedAt`` to mirror the upstream JSON). Adapters can rename if
  needed without changing how qmesh produces the data.
* We extract one ``Result`` row per metric we can faithfully read out of a
  manifest. For execution manifests that's typically a fidelity or success
  probability (best-effort from `counts`); for ftmode manifests it's the
  logical error rate (a much stronger signal, the headline FT-benchmark
  metric).
* Live posts go through `submit_to_metriq` with `dry_run=False`; tests use
  the dry-run path so they pass without network access.
"""

from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from qmesh import __version__
from qmesh.provenance.manifest import _canonical_json

# --------------------------------------------------------------------------- #
# Loading helpers                                                             #
# --------------------------------------------------------------------------- #


def _load_manifest_dict(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _manifest_hash_from_dict(data: dict[str, Any]) -> str:
    """Re-derive the manifest hash from on-disk JSON (excluding signature)."""
    d = {k: v for k, v in data.items() if k != "signature"}
    return sha256(_canonical_json(d)).hexdigest()


def _signature_value(data: dict[str, Any]) -> str:
    sig = data.get("signature") or {}
    return sig.get("value", "")


def _signature_alg(data: dict[str, Any]) -> str:
    sig = data.get("signature") or {}
    return sig.get("alg", "unsigned")


def _public_key(data: dict[str, Any]) -> str:
    return (data.get("submitter") or {}).get("public_key", "")


# --------------------------------------------------------------------------- #
# Metric extraction                                                           #
# --------------------------------------------------------------------------- #


def _ftmode_metrics(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull metrics out of an ftmode manifest."""
    out: list[dict[str, Any]] = []
    ft = data.get("ftmode") or {}
    exe = data.get("execution") or {}

    if "logical_error_rate" in exe:
        out.append({
            "metric_name": "logical_error_rate",
            "metric_value": float(exe["logical_error_rate"]),
            "metric_unit": "fraction",
        })
    if "logical_error_count" in exe:
        out.append({
            "metric_name": "logical_error_count",
            "metric_value": float(exe["logical_error_count"]),
            "metric_unit": "count",
        })
    re_block = ft.get("resource_estimate") or {}
    if "estimated_logical_error_rate" in re_block:
        out.append({
            "metric_name": "estimated_logical_error_rate",
            "metric_value": float(re_block["estimated_logical_error_rate"]),
            "metric_unit": "fraction",
        })
    if "physical_qubits" in re_block:
        out.append({
            "metric_name": "physical_qubits_required",
            "metric_value": float(re_block["physical_qubits"]),
            "metric_unit": "count",
        })
    return out


def _execution_metrics(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull headline metrics out of a non-FT execution manifest."""
    out: list[dict[str, Any]] = []
    exe = data.get("execution") or {}

    counts = exe.get("counts") or {}
    if counts:
        total = sum(counts.values())
        if total > 0:
            # Pick the most-frequent bitstring as a fidelity proxy.
            top_bs, top_n = max(counts.items(), key=lambda kv: kv[1])
            out.append({
                "metric_name": "top_bitstring_fraction",
                "metric_value": top_n / total,
                "metric_unit": "fraction",
                "qualifier": top_bs,
            })
    if "wall_seconds" in exe:
        out.append({
            "metric_name": "wall_time",
            "metric_value": float(exe["wall_seconds"]),
            "metric_unit": "seconds",
        })
    return out


def _shots_for(data: dict[str, Any]) -> int:
    exe = data.get("execution") or {}
    return int(exe.get("shots", 0) or 0)


def _task_id_for(data: dict[str, Any]) -> str:
    circ = data.get("circuit") or {}
    h = circ.get("ir_hash_sha256", "unknown")
    return h


def _method_for(data: dict[str, Any]) -> str:
    bk = data.get("backend") or {}
    fr = data.get("frontend") or {}
    return f"{fr.get('name','qmesh')}@{fr.get('version','?')} -> {bk.get('name','?')}"


def _evidence_for(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_hash": _manifest_hash_from_dict(data),
        "signature_alg": _signature_alg(data),
        "signature_value": _signature_value(data),
        "public_key": _public_key(data),
        "qmesh_version": data.get("qmesh_version", __version__),
        "submitted_at": data.get("submitted_at"),
    }


def _results_from_manifest_dict(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("ftmode"):
        metrics = _ftmode_metrics(data)
    else:
        metrics = _execution_metrics(data)

    task_id = _task_id_for(data)
    method = _method_for(data)
    evidence = _evidence_for(data)
    shots = _shots_for(data)
    evaluated_at = data.get("submitted_at") or time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )

    rows: list[dict[str, Any]] = []
    for m in metrics:
        row = {
            "task_id": task_id,
            "method_name": method,
            "metric_name": m["metric_name"],
            "metric_value": m["metric_value"],
            "metric_unit": m["metric_unit"],
            "sample_size": shots,
            "evaluatedAt": evaluated_at,
            "evidence": dict(evidence),
        }
        if "qualifier" in m:
            row["evidence"]["qualifier"] = m["qualifier"]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def metriq_submission_from_manifest(
    manifest_path: str | Path,
    *,
    name: str | None = None,
    repo: str | None = None,
    commit: str | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a Metriq submission dict around a single manifest."""
    p = Path(manifest_path)
    data = _load_manifest_dict(p)
    rows = _results_from_manifest_dict(data)
    bk = (data.get("backend") or {}).get("name", "?")
    sub: dict[str, Any] = {
        "name": name or f"qmesh run {data.get('submitted_at','?')} on {bk}",
        "repo": repo,
        "commit": commit,
        "tags": list(tags or _default_tags(data)),
        "results": rows,
    }
    return sub


def export_manifest(
    manifest_path: str | Path,
    *,
    name: str | None = None,
    repo: str | None = None,
    commit: str | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Export a single manifest as a Metriq submission dict."""
    return metriq_submission_from_manifest(
        manifest_path, name=name, repo=repo, commit=commit, tags=tags,
    )


def export_run_dir(
    run_dir: str | Path,
    *,
    name: str | None = None,
    repo: str | None = None,
    commit: str | None = None,
    tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Walk a ledger dir, gather all manifest .json files, fold into one
    Metriq submission with combined results[]."""
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"run dir not found: {root}")

    rows: list[dict[str, Any]] = []
    n = 0
    if root.is_file() and root.suffix == ".json":
        rows.extend(_results_from_manifest_dict(_load_manifest_dict(root)))
        n = 1
    else:
        for p in sorted(root.rglob("*.json")):
            try:
                data = _load_manifest_dict(p)
            except Exception:
                continue
            # Heuristic: real qmesh manifests carry circuit + qmesh_version.
            if "qmesh_version" not in data or "circuit" not in data:
                continue
            rows.extend(_results_from_manifest_dict(data))
            n += 1

    return {
        "name": name or f"qmesh ledger export ({root.name}, {n} manifests)",
        "repo": repo,
        "commit": commit,
        "tags": list(tags or ["qmesh", "ledger-export"]),
        "results": rows,
    }


def _default_tags(data: dict[str, Any]) -> list[str]:
    tags = ["qmesh", f"qmesh-{data.get('qmesh_version','?')}"]
    if data.get("ftmode"):
        tags.append("ftmode")
        ft = data["ftmode"]
        code = ft.get("code") or {}
        if "name" in code:
            tags.append(code["name"])
    bk = (data.get("backend") or {}).get("name")
    if bk:
        tags.append(f"backend:{bk}")
    return tags


def submit_to_metriq(
    submission: dict[str, Any],
    endpoint: str = "https://metriq.info/api/submission",
    *,
    dry_run: bool = True,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Post a Metriq submission to an HTTP endpoint.

    Tests pass `dry_run=True` so this never makes a network call: we return
    a ``{"dry_run": True, "endpoint": ..., "payload": submission}`` echo.

    Live mode lazy-imports `requests` and posts the JSON; the response body
    is returned unchanged. We do not try to handle every possible error mode
    of the Metriq endpoint here — the caller is expected to inspect
    `status_code`/`error` in the returned dict and decide how to react.
    """
    if dry_run:
        return {
            "dry_run": True,
            "endpoint": endpoint,
            "payload": submission,
            "n_results": len(submission.get("results", [])),
        }

    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "live submit_to_metriq requires `requests`: "
            "`pip install requests` or call with dry_run=True."
        ) from e

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(  # noqa: S113 — timeout passed explicitly
        endpoint, json=submission, headers=headers, timeout=timeout,
    )
    out: dict[str, Any] = {
        "dry_run": False,
        "endpoint": endpoint,
        "status_code": resp.status_code,
        "n_results": len(submission.get("results", [])),
    }
    try:
        out["response"] = resp.json()
    except ValueError:
        out["response_text"] = resp.text
    return out
