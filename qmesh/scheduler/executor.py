"""qmesh.scheduler.executor — execute a DAG with checkpointing + resume.

Topological execution. Independent nodes (no shared deps) run in parallel
across a thread pool. ClassicalTask runs inline; QPUPrimitive submits
through `qmesh.api.submit`; Fanout expands then runs in parallel; Barrier
is a no-op once all deps complete.

Provenance: every executed node either produces a `manifest_hash` (QPU
nodes) or contributes its summary to an aggregate `dag_run_manifest.json`
file in the ledger. A separate `events.ndjson` event log captures every
state transition. Each NodeResult is also persisted as
`node_results/{node_id}.json` so a crashed run can be resumed via
`DAGExecutor.resume(dag, original_run_dir)` — successful nodes are
skipped, failed/incomplete nodes re-execute, and the new manifest
references `parent_run_id` for an unbroken audit chain (Phase 4β).
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qmesh import __version__
from qmesh.api import submit as _submit
from qmesh.provenance.manifest import Manifest, ManifestSigner
from qmesh.scheduler.dag import (
    DAG,
    Barrier,
    BellPairClaim,
    ClassicalTask,
    EntanglementBarrier,
    Fanout,
    MCMRegion,
    Node,
    NodeResult,
    QPUPrimitive,
)
from qmesh.scheduler.hpc import HPCQPUPrimitive


@dataclass(slots=True)
class DAGRun:
    """Aggregate result of a DAG execution."""

    run_id: str
    started_at: str
    finished_at: str
    duration_seconds: float
    results: dict[str, NodeResult]
    context: dict[str, Any]
    manifest_hash: str
    event_log_path: Path
    manifest_path: Path


class DAGExecutor:
    """Topological executor with optional thread parallelism."""

    def __init__(
        self,
        *,
        ledger_dir: Path | str = "ledger/dag",
        max_workers: int = 4,
        sign: bool = True,
    ) -> None:
        self.ledger_dir = Path(ledger_dir)
        self.max_workers = max_workers
        self.sign = sign

    def run(self, dag: DAG, *, context: dict[str, Any] | None = None) -> DAGRun:
        return self._run(dag, context=context, prefilled={}, lineage=None)

    def resume(
        self,
        dag: DAG,
        original_run_dir: Path | str,
        *,
        context: dict[str, Any] | None = None,
    ) -> DAGRun:
        """Resume a previously failed or partial DAG run.

        The caller supplies the same DAG (callables can't be re-hydrated
        from JSON; only node IDs are checked for shape match). Nodes whose
        persisted NodeResult shows ok=True are skipped — their outputs are
        re-injected into the new run's context and recorded in the new
        manifest as `skipped: true`. Failed and never-started nodes
        re-execute. The new manifest carries a first-class `lineage` block
        with `parent_run_id`, `parent_manifest_hash`, and the parent's
        signature value so a verifier can walk the chain back unbroken.
        """
        original_run_dir = Path(original_run_dir)
        if not original_run_dir.exists():
            raise FileNotFoundError(f"original run dir not found: {original_run_dir}")

        prefilled = self._load_prefilled_results(original_run_dir)
        lineage = self._read_lineage(original_run_dir)
        self._validate_dag_against_original(dag, original_run_dir)

        return self._run(
            dag, context=context, prefilled=prefilled, lineage=lineage,
        )

    def resume_info(self, dag: DAG, original_run_dir: Path | str) -> dict[str, Any]:
        """Inspect a run dir and report which nodes would be skipped vs re-run."""
        original_run_dir = Path(original_run_dir)
        prefilled = self._load_prefilled_results(original_run_dir)
        lineage = self._read_lineage(original_run_dir) or {}
        # Filter to top-level nodes for the human-facing skip/rerun split.
        top_level_skip = sorted(nid for nid in prefilled if nid in dag.nodes)
        rerun = sorted(nid for nid in dag.nodes if nid not in prefilled)
        # Children (e.g. Fanout::child_*) skipped on resume — informational.
        child_skip = sorted(nid for nid in prefilled if nid not in dag.nodes)
        return {
            "parent_run_id": lineage.get("parent_run_id"),
            "parent_manifest_hash": lineage.get("parent_manifest_hash"),
            "original_run_dir": str(original_run_dir),
            "skip": top_level_skip,
            "rerun": rerun,
            "child_skip": child_skip,
            "n_total": len(dag.nodes),
            "n_skip": len(top_level_skip),
            "n_rerun": len(rerun),
            "n_child_skip": len(child_skip),
        }

    # ---------- internal ----------

    def _run(
        self,
        dag: DAG,
        *,
        context: dict[str, Any] | None,
        prefilled: dict[str, NodeResult],
        lineage: dict[str, Any] | None,
    ) -> DAGRun:
        run_id = uuid.uuid4().hex[:12]
        started = time.time()
        started_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
        run_dir = self.ledger_dir / started_iso[:10] / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        event_log = run_dir / "events.ndjson"
        results_dir = run_dir / "node_results"
        results_dir.mkdir(exist_ok=True)

        # Persist the DAG spec FIRST — a crash at level 0 still leaves a
        # readable run dir for resume_info / dag-status.
        (run_dir / "dag.json").write_text(
            json.dumps(dag.to_dict(), indent=2, default=str)
        )

        ctx = dict(context) if context else {}
        results: dict[str, NodeResult] = {}
        parent_run_id = (lineage or {}).get("parent_run_id")

        # Seed prefilled (resumed) results so downstream nodes see them in ctx.
        # Only top-level node IDs (those in dag.nodes) belong in `results` — the
        # full prefilled map (including Fanout child IDs) flows through to
        # _execute_node.
        for nid, nr in prefilled.items():
            if nid in dag.nodes:
                results[nid] = nr
                ctx[nid] = nr.output

        emit_lock = threading.Lock()

        def emit(event: dict) -> None:
            event = {"ts": time.time(), **event}
            line = json.dumps(event, default=str) + "\n"
            with emit_lock, event_log.open("a") as f:
                f.write(line)

        emit({
            "event": "run_start", "run_id": run_id,
            "dag": dag.to_dict(), "qmesh_version": __version__,
            "parent_run_id": parent_run_id,
            "prefilled_node_ids": sorted(results.keys()),
        })

        try:
            order = dag.toposort()
        except ValueError as e:
            emit({"event": "topology_error", "error": str(e)})
            raise

        levels = self._level_groups(order)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for level_idx, level in enumerate(levels):
                pending = [n for n in level if n.id not in results]
                skipped_ids = [n.id for n in level if n.id in results]
                emit({"event": "level_start", "level": level_idx,
                      "node_ids": [n.id for n in level],
                      "skipped_resume": skipped_ids})
                for sid in skipped_ids:
                    emit({"event": "node_skipped_resume", "node_id": sid,
                          "kind": results[sid].kind,
                          "manifest_hash": results[sid].manifest_hash})
                futures: dict[str, Future[NodeResult]] = {}
                for node in pending:
                    futures[node.id] = pool.submit(
                        self._execute_node, node, ctx, dag, run_dir, prefilled,
                    )
                for node in pending:
                    nr = futures[node.id].result()
                    results[node.id] = nr
                    self._persist_node_result(results_dir, nr)
                    if nr.ok:
                        ctx[node.id] = nr.output
                        emit({"event": "node_ok", "node_id": node.id,
                              "kind": nr.kind, "duration_s": nr.duration_seconds,
                              "manifest_hash": nr.manifest_hash})
                    else:
                        emit({"event": "node_failed", "node_id": node.id,
                              "kind": nr.kind, "error": nr.error})
                emit({"event": "level_end", "level": level_idx})

        finished = time.time()
        finished_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished))

        agg = Manifest.new(
            qmesh_version=__version__,
            ir_hash=f"dag-{run_id}",
            ir_path=str(run_dir / "dag.json"),
            frontend={"name": "qmesh.scheduler", "version": __version__},
            backend={"name": "qmesh.scheduler.dag", "vendor": "qmesh",
                     "is_simulator": True, "qubit_count": 0,
                     "fidelity_2q_typical": 1.0},
        )
        n_top_level_skipped = sum(1 for nid in results if nid in prefilled)
        agg.execution = {
            "run_id": run_id,
            "duration_seconds": finished - started,
            "n_nodes": len(results),
            "n_failed": sum(1 for r in results.values() if not r.ok),
            "node_results": [
                {
                    "id": r.node_id, "kind": r.kind, "ok": r.ok,
                    "duration_s": r.duration_seconds,
                    "manifest_hash": r.manifest_hash,
                    "metadata": r.metadata,
                    "skipped": r.node_id in prefilled,
                }
                for r in results.values()
            ],
        }
        if lineage is not None:
            # Carry n_skipped_resume that reflects what actually got skipped
            # at the top level of this DAG (children may also have been
            # skipped — see Fanout branch in _execute_node).
            agg.lineage = {**lineage, "n_skipped_resume": n_top_level_skipped}

        if self.sign:
            agg = ManifestSigner().sign(agg)
        manifest_path = run_dir / "dag_run_manifest.json"
        manifest_path.write_text(agg.to_json())
        emit({"event": "run_finish", "manifest_hash": agg.hash(),
              "parent_run_id": parent_run_id})

        return DAGRun(
            run_id=run_id,
            started_at=started_iso,
            finished_at=finished_iso,
            duration_seconds=finished - started,
            results=results,
            context=ctx,
            manifest_hash=agg.hash(),
            event_log_path=event_log,
            manifest_path=manifest_path,
        )

    @staticmethod
    def _persist_node_result(results_dir: Path, nr: NodeResult) -> None:
        """Snapshot a NodeResult to disk so resume() can pick it up later."""
        payload = {
            "node_id": nr.node_id,
            "kind": nr.kind,
            "ok": nr.ok,
            "output": nr.output,
            "duration_seconds": nr.duration_seconds,
            "metadata": nr.metadata,
            "error": nr.error,
            "manifest_hash": nr.manifest_hash,
        }
        (results_dir / f"{nr.node_id}.json").write_text(
            json.dumps(payload, default=str)
        )

    @staticmethod
    def _load_prefilled_results(original_run_dir: Path) -> dict[str, NodeResult]:
        """Load only ok=True NodeResults from a prior run dir."""
        results_dir = original_run_dir / "node_results"
        if not results_dir.exists():
            return {}
        prefilled: dict[str, NodeResult] = {}
        for jf in sorted(results_dir.glob("*.json")):
            data = json.loads(jf.read_text())
            if not data.get("ok"):
                continue
            prefilled[data["node_id"]] = NodeResult(
                node_id=data["node_id"],
                kind=data["kind"],
                ok=data["ok"],
                output=data.get("output"),
                duration_seconds=data.get("duration_seconds", 0.0),
                metadata=data.get("metadata", {}) or {},
                error=data.get("error"),
                manifest_hash=data.get("manifest_hash"),
            )
        return prefilled

    @staticmethod
    def _read_lineage(original_run_dir: Path) -> dict[str, Any] | None:
        """Build the lineage block from a parent run's manifest.

        Captures parent_run_id, parent_manifest_hash, and the parent's
        ed25519 signature value so an auditor can detect parent-run
        substitution (the file pointed at by the new manifest's
        `parent_run_id` must hash to `parent_manifest_hash`).
        """
        manifest_path = original_run_dir / "dag_run_manifest.json"
        if not manifest_path.exists():
            return None
        data = json.loads(manifest_path.read_text())
        execution = data.get("execution", {})
        signature = data.get("signature") or {}
        # Recompute the parent's content hash from the on-disk manifest:
        # this is what `Manifest.hash()` would produce on the same object.
        body = {k: v for k, v in data.items() if k != "signature"}
        from hashlib import sha256
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                                default=str).encode()
        parent_hash = sha256(canonical).hexdigest()
        return {
            "parent_run_id": execution.get("run_id"),
            "parent_manifest_hash": parent_hash,
            "parent_manifest_path": str(manifest_path),
            "parent_signature_alg": signature.get("alg"),
            "parent_signature_value": signature.get("value"),
        }

    @staticmethod
    def _validate_dag_against_original(dag: DAG, original_run_dir: Path) -> None:
        original_dag_path = original_run_dir / "dag.json"
        if not original_dag_path.exists():
            return  # nothing to compare against; trust the caller
        original_spec = json.loads(original_dag_path.read_text())
        original_ids = {n["id"] for n in original_spec.get("nodes", [])}
        current_ids = set(dag.nodes.keys())
        if current_ids != original_ids:
            missing = original_ids - current_ids
            extra = current_ids - original_ids
            raise ValueError(
                f"DAG node IDs don't match the original run "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )

    # ---------- helpers ----------

    @staticmethod
    def _level_groups(order: list[Node]) -> list[list[Node]]:
        """Group nodes by execution level (max depth in DAG)."""
        depth: dict[str, int] = {}
        levels: list[list[Node]] = []
        for n in order:
            d = 0
            for dep in n.depends_on:
                d = max(d, depth[dep] + 1)
            depth[n.id] = d
            while len(levels) <= d:
                levels.append([])
            levels[d].append(n)
        return levels

    def _execute_node(
        self,
        node: Node,
        ctx: dict[str, Any],
        dag: DAG,
        run_dir: Path,
        prefilled: dict[str, NodeResult] | None = None,
    ) -> NodeResult:
        prefilled = prefilled or {}
        t0 = time.time()
        try:
            if isinstance(node, ClassicalTask):
                output = node.fn(ctx) if node.fn else {}  # type: ignore[misc]
                return NodeResult(
                    node_id=node.id, kind="classical", ok=True,
                    output=output, duration_seconds=time.time() - t0,
                )

            if isinstance(node, QPUPrimitive):
                module = (
                    node.module if node.module is not None
                    else node.module_factory(ctx)  # type: ignore[misc]
                )
                if node.backend == "auto":
                    from qmesh.router import Objective, choose
                    obj = node.objective or Objective()
                    bk, _ = choose(module, obj, shots=node.shots)
                    backend_name = bk.capabilities.name
                else:
                    backend_name = node.backend

                result, manifest = _submit(
                    module, backend=backend_name, shots=node.shots,
                    ledger_dir=run_dir, sign=node.sign,
                )
                return NodeResult(
                    node_id=node.id, kind="qpu", ok=True,
                    output={"counts": result.counts, "shots": result.shots},
                    duration_seconds=time.time() - t0,
                    manifest_hash=manifest.hash(),
                    metadata={"backend": backend_name,
                              "wall_seconds": result.wall_seconds},
                )

            if isinstance(node, Fanout):
                # Deterministic child IDs so resume can match instance-by-instance:
                # `{parent_id}::child_{i}` is independent of what the template
                # generates for `id`.
                results_dir = run_dir / "node_results"
                results_dir.mkdir(exist_ok=True)
                child_results: dict[int, NodeResult] = {}
                pending_pairs: list[tuple[int, Node]] = []
                for i, kw in enumerate(node.fanout_kwargs):
                    child_id = f"{node.id}::child_{i}"
                    cached = prefilled.get(child_id)
                    if cached is not None and cached.ok:
                        child_results[i] = cached
                        continue
                    inst = node.template(kw)  # type: ignore[misc]
                    inst.id = child_id
                    inst.depends_on = []  # gathered by Fanout itself
                    pending_pairs.append((i, inst))

                if pending_pairs:
                    with ThreadPoolExecutor(
                        max_workers=min(self.max_workers, max(1, len(pending_pairs)))
                    ) as pool:
                        idx_futures = [
                            (i, pool.submit(
                                self._execute_node, inst, ctx, dag, run_dir, prefilled,
                            ))
                            for i, inst in pending_pairs
                        ]
                        for i, fut in idx_futures:
                            nr = fut.result()
                            child_results[i] = nr
                            self._persist_node_result(results_dir, nr)

                # Order matches fanout_kwargs.
                ordered = [child_results[i] for i in range(len(node.fanout_kwargs))]
                n_skipped = sum(1 for i in range(len(node.fanout_kwargs))
                                if f"{node.id}::child_{i}" in prefilled)
                return NodeResult(
                    node_id=node.id, kind="fanout",
                    ok=all(r.ok for r in ordered),
                    output=[r.output for r in ordered],
                    duration_seconds=time.time() - t0,
                    metadata={
                        "n_instances": len(node.fanout_kwargs),
                        "n_skipped_resume": n_skipped,
                        "child_ids": [f"{node.id}::child_{i}"
                                      for i in range(len(node.fanout_kwargs))],
                    },
                )

            if isinstance(node, MCMRegion):
                # Phase 4α: passthrough to backend that supports if_test
                from qmesh.backends.registry import get
                bk = get(node.backend)
                if not bk.capabilities.measurement_feedforward:
                    raise ValueError(
                        f"MCMRegion requires backend with measurement_feedforward; "
                        f"{node.backend} doesn't support it"
                    )
                if node.module is None:
                    raise ValueError("MCMRegion requires `module`")
                result, manifest = _submit(
                    node.module, backend=node.backend, shots=node.shots,
                    ledger_dir=run_dir, sign=True,
                )
                return NodeResult(
                    node_id=node.id, kind="mcm", ok=True,
                    output={"counts": result.counts, "shots": result.shots},
                    duration_seconds=time.time() - t0,
                    manifest_hash=manifest.hash(),
                    metadata={"feedforward_latency_budget_ns":
                              node.feedforward_latency_budget_ns},
                )

            # -------- Phase 4β: HPC + entanglement (BEGIN) --------

            if isinstance(node, HPCQPUPrimitive):
                module = (
                    node.module if node.module is not None
                    else node.module_factory(ctx)  # type: ignore[misc]
                )
                hpc_t0 = time.time()
                hpc_result = node.connector.run(  # type: ignore[union-attr]
                    module=module,
                    backend=node.backend,
                    shots=node.shots,
                    sign=node.sign,
                    partition=node.partition,
                    walltime=node.walltime,
                    nodes=node.nodes,
                    cpus_per_task=node.cpus_per_task,
                    extra_directives=list(node.extra_sbatch),
                )
                walltime_actual = time.time() - hpc_t0
                hpc_block = {
                    "scheduler": hpc_result.scheduler,
                    "job_id": hpc_result.job_id,
                    "partition": hpc_result.partition,
                    "walltime_request": hpc_result.walltime_request,
                    "walltime_actual": walltime_actual,
                    "nodes": node.nodes,
                    "cpus_per_task": node.cpus_per_task,
                    "out_dir": str(hpc_result.out_dir),
                    "submit_dir": str(hpc_result.submit_dir),
                    "inner_manifest_hash": hpc_result.inner_manifest_hash,
                    "backend": hpc_result.backend_name,
                }
                return NodeResult(
                    node_id=node.id, kind="hpc_qpu", ok=True,
                    output={
                        "counts": hpc_result.counts,
                        "shots": hpc_result.shots,
                        "hpc": hpc_block,
                    },
                    duration_seconds=time.time() - t0,
                    manifest_hash=hpc_result.inner_manifest_hash,
                    metadata={"hpc": hpc_block,
                              "backend": hpc_result.backend_name,
                              "walltime_actual": walltime_actual},
                )

            if isinstance(node, EntanglementBarrier):
                # Wait for `expected_bell_pairs` claims on this photonic_link.
                deadline = t0 + node.timeout_seconds
                got = 0
                while time.time() < deadline:
                    bucket = (ctx.get("bell_pairs") or {}).get(
                        node.photonic_link, []
                    )
                    got = len(bucket)
                    if got >= node.expected_bell_pairs:
                        break
                    time.sleep(node.poll_interval_seconds)
                else:
                    bucket = (ctx.get("bell_pairs") or {}).get(
                        node.photonic_link, []
                    )
                    got = len(bucket)
                wait_seconds = time.time() - t0
                if got < node.expected_bell_pairs:
                    raise TimeoutError(
                        f"EntanglementBarrier {node.id!r}: only {got} of "
                        f"{node.expected_bell_pairs} Bell-pair claims arrived "
                        f"on link {node.photonic_link!r} within "
                        f"{node.timeout_seconds:.1f}s"
                    )
                return NodeResult(
                    node_id=node.id, kind="entanglement_barrier", ok=True,
                    output={
                        "photonic_link": node.photonic_link,
                        "claims_received": got,
                        "expected_bell_pairs": node.expected_bell_pairs,
                        "wait_seconds": wait_seconds,
                    },
                    duration_seconds=time.time() - t0,
                    metadata={
                        "photonic_link": node.photonic_link,
                        "expected_bell_pairs": node.expected_bell_pairs,
                        "claims_received": got,
                        "wait_seconds": wait_seconds,
                    },
                )

            # -------- Phase 4β: HPC + entanglement (END) --------

            if isinstance(node, Barrier):
                return NodeResult(
                    node_id=node.id, kind="barrier", ok=True,
                    output=None, duration_seconds=time.time() - t0,
                )

            raise NotImplementedError(f"unknown node kind: {type(node).__name__}")

        except Exception as e:
            return NodeResult(
                node_id=node.id, kind=getattr(node, "kind", lambda: "?")(),
                ok=False, output=None, duration_seconds=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
                metadata={"traceback": traceback.format_exc()[:1024]},
            )


def execute(dag: DAG, *, context: dict[str, Any] | None = None,
            ledger_dir: Path | str = "ledger/dag",
            max_workers: int = 4, sign: bool = True) -> DAGRun:
    """Run a DAG end-to-end. See `DAGExecutor` for tuning knobs."""
    return DAGExecutor(ledger_dir=ledger_dir, max_workers=max_workers,
                       sign=sign).run(dag, context=context)


def resume(
    dag: DAG,
    original_run_dir: Path | str,
    *,
    context: dict[str, Any] | None = None,
    ledger_dir: Path | str = "ledger/dag",
    max_workers: int = 4,
    sign: bool = True,
) -> DAGRun:
    """Resume a partial DAG run. See `DAGExecutor.resume` for semantics."""
    return DAGExecutor(ledger_dir=ledger_dir, max_workers=max_workers,
                       sign=sign).resume(dag, original_run_dir, context=context)


__all__ = ["DAGExecutor", "DAGRun", "execute", "resume"]
