"""Phase 4α + 4β tests: DAG topology, executor, parallelism, cross-modality, resume."""

from __future__ import annotations

import json

import pytest

import qmesh
from qmesh.scheduler import (
    DAG,
    Barrier,
    BellPairClaim,
    ClassicalTask,
    DAGExecutor,
    EntanglementBarrier,
    Fanout,
    HPCQPUPrimitive,
    MockHPCConnector,
    PBSConnector,
    QPUPrimitive,
    SLURMConnector,
    execute,
    resume,
)
from qmesh.scheduler.dag import _new_id


# ---------- topology ----------

def test_toposort_respects_deps():
    dag = DAG()
    a = dag.add(ClassicalTask(id="a", fn=lambda c: {"v": 1}))
    b = dag.add(ClassicalTask(id="b", fn=lambda c: {"v": 2}, depends_on=["a"]))
    c = dag.add(ClassicalTask(id="c", fn=lambda c: {"v": 3}, depends_on=["a"]))
    d = dag.add(ClassicalTask(id="d", fn=lambda ctx: {"v": 4},
                              depends_on=["b", "c"]))
    order = [n.id for n in dag.toposort()]
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_cycle_detected():
    """Cycles can't be built via DAG.add (which validates deps exist),
    so to test toposort's cycle detector we install nodes directly."""
    dag = DAG()
    a = ClassicalTask(id="a", fn=lambda c: c, depends_on=["b"])
    b = ClassicalTask(id="b", fn=lambda c: c, depends_on=["a"])
    dag.nodes["a"] = a
    dag.nodes["b"] = b
    with pytest.raises(ValueError, match="cycle"):
        dag.toposort()


def test_add_validates_missing_dep():
    dag = DAG()
    with pytest.raises(ValueError):
        dag.add(ClassicalTask(id="x", fn=lambda c: c, depends_on=["nope"]))


# ---------- executor ----------

def test_executor_runs_classical_chain(tmp_path):
    dag = DAG()
    a = dag.add(ClassicalTask(id="a", fn=lambda c: {"v": 10}))
    b = dag.add(ClassicalTask(id="b",
                              fn=lambda ctx: {"v": ctx["a"]["v"] * 2},
                              depends_on=["a"]))
    run = execute(dag, ledger_dir=tmp_path)
    assert run.results[a.id].ok
    assert run.results[b.id].ok
    assert run.context[b.id] == {"v": 20}


def test_executor_classical_failure_recorded(tmp_path):
    def boom(ctx):
        raise RuntimeError("intentional")
    dag = DAG()
    a = dag.add(ClassicalTask(id="a", fn=boom))
    run = execute(dag, ledger_dir=tmp_path)
    assert not run.results[a.id].ok
    assert "intentional" in run.results[a.id].error


def test_qpu_node_emits_manifest(tmp_path):
    pytest.importorskip("qiskit_aer")
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    dag = DAG()
    n = dag.add(QPUPrimitive(name="prep", module=c.module,
                             backend="qmesh.aer", shots=512, sign=False))
    run = execute(dag, ledger_dir=tmp_path)
    assert run.results[n.id].ok
    assert run.results[n.id].manifest_hash is not None
    counts = run.context[n.id]["counts"]
    assert sum(counts.values()) == 512


def test_module_factory_consumes_upstream_context(tmp_path):
    pytest.importorskip("qiskit_aer")
    dag = DAG()
    derive = dag.add(ClassicalTask(
        id="derive", fn=lambda c: {"theta": 1.234},
    ))

    def factory(ctx):
        with qmesh.circuit("p", n_qubits=1, n_bits=1) as c:
            c.ry(ctx["derive"]["theta"], 0)
            c.measure(0, 0)
        return c.module

    n = dag.add(QPUPrimitive(
        name="p", module_factory=factory, backend="qmesh.aer",
        shots=512, sign=False, depends_on=["derive"],
    ))
    run = execute(dag, ledger_dir=tmp_path)
    assert run.results[n.id].ok
    assert sum(run.context[n.id]["counts"].values()) == 512


def test_aggregate_manifest_signed_and_persisted(tmp_path):
    dag = DAG()
    dag.add(ClassicalTask(id="a", fn=lambda c: {"v": 1}))
    run = execute(dag, ledger_dir=tmp_path)
    assert run.manifest_path.exists()
    manifest_data = json.loads(run.manifest_path.read_text())
    assert manifest_data["signature"]["alg"] == "ed25519"
    assert manifest_data["execution"]["run_id"] == run.run_id


def test_event_log_captures_transitions(tmp_path):
    dag = DAG()
    a = dag.add(ClassicalTask(id="a", fn=lambda c: {"v": 1}))
    b = dag.add(ClassicalTask(id="b", fn=lambda c: {"v": 2}, depends_on=["a"]))
    run = execute(dag, ledger_dir=tmp_path)
    events = [json.loads(line) for line in run.event_log_path.read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert "run_start" in kinds
    assert "level_start" in kinds
    assert "node_ok" in kinds
    assert "run_finish" in kinds


def test_parallel_qpu_nodes_inside_one_level(tmp_path):
    pytest.importorskip("qiskit_aer")
    # Build a fan-out DAG: 4 independent QPU nodes share no deps,
    # so the executor places them in level 0.
    dag = DAG()
    ids = []
    for i in range(4):
        with qmesh.circuit(f"c{i}", n_qubits=1, n_bits=1) as c:
            c.h(0); c.measure(0, 0)
        ids.append(dag.add(QPUPrimitive(
            id=f"n{i}", name=f"node{i}", module=c.module,
            backend="qmesh.aer", shots=256, sign=False,
        )).id)
    run = execute(dag, ledger_dir=tmp_path, max_workers=4)
    for nid in ids:
        assert run.results[nid].ok


# ---------- cross-modality ----------

def test_cross_modality_dag_runs(tmp_path):
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("pulser")
    pytest.importorskip("strawberryfields")

    from pulser import Pulse, Register, Sequence
    from pulser.devices import MockDevice
    import strawberryfields as sf
    from strawberryfields import ops as sf_ops
    from qmesh.frontends.pulser import from_pulser
    from qmesh.frontends.sf import from_sf

    dag = DAG()
    with qmesh.circuit("g", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    bell = dag.add(QPUPrimitive(
        id="bell", module=c.module, backend="qmesh.aer", shots=128, sign=False,
    ))

    def derive(ctx):
        return {"r": 0.3}

    derived = dag.add(ClassicalTask(id="der", fn=derive, depends_on=["bell"]))

    def cv_factory(ctx):
        prog = sf.Program(2)
        with prog.context as q:
            sf_ops.Sgate(ctx["der"]["r"]) | q[0]
            sf_ops.BSgate(0.5, 0) | (q[0], q[1])
            sf_ops.MeasureFock() | q[0]
            sf_ops.MeasureFock() | q[1]
        return from_sf(prog)

    cv = dag.add(QPUPrimitive(
        id="cv", module_factory=cv_factory, backend="qmesh.sf.gaussian",
        shots=20, sign=False, depends_on=["der"],
    ))

    run = execute(dag, ledger_dir=tmp_path)
    assert run.results[bell.id].ok
    assert run.results[derived.id].ok
    assert run.results[cv.id].ok


# ---------- fanout ----------

def test_fanout_runs_all_instances(tmp_path):
    def make(kw):
        return ClassicalTask(id=f"inst_{kw['i']}", fn=lambda c, kw=kw: {"i": kw["i"]})

    dag = DAG()
    fan = dag.add(Fanout(
        id="fan", template=make, fanout_kwargs=[{"i": 0}, {"i": 1}, {"i": 2}],
    ))
    run = execute(dag, ledger_dir=tmp_path)
    assert run.results[fan.id].ok
    assert len(run.context[fan.id]) == 3


# ---------- barrier ----------

def test_barrier_synchronises(tmp_path):
    dag = DAG()
    a = dag.add(ClassicalTask(id="a", fn=lambda c: {"v": 1}))
    b = dag.add(ClassicalTask(id="b", fn=lambda c: {"v": 2}))
    bar = dag.add(Barrier(id="bar", depends_on=["a", "b"]))
    last = dag.add(ClassicalTask(id="last",
                                 fn=lambda ctx: {"both": ctx["a"]["v"] + ctx["b"]["v"]},
                                 depends_on=["bar"]))
    run = execute(dag, ledger_dir=tmp_path)
    assert run.results[bar.id].ok
    assert run.context[last.id] == {"both": 3}


# ---------- Phase 4β: checkpoint resume ----------

def _build_three_node_dag(*, second_fails: bool):
    """a → b → c. b raises iff second_fails."""
    dag = DAG()
    dag.add(ClassicalTask(id="a", fn=lambda c: {"v": 10}))

    def b_fn(ctx):
        if second_fails:
            raise RuntimeError("boom-b")
        return {"v": ctx["a"]["v"] * 2}
    dag.add(ClassicalTask(id="b", fn=b_fn, depends_on=["a"]))
    dag.add(ClassicalTask(
        id="c",
        fn=lambda ctx: {"v": ctx["b"]["v"] + 1},
        depends_on=["b"],
    ))
    return dag


def test_node_results_are_persisted(tmp_path):
    dag = DAG()
    dag.add(ClassicalTask(id="a", fn=lambda c: {"v": 7}))
    run = execute(dag, ledger_dir=tmp_path)
    snap = run.manifest_path.parent / "node_results" / "a.json"
    assert snap.exists()
    data = json.loads(snap.read_text())
    assert data["node_id"] == "a"
    assert data["ok"] is True
    assert data["output"] == {"v": 7}


def test_resume_after_partial_failure(tmp_path):
    # 1) First run: middle node fails. a is durable; b/c never landed.
    dag = _build_three_node_dag(second_fails=True)
    first = execute(dag, ledger_dir=tmp_path)
    assert first.results["a"].ok
    assert not first.results["b"].ok
    assert not first.results["c"].ok        # never got to run; classical-failure cascade
    original_dir = first.manifest_path.parent

    # 2) Build a fresh DAG with the bug fixed and the same node IDs.
    fixed = _build_three_node_dag(second_fails=False)
    resumed = resume(fixed, original_dir, ledger_dir=tmp_path)

    # a is skipped; b and c re-run successfully.
    assert resumed.results["a"].ok
    assert resumed.results["b"].ok
    assert resumed.results["c"].ok
    assert resumed.context["c"] == {"v": 21}
    # The new manifest references the original through a first-class lineage block.
    manifest = json.loads(resumed.manifest_path.read_text())
    assert manifest["lineage"]["parent_run_id"] == first.run_id
    assert manifest["lineage"]["n_skipped_resume"] == 1
    # The parent_manifest_hash is reproducible from the on-disk parent manifest.
    parent_data = json.loads(first.manifest_path.read_text())
    body = {k: v for k, v in parent_data.items() if k != "signature"}
    import hashlib
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   default=str).encode()
    ).hexdigest()
    assert manifest["lineage"]["parent_manifest_hash"] == expected
    # The parent's ed25519 signature value is forwarded for offline auditing.
    assert manifest["lineage"]["parent_signature_alg"] == "ed25519"
    assert manifest["lineage"]["parent_signature_value"]


def test_resume_emits_skip_events_and_marks_skipped(tmp_path):
    dag = _build_three_node_dag(second_fails=True)
    first = execute(dag, ledger_dir=tmp_path)
    fixed = _build_three_node_dag(second_fails=False)
    resumed = resume(fixed, first.manifest_path.parent, ledger_dir=tmp_path)

    events = [
        json.loads(line) for line in resumed.event_log_path.read_text().splitlines()
    ]
    skipped_events = [e for e in events if e["event"] == "node_skipped_resume"]
    assert {e["node_id"] for e in skipped_events} == {"a"}

    manifest = json.loads(resumed.manifest_path.read_text())
    by_id = {r["id"]: r for r in manifest["execution"]["node_results"]}
    assert by_id["a"]["skipped"] is True
    assert by_id["b"]["skipped"] is False
    assert by_id["c"]["skipped"] is False


def test_resume_of_fully_completed_run_is_idempotent(tmp_path):
    """If everything already succeeded, resume reruns nothing."""
    counter = {"a": 0, "b": 0}

    def make_dag():
        d = DAG()
        d.add(ClassicalTask(id="a", fn=lambda c: (counter.__setitem__("a", counter["a"] + 1), {"v": 1})[1]))
        d.add(ClassicalTask(
            id="b", fn=lambda c: (counter.__setitem__("b", counter["b"] + 1), {"v": c["a"]["v"] + 1})[1],
            depends_on=["a"],
        ))
        return d

    first = execute(make_dag(), ledger_dir=tmp_path)
    assert counter == {"a": 1, "b": 1}
    resumed = resume(make_dag(), first.manifest_path.parent, ledger_dir=tmp_path)
    # Counter stays at 1,1 — the bodies didn't fire on the resumed run.
    assert counter == {"a": 1, "b": 1}
    # But the manifest still references both nodes as skipped.
    manifest = json.loads(resumed.manifest_path.read_text())
    assert manifest["lineage"]["n_skipped_resume"] == 2
    assert manifest["execution"]["n_failed"] == 0
    assert resumed.context["a"] == {"v": 1}
    assert resumed.context["b"] == {"v": 2}


def test_resume_rejects_dag_shape_mismatch(tmp_path):
    first = execute(_build_three_node_dag(second_fails=True), ledger_dir=tmp_path)
    bad = DAG()
    bad.add(ClassicalTask(id="a", fn=lambda c: {"v": 1}))
    bad.add(ClassicalTask(id="x", fn=lambda c: {"v": 2}, depends_on=["a"]))
    with pytest.raises(ValueError, match="don't match"):
        resume(bad, first.manifest_path.parent, ledger_dir=tmp_path)


def test_resume_info_reports_skip_and_rerun(tmp_path):
    first = execute(_build_three_node_dag(second_fails=True), ledger_dir=tmp_path)
    info = DAGExecutor(ledger_dir=tmp_path).resume_info(
        _build_three_node_dag(second_fails=False),
        first.manifest_path.parent,
    )
    assert info["parent_run_id"] == first.run_id
    assert info["skip"] == ["a"]
    assert sorted(info["rerun"]) == ["b", "c"]
    assert info["n_total"] == 3
    assert info["n_skip"] == 1
    assert info["n_rerun"] == 2


def test_resume_fanout_skips_completed_children(tmp_path):
    """Fanout-resume: each child instance is persisted with a deterministic
    `{parent_id}::child_{i}` ID. On resume, children that completed on the
    first run are skipped — only failed children re-execute. This is the
    bug the audit flagged: previously the parent Fanout was treated atomically,
    so any sibling failure forced ALL children to re-run."""
    fail_indices = {0, 2}     # children 0 and 2 fail on the first run
    fire_count = {"calls": []}

    def make_dag():
        d = DAG()

        def template(kw):
            i = kw["i"]
            should_fail = i in fail_indices

            def fn(ctx, _i=i, _should_fail=should_fail):
                fire_count["calls"].append(_i)
                if _should_fail:
                    raise RuntimeError(f"child-{_i} bug")
                return {"i": _i, "v": _i * 10}
            return ClassicalTask(fn=fn)

        d.add(Fanout(
            id="fan", template=template,
            fanout_kwargs=[{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}],
        ))
        return d

    first = execute(make_dag(), ledger_dir=tmp_path)
    assert not first.results["fan"].ok
    # All four children fired exactly once on the first run.
    assert sorted(fire_count["calls"]) == [0, 1, 2, 3]

    # Surgically mark the failing children as fixed.
    fail_indices.clear()
    fire_count["calls"] = []

    resumed = resume(make_dag(), first.manifest_path.parent, ledger_dir=tmp_path)
    # Only children 0 and 2 should re-fire — 1 and 3 are loaded from disk.
    assert sorted(fire_count["calls"]) == [0, 2]
    assert resumed.results["fan"].ok
    assert [r["i"] for r in resumed.results["fan"].output] == [0, 1, 2, 3]
    # Each child has a deterministic stable ID.
    assert resumed.results["fan"].metadata["child_ids"] == [
        "fan::child_0", "fan::child_1", "fan::child_2", "fan::child_3",
    ]
    assert resumed.results["fan"].metadata["n_skipped_resume"] == 2


def test_resume_info_reports_fanout_child_skips(tmp_path):
    fail_indices = {1}

    def make_dag():
        d = DAG()

        def template(kw):
            i = kw["i"]
            def fn(ctx, _i=i):
                if _i in fail_indices:
                    raise RuntimeError(f"child-{_i} bug")
                return {"i": _i}
            return ClassicalTask(fn=fn)

        d.add(Fanout(
            id="fan", template=template,
            fanout_kwargs=[{"i": 0}, {"i": 1}, {"i": 2}],
        ))
        return d

    first = execute(make_dag(), ledger_dir=tmp_path)
    info = DAGExecutor(ledger_dir=tmp_path).resume_info(
        make_dag(), first.manifest_path.parent,
    )
    # Top-level fan stays in rerun (one child still failed); two children skip.
    assert info["rerun"] == ["fan"]
    assert info["child_skip"] == ["fan::child_0", "fan::child_2"]
    assert info["n_child_skip"] == 2


def test_resume_qpu_node_skips_without_re_submitting(tmp_path):
    """A QPU node that succeeded on the first run is not re-submitted on
    resume. We prove this by (a) making the downstream classical node
    fail the first time so resume actually triggers, (b) checking the
    counter — if bell were re-run, the QPU shot count would double."""
    pytest.importorskip("qiskit_aer")
    fail_agg = {"flag": True}
    agg_runs = {"n": 0}

    def make_dag():
        d = DAG()
        with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
            c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
        d.add(QPUPrimitive(
            id="bell", module=c.module, backend="qmesh.aer",
            shots=128, sign=False,
        ))

        def agg_fn(ctx):
            agg_runs["n"] += 1
            if fail_agg["flag"]:
                raise RuntimeError("agg-bug")
            return {"shots": ctx["bell"]["shots"]}

        d.add(ClassicalTask(id="agg", fn=agg_fn, depends_on=["bell"]))
        return d

    first = execute(make_dag(), ledger_dir=tmp_path)
    assert first.results["bell"].ok
    assert not first.results["agg"].ok
    assert agg_runs["n"] == 1

    # Fix the bug and resume.
    fail_agg["flag"] = False
    resumed = resume(make_dag(), first.manifest_path.parent, ledger_dir=tmp_path)
    # bell skipped: agg ran once more, but bell did NOT (no extra QPU submit).
    assert agg_runs["n"] == 2
    assert resumed.results["bell"].ok
    assert resumed.results["agg"].ok
    assert resumed.context["agg"] == {"shots": 128}

    manifest = json.loads(resumed.manifest_path.read_text())
    by_id = {r["id"]: r for r in manifest["execution"]["node_results"]}
    assert by_id["bell"]["skipped"] is True
    assert by_id["agg"]["skipped"] is False
    # The skipped QPU node still carries its original manifest_hash through.
    assert by_id["bell"]["manifest_hash"] == first.results["bell"].manifest_hash


# ---------- Phase 4β: HPC connectors ----------

def _bell_module():
    with qmesh.circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    return c.module


def test_mock_hpc_connector_runs_qpu_node(tmp_path):
    pytest.importorskip("qiskit_aer")
    connector = MockHPCConnector(work_dir=tmp_path / "hpc")
    dag = DAG()
    n = dag.add(HPCQPUPrimitive(
        id="hpc1", name="bell-on-hpc", module=_bell_module(),
        backend="qmesh.aer", shots=256, sign=False,
        connector=connector, partition="gpu", walltime="00:10:00",
    ))
    run = execute(dag, ledger_dir=tmp_path / "dag")
    nr = run.results[n.id]
    assert nr.ok, nr.error
    # The hpc block lives in node metadata + on the inner output, and it
    # is reflected in the aggregate manifest's execution.node_results entry.
    assert nr.metadata["hpc"]["scheduler"] == "mock"
    assert nr.metadata["hpc"]["partition"] == "gpu"
    assert nr.metadata["hpc"]["walltime_request"] == "00:10:00"
    assert nr.metadata["hpc"]["job_id"].startswith("mock-")
    assert nr.metadata["hpc"]["inner_manifest_hash"] is not None
    counts = nr.output["counts"]
    assert sum(counts.values()) == 256
    # Inner manifest is signed-by-default (sign=False here only to keep
    # the test fast — flip to True to verify the path).
    inner_manifest = nr.output["hpc"]["inner_manifest_hash"]
    assert isinstance(inner_manifest, str) and len(inner_manifest) == 64
    # Aggregate DAG run manifest carries the hpc block on its node entry.
    agg = json.loads(run.manifest_path.read_text())
    by_id = {r["id"]: r for r in agg["execution"]["node_results"]}
    assert by_id[n.id]["metadata"]["hpc"]["scheduler"] == "mock"


def test_slurm_connector_renders_sbatch_script(tmp_path):
    connector = SLURMConnector(work_dir=tmp_path / "slurm")
    submit_dir = tmp_path / "slurm" / "fake"
    out_dir = submit_dir / "out"
    out_dir.mkdir(parents=True)
    script = connector.render_script(
        module=_bell_module(), backend="qmesh.aer", shots=512, sign=True,
        partition="gpu-debug", walltime="01:30:00", nodes=2, cpus_per_task=4,
        extra_directives=["#SBATCH --gres=gpu:1"],
        submit_dir=submit_dir, out_dir=out_dir,
    )
    # SLURM directives, time, partition, gres extra, qmesh entrypoint.
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH --partition=gpu-debug" in script
    assert "#SBATCH --time=01:30:00" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "from qmesh.api import submit" in script
    assert "qmesh.aer" in script
    # Submit command is `sbatch <script>`, job-id parser handles standard reply.
    assert connector._submit_cmd(submit_dir / "submit.sh")[0] == "sbatch"
    assert connector._parse_job_id("Submitted batch job 987654\n") == "987654"
    assert connector._is_complete("") is True
    assert connector._is_complete("12345 RUNNING") is False


def test_pbs_connector_renders_qsub_script(tmp_path):
    connector = PBSConnector(work_dir=tmp_path / "pbs")
    submit_dir = tmp_path / "pbs" / "fake"
    out_dir = submit_dir / "out"
    out_dir.mkdir(parents=True)
    script = connector.render_script(
        module=_bell_module(), backend="qmesh.aer", shots=128, sign=False,
        partition="batch", walltime="00:20:00", nodes=1, cpus_per_task=8,
        extra_directives=["#PBS -A QMESH_ALLOC"],
        submit_dir=submit_dir, out_dir=out_dir,
    )
    assert script.startswith("#!/bin/bash")
    assert "#PBS -q batch" in script
    assert "#PBS -l walltime=00:20:00" in script
    assert "#PBS -l select=1:ncpus=8" in script
    assert "#PBS -A QMESH_ALLOC" in script
    assert "from qmesh.api import submit" in script
    assert connector._submit_cmd(submit_dir / "submit.sh")[0] == "qsub"
    # Typical qsub stdout is just the job id.
    assert connector._parse_job_id("12345.head.cluster\n") == "12345.head.cluster"
    assert connector._is_complete("Unknown Job Id 12345") is True


# ---------- Phase 4β: entanglement-aware barriers ----------

def test_entanglement_barrier_waits_for_bell_pairs(tmp_path):
    dag = DAG()
    dag.add(BellPairClaim(
        id="claim_a", photonic_link="link-A",
        producer_node_id="qpu_a",
    ))
    dag.add(BellPairClaim(
        id="claim_b", photonic_link="link-A",
        producer_node_id="qpu_b",
    ))
    bar = dag.add(EntanglementBarrier(
        id="bar", photonic_link="link-A",
        expected_bell_pairs=2, timeout_seconds=2.0,
        depends_on=["claim_a", "claim_b"],
    ))
    dag.add(ClassicalTask(
        id="downstream",
        fn=lambda ctx: {
            "got": ctx["bar"]["claims_received"],
            "link": ctx["bar"]["photonic_link"],
        },
        depends_on=["bar"],
    ))
    run = execute(dag, ledger_dir=tmp_path)
    assert run.results[bar.id].ok
    assert run.context[bar.id]["claims_received"] == 2
    assert run.context["downstream"] == {"got": 2, "link": "link-A"}
    # Both claims are visible in the run context under the link name.
    assert len(run.context["bell_pairs"]["link-A"]) == 2


def test_entanglement_barrier_times_out_when_pairs_missing(tmp_path):
    dag = DAG()
    dag.add(BellPairClaim(
        id="claim_a", photonic_link="link-A", producer_node_id="qpu_a",
    ))
    bar = dag.add(EntanglementBarrier(
        id="bar", photonic_link="link-A",
        expected_bell_pairs=2, timeout_seconds=0.2,
        poll_interval_seconds=0.02,
        depends_on=["claim_a"],
    ))
    run = execute(dag, ledger_dir=tmp_path)
    nr = run.results[bar.id]
    assert not nr.ok
    assert "TimeoutError" in nr.error
    assert "1 of 2" in nr.error
    assert "link-A" in nr.error
