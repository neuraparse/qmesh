"""Phase 4β: HPC-dispatched QPU nodes + entanglement-aware Barrier.

We build a DAG that mirrors the IonQ photonic-interconnect / Cisco
Universal Quantum Switch flow shipping in 2026:

  qpu_a (HPC-dispatched Bell prep) ─┐
                                    ├─→ EntanglementBarrier(link-A, expect=2)
  qpu_b (HPC-dispatched Bell prep) ─┘                  │
                                                       ↓
                                              classical aggregator

Each HPC node is wrapped through a `MockHPCConnector` (CI has no real
SLURM/PBS) — the connector renders a real sbatch-style script for audit,
then runs the IR locally. Each HPC node's downstream `BellPairClaim`
helper publishes a Bell-pair claim on `link-A`. The
`EntanglementBarrier` waits for both claims before the aggregator runs.

The aggregate run manifest carries:
  - one `hpc` block per HPCQPUPrimitive (scheduler / job_id / partition /
    walltime_request / walltime_actual / inner_manifest_hash);
  - one `entanglement_barrier` node entry with claims_received + wait
    duration;
  - the usual signed ed25519 envelope.

Run:
    PYTHONPATH=. python3 examples/dag_hpc.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rich import print
from rich.panel import Panel
from rich.table import Table

import qmesh
from qmesh.ir.builder import circuit
from qmesh.scheduler import (
    DAG,
    BellPairClaim,
    ClassicalTask,
    EntanglementBarrier,
    HPCQPUPrimitive,
    MockHPCConnector,
    execute,
)


LEDGER = Path("ledger/dag_hpc_demo")


def _bell_module(label: str):
    with circuit(label, n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    return c.module


def main() -> None:
    print(Panel.fit(
        "[bold]Phase 4β: HPC connectors + entanglement-aware Barrier[/]\n\n"
        "Two HPCQPUPrimitive nodes dispatch Bell prep through MockHPCConnector\n"
        "(emulating SLURM). Each publishes a Bell-pair claim on link-A.\n"
        "EntanglementBarrier waits for both claims before the aggregator runs.",
        title="qmesh.scheduler  •  HPC + entanglement"
    ))

    LEDGER.mkdir(parents=True, exist_ok=True)
    connector_a = MockHPCConnector(
        work_dir=LEDGER / "hpc_a", scheduler_emulated="slurm-fugaku",
    )
    connector_b = MockHPCConnector(
        work_dir=LEDGER / "hpc_b", scheduler_emulated="slurm-fugaku",
    )

    dag = DAG()

    qpu_a = dag.add(HPCQPUPrimitive(
        id="qpu_a", name="bell-on-fugaku-A",
        module=_bell_module("bell_a"),
        backend="qmesh.aer", shots=512, sign=False,
        connector=connector_a,
        partition="gpu", walltime="00:15:00", nodes=1, cpus_per_task=4,
        extra_sbatch=["#SBATCH --gres=gpu:1"],
    ))
    qpu_b = dag.add(HPCQPUPrimitive(
        id="qpu_b", name="bell-on-fugaku-B",
        module=_bell_module("bell_b"),
        backend="qmesh.aer", shots=512, sign=False,
        connector=connector_b,
        partition="gpu", walltime="00:15:00", nodes=1, cpus_per_task=4,
        extra_sbatch=["#SBATCH --gres=gpu:1"],
    ))

    claim_a = dag.add(BellPairClaim(
        id="claim_a", photonic_link="link-A",
        producer_node_id="qpu_a", fidelity=0.991,
        depends_on=["qpu_a"],
    ))
    claim_b = dag.add(BellPairClaim(
        id="claim_b", photonic_link="link-A",
        producer_node_id="qpu_b", fidelity=0.987,
        depends_on=["qpu_b"],
    ))

    bar = dag.add(EntanglementBarrier(
        id="bar", photonic_link="link-A",
        expected_bell_pairs=2, timeout_seconds=10.0,
        depends_on=["claim_a", "claim_b"],
    ))

    def aggregate(ctx):
        return {
            "qpu_a_counts": ctx["qpu_a"]["counts"],
            "qpu_b_counts": ctx["qpu_b"]["counts"],
            "claims": ctx["bell_pairs"]["link-A"],
            "bar": ctx["bar"],
        }

    dag.add(ClassicalTask(
        id="agg", name="aggregate",
        fn=aggregate, depends_on=["bar"],
    ))

    run = execute(dag, ledger_dir=LEDGER)

    # ---- present results ----

    table = Table(title="DAG nodes")
    table.add_column("node", style="cyan")
    table.add_column("kind")
    table.add_column("ok", justify="center")
    table.add_column("dur (s)", justify="right")
    table.add_column("note", overflow="fold")
    for nid, r in run.results.items():
        note = ""
        if r.kind == "hpc_qpu":
            hpc = r.metadata["hpc"]
            note = (f"job={hpc['job_id']} partition={hpc['partition']} "
                    f"walltime={hpc['walltime_request']}")
        elif r.kind == "entanglement_barrier":
            note = (f"link={r.metadata['photonic_link']} "
                    f"got={r.metadata['claims_received']}/"
                    f"{r.metadata['expected_bell_pairs']} "
                    f"wait={r.metadata['wait_seconds']:.4f}s")
        elif r.kind == "bell_pair_claim":
            note = "claim published to ctx['bell_pairs']['link-A']"
        table.add_row(
            nid, r.kind, "[green]OK[/]" if r.ok else "[red]FAIL[/]",
            f"{r.duration_seconds:.4f}", note,
        )
    print(table)

    # HPC blocks
    for nid in ("qpu_a", "qpu_b"):
        hpc = run.results[nid].metadata["hpc"]
        print(Panel.fit(
            "\n".join(f"  {k:<22} {v}" for k, v in hpc.items()),
            title=f"manifest.execution.hpc — {nid}",
        ))

    # Barrier wait
    bar_meta = run.results["bar"].metadata
    print(Panel.fit(
        f"  photonic_link          {bar_meta['photonic_link']}\n"
        f"  expected_bell_pairs    {bar_meta['expected_bell_pairs']}\n"
        f"  claims_received        {bar_meta['claims_received']}\n"
        f"  wait_seconds           {bar_meta['wait_seconds']:.6f}",
        title="EntanglementBarrier — barrier wait",
    ))

    # Audit pointer
    manifest = json.loads(run.manifest_path.read_text())
    print(Panel.fit(
        f"  run_id              {manifest['execution']['run_id']}\n"
        f"  signed              {manifest.get('signature', {}).get('alg')}\n"
        f"  manifest path       {run.manifest_path}\n"
        f"  n_nodes             {manifest['execution']['n_nodes']}\n"
        f"  n_failed            {manifest['execution']['n_failed']}",
        title="aggregate run manifest",
    ))


if __name__ == "__main__":
    main()
