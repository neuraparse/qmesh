"""qmesh.scheduler — hybrid DAG executor.

Phase 4α: typed DAG nodes (ClassicalTask, QPUPrimitive, Barrier, Fanout,
MCMRegion), topological executor with thread-pool parallelism inside each
DAG level, append-only event log, signed aggregate manifest.

Phase 4β (in progress): checkpoint-based resume after crash —
`DAGExecutor.resume(dag, original_run_dir)` reads
`{run_dir}/node_results/*.json` snapshots, skips successful nodes, and
re-executes failed/incomplete ones; the new manifest carries
`parent_run_id` for an audit chain. Still-queued: cross-modality
`ChannelOp` stitching, SLURM/PBS HPC connectors, entanglement-aware
Barrier semantics.

Public API:

    from qmesh.scheduler import DAG, ClassicalTask, QPUPrimitive, execute
    dag = DAG()
    a = dag.add(QPUPrimitive(name="prep", module=prep_module, backend="qmesh.aer"))
    b = dag.add(ClassicalTask(name="post",
                              fn=lambda ctx: {"score": some_fn(ctx[a.id])},
                              depends_on=[a.id]))
    run = execute(dag)
    print(run.context[b.id])
"""

from __future__ import annotations

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
from qmesh.scheduler.executor import DAGExecutor, DAGRun, execute, resume
from qmesh.scheduler.hpc import (
    HPCConnector,
    HPCQPUPrimitive,
    HPCResult,
    HPCSubmission,
    MockHPCConnector,
    PBSConnector,
    SLURMConnector,
)

__all__ = [
    "DAG",
    "Node",
    "NodeResult",
    "ClassicalTask",
    "QPUPrimitive",
    "Barrier",
    "Fanout",
    "MCMRegion",
    "EntanglementBarrier",
    "BellPairClaim",
    "DAGExecutor",
    "DAGRun",
    "execute",
    "resume",
    "HPCConnector",
    "HPCQPUPrimitive",
    "HPCResult",
    "HPCSubmission",
    "MockHPCConnector",
    "PBSConnector",
    "SLURMConnector",
]
