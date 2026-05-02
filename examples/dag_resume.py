"""Phase 4β: checkpoint resume on a hybrid DAG.

We build a 4-node DAG (prep QPU → derive classical → score QPU → publish
classical). The "publish" step intentionally fails on the first run.
After the failure, we fix the bug, call `qmesh.scheduler.resume()`, and
the executor:

  - skips the two QPU nodes whose results are already on disk (no second
    Aer submission, no fresh manifest_hash);
  - re-runs only the previously-failed node;
  - emits a new aggregate manifest whose `parent_run_id` chains back to
    the original run so an auditor can walk the lineage.

Run:
    PYTHONPATH=. python3 examples/dag_resume.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rich import print
from rich.panel import Panel
from rich.table import Table

import qmesh
from qmesh.ir.builder import circuit
from qmesh.scheduler import DAG, ClassicalTask, QPUPrimitive, execute, resume


LEDGER = Path("ledger/dag_resume_demo")


def _bell_module():
    with circuit("bell", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    return c.module


def _ghz3_module():
    with circuit("ghz3", n_qubits=3, n_bits=3) as c:
        c.h(0); c.cx(0, 1); c.cx(1, 2)
        c.measure(0, 0); c.measure(1, 1); c.measure(2, 2)
    return c.module


def _make_dag(*, publish_fails: bool) -> DAG:
    dag = DAG()
    dag.add(QPUPrimitive(
        id="prep", name="bell", module=_bell_module(),
        backend="qmesh.aer", shots=512, sign=False,
    ))

    dag.add(ClassicalTask(
        id="derive", name="derive-weights",
        fn=lambda ctx: {"weight": ctx["prep"]["counts"].get("00", 0) / 512},
        depends_on=["prep"],
    ))

    dag.add(QPUPrimitive(
        id="score", name="ghz3", module=_ghz3_module(),
        backend="qmesh.aer", shots=512, sign=False,
        depends_on=["derive"],
    ))

    def publish(ctx):
        if publish_fails:
            raise RuntimeError("intentional publish bug")
        return {
            "weight": ctx["derive"]["weight"],
            "ghz_000": ctx["score"]["counts"].get("000", 0),
            "ghz_111": ctx["score"]["counts"].get("111", 0),
        }

    dag.add(ClassicalTask(
        id="publish", name="publish-summary",
        fn=publish, depends_on=["score"],
    ))
    return dag


def _show(run, label: str) -> None:
    table = Table(title=label)
    table.add_column("node", style="cyan")
    table.add_column("ok", justify="center")
    table.add_column("dur (s)", justify="right")
    table.add_column("manifest", overflow="fold")
    for nid, r in run.results.items():
        table.add_row(
            nid, "✓" if r.ok else "✗", f"{r.duration_seconds:.3f}",
            (r.manifest_hash or "")[:18] or "—",
        )
    print(table)
    print(f"[dim]aggregate manifest: {run.manifest_path}[/]")


def main() -> None:
    print(Panel.fit("[bold]Phase 4β: checkpoint resume[/]\n"
                    "First run lets `publish` fail, then resume() fixes it.",
                    title="qmesh.scheduler.resume"))

    LEDGER.mkdir(parents=True, exist_ok=True)

    first = execute(_make_dag(publish_fails=True), ledger_dir=LEDGER)
    _show(first, "run #1 — publish fails")
    print(f"[red]first run failed nodes:[/] "
          f"{[k for k, r in first.results.items() if not r.ok]}\n")

    # Fix and resume.
    resumed = resume(_make_dag(publish_fails=False),
                     first.manifest_path.parent, ledger_dir=LEDGER)
    _show(resumed, "run #2 — resumed")

    manifest = json.loads(resumed.manifest_path.read_text())
    lineage = manifest["lineage"]
    print(Panel.fit(
        f"parent_run_id         {lineage['parent_run_id']}\n"
        f"parent_manifest_hash  {lineage['parent_manifest_hash'][:32]}…\n"
        f"parent_signature_alg  {lineage['parent_signature_alg']}\n"
        f"n_skipped_resume      {lineage['n_skipped_resume']}\n"
        f"n_failed              {manifest['execution']['n_failed']}\n"
        f"published             {resumed.context['publish']}",
        title="audit chain",
    ))


if __name__ == "__main__":
    main()
