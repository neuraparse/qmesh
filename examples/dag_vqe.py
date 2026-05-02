"""VQE on H2 expressed as a hybrid DAG.

Each Pauli observable in H is a separate QPUPrimitive node; they fan out
in parallel under a single Barrier; a ClassicalTask sums them into the
energy estimate. The whole iteration is one DAG run with a signed
aggregate manifest.

This is the same physics as `examples/vqe_h2.py` but expressed as a graph
the scheduler executes — proving the scheduler integrates with real
algorithm work.

Run:
    PYTHONPATH=. python3 examples/dag_vqe.py
"""

from __future__ import annotations

from rich import print
from rich.panel import Panel

import qmesh
from qmesh.ir.builder import circuit
from qmesh.scheduler import DAG, Barrier, ClassicalTask, QPUPrimitive, execute


H_TERMS = [
    ("II", -1.0523732),
    ("IZ",  0.39793742),
    ("ZI", -0.39793742),
    ("ZZ", -0.0112801),
    ("XX",  0.18093119),
]


def _ansatz(theta: list[float]) -> qmesh.Module:
    with circuit("h2_ansatz", n_qubits=2, n_bits=2) as c:
        c.ry(theta[0], 0); c.ry(theta[1], 1)
        c.cx(0, 1)
        c.ry(theta[2], 0); c.ry(theta[3], 1)
    return c.module


def _measure_in(basis: str, theta: list[float]) -> qmesh.Module:
    """Build prep + basis-rotation + measurement IR."""
    with circuit(f"h2_{basis}", n_qubits=2, n_bits=2) as c:
        c.ry(theta[0], 0); c.ry(theta[1], 1)
        c.cx(0, 1)
        c.ry(theta[2], 0); c.ry(theta[3], 1)
        for i, b in enumerate(reversed(basis)):
            if b == "X":
                c.h(i)
            elif b == "Y":
                c.sdg(i); c.h(i)
        c.measure(0, 0); c.measure(1, 1)
    return c.module


def _expectation(counts: dict[str, int], basis: str) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    val = 0.0
    for bitstring, n in counts.items():
        sign = 1
        for i, b in enumerate(reversed(basis)):
            bit = int(bitstring[len(bitstring) - 1 - i])
            if b == "I":
                continue
            if bit == 1:
                sign *= -1
        val += sign * n / total
    return val


def build_energy_dag(theta: list[float], shots: int = 2048) -> DAG:
    """Build a DAG that computes <H> for the given angles."""
    dag = DAG(metadata={"theta": theta, "shots": shots})

    qpu_node_ids: list[tuple[str, float, str]] = []  # (node_id, coeff, basis)
    for basis, coeff in H_TERMS:
        if basis == "II":
            continue
        node = dag.add(QPUPrimitive(
            name=f"prep_meas_{basis}",
            module=_measure_in(basis, theta),
            backend="qmesh.aer",
            shots=shots,
            sign=False,                  # avoid signing per-shot batches
        ))
        qpu_node_ids.append((node.id, coeff, basis))

    barrier = dag.add(Barrier(
        name="all_terms_done",
        depends_on=[nid for nid, _, _ in qpu_node_ids],
    ))

    def gather_energy(ctx: dict) -> dict:
        e = sum(c for b, c in H_TERMS if b == "II")  # identity term
        per_term = {}
        for nid, coeff, basis in qpu_node_ids:
            counts = ctx[nid]["counts"]
            ev = _expectation(counts, basis)
            per_term[basis] = ev
            e += coeff * ev
        return {"energy": e, "per_term_expectations": per_term}

    dag.add(ClassicalTask(
        name="gather_energy",
        fn=gather_energy,
        depends_on=[barrier.id],
    ))
    return dag


def main() -> None:
    print(Panel.fit(
        "[bold cyan]VQE on H2 as a DAG[/]\n\n"
        "5 Hamiltonian terms → 4 QPU nodes (II is constant) → Barrier → ClassicalTask.\n"
        "All 4 QPU nodes run in parallel inside one DAG level.",
        title="qmesh.scheduler",
    ))

    # Use the angles VQE optimisation found in the linear example
    theta = [0.2015, -0.2193, 3.1485, 0.1923]   # near-optimal H2 angles
    dag = build_energy_dag(theta, shots=4096)
    run = execute(dag, ledger_dir="ledger/dag_vqe")

    energy_node = next(n for n in dag.nodes.values()
                       if isinstance(n, ClassicalTask))
    payload = run.context[energy_node.id]

    print()
    print(f"[bold]E(theta) = {payload['energy']:.5f}[/]")
    print(f"FCI ground state         = -1.857")
    print()
    print(f"per-term expectations:")
    for k, v in payload["per_term_expectations"].items():
        print(f"  ⟨{k}⟩ = {v:+.4f}")
    print()
    print(f"[dim]run id: {run.run_id}[/]")
    print(f"[dim]aggregate manifest: {run.manifest_path}[/]")
    print(f"[dim]wall: {run.duration_seconds:.3f}s   "
          f"({len(run.results)} nodes, "
          f"{sum(1 for r in run.results.values() if r.kind == 'qpu')} QPU)[/]")


if __name__ == "__main__":
    main()
