"""qmesh.scheduler.dag — typed hybrid execution graph.

Nodes:
  - ClassicalTask  — pure Python callable; reads context, returns updates.
  - QPUPrimitive   — a qmesh.ir Module submitted to a backend (or via the
                     router). Result counts go into context under `node_id`.
  - MCMRegion      — a sub-graph that runs as a single shot with mid-circuit
                     measurement and live classical conditioning. Pinned to
                     one backend for the feedforward-latency budget.
  - Fanout         — replicate a child node N times with different params,
                     execute in parallel; gather results into a list.
  - Barrier        — explicit sync point; can be entanglement-aware in
                     Phase 4β.

Edges are declared with `depends_on=[node_id, ...]`. The `DAG.add(node)`
method validates the edge endpoints exist; `DAG.toposort()` returns nodes
in execution order.

Provenance: the executor signs an aggregate manifest enumerating every
node's individual manifest hash + its decision context.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from qmesh.ir.module import Module


def _new_id(prefix: str = "node") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass(slots=True)
class NodeResult:
    node_id: str
    kind: str
    ok: bool
    output: Any
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    manifest_hash: str | None = None


@dataclass(slots=True)
class Node(ABC):
    """Abstract DAG node."""

    id: str = field(default_factory=lambda: _new_id())
    name: str = ""
    depends_on: list[str] = field(default_factory=list)

    @abstractmethod
    def kind(self) -> str: ...


@dataclass(slots=True)
class ClassicalTask(Node):
    """Pure Python step. Reads/writes the DAG context dict."""

    fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    description: str = ""

    def kind(self) -> str:
        return "classical"

    def __post_init__(self) -> None:
        if self.fn is None:
            raise ValueError("ClassicalTask requires a `fn`")


@dataclass(slots=True)
class QPUPrimitive(Node):
    """Run a qmesh.ir Module on a backend (or via the router).

    `module_factory` builds the IR fresh from `context` at execution time —
    this is how upstream nodes inject parameters (e.g. classical-optimised
    angles for a VQE ansatz, or measurement results that drive a Rydberg
    pulse amplitude).
    """

    module: Module | None = None
    module_factory: Callable[[dict[str, Any]], Module] | None = None
    backend: str = "auto"
    shots: int = 1024
    objective: Any = None    # qmesh.router.Objective when backend=='auto'
    sign: bool = True

    def kind(self) -> str:
        return "qpu"

    def __post_init__(self) -> None:
        if self.module is None and self.module_factory is None:
            raise ValueError(
                "QPUPrimitive requires either `module` or `module_factory`"
            )


@dataclass(slots=True)
class Barrier(Node):
    """Synchronisation point. Always runs after all `depends_on` complete."""

    def kind(self) -> str:
        return "barrier"


# -------- Phase 4β: entanglement-aware barriers (BEGIN clearly-delimited) --------

@dataclass(slots=True)
class EntanglementBarrier(Barrier):
    """Barrier that waits for `expected_bell_pairs` Bell-pair claims to land
    on `photonic_link` before classical-conditioned downstream work runs.

    Models the IonQ photonic-interconnect / Cisco Universal Quantum Switch
    flow (April 2026): two QPUs entangle through a fiber link, register
    "I have a Bell pair on link-A" claims into the run context, and the
    barrier only completes once `expected_bell_pairs` claims have landed.

    α scope: claims live in `ctx['bell_pairs'][photonic_link]` as a list of
    `BellPairClaim` payload dicts. β: integrate with a real photonic-link
    middleware service (Cisco UQS RPC, IonQ's verified-link API).
    """

    photonic_link: str = "link-A"
    expected_bell_pairs: int = 1
    timeout_seconds: float = 60.0
    poll_interval_seconds: float = 0.05

    def kind(self) -> str:
        return "entanglement_barrier"


@dataclass(slots=True)
class BellPairClaim(ClassicalTask):
    """Helper ClassicalTask that publishes a Bell-pair claim into the run
    context. Producers depend on the upstream QPU node that actually
    generated the entangled state — the claim itself is just a token saying
    "this run produced one Bell pair on `photonic_link`".

    Output: the claim payload dict is appended to
    `ctx['bell_pairs'][photonic_link]`. The BellPairClaim's own
    `ctx[node_id]` entry is the same dict, so downstream nodes can read it
    directly if they prefer not to walk the link list.
    """

    photonic_link: str = "link-A"
    pair_id: str = ""
    fidelity: float = 0.95
    producer_node_id: str = ""

    def __post_init__(self) -> None:
        # Override the parent's `fn is None` check by providing our own fn.
        if not self.pair_id:
            self.pair_id = _new_id("pair")

        # Capture the (post-init) attributes so the closure stays stable.
        link = self.photonic_link
        pid = self.pair_id
        fid = self.fidelity
        producer = self.producer_node_id or self.id

        def _publish(ctx: dict[str, Any]) -> dict[str, Any]:
            claim = {
                "pair_id": pid,
                "photonic_link": link,
                "fidelity": fid,
                "producer_node_id": producer,
                "claimed_at": _now_iso(),
            }
            # Mutate ctx in-place: the executor reads from this dict directly,
            # not from `ctx[self.id]` — that's how the EntanglementBarrier
            # discovers claims published in parallel by sibling nodes.
            link_bucket = ctx.setdefault("bell_pairs", {}).setdefault(link, [])
            link_bucket.append(claim)
            return claim

        self.fn = _publish

    def kind(self) -> str:
        return "bell_pair_claim"


def _now_iso() -> str:
    import time as _t
    return _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())

# -------- Phase 4β: entanglement-aware barriers (END) --------


@dataclass(slots=True)
class Fanout(Node):
    """Replicate a child template N times with per-instance kwargs.

    The executor runs each instance in parallel (when the runtime supports
    it) and writes a list of results into context[self.id].
    """

    template: Callable[[dict[str, Any]], Node] | None = None
    fanout_kwargs: list[dict[str, Any]] = field(default_factory=list)

    def kind(self) -> str:
        return "fanout"

    def __post_init__(self) -> None:
        if self.template is None:
            raise ValueError("Fanout requires a `template` callable")


@dataclass(slots=True)
class MCMRegion(Node):
    """Mid-circuit-measurement region — a single shot with conditional gates.

    Phase 4α: passthrough to a `QPUPrimitive` whose backend supports
    measurement_feedforward; rejected on backends that don't.
    Phase 4β: real lowering of MCM regions with classical-feedforward
    sub-graphs.
    """

    module: Module | None = None
    backend: str = "qmesh.aer"  # Aer supports if_test
    shots: int = 1024
    feedforward_latency_budget_ns: int = 200_000

    def kind(self) -> str:
        return "mcm"


@dataclass(slots=True)
class DAG:
    """A hybrid execution graph."""

    nodes: dict[str, Node] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError(f"node id {node.id!r} already in DAG")
        for dep in node.depends_on:
            if dep not in self.nodes:
                raise ValueError(f"node {node.id!r} depends on missing {dep!r}")
        self.nodes[node.id] = node
        return node

    def toposort(self) -> list[Node]:
        order: list[Node] = []
        visited: set[str] = set()
        temp: set[str] = set()

        def visit(n: Node) -> None:
            if n.id in visited:
                return
            if n.id in temp:
                raise ValueError(f"cycle through node {n.id!r}")
            temp.add(n.id)
            for dep_id in n.depends_on:
                visit(self.nodes[dep_id])
            temp.remove(n.id)
            visited.add(n.id)
            order.append(n)

        for n in self.nodes.values():
            visit(n)
        return order

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "kind": n.kind(),
                    "depends_on": list(n.depends_on),
                    "extra": _node_summary(n),
                }
                for n in self.nodes.values()
            ],
        }


def _node_summary(n: Node) -> dict:
    if isinstance(n, QPUPrimitive):
        return {
            "backend": n.backend,
            "shots": n.shots,
            "module_hash": n.module.hash() if n.module is not None else None,
            "factory": n.module_factory is not None,
        }
    if isinstance(n, ClassicalTask):
        return {"description": n.description, "fn": n.fn.__name__ if n.fn else None}
    if isinstance(n, Fanout):
        return {"n_instances": len(n.fanout_kwargs)}
    if isinstance(n, MCMRegion):
        return {"backend": n.backend, "shots": n.shots,
                "ff_budget_ns": n.feedforward_latency_budget_ns}
    return {}


__all__ = [
    "DAG", "Node", "NodeResult",
    "ClassicalTask", "QPUPrimitive", "Barrier", "Fanout", "MCMRegion",
    "EntanglementBarrier", "BellPairClaim",
]
