"""qmesh.compiler — pass pipeline operating on qmesh.ir.

Phase 0: stub passes. Phase 1+: TKET, BQSKit, MQT integration; RL passes
(SABRE-RL, AlphaTensor-Quantum, ZX+RL).

The contract:
    pass(module: Module, ctx: PassContext) -> Module
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from qmesh.ir.module import Module


@dataclass(slots=True)
class PassContext:
    backend_name: str
    options: dict[str, object] = field(default_factory=dict)
    # provenance: passes append themselves to here
    log: list[dict[str, object]] = field(default_factory=list)


Pass = Callable[[Module, PassContext], Module]


def pipeline(*passes: Pass) -> Pass:
    def run(module: Module, ctx: PassContext) -> Module:
        for p in passes:
            module = p(module, ctx)
            ctx.log.append({"pass": p.__name__})
        return module

    return run


def canonicalise(module: Module, ctx: PassContext) -> Module:  # noqa: ARG001
    """No-op canonicaliser placeholder."""
    return module


def constant_fold(module: Module, ctx: PassContext) -> Module:  # noqa: ARG001
    """Fold trivially-zero rotations etc. Phase-1 work."""
    return module


def gate_decompose(module: Module, ctx: PassContext) -> Module:  # noqa: ARG001
    """Decompose into backend's native gate set. Phase-1 work."""
    return module


__all__ = ["Pass", "PassContext", "pipeline", "canonicalise",
           "constant_fold", "gate_decompose"]
