"""qmesh.mitigation — error-mitigation pipeline orchestrator.

Phase 0 stub. Phase 1: wraps Mitiq 1.0 (ZNE, PEC, DDD, LRE, CDR, REM, PT)
and selects a Pareto-optimal stack from (bias, variance, $) given (circuit,
backend, budget).

Public API sketch:

    qmesh.mitigation.auto(circuit, backend, budget) -> MitigationStack
    qmesh.mitigation.apply(stack, run_callable) -> mitigated_estimator
"""

from __future__ import annotations
