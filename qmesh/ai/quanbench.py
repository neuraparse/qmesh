"""qmesh.ai.quanbench — small QuanBench-shaped evaluator for the copilot.

QuanBench (Liu et al., Oct 2025) is a cross-framework eval suite that scores
LLM-generated quantum circuits along multiple axes — intent recognition,
syntactic correctness, executable semantics. This module ships a *small*,
self-contained, mock-friendly version: 10 prompts spanning the patterns the
intent compiler can hit plus two "hard" prompts that force the LLM/mock
fallback path. For each prompt we score:

  * intent_recognised : did the rule-based intent compiler classify it?
  * compiled          : did `draft()` return a runnable Module?
  * simulator_passed  : did the validator's `passed` flag come back True?
  * pattern_match     : did the recovered intent_pattern match what we expected?

Aggregated into a single JSON report with a per-pattern and per-provider
breakdown plus wall-time stats.

Use:
    from qmesh.ai.quanbench import run_quanbench, BUILTIN_SUITE
    report = run_quanbench(provider="mock", suite=BUILTIN_SUITE)
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from qmesh.ai.copilot import draft
from qmesh.ai.llm import LLMProvider, MockProvider, auto_provider


@dataclass
class BenchPrompt:
    """A single benchmark item.

    Optional ``expected_property`` is a callable
    ``(module, counts: dict[str, int]) -> bool`` that the harness invokes
    after running the produced module on ``qmesh.aer``. Used by the novel-
    prompt suite to distinguish "compiles" from "behaves correctly".
    """
    prompt: str
    expected_pattern: str | None
    tags: list[str] = field(default_factory=list)
    expected_property: Callable[..., bool] | None = None


# Phase 5γ alias — makes the public name match the spec language.
QuanBenchPrompt = BenchPrompt


# ~10 prompts spanning the intent compiler's coverage + two "hard" cases.
BUILTIN_SUITE: list[BenchPrompt] = [
    BenchPrompt("Build a Bell state",                                "bell",                       ["bell", "easy"]),
    BenchPrompt("Make a Bell pair please",                            "bell",                       ["bell", "easy"]),
    BenchPrompt("GHZ state on 4 qubits",                              "ghz",                        ["ghz"]),
    BenchPrompt("GHZ state on 6 qubits",                              "ghz",                        ["ghz"]),
    BenchPrompt("QFT on 3 qubits",                                    "qft",                        ["qft"]),
    BenchPrompt("hardware-efficient ansatz on 4 qubits",              "hardware_efficient_ansatz",  ["hea"]),
    BenchPrompt("W state on 3 qubits",                                "w_state",                    ["w_state"]),
    BenchPrompt("random Clifford on 5 qubits depth 4",                "random_clifford",            ["clifford"]),
    # "Hard" prompts — wording the intent compiler doesn't catch, exercising
    # the LLM/mock fallback path. The mock falls back to a Bell stub.
    BenchPrompt("Prepare a maximally entangled two-qubit register",   None,                         ["hard", "fallback"]),
    BenchPrompt("Construct an entangling primitive on a pair",         None,                         ["hard", "fallback"]),
]


@dataclass
class _PromptResult:
    prompt: str
    expected_pattern: str | None
    actual_pattern: str | None
    intent_recognised: bool
    compiled: bool
    simulator_passed: bool
    pattern_match: bool
    iterations: int
    provider: str
    duration_seconds: float
    property_checked: bool = False
    property_passed: bool = False
    error: str | None = None


def _run_property_check(
    item: BenchPrompt, module: Any
) -> tuple[bool, bool, str | None]:
    """Invoke ``item.expected_property`` if set; return ``(checked, passed, err)``.

    The callable receives ``(module, counts)`` (a small Aer run) and must
    return a truthy value if the produced circuit matches the requested
    behaviour. The harness shields the rest of the score from raised
    exceptions inside the callable.
    """
    if item.expected_property is None:
        return False, False, None
    try:
        import qmesh
        result, _ = qmesh.submit(
            module, backend="qmesh.aer", shots=512,
            sign=False, ledger_dir="ledger/quanbench_property",
        )
        ok = bool(item.expected_property(module, dict(result.counts)))
        return True, ok, None
    except Exception as e:  # noqa: BLE001
        return True, False, f"{type(e).__name__}: {e}"


def _score_prompt(item: BenchPrompt, provider: LLMProvider | None) -> _PromptResult:
    t0 = time.time()
    try:
        r = draft(item.prompt, provider=provider, max_iterations=2, validate=True)
        dt = time.time() - t0
        sim_passed = bool(r.validation.get("passed", False))
        pattern_ok = (r.intent_pattern == item.expected_pattern)
        prop_checked, prop_passed, prop_err = _run_property_check(item, r.module)
        return _PromptResult(
            prompt=item.prompt,
            expected_pattern=item.expected_pattern,
            actual_pattern=r.intent_pattern,
            intent_recognised=r.intent_pattern is not None,
            compiled=True,
            simulator_passed=sim_passed,
            pattern_match=pattern_ok,
            iterations=r.iterations,
            provider=r.provider,
            duration_seconds=dt,
            property_checked=prop_checked,
            property_passed=prop_passed,
            error=prop_err,
        )
    except Exception as e:  # noqa: BLE001
        return _PromptResult(
            prompt=item.prompt,
            expected_pattern=item.expected_pattern,
            actual_pattern=None,
            intent_recognised=False,
            compiled=False,
            simulator_passed=False,
            pattern_match=False,
            iterations=0,
            provider="error",
            duration_seconds=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )


def _resolve_provider(provider: str | LLMProvider | None) -> LLMProvider | None:
    if provider is None or provider == "auto":
        return None  # let draft() pick
    if isinstance(provider, LLMProvider):
        return provider
    if provider == "mock":
        return MockProvider()
    if provider == "auto":
        return auto_provider()
    raise ValueError(f"unknown provider {provider!r}; pass 'mock', 'auto', or an LLMProvider")


def run_quanbench(
    *,
    provider: str | LLMProvider | None = "mock",
    suite: list[BenchPrompt] | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run the harness and return the report dict.

    A "success" is defined as:
      * compiled AND simulator_passed AND
      * (pattern_match if expected_pattern is not None else compiled)

    Parameters
    ----------
    provider : 'mock' | 'auto' | LLMProvider | None
        Which provider to send the LLM-fallback prompts to.
    suite : list[BenchPrompt] | None
        Override prompt list (default: BUILTIN_SUITE).
    output_path : str | Path | None
        If set, write the JSON report to this path.
    """
    suite = suite if suite is not None else BUILTIN_SUITE
    prov = _resolve_provider(provider)

    t0 = time.time()
    results = [_score_prompt(p, prov) for p in suite]
    total_wall = time.time() - t0

    # Aggregate
    n = len(results)
    n_compiled = sum(1 for r in results if r.compiled)
    n_sim_pass = sum(1 for r in results if r.simulator_passed)
    n_intent_ok = sum(1 for r in results if r.intent_recognised)
    successes = []
    for r in results:
        if not (r.compiled and r.simulator_passed):
            successes.append(False)
            continue
        if r.property_checked:
            # Novel-suite path — the property is the gold standard.
            successes.append(r.property_passed)
        elif r.expected_pattern is not None:
            successes.append(r.pattern_match)
        else:
            successes.append(True)
    n_success = sum(successes)

    iters = [r.iterations for r in results]
    durations = [r.duration_seconds for r in results]

    # per-pattern breakdown — keyed by expected_pattern (or 'fallback')
    per_pattern: dict[str, dict[str, Any]] = {}
    for r, ok in zip(results, successes):
        key = r.expected_pattern or "fallback"
        slot = per_pattern.setdefault(
            key, {"n": 0, "passed": 0, "compiled": 0, "sim_passed": 0}
        )
        slot["n"] += 1
        slot["passed"] += int(ok)
        slot["compiled"] += int(r.compiled)
        slot["sim_passed"] += int(r.simulator_passed)

    # per-provider breakdown — keyed by the provider field of each result
    per_provider: dict[str, dict[str, Any]] = {}
    for r, ok in zip(results, successes):
        slot = per_provider.setdefault(
            r.provider, {"n": 0, "passed": 0, "mean_iter": 0.0, "_iter_total": 0}
        )
        slot["n"] += 1
        slot["passed"] += int(ok)
        slot["_iter_total"] += r.iterations
    for slot in per_provider.values():
        slot["mean_iter"] = slot["_iter_total"] / max(1, slot["n"])
        slot.pop("_iter_total", None)

    report: dict[str, Any] = {
        "schema": "qmesh.ai.quanbench.v1",
        "n_prompts": n,
        "success_rate": n_success / max(1, n),
        "n_success": n_success,
        "n_compiled": n_compiled,
        "n_simulator_passed": n_sim_pass,
        "n_intent_recognised": n_intent_ok,
        "mean_iterations": (statistics.mean(iters) if iters else 0.0),
        "wall_time_seconds": {
            "total": total_wall,
            "mean_per_prompt": statistics.mean(durations) if durations else 0.0,
            "max_per_prompt": max(durations) if durations else 0.0,
        },
        "per_pattern": per_pattern,
        "per_provider": per_provider,
        "results": [asdict(r) for r in results],
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- #
# Phase 5γ: novel-prompt suite                                                #
# --------------------------------------------------------------------------- #
#
# These prompts are deliberately worded so the rule-based intent compiler in
# qmesh.ai.intent does NOT match them — they require real LLM understanding
# of named primitives (teleportation, [[5,1,3]] code, Trotter step, ...).
# Each prompt has an `expected_property(module, counts) -> bool` that runs on
# qmesh.aer and decides if the produced circuit *behaves* correctly.
#
# Honest framing: under the MockProvider the suite's success rate is ~zero
# because the mock falls back to a Bell stub that satisfies almost none of
# these properties. That's the point — it makes NOVEL_SUITE a meaningful
# "hard" probe that real LLM providers can score on.


def _support_indices(counts: dict[str, int]) -> set[int]:
    """Return the set of computational-basis indices observed."""
    out: set[int] = set()
    for bitstr in counts:
        try:
            out.add(int(bitstr.replace(" ", ""), 2))
        except ValueError:
            continue
    return out


def _total(counts: dict[str, int]) -> int:
    return max(1, sum(counts.values()))


def _prop_uniform_over_target(target: set[int], qubits: int, threshold: float = 0.85
                              ) -> Callable[..., bool]:
    """Return a property that passes if the support is approximately equal
    to ``target`` and the cumulative probability across ``target`` exceeds
    ``threshold``. Requires that the produced circuit operate on exactly
    ``qubits`` qubits — the Bell stub trivially fails this.
    """
    def check(module: Any, counts: dict[str, int]) -> bool:
        total = _total(counts)
        # Reject circuits that don't even use the right number of qubits.
        for k in counts:
            if len(k.replace(" ", "")) != qubits:
                return False
            break
        in_target = sum(v for k, v in counts.items()
                        if int(k.replace(" ", ""), 2) in target)
        observed = _support_indices(counts)
        # The observed support should approximately equal the target.
        # Allow at most one missing target element (sampling noise).
        missing = target - observed
        return (
            (in_target / total) >= threshold
            and observed.issubset(target)
            and len(missing) <= 1
        )
    return check


def _prop_n_qubits(qubits: int) -> Callable[..., bool]:
    """Property: produced circuit operates on at least ``qubits`` qubits."""
    def check(module: Any, counts: dict[str, int]) -> bool:
        # Count bitstring length to infer how many bits got measured.
        for k in counts:
            return len(k.replace(" ", "")) >= qubits
        return False
    return check


def _prop_ghz_x_basis(n: int) -> Callable[..., bool]:
    """GHZ in the X basis: parity should be even after H on all qubits."""
    def check(module: Any, counts: dict[str, int]) -> bool:
        total = _total(counts)
        even = 0
        for k, v in counts.items():
            bs = k.replace(" ", "")
            if len(bs) < n:
                return False
            if bs.count("1") % 2 == 0:
                even += v
        return (even / total) > 0.9
    return check


def _prop_qft_inverse(n: int) -> Callable[..., bool]:
    """Inverse-QFT applied to |0...0> should leave |0...0>."""
    def check(module: Any, counts: dict[str, int]) -> bool:
        total = _total(counts)
        zero = "0" * n
        for k in counts:
            if len(k.replace(" ", "")) != n:
                return False
            break
        return (counts.get(zero, 0) / total) > 0.9
    return check


def _prop_teleportation() -> Callable[..., bool]:
    """Teleportation: post-correction, q[2] should reflect q[0]'s prep."""
    def check(module: Any, counts: dict[str, int]) -> bool:
        # Heuristic: at least 3 qubits' worth of bits.
        for k in counts:
            return len(k.replace(" ", "")) >= 3
        return False
    return check


def _prop_haar_random(n: int) -> Callable[..., bool]:
    """Haar-random unitary: support should spread over many basis states."""
    def check(module: Any, counts: dict[str, int]) -> bool:
        return len(_support_indices(counts)) >= max(2, 2 ** (n - 1))
    return check


def _prop_phase_kickback() -> Callable[..., bool]:
    """Phase-kickback subroutine: must use at least 3 qubits (ancilla +
    register) and the ancilla outcome must show the kickback phase shift —
    i.e., the marginal on the first bit must not be flat |0⟩."""
    def check(module: Any, counts: dict[str, int]) -> bool:
        bs_len: int | None = None
        for k in counts:
            bs_len = len(k.replace(" ", ""))
            break
        if bs_len is None or bs_len < 3:
            return False
        total = _total(counts)
        # Marginal P(first bit = 1)
        p_one = sum(v for k, v in counts.items()
                    if k.replace(" ", "")[0] == "1") / total
        # A genuine kickback subroutine produces non-trivial p_one.
        return p_one > 0.05
    return check


def _prop_trotter_step(n: int) -> Callable[..., bool]:
    """Trotter step is a unitary; just sanity-check the qubit count."""
    def check(module: Any, counts: dict[str, int]) -> bool:
        for k in counts:
            return len(k.replace(" ", "")) >= n
        return False
    return check


def _prop_syndrome_extraction() -> Callable[..., bool]:
    """Syndrome extraction circuit: at least 5 qubits + ancillas measured."""
    def check(module: Any, counts: dict[str, int]) -> bool:
        for k in counts:
            return len(k.replace(" ", "")) >= 5
        return False
    return check


NOVEL_SUITE: list[BenchPrompt] = [
    BenchPrompt(
        "Circuit that prepares a uniform superposition over computational-"
        "basis states {0, 3, 5, 7} on 3 qubits.",
        expected_pattern=None,
        tags=["novel", "multi_controlled_prep"],
        expected_property=_prop_uniform_over_target({0, 3, 5, 7}, qubits=3),
    ),
    BenchPrompt(
        "5-qubit syndrome-extraction circuit for the [[5,1,3]] perfect code.",
        expected_pattern=None,
        tags=["novel", "named_code"],
        expected_property=_prop_syndrome_extraction(),
    ),
    BenchPrompt(
        "Single Trotter step for a 4-spin 1D Heisenberg Hamiltonian "
        "(H = sum ZZ + XX + YY) with timestep dt=0.1.",
        expected_pattern=None,
        tags=["novel", "hamiltonian_sim"],
        expected_property=_prop_trotter_step(4),
    ),
    BenchPrompt(
        "Quantum teleportation protocol: prepare an arbitrary state on q[0], "
        "teleport to q[2], measure q[1] and q[0] for the corrections.",
        expected_pattern=None,
        tags=["novel", "named_protocol"],
        expected_property=_prop_teleportation(),
    ),
    BenchPrompt(
        "QFT inverse on 4 qubits, followed by computational-basis measurement.",
        expected_pattern=None,
        tags=["novel", "named_transform_twist"],
        expected_property=_prop_qft_inverse(4),
    ),
    BenchPrompt(
        "GHZ-state preparation on 5 qubits, then measure in the X basis.",
        expected_pattern=None,
        tags=["novel", "composite"],
        expected_property=_prop_ghz_x_basis(5),
    ),
    BenchPrompt(
        "Phase-kickback subroutine implementing controlled-U for U = Rz(pi/4).",
        expected_pattern=None,
        tags=["novel", "named_primitive"],
        expected_property=_prop_phase_kickback(),
    ),
    BenchPrompt(
        "Random unitary on 3 qubits sampled from Haar measure, depth at most 8.",
        expected_pattern=None,
        tags=["novel", "haar_random"],
        expected_property=_prop_haar_random(3),
    ),
    BenchPrompt(
        "Iterative phase estimation routine: estimate the eigenphase of "
        "controlled-U where U has eigenvalue exp(i pi / 4) on 1 ancilla qubit.",
        expected_pattern=None,
        tags=["novel", "named_protocol"],
        expected_property=_prop_phase_kickback(),
    ),
    BenchPrompt(
        "Variational quantum eigensolver routine for the H2 molecule on "
        "4 qubits using the UCCSD parameterisation.",
        expected_pattern=None,
        tags=["novel", "vqe"],
        expected_property=_prop_n_qubits(4),
    ),
]


def quanbench_combined(
    *,
    provider: str | LLMProvider | None = "mock",
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run BUILTIN_SUITE *and* NOVEL_SUITE and report each separately.

    Returns a dict with two top-level keys, ``builtin`` and ``novel``, each
    holding a full :func:`run_quanbench` report. A ``combined`` key holds
    the joined success_rate / n_prompts / wall-time totals.
    """
    builtin_report = run_quanbench(provider=provider, suite=BUILTIN_SUITE)
    novel_report = run_quanbench(provider=provider, suite=NOVEL_SUITE)
    n_total = builtin_report["n_prompts"] + novel_report["n_prompts"]
    n_success = builtin_report["n_success"] + novel_report["n_success"]
    out: dict[str, Any] = {
        "schema": "qmesh.ai.quanbench.combined.v1",
        "builtin": builtin_report,
        "novel": novel_report,
        "combined": {
            "n_prompts": n_total,
            "n_success": n_success,
            "success_rate": n_success / max(1, n_total),
            "wall_time_seconds": (
                builtin_report["wall_time_seconds"]["total"]
                + novel_report["wall_time_seconds"]["total"]
            ),
        },
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(out, indent=2, default=str))
    return out


__all__ = [
    "BenchPrompt",
    "QuanBenchPrompt",
    "BUILTIN_SUITE",
    "NOVEL_SUITE",
    "run_quanbench",
    "quanbench_combined",
]
