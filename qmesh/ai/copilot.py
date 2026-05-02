"""qmesh.ai.copilot — natural-language → IR with QCoder-style validation.

Pipeline:
  prompt → intent compiler (rule-based) → if hit, return IR directly
        → otherwise LLM provider → QASM 3 → parse → IR
        → simulate (`qmesh.aer`)  → property checks → optionally re-prompt

Each draft is captured with full provenance: prompt text, provider name,
model version, intent pattern (if matched), iteration count, validation
results. The Module's `metadata` carries this so it lands in the manifest
when the user runs the circuit.

QCoder-style validation: lightweight property checks for common cases
(GHZ, Bell, parity, sample-distribution shape). Phase 5β work: structured
output validation via constrained LLM decoding, full RLHF feedback loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from qmesh.ai.intent import compile_intent
from qmesh.ai.llm import LLMProvider, auto_provider
from qmesh.frontends.qasm3 import parse as parse_qasm3
from qmesh.ir.module import Module

SYSTEM_PROMPT = """You are a quantum-circuit assistant. Given a natural-language
request, produce **OpenQASM 3.0** source for a circuit that satisfies it. Output
only the QASM source, no prose, no markdown fences. Use the standard gate library
via `include "stdgates.inc";`. Always declare `qubit[N] q;` and `bit[M] c;`,
apply gates, then measure.
"""


@dataclass(slots=True)
class DraftResult:
    """Output of `draft()`: an IR Module + full provenance trace."""

    module: Module
    description: str
    iterations: int
    provider: str
    model: str
    intent_pattern: str | None
    qasm_source: str | None
    validation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _extract_qasm(text: str) -> str:
    """Pull QASM out of LLM output (strip markdown fences, etc.)."""
    text = text.strip()
    # ```qasm ... ``` fences
    m = re.search(r"```(?:qasm3?|openqasm)?\s*\n(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1).strip()
    # Remove leading lines without OPENQASM header
    if "OPENQASM" not in text.split("\n", 5)[0]:
        # Find the OPENQASM line
        idx = text.find("OPENQASM")
        if idx >= 0:
            text = text[idx:]
    return text


def _ghz_validator(module: Module, n_expected: int) -> dict[str, Any]:
    """Validate that the module produces a GHZ-shaped count distribution."""
    import qmesh
    result, _ = qmesh.submit(module, backend="qmesh.aer", shots=1024,
                            sign=False, ledger_dir="ledger/ai_validate")
    total = sum(result.counts.values())
    good = "0" * n_expected
    bad = "1" * n_expected
    p_ghz = (result.counts.get(good, 0) + result.counts.get(bad, 0)) / total
    return {"backend": "qmesh.aer", "shots": total, "p_ghz_pattern": p_ghz,
            "passed": p_ghz > 0.95}


def _bell_validator(module: Module) -> dict[str, Any]:
    import qmesh
    result, _ = qmesh.submit(module, backend="qmesh.aer", shots=1024,
                            sign=False, ledger_dir="ledger/ai_validate")
    total = sum(result.counts.values())
    p_even = (result.counts.get("00", 0) + result.counts.get("11", 0)) / total
    return {"backend": "qmesh.aer", "shots": total, "p_even_parity": p_even,
            "passed": p_even > 0.95}


def _generic_validator(module: Module) -> dict[str, Any]:
    """Catch-all: just verify the circuit parses & runs without error."""
    import qmesh
    try:
        result, _ = qmesh.submit(module, backend="qmesh.aer", shots=256,
                                sign=False, ledger_dir="ledger/ai_validate")
        return {"backend": "qmesh.aer", "shots": result.shots,
                "n_outcomes": len(result.counts), "passed": True}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "passed": False}


def _select_validator(prompt: str) -> Callable[[Module], dict[str, Any]]:
    p_lower = prompt.lower()
    if "bell" in p_lower:
        return _bell_validator
    m = re.search(r"ghz.*?(\d+)", p_lower)
    if m:
        n = int(m.group(1))
        return lambda module: _ghz_validator(module, n)
    return _generic_validator


def draft(
    prompt: str,
    *,
    provider: LLMProvider | None = None,
    max_iterations: int = 2,
    validate: bool = True,
) -> DraftResult:
    """Draft a circuit from a natural-language prompt.

    Pipeline:
      1. Try the rule-based intent compiler (fast, deterministic).
      2. If it doesn't match, call the LLM provider for QASM 3.
      3. Parse + simulate + property-check.
      4. If validation fails and we have iterations left, re-prompt with
         the failure context.
    """
    notes: list[str] = []
    intent = compile_intent(prompt)
    if intent is not None:
        notes.append(f"intent compiler matched pattern '{intent.pattern}'")
        validation = {}
        if validate:
            validation = _select_validator(prompt)(intent.module)
        return DraftResult(
            module=intent.module, description=intent.description,
            iterations=0, provider="intent-rules", model="qmesh.ai.intent",
            intent_pattern=intent.pattern, qasm_source=None,
            validation=validation, notes=notes,
        )

    prov = provider or auto_provider()
    notes.append(f"intent compiler did not match; using LLM provider {prov.name}")

    last_qasm = ""
    last_validation: dict[str, Any] = {}
    for it in range(max_iterations):
        sys = SYSTEM_PROMPT
        user = prompt
        if it > 0 and last_validation and not last_validation.get("passed", False):
            user = (
                f"Previous draft did not satisfy validation: {last_validation}.\n"
                f"Original request: {prompt}\n"
                f"Please produce a corrected QASM 3.0 program."
            )
        completion = prov.complete(user, system=sys)
        last_qasm = _extract_qasm(completion.text)
        try:
            module = parse_qasm3(last_qasm)
        except Exception as e:  # noqa: BLE001
            notes.append(f"iter {it}: parse failed ({e!r}); retrying")
            last_validation = {"passed": False, "parse_error": str(e)}
            continue

        if not validate:
            return DraftResult(
                module=module, description=prompt, iterations=it + 1,
                provider=completion.provider, model=completion.model,
                intent_pattern=None, qasm_source=last_qasm,
                validation={}, notes=notes,
            )
        last_validation = _select_validator(prompt)(module)
        if last_validation.get("passed", False):
            notes.append(f"iter {it}: validation passed")
            return DraftResult(
                module=module, description=prompt, iterations=it + 1,
                provider=completion.provider, model=completion.model,
                intent_pattern=None, qasm_source=last_qasm,
                validation=last_validation, notes=notes,
            )
        notes.append(f"iter {it}: validation failed: {last_validation}")

    raise RuntimeError(
        f"copilot exhausted {max_iterations} iterations without passing "
        f"validation. Last validation: {last_validation}\n"
        f"Last QASM:\n{last_qasm[:500]}"
    )


__all__ = ["DraftResult", "draft"]
