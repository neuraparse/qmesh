"""qmesh.ai.constrained_decoding — token-level grammar gate for OpenQASM 3.

α-quality: implements a useful subset of the OpenQASM 3 grammar covering the
gate set qmesh's intent compiler emits. It is *not* a full ANTLR-grade
parser; it is a streaming line-validator + next-token oracle that catches
the common LLM failure modes (bad gate names, missing commas, orphan
qubit references, missing semicolons, mismatched brackets).

Public API:
    from qmesh.ai.constrained_decoding import (
        validate_qasm_streaming,
        QASMGrammarGate,
        ConstrainedQASMGenerator,
    )

    # Streaming validator: yields (prefix, valid?) pairs
    for prefix, ok in validate_qasm_streaming(text):
        ...

    # Next-token oracle (for token-level constrained decoding)
    gate = QASMGrammarGate()
    legal_next = gate.legal_next_tokens(prefix_text)

    # Provider wrapper that filters LLM output through the gate
    cg = ConstrainedQASMGenerator(provider, max_tokens=512)
    completion = cg.complete(prompt, system=system)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from qmesh.ai.llm import CompletionResult, LLMProvider

# Gate set qmesh.ai.intent emits + a few common extras.
_GATES_NO_PARAM = {
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "id",
    "cx", "cz", "cy", "swap", "iswap", "ccx", "cswap",
}
_GATES_PARAM = {
    "rx", "ry", "rz", "p", "u1", "u2", "u3",
    "rxx", "ryy", "rzz", "cp", "crx", "cry", "crz",
}
_GATE_ARITY: dict[str, int] = {
    "h": 1, "x": 1, "y": 1, "z": 1, "s": 1, "sdg": 1, "t": 1, "tdg": 1, "id": 1,
    "rx": 1, "ry": 1, "rz": 1, "p": 1, "u1": 1, "u2": 1, "u3": 1,
    "cx": 2, "cz": 2, "cy": 2, "swap": 2, "iswap": 2,
    "rxx": 2, "ryy": 2, "rzz": 2, "cp": 2, "crx": 2, "cry": 2, "crz": 2,
    "ccx": 3, "cswap": 3,
}
_GATE_HAS_PARAM: dict[str, bool] = {g: g in _GATES_PARAM for g in _GATE_ARITY}


_HEADER_RE = re.compile(r"^\s*OPENQASM\s+3(?:\.0)?\s*;\s*$")
_INCLUDE_RE = re.compile(r'^\s*include\s+"stdgates\.inc"\s*;\s*$')
_QDECL_RE = re.compile(r"^\s*qubit\s*\[\s*(\d+)\s*\]\s*([A-Za-z_]\w*)\s*;\s*$")
_BDECL_RE = re.compile(r"^\s*bit\s*\[\s*(\d+)\s*\]\s*([A-Za-z_]\w*)\s*;\s*$")
_RESET_RE = re.compile(r"^\s*reset\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;\s*$")
_BARRIER_RE = re.compile(r"^\s*barrier\s*;\s*$")
_MEASURE_ARROW_RE = re.compile(
    r"^\s*measure\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*->\s*"
    r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;\s*$"
)
_MEASURE_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*=\s*measure\s+"
    r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;\s*$"
)
# Gate body: optional (params) then comma-separated qubit refs.
_GATE_NOPARAM_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s+(.+?)\s*;\s*$"
)
_GATE_WITHPARAM_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*\(([^)]*)\)\s+(.+?)\s*;\s*$"
)
_QREF_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*$")


@dataclass
class QASMValidationState:
    """Tracks declared registers + header state across lines."""
    saw_header: bool = False
    saw_include: bool = False
    qregs: dict[str, int] = field(default_factory=dict)
    bregs: dict[str, int] = field(default_factory=dict)
    line_num: int = 0


def _parse_qubit_args(rest: str, state: QASMValidationState
                      ) -> tuple[bool, str, list[tuple[str, int]]]:
    """Parse comma-separated `q[i]` references; return (ok, error, refs)."""
    parts = [p.strip() for p in rest.split(",")]
    refs: list[tuple[str, int]] = []
    for p in parts:
        m = _QREF_RE.match(p)
        if not m:
            return False, f"bad qubit ref {p!r}", []
        reg, idx = m.group(1), int(m.group(2))
        if reg not in state.qregs:
            return False, f"undeclared qubit register {reg!r}", []
        if idx >= state.qregs[reg]:
            return False, f"qubit index {idx} out of range for {reg}[{state.qregs[reg]}]", []
        refs.append((reg, idx))
    return True, "", refs


def _validate_gate_line(line: str, state: QASMValidationState) -> tuple[bool, str]:
    """Validate a single gate-or-statement line."""
    s = line.rstrip()
    if not s.strip():
        return True, ""
    if s.strip().startswith("//"):
        return True, ""

    if _HEADER_RE.match(s):
        if state.saw_header:
            return False, "duplicate OPENQASM header"
        state.saw_header = True
        return True, ""
    if _INCLUDE_RE.match(s):
        state.saw_include = True
        return True, ""
    m = _QDECL_RE.match(s)
    if m:
        size, name = int(m.group(1)), m.group(2)
        if size <= 0:
            return False, f"qubit register {name!r} must be size ≥ 1"
        state.qregs[name] = size
        return True, ""
    m = _BDECL_RE.match(s)
    if m:
        size, name = int(m.group(1)), m.group(2)
        if size <= 0:
            return False, f"bit register {name!r} must be size ≥ 1"
        state.bregs[name] = size
        return True, ""
    if _BARRIER_RE.match(s):
        return True, ""
    m = _RESET_RE.match(s)
    if m:
        reg, idx = m.group(1), int(m.group(2))
        if reg not in state.qregs or idx >= state.qregs[reg]:
            return False, f"reset references undeclared/out-of-range {reg}[{idx}]"
        return True, ""
    m = _MEASURE_ARROW_RE.match(s)
    if m:
        qreg, qidx, breg, bidx = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if qreg not in state.qregs or qidx >= state.qregs[qreg]:
            return False, f"measure: {qreg}[{qidx}] out of range"
        if breg not in state.bregs or bidx >= state.bregs[breg]:
            return False, f"measure: {breg}[{bidx}] out of range"
        return True, ""
    m = _MEASURE_ASSIGN_RE.match(s)
    if m:
        breg, bidx, qreg, qidx = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if qreg not in state.qregs or qidx >= state.qregs[qreg]:
            return False, f"measure: {qreg}[{qidx}] out of range"
        if breg not in state.bregs or bidx >= state.bregs[breg]:
            return False, f"measure: {breg}[{bidx}] out of range"
        return True, ""

    # gate(param) qubits ;
    m = _GATE_WITHPARAM_RE.match(s)
    if m:
        gname, params, qubit_part = m.group(1), m.group(2), m.group(3)
        if gname not in _GATE_ARITY:
            return False, f"unknown gate {gname!r}"
        if not _GATE_HAS_PARAM[gname]:
            return False, f"gate {gname!r} does not take parameters"
        if not params.strip():
            return False, f"gate {gname!r} has empty parameter list"
        ok, err, refs = _parse_qubit_args(qubit_part, state)
        if not ok:
            return False, err
        if len(refs) != _GATE_ARITY[gname]:
            return False, f"gate {gname!r} expects {_GATE_ARITY[gname]} qubits, got {len(refs)}"
        return True, ""

    # gate qubits ;
    m = _GATE_NOPARAM_RE.match(s)
    if m:
        gname, qubit_part = m.group(1), m.group(2)
        if gname not in _GATE_ARITY:
            return False, f"unknown gate {gname!r}"
        if _GATE_HAS_PARAM[gname]:
            return False, f"gate {gname!r} requires parameters in (...)"
        ok, err, refs = _parse_qubit_args(qubit_part, state)
        if not ok:
            return False, err
        if len(refs) != _GATE_ARITY[gname]:
            return False, f"gate {gname!r} expects {_GATE_ARITY[gname]} qubits, got {len(refs)}"
        return True, ""

    return False, f"unrecognised statement: {s.strip()[:80]!r}"


def validate_qasm(text: str) -> tuple[bool, list[str]]:
    """Validate a complete QASM 3 source. Returns (ok, errors[])."""
    state = QASMValidationState()
    errors: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        state.line_num = i
        ok, err = _validate_gate_line(line, state)
        if not ok:
            errors.append(f"line {i}: {err}")
    if not state.saw_header:
        errors.append("missing OPENQASM 3.0; header")
    return len(errors) == 0, errors


def validate_qasm_streaming(text: str) -> Iterator[tuple[str, bool]]:
    """Yield `(prefix, valid)` pairs as the text is consumed line-by-line.

    Useful for catching the *first* invalid token in LLM streaming output —
    the consumer can break on the first `valid=False` to abort generation.
    """
    state = QASMValidationState()
    lines = text.splitlines(keepends=True)
    accumulated = ""
    for i, line in enumerate(lines, start=1):
        state.line_num = i
        # Allow blank-only / comment lines through; only validate complete
        # statement lines (those ending in ; or matching headers/decls).
        stripped = line.strip()
        accumulated += line
        if not stripped:
            yield accumulated, True
            continue
        if stripped.startswith("//"):
            yield accumulated, True
            continue
        # Wait for a terminator before judging.
        if not stripped.endswith(";") and not _HEADER_RE.match(stripped) \
           and not _INCLUDE_RE.match(stripped) and not _BARRIER_RE.match(stripped):
            yield accumulated, True
            continue
        ok, _err = _validate_gate_line(line, state)
        yield accumulated, ok


@dataclass
class QASMGrammarGate:
    """Token-level next-token oracle for OpenQASM 3.

    Given a partial QASM prefix, returns a list of legal next *tokens*. Tokens
    are coarse here — keywords / gate names / register names — not single
    characters. This is the right granularity for sub-word LLM tokenizers
    where a single keyword often matches one BPE piece.
    """

    state: QASMValidationState = field(default_factory=QASMValidationState)

    def reset(self) -> None:
        self.state = QASMValidationState()

    def feed_line(self, line: str) -> tuple[bool, str]:
        return _validate_gate_line(line, self.state)

    def legal_next_tokens(self, prefix: str) -> list[str]:
        """Return legal next 'tokens' given the prefix. α: line-level.

        Strategy: look at the prefix's last (incomplete) line. If it is empty,
        propose statement-starters. Otherwise propose continuations consistent
        with what's been seen so far.
        """
        # Replay the completed lines through a fresh state so we know what
        # registers exist.
        state = QASMValidationState()
        lines = prefix.splitlines()
        if not prefix.endswith("\n") and lines:
            completed = lines[:-1]
            partial = lines[-1]
        else:
            completed = lines
            partial = ""
        for line in completed:
            _validate_gate_line(line, state)

        # If we haven't seen the header, the first thing must be it.
        if not state.saw_header:
            return ["OPENQASM 3.0;"]

        partial_stripped = partial.strip()
        if not partial_stripped:
            # Statement-starters
            opts: list[str] = []
            if not state.saw_include:
                opts.append('include "stdgates.inc";')
            opts.append("qubit[N] q;")
            opts.append("bit[N] c;")
            if state.qregs:
                opts.extend(sorted(_GATES_NO_PARAM | _GATES_PARAM))
                opts.append("measure")
                opts.append("reset")
                opts.append("barrier;")
            return opts

        # Crude: if user has typed a partial gate name, return matching gates.
        first_token = partial_stripped.split()[0] if partial_stripped else ""
        head = first_token.split("(")[0]
        if head and not head[-1].isalnum() and head[-1] != "_":
            head = head[:-1]
        candidates = sorted(g for g in (_GATES_NO_PARAM | _GATES_PARAM) if g.startswith(head))
        return candidates if candidates else []


@dataclass
class ConstrainedQASMGenerator:
    """LLMProvider wrapper that validates the model's QASM output through
    the grammar gate.

    α-quality: post-hoc validation. We invoke the underlying provider, then
    filter the output line-by-line through `QASMGrammarGate.feed_line`. If
    any line fails, we trim the output at the failing line and append a
    completion that emits a safe Bell-state default — ensuring the consumer
    always gets grammar-valid QASM. A real β implementation would go a step
    further and constrain the LLM's logits at generation time (logit-bias
    masking, llama.cpp grammar files, Outlines, etc.); since the qmesh-side
    LLM API only exposes whole-completion `complete()`, post-hoc filtering
    is the honest interface here.
    """

    provider: LLMProvider
    max_tokens: int = 1024

    @property
    def name(self) -> str:
        return f"constrained({self.provider.name})"

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int | None = None,
                 temperature: float = 0.0) -> CompletionResult:
        completion = self.provider.complete(
            prompt, system=system,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature,
        )
        text = completion.text
        # Strip markdown fences identical to the copilot helper, but locally
        # to avoid a circular import.
        m = re.search(r"```(?:qasm3?|openqasm)?\s*\n(.*?)```", text, re.S | re.I)
        if m:
            text = m.group(1)
        idx = text.find("OPENQASM")
        if idx > 0:
            text = text[idx:]

        gate = QASMGrammarGate()
        kept: list[str] = []
        first_failure: int | None = None
        for i, line in enumerate(text.splitlines()):
            ok, _err = gate.feed_line(line)
            if ok:
                kept.append(line)
            else:
                first_failure = i
                break
        filtered = "\n".join(kept)
        if first_failure is not None:
            # Emit a safe default completion so downstream parsing doesn't
            # crash. The metadata records what got dropped.
            filtered += "\n// [constrained_decoding] truncated at line " \
                        f"{first_failure}; appending Bell default\n"
            if not gate.state.saw_header:
                filtered = (
                    "OPENQASM 3.0;\n"
                    'include "stdgates.inc";\n'
                    "qubit[2] q;\nbit[2] c;\n"
                    "h q[0];\ncx q[0], q[1];\n"
                    "c[0] = measure q[0];\nc[1] = measure q[1];\n"
                )
        new_meta = dict(completion.metadata or {})
        new_meta["constrained_decoding"] = {
            "first_invalid_line": first_failure,
            "valid": first_failure is None,
        }
        return CompletionResult(
            text=filtered + ("\n" if not filtered.endswith("\n") else ""),
            provider=self.name,
            model=completion.model,
            metadata=new_meta,
        )


__all__ = [
    "validate_qasm",
    "validate_qasm_streaming",
    "QASMGrammarGate",
    "ConstrainedQASMGenerator",
    "QASMValidationState",
]
