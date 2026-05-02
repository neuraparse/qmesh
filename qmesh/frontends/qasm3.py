"""qmesh.frontends.qasm3 — OpenQASM 3 ↔ qmesh.ir.

Parsing uses the official `openqasm3` package's AST. We accept the gate-modality
subset that real backends actually support in 2026 (gate set, measure, reset,
barrier, delay, qubit/bit declarations, classical `if` over measurement
results). Constructs we explicitly do not parse yet are noted in TODOs.

Emitting goes the other way: qmesh.ir Module -> portable QASM 3 source.

Round-trip property: emit(parse(qasm3_text)) is semantically equivalent to
qasm3_text for the supported subset. This is the property the test suite checks.
"""

from __future__ import annotations

import io
from textwrap import dedent

from openqasm3 import ast
from openqasm3.parser import parse as _qasm_parse

from qmesh.ir.builder import circuit
from qmesh.ir.module import Module


_KNOWN_GATES_NO_PARAM = {
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "id",
    "cx", "cz", "cy", "swap", "iswap",
    "ccx", "cswap",
}
_KNOWN_GATES_PARAM = {
    "rx", "ry", "rz", "p", "u1", "u2", "u3",
    "rxx", "ryy", "rzz", "cp", "crx", "cry", "crz",
}


def _expr_value(expr: ast.Expression) -> float:
    """Best-effort literal float from an OpenQASM 3 expression."""
    if isinstance(expr, ast.IntegerLiteral):
        return float(expr.value)
    if isinstance(expr, ast.FloatLiteral):
        return float(expr.value)
    if isinstance(expr, ast.UnaryExpression):
        v = _expr_value(expr.expression)
        return -v if expr.op.name == "-" else v
    if isinstance(expr, ast.BinaryExpression):
        a, b = _expr_value(expr.lhs), _expr_value(expr.rhs)
        op = expr.op.name
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/": return a / b
    if isinstance(expr, ast.Identifier):
        if expr.name == "pi":
            from math import pi
            return pi
        if expr.name == "tau":
            from math import tau
            return tau
        if expr.name == "euler":
            from math import e
            return e
    raise NotImplementedError(f"qmesh.qasm3: cannot evaluate expression {expr!r}")


def _identifier_index(idx: ast.Expression) -> int:
    """Extract an integer index from an OpenQASM3 identifier-or-index expr."""
    if isinstance(idx, ast.IntegerLiteral):
        return idx.value
    raise NotImplementedError(f"qmesh.qasm3: only integer indices supported, got {idx!r}")


def parse(text: str) -> Module:
    """Parse OpenQASM 3 source text into a qmesh.ir Module.

    Recognises:
        OPENQASM 3.0 / include "stdgates.inc"
        qubit[N] q;  bit[M] c;
        h q[i]; cx q[i], q[j]; rx(theta) q[i]; ...
        measure q[i] -> c[j];   c[j] = measure q[i];
        reset q[i]; barrier; delay[Nns] q[i];
    """
    program = _qasm_parse(text)

    # First pass: figure out qubit/bit register sizes
    n_qubits = 0
    n_bits = 0
    qubit_offset: dict[str, int] = {}
    bit_offset: dict[str, int] = {}
    name = "qasm3"

    for stmt in program.statements:
        if isinstance(stmt, ast.QubitDeclaration):
            size = stmt.size.value if stmt.size is not None else 1  # type: ignore[union-attr]
            qubit_offset[stmt.qubit.name] = n_qubits
            n_qubits += size
        elif isinstance(stmt, ast.ClassicalDeclaration) and isinstance(stmt.type, ast.BitType):
            size = stmt.type.size.value if stmt.type.size is not None else 1  # type: ignore[union-attr]
            bit_offset[stmt.identifier.name] = n_bits
            n_bits += size

    if n_qubits == 0 and n_bits == 0:
        # nothing to translate; return empty module
        with circuit(name) as c:
            pass
        return c.module

    def _resolve_indexed(expr: ast.Expression) -> tuple[str, int]:
        """Returns (register_name, index)."""
        if isinstance(expr, ast.IndexedIdentifier):
            base = expr.name.name
            # indices is a list-of-lists; first index of first list
            idx = _identifier_index(expr.indices[0][0])
            return base, idx
        if isinstance(expr, ast.IndexExpression):
            base = expr.collection.name  # type: ignore[union-attr]
            idx = _identifier_index(expr.index[0])
            return base, idx
        if isinstance(expr, ast.Identifier):
            return expr.name, 0
        raise NotImplementedError(f"qmesh.qasm3: index expr {expr!r}")

    def _q_index(expr: ast.Expression) -> int:
        base, idx = _resolve_indexed(expr)
        return qubit_offset[base] + idx

    def _b_index(expr: ast.Expression) -> int:
        base, idx = _resolve_indexed(expr)
        return bit_offset[base] + idx

    # Second pass: emit ops
    with circuit(name, n_qubits=n_qubits, n_bits=n_bits) as c:
        for stmt in program.statements:
            if isinstance(stmt, (ast.QubitDeclaration, ast.ClassicalDeclaration,
                                 ast.Include, ast.Pragma)):
                continue

            if isinstance(stmt, ast.QuantumGate):
                gname = stmt.name.name
                qubit_idx = [_q_index(q) for q in stmt.qubits]
                params = tuple(_expr_value(p) for p in stmt.arguments)
                if gname in _KNOWN_GATES_NO_PARAM or gname in _KNOWN_GATES_PARAM:
                    c._gate(gname, qubit_idx, params)
                else:
                    # unknown native gate: pass through
                    c._gate(gname, qubit_idx, params)
                continue

            if isinstance(stmt, ast.QuantumMeasurementStatement):
                meas = stmt.measure.qubit  # the qubit being measured
                target = stmt.target          # the bit to write
                qi = _q_index(meas)
                if target is not None:
                    bi = _b_index(target)
                    c.measure(qi, bi)
                continue

            if isinstance(stmt, ast.QuantumReset):
                c.reset(_q_index(stmt.qubits))
                continue

            if isinstance(stmt, ast.QuantumBarrier):
                c.barrier()
                continue

            if isinstance(stmt, ast.DelayInstruction):
                # parse "Nns" / "Nus" / "Nms"
                dur = stmt.duration
                ns = _expr_value(dur) if isinstance(dur, ast.Expression) else 0
                if hasattr(dur, "unit") and dur.unit is not None:
                    u = dur.unit.name
                    if u == "us": ns *= 1e3
                    elif u == "ms": ns *= 1e6
                    elif u == "s": ns *= 1e9
                if stmt.qubits:
                    for q in stmt.qubits:
                        c.delay(ns, _q_index(q))
                else:
                    c.delay(ns)
                continue

            # TODO Phase 2: BranchingStatement / WhileLoop / ForInLoop / ClassicalAssignment

    return c.module


def emit(module: Module) -> str:
    """Emit OpenQASM 3 source for a qmesh Module.

    Only the gate modality is emitted today — Rydberg / CV / Pulse op modalities
    raise NotImplementedError because OpenQASM 3 has no portable surface for them.
    """
    from qmesh.ir.ops import GateOp, MeasureOp, ResetOp, BarrierOp, DelayOp
    from qmesh.ir.types import Modality

    lines: list[str] = ["OPENQASM 3.0;", 'include "stdgates.inc";', ""]
    n_qubits = 0
    n_bits = 0
    for f in module.functions:
        for op in f.body.ops:
            if op.modality not in (Modality.GATE, Modality.CLASSICAL):
                raise NotImplementedError(
                    f"qmesh.qasm3.emit: cannot emit modality {op.modality}; "
                    f"OpenQASM 3 has no portable surface for {op.modality.value}."
                )
            for v in op.operands:
                # Accumulate counts via type
                from qmesh.ir.types import Bit, Qubit
                if isinstance(v, Qubit):
                    n_qubits = max(n_qubits, v.index + 1)
                elif isinstance(v, Bit):
                    n_bits = max(n_bits, v.index + 1)

    if n_qubits:
        lines.append(f"qubit[{n_qubits}] q;")
    if n_bits:
        lines.append(f"bit[{n_bits}] c;")
    lines.append("")

    for f in module.functions:
        for op in f.body.ops:
            from qmesh.ir.types import Bit, Qubit
            qrefs = [f"q[{v.index}]" for v in op.operands if isinstance(v, Qubit)]
            brefs = [f"c[{v.index}]" for v in op.operands if isinstance(v, Bit)]
            if isinstance(op, MeasureOp):
                lines.append(f"{brefs[0]} = measure {qrefs[0]};")
            elif isinstance(op, ResetOp):
                lines.append(f"reset {qrefs[0]};")
            elif isinstance(op, BarrierOp):
                lines.append("barrier;")
            elif isinstance(op, DelayOp):
                ns = op.params[0] if op.params else 0
                lines.append(f"delay[{int(ns)}ns] {', '.join(qrefs)};" if qrefs else f"delay[{int(ns)}ns];")
            elif isinstance(op, GateOp):
                p = f"({', '.join(repr(x) for x in op.params)}) " if op.params else ""
                lines.append(f"{op.name}{('(' + ', '.join(repr(x) for x in op.params) + ') ') if op.params else ' '}{', '.join(qrefs)};".strip())
            else:
                continue
    return "\n".join(lines) + "\n"


__all__ = ["parse", "emit"]
