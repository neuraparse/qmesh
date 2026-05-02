"""qmesh.frontends.qbraid — qBraid SDK ↔ qmesh.ir.

Phase 6β SDK plugin. qBraid (https://github.com/qBraid/qBraid) provides
*transpiler* utilities that convert across many quantum SDKs by going through
its own ``qbraid.programs`` IR. We integrate at that layer:

  * ``from_qbraid(program) -> Module`` — accept any qbraid-wrapped program
    (Cirq, Qiskit, Braket, PyQuil, OpenQASM3 source) and route through
    qmesh's existing frontends. We use qbraid to detect the program kind
    and to optionally transpile to OpenQASM 3 as a normal-form intermediate.

  * ``from_program_spec(spec) -> Module`` — accept a qBraid ``ProgramSpec``
    (produced by `qbraid.programs.load_program`) directly.

The implementation lazy-imports `qbraid` and falls back to a clear runtime
error when missing.
"""

from __future__ import annotations

from typing import Any

from qmesh.ir.module import Module


def _qbraid_available() -> bool:
    try:
        import qbraid  # noqa: F401
        return True
    except ImportError:
        return False


def from_qbraid(program: Any, *, name: str | None = None) -> Module:
    """Convert any qBraid-supported quantum program to qmesh IR.

    Strategy:
      1. Use ``qbraid.transpiler.transpile`` to lower the input program to
         OpenQASM 3 source.
      2. Hand that source to :func:`qmesh.frontends.qasm3.parse`.

    This avoids re-implementing every SDK-specific lowering and inherits
    qBraid's transpiler coverage.
    """
    if not _qbraid_available():
        raise RuntimeError(
            "qbraid not installed; `pip install qbraid>=0.7` to enable."
        )
    from qbraid.transpiler import transpile  # type: ignore[import-not-found]
    from qmesh.frontends.qasm3 import parse as parse_qasm3

    qasm_src = transpile(program, "qasm3")
    module = parse_qasm3(qasm_src)
    if name and module.functions:
        module.functions[0].name = name
    return module


def from_program_spec(spec: Any) -> Module:
    """Lower a `qbraid.programs.ProgramSpec` to a qmesh Module.

    Convenience wrapper that pulls the underlying program out and delegates
    to :func:`from_qbraid`.
    """
    if not _qbraid_available():
        raise RuntimeError(
            "qbraid not installed; `pip install qbraid>=0.7` to enable."
        )
    program = getattr(spec, "program", spec)
    return from_qbraid(program, name=getattr(spec, "name", None))


__all__ = ["from_qbraid", "from_program_spec"]
