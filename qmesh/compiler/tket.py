"""qmesh.compiler.tket — pytket pass wrapper.

Lowers qmesh.ir → pytket.Circuit, applies a named pass, lifts back.
Lazy import so qmesh works without pytket installed.

Available passes (all from `pytket.passes`):
    FullPeepholeOptimise   — aggressive optimisation
    CliffordSimp           — Clifford-only simplification
    DecomposeBoxes         — un-pack composite boxes
    RemoveRedundancies     — drop trivially identity gates
    SquashRZPhasedX        — Rebase to {RZ, PhasedX} (IBM/IonQ-friendly)
    SynthesiseTket         — heuristic re-synthesis
    PauliSimp              — Pauli-frame optimisation
"""

from __future__ import annotations

from qmesh.compiler import PassContext
from qmesh.ir.module import Module


def tket_pass(name: str, **kwargs: object):
    """Return a qmesh.compiler.Pass that applies the named pytket pass."""
    def run(module: Module, ctx: PassContext) -> Module:
        from pytket import passes as _tket_passes
        from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

        # NOTE: pytket's qasm helpers actually consume QASM 2 string; we use
        # qiskit as the bridge for portability.
        from qmesh.backends.aer_sim import _ir_to_qiskit
        from qmesh.frontends.qiskit import from_qiskit

        qc = _ir_to_qiskit(module)
        tk = qiskit_to_tk(qc)

        pass_class = getattr(_tket_passes, name, None)
        if pass_class is None:
            raise ValueError(
                f"unknown pytket pass '{name}'. "
                f"Examples: FullPeepholeOptimise, CliffordSimp, RemoveRedundancies"
            )
        pass_class(**kwargs).apply(tk)

        # back to qmesh.ir via qiskit
        qc_out = tk_to_qiskit(tk)
        out = from_qiskit(qc_out)
        ctx.log.append({
            "pass": f"tket.{name}",
            "kwargs": kwargs,
            "input_hash": module.hash(),
            "output_hash": out.hash(),
        })
        return out

    run.__name__ = f"tket_{name}"
    return run
