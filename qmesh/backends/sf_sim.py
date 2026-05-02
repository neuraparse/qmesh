"""qmesh.backends.sf_sim — Strawberry Fields backends.

Two registered profiles:
- `qmesh.sf.fock`     — Fock backend (truncated photon-number basis)
- `qmesh.sf.gaussian` — Gaussian backend (covariance-matrix simulation)

Round-trip path: qmesh.ir CVOps → SF Program → SF backend → counts.
"""

from __future__ import annotations

import time
from typing import Any

from qmesh.backends.base import Backend, Capabilities, ControlLevel, RunResult
from qmesh.backends.registry import register
from qmesh.ir.module import Module
from qmesh.ir.ops import CVOp, MeasureOp
from qmesh.ir.types import Modality, Qumode


def _ir_to_sf(module: Module):
    """Lower a CV-modality qmesh.ir Module into a SF Program."""
    import strawberryfields as sf
    from strawberryfields import ops as sf_ops

    # Collect modes
    modes: list[Qumode] = []
    for f in module.functions:
        for v in f.inputs:
            if isinstance(v, Qumode):
                modes.append(v)
    if not modes:
        raise ValueError("no Qumode operands found; not a CV-modality module")

    n_modes = len(modes)
    program = sf.Program(n_modes)
    with program.context as q:
        for f in module.functions:
            for op in f.body.ops:
                if isinstance(op, CVOp):
                    op_class = getattr(sf_ops, op.name, None)
                    if op_class is None:
                        raise NotImplementedError(
                            f"sf backend has no op '{op.name}'"
                        )
                    mode_idx = [v.index for v in op.operands if isinstance(v, Qumode)]
                    instance = op_class(*op.params)
                    if len(mode_idx) == 1:
                        instance | q[mode_idx[0]]  # noqa: B015
                    else:
                        instance | tuple(q[i] for i in mode_idx)  # noqa: B015
                elif isinstance(op, MeasureOp):
                    basis = op.attrs.get("basis", "MeasureFock")
                    op_class = getattr(sf_ops, basis, sf_ops.MeasureFock)
                    qm = next(v for v in op.operands if isinstance(v, Qumode))
                    op_class() | q[qm.index]  # noqa: B015
    return program


class SFBackend(Backend):
    def __init__(self, *, kind: str = "fock", cutoff_dim: int = 10) -> None:
        self._kind = kind
        self._cutoff_dim = cutoff_dim

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=f"qmesh.sf.{self._kind}",
            vendor="qmesh+strawberryfields",
            modalities={Modality.CV},
            qubit_count=8,                     # CV "modes"; Fock cutoff limits scale
            native_gates=set(),                # CV ops; named, not gate set
            measurement_feedforward=False,
            classical_control=ControlLevel.NONE,
            cost_per_shot_usd=0.0,
            queue_depth=0,
            fidelity_2q_typical=1.0,
            is_simulator=True,
            notes=f"Strawberry Fields {self._kind} backend"
                  f"{f' (cutoff={self._cutoff_dim})' if self._kind == 'fock' else ''}",
        )

    def run(self, module: Module, shots: int = 100, **_: Any) -> RunResult:
        import strawberryfields as sf

        ok, why = self.accepts(module)
        if not ok:
            raise ValueError(f"sf backend rejected module: {why}")

        program = _ir_to_sf(module)
        backend_kwargs: dict[str, Any] = {}
        if self._kind == "fock":
            backend_kwargs["cutoff_dim"] = self._cutoff_dim

        t0 = time.time()
        # SF engines hold per-run state; re-create per shot to keep statistics
        # IID. (Engine reuse via reset() is fragile across SF versions.)
        counts: dict[str, int] = {}
        for _ in range(shots):
            engine = sf.Engine(self._kind, backend_options=backend_kwargs)
            res = engine.run(program)
            samples = res.samples
            if samples is not None and len(samples):
                key = ",".join(str(int(s)) for s in samples[0])
            else:
                key = ""
            counts[key] = counts.get(key, 0) + 1
        wall = time.time() - t0

        return RunResult(
            counts=counts,
            shots=shots,
            wall_seconds=wall,
            qpu_seconds=wall,
            cost_usd=0.0,
            backend_metadata={
                "kind": self._kind,
                "cutoff_dim": self._cutoff_dim if self._kind == "fock" else None,
                "n_modes": program.num_subsystems,
            },
        )


register(SFBackend(kind="fock", cutoff_dim=8))
register(SFBackend(kind="gaussian"))
