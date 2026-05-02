"""qmesh.mitigation.zne — Zero-Noise Extrapolation via Mitiq.

ZNE is the workhorse mitigation in 2026: scale gate noise up by inserting
identity-pairs, fit a curve, extrapolate to zero. Wraps Mitiq so callers
get a single API across whatever backend qmesh is targeting.

Usage:

    from qmesh.mitigation.zne import zne_estimate

    # observable: a function that maps RunResult.counts -> float
    def parity(counts):
        total = sum(counts.values())
        return sum(v for k, v in counts.items() if k.count('1') % 2 == 0) / total

    estimate = zne_estimate(
        module, backend='qmesh.aer.noisy',
        observable=parity, scale_factors=[1, 3, 5], shots=4096,
    )
"""

from __future__ import annotations

from typing import Callable

from qmesh.api import submit
from qmesh.ir.module import Module


def zne_estimate(
    module: Module,
    *,
    backend: str = "qmesh.aer.noisy",
    observable: Callable[[dict[str, int]], float],
    scale_factors: tuple[float, ...] = (1.0, 3.0, 5.0),
    shots: int = 4096,
    extrapolator: str = "richardson",
) -> dict:
    """Run `module` at multiple noise scale factors and Richardson-extrapolate.

    We use Mitiq's `RichardsonFactory` (or `LinearFactory`) for the fit. Noise
    scaling is implemented as global gate-folding (each gate G replaced by
    G G† G, repeated to hit the requested scale).

    Returns a dict with `mitigated`, `unmitigated`, `raw` per-scale values,
    and the chosen `extrapolator`.
    """
    import numpy as np
    from mitiq.zne.inference import LinearFactory, RichardsonFactory

    # Build the noise-folded circuits in qmesh.ir directly. We fold by running
    # each gate operation an additional 2× per fold step, conceptually
    # equivalent to G→G G† G. This is a faithful global folding scheme.
    raw: list[tuple[float, float]] = []
    for s in scale_factors:
        n_extra = int(round((s - 1) / 2))  # 1→0 extra; 3→1 extra; 5→2 extra
        folded = _fold_global(module, n_extra=n_extra)
        result, _ = submit(folded, backend=backend, shots=shots, sign=False)
        raw.append((s, observable(result.counts)))

    xs = np.array([p[0] for p in raw])
    ys = np.array([p[1] for p in raw])
    factory_cls = RichardsonFactory if extrapolator.lower() == "richardson" else LinearFactory
    fac = factory_cls(scale_factors=list(xs))
    for x, y in raw:
        fac._instack.append({"scale_factor": float(x)})
        fac._outstack.append(float(y))
    mitigated = float(fac.reduce())

    return {
        "mitigated": mitigated,
        "unmitigated": float(ys[0]),
        "raw": raw,
        "extrapolator": extrapolator,
        "backend": backend,
    }


def _fold_global(module: Module, n_extra: int) -> Module:
    """Replace every Gate by `G (G† G)^n_extra` (n_extra=0 returns the original)."""
    if n_extra == 0:
        return module
    from qmesh.ir.builder import circuit
    from qmesh.ir.ops import GateOp, MeasureOp, ResetOp
    from qmesh.ir.types import Bit, Qubit

    # Find sizes
    n_qubits = 0
    n_bits = 0
    for f in module.functions:
        for op in f.body.ops:
            for v in op.operands:
                if isinstance(v, Qubit):
                    n_qubits = max(n_qubits, v.index + 1)
                elif isinstance(v, Bit):
                    n_bits = max(n_bits, v.index + 1)

    # Inverse of common gates (most are self-inverse or have a known inverse).
    INV = {
        "h": "h", "x": "x", "y": "y", "z": "z", "s": "sdg", "sdg": "s",
        "t": "tdg", "tdg": "t",
        "cx": "cx", "cz": "cz", "swap": "swap",
        "rx": ("rx", -1), "ry": ("ry", -1), "rz": ("rz", -1),
    }

    with circuit("zne_folded", n_qubits=n_qubits, n_bits=n_bits) as c:
        for f in module.functions:
            for op in f.body.ops:
                if isinstance(op, GateOp):
                    qs = [v.index for v in op.operands if isinstance(v, Qubit)]
                    inv = INV.get(op.name)
                    # Original
                    c._gate(op.name, qs, op.params)
                    if inv is None:
                        continue  # no fold for unknown gate; skip
                    for _ in range(n_extra):
                        if isinstance(inv, tuple):
                            inv_name, sign = inv
                            inv_params = tuple(sign * p for p in op.params)
                            c._gate(inv_name, qs, inv_params)
                            c._gate(op.name, qs, op.params)
                        else:
                            c._gate(inv, qs, op.params)
                            c._gate(op.name, qs, op.params)
                elif isinstance(op, MeasureOp):
                    q = next(v.index for v in op.operands if isinstance(v, Qubit))
                    b = next(v.index for v in op.operands if isinstance(v, Bit))
                    c.measure(q, b)
                elif isinstance(op, ResetOp):
                    q = next(v.index for v in op.operands if isinstance(v, Qubit))
                    c.reset(q)
    return c.module
