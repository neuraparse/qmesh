"""qmesh.frontends.mrmustard — MrMustard ↔ qmesh.ir.

Strawberry Fields entered maintenance mode in 2026; **MrMustard** is the
supported successor for Xanadu's photonic-CV stack. It is a differentiable
Gaussian + Fock engine designed around three high-level concepts:

    - States   (Vacuum, SqueezedVacuum, ...)        — sit on `modes`
    - Transformations  (Sgate, Dgate, BSgate, ...)  — applied to modes
    - Measurements    (Homodyne, Heterodyne,
                       Number, PNR)                 — terminal nodes

A MrMustard `Circuit` chains these via `>>` composition; each component
exposes `.modes` (a tuple of mode indices) and a name (its class name)
and parameter values (`.parameters`, often a dict of TensorFlow/jax
scalars, or `.params` on a real `Transformation`).

We translate:

    MrMustard Circuit → qmesh.ir Module with CVOps

mirroring `qmesh.frontends.sf` exactly, so the same `qmesh.sf.fock` /
`qmesh.sf.gaussian` backends already accept the lowered IR (the CV ops
share names: `Sgate`, `BSgate`, `Dgate`, `Rgate`, `MeasureFock`,
`MeasureHomodyne`).

α-quality: real MrMustard installs duck-type into the same shape. If
MrMustard isn't installed (today, on this host, May 2026 — Bloqade is
still pre-1.0 and MrMustard hasn't shipped to PyPI), tests use the
shim defined below — the same shape any `Circuit.components` walk
yields. Tests gate on `pytest.importorskip("mrmustard")`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from qmesh.ir.builder import circuit
from qmesh.ir.module import Module
from qmesh.ir.ops import CVOp, MeasureOp
from qmesh.ir.types import Bit, Modality

if TYPE_CHECKING:
    pass


# ---------- α-quality shim ----------

@dataclass(slots=True)
class MMComponent:
    """One MrMustard component (gate or measurement).

    `name` is the component class name, e.g. `"Sgate"` or `"MeasureFock"`.
    `modes` is a tuple of mode indices the component acts on.
    `params` is a tuple of scalar parameters (resolved to floats —
    differentiable parameters get `.numpy()`'d at export time, mirroring
    what MrMustard's own backend-export pipeline does).
    """
    name: str
    modes: tuple[int, ...]
    params: tuple[float, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class MMCircuit:
    """α-quality shim of a MrMustard `Circuit`.

    A real `mrmustard.lab.Circuit` lowers to the same representation via
    `circuit.components` (or `circuit.ops` on older versions); each
    component exposes `.modes` and a class name. Tests build one of
    these directly.
    """
    n_modes: int
    components: list[MMComponent] = field(default_factory=list)


# ---------- public translator ----------

def from_mrmustard(prog: Any, *, name: str = "mrmustard_program") -> Module:
    """Translate a MrMustard Circuit into a qmesh.ir Module.

    Accepts:
        - an `MMCircuit` shim (this file's dataclass)
        - a real `mrmustard.lab.Circuit` (duck-typed: walk `.components`
          / `.ops`, read `.modes` and `.parameters` on each)
    """
    n_modes, comps = _normalise(prog)
    n_bits = sum(1 for c in comps if c.name.startswith("Measure"))

    with circuit(name, n_qubits=0, n_bits=n_bits, n_qumodes=n_modes) as c:
        meas_idx = 0
        for comp in comps:
            mode_idx = list(comp.modes)
            if comp.name.startswith("Measure"):
                # one Bit per measured mode — match SF's pattern
                for mi in mode_idx:
                    c._region.append(MeasureOp(
                        operands=(c.qumodes[mi],
                                  Bit(name=f"c{meas_idx}", index=meas_idx)),
                        modality=Modality.CV,
                        attrs={"basis": comp.name, "frontend": "mrmustard"},
                    ))
                    meas_idx += 1
            else:
                # Standard CV op
                c._region.append(CVOp(
                    name=comp.name,
                    operands=tuple(c.qumodes[i] for i in mode_idx),
                    params=tuple(float(p) for p in comp.params),
                    modality=Modality.CV,
                    attrs={"frontend": "mrmustard"},
                ))

    c.module.metadata["frontend"] = "mrmustard"
    return c.module


# ---------- duck-type normaliser ----------

def _normalise(prog: Any) -> tuple[int, list[MMComponent]]:
    """Coerce a MrMustard-shaped object into (n_modes, components)."""
    if isinstance(prog, MMCircuit):
        return prog.n_modes, list(prog.components)

    # Real MrMustard Circuit
    n_modes = int(
        getattr(prog, "num_modes", None)
        or getattr(prog, "n_modes", None)
        or _max_mode_idx(prog) + 1
    )
    raw = getattr(prog, "components", None) or getattr(prog, "ops", None) or []
    comps: list[MMComponent] = []
    for c in raw:
        comps.append(MMComponent(
            name=type(c).__name__,
            modes=tuple(getattr(c, "modes", ())),
            params=_extract_params(c),
        ))
    return n_modes, comps


def _max_mode_idx(prog: Any) -> int:
    raw = getattr(prog, "components", None) or getattr(prog, "ops", None) or []
    m = -1
    for c in raw:
        for mi in getattr(c, "modes", ()):
            if int(mi) > m:
                m = int(mi)
    return m


def _extract_params(c: Any) -> tuple[float, ...]:
    """Best-effort scalar extraction from a MrMustard component.

    Real MrMustard parameters are either plain floats, `Variable`
    objects (with `.value`), or backend-tensor scalars (TF / jax /
    torch). We unwrap each of those to a Python float.
    """
    candidates: list[Any] = []
    p = getattr(c, "parameters", None)
    if p is not None:
        if isinstance(p, dict):
            candidates.extend(p.values())
        else:
            try:
                candidates.extend(list(p))
            except TypeError:
                candidates.append(p)
    raw_params = getattr(c, "params", None)
    if raw_params is not None and not candidates:
        try:
            candidates.extend(list(raw_params))
        except TypeError:
            candidates.append(raw_params)

    out: list[float] = []
    for v in candidates:
        if v is None:
            continue
        for getter in ("value", "numpy", "item", "_numpy"):
            if hasattr(v, getter):
                got = getattr(v, getter)
                v = got() if callable(got) else got
                break
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return tuple(out)


__all__ = [
    "from_mrmustard",
    "MMCircuit",
    "MMComponent",
]
