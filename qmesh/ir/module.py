"""qmesh.ir.module — Module / Function / Region containers + canonical hashing."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from qmesh.ir.ops import Op
from qmesh.ir.types import QType


@dataclass(slots=True)
class Region:
    """Sequence of ops; may nest for control flow."""

    ops: list[Op] = field(default_factory=list)
    # nested regions appear as ClassicalOp.attrs["body"] = Region
    label: str | None = None

    def append(self, op: Op) -> None:
        self.ops.append(op)

    def structural_digest(self) -> bytes:
        """Hash of op sequence only — no label."""
        h = sha256()
        for op in self.ops:
            h.update(op.digest())
            for v in op.attrs.values():
                if isinstance(v, Region):
                    h.update(v.structural_digest())
        return h.digest()

    def digest(self) -> bytes:
        """Full digest including label."""
        h = sha256()
        if self.label:
            h.update(self.label.encode())
        h.update(self.structural_digest())
        return h.digest()


@dataclass(slots=True)
class Function:
    name: str
    inputs: tuple[QType, ...] = field(default_factory=tuple)
    outputs: tuple[QType, ...] = field(default_factory=tuple)
    body: Region = field(default_factory=Region)
    params: dict[str, float] = field(default_factory=dict)

    def digest(self) -> bytes:
        h = sha256()
        h.update(self.name.encode())
        for v in self.inputs:
            h.update(repr(v).encode())
        for v in self.outputs:
            h.update(repr(v).encode())
        for k in sorted(self.params):
            h.update(k.encode())
            h.update(repr(self.params[k]).encode())
        h.update(self.body.digest())
        return h.digest()


@dataclass(slots=True)
class Module:
    """Top-level container. Hash is reproducible across runs and machines."""

    functions: list[Function] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, f: Function) -> Function:
        self.functions.append(f)
        return f

    def hash(self) -> str:
        h = sha256()
        for f in self.functions:
            h.update(f.digest())
        for k in sorted(self.metadata):
            h.update(k.encode())
            h.update(repr(self.metadata[k]).encode())
        return h.hexdigest()

    def semantic_hash(self) -> str:
        """Hash that excludes cosmetic naming (function name, region labels).

        Two modules with the same structure but different `Function.name` /
        region labels (the case across frontends) hash identically under
        `semantic_hash`. The full `hash()` includes naming so it doubles as
        an identity check including provenance.
        """
        h = sha256()
        for f in self.functions:
            h.update(f.body.structural_digest())
            for k in sorted(f.params):
                h.update(k.encode())
                h.update(repr(f.params[k]).encode())
        return h.hexdigest()

    def to_text(self) -> str:
        """Pretty-print the IR. Stable for diffing."""
        out: list[str] = [f"module @{self.hash()[:12]} {{"]
        for f in self.functions:
            out.append(f"  func @{f.name}({', '.join(str(i) for i in f.inputs)}) " f"-> ({', '.join(str(o) for o in f.outputs)}) {{")
            for op in f.body.ops:
                operands = ", ".join(str(o) for o in op.operands)
                params = ", ".join(repr(p) for p in op.params)
                attrs = ""
                if op.attrs:
                    attrs = " {" + ", ".join(f"{k}={v!r}" for k, v in sorted(op.attrs.items())) + "}"
                out.append(f"    {op.modality.value}.{op.name} {operands}" + (f" : ({params})" if params else "") + attrs)
            out.append("  }")
        out.append("}")
        return "\n".join(out)


__all__ = ["Module", "Function", "Region"]
