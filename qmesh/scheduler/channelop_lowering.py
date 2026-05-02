"""qmesh.scheduler.channelop_lowering — first-class ChannelOp → DAG.

Phase 3β α: lower a single qmesh.ir Module that mixes modalities (gate
+ rydberg + cv + pulse ...) joined by `ChannelOp`s into a
`qmesh.scheduler.DAG` that the existing executor can run end-to-end.

Algorithm:

    1. Walk the Module's ops in order. Partition them into "modality-
       coherent blocks" — maximal runs of ops whose modality matches the
       block's modality. ChannelOps split blocks; classical / barrier /
       delay ops attach to the surrounding block.

    2. For each non-ChannelOp block, emit a `QPUPrimitive` whose
       `module_factory` rebuilds a single-modality Module from the block
       (so per-node manifests sign just that block's IR hash). Pick a
       backend per modality:
           gate    → qmesh.aer        (or "auto" via the router)
           rydberg → qmesh.pulser
           cv      → qmesh.sf.gaussian
           pulse   → qmesh.statevec  (no first-class pulse backend yet)

    3. For each ChannelOp, emit a `ClassicalTask` "channel" stub that:
           - depends on the upstream QPUPrimitive
           - reads `payload["forward"]` from that node's output
             (e.g. a measurement-derived statistic)
           - writes `payload["into_param"]` into the context dict so the
             downstream `QPUPrimitive`'s `module_factory` can pick it up
       Then mark the next block's QPUPrimitive as depending on the
       channel task.

Phase 3β α: forward-direction (`gate->rydberg`, `gate->cv`,
`rydberg->cv`).
Phase 3γ: reverse-direction (`rydberg->gate`, `cv->gate`,
`cv->rydberg`) and pulse-modality bridges (`gate->pulse`, `pulse->gate`).

Bridge payload semantics (Phase 3γ additions):

    rydberg->gate
        upstream: Pulser counts dict (keys are bitstrings like "01").
        forward:  "p_excited"   — per-atom excitation probability.
        emits:    `{into_param: <mean p_excited>,
                    "p_excited_per_atom": [p0, p1, ...],
                    "stat": <mean>}`
        injector: scales params[0] of the first parametric gate
                  (rx/ry/rz/p/u3/rxx/ryy/rzz/cp/crx/cry/crz) by the
                  forwarded scalar (so a fully-excited Rydberg readout
                  becomes a unity-scale rotation, half-excited halves it,
                  etc.).

    cv->gate
        upstream: SF Fock counts (keys are comma-joined integers
                  like "0,2,1").
        forward:  "n_avg"       — mean photon number per mode.
        emits:    `{into_param: <mean n_avg>,
                    "n_avg_per_mode": [n0, n1, ...]}`
        injector: same scale-the-first-parametric-gate pattern.

    cv->rydberg
        upstream: SF counts (Fock or homodyne).
        forward:  "homodyne_x"  — mean homodyne value if the upstream
                  block measured a homodyne basis; else falls back to
                  `n_avg`.
        emits:    `{into_param: <homodyne_x or n_avg> * scale + bias}`
        injector: sets params[0] (Rabi amplitude) of the first
                  RydbergOp — same set-not-scale convention as the
                  forward `gate->rydberg` arrow.

    gate->pulse
        upstream: gate-mode counts (Aer-style bitstrings).
        forward:  default `p_one` — fraction of shots with any '1'.
        emits:    `{"pulse_drive_amp_scale": <stat>*scale+bias, ...}`
        injector: scales params[0] (amplitude) of the first PulseOp.

    pulse->gate
        upstream: pulse-mode counts (whatever the pulse backend returns;
                  the α qmesh.statevec fallback returns gate-style
                  bitstrings).
        forward:  default `p_one`.
        emits:    `{into_param: <stat>*scale+bias}`
        injector: scales params[0] of the first parametric gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from qmesh.ir.module import Function, Module, Region
from qmesh.ir.ops import (
    BarrierOp,
    CVOp,
    ChannelOp,
    ClassicalOp,
    DelayOp,
    GateOp,
    MeasureOp,
    Op,
    PulseOp,
    ResetOp,
    RydbergOp,
)
from qmesh.ir.types import Atom, Bit, Modality, Qubit, Qumode
from qmesh.scheduler.dag import DAG, ClassicalTask, QPUPrimitive


# Default backend per modality. Callers can override via `backend_for`.
_DEFAULT_BACKEND: dict[Modality, str] = {
    Modality.GATE: "qmesh.aer",
    Modality.RYDBERG: "qmesh.pulser",
    Modality.CV: "qmesh.sf.gaussian",
    Modality.PULSE: "qmesh.statevec",  # fallback; pulse backends out of scope
}

# Phase 3β α + Phase 3γ: explicitly handled channel kinds.
SUPPORTED_KINDS: set[str] = {
    # Phase 3β α — forward arrows.
    "gate->rydberg",
    "gate->cv",
    "rydberg->cv",
    # Phase 3γ — reverse arrows.
    "rydberg->gate",
    "cv->gate",
    "cv->rydberg",
    # Phase 3γ — pulse-modality bridges.
    "gate->pulse",
    "pulse->gate",
}

# Anything still on the roadmap. Empty for now — kept as a hook so an
# unsupported kind raises ValueError early in `lower_module_to_dag` (the
# old free-pass for "deferred" kinds is gone now that they're real).
DEFERRED_KINDS: set[str] = set()


@dataclass(slots=True)
class ModalityBlock:
    """A maximal run of ops sharing a modality."""
    modality: Modality
    ops: list[Op] = field(default_factory=list)
    # operand-set: which Qubits / Atoms / Qumodes this block touches (so the
    # rebuilt sub-module declares only the correct typed inputs)
    qubits: set[int] = field(default_factory=set)
    atoms: set[int] = field(default_factory=set)
    qumodes: set[int] = field(default_factory=set)
    bits: set[int] = field(default_factory=set)


# ----------------- partitioning -----------------

# Ops that are tagged classical/control but logically belong with a modality
# block (a measurement or a barrier on qubits is part of the gate block).
_NEUTRAL = (BarrierOp, DelayOp, ResetOp)


def _block_modality_for(op: Op) -> Modality | None:
    """Return the modality this op contributes to, or None if it's a divider."""
    if isinstance(op, ChannelOp):
        return None
    if isinstance(op, MeasureOp):
        # Measurements are typed by their explicit `op.modality` (frontends
        # set it to RYDBERG / CV / GATE). If unset, infer from operand.
        if op.modality and op.modality != Modality.CLASSICAL:
            return op.modality
        for v in op.operands:
            if isinstance(v, Qubit):
                return Modality.GATE
            if isinstance(v, Atom):
                return Modality.RYDBERG
            if isinstance(v, Qumode):
                return Modality.CV
        return Modality.GATE
    if isinstance(op, _NEUTRAL):
        return op.modality if op.modality != Modality.CLASSICAL else None
    if isinstance(op, ClassicalOp):
        return None  # classical ops attach to the previous block
    return op.modality


def _record_operands(block: ModalityBlock, op: Op) -> None:
    for v in op.operands:
        if isinstance(v, Qubit):
            block.qubits.add(v.index)
        elif isinstance(v, Atom):
            block.atoms.add(v.index)
        elif isinstance(v, Qumode):
            block.qumodes.add(v.index)
        elif isinstance(v, Bit):
            block.bits.add(v.index)


def partition(module: Module) -> tuple[list[ModalityBlock], list[tuple[ChannelOp, int, int]]]:
    """Walk a Module, return (blocks, channel_edges).

    Each `(channel_op, upstream_block_idx, downstream_block_idx)` tuple
    identifies which two blocks a ChannelOp bridges.
    """
    blocks: list[ModalityBlock] = []
    edges: list[tuple[ChannelOp, int, int]] = []
    pending_channel: ChannelOp | None = None
    last_block_idx: int | None = None

    for f in module.functions:
        cur: ModalityBlock | None = None
        for op in f.body.ops:
            if isinstance(op, ChannelOp):
                # Close out the current block. Defer recording the edge until
                # the *next* modality block opens.
                if cur is not None:
                    blocks.append(cur)
                    last_block_idx = len(blocks) - 1
                    cur = None
                pending_channel = op
                continue

            mod = _block_modality_for(op)
            if mod is None:
                # Pure classical / unknown — attach to the current block if any.
                if cur is not None:
                    cur.ops.append(op)
                    _record_operands(cur, op)
                continue

            if cur is None or cur.modality != mod:
                if cur is not None:
                    blocks.append(cur)
                    last_block_idx = len(blocks) - 1
                cur = ModalityBlock(modality=mod, ops=[])
                # If a channel was waiting, this is its downstream block.
                if pending_channel is not None and last_block_idx is not None:
                    edges.append((pending_channel, last_block_idx, len(blocks)))
                    pending_channel = None

            cur.ops.append(op)
            _record_operands(cur, op)

        if cur is not None:
            blocks.append(cur)

    # Dangling channel (no following modality block) — drop it with a warning
    # baked into Module metadata. We keep the lowering robust.
    return blocks, edges


# ----------------- module rebuilding per block -----------------

def _build_sub_module(block: ModalityBlock, *, name: str) -> Module:
    """Rebuild a single-modality Module from a block's ops.

    The new Module has typed inputs that match exactly the operands the
    block touches (and Bits as outputs). Operand identities are preserved
    by reference so digests roll up cleanly. Atom positions / Qumode
    cutoffs from the parent module's actual operands flow through.
    """
    qubits = sorted(block.qubits)
    atoms = sorted(block.atoms)
    qumodes = sorted(block.qumodes)
    bits = sorted(block.bits)

    # Pull the *actual* typed values from the block's ops so per-op metadata
    # (Atom.position, Qumode.cutoff) is preserved.
    seen_qubit: dict[int, Qubit] = {}
    seen_atom: dict[int, Atom] = {}
    seen_qumode: dict[int, Qumode] = {}
    for op in block.ops:
        for v in op.operands:
            if isinstance(v, Qubit):
                seen_qubit.setdefault(v.index, v)
            elif isinstance(v, Atom):
                seen_atom.setdefault(v.index, v)
            elif isinstance(v, Qumode):
                seen_qumode.setdefault(v.index, v)

    inputs: list[Any] = []
    inputs.extend(seen_qubit.get(i, Qubit(name=f"q{i}", index=i)) for i in qubits)
    inputs.extend(seen_atom.get(i, Atom(name=f"a{i}", index=i)) for i in atoms)
    inputs.extend(seen_qumode.get(i, Qumode(name=f"m{i}", index=i)) for i in qumodes)
    output_bits = tuple(Bit(name=f"c{i}", index=i) for i in bits)

    region = Region(label=name, ops=list(block.ops))
    fn = Function(name=name, inputs=tuple(inputs), outputs=output_bits, body=region)
    sub = Module()
    sub.add(fn)
    sub.metadata = {"modality": block.modality.value, "lowered_from": "ChannelOp"}
    return sub


# ----------------- channel-task factories -----------------

def _bridge_fn_factory(
    *,
    upstream_node_id: str,
    payload: dict[str, Any],
    kind: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a ClassicalTask body that forwards a stat from upstream → context.

    Reads `ctx[upstream_node_id]["counts"]`, computes a scalar according
    to `payload["forward"]`, scales by `payload["scale"]` (default 1.0)
    + `payload["bias"]` (default 0.0), and writes it under
    `payload["into_param"]` plus a few diagnostic fields. The downstream
    QPUPrimitive's `module_factory` reads that key from the same ctx.

    Supported `forward` modes:
        - "p_excited"   — fraction of shots with at least one '1' bit.
                          Also emits `p_excited_per_atom` (per-position).
        - "parity_even" — fraction of shots with even Hamming weight
        - "p_one"       — P(any '1' anywhere) — synonym of p_excited
        - "n_avg"       — mean photon number across modes (CV outcome
                          strings are comma-joined integers). Also emits
                          `n_avg_per_mode` (per-position).
        - "homodyne_x"  — mean homodyne value if upstream measured a
                          homodyne basis; otherwise falls back to n_avg.
    """
    forward = payload.get("forward", "p_excited")
    into = payload.get("into_param", "amp_rad_per_us")
    scale = float(payload.get("scale", 1.0))
    bias = float(payload.get("bias", 0.0))
    default_value = float(payload.get("default", 0.0))

    def fn(ctx: dict[str, Any]) -> dict[str, Any]:
        upstream = ctx.get(upstream_node_id, {}) or {}
        counts = upstream.get("counts", {}) or {}
        total = sum(counts.values()) or 0
        per_atom: list[float] = []
        per_mode: list[float] = []

        if total == 0:
            stat = default_value
        elif forward in ("p_excited", "p_one"):
            stat = sum(v for k, v in counts.items() if "1" in k) / total
            # Per-atom p_excited: which positions go '1' across shots.
            # Pulser keys are bitstrings like "01"; gate keys may be either
            # "01" (Aer "00 01" → "01") or comma-joined ints. Treat both.
            n_pos = max(
                (len(_split_outcome(k)) for k in counts if k), default=0,
            )
            if n_pos:
                per_atom = [0.0] * n_pos
                for k, v in counts.items():
                    parts = _split_outcome(k)
                    for i, sym in enumerate(parts):
                        if i < n_pos and sym not in ("0", "", None):
                            try:
                                if int(sym) > 0:
                                    per_atom[i] += v
                            except (TypeError, ValueError):
                                continue
                per_atom = [c / total for c in per_atom]
        elif forward == "parity_even":
            stat = sum(v for k, v in counts.items()
                       if k.replace(",", "").count("1") % 2 == 0) / total
        elif forward == "n_avg":
            tally = 0.0
            n_pos = 0
            for k, v in counts.items():
                vals = [int(x) for x in k.split(",") if x.strip().lstrip("-").isdigit()]
                tally += (sum(vals) / len(vals) if vals else 0.0) * v
                if len(vals) > n_pos:
                    n_pos = len(vals)
            stat = tally / total
            if n_pos:
                per_mode = [0.0] * n_pos
                for k, v in counts.items():
                    vals = [int(x) for x in k.split(",") if x.strip().lstrip("-").isdigit()]
                    for i in range(min(len(vals), n_pos)):
                        per_mode[i] += vals[i] * v
                per_mode = [c / total for c in per_mode]
        elif forward == "homodyne_x":
            # Best-effort: SF homodyne samples are floats. Our SF backend
            # currently casts to int via str(int(s)) — so for now this falls
            # back to n_avg semantics. A future SF backend that preserves
            # homodyne floats can override by writing keys with decimal
            # points; we detect that here.
            tally = 0.0
            saw_float = False
            for k, v in counts.items():
                vals_f: list[float] = []
                for x in k.split(","):
                    x = x.strip()
                    if not x:
                        continue
                    try:
                        f = float(x)
                        vals_f.append(f)
                        if "." in x:
                            saw_float = True
                    except ValueError:
                        continue
                if vals_f:
                    tally += (sum(vals_f) / len(vals_f)) * v
            stat = tally / total if total else default_value
            if not saw_float:
                # Fall back to n_avg (the int-coerced photon counts).
                pass
        else:
            stat = default_value

        value = stat * scale + bias
        out: dict[str, Any] = {
            into: value,
            "stat": stat,
            "forward": forward,
            "kind": kind,
            "upstream_node_id": upstream_node_id,
        }
        if per_atom:
            out["p_excited_per_atom"] = per_atom
        if per_mode:
            out["n_avg_per_mode"] = per_mode
        return out

    fn.__name__ = f"channel_{kind.replace('->', '_to_')}"
    return fn


def _split_outcome(key: str) -> list[str]:
    """Split a measurement-outcome string into per-position symbols.

    Handles both bitstring form ("0101") and comma-joined form ("0,1,0,1")
    so the same code path serves Pulser counts and SF counts.
    """
    if "," in key:
        return [s.strip() for s in key.split(",")]
    # Aer sometimes returns space-separated registers ("00 01"); collapse.
    flat = key.replace(" ", "")
    return list(flat)


# Names of parametric gate ops eligible for `→gate` injection.
_PARAMETRIC_GATE_NAMES = frozenset({
    "rx", "ry", "rz", "p", "u3",
    "rxx", "ryy", "rzz", "cp", "crx", "cry", "crz",
})


def _is_target_op(op: Op, modality: Modality) -> bool:
    """Whether this op is the first-match target for parameter injection."""
    if modality == Modality.RYDBERG and isinstance(op, RydbergOp):
        return True
    if modality == Modality.CV and isinstance(op, CVOp):
        return True
    if modality == Modality.GATE and isinstance(op, GateOp):
        # Only parametric gates can carry an injected scalar meaningfully.
        return op.name in _PARAMETRIC_GATE_NAMES and len(op.params) >= 1
    if modality == Modality.PULSE and isinstance(op, PulseOp):
        return len(op.params) >= 1
    return False


def _default_patch_mode(modality: Modality) -> str:
    """`set` for upstream→{rydberg,cv} (existing semantics), `scale` for
    upstream→{gate,pulse} (Phase 3γ — multiply the existing rotation /
    amplitude by the forwarded scalar so a 'fully excited' upstream
    becomes a unity-scale rotation).
    """
    if modality in (Modality.GATE, Modality.PULSE):
        return "scale"
    return "set"


def _module_factory_factory(
    *, sub_module: Module, modality: Modality, channel_node_id: str | None,
    payload: dict[str, Any] | None,
) -> Callable[[dict[str, Any]], Module]:
    """Build a `QPUPrimitive.module_factory` that overrides one param from ctx.

    The override applies to the first matching op in the downstream block:
        - `rydberg` blocks → first RydbergOp's params[0] (Rabi amplitude).
        - `cv` blocks      → first CVOp's params[0] (e.g. squeezing r).
        - `gate` blocks    → first parametric gate's params[0] (rx/ry/rz...).
        - `pulse` blocks   → first PulseOp's params[0] (amplitude).

    `payload["patch_mode"]` controls how the forwarded scalar combines
    with the op's existing params[0]:
        - "set"   — replace params[0] with the forwarded value (used for
                    `→rydberg` / `→cv` / forward arrows).
        - "scale" — multiply params[0] by the forwarded value (used for
                    `→gate` / `→pulse` / reverse and pulse arrows so an
                    upstream measurement modulates a downstream rotation).

    If there's no channel context, the sub-module is returned unchanged.
    """
    payload = payload or {}
    into = payload.get("into_param")
    patch_mode = payload.get("patch_mode") or _default_patch_mode(modality)

    def factory(ctx: dict[str, Any]) -> Module:
        if channel_node_id is None or into is None:
            return sub_module
        ch_out = ctx.get(channel_node_id, {}) or {}
        if into not in ch_out:
            return sub_module
        new_value = float(ch_out[into])

        # Rebuild the module shallowly with the first matching op patched.
        new_mod = Module()
        for f in sub_module.functions:
            new_ops: list[Op] = []
            patched = False
            for op in f.body.ops:
                if not patched and _is_target_op(op, modality):
                    if patch_mode == "scale":
                        old_p0 = float(op.params[0]) if op.params else 0.0
                        new_p0 = old_p0 * new_value
                    else:  # "set"
                        new_p0 = new_value
                    new_params = (new_p0,) + tuple(op.params[1:])
                    new_op = type(op)(
                        name=op.name, operands=op.operands, params=new_params,
                        modality=op.modality,
                        attrs={**op.attrs, "channelop_patched": True,
                               "channelop_patch_mode": patch_mode},
                    )
                    new_ops.append(new_op)
                    patched = True
                else:
                    new_ops.append(op)
            new_region = Region(label=f.body.label, ops=new_ops)
            new_mod.add(Function(
                name=f.name, inputs=f.inputs, outputs=f.outputs, body=new_region,
                params=dict(f.params),
            ))
        new_mod.metadata = {**sub_module.metadata, "channelop_patched": True}
        return new_mod

    factory.__name__ = f"factory_{modality.value}"
    return factory


# ----------------- public lowering -----------------

def lower_module_to_dag(
    module: Module,
    *,
    backend_for: dict[Modality, str] | None = None,
    shots: int = 256,
    sign: bool = True,
    name_prefix: str = "node",
) -> DAG:
    """Lower a multi-modality Module into a runnable DAG.

    Produces, for a Module of shape  [GATE block] -> ChannelOp -> [RYDBERG]
    -> ChannelOp -> [CV]:

        QPU(gate)  →  ClassicalTask(channel gate→ryd)  →  QPU(rydberg)
                                                           ↓
                              ClassicalTask(channel ryd→cv)  →  QPU(cv)

    Each QPU node carries its own per-modality sub-module hash; the DAG
    aggregate manifest references all of them.
    """
    backend_for = {**_DEFAULT_BACKEND, **(backend_for or {})}
    blocks, edges = partition(module)
    if not blocks:
        raise ValueError("lower_module_to_dag: module has no modality-coherent blocks")

    dag = DAG()
    dag.metadata = {
        "lowered_from": "qmesh.ir.Module",
        "source_module_hash": module.hash(),
        "n_blocks": len(blocks),
        "n_channels": len(edges),
        "channel_kinds": [e[0].kind for e in edges],
    }

    # Map block_idx -> dependency for the *next* QPU node (initially the previous QPU).
    block_to_qpu_id: dict[int, str] = {}
    # And: block_idx -> incoming-channel-node-id (for module_factory wiring)
    block_to_channel_id: dict[int, str] = {}
    block_to_payload: dict[int, dict[str, Any]] = {}

    # First pass: pre-create channel-task wiring info.
    edges_by_dst: dict[int, tuple[ChannelOp, int]] = {}
    for ch_op, upstream_idx, downstream_idx in edges:
        edges_by_dst[downstream_idx] = (ch_op, upstream_idx)

    # Second pass: emit QPU + ClassicalTask nodes in toposort order.
    for idx, block in enumerate(blocks):
        sub_mod = _build_sub_module(block, name=f"{name_prefix}_block{idx}_{block.modality.value}")
        backend_name = backend_for.get(block.modality, "qmesh.statevec")

        # Find the channel that feeds this block (if any) for module_factory wiring.
        upstream_qpu_id: str | None = None
        channel_node_id: str | None = None
        payload: dict[str, Any] | None = None
        depends_on: list[str] = []

        if idx in edges_by_dst:
            ch_op, upstream_idx = edges_by_dst[idx]
            kind = ch_op.kind or f"{ch_op.source_modality.value}->{ch_op.target_modality.value}"
            if kind not in SUPPORTED_KINDS:
                raise ValueError(
                    f"channelop_lowering: unsupported ChannelOp kind {kind!r}. "
                    f"Supported kinds: {sorted(SUPPORTED_KINDS)}. "
                    f"(DEFERRED_KINDS is currently empty — kinds outside "
                    f"SUPPORTED_KINDS are not yet implemented.)"
                )
            payload = dict(ch_op.payload or {})
            upstream_qpu_id = block_to_qpu_id[upstream_idx]
            channel_id = f"channel_{idx}_{kind.replace('->', '_to_')}"
            ct = ClassicalTask(
                id=channel_id,
                name=f"channel {kind}",
                fn=_bridge_fn_factory(
                    upstream_node_id=upstream_qpu_id, payload=payload, kind=kind,
                ),
                description=f"ChannelOp lowering: {kind}",
                depends_on=[upstream_qpu_id],
            )
            dag.add(ct)
            channel_node_id = channel_id
            depends_on.append(channel_id)
            block_to_channel_id[idx] = channel_id
            block_to_payload[idx] = payload

        # Emit the QPU node.
        qpu_id = f"qpu_{idx}_{block.modality.value}"
        if channel_node_id is not None:
            factory = _module_factory_factory(
                sub_module=sub_mod, modality=block.modality,
                channel_node_id=channel_node_id, payload=payload,
            )
            qpu_node = QPUPrimitive(
                id=qpu_id,
                name=f"{block.modality.value}-block #{idx}",
                module_factory=factory,
                backend=backend_name,
                shots=shots,
                sign=sign,
                depends_on=depends_on,
            )
        else:
            qpu_node = QPUPrimitive(
                id=qpu_id,
                name=f"{block.modality.value}-block #{idx}",
                module=sub_mod,
                backend=backend_name,
                shots=shots,
                sign=sign,
                depends_on=depends_on,
            )
        dag.add(qpu_node)
        block_to_qpu_id[idx] = qpu_id

    return dag


__all__ = [
    "lower_module_to_dag",
    "partition",
    "ModalityBlock",
    "SUPPORTED_KINDS",
    "DEFERRED_KINDS",
]
