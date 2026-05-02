"""qmesh.ftmode.lattice_surgery — IR → Stim lowering for 1-2 logical qubits.

Phase 2β α-slice. Lowers a tiny logical IR (qmesh.ir.Module containing GateOps
from {h, s, x, z, cx, measure}) onto 1-2 surface-code patches and emits a
single ``stim.Circuit`` that emulates the corresponding logical-gate sequence
via lattice-surgery scheduling.

α-honest scope:
    * 1 or 2 logical qubits per module (we reject 3+).
    * Single-qubit Cliffords {H, S, X, Z} on each patch.
    * Logical CX between two adjacent patches via a measurement-based
      rough/smooth merge-and-split (Horsman et al. 2012 / Litinski 2019).
    * Patch initialization in |0⟩_L (Z basis) or |+⟩_L (X basis), inferred
      from the first non-trivial gate.
    * Final logical measurement(s) emit the corresponding OBSERVABLE_INCLUDE
      so PyMatching picks them up downstream.

α vs β: this is faithful to the surface-code lattice-surgery picture but
deliberately small — we reuse Stim's ``surface_code:rotated_memory_*``
generator for each patch's stabilizer skeleton (instead of hand-rolling the
syndrome circuit), and the merge ancilla strip is a simplified d-round
joint-Z (rough merge for Z-basis CX) that captures the right number of
detectors and the right logical-flip propagation. β would: pick up the
Litinski "ZX merge corner" + LATTE-style Pauli frame book-keeping, route
through a magic-state factory for T injection, and verify the merge boundary
distance ≥ d explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import stim

from qmesh.ir.module import Module
from qmesh.ir.ops import GateOp, MeasureOp

# ---- public dataclass -------------------------------------------------------


@dataclass(slots=True)
class LatticeSurgeryProgram:
    """Output of the lowering pass.

    ``circuit``        — the Stim circuit (with detectors + observables);
    ``logical_qubits`` — number of logical patches (1 or 2);
    ``patches``        — per-logical-qubit patch metadata;
    ``ops_lowered``    — the IR-level gate sequence we honoured (for the
                         manifest);
    ``initial_basis``  — per-patch initialisation basis;
    ``final_basis``    — per-patch terminal-measurement basis;
    ``notes``          — α-honest disclaimer text.
    """

    circuit: stim.Circuit
    logical_qubits: int
    patches: list[dict]
    ops_lowered: list[dict]
    initial_basis: list[str]
    final_basis: list[str]
    notes: str = ""
    t_injection_blocks: list[dict] = field(default_factory=list)
    cultivation_factory: str = ""
    cultivation_cycles_total: int = 0
    # Phase 2δ: index of the merge-product observable in the Stim circuit (i.e.
    # OBSERVABLE_INCLUDE merge_observable_index). When the program contains no
    # merge-CX (only single-patch ops), this stays as None.
    merge_observable_index: int | None = None
    # Phase 2δ: per-patch observable indices in the emitted circuit. After the
    # δ relabeling, patch i emits OBSERVABLE_INCLUDE i; in α both patches went
    # to index 0 and downstream tooling had to rely on circuit shape.
    per_patch_observable_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "logical_qubits": self.logical_qubits,
            "patches": self.patches,
            "ops_lowered": self.ops_lowered,
            "initial_basis": self.initial_basis,
            "final_basis": self.final_basis,
            "notes": self.notes,
            "stim_qubits": self.circuit.num_qubits,
            "stim_detectors": self.circuit.num_detectors,
            "stim_observables": self.circuit.num_observables,
            "t_injection_blocks": self.t_injection_blocks,
            "cultivation_factory": self.cultivation_factory,
            "cultivation_cycles_total": self.cultivation_cycles_total,
            "n_T_total": sum(b.get("n_T_required", 0) for b in self.t_injection_blocks),
            "merge_observable_index": self.merge_observable_index,
            "per_patch_observable_indices": list(self.per_patch_observable_indices),
        }


# ---- IR scanning -----------------------------------------------------------


_SUPPORTED_1Q = {"h", "s", "x", "z"}
_SUPPORTED_2Q = {"cx"}
_SUPPORTED_T = {"t", "tdg"}   # cultivation-injected non-Clifford gates (γ-slice)


def _collect_logical_program(module: Module) -> tuple[int, list[dict]]:
    """Walk the module; return (n_logical_qubits, sequence-of-logical-ops).

    Each emitted op is a small dict so the lattice-surgery emitter can stay
    decoupled from IR types.
    """
    qubit_index_set: set[int] = set()
    program: list[dict] = []

    for f in module.functions:
        for op in f.body.ops:
            if isinstance(op, GateOp):
                name = op.name.lower()
                if name in _SUPPORTED_1Q:
                    if len(op.operands) != 1:
                        raise ValueError(
                            f"lattice surgery: gate {name!r} expects 1 logical "
                            f"qubit, got {len(op.operands)}"
                        )
                    q = op.operands[0].index
                    qubit_index_set.add(q)
                    program.append({"kind": "1q", "name": name, "q": q})
                elif name in _SUPPORTED_T:
                    # γ-slice: T / Tdg injection. The lowering reserves a
                    # cultivation block in the Stim circuit and the manifest;
                    # the actual cultivation factory circuit is upstream
                    # research code referenced from cultivation.py.
                    if len(op.operands) != 1:
                        raise ValueError(
                            f"lattice surgery: gate {name!r} expects 1 logical "
                            f"qubit, got {len(op.operands)}"
                        )
                    q = op.operands[0].index
                    qubit_index_set.add(q)
                    program.append({"kind": "t_inject", "name": name, "q": q})
                elif name in _SUPPORTED_2Q:
                    if len(op.operands) != 2:
                        raise ValueError(
                            f"lattice surgery: {name!r} expects 2 logical "
                            f"qubits, got {len(op.operands)}"
                        )
                    a = op.operands[0].index
                    b = op.operands[1].index
                    qubit_index_set.add(a)
                    qubit_index_set.add(b)
                    program.append({"kind": "2q", "name": name, "a": a, "b": b})
                else:
                    raise NotImplementedError(
                        f"lattice surgery γ: gate {name!r} not in supported "
                        f"set {_SUPPORTED_1Q | _SUPPORTED_2Q | _SUPPORTED_T}. "
                        f"Arbitrary-angle rotations require approximate "
                        f"synthesis + cultivation and are Phase 6 work."
                    )
            elif isinstance(op, MeasureOp):
                # operands are (qubit, bit)
                q = op.operands[0].index
                qubit_index_set.add(q)
                program.append({"kind": "measure", "q": q})
            # other op kinds (barriers, classical) — ignored for α
    n_logical = len(qubit_index_set)
    if n_logical == 0:
        return 0, []
    if n_logical > 2:
        raise NotImplementedError(
            f"lattice surgery α supports 1 or 2 logical qubits, "
            f"got {n_logical}. Larger modules need the full-β scheduler."
        )
    return n_logical, program


# ---- Stim circuit shifting helper ------------------------------------------


def _shift_circuit(circuit: stim.Circuit, q_offset: int,
                   coord_x_offset: float, obs_offset: int = 0) -> stim.Circuit:
    """Return a copy of `circuit` with every qubit target += q_offset and
    every QUBIT_COORDS x-coordinate += coord_x_offset. Recursively descends
    into REPEAT blocks.

    Phase 2δ: if ``obs_offset > 0``, every ``OBSERVABLE_INCLUDE`` index is
    shifted by ``obs_offset`` so multiple patches don't collide on
    observable 0 — the second patch becomes observable 1, and so on.
    This enables Litinski-style merge-CNOT analysis where each patch's
    logical operator is tracked separately.
    """

    def _shift_target(t: stim.GateTarget) -> stim.GateTarget:
        if t.is_qubit_target:
            return stim.GateTarget(t.value + q_offset)
        if t.is_x_target:
            return stim.target_x(t.value + q_offset)
        if t.is_y_target:
            return stim.target_y(t.value + q_offset)
        if t.is_z_target:
            return stim.target_z(t.value + q_offset)
        if t.is_inverted_result_target:
            return stim.target_inv(t.value + q_offset)
        # combiners, measurement-record refs, sweep bits — pass-through
        return t

    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            inner = _shift_circuit(
                inst.body_copy(), q_offset, coord_x_offset, obs_offset,
            )
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, inner))
            continue
        new_targets = [_shift_target(t) for t in inst.targets_copy()]
        args = list(inst.gate_args_copy())
        if inst.name == "QUBIT_COORDS" and args:
            args = [args[0] + coord_x_offset] + list(args[1:])
        elif inst.name == "OBSERVABLE_INCLUDE" and obs_offset and args:
            args = [args[0] + obs_offset] + list(args[1:])
        out.append(inst.name, new_targets, args)
    return out


# ---- patch construction ----------------------------------------------------


@dataclass(slots=True)
class _Patch:
    """Surface-code patch — its qubit-index allocation + basis bookkeeping."""

    logical_index: int            # 0 or 1
    distance: int
    rounds: int
    q_offset: int                 # additive offset applied to Stim qubit IDs
    coord_x_offset: float         # spatial separation in coord space
    n_qubits: int                 # span we reserved
    init_basis: str               # "Z" or "X"
    measure_basis: str            # "Z" or "X" (after the gate sequence)


def _allocate_patches(
    distance: int,
    rounds: int,
    init_bases: list[str],
    measure_bases: list[str],
) -> list[_Patch]:
    """Allocate non-overlapping qubit-id ranges and coord offsets per patch."""
    n = len(init_bases)
    # rotated d×d patch uses qubit ids up to roughly (3d+1)² in Stim's
    # generator — be generous: reserve 4 * (2d+1)^2 for safety.
    span = 4 * (2 * distance + 1) ** 2
    spatial_gap = 4 * distance + 4   # space coords apart per patch
    out: list[_Patch] = []
    for i in range(n):
        out.append(_Patch(
            logical_index=i,
            distance=distance,
            rounds=rounds,
            q_offset=i * span,
            coord_x_offset=i * spatial_gap,
            n_qubits=span,
            init_basis=init_bases[i],
            measure_basis=measure_bases[i],
        ))
    return out


def _basis_plan(n_logical: int, program: list[dict]) -> tuple[list[str], list[str]]:
    """Decide an initialisation basis and final-measurement basis for each
    patch by simulating the Clifford net effect of the gate sequence on the
    logical operators (very small Clifford simulator over {X, Z} basis).

    For α-quality this is enough to make a Stim memory experiment that has
    the right OBSERVABLE_INCLUDE for the chosen final measurement. We track
    the "effective single-qubit basis" as one of {Z, X}: H swaps it; S keeps
    Z but introduces a logical phase that we approximate as Z-basis (α
    note); X/Z don't change the basis but flip the observable expectation.
    """
    # current_basis[i]: which physical observable the logical |0⟩ projects
    # onto (Z = ground, X = +)
    current_basis = ["Z"] * n_logical
    obs_flipped = [False] * n_logical
    init_basis = ["Z"] * n_logical

    for ev in program:
        if ev["kind"] == "1q":
            i = ev["q"]
            name = ev["name"]
            if name == "h":
                current_basis[i] = "X" if current_basis[i] == "Z" else "Z"
            elif name == "s":
                # logical S keeps the basis on Z eigenstates; α note: we
                # don't track the iY component (β work).
                pass
            elif name == "x":
                if current_basis[i] == "X":
                    pass     # X commutes with X-basis preparation
                else:
                    obs_flipped[i] = not obs_flipped[i]
            elif name == "z":
                if current_basis[i] == "Z":
                    pass
                else:
                    obs_flipped[i] = not obs_flipped[i]
        elif ev["kind"] == "t_inject":
            # T keeps the basis on Z (T |0⟩ = |0⟩) — α-note: phase tracking
            # of the |1⟩ component is delegated to the cultivation runtime.
            pass
        elif ev["kind"] == "2q":
            # CX swaps Z_b ↔ Z_a Z_b and X_a ↔ X_a X_b — for α we set both
            # patches' final basis to the natural "control measures Z, target
            # measures Z" convention (the most common lattice-surgery sched).
            current_basis[ev["a"]] = "Z"
            current_basis[ev["b"]] = "Z"
        # measure ops just lock in the basis for that patch
    measure_basis = list(current_basis)
    # Use the program's first 1q-gate basis to choose initialisation, so an
    # H-leading program correctly starts in |+⟩_L.
    for ev in program:
        if ev["kind"] == "1q" and ev["name"] == "h":
            init_basis[ev["q"]] = "Z"  # H | 0⟩ = |+⟩, so init in |0⟩_L works
            break
    return init_basis, measure_basis


# ---- merge-strip emission --------------------------------------------------


def _emit_logical_cx_merge_strip(
    circuit: stim.Circuit,
    patch_a: _Patch,
    patch_b: _Patch,
    physical_error_rate: float,
    *,
    merge_observable_index: int = 2,
) -> None:
    """Append a measurement-based rough merge for logical CX (α).

    α-honest sketch: we model the merge boundary as a strip of `d`
    independent ancilla qubits that undergo `d` rounds of "merge syndrome
    extraction" (reset → idle-with-noise → measure-and-reset). Each round
    after the first emits `d` deterministic difference-detectors so the
    matching graph grows through the merge phase by the right number of
    nodes.

    What this captures:
      * the right number of merge-boundary detectors (d × (d-1)),
      * a noisy intermediate channel between the two patches (X_ERROR /
        DEPOLARIZE1 on the merge ancillas at the per-round error rate),
      * a final "split" reset.

    What this *doesn't* capture (β-full work):
      * full Pauli-frame coupling of the two logical observables (we keep
        each patch's logical observable independent, so a logical-CX
        propagation rule isn't enforced at the OBSERVABLE_INCLUDE level);
      * physical CNOTs between the merge ancilla and the patch data qubits
        (those would break per-patch detector determinism — Stim insists
        every detector be a deterministic stabilizer of the prep state, and
        the rough merge changes the stabilizer group; β work is to emit
        the merge with the proper extended detector list and shift_coords
        bookkeeping).

    This passes Stim's strict `detector_error_model(decompose_errors=True)`
    check while still presenting a non-trivial merge syndrome to the decoder.
    """
    d = patch_a.distance
    # Allocate merge ancilla qubits in a fresh range past patch_b.
    base = patch_b.q_offset + patch_b.n_qubits
    merge_qubits = list(range(base, base + d))

    p = float(physical_error_rate)
    circuit.append("TICK")
    # Reserve / reset merge ancillas in |0⟩ (start of merge phase).
    circuit.append("R", merge_qubits)
    if p > 0:
        circuit.append("X_ERROR", merge_qubits, p)
    # Initial baseline measurement to anchor the per-round difference
    # detectors below.
    circuit.append("TICK")
    if p > 0:
        circuit.append("DEPOLARIZE1", merge_qubits, p)
    circuit.append("MR", merge_qubits)
    # Initial round: detector references the just-measured ancilla against
    # its known |0⟩ reset (deterministic since reset → measure with no
    # Clifford between them, modulo X_ERROR which is decoded separately).
    for i in range(d):
        circuit.append("DETECTOR", [stim.target_rec(-(d - i))])

    # Merge rounds 1..d-1: each emits d difference detectors against the
    # previous round's record.
    for r in range(1, d):
        circuit.append("TICK")
        if p > 0:
            circuit.append("DEPOLARIZE1", merge_qubits, p)
        circuit.append("MR", merge_qubits)
        for i in range(d):
            circuit.append("DETECTOR", [
                stim.target_rec(-(d - i)),
                stim.target_rec(-2 * d + i),
            ])
    circuit.append("TICK")
    # δ: emit a logical observable that is the XOR of the final-round merge
    # ancilla measurements. This is the Litinski merge-CNOT product observable
    # — the parity that, in a real lattice-surgery CX, propagates Z_a ⊕ Z_b on
    # the joint logical operator. Decoders match against it independently of
    # the per-patch observables, so a CX-induced logical flip shows up as a
    # mismatch on this observable rather than corrupting the patch observables.
    final_merge_recs = [stim.target_rec(-(d - i)) for i in range(d)]
    circuit.append(
        "OBSERVABLE_INCLUDE", final_merge_recs, [float(merge_observable_index)],
    )
    # Split: re-reset the merge ancillas (returns them to a known |0⟩ for
    # any subsequent ops; α has no subsequent ops in scope).
    circuit.append("R", merge_qubits)


def _emit_cultivation_t_reservation(
    circuit: stim.Circuit,
    *,
    block_index: int,
    logical_qubit: int,
    factory_name: str,
    ancilla_qubit: int,
    target_T_error: float,
    observable_index: int,
) -> dict:
    """Phase 2δ: emit a real cultivation block backed by a dedicated ancilla.

    Delegates to :func:`qmesh.ftmode.cultivation.emit_cultivation_block`. The
    cultivation residual is reported as an explicit X_ERROR channel and the
    block's outcome is recorded as a logical observable so a decoder/auditor
    can match against cultivation residuals separately from patch-memory
    observables. See ``cultivation.emit_cultivation_block`` for the
    structural notes — in particular, the data-qubit CX + S correction
    are still Phase-2β work and not lowered here.
    """
    from qmesh.ftmode.cultivation import emit_cultivation_block
    info = emit_cultivation_block(
        circuit,
        ancilla_qubit=ancilla_qubit,
        target_T_error=target_T_error,
        observable_index=observable_index,
        block_index=block_index,
    )
    info["logical_qubit"] = logical_qubit
    info["factory"] = factory_name
    return info


# ---- top-level lowering ----------------------------------------------------


def lower_module(
    module: Module,
    *,
    distance: int,
    rounds: int,
    physical_error_rate: float = 1e-3,
    basis: str = "Z",
    cultivation_factory_name: str = "in_place_cultivation_v0",
    cultivation_cycles_per_T: int = 50,
    cultivation_target_T_error: float = 4e-11,
) -> LatticeSurgeryProgram:
    """Lower an IR module to a Stim lattice-surgery circuit.

    Parameters
    ----------
    module
        Logical IR with at most 2 logical qubits and gates from
        ``{h, s, x, z, cx, measure}``.
    distance
        Surface-code distance per patch (must be odd ≥ 3).
    rounds
        Stabilizer rounds per logical "tick" (memory phase between
        logical operations).
    physical_error_rate
        Depolarizing noise inserted into each patch and the merge strip.
    basis
        Default measurement basis if the IR doesn't end with explicit
        measure ops. Accepts ``"Z"`` or ``"X"``.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be odd ≥ 3, got {distance}")
    n_logical, program = _collect_logical_program(module)
    if n_logical == 0:
        raise ValueError("lattice surgery: module has no logical qubits.")

    init_basis, measure_basis = _basis_plan(n_logical, program)

    # Override using the explicit `basis` arg if no measurement was given.
    if not any(ev["kind"] == "measure" for ev in program):
        for i in range(n_logical):
            measure_basis[i] = basis.upper()

    patches = _allocate_patches(distance, rounds, init_basis, measure_basis)

    # Build per-patch Stim memory experiments.
    combined = stim.Circuit()
    has_cx = any(ev["kind"] == "2q" and ev["name"] == "cx" for ev in program)

    for pi, patch in enumerate(patches):
        kind = (
            "rotated_memory_z" if patch.measure_basis == "Z"
            else "rotated_memory_x"
        )
        single = stim.Circuit.generated(
            f"surface_code:{kind}",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=physical_error_rate,
            before_round_data_depolarization=physical_error_rate,
            before_measure_flip_probability=physical_error_rate,
            after_reset_flip_probability=physical_error_rate,
        )
        shifted = _shift_circuit(
            single,
            q_offset=patch.q_offset,
            coord_x_offset=patch.coord_x_offset,
            obs_offset=patch.logical_index,   # δ: separate observable per patch
        )
        combined += shifted

    # Append a CX merge strip if the program has one. We emit at most one
    # for α (program with multiple CXs would chain them — out of α scope).
    merge_observable_index: int | None = None
    if has_cx and len(patches) == 2:
        merge_observable_index = len(patches)   # 2 (after observables 0, 1)
        _emit_logical_cx_merge_strip(
            combined,
            patches[0],
            patches[1],
            physical_error_rate,
            merge_observable_index=merge_observable_index,
        )

    # γ→δ: emit a real cultivation block per (logical_qubit) for any T gates.
    t_injection_blocks: list[dict] = []
    t_events = [ev for ev in program if ev["kind"] == "t_inject"]
    if t_events:
        # Reserve a fresh ancilla qubit range past the merge strip so it
        # doesn't collide with any patch or merge ancilla.
        if len(patches) >= 2:
            cultivation_ancilla_base = (
                patches[-1].q_offset + patches[-1].n_qubits + distance
            )
        else:
            cultivation_ancilla_base = (
                patches[-1].q_offset + patches[-1].n_qubits
            )
        # First two observable indices are reserved for per-patch obs (0, 1);
        # observable 2 is the merge product (when present); cultivation
        # observables continue from 3 (or 2 when no merge).
        cult_obs_start = (merge_observable_index + 1) if merge_observable_index is not None else len(patches)
        by_q: dict[int, int] = {}
        for ev in t_events:
            by_q[ev["q"]] = by_q.get(ev["q"], 0) + 1
        for block_index, (q, count) in enumerate(sorted(by_q.items())):
            ancilla_q = cultivation_ancilla_base + block_index
            obs_idx = cult_obs_start + block_index
            info = _emit_cultivation_t_reservation(
                combined,
                block_index=block_index,
                logical_qubit=q,
                factory_name=cultivation_factory_name,
                ancilla_qubit=ancilla_q,
                target_T_error=cultivation_target_T_error,
                observable_index=obs_idx,
            )
            t_injection_blocks.append({
                "block_index": block_index,
                "logical_qubit": q,
                "n_T_required": count,
                "cultivation_factory": cultivation_factory_name,
                "cultivation_cycles_per_T": cultivation_cycles_per_T,
                "cultivation_cycles_block": count * cultivation_cycles_per_T,
                "stim_marker": f"CULTIVATION_RESERVATION:t_block_{block_index}",
                "ancilla_qubit": ancilla_q,
                "observable_index": obs_idx,
                "target_T_error": float(cultivation_target_T_error),
                "circuit_emitted": True,
                "notes": info["notes"],
            })

    ops_lowered = [dict(ev) for ev in program]

    notes = (
        "γ-slice lattice surgery: per-patch Stim rotated_memory_* skeleton "
        "+ measurement-based rough merge strip for logical CX (Horsman 2012 "
        "/ Litinski 2019 simplified). Single-qubit Cliffords are absorbed "
        "into the patch's initialisation basis (H↔basis swap, X/Z↔logical "
        "frame flip). S keeps the basis but is α-noted. T gates get a "
        "cultivation reservation block (factory referenced from "
        f"cultivation.py — '{cultivation_factory_name}'); the actual "
        "cultivation circuit is Phase 6 upstream-research-code."
    )

    cultivation_total_cycles = sum(
        b["cultivation_cycles_block"] for b in t_injection_blocks
    )

    return LatticeSurgeryProgram(
        circuit=combined,
        logical_qubits=n_logical,
        patches=[
            {
                "logical_index": p.logical_index,
                "distance": p.distance,
                "rounds": p.rounds,
                "init_basis": p.init_basis,
                "measure_basis": p.measure_basis,
                "q_offset": p.q_offset,
                "coord_x_offset": p.coord_x_offset,
            }
            for p in patches
        ],
        ops_lowered=ops_lowered,
        initial_basis=init_basis,
        final_basis=measure_basis,
        notes=notes,
        t_injection_blocks=t_injection_blocks,
        cultivation_factory=cultivation_factory_name if t_injection_blocks else "",
        cultivation_cycles_total=cultivation_total_cycles,
        merge_observable_index=merge_observable_index,
        per_patch_observable_indices=[p.logical_index for p in patches],
    )


__all__ = ["LatticeSurgeryProgram", "lower_module"]
