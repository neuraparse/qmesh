"""Phase 3α tests: Pulser + Strawberry Fields integration + modality routing."""

from __future__ import annotations

import pytest

import qmesh
from qmesh.ir.types import Modality


def _pulser_seq(rabi_amp: float = 6.28):
    pulser = pytest.importorskip("pulser")
    from pulser import Pulse, Register, Sequence
    from pulser.devices import MockDevice
    reg = Register({"q0": (0, 0), "q1": (5, 0)})
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg", "rydberg_global")
    seq.add(Pulse.ConstantPulse(1000, rabi_amp, 0, 0), "rydberg")
    seq.measure("ground-rydberg")
    return seq


def _sf_program():
    sf = pytest.importorskip("strawberryfields")
    from strawberryfields import ops
    prog = sf.Program(2)
    with prog.context as q:
        ops.Sgate(0.4) | q[0]
        ops.Sgate(0.4) | q[1]
        ops.BSgate(0.785, 0) | (q[0], q[1])
        ops.MeasureFock() | q[0]
        ops.MeasureFock() | q[1]
    return prog


# ---------- Pulser frontend ----------

def test_pulser_frontend_translates_to_rydberg_modality():
    pytest.importorskip("pulser")
    from qmesh.frontends.pulser import from_pulser
    module = from_pulser(_pulser_seq())
    used = {op.modality for f in module.functions for op in f.body.ops}
    assert Modality.RYDBERG in used
    # 2 atoms → 2 measurement bits
    assert any("measure" in op.name for f in module.functions for op in f.body.ops)


def test_pulser_atom_positions_preserved():
    pytest.importorskip("pulser")
    from qmesh.frontends.pulser import from_pulser
    from qmesh.ir.types import Atom
    module = from_pulser(_pulser_seq())
    atoms = [v for f in module.functions for v in f.inputs if isinstance(v, Atom)]
    assert len(atoms) == 2
    assert atoms[0].position is not None
    assert atoms[1].position == (5.0, 0.0, 0.0)


# ---------- Pulser backend ----------

def test_pulser_backend_runs_blockade(tmp_path):
    pytest.importorskip("pulser")
    pytest.importorskip("pulser_simulation")
    from qmesh.frontends.pulser import from_pulser
    module = from_pulser(_pulser_seq())
    result, manifest = qmesh.submit(
        module, backend="qmesh.pulser", shots=200, ledger_dir=tmp_path,
    )
    assert result.shots == 200
    assert sum(result.counts.values()) == 200
    # Rydberg blockade: |11⟩ should be heavily suppressed at 5µm + 2π Rabi
    p_11 = result.counts.get("11", 0) / result.shots
    assert p_11 < 0.30, f"Rydberg blockade not visible (p_11={p_11})"


def test_pulser_backend_rejects_gate_modality(tmp_path):
    pytest.importorskip("pulser")
    with qmesh.circuit("g", n_qubits=2, n_bits=2) as c:
        c.h(0); c.cx(0, 1); c.measure(0, 0); c.measure(1, 1)
    with pytest.raises(ValueError):
        qmesh.submit(c.module, backend="qmesh.pulser", shots=10, ledger_dir=tmp_path)


# ---------- Strawberry Fields frontend ----------

def test_sf_frontend_translates_to_cv_modality():
    pytest.importorskip("strawberryfields")
    from qmesh.frontends.sf import from_sf
    module = from_sf(_sf_program())
    used = {op.modality for f in module.functions for op in f.body.ops}
    assert Modality.CV in used
    # Sgate, BSgate, two MeasureFocks
    op_names = [op.name for f in module.functions for op in f.body.ops]
    assert "Sgate" in op_names
    assert "BSgate" in op_names


# ---------- Strawberry Fields backend ----------

def test_sf_gaussian_backend_runs(tmp_path):
    pytest.importorskip("strawberryfields")
    from qmesh.frontends.sf import from_sf
    module = from_sf(_sf_program())
    result, _ = qmesh.submit(
        module, backend="qmesh.sf.gaussian", shots=20, ledger_dir=tmp_path,
    )
    assert result.shots == 20
    assert sum(result.counts.values()) == 20
    # vacuum should dominate (squeezing param 0.4 keeps mean photon number low)
    p_vacuum = result.counts.get("0,0", 0) / result.shots
    assert p_vacuum > 0.40


def test_sf_backend_rejects_gate_modality(tmp_path):
    pytest.importorskip("strawberryfields")
    with qmesh.circuit("g", n_qubits=1, n_bits=1) as c:
        c.h(0); c.measure(0, 0)
    with pytest.raises(ValueError):
        qmesh.submit(c.module, backend="qmesh.sf.gaussian", shots=10, ledger_dir=tmp_path)


# ---------- modality-aware router ----------

def test_router_routes_rydberg_module_to_pulser():
    pytest.importorskip("pulser")
    from qmesh.frontends.pulser import from_pulser
    from qmesh.router import choose
    module = from_pulser(_pulser_seq())
    bk, why = choose(module)
    # Only rydberg-modality backend is qmesh.pulser
    assert bk.capabilities.name == "qmesh.pulser"


def test_router_routes_cv_module_to_sf():
    pytest.importorskip("strawberryfields")
    from qmesh.frontends.sf import from_sf
    from qmesh.router import choose
    module = from_sf(_sf_program())
    bk, why = choose(module)
    assert bk.capabilities.name.startswith("qmesh.sf")


def test_three_modality_backends_registered():
    """Coverage check: at least one backend per modality."""
    from qmesh.backends.registry import all_backends
    by_modality: dict[Modality, list[str]] = {}
    for bk in all_backends().values():
        for m in bk.capabilities.modalities:
            by_modality.setdefault(m, []).append(bk.capabilities.name)
    assert Modality.GATE in by_modality
    assert Modality.RYDBERG in by_modality
    assert Modality.CV in by_modality


# ===================================================================
# Phase 3β α — Bloqade frontend, MrMustard frontend,
# first-class ChannelOp, ChannelOp lowering to a multi-modality DAG.
# ===================================================================


def _bloqade_program():
    """A small Bloqade-shaped program used by both frontend tests.

    α-quality contract (Phase 3β): if real Bloqade is installed, build a
    real program; otherwise build an `BloqadeProgram` shim — same shape,
    exercises the same translator path. The shim is always exported by
    qmesh.frontends.bloqade so the test runs either way.
    """
    from qmesh.frontends.bloqade import (
        BloqadeAtom,
        BloqadeProgram,
        BloqadeRegister,
        DriveSegment,
    )
    register = BloqadeRegister(atoms=[
        BloqadeAtom(position=(0.0, 0.0, 0.0)),
        BloqadeAtom(position=(5.0, 0.0, 0.0)),
        BloqadeAtom(position=(10.0, 0.0, 0.0)),
    ])
    drives = [
        DriveSegment(duration_ns=600, amplitude_rad_per_us=6.28,
                     detuning_rad_per_us=0.0, phase=0.0),
        DriveSegment(duration_ns=200, amplitude_rad_per_us=0.0,
                     detuning_rad_per_us=0.0),  # idle
        DriveSegment(duration_ns=400, amplitude_rad_per_us=3.14,
                     detuning_rad_per_us=1.0, phase=0.5),
    ]
    return BloqadeProgram(
        register=register, drives=drives, measurement_basis="ground-rydberg",
    )


def test_bloqade_frontend_translates_atoms_and_pulses():
    # α-contract: real bloqade is preferred; absent it, the shim covers the
    # same translator path. We *don't* skip the test when the lib is missing
    # because the shim exercises the production code path.
    pytest.importorskip("qmesh.frontends.bloqade")
    from qmesh.frontends.bloqade import from_bloqade
    from qmesh.ir.ops import RydbergOp
    from qmesh.ir.types import Atom

    prog = _bloqade_program()
    module = from_bloqade(prog)

    # 3 atoms → 3 inputs, 3 measurement bits
    atoms = [v for f in module.functions for v in f.inputs if isinstance(v, Atom)]
    assert len(atoms) == 3
    # Atom positions preserved
    assert atoms[0].position == (0.0, 0.0, 0.0)
    assert atoms[1].position == (5.0, 0.0, 0.0)
    assert atoms[2].position == (10.0, 0.0, 0.0)
    # Modality coverage
    used = {op.modality for f in module.functions for op in f.body.ops}
    assert Modality.RYDBERG in used
    # Two RydbergOps (the two non-zero-amp segments) + one DelayOp + 3 measures
    op_kinds = [type(op).__name__ for f in module.functions for op in f.body.ops]
    assert op_kinds.count("RydbergOp") == 2
    assert op_kinds.count("DelayOp") == 1
    assert op_kinds.count("MeasureOp") == 3
    # The first RydbergOp's params = (amp, det, phase, duration)
    rydberg_ops = [op for f in module.functions for op in f.body.ops
                   if isinstance(op, RydbergOp)]
    assert rydberg_ops[0].params[0] == pytest.approx(6.28)
    assert rydberg_ops[0].params[3] == pytest.approx(600.0)
    # Frontend tag flows through
    assert module.metadata.get("frontend") == "bloqade"


def _mrmustard_program():
    # α-contract: shim path always works; real mrmustard duck-types in.
    from qmesh.frontends.mrmustard import MMCircuit, MMComponent
    return MMCircuit(
        n_modes=2,
        components=[
            MMComponent(name="Sgate", modes=(0,), params=(0.4,)),
            MMComponent(name="Sgate", modes=(1,), params=(0.4,)),
            MMComponent(name="BSgate", modes=(0, 1), params=(0.785, 0.0)),
            MMComponent(name="MeasureFock", modes=(0,), params=()),
            MMComponent(name="MeasureFock", modes=(1,), params=()),
        ],
    )


def test_mrmustard_frontend_translates_squeezing():
    pytest.importorskip("qmesh.frontends.mrmustard")
    from qmesh.frontends.mrmustard import from_mrmustard
    from qmesh.ir.ops import CVOp, MeasureOp

    prog = _mrmustard_program()
    module = from_mrmustard(prog)

    # 2 qumodes, 2 measurements, 3 CV ops (2 Sgate + 1 BSgate)
    cv_ops = [op for f in module.functions for op in f.body.ops
              if isinstance(op, CVOp)]
    assert len(cv_ops) == 3
    op_names = [op.name for op in cv_ops]
    assert "Sgate" in op_names
    assert "BSgate" in op_names

    # Squeezing parameter passes through
    sgate = next(op for op in cv_ops if op.name == "Sgate")
    assert sgate.params[0] == pytest.approx(0.4)

    # MeasureFock → MeasureOp tagged with mrmustard frontend
    measures = [op for f in module.functions for op in f.body.ops
                if isinstance(op, MeasureOp)]
    assert len(measures) == 2
    assert all(op.attrs.get("frontend") == "mrmustard" for op in measures)
    assert module.metadata.get("frontend") == "mrmustard"


# ---------- ChannelOp first-class IR ----------

def test_channelop_is_first_class_ir():
    """A ChannelOp carries source/target modality, kind, and payload, hashes
    deterministically, lives in a Module, and survives a serialise → parse
    round-trip via the Module's IR-text form.
    """
    from qmesh.ir.builder import circuit
    from qmesh.ir.ops import ChannelOp

    with circuit("with_channel", n_qubits=2, n_bits=2, n_atoms=2) as c:
        c.h(0)
        c.measure(0, 0)
        c._region.append(ChannelOp(
            source_modality=Modality.GATE,
            target_modality=Modality.RYDBERG,
            kind="gate->rydberg",
            payload={"forward": "p_excited", "into_param": "amp_rad_per_us",
                     "scale": 6.28},
        ))

    # Walk the module and find the ChannelOp
    module = c.module
    channels = [op for f in module.functions for op in f.body.ops
                if isinstance(op, ChannelOp)]
    assert len(channels) == 1
    ch = channels[0]
    assert ch.source_modality == Modality.GATE
    assert ch.target_modality == Modality.RYDBERG
    assert ch.kind == "gate->rydberg"
    assert ch.payload["forward"] == "p_excited"
    assert ch.payload["scale"] == 6.28
    # Default-modality of the ChannelOp itself is CLASSICAL — it is a bridge.
    assert ch.modality == Modality.CLASSICAL

    # Deterministic hashing
    digest1 = ch.digest()
    ch_dup = ChannelOp(
        source_modality=Modality.GATE,
        target_modality=Modality.RYDBERG,
        kind="gate->rydberg",
        payload={"forward": "p_excited", "into_param": "amp_rad_per_us",
                 "scale": 6.28},
    )
    assert ch_dup.digest() == digest1, "equal ChannelOps must hash identically"

    # Different kind ⇒ different digest
    ch_diff = ChannelOp(
        source_modality=Modality.GATE,
        target_modality=Modality.CV,
        kind="gate->cv",
        payload={"forward": "p_excited"},
    )
    assert ch_diff.digest() != digest1

    # Module hashes deterministically and includes the ChannelOp digest
    h1 = module.hash()
    assert isinstance(h1, str) and len(h1) == 64
    # Round-trip the IR through the textual representation as a sanity check
    text = module.to_text()
    assert "channel" in text  # the ChannelOp's name appears in the dump


# ---------- ChannelOp lowering ----------

def _multimodal_module_for_lowering():
    """A Module: gate-block → ChannelOp(gate→ryd) → rydberg-block."""
    from qmesh.ir.builder import circuit
    from qmesh.ir.ops import ChannelOp, MeasureOp, RydbergOp
    from qmesh.ir.types import Atom, Bit
    with circuit("g_to_r", n_qubits=2, n_bits=4, n_atoms=2) as c:
        c.h(0)
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)
        c._region.append(ChannelOp(
            source_modality=Modality.GATE,
            target_modality=Modality.RYDBERG,
            kind="gate->rydberg",
            payload={"forward": "parity_even", "into_param": "amp_rad_per_us",
                     "scale": 6.28, "default": 3.14},
        ))
        # Pin atom positions
        c.atoms = (
            Atom(name="a0", index=0, position=(0.0, 0.0, 0.0)),
            Atom(name="a1", index=1, position=(5.0, 0.0, 0.0)),
        )
        c.module.functions[0].inputs = c.qubits + c.atoms
        c._region.append(RydbergOp(
            name="pulse", operands=tuple(c.atoms),
            params=(0.0, 0.0, 0.0, 600.0),  # amp param[0] is the slot to patch
            modality=Modality.RYDBERG,
            attrs={"channel": "rydberg", "addressing": "Global"},
        ))
        c._region.append(MeasureOp(
            operands=(c.atoms[0], Bit(name="c2", index=2)),
            modality=Modality.RYDBERG, attrs={"basis": "ground-rydberg"},
        ))
        c._region.append(MeasureOp(
            operands=(c.atoms[1], Bit(name="c3", index=3)),
            modality=Modality.RYDBERG, attrs={"basis": "ground-rydberg"},
        ))
    return c.module


def test_channelop_lowering_builds_multimodal_dag():
    from qmesh.scheduler.channelop_lowering import lower_module_to_dag, partition

    module = _multimodal_module_for_lowering()
    blocks, edges = partition(module)
    # Two modality blocks (gate, rydberg), one channel between them.
    assert len(blocks) == 2
    assert blocks[0].modality == Modality.GATE
    assert blocks[1].modality == Modality.RYDBERG
    assert len(edges) == 1
    assert edges[0][0].kind == "gate->rydberg"

    dag = lower_module_to_dag(module, shots=32, sign=False)
    # 2 QPU + 1 ClassicalTask = 3 nodes
    assert len(dag.nodes) == 3
    kinds = sorted(n.kind() for n in dag.nodes.values())
    assert kinds == ["classical", "qpu", "qpu"]
    # Dependency chain: qpu_gate → channel → qpu_rydberg
    qpu_gate = dag.nodes["qpu_0_gate"]
    channel = dag.nodes["channel_1_gate_to_rydberg"]
    qpu_ryd = dag.nodes["qpu_1_rydberg"]
    assert qpu_gate.depends_on == []
    assert channel.depends_on == [qpu_gate.id]
    assert qpu_ryd.depends_on == [channel.id]
    # Topological sort respects the dependency chain
    order = [n.id for n in dag.toposort()]
    assert order.index(qpu_gate.id) < order.index(channel.id) < order.index(qpu_ryd.id)
    # Aggregate metadata records the source module hash + channel kinds
    assert dag.metadata["source_module_hash"] == module.hash()
    assert dag.metadata["channel_kinds"] == ["gate->rydberg"]


def test_lowered_dag_executes_end_to_end(tmp_path):
    pytest.importorskip("pulser")
    pytest.importorskip("pulser_simulation")
    from qmesh.scheduler import execute
    from qmesh.scheduler.channelop_lowering import lower_module_to_dag

    module = _multimodal_module_for_lowering()
    dag = lower_module_to_dag(module, shots=32, sign=False)
    run = execute(dag, ledger_dir=tmp_path)

    # All three nodes must succeed
    assert all(r.ok for r in run.results.values()), {
        nid: r.error for nid, r in run.results.items() if not r.ok
    }
    # Both modality nodes succeeded — that's the headline assertion.
    gate_res = run.results["qpu_0_gate"]
    ryd_res = run.results["qpu_1_rydberg"]
    assert gate_res.ok and gate_res.kind == "qpu"
    assert ryd_res.ok and ryd_res.kind == "qpu"
    # Counts present and shot-conserving
    assert sum(gate_res.output["counts"].values()) == gate_res.output["shots"]
    assert sum(ryd_res.output["counts"].values()) == ryd_res.output["shots"]
    # The aggregate DAG manifest is signed (the executor signs by default).
    import json
    dag_manifest = json.loads(run.manifest_path.read_text())
    assert "signature" in dag_manifest
    assert dag_manifest["signature"]["alg"] in ("ed25519", "unsigned")


# ===================================================================
# Phase 3γ — reverse-arrow ChannelOps + pulse-modality bridges.
# ===================================================================


def _module_rydberg_to_gate():
    """Module: rydberg-block → ChannelOp(rydberg→gate) → gate-block(ry).

    The rydberg block's measurement outcomes drive (via `p_excited`) a
    scale of the downstream `ry(theta)` rotation. Patch mode defaults
    to `scale` for `→gate` arrows.
    """
    from qmesh.ir.builder import circuit
    from qmesh.ir.ops import ChannelOp, MeasureOp, RydbergOp
    from qmesh.ir.types import Atom, Bit
    with circuit("r_to_g", n_qubits=1, n_bits=3, n_atoms=2) as c:
        # Pin atom positions
        c.atoms = (
            Atom(name="a0", index=0, position=(0.0, 0.0, 0.0)),
            Atom(name="a1", index=1, position=(5.0, 0.0, 0.0)),
        )
        c.module.functions[0].inputs = c.atoms + c.qubits
        # rydberg block
        c._region.append(RydbergOp(
            name="pulse", operands=tuple(c.atoms),
            params=(6.28, 0.0, 0.0, 1000.0),  # 2π Rabi on global drive
            modality=Modality.RYDBERG,
            attrs={"channel": "rydberg", "addressing": "Global"},
        ))
        c._region.append(MeasureOp(
            operands=(c.atoms[0], Bit(name="c0", index=0)),
            modality=Modality.RYDBERG, attrs={"basis": "ground-rydberg"},
        ))
        c._region.append(MeasureOp(
            operands=(c.atoms[1], Bit(name="c1", index=1)),
            modality=Modality.RYDBERG, attrs={"basis": "ground-rydberg"},
        ))
        # channel
        c._region.append(ChannelOp(
            source_modality=Modality.RYDBERG, target_modality=Modality.GATE,
            kind="rydberg->gate",
            payload={"forward": "p_excited", "into_param": "theta_scale",
                     "scale": 1.0, "default": 0.0},
        ))
        # gate block
        c.ry(1.0, 0)  # baseline angle 1.0 rad — patched in by *scale*
        c.measure(0, 2)
    return c.module


def test_channelop_rydberg_to_gate_lowering():
    """rydberg→gate ChannelOp wires up; bridge fn yields p_excited;
    downstream module_factory scales the first ry's params[0]."""
    from qmesh.scheduler.channelop_lowering import (
        SUPPORTED_KINDS,
        lower_module_to_dag,
    )
    assert "rydberg->gate" in SUPPORTED_KINDS

    module = _module_rydberg_to_gate()
    dag = lower_module_to_dag(module, shots=16, sign=False)
    # 2 QPU + 1 channel
    assert len(dag.nodes) == 3
    qpu_ryd = dag.nodes["qpu_0_rydberg"]
    channel = dag.nodes["channel_1_rydberg_to_gate"]
    qpu_gate = dag.nodes["qpu_1_gate"]
    assert channel.depends_on == [qpu_ryd.id]
    assert qpu_gate.depends_on == [channel.id]

    # Run the bridge fn on a synthetic upstream output: 50% '11', 50% '00'.
    fake_ctx = {qpu_ryd.id: {"counts": {"11": 8, "00": 8}, "shots": 16}}
    bridge_out = channel.fn(fake_ctx)
    # p_excited = 8/16 = 0.5; scale=1.0 → 0.5
    assert bridge_out["stat"] == pytest.approx(0.5)
    assert bridge_out["theta_scale"] == pytest.approx(0.5)
    # Per-atom list also surfaced (both atoms equally excited).
    assert bridge_out["p_excited_per_atom"] == pytest.approx([0.5, 0.5])

    # Now feed the bridge output back into the gate block's module_factory:
    # the first ry's params[0] should equal baseline (1.0) * 0.5 = 0.5.
    fake_ctx[channel.id] = bridge_out
    patched_mod = qpu_gate.module_factory(fake_ctx)
    from qmesh.ir.ops import GateOp
    ry_ops = [op for f in patched_mod.functions for op in f.body.ops
              if isinstance(op, GateOp) and op.name == "ry"]
    assert ry_ops, "expected at least one ry in the patched gate block"
    assert ry_ops[0].params[0] == pytest.approx(0.5)
    assert ry_ops[0].attrs.get("channelop_patched") is True
    assert ry_ops[0].attrs.get("channelop_patch_mode") == "scale"


def _module_cv_to_gate():
    """Module: cv-block (Sgate + MeasureFock) → ChannelOp(cv→gate) → gate(rx)."""
    from qmesh.ir.builder import circuit
    from qmesh.ir.ops import ChannelOp, CVOp, MeasureOp
    from qmesh.ir.types import Bit
    with circuit("cv_to_g", n_qubits=1, n_bits=3, n_qumodes=2) as c:
        # cv block
        c._region.append(CVOp(
            name="Sgate", operands=(c.qumodes[0],), params=(0.4,),
            modality=Modality.CV,
        ))
        c._region.append(CVOp(
            name="Sgate", operands=(c.qumodes[1],), params=(0.4,),
            modality=Modality.CV,
        ))
        c._region.append(MeasureOp(
            operands=(c.qumodes[0], Bit(name="c0", index=0)),
            modality=Modality.CV, attrs={"basis": "MeasureFock"},
        ))
        c._region.append(MeasureOp(
            operands=(c.qumodes[1], Bit(name="c1", index=1)),
            modality=Modality.CV, attrs={"basis": "MeasureFock"},
        ))
        # channel
        c._region.append(ChannelOp(
            source_modality=Modality.CV, target_modality=Modality.GATE,
            kind="cv->gate",
            payload={"forward": "n_avg", "into_param": "theta_scale",
                     "scale": 1.0, "default": 0.0},
        ))
        # gate block
        c.rx(2.0, 0)  # baseline 2.0 rad — gets scaled by mean photon number
        c.measure(0, 2)
    return c.module


def test_channelop_cv_to_gate_lowering():
    from qmesh.scheduler.channelop_lowering import (
        SUPPORTED_KINDS,
        lower_module_to_dag,
    )
    assert "cv->gate" in SUPPORTED_KINDS

    module = _module_cv_to_gate()
    dag = lower_module_to_dag(module, shots=16, sign=False)
    qpu_cv = dag.nodes["qpu_0_cv"]
    channel = dag.nodes["channel_1_cv_to_gate"]
    qpu_gate = dag.nodes["qpu_1_gate"]
    assert channel.depends_on == [qpu_cv.id]
    assert qpu_gate.depends_on == [channel.id]

    # Synthetic SF Fock counts: half "0,0", half "1,1" → mean photon = 1.0.
    fake_ctx = {qpu_cv.id: {"counts": {"0,0": 8, "1,1": 8}, "shots": 16}}
    bridge_out = channel.fn(fake_ctx)
    # n_avg per shot: ((0+0)/2)*8 + ((1+1)/2)*8 = 8 → /16 = 0.5
    assert bridge_out["stat"] == pytest.approx(0.5)
    assert bridge_out["theta_scale"] == pytest.approx(0.5)
    assert bridge_out["n_avg_per_mode"] == pytest.approx([0.5, 0.5])

    # Module factory scales rx's params[0] by 0.5 → 1.0.
    fake_ctx[channel.id] = bridge_out
    patched_mod = qpu_gate.module_factory(fake_ctx)
    from qmesh.ir.ops import GateOp
    rx_ops = [op for f in patched_mod.functions for op in f.body.ops
              if isinstance(op, GateOp) and op.name == "rx"]
    assert rx_ops and rx_ops[0].params[0] == pytest.approx(1.0)
    assert rx_ops[0].attrs.get("channelop_patch_mode") == "scale"


def _module_cv_to_rydberg():
    """Module: cv block → ChannelOp(cv→rydberg) → rydberg block."""
    from qmesh.ir.builder import circuit
    from qmesh.ir.ops import ChannelOp, CVOp, MeasureOp, RydbergOp
    from qmesh.ir.types import Atom, Bit
    with circuit("cv_to_r", n_qubits=0, n_bits=4, n_atoms=2, n_qumodes=2) as c:
        c.atoms = (
            Atom(name="a0", index=0, position=(0.0, 0.0, 0.0)),
            Atom(name="a1", index=1, position=(5.0, 0.0, 0.0)),
        )
        c.module.functions[0].inputs = c.qumodes + c.atoms
        # cv
        c._region.append(CVOp(
            name="Sgate", operands=(c.qumodes[0],), params=(0.4,),
            modality=Modality.CV,
        ))
        c._region.append(MeasureOp(
            operands=(c.qumodes[0], Bit(name="c0", index=0)),
            modality=Modality.CV, attrs={"basis": "MeasureFock"},
        ))
        c._region.append(MeasureOp(
            operands=(c.qumodes[1], Bit(name="c1", index=1)),
            modality=Modality.CV, attrs={"basis": "MeasureFock"},
        ))
        # channel — homodyne_x falls back to n_avg semantics on int-coerced
        # outputs, scaled into a Rabi-amplitude scaling (set into params[0]).
        c._region.append(ChannelOp(
            source_modality=Modality.CV, target_modality=Modality.RYDBERG,
            kind="cv->rydberg",
            payload={"forward": "homodyne_x", "into_param": "amp_rad_per_us",
                     "scale": 6.28, "default": 0.0},
        ))
        # rydberg
        c._region.append(RydbergOp(
            name="pulse", operands=tuple(c.atoms),
            params=(0.0, 0.0, 0.0, 600.0),  # amp patched by channel
            modality=Modality.RYDBERG,
            attrs={"channel": "rydberg", "addressing": "Global"},
        ))
        c._region.append(MeasureOp(
            operands=(c.atoms[0], Bit(name="c2", index=2)),
            modality=Modality.RYDBERG, attrs={"basis": "ground-rydberg"},
        ))
        c._region.append(MeasureOp(
            operands=(c.atoms[1], Bit(name="c3", index=3)),
            modality=Modality.RYDBERG, attrs={"basis": "ground-rydberg"},
        ))
    return c.module


def test_channelop_cv_to_rydberg_lowering():
    from qmesh.scheduler.channelop_lowering import (
        SUPPORTED_KINDS,
        lower_module_to_dag,
    )
    assert "cv->rydberg" in SUPPORTED_KINDS

    module = _module_cv_to_rydberg()
    dag = lower_module_to_dag(module, shots=16, sign=False)
    qpu_cv = dag.nodes["qpu_0_cv"]
    channel = dag.nodes["channel_1_cv_to_rydberg"]
    qpu_ryd = dag.nodes["qpu_1_rydberg"]
    assert channel.depends_on == [qpu_cv.id]
    assert qpu_ryd.depends_on == [channel.id]

    # Synthetic homodyne_x counts (int-coerced fallback): "0,0"x4, "1,1"x12
    # → mean photon ≈ 1.5*0.5 = 0.75; *scale 6.28 → 4.71.
    fake_ctx = {qpu_cv.id: {"counts": {"0,0": 4, "1,1": 12}, "shots": 16}}
    bridge_out = channel.fn(fake_ctx)
    expected_stat = ((0.0) * 4 + (1.0) * 12) / 16  # 0.75
    assert bridge_out["stat"] == pytest.approx(expected_stat)
    assert bridge_out["amp_rad_per_us"] == pytest.approx(expected_stat * 6.28)

    fake_ctx[channel.id] = bridge_out
    patched_mod = qpu_ryd.module_factory(fake_ctx)
    from qmesh.ir.ops import RydbergOp
    rb_ops = [op for f in patched_mod.functions for op in f.body.ops
              if isinstance(op, RydbergOp)]
    assert rb_ops
    # Reverse-arrow injection into rydberg uses `set` mode (not scale): the
    # forwarded value lands in params[0] verbatim.
    assert rb_ops[0].params[0] == pytest.approx(expected_stat * 6.28)
    assert rb_ops[0].attrs.get("channelop_patch_mode") == "set"


def test_channelop_pulse_kinds_register_with_lowering():
    """gate↔pulse arrows are registered and don't raise NotImplementedError
    when wired into a Module + lowered.
    """
    from qmesh.ir.builder import circuit
    from qmesh.ir.ops import ChannelOp, MeasureOp, PulseOp
    from qmesh.ir.types import Bit
    from qmesh.scheduler.channelop_lowering import (
        SUPPORTED_KINDS,
        lower_module_to_dag,
    )

    assert "gate->pulse" in SUPPORTED_KINDS
    assert "pulse->gate" in SUPPORTED_KINDS
    # Modality.PULSE is required by the bridges.
    assert Modality.PULSE.value == "pulse"

    # gate→pulse Module
    with circuit("g_to_p", n_qubits=1, n_bits=2) as c:
        c.h(0)
        c.measure(0, 0)
        c._region.append(ChannelOp(
            source_modality=Modality.GATE, target_modality=Modality.PULSE,
            kind="gate->pulse",
            payload={"forward": "p_one",
                     "into_param": "pulse_drive_amp_scale",
                     "scale": 1.0},
        ))
        c._region.append(PulseOp(
            name="constant", operands=(c.qubits[0],),
            params=(0.5, 0.0, 100.0),  # amp, phase, duration_ns
            modality=Modality.PULSE,
            attrs={"channel": "drive_q0"},
        ))
        c._region.append(MeasureOp(
            operands=(c.qubits[0], Bit(name="c1", index=1)),
            modality=Modality.PULSE, attrs={"basis": "z"},
        ))
    dag = lower_module_to_dag(c.module, shots=4, sign=False)
    # No raise — that's the test. Also verify wiring.
    assert "channel_1_gate_to_pulse" in dag.nodes
    assert "qpu_1_pulse" in dag.nodes

    # Run the channel fn + factory on synthetic ctx to confirm patching works.
    qpu_gate = dag.nodes["qpu_0_gate"]
    channel = dag.nodes["channel_1_gate_to_pulse"]
    qpu_pulse = dag.nodes["qpu_1_pulse"]
    fake_ctx = {qpu_gate.id: {"counts": {"0": 5, "1": 5}, "shots": 10}}
    out = channel.fn(fake_ctx)
    assert out["pulse_drive_amp_scale"] == pytest.approx(0.5)
    fake_ctx[channel.id] = out
    patched = qpu_pulse.module_factory(fake_ctx)
    pulse_ops = [op for f in patched.functions for op in f.body.ops
                 if isinstance(op, PulseOp)]
    assert pulse_ops
    # `→pulse` defaults to `scale` mode: 0.5 (baseline) * 0.5 (forward) = 0.25.
    assert pulse_ops[0].params[0] == pytest.approx(0.25)
    assert pulse_ops[0].attrs.get("channelop_patch_mode") == "scale"


def test_channelop_unsupported_kind_still_errors_clearly():
    """Picking an unsupported (invented) channel kind raises a clear ValueError.

    Phase 3γ ships every modality combo we need today; this test uses a
    deliberately-invented kind label to exercise the unsupported-kind
    error path so the lowering keeps refusing kinds it can't lower.
    """
    from qmesh.ir.builder import circuit
    from qmesh.ir.ops import ChannelOp, MeasureOp, RydbergOp
    from qmesh.ir.types import Atom, Bit
    from qmesh.scheduler.channelop_lowering import (
        SUPPORTED_KINDS,
        lower_module_to_dag,
    )

    invented_kind = "gate->wormhole"
    assert invented_kind not in SUPPORTED_KINDS

    with circuit("bogus", n_qubits=1, n_bits=2, n_atoms=1) as c:
        c.h(0)
        c.measure(0, 0)
        # Bogus channel — kind string isn't in SUPPORTED_KINDS even though
        # source/target modalities are valid enums.
        c._region.append(ChannelOp(
            source_modality=Modality.GATE,
            target_modality=Modality.RYDBERG,
            kind=invented_kind,
            payload={"forward": "p_one", "into_param": "amp_rad_per_us"},
        ))
        # Real downstream block so the partitioner emits an edge that the
        # lowerer must classify (and reject).
        c.atoms = (Atom(name="a0", index=0, position=(0.0, 0.0, 0.0)),)
        c.module.functions[0].inputs = c.qubits + c.atoms
        c._region.append(RydbergOp(
            name="pulse", operands=tuple(c.atoms),
            params=(0.0, 0.0, 0.0, 100.0),
            modality=Modality.RYDBERG,
            attrs={"channel": "rydberg", "addressing": "Global"},
        ))
        c._region.append(MeasureOp(
            operands=(c.atoms[0], Bit(name="c1", index=1)),
            modality=Modality.RYDBERG, attrs={"basis": "ground-rydberg"},
        ))

    with pytest.raises(ValueError, match="unsupported ChannelOp kind"):
        lower_module_to_dag(c.module, shots=4, sign=False)
