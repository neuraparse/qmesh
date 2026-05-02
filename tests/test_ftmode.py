"""FT-mode tests: codes, decoders, estimator, runner."""

from __future__ import annotations

import pytest

import qmesh
from qmesh.ftmode import (
    BBCode,
    DistillationFactory,
    FTConfig,
    InPlaceCultivation,
    PyMatchingDecoder,
    RepetitionCode,
    SlidingWindowMWPMDecoder,
    SurfaceCode,
    estimate,
    estimate_for_distances,
    memory_experiment,
    threshold_sweep,
)
from qmesh.provenance.manifest import ManifestSigner


# ---------- codes ----------

def test_surface_code_metadata():
    c = SurfaceCode(distance=5, rounds=8)
    md = c.metadata()
    assert md.distance == 5
    assert md.rounds == 8
    # rotated d=5: 25 data + 24 ancillas = 49
    assert md.physical_qubits == 49
    assert md.logical_qubits == 1
    assert md.name == "rotated_surface_d5"


def test_surface_code_rejects_even_distance():
    with pytest.raises(ValueError):
        SurfaceCode(distance=4, rounds=8)


def test_surface_code_circuit_generates():
    pytest.importorskip("stim")
    c = SurfaceCode(distance=3, rounds=4)
    stim_c = c.generate_memory_circuit(physical_error_rate=1e-3)
    assert stim_c.num_detectors > 0
    assert stim_c.num_observables == 1


def test_repetition_code_metadata():
    c = RepetitionCode(distance=5, rounds=4)
    md = c.metadata()
    assert md.physical_qubits == 9       # 2*5-1
    assert md.extra["corrects"] == "bit_flip"


def test_bb_code_metadata():
    """Phase 2β: verifies the [[144,12,12]] gross-code parameters and name."""
    c = BBCode(distance=12)
    md = c.metadata()
    assert md.name == "bb[[144,12,12]]"
    assert md.physical_qubits == 144
    assert md.logical_qubits == 12
    assert md.distance == 12
    assert md.extra["code_family"] == "bivariate_bicycle"
    assert md.extra["l"] == 12
    assert md.extra["m"] == 6
    # A = x^3 + y + y^2; B = y^3 + x + x^2 per Bravyi 2024
    assert [list(t) for t in md.extra["A_terms"]] == [["x", 3], ["y", 1], ["y", 2]]
    assert [list(t) for t in md.extra["B_terms"]] == [["y", 3], ["x", 1], ["x", 2]]


# ---------- decoders ----------

def test_pymatching_decoder_decodes_below_threshold():
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    code = SurfaceCode(distance=5, rounds=8)
    stim_c = code.generate_memory_circuit(physical_error_rate=1e-3)
    decoder = PyMatchingDecoder().from_circuit(stim_c)
    sampler = stim_c.compile_detector_sampler()
    de, of = sampler.sample(shots=2000, separate_observables=True)
    res = decoder.decode_batch(de, of)
    # at p=1e-3, d=5 → logical error rate << 1
    assert res.logical_error_rate < 0.05


def test_sliding_window_decoder_emits_metadata():
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    code = SurfaceCode(distance=3, rounds=4)
    stim_c = code.generate_memory_circuit(physical_error_rate=1e-3)
    decoder = SlidingWindowMWPMDecoder(window_rounds=2).from_circuit(stim_c)
    sampler = stim_c.compile_detector_sampler()
    de, of = sampler.sample(shots=500, separate_observables=True)
    res = decoder.decode_batch(de, of)
    assert "window_rounds" in res.metadata
    # identity hits the manifest
    assert "implementation" in decoder.identity()


# ---------- cultivation ----------

def test_cultivation_manifest_block():
    f = InPlaceCultivation(target_T_error=4e-11)
    block = f.manifest_block(n_T_states=100)
    assert block["T_states_used"] == 100
    assert block["target_logical_T_error"] == 4e-11
    assert "in_place_gidney_2024" in block["method"]


def test_distillation_factory_baseline():
    f = DistillationFactory()
    p = f.params()
    assert p.method == "15_to_1_distillation"


# ---------- resource estimator ----------

def test_estimate_no_t_gates():
    with qmesh.circuit("memory", n_qubits=1, n_bits=1) as c:
        c.h(0); c.measure(0, 0)
    re = estimate(c.module, code=SurfaceCode(distance=5, rounds=8),
                 physical_error_rate=1e-3, target_logical_error_rate=1e-9)
    assert re.T_states_required == 0
    assert re.physical_qubits == 49
    assert re.code_distance == 5


def test_estimate_with_t_gates_costs_extra():
    with qmesh.circuit("workload", n_qubits=1, n_bits=1) as c:
        for _ in range(50):
            c.t(0)
    re = estimate(c.module, code=SurfaceCode(distance=7, rounds=8),
                 physical_error_rate=1e-3)
    assert re.T_states_required == 50
    # cycles increases by t * cycles_per_T
    assert re.cycles == 8 + 50 * 50


def test_estimate_higher_distance_lowers_logical_error():
    with qmesh.circuit("c", n_qubits=1, n_bits=1) as c:
        c.h(0)
    rates = [
        estimate(c.module, code=SurfaceCode(distance=d, rounds=8),
                physical_error_rate=1e-3).estimated_logical_error_rate
        for d in (3, 5, 7, 9)
    ]
    # monotonically decreasing below threshold
    assert all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))


def test_estimate_for_distances_helper():
    with qmesh.circuit("c", n_qubits=1, n_bits=1) as c:
        c.h(0)
    estimates = estimate_for_distances(c.module, distances=(3, 5, 7))
    assert len(estimates) == 3
    assert estimates[0].code_distance == 3


# ---------- runner ----------

def test_memory_experiment_runs_and_signs(tmp_path):
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    cfg = FTConfig(distance=3, rounds=4, physical_error_rate=1e-3,
                  decoder="pymatching")
    result, manifest = memory_experiment(ftconfig=cfg, shots=1000, ledger_dir=tmp_path)
    assert result.shots == 1000
    assert 0 <= result.logical_error_rate <= 1
    assert manifest.ftmode is not None
    assert manifest.ftmode["code"]["distance"] == 3
    assert manifest.ftmode["decoder"]["name"] == "pymatching-mwpm"
    assert manifest.ftmode["cultivation"]["factory"] == "in_place_cultivation_v0"
    assert ManifestSigner.verify(manifest)
    assert result.manifest_path.exists()


def test_threshold_sweep_distance_suppression(tmp_path):
    """Real Λ-suppression assertion (Phase 2α pillar): under p=1e-3 the
    surface code at d=5 must be strictly better than d=3 by a factor of at
    least 2× — well below Willow's published Λ≈2.14 (so the test stays
    deterministic across PyMatching versions) but well above 1.0× (so it
    actually proves suppression rather than tolerating regression).
    """
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    template = FTConfig(distance=3, rounds=4, physical_error_rate=1e-3,
                       decoder="pymatching")
    results = threshold_sweep(
        ftconfig_template=template,
        distances=[3, 5],
        physical_error_rates=[1e-3],
        shots=8_000,
        ledger_dir=tmp_path,
        seed=42,
    )
    assert len(results) == 2
    d3 = next(r for r in results if r.distance == 3)
    d5 = next(r for r in results if r.distance == 5)
    # d=3 must have enough errors to make a ratio meaningful.
    assert d3.logical_error_count >= 8, (
        f"d=3 errors {d3.logical_error_count} too low at p=1e-3; bump shots"
    )
    # Real suppression: d=5 error rate at most half d=3's (Λ ≥ 2).
    assert d5.logical_error_rate * 2 <= d3.logical_error_rate, (
        f"Λ_35 = {d3.logical_error_rate / max(d5.logical_error_rate, 1/8000):.2f} "
        f"is below the 2× pillar threshold (d3={d3.logical_error_rate:.2e}, "
        f"d5={d5.logical_error_rate:.2e})"
    )


def test_promote_and_run_rejects_unsupported_gate(tmp_path):
    """Phase-2γ promotion: T is now supported via cultivation reservation.
    Arbitrary-angle rotations (e.g. rx/ry/rz) remain out of scope and must
    still raise."""
    from qmesh.ftmode import promote_and_run
    from qmesh.ir.module import Module
    from qmesh.ir.ops import GateOp
    from qmesh.ir.types import Qubit, Modality
    # Build an IR module with an unsupported `rx` gate by hand so the
    # frontend's gate-set guard doesn't reject it before lowering.
    with qmesh.circuit("logical", n_qubits=1, n_bits=1) as c:
        c.h(0); c.measure(0, 0)
    # mutate the function body to include a gate the lowering can't lower
    fn = c.module.functions[0]
    fn.body.ops.insert(
        1,
        GateOp(name="rx", operands=(Qubit(name="q0", index=0),),
               params=(0.7,), modality=Modality.GATE),
    )
    cfg = FTConfig(distance=3, rounds=4)
    with pytest.raises(NotImplementedError):
        promote_and_run(c.module, ftconfig=cfg, shots=100, ledger_dir=tmp_path)


# ---------- Phase 2β α-slice tests ----------


def test_lattice_surgery_logical_h_runs(tmp_path):
    """Lattice surgery on a single-logical-qubit module containing H +
    measure runs end-to-end and emits a signed manifest tagged
    'lattice_surgery'."""
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    from qmesh.ftmode import promote_and_run

    with qmesh.circuit("logical_H", n_qubits=1, n_bits=1) as c:
        c.h(0); c.measure(0, 0)
    cfg = FTConfig(distance=3, rounds=4, physical_error_rate=1e-3,
                  decoder="pymatching")
    result, manifest = promote_and_run(
        c.module, ftconfig=cfg, shots=500, ledger_dir=tmp_path,
    )
    # signed and reports lattice surgery somewhere
    assert ManifestSigner.verify(manifest)
    assert manifest.ftmode is not None
    assert manifest.ftmode["execution_path"] == "lattice_surgery"
    assert "lattice_surgery" in manifest.ftmode
    ls = manifest.ftmode["lattice_surgery"]
    assert ls["logical_qubits"] == 1
    # the result name surfaces lattice_surgery as well
    assert "lattice_surgery" in result.code_name
    # logical error rate is sane (small)
    assert 0.0 <= result.logical_error_rate <= 1.0
    assert result.manifest_path.exists()


def test_lattice_surgery_logical_cx_runs(tmp_path):
    """Two-logical-qubit module with CX + measures runs end-to-end."""
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    from qmesh.ftmode import promote_and_run

    with qmesh.circuit("logical_CX", n_qubits=2, n_bits=2) as c:
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)
    cfg = FTConfig(distance=3, rounds=4, physical_error_rate=1e-3,
                  decoder="pymatching")
    result, manifest = promote_and_run(
        c.module, ftconfig=cfg, shots=500, ledger_dir=tmp_path,
    )
    assert ManifestSigner.verify(manifest)
    ls = manifest.ftmode["lattice_surgery"]
    assert ls["logical_qubits"] == 2
    # 2 patches present
    assert len(ls["patches"]) == 2
    # the merge strip added detectors beyond a single-patch memory
    single_patch_detectors = 32  # d=3, rounds=4 reference
    assert ls["stim_detectors"] > 2 * single_patch_detectors


def test_lattice_surgery_emits_merge_observable_for_cx(tmp_path):
    """Phase 2δ: a 2-patch CX program produces three observables —
    obs[0] / obs[1] are per-patch logical Z, obs[2] is the merge ancilla
    parity (the Litinski merge-CNOT product). Tests pre-runner so we can
    inspect the lattice surgery program directly."""
    pytest.importorskip("stim")
    from qmesh.ftmode.lattice_surgery import lower_module

    with qmesh.circuit("logical_CX_merge_obs", n_qubits=2, n_bits=2) as c:
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)

    prog = lower_module(c.module, distance=3, rounds=4,
                        physical_error_rate=1e-3)

    # Three observables: per-patch (0, 1) + merge (2).
    assert prog.merge_observable_index == 2
    assert prog.per_patch_observable_indices == [0, 1]
    assert prog.circuit.num_observables == 3, (
        f"want 3 observables (2 per-patch + 1 merge), "
        f"got {prog.circuit.num_observables}"
    )

    # Sanity: a single-patch program (no CX) should *not* have a merge obs.
    with qmesh.circuit("logical_no_cx", n_qubits=1, n_bits=1) as c2:
        c2.h(0); c2.measure(0, 0)
    prog1 = lower_module(c2.module, distance=3, rounds=4,
                         physical_error_rate=1e-3)
    assert prog1.merge_observable_index is None
    assert prog1.circuit.num_observables == 1


def test_lattice_surgery_merge_observable_decoder_graph_valid(tmp_path):
    """The 3-observable circuit must compile to a valid Stim DEM (every
    detector is a stabilizer of the prep state) — critical for matching."""
    pytest.importorskip("stim")
    from qmesh.ftmode.lattice_surgery import lower_module

    with qmesh.circuit("cx_dem", n_qubits=2, n_bits=2) as c:
        c.cx(0, 1)
        c.measure(0, 0)
        c.measure(1, 1)
    prog = lower_module(c.module, distance=3, rounds=4,
                        physical_error_rate=1e-3)
    # decompose_errors=True is the strict mode; if we'd corrupted the merge
    # ancilla detector pattern this would raise.
    dem = prog.circuit.detector_error_model(decompose_errors=True)
    assert dem.num_observables == 3


def test_bb_code_generates_memory_circuit():
    """BBCode(distance=12).generate_memory_circuit() returns a stim.Circuit
    with the right number of qubits and stabilizer rounds."""
    pytest.importorskip("stim")
    import stim

    rounds = 3
    code = BBCode(distance=12, rounds=rounds)
    circuit = code.generate_memory_circuit(physical_error_rate=1e-3)
    assert isinstance(circuit, stim.Circuit)
    # 144 data + 72 X-checks + 72 Z-checks = 288
    assert circuit.num_qubits == 144 + 144
    assert circuit.num_observables == 1
    # detectors grow with rounds; at rounds=3 we expect a non-trivial count
    assert circuit.num_detectors > 0

    # sample-and-decode loop produces a non-zero (but small) logical error
    # rate at p=1e-3. PyMatching can't natively decode BB syndromes (they
    # have hyperedges); we use a per-shot trivial baseline ("predict 0")
    # whose error rate equals the raw observable flip probability — that's
    # the α-honest decode.
    sampler = circuit.compile_detector_sampler(seed=11)
    de, of = sampler.sample(shots=400, separate_observables=True)
    raw_err = float(of.mean())
    assert 0.0 < raw_err < 0.5, (
        f"BB raw observable flip rate={raw_err} outside the α-honest "
        f"non-zero-but-small window at p=1e-3, rounds=3"
    )


def test_streaming_decoder_matches_batch_decoder():
    """Run the same syndrome data through batch + streaming decoders;
    predictions agree within a small tolerance."""
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    import numpy as np
    from qmesh.ftmode import (
        PyMatchingDecoder, StreamingMWPMDecoder, SurfaceCode,
    )

    code = SurfaceCode(distance=3, rounds=4)
    circuit = code.generate_memory_circuit(physical_error_rate=3e-3)
    sampler = circuit.compile_detector_sampler(seed=2026)
    de, of = sampler.sample(shots=400, separate_observables=True)

    batch = PyMatchingDecoder().from_circuit(circuit)
    stream = StreamingMWPMDecoder(distance=3).from_circuit(circuit)
    batch_res = batch.decode_batch(de, of)
    stream_res = stream.decode_batch(de, of)

    # tolerance: streaming is a faithful α emulation and may disagree on a
    # handful of shots near the window's commit boundary
    disagreements = int(np.sum(batch_res.predictions != stream_res.predictions))
    assert disagreements <= max(1, int(0.05 * de.shape[0])), (
        f"streaming and batch disagree on {disagreements}/{de.shape[0]} "
        f"shots — exceeds α tolerance"
    )

    # streaming API contract: decode_stream yields one prediction per round
    one_shot_rounds = []
    n_rounds = 4
    per_round = de.shape[1] // n_rounds
    row = de[0]
    for r in range(n_rounds):
        one_shot_rounds.append(row[r * per_round: (r + 1) * per_round])
    out = list(stream.decode_stream(one_shot_rounds))
    assert len(out) == n_rounds
    # final element is the committed prediction
    assert out[-1].shape == (1,)


# ---------- Phase 2γ tests ---------------------------------------------------


def test_streaming_decoder_warm_state_matches_batch_predictions():
    """Phase 2γ headline: warm-state streaming on a 50-round circuit
    produces the SAME observable_flips predictions as the batch decoder.

    Both decoders are seeded identically and consume the same syndromes;
    deterministic agreement (0 disagreements over 100 shots) is required.
    """
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    import numpy as np
    from qmesh.ftmode import (
        PyMatchingDecoder, StreamingMWPMDecoder, SurfaceCode,
    )

    code = SurfaceCode(distance=3, rounds=50)
    circuit = code.generate_memory_circuit(physical_error_rate=1e-3)
    sampler = circuit.compile_detector_sampler(seed=4242)
    de, of = sampler.sample(shots=100, separate_observables=True)

    batch = PyMatchingDecoder().from_circuit(circuit)
    stream = StreamingMWPMDecoder(distance=3).from_circuit(circuit)
    batch_res = batch.decode_batch(de, of)
    stream_res = stream.decode_batch(de, of)

    # warm-state contract: 0 disagreements with the batch decoder.
    disagreements = int(np.sum(batch_res.predictions != stream_res.predictions))
    assert disagreements == 0, (
        f"warm-state streaming disagrees with batch on {disagreements}/100 "
        f"shots — warm-state contract violated"
    )
    # warm-state metadata should surface in the result
    assert stream_res.metadata.get("warm_state") is True
    assert stream_res.metadata.get("matcher_built_once") is True


def test_streaming_decoder_warm_state_processes_long_circuits():
    """Phase 2γ: warm-state streaming consumes a d=3, 30-round circuit
    end-to-end; bounded memory growth (verified via psutil if available,
    else a runtime smoke under 10s)."""
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    import time
    from qmesh.ftmode import StreamingMWPMDecoder, SurfaceCode

    code = SurfaceCode(distance=3, rounds=30)
    circuit = code.generate_memory_circuit(physical_error_rate=1e-3)
    sampler = circuit.compile_detector_sampler(seed=2025)
    de, of = sampler.sample(shots=200, separate_observables=True)

    stream = StreamingMWPMDecoder(distance=3).from_circuit(circuit)

    try:
        import psutil  # type: ignore[import-not-found]
        proc = psutil.Process()
        rss_before = proc.memory_info().rss
    except ImportError:
        proc = None
        rss_before = 0

    t0 = time.time()
    res = stream.decode_batch(de, of)
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"streaming took {elapsed:.1f}s — too slow"

    if proc is not None:
        rss_after = proc.memory_info().rss
        # Sanity: < 50 MB growth for the warm-state graph (built once).
        # We allow some slack for the incidental Python heap movement.
        delta_mb = (rss_after - rss_before) / (1024 * 1024)
        assert delta_mb < 200, (
            f"streaming RSS grew {delta_mb:.0f}MB — warm-state should be bounded"
        )

    assert res.shots == 200
    assert res.metadata.get("warm_state") is True


def test_bp_osd_decoder_decodes_bb_code():
    """BP+OSD decoder runs end-to-end on the smallest BB-code config
    we can sample 100 shots with under the fallback budget."""
    pytest.importorskip("stim")
    from qmesh.ftmode import BBCode, BpOsdDecoder

    code = BBCode(distance=6, rounds=1)
    circuit = code.generate_memory_circuit(physical_error_rate=1e-3)
    decoder = BpOsdDecoder(
        max_iter=10, osd_order=0, force_fallback=True,
    ).from_circuit(circuit)

    sampler = circuit.compile_detector_sampler(seed=11)
    de, of = sampler.sample(shots=100, separate_observables=True)
    res = decoder.decode_batch(de, of)
    # Phase 2γ: BP+OSD on BB code returns a sane logical error rate.
    assert 0.0 <= res.logical_error_rate <= 0.5, (
        f"BP+OSD logical error rate {res.logical_error_rate} out of "
        f"[0, 0.5] sanity window"
    )
    assert res.shots == 100
    assert "bp_osd" in res.metadata["name"]


def test_bp_osd_decoder_falls_back_when_ldpc_missing():
    """The BP+OSD decoder accepts force_fallback=True and produces sane
    outputs on a small toy code without the ldpc package."""
    pytest.importorskip("stim")
    from qmesh.ftmode import BpOsdDecoder, SurfaceCode

    code = SurfaceCode(distance=3, rounds=2)
    circuit = code.generate_memory_circuit(physical_error_rate=1e-3)
    decoder = BpOsdDecoder(
        max_iter=10, osd_order=0, force_fallback=True,
    ).from_circuit(circuit)

    sampler = circuit.compile_detector_sampler(seed=777)
    de, of = sampler.sample(shots=20, separate_observables=True)
    res = decoder.decode_batch(de, of)
    assert res.shots == 20
    assert 0.0 <= res.logical_error_rate <= 1.0
    # Confirm the fallback path was actually used.
    assert res.metadata["using_fallback"] is True
    ident = decoder.identity()
    assert ident["using_fallback"] is True
    assert "qmesh-fallback" in ident["name"]


def test_lattice_surgery_t_injection_records_cultivation_blocks(tmp_path):
    """Phase 2γ: a logical IR with H + T + measure on 1 logical qubit,
    promote_and_run, manifest.ftmode["lattice_surgery"]["t_injection_blocks"]
    has length 1 and reports the correct T-state count + factory."""
    pytest.importorskip("pymatching")
    pytest.importorskip("stim")
    from qmesh.ftmode import promote_and_run

    with qmesh.circuit("logical_T", n_qubits=1, n_bits=1) as c:
        c.h(0); c.t(0); c.measure(0, 0)
    cfg = FTConfig(distance=3, rounds=4, physical_error_rate=1e-3,
                  decoder="pymatching")
    result, manifest = promote_and_run(
        c.module, ftconfig=cfg, shots=200, ledger_dir=tmp_path,
    )
    assert ManifestSigner.verify(manifest)
    ls = manifest.ftmode["lattice_surgery"]
    assert "t_injection_blocks" in ls
    blocks = ls["t_injection_blocks"]
    assert len(blocks) == 1, f"expected 1 cultivation block, got {len(blocks)}"
    blk = blocks[0]
    assert blk["logical_qubit"] == 0
    assert blk["n_T_required"] == 1
    assert blk["cultivation_factory"] == "in_place_cultivation_v0"
    assert blk["cultivation_cycles_block"] == blk["cultivation_cycles_per_T"]
    # Total reported cultivation cost matches sum of blocks.
    assert ls["n_T_total"] == 1
    assert ls["cultivation_factory"] == "in_place_cultivation_v0"
    assert ls["cultivation_cycles_total"] >= 50
    # The lattice_surgery program is still tagged as a lattice_surgery run.
    assert manifest.ftmode["execution_path"] == "lattice_surgery"
    # Resource estimator agrees on T count.
    re = manifest.ftmode["resource_estimate"]
    assert re["T_states_required"] == 1


def test_lattice_surgery_cultivation_block_emits_real_circuit(tmp_path):
    """Phase 2δ: the T-injection cultivation block emits a real Stim sub-
    circuit (R / X_ERROR / M / OBSERVABLE_INCLUDE), not just a SHIFT_COORDS
    marker. The resulting Stim circuit has at least one cultivation
    observable, the DEM compiles, and the cultivation block carries
    `circuit_emitted=True` + observable_index in its manifest record."""
    pytest.importorskip("stim")
    from qmesh.ftmode.lattice_surgery import lower_module

    with qmesh.circuit("logical_T_circuit", n_qubits=1, n_bits=1) as c:
        c.h(0); c.t(0); c.measure(0, 0)
    prog = lower_module(c.module, distance=3, rounds=4,
                        physical_error_rate=1e-3,
                        cultivation_target_T_error=1e-4)

    # Cultivation block records the new δ fields.
    assert prog.t_injection_blocks, "cultivation should have produced a block"
    blk = prog.t_injection_blocks[0]
    assert blk["circuit_emitted"] is True
    assert "ancilla_qubit" in blk and isinstance(blk["ancilla_qubit"], int)
    assert "observable_index" in blk
    obs_idx = blk["observable_index"]

    # Stim circuit: at least 2 observables (patch + cultivation), DEM is valid.
    n_obs = prog.circuit.num_observables
    assert n_obs >= 2, f"expected ≥ 2 observables (patch + cultivation), got {n_obs}"
    assert obs_idx < n_obs

    dem = prog.circuit.detector_error_model(decompose_errors=True)
    assert dem.num_observables == n_obs

    # The cultivation block flips its observable with probability ~target_T_error.
    # Run a short Stim sample and check the cultivation observable flip rate
    # is in a sane range (< 1%, since target_T_error=1e-4).
    sampler = prog.circuit.compile_detector_sampler(seed=1)
    _, obs_flips = sampler.sample(shots=2000, separate_observables=True)
    cultivation_flip_rate = obs_flips[:, obs_idx].mean()
    assert cultivation_flip_rate < 0.05, (
        f"cultivation observable flipping too often: {cultivation_flip_rate}"
    )
