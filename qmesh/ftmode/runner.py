"""qmesh.ftmode.runner — FT-mode runtime.

The runner glues code generation + sampling + decoding + manifest population.

Phase-2α surface (in-scope): logical-memory experiments. The user provides a
code and a decoder; the runner generates the Stim circuit, samples shots,
runs the decoder, and emits a manifest with `ftmode` block populated.

Phase-2β scope (future): logical-gate sequences (CX, S, H, T injection)
between multiple logical qubits via lattice surgery. The framework already
allocates a `cultivation` slot in the manifest so T-gate counts can be
recorded; the actual circuit emission is the upstream-research-code item.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qmesh import __version__
from qmesh.ftmode.estimate import estimate
from qmesh.provenance.manifest import Manifest, ManifestSigner, _canonical_json

if TYPE_CHECKING:
    from qmesh.ftmode import FTConfig


@dataclass(slots=True)
class FTResult:
    logical_error_rate: float
    logical_error_count: int
    shots: int
    rounds: int
    distance: int
    code_name: str
    decoder_name: str
    wall_seconds: float
    decode_seconds: float
    sample_seconds: float
    manifest_path: Path | None = None


def memory_experiment(
    *,
    ftconfig: "FTConfig",
    shots: int = 30_000,
    ledger_dir: Path | str = "ledger/ft",
    sign: bool = True,
    seed: int | None = None,
    chain_to: Path | str | None = None,
) -> tuple[FTResult, Manifest]:
    """Run a memory experiment for the given FTConfig and emit a manifest.

    Pass `seed` to make the Stim sampler deterministic — useful in tests
    that assert numerical Λ-suppression bounds.

    Pass `chain_to` (path to a prior FT manifest JSON) to link this run
    to that one in the manifest's ftmode.cultivation_audit.cumulative_ft_chain
    slot — so a campaign of FT runs forms a cryptographic chain by hash.
    """
    from qmesh.ftmode import FTConfig as _FTC  # type: ignore[unused-ignore]
    if not isinstance(ftconfig, _FTC):
        raise TypeError("ftconfig must be an FTConfig instance")

    code = ftconfig.build_code()
    decoder = ftconfig.build_decoder()

    # 1. Generate the FT circuit
    t0 = time.time()
    stim_circuit = code.generate_memory_circuit(
        basis=ftconfig.basis,           # type: ignore[arg-type]
        physical_error_rate=ftconfig.physical_error_rate,
    )

    # 2. Sample detectors + observables
    sampler = stim_circuit.compile_detector_sampler(seed=seed)
    t_s = time.time()
    detection_events, observable_flips = sampler.sample(
        shots=shots, separate_observables=True,
    )
    sample_wall = time.time() - t_s

    # 3. Decode
    decoder.from_circuit(stim_circuit)
    decode_result = decoder.decode_batch(detection_events, observable_flips)

    total_wall = time.time() - t0

    # 4. Resource estimate (informational; embedded in manifest)
    from qmesh.ir.module import Module as _Module
    empty_module = _Module()  # memory experiment has no logical T gates
    re = estimate(
        empty_module,
        code=code,
        physical_error_rate=ftconfig.physical_error_rate,
        target_logical_error_rate=ftconfig.target_logical_error,
    )

    # 5. Build the FT manifest
    manifest = _build_manifest(
        ftconfig=ftconfig,
        code=code,
        decoder=decoder,
        decode_result=decode_result,
        sample_wall=sample_wall,
        total_wall=total_wall,
        resource_estimate=re,
        stim_circuit_repr=str(stim_circuit)[:1024],
        chain_to=chain_to,
    )
    if sign:
        manifest = ManifestSigner().sign(manifest)

    out_dir = Path(ledger_dir) / manifest.submitted_at[:10]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{manifest.hash()[:16]}.json"
    manifest_path.write_text(manifest.to_json())

    md = code.metadata()
    return (
        FTResult(
            logical_error_rate=decode_result.logical_error_rate,
            logical_error_count=decode_result.logical_error_count,
            shots=shots,
            rounds=md.rounds,
            distance=md.distance,
            code_name=md.name,
            decoder_name=decoder.name,
            wall_seconds=total_wall,
            decode_seconds=decode_result.wall_seconds,
            sample_seconds=sample_wall,
            manifest_path=manifest_path,
        ),
        manifest,
    )


def _decoder_audit_block(decoder) -> dict:
    """Build the decoder_audit block: identity + (when applicable) a sha256
    of the persisted weights so 'which model produced which logical error
    rate' is auditable from the manifest alone."""
    out = dict(decoder.identity())
    # NeuralDecoder carries an optional weights_path; if it's set & exists,
    # hash the file and embed it. We import lazily to keep this file free of
    # an import cycle on the decoders subpackage at module-load time.
    weights_path = getattr(decoder, "weights_path", None)
    if weights_path:
        wp = Path(weights_path)
        if wp.exists() and wp.is_file():
            h = sha256()
            with wp.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 16), b""):
                    h.update(chunk)
            out["weights_sha256"] = h.hexdigest()
            out["weights_size_bytes"] = wp.stat().st_size
        else:
            out["weights_sha256"] = None
            out["weights_note"] = (
                f"weights_path set but file unreadable: {weights_path!r}"
            )
    return out


def _cultivation_audit_block(
    cultivation,
    *,
    n_T_states: int,
    chain_to: Path | str | None,
) -> dict | None:
    """Wrap the standard cultivation manifest_block with audit metadata —
    a signing timestamp and a cumulative_ft_chain slot pointing at a prior
    FT manifest's hash, so a campaign of FT runs forms a chain."""
    if cultivation is None:
        return None
    base = cultivation.manifest_block(n_T_states=n_T_states)
    prev_hash: str | None = None
    chain_to_path: str | None = None
    if chain_to is not None:
        p = Path(chain_to)
        chain_to_path = str(p)
        if p.exists():
            prior = json.loads(p.read_text())
            d = {k: v for k, v in prior.items() if k != "signature"}
            prev_hash = sha256(_canonical_json(d)).hexdigest()
        else:
            prev_hash = None
    return {
        "audit_signed_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "cumulative_ft_chain": prev_hash,
        "chain_to_path": chain_to_path,
        "snapshot": base,
    }


def _build_manifest(
    *,
    ftconfig: "FTConfig",
    code,
    decoder,
    decode_result,
    sample_wall: float,
    total_wall: float,
    resource_estimate,
    stim_circuit_repr: str,
    chain_to: Path | str | None = None,
) -> Manifest:
    md = code.metadata()
    cultivation = ftconfig.cultivation

    backend = {
        "name": "qmesh.ftmode.stim",
        "vendor": "qmesh+stim",
        "is_simulator": True,
        "physical_error_rate": ftconfig.physical_error_rate,
    }

    manifest = Manifest.new(
        qmesh_version=__version__,
        ir_hash="ft-experiment-" + md.name,
        ir_path=None,
        frontend={"name": "qmesh.ftmode.runner", "version": __version__},
        backend=backend,
    )
    manifest.ftmode = {
        "code": md.to_dict(),
        "decoder": decoder.identity(),
        "decoder_audit": _decoder_audit_block(decoder),
        "cultivation": (
            cultivation.manifest_block(n_T_states=resource_estimate.T_states_required)
            if cultivation is not None else None
        ),
        "cultivation_audit": _cultivation_audit_block(
            cultivation,
            n_T_states=resource_estimate.T_states_required,
            chain_to=chain_to,
        ),
        "target_logical_error_rate": ftconfig.target_logical_error,
        "physical_error_rate": ftconfig.physical_error_rate,
        "basis": ftconfig.basis,
        "stim_circuit_repr_preview": stim_circuit_repr,
        "resource_estimate": resource_estimate.to_dict(),
    }
    manifest.execution = {
        "shots": decode_result.shots,
        "logical_error_count": decode_result.logical_error_count,
        "logical_error_rate": decode_result.logical_error_rate,
        "wall_seconds": total_wall,
        "sample_seconds": sample_wall,
        "decode_seconds": decode_result.wall_seconds,
    }
    return manifest


def threshold_sweep(
    *,
    ftconfig_template: "FTConfig",
    distances: list[int] | None = None,
    physical_error_rates: list[float] | None = None,
    shots: int = 10_000,
    ledger_dir: Path | str = "ledger/ft",
    seed: int | None = None,
) -> list[FTResult]:
    """Sweep over (distance, physical_error_rate) and return results.

    Useful for empirical threshold extraction and Λ-curve plotting. Pass
    `seed` for reproducible sampling — each (d, p) pair gets a derived
    deterministic seed so the sweep as a whole is reproducible.
    """
    distances = distances or [3, 5, 7]
    physical_error_rates = physical_error_rates or [1e-3, 3e-3, 1e-2]

    results: list[FTResult] = []
    for di, d in enumerate(distances):
        for pi, p in enumerate(physical_error_rates):
            cfg = _replace(ftconfig_template, distance=d, physical_error_rate=p)
            sub_seed = (
                None if seed is None
                else seed * 10_000 + di * 100 + pi
            )
            r, _ = memory_experiment(
                ftconfig=cfg, shots=shots, ledger_dir=ledger_dir, seed=sub_seed,
            )
            results.append(r)
    return results


def promote_and_run(
    module,                              # qmesh.ir.Module
    *,
    ftconfig: "FTConfig",
    shots: int = 10_000,
    ledger_dir: Path | str = "ledger/ft",
    sign: bool = True,
    seed: int | None = None,
    chain_to: Path | str | None = None,
):
    """Promote a logical IR Module to FT mode and run.

    Phase 2β α-slice supports two execution paths:

      * Empty module — equivalent to ``memory_experiment(ftconfig)``.
      * 1-2 logical-qubit module containing gates from
        ``{h, s, x, z, cx, measure}`` — lowered to a Stim lattice-surgery
        circuit via :mod:`qmesh.ftmode.lattice_surgery` and executed with
        the same sample-and-decode loop as ``memory_experiment``. The
        resulting manifest carries an ``ftmode.lattice_surgery`` block in
        addition to the standard code/decoder/cultivation/RE block.
    """
    from qmesh.ir.module import Module as _Module
    if not isinstance(module, _Module):
        raise TypeError("module must be a qmesh.ir.Module")

    has_ops = any(f.body.ops for f in module.functions)
    if not has_ops:
        return memory_experiment(
            ftconfig=ftconfig, shots=shots, ledger_dir=ledger_dir,
            sign=sign, seed=seed, chain_to=chain_to,
        )
    return _lattice_surgery_run(
        module,
        ftconfig=ftconfig,
        shots=shots,
        ledger_dir=ledger_dir,
        sign=sign,
        seed=seed,
        chain_to=chain_to,
    )


def _lattice_surgery_run(
    module,
    *,
    ftconfig: "FTConfig",
    shots: int,
    ledger_dir: Path | str,
    sign: bool,
    seed: int | None,
    chain_to: Path | str | None = None,
) -> tuple[FTResult, Manifest]:
    """Lower the IR module via lattice surgery and run sample+decode."""
    from qmesh.ftmode.lattice_surgery import lower_module
    from qmesh.ir.module import Module as _Module  # noqa: F401  (re-import for clarity)

    code = ftconfig.build_code()
    decoder = ftconfig.build_decoder()

    t0 = time.time()
    program = lower_module(
        module,
        distance=ftconfig.distance,
        rounds=ftconfig.rounds,
        physical_error_rate=ftconfig.physical_error_rate,
        basis=ftconfig.basis,
    )
    stim_circuit = program.circuit

    sampler = stim_circuit.compile_detector_sampler(seed=seed)
    t_s = time.time()
    detection_events, observable_flips = sampler.sample(
        shots=shots, separate_observables=True,
    )
    sample_wall = time.time() - t_s

    decoder.from_circuit(stim_circuit)
    decode_result = decoder.decode_batch(detection_events, observable_flips)

    total_wall = time.time() - t0

    re = estimate(
        module,
        code=code,
        physical_error_rate=ftconfig.physical_error_rate,
        target_logical_error_rate=ftconfig.target_logical_error,
    )

    manifest = _build_manifest(
        ftconfig=ftconfig,
        code=code,
        decoder=decoder,
        decode_result=decode_result,
        sample_wall=sample_wall,
        total_wall=total_wall,
        resource_estimate=re,
        stim_circuit_repr=str(stim_circuit)[:1024],
        chain_to=chain_to,
    )
    # Tag this run as a lattice-surgery execution. Preserves the standard
    # ftmode/code/decoder/cultivation/RE block; just adds the lowering
    # provenance so manifests can be filtered downstream.
    manifest.ftmode["execution_path"] = "lattice_surgery"
    manifest.ftmode["lattice_surgery"] = program.to_dict()

    if sign:
        manifest = ManifestSigner().sign(manifest)

    out_dir = Path(ledger_dir) / manifest.submitted_at[:10]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{manifest.hash()[:16]}.json"
    manifest_path.write_text(manifest.to_json())

    md = code.metadata()
    return (
        FTResult(
            logical_error_rate=decode_result.logical_error_rate,
            logical_error_count=decode_result.logical_error_count,
            shots=shots,
            rounds=md.rounds,
            distance=md.distance,
            code_name=f"lattice_surgery({md.name},logical={program.logical_qubits})",
            decoder_name=decoder.name,
            wall_seconds=total_wall,
            decode_seconds=decode_result.wall_seconds,
            sample_seconds=sample_wall,
            manifest_path=manifest_path,
        ),
        manifest,
    )


def _replace(cfg, **changes):
    """Local helper: dataclass.replace doesn't work on slotted dataclasses
    with default factory fields cleanly across all Python versions."""
    from dataclasses import fields, replace
    return replace(cfg, **changes)
