# qmesh — architecture

## 1. The IR (`qmesh.ir`)

The IR is **typed, modality-aware, and provenance-friendly**. It distinguishes
four execution modalities and gives them a shared module/function/region/op
hierarchy:

```
Module
└─ Function (typed, may be parameterised)
   └─ Region (sequence of Ops, possibly nested for control flow)
      ├─ GateOp           # discrete unitary, optionally controlled
      ├─ MeasureOp        # mid-circuit or terminal
      ├─ ResetOp / DelayOp / BarrierOp
      ├─ PulseOp          # pulse-level (amp, phase, duration, channel)
      ├─ RydbergOp        # global laser, blockade window, atom-set
      ├─ CVOp             # Squeeze, Displace, BS, MeasureHomodyne ...
      ├─ ChannelOp        # cross-modality entanglement carrier
      ├─ ClassicalOp      # arithmetic, branching, loops
      └─ QECOp            # syndrome extract, decoder hand-off, code switch
```

Types: `qubit`, `qumode` (CV), `atom` (Rydberg site), `bit`, `int`, `float`,
`channel<src_modality, dst_modality>`. Region attributes carry intent metadata
(e.g. `@feedforward(latency_budget=200ns)`, `@logical(code=BB(144,12,12))`).

Key design decisions:
- **MLIR-shaped, but Pythonic to author.** Inspired by the MLIR-Quantum dialect
  and Catalyst's `Quantum` dialect, but the user-facing builder is a plain Python
  DSL. We can lower to MLIR if/when the QIR-2.0 + MLIR-Quantum standard hardens.
- **No leaky abstractions.** Modalities are first-class — a `RydbergOp` is not a
  fake `GateOp`. Compilers refuse to emit cross-modality ops to a backend that
  doesn't support that channel.
- **Manifest-friendly.** Every Op carries a stable hash; the IR serialiser is
  deterministic so two semantically equal modules hash to the same value
  (canonicalised: ordered SSA value renaming, stable region serialisation).

Serialisation: a **CBOR** binary form (compact, deterministic) and a JSON
human-readable form. Source IR is the binary form; manifests embed the hash.

## 2. Frontends (`qmesh.frontends`)

Each frontend is a one-way translator from a vendor SDK or source language to
qmesh.ir:

| Frontend                    | Source                  | Modalities         | Status |
|-----------------------------|--------------------------|--------------------|--------|
| `qmesh.frontends.qiskit`    | `QuantumCircuit`        | gate, dynamic      | ✅ shipped (Phase 1) |
| `qmesh.frontends.cirq`      | `cirq.Circuit`          | gate               | ✅ shipped (Phase 1) |
| `qmesh.frontends.qasm3`     | OpenQASM 3.0 source     | gate, dynamic      | ✅ shipped (Phase 1) |
| `qmesh.frontends.pulser`    | `pulser.Sequence`       | rydberg-analog     | ✅ shipped (Phase 3α) |
| `qmesh.frontends.sf`        | Strawberry Fields prog. | photonic-CV        | ✅ shipped (Phase 3α) |
| `qmesh.frontends.pennylane` | `qml.tape.QuantumTape`  | gate               | 🔜 Phase 3β |
| `qmesh.frontends.bloqade`   | Bloqade program (QuEra) | rydberg-analog     | ✅ shipped (Phase 3β α; shim until Bloqade hits PyPI) |
| `qmesh.frontends.mrmustard` | Xanadu MrMustard        | photonic-CV        | ✅ shipped (Phase 3β α; SF in maintenance mode) |
| `qmesh.frontends.qir`       | QIR LLVM bitcode        | gate (limited)     | 🔜 Phase 3β |

A frontend's job is *only* translation. Optimisation happens later in
`qmesh.compiler`.

## 3. Backends (`qmesh.backends`)

Each backend declares **capabilities** (`Capabilities` object) that the router
uses for matching:

```python
class Capabilities:
    modality: Set[Modality]              # {GATE}, {RYDBERG}, {CV}, ...
    native_gates: Set[GateName]
    qubit_count: int
    connectivity: ConnectivityGraph
    measurement_feedforward: bool        # MCM → conditional gate
    feedforward_latency_ns: Optional[int]
    classical_control: ControlLevel      # NONE | IF | LOOP | FULL
    pulse_access: bool
    cost_per_shot_usd: Optional[float]
    queue_depth: Optional[int]
    fidelity_2q_typical: float
    calibration_url: Optional[str]       # for live ingestion
```

Backends:

- `ibm` — Qiskit Runtime (Heron, Flamingo, Loon when public). QRMI for vendor-
  neutral submission.
- `ionq` — direct + via Braket (for photonic-interconnect-aware routing post-2026).
- `quantinuum` — pytket-quantinuum + qnexus.
- `quera` — Bloqade native, neutral-atom analog + Gemini gate-mode.
- `pasqal` — Pulser native, Orion-series.
- `xanadu` — Strawberry Fields, Aurora photonic-CV.
- `rigetti` — Quil/pyquil.
- `oqc` — Lucy/Toshiko via OpenQASM 3.
- `iqm` — Crystal/Halocene (5-logical-qubit features end-2026).
- `braket` — multi-vendor wrapper; dynamic-circuit dialect for IQM Garnet.
- `cudaq_sim` — CUDA-Q simulators (`nvidia`, `nvidia-mgpu`, `tensornet`,
  `density-matrix`).
- `aer_sim` — Qiskit Aer (state-vector, MPS, density, calibration ingest).
- `stim_sim` — Stim Clifford (used for QEC paths and verification).

## 4. Compiler (`qmesh.compiler`)

Pass pipeline operating on `qmesh.ir`:

```
canonicalise → modality_split → constant_fold → gate_decompose →
ftmode_promote (optional) → routing → mitigation_orchestrate (optional) →
final_layout → backend_lower
```

Canonical pass library wraps mature tools as **passes**, not as monoliths:

- `tket_pass(name, **kwargs)` — any TKET pass (CliffordSimp, FullPeepholeOptimise,
  RoutingPass, ZXGraphlikeOptimisation, …) [docs.quantinuum.com].
- `bqskit_pass(synthesis_strategy)` — BQSKit unitary resynthesis [bqskit.lbl.gov].
- `mqt_pass(name)` — Munich Quantum Toolkit pass (DD-based, ZX, predictor)
  [mqt.readthedocs.io].
- `rl_pass(model_id)` — RL transpiler models (SABRE-RL, AlphaTensor-Quantum-T,
  ZX-RL) loaded from local cache or HF Hub. Behind feature flags until benchmarked.
- `cuda_q_lower()` — emit a CUDA-Q kernel for hybrid execution.

## 5. FT mode (`qmesh.ftmode`)

A **single flag** on submission promotes physical → logical:

```python
qmesh.submit(circuit, backend="ibm:kookaburra", ft=FTConfig(
    code="auto",                    # → BB(144,12,12) on Kookaburra; surface elsewhere
    decoder="alphaqubit2",          # | "pymatching" | "stim+belief_propagation"
    cultivation=Cultivation.AUTO,   # cultivation pools, T-state target rate
    target_logical_error=1e-9,
))
```

Picks per backend:
- IBM Kookaburra → BB qLDPC.
- Google Willow / Heron defaults → surface code.
- Quantinuum Helios → trapped-ion-friendly Floquet variant when available.
- Atom Computing Magne / QuEra Gemini → neutral-atom code overlays.

Resource estimation (logical qubits, T-states, cycles, wall time, $) emitted as
part of the submission preview and embedded in the manifest. Microsoft Resource
Estimator interop via QIR export when applicable.

## 6. Mitigation (`qmesh.mitigation`)

Wraps Mitiq 1.0 (ZNE, PEC, DDD, LRE, CDR, REM, PT) and adds an
**auto-orchestrator**: given (circuit, backend, budget), picks a Pareto-optimal
mitigation stack across `bias`, `variance`, and `$`. Approach:

1. Profile circuit (depth, 2Q-gate count, T-count, structure flags).
2. Pull live calibration from backend (T1/T2, readout error, gate fidelities).
3. Score candidate stacks via a learned cost model (offline trained on Stim +
   Aer noisy traces; cheap to query).
4. Emit chosen stack into manifest; results carry uncertainty interval.

Q-CTRL Fire Opal is closed-source and best-in-class today — qmesh is the
**open-source alternative aimed at parity**.

## 7. Router (`qmesh.router`)

Submission contract:

```python
qmesh.submit(circuit, objective=Objective(
    mode="max_fidelity_under_budget",   # | "min_cost_above_fidelity"
    budget_usd=10.0,
    min_fidelity=0.90,
    max_wall_seconds=600,
))
```

Router scoring per backend:

```
score = w_fid · estimated_fidelity
      − w_cost · estimated_cost_usd / budget_usd
      − w_queue · estimated_wait_seconds / max_wall_seconds
      − w_latency · feedforward_latency_penalty (if MCM in circuit)
      + w_bonus · native_gate_match
```

`estimated_fidelity` = transpiler-output 2Q gate count × (1 − typical_2q_err) +
mitigation lift. Live calibration ingestion via the backend's `calibration_url`.
Falls back to vendor-published numbers when live unavailable.

Algorithmic basis: MQT Predictor (top-3 device choice in 98% of cases) +
Ariadne (academic) + custom learned cost model. We productize what the academic
community already proved feasible.

## 8. Scheduler (`qmesh.scheduler`)

Hybrid DAG executor. Nodes are one of:

- `ClassicalTask` — Python callable, runs on the orchestrator host or HPC node.
- `QPUPrimitive` — a qmesh.ir submission to a backend.
- `MCMRegion` — a sub-graph that runs as a single shot with mid-circuit
  measurement and live classical conditioning. Pinned to one backend for the
  feedforward-latency budget to hold.
- `Barrier` — sync point; can be entanglement-aware (waits for all entanglement
  channels to terminate before proceeding).

Persisted via append-only event log (replayable). Backend integration: Braket
Hybrid Jobs as one executor profile, Qiskit Runtime sessions as another, raw
SLURM as a third. CUDA-Q kernels are the portable middle IR for the QPUPrimitive
node (lowers to `nvidia-mgpu` for sim, to vendor backends for hardware).

## 9. Provenance (`qmesh.provenance`)

The manifest is the central artifact:

```json
{
  "qmesh_version": "0.1.0",
  "submitted_at": "2026-05-01T22:14:03Z",
  "submitter": { "id": "ed25519:abcd...", "host": "..." },
  "circuit": {
    "ir_hash_sha256": "...",
    "ir_cbor_path": "ledger/2026/05/01/...cbor"
  },
  "frontend": { "name": "qiskit", "version": "2.2.0" },
  "compiler": {
    "passes": [
      { "name": "tket.FullPeepholeOptimise", "version": "1.36" },
      { "name": "qmesh.routing.SABRE_RL", "version": "0.1.0", "model_hash": "..." }
    ],
    "input_hash": "...",
    "output_hash": "..."
  },
  "ftmode": {
    "code": "BB[[144,12,12]]",
    "decoder": "alphaqubit2-d11",
    "cultivation": { "method": "in_place", "T_states_used": 142 }
  },
  "mitigation": {
    "stack": [
      { "name": "ZNE", "scale_factors": [1, 3, 5], "extrapolator": "Richardson" },
      { "name": "REM",  "calibration_run_id": "..." }
    ]
  },
  "backend": {
    "vendor": "ibm",
    "device": "kookaburra",
    "calibration_snapshot_hash": "...",
    "queue_position_at_submit": 3
  },
  "execution": {
    "shots": 4096,
    "wall_seconds": 287.4,
    "qpu_seconds": 11.2,
    "cost_usd": 4.31,
    "raw_counts_path": "ledger/2026/05/01/...counts.cbor",
    "post_processed_path": "ledger/2026/05/01/...post.json"
  },
  "signature": { "alg": "ed25519", "value": "..." }
}
```

Replay: `qmesh replay <manifest>` re-emits the IR, applies the recorded
compiler chain, attempts to reproduce on the same backend (or warns if device
calibration has drifted). Diff: `qmesh diff <m1> <m2>` shows which step
diverged. Metriq export: one command writes a Metriq-schema-validated row.

## 10. AI (`qmesh.ai`)

- **Copilot.** Wraps Qiskit Code Assistant (granite-8b-qiskit) and any local
  Llama / Granite / Qwen Coder; routes prompts via a thin adapter so users can
  swap models. Exposes `qmesh ai draft "VQE on H2 with Hartree-Fock init"`,
  with simulator-feedback iteration (QCoder-style).
- **Intent compiler.** Classiq-inspired: high-level functional spec → IR.
  Open-source; rule-based first, model-augmented later.
- **RL hooks.** Unified loader for SABRE-RL, AlphaTensor-Quantum-T, ZX-RL
  models — they all become `qmesh.compiler.rl_pass(...)`.
- **Open neural decoder pipeline.** Generate Stim syndrome data → train a
  PyTorch decoder → sign weights → slot into FT mode. Bridges users without
  AlphaQubit-2 access.

## 11. Deployment

Two deployment shapes from day 1:

- **Library mode.** `pip install qmesh`; runs on the user's workstation; talks
  to vendor cloud APIs directly; ledger on local disk.
- **Service mode.** `docker compose up`; Postgres + object store + Redis;
  exposes a REST/gRPC API + a small web UI for the ledger; runs the LLM copilot
  via local Ollama or external Anthropic/OpenAI key. Optional SLURM/PBS
  connector for HPC-coupled installs (RIKEN/NERSC/ORNL/JSC pattern).

This repo includes a single-file `docker-compose.yml` for the service mode
preview.

## 12. Non-goals (deliberately)

- **We are not building a new vendor SDK.** Frontends import existing SDK
  objects; backends call existing SDKs.
- **We are not reinventing TKET / BQSKit / Mitiq / Stim.** We compose them.
- **We are not building a new simulator.** CUDA-Q + Aer + Stim cover it.
- **No proprietary closed-source modules.** Apache 2.0 throughout. Closed-source
  decoders / mitigators can plug in via interfaces, but qmesh's defaults are open.
