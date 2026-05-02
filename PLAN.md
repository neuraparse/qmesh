# qmesh — research-backed plan

Date: 2026-05-01. Author: bayram + Claude (research synthesis from 6 parallel agents
+ targeted web searches). All facts cited inline `[domain]`; full sources in §7.

## 1. Why now — the 2026 inflection

Two structural shifts in 2025-2026 changed what a quantum framework needs to be:

### 1.1 Logical-qubit native era
- **Google Willow** (Dec 2024, Nature): first below-threshold demonstration —
  Λ ≈ 2.14 between distance 3/5/7, **0.143% logical error per cycle at d=7**,
  ~2.4× better than physical lifetime [nature.com]. AlphaQubit 2 (March 2026,
  arXiv 2512.07737) decodes surface d=11 + color d=9 in **<1 µs/cycle** on
  commercial accelerators [arxiv.org].
- **IBM Loon** (Nov 2025) testbed for qLDPC routing; **Kookaburra** (per
  IBM's published 2029-FTQC roadmap, slated for 2026) is intended to be
  the first IBM module to encode data in the [[144,12,12]] bivariate-
  bicycle gross code with an attached Logical Processing Unit — public
  demonstration is still pending as of May 2026; Cockatoo (2027), full
  FTQC by 2029 [ibm.com/quantum, postquantum.com].
- **Quantinuum Helios** (Nov 2025) — 98 trapped-ion qubits, 99.921% 2Q fidelity;
  **94 protected logical qubits in a GHZ state** demonstrated March 2026
  [quantinuum.com, thequantuminsider.com].
- **Atom Computing + Microsoft Magne**: 50 logical from ~1,200 atoms, operational
  early 2027 [spectrum.ieee.org].
- **IonQ**: 99.99% 2Q fidelity (Oct 2025); 256-qubit prototype 2026; "Walking Cat"
  blueprint targets ~1,600 logical by 2028 — qLDPC-first not surface
  [ionq.com, postquantum.com].
- **Magic state cultivation** (Gidney & Shutty, arXiv 2409.17595) replaced
  multi-stage distillation with in-place growth: **2×10⁻⁹ logical error at 10⁻³
  noise**, falling to **4×10⁻¹¹ at 5×10⁻⁴**, ~10× qubit-round savings. Cultivation
  is now baked into resource estimates as default; classical distillation factories
  are deprecating.
- **RSA-2048 estimate**: Gidney May 2025 — **<1M noisy qubits in <1 week**
  (arXiv 2505.15917), a 20× qubit reduction vs his 2019 number. NSA CNSA 2.0
  mandates quantum-safe NSS by Jan 2027.

### 1.2 Modality-interconnect era
- **IonQ photonic interconnect** (April 14, 2026) — first commercial-system
  photonic link verified entanglement between two independent trapped-ion
  systems via a central detection hub; AFRL-funded; Milestone 2/4 of IonQ's
  roadmap [ionq.com].
- **Cisco Universal Quantum Switch** (April 23, 2026) routes entangled
  photons over telecom fiber at room temperature; ≤4% fidelity loss, 1 ns
  reconfig, <1 W [cisco.com]. Earlier in 2026, **Qunnect + Cisco** ran a
  metro-scale entanglement-swapping demo over 17.6 km of NYC fiber
  (announced Feb 18, 2026 — predates the switch unveil)
  [thequantuminsider.com].
- **PsiQuantum Omega** chiplet (Nature 2024): SPAM 99.98%, chip-to-chip qubit
  interconnect 99.72%; $1B raise Sept 2025; targets 1M physical qubits late 2020s;
  Chicago + Brisbane sites announced March 2026 [psiquantum.com].
- **Xanadu Aurora** photonic-CV; **Pasqal Orion**, **QuEra Gemini** (neutral atom);
  **IQM Halocene 150-qubit** end-2026 with 5-logical-qubit features
  [meetiqm.com].
- **CUDA-Q v0.14** (March 16, 2026) hits 75% of public QPUs across
  modalities and now ships **`cudaq-realtime`** with QEC-library real-time
  decoding on Quantinuum and deeper Aquila (neutral-atom) integration via
  NERSC's QCAN — closing earlier gaps but still not first-class for CV
  modality. Notably, `cudaq-realtime` competes directly with qmesh's
  decoder-swap niche (§2 #6) [nvidia.github.io].

### 1.2.1 Q1–Q2 2026 reality updates (added 2026-05-01)
- **QuEra + Harvard + MIT 2:1 physical-to-logical ratio** (April 20, 2026,
  Nat. Commun.) — reconfigurable neutral atoms + qLDPC; simulated
  "Teraquop" regime. Strengthens qLDPC memory story and outweighs
  Kookaburra-pending claims for the §1.1 logical-era framing.
- **Innsbruck / RWTH / FZJ / AQT measurement-free universal FT logical
  computation** (April 2026, Nat. Commun.) — Grover on 3 logical qubits
  *without* mid-circuit measurement. Implication: `MCMRegion` semantics
  in qmesh.scheduler must accommodate measurement-free logical compute as
  a peer to MCM-driven feedforward.
- **Google Quantum AI neutral-atom pivot** (March 2026) — Adam Kaufman
  joins in Boulder; superconducting Willow continues in Santa Barbara;
  Willow Early Access Program opens. Breaks the "modality monoculture
  per vendor" assumption baked into §1.2.
- **CNSA 2.0 algorithm finalization** (April 2026 update) — ML-DSA-87 +
  ML-KEM-1024 mandated exclusively per draft-jenkins-cnsa2-pkix-profile
  §4. Sharpens the §1.1 Q-day citation and the mandate timeline through
  Jan 2027 [pqcinformation.com].

The implication: a single program in 2026 will plausibly want to run a photonic
linker + a trapped-ion compute step + a neutral-atom memory + a superconducting
fast-feedback shot, with logical-qubit guarantees on each. Today no IR captures
this. No scheduler reasons across it.

### 1.3 Compiler & AI inflection
- **Qiskit Code Assistant** (granite-8b-qiskit, IBM) shipped; **PennyLang**,
  **Agent-Q**, **QUASAR**, **QCoder** academic frameworks; **QuanBench** (Oct
  2025) is the cross-framework eval [arxiv.org].
- **AlphaTensor-Quantum** (Nature MI) reaches production-grade T-count reduction;
  **RL transpilers** beat SABRE by ~50% fewer SWAPs; **ZX-calculus + RL**
  (Quantum journal May 2025) is the second-most credible school [quantum-journal.org].
- **Mitiq 1.0** (Unitary Foundation) froze its public API for ZNE, PEC, DDD,
  LRE, CDR, REM, PT [mitiq.readthedocs.io]; AWS Braket integrated Mitiq via
  Program Sets.
- **HPC + QC actually deployed**: RIKEN Fugaku ↔ Reimei (Q1 2026), NERSC QCAN
  (2026 call), ORNL + Infleqtion + GB200 NVL72, JSC NVIDIA DGX Quantum
  [riken.jp, hpcwire.com, olcf.ornl.gov, nvidianews.nvidia.com].

## 2. Gap analysis

The gap analysis (full report from research agent, [docs/research/gaps.md]) ranked
seven unfilled niches by impact × tractability:

| # | Niche | Movers in 2026 | Status |
|---|---|---|---|
| 1 | Cost+fidelity backend router as paid API | MQT Predictor, Ariadne, qBraid | Research, no SLA SaaS |
| 2 | Reproducibility-as-runtime (signed manifests) | Unitary Foundation Metriq | Early; needs vendor buy-in |
| 3 | Auto-EM compiler (mitigation pipeline orchestrator) | Mitiq + Q-CTRL Fire Opal (closed) | Closed beats open |
| 4 | Dynamic-circuit portability shim | OpenQASM TSC, IBM, AWS | Spec ahead of runtimes |
| 5 | Mixed-modality IR (gate + Rydberg + CV + pulse) | TUM/MQT, Pasqal, Xanadu | Slow; no consortium |
| 6 | FT-mode provenance + decoder swap layer | IBM, IonQ, Riverlane | Each vendor siloed |
| 7 | Cross-provider hybrid scheduler (quantum-aware DAGs) | Qurator (arXiv Apr'26), MQSS | Preprints only |

## 3. qmesh thesis

A new framework that simultaneously owns niches **#1, #2, #3, #5, #6, #7** is the
shortest path from "academic glue code" to "the layer everything else builds on."
Niche **#4** is a side-effect (the IR makes dynamic circuits portable for free).

The novel angle is the integration. Each of these has primitives shipping in 2026
research; nobody has wired them together as a single product. qmesh is built around
two non-negotiable pillars:

- **Modality-agnostic IR.** A typed IR distinguishing CV modes, Rydberg-blockade
  ops, discrete gates, and pulse-level operations, with explicit cross-modality
  channels. Compiles down to QASM3 / OpenPulse / Pulser / Bloqade /
  Strawberry Fields / QIR. (Pulls on TUM-MQT MLIR-Quantum work + Catalyst
  + Infleqtion Multistaq lessons [infleqtion.com].)
- **Provenance-by-default.** Every execution emits a signed manifest:
  `{circuit_hash, transpiler_chain, calibration_snapshot, mitigation_chain,
  decoder_identity, code_layout, cultivation_params, raw_counts, post_processed}`.
  Replay any past run from the manifest; diff results across versions.
  Metriq-compatible export.

## 4. Architecture (high level)

See `ARCHITECTURE.md` for full detail. The eight subpackages of `qmesh/`:

- **`qmesh.ir`** — modality-agnostic typed IR + serialization.
- **`qmesh.frontends`** — adapters from Qiskit / Cirq / PennyLane / OpenQASM3 /
  Pulser / Bloqade / Strawberry Fields → qmesh.ir.
- **`qmesh.backends`** — adapters from qmesh.ir → IBM / IonQ / Quantinuum / QuEra
  / Pasqal / Rigetti / Xanadu / Braket / CUDA-Q simulators.
- **`qmesh.compiler`** — transpiler passes; wraps **TKET** (best gate-based
  optimizer), **BQSKit** (best resynthesis), **MQT** passes, plus RL-transpiler
  hooks for AlphaTensor-Quantum / SABRE-RL / ZX+RL when available.
- **`qmesh.mitigation`** — Mitiq 1.0 wrapper + auto-pipeline orchestrator
  (chooses ZNE / PEC / DDD / REM / CDR per circuit/device/budget Pareto).
- **`qmesh.ftmode`** — promote a physical-qubit circuit to logical: surface,
  qLDPC ([[144,12,12]] BB), Floquet codes per backend topology; magic-state
  cultivation primitives by default; pluggable decoder slot (PyMatching / Stim
  / AlphaQubit-2-style neural / FPGA hooks). Microsoft-Resource-Estimator-
  compatible budget pass.
- **`qmesh.router`** — submit OpenQASM3 / qmesh.ir + an objective (max fidelity ≤
  $X, or min $ ≥ F fidelity), router scores each backend on live calibration,
  queue depth, MCM-feedforward latency, native-gate match, price, and routes.
- **`qmesh.scheduler`** — hybrid DAG executor (classical task / QPU primitive /
  MCM-conditioned subgraph) with checkpoints; CUDA-Q kernels as the portable
  middle IR; Braket Hybrid Jobs / Qiskit Runtime sessions / NEXUS as backends.
- **`qmesh.provenance`** — signed manifest + replay ledger + Metriq export.
- **`qmesh.ai`** — LLM copilot (wraps Granite Qiskit-Code-Assistant + Classiq-style
  intent compiler); QCoder-style simulator-feedback loop; RL transpiler glue.

## 5. Roadmap

**Phase 0 — May 2026 (now): scaffold (this commit).**
- Repo, package, IR sketch, frontend/backend interfaces, hello-world example
  running through 2 simulator backends with provenance manifests emitted.

**Phase 1 — Q3 2026: gate-based MVP.**
- IR locked for gate + dynamic-circuit subset (OpenQASM 3 Level 0).
- Frontend: Qiskit, Cirq, OpenQASM3.
- Backend: Qiskit Aer, CUDA-Q `nvidia` + `tensornet`, Stim (for Clifford), IBM
  Runtime (read-only via QRMI), IonQ via Braket, Quantinuum via pytket-extensions.
- Compiler: TKET pass-pipeline integration; one RL pass (SABRE-RL or
  AlphaTensor-Quantum) behind a feature flag.
- Mitigation: Mitiq ZNE + REM auto-pipeline; auto-EM Pareto over budget objective.
- Provenance: manifest schema v1, signing via ed25519, file-based ledger.
- CLI: `qmesh submit`, `qmesh replay`, `qmesh diff`.

**Phase 2α — DONE (2026-05-01):**
- ✅ Surface code primitives via Stim (`SurfaceCode`, `UnrotatedSurfaceCode`,
  `RepetitionCode`); BB-code metadata stub (`BBCode`).
- ✅ PyMatching MWPM decoder + sliding-window emulation.
- ✅ Neural-decoder slot (`NeuralDecoder`) — PyTorch MLP skeleton with
  pluggable weights for AlphaQubit-2-class drop-in.
- ✅ `InPlaceCultivation` factory (Gidney 2024 parameter capture).
- ✅ Microsoft-RE-shaped resource estimator with surface-code suppression
  model `p_L ≈ 0.1 (p / p_th)^((d+1)/2)`.
- ✅ `memory_experiment()` + `threshold_sweep()` runners — emit signed FT
  manifests with full code/decoder/cultivation/RE block.
- ✅ End-to-end example (`examples/ft_logical_memory.py`) verified.
- ✅ 16 tests covering codes, decoders, estimator, cultivation, runner.

**Phase 2β α — DONE (2026-05-01):**
- ✅ Lattice surgery for 1-2 logical qubits: `qmesh.ftmode.lattice_surgery`
  lowers a logical IR Module on {h, s, x, z, cx, measure} into a
  Stim circuit using rotated-surface patches + measurement-based merge
  for logical CX. `promote_and_run(module, ftconfig)` now accepts non-
  empty modules; manifest carries `execution_path: "lattice_surgery"`
  and a full lattice_surgery lowering block.
- ✅ BB qLDPC `[[144,12,12]]` gross code: `BBCode.generate_memory_circuit()`
  emits a real Stim circuit with parity-check construction
  `A = x³ + y + y²`, `B = y³ + x + x²` over `Z_l × Z_m` (l=12, m=6).
  Real logical operator computed via GF(2) Gaussian elimination on
  ker(H_X) \ rowspan(H_Z). Determinism verified at p=0.
- ✅ True streaming MWPM decoder: `qmesh.ftmode.decoders.streaming` —
  `StreamingMWPMDecoder` with per-round generator API
  (`decode_stream`), bounded buffer (window=2d, commit_radius=d).
  Selectable via `FTConfig(decoder="streaming")`.
- 4 new tests covering lattice surgery (1-q + 2-q), BB-code metadata
  + circuit, streaming-decoder agreement.
- Demo: `examples/ft_lattice_surgery.py` — d=3, p=1e-3, 4k shots,
  logical err 0.0015, manifest signed.

**Phase 2γ α — DONE (2026-05-01):**
- ✅ Warm-state incremental matching: `qmesh.ftmode.decoders.streaming`
  now builds the `pymatching.Matching` once in `from_circuit()` and
  reuses it across every `decode_stream`/`decode_batch` call (no
  per-shot rebuild). PyMatching 2.3 has no public Blossom warm-start
  hook, so true Blossom-incremental is deferred — but the
  build-once-reuse-everywhere pattern matches batch decoder predictions
  100% on a 50-round circuit.
- ✅ BP+OSD decoder: `qmesh.ftmode.decoders.bp_osd.BpOsdDecoder` —
  lazy-imports `ldpc` PyPI package; if absent ships an in-package
  fallback (min-sum BP 50-iter + OSD-1) that's correct on small
  examples. Wired into `FTConfig(decoder="bp_osd")`.
- ✅ Cultivation-T injection: `qmesh.ftmode.lattice_surgery` now
  accepts `t`/`tdg` gates in the input module, emits SHIFT_COORDS
  cultivation-reservation markers in the lowered Stim circuit, and
  populates `manifest.ftmode["lattice_surgery"]["t_injection_blocks"]`
  with `{block_index, logical_qubit, n_T_required, cultivation_factory,
  cultivation_cycles_per_T, cultivation_cycles_block, stim_marker}`.
  Resource estimator counts T cycles correctly.
- 5 new tests + demo `examples/ft_t_injection.py`: H + T + measure on
  d=3 logical qubit, 2000 shots, logical err ~5e-4, 1 cultivation
  block reserved (50 cycles), signed manifest verified.

**Phase 2δ — DONE (2026-05-02):**
- ✅ Litinski merge-CNOT extended observable: `_shift_circuit` now
  carries an `obs_offset` arg so each patch's `OBSERVABLE_INCLUDE` lands
  on a distinct index (patch i → observable i instead of all-patches→0).
  `_emit_logical_cx_merge_strip` appends an additional
  `OBSERVABLE_INCLUDE merge_observable_index` whose targets are the
  final-round merge-ancilla measurements — this is the
  Litinski merge-CNOT product observable that captures the CX
  Pauli-frame action. `LatticeSurgeryProgram` exposes
  `merge_observable_index` and `per_patch_observable_indices`. A 2-patch
  CX program now emits 3 observables (2 per-patch + 1 merge); single-
  patch programs still emit 1 (no merge).
- ✅ Cultivation circuit emission: `qmesh.ftmode.cultivation.emit_cultivation_block`
  produces a real Stim sub-circuit
  (`R / X_ERROR(target_T_error) / M / OBSERVABLE_INCLUDE`) on a
  dedicated cultivation ancilla per logical T injection. The DEM
  carries explicit per-block flip channels; the cultivation outcome is
  a real logical observable that decoders match against. The data-qubit
  CX + S correction is still β work and explicitly noted; what ships is
  a *real* Stim block, not a SHIFT_COORDS marker. `lower_module(...,
  cultivation_target_T_error=...)` plumbs the residual error rate.
- ✅ 3 new tests: merge observable count + per-patch observable
  indices, full DEM compilation with `decompose_errors=True`, real
  cultivation flip-rate < 5% sampled at target_T_error=1e-4.

**Phase 2δ — next (research-grade):**
- True Blossom-incremental matching via the C++-level pymatching API
  (current `streaming` decoder uses warm-state matching with 100% batch
  parity, which is the Python-API ceiling).
- Full data-qubit T injection: CX from cultivation ancilla into the
  patch + conditional S correction inside the lattice-surgery scheduler.

**Phase 3α — DONE (2026-05-01):**
- ✅ Rydberg-analog IR exercised: `qmesh.frontends.pulser` translates Pasqal
  Pulser sequences (Register, Pulse, channels, measurements) into RydbergOps
  with atom positions and channel-addressing attrs preserved.
- ✅ `qmesh.backends.pulser` runs the IR through QutipEmulator; Rydberg
  blockade physics verified (`|11⟩` heavily suppressed in 5µm + 2π Rabi).
- ✅ Photonic-CV IR exercised: `qmesh.frontends.sf` translates Strawberry
  Fields Programs into CVOps (Sgate, BSgate, Dgate, Rgate, MeasureFock,
  MeasureHomodyne).
- ✅ `qmesh.backends.sf.fock` + `qmesh.backends.sf.gaussian` run CV programs;
  two-mode squeezed vacuum + BS verified.
- ✅ Modality-aware router refuses cross-modality submissions (gate→pulser,
  cv→aer all rejected with clear error messages).
- ✅ Cross-modality example demonstrates 3 modalities + 3 backends + 3 signed
  manifests in one program.
- ✅ 10 tests covering frontends, backends, router behavior, modality coverage.

**Phase 3β α — DONE (2026-05-01):**
- ✅ Bloqade frontend: `qmesh.frontends.bloqade.from_bloqade(prog) → Module`
  with a `BloqadeProgram` α-shim for the (still-unreleased-on-PyPI)
  Bloqade ecosystem; real Bloqade duck-types in via `.to_segments()`.
  Atom positions, per-segment Rabi amp/detuning/phase/duration preserved.
- ✅ MrMustard frontend: `qmesh.frontends.mrmustard.from_mrmustard(prog)
  → Module` with an `MMCircuit` shim mirroring `qmesh.frontends.sf`. Real
  MrMustard duck-types through `.components`/`.ops`; differentiable
  params unwrapped via `.value`/`.numpy()`. SF is in maintenance mode —
  MrMustard is the supported successor.
- ✅ ChannelOp first-class: upgraded from placeholder in `qmesh/ir/ops.py`
  to a real op carrying `(source_modality, target_modality, kind,
  payload)`. Custom `digest()` rolls those into module hashing.
- ✅ ChannelOp lowering: `qmesh.scheduler.channelop_lowering.lower_module_to_dag`
  partitions a Module by modality, emits one QPUPrimitive per modality-
  coherent block, inserts ClassicalTask "channel" stubs for each ChannelOp
  forwarding `parity_even` / `p_excited` / `p_one` / `n_avg` derived
  payloads. Supports gate→rydberg, gate→cv, rydberg→cv kinds; reverse
  arrows + pulse modality flagged as Phase 3γ.
- 5 new tests + demo: `examples/cross_modality_channelop.py` — ONE Module
  with gate Bell + ChannelOp + Rydberg drive + ChannelOp + CV BS-readout
  → 5-node DAG (3 QPU + 2 classical bridges) → executed on 3 modalities,
  signed end-to-end.

**Phase 3γ α — DONE (2026-05-01):**
- ✅ Reverse-arrow ChannelOps: `rydberg→gate`, `cv→gate`, `cv→rydberg`
  added to `qmesh.scheduler.channelop_lowering.SUPPORTED_KINDS`. Each
  has a payload extractor (`p_excited_per_atom`, `n_avg_per_mode`,
  `homodyne_x` with `n_avg` fallback) and a parameter injector that
  patches the first matching downstream op (parametric `GateOp` /
  `RydbergOp` / `CVOp` / `PulseOp`) in either `set` or `scale` mode.
- ✅ Pulse-modality bridges: `gate→pulse` and `pulse→gate` register in
  SUPPORTED_KINDS. `Modality.PULSE` and `PulseOp` were already in IR;
  pulse-modality execution still falls back to statevec (no real
  OpenPulse backend yet — IR + bridge plumbing is what ships).
- 5 new tests + demo `examples/cross_modality_reverse.py`: rydberg
  readout → ChannelOp(rydberg→gate, p_excited=0.94 → ry-θ scale) →
  gate compute → ChannelOp(gate→cv, parity_even) → cv squeeze+Fock,
  signed aggregate manifest.

**Phase 3δ — DONE (2026-05-02):**
- ✅ OpenPulse backend skeleton: `qmesh.backends.openpulse_sim`
  registers `qmesh.openpulse` with `pulse_access=True`. Lowers any
  gate-modality module to a vendor-neutral `qmesh.openpulse.v0`
  schedule descriptor (per-event qubit assignments, start_ns,
  duration_ns, amplitude, sigma_ns, plus virtual-Z flags). When
  `qiskit.pulse` is importable (Qiskit 1.x) it additionally constructs
  a real `qiskit.pulse.Schedule`; otherwise it gracefully α-degrades
  (Qiskit 2.x dropped the pulse module). Counts come through the
  best-installed gate-level fallback (`qmesh.statevec` →
  `qmesh.aer` → `qmesh.stim`) so the backend always returns valid
  results. `RunResult.backend_metadata` records
  `schedule_descriptor`, `qiskit_pulse_available`, `execution_path`,
  and the fallback backend name.
- ✅ 3 new tests: registration + capabilities, Bell circuit produces
  4-event descriptor (h + cx + 2× measure) with positive total
  duration and valid 50/50 fallback counts, capabilities note
  truthfully reports `qiskit.pulse` availability.

**Phase 3δ — next (vendor-API gated):**
- Photonic-link middleware integration: route the bridge through the
  IonQ photonic interconnect API or Cisco UQS once those are public.
- Calibrated DRAG + cross-resonance pulse synthesis (today: single
  Gaussian envelope per gate; vendor-specific calibration is a
  hardware-bring-up item).
- Full parameter-name targeting (today: `params[0]` only).

**Phase 4α — DONE (2026-05-01):**
- ✅ Typed DAG nodes: `ClassicalTask`, `QPUPrimitive`, `Barrier`, `Fanout`,
  `MCMRegion` (passthrough today).
- ✅ Topological executor with thread-pool parallelism inside each DAG level
  (independent nodes run concurrently; deps respected).
- ✅ Append-only NDJSON event log per run + signed aggregate run manifest.
- ✅ `module_factory` lets upstream classical results parametrise downstream
  QPU programs (the missing piece for VQE/QAOA/feedback loops).
- ✅ `qmesh dag-status` CLI + `/dag_runs/{id}` REST endpoint.
- ✅ Two end-to-end examples: VQE-DAG (4 parallel observable QPU nodes +
  barrier + classical aggregator) and cross-modality DAG (gate →
  classical → Rydberg → classical → CV chain).
- ✅ 13 tests covering topology, parallelism, failure handling, manifest
  signing, event log, fanout, barrier, cross-modality.

**Phase 4β — DONE (2026-05-01):**
- ✅ Real checkpoint resume: every node's NodeResult is snapshotted to
  `{run_dir}/node_results/{node_id}.json`; `qmesh.scheduler.resume(dag,
  original_run_dir)` skips ok-results (including individual Fanout
  children via deterministic `{parent}::child_{i}` IDs), re-runs
  failures, and emits a first-class **`lineage` block** on the Manifest
  carrying `parent_run_id`, `parent_manifest_hash`, and the parent's
  ed25519 signature value (so an auditor can detect parent
  substitution). CLI: `qmesh dag-resume-info <run_dir>`. Demo:
  `examples/dag_resume.py`. 9 tests.
- ✅ Cross-modality ChannelOp auto-lowering — see Phase 3β α.
- ✅ HPC connectors: `qmesh.scheduler.hpc` — `SLURMConnector`,
  `PBSConnector`, `MockHPCConnector` (renders `sbatch`/`qsub` scripts;
  mock executor runs locally for CI). `HPCQPUPrimitive` is a Node
  subtype; manifests carry an `hpc` block with scheduler, job_id,
  partition, walltime_request/actual, nodes, cpus_per_task,
  inner_manifest_hash. CLI: `qmesh hpc-status <job_id>
  --scheduler {slurm|pbs|mock}`.
- ✅ Entanglement-aware Barrier: `EntanglementBarrier(Barrier)` +
  `BellPairClaim(ClassicalTask)`. Producers publish bell-pair claims
  into `ctx['bell_pairs'][photonic_link]`; barrier waits until
  `expected_bell_pairs` arrive or `timeout_ns` fires. Models the
  IonQ photonic interconnect / Cisco UQS flow where two QPUs
  entangle via fiber before classical-conditioned downstream work
  proceeds.
- 5 new tests covering Mock-HPC + SLURM/PBS script renders +
  Bell-pair barrier success + timeout. Demo: `examples/dag_hpc.py`.

**Phase 5α — DONE (2026-05-01):**
- ✅ Rule-based intent compiler — 6 patterns (Bell, GHZ, QFT, HEA, W-state,
  random Clifford) emit IR directly without LLM round-trip.
- ✅ LLM provider abstraction — Ollama / Anthropic / OpenAI / Mock; auto-
  selected via env vars; mock fallback always works (CI / no-key tutorials).
- ✅ Copilot with QCoder-style validation loop — generated QASM is parsed,
  simulated on `qmesh.aer`, checked against pattern-specific properties
  (Bell parity, GHZ-pattern probability, generic runnability), re-prompted
  on failure.
- ✅ Neural-decoder training pipeline — `qmesh.ai.train_neural_decoder`
  generates Stim syndrome data, trains a PyTorch MLP, persists sha256-signed
  weights, returns a `NeuralDecoder` ready to slot into FT-mode runs.
- ✅ `qmesh ai-draft` and `qmesh ai-train-decoder` CLI commands.
- ✅ 14 tests across intent compiler, mock provider, copilot pipeline,
  decoder training.

**Phase 5β α — DONE (2026-05-01):**
- ✅ Transformer-based syndrome decoder: `qmesh.ai.transformer_decoder`
  — 2-layer / 4-head / d_model=64 (~50K params) encoder over per-round
  detector tokens, mean-pool → observable head. Mirrors the MLP entry
  point (`train_transformer_decoder`); same signed-weights manifest
  format. On d=3, p=1e-3, 5K shots, 4 epochs, seed=42: val_acc ≈ 0.97.
  Loads back into the existing `NeuralDecoder` slot via
  `load_transformer_decoder()` so FT-mode runs are decoder-agnostic.
- ✅ Constrained QASM 3 decoding: `qmesh.ai.constrained_decoding` —
  line-level grammar gate covering the gate set qmesh's intent
  compiler emits. APIs: `validate_qasm`, `validate_qasm_streaming`,
  `QASMGrammarGate.legal_next_tokens(prefix)`,
  `ConstrainedQASMGenerator(provider, max_tokens)` post-hoc filter
  with safe Bell-stub fallback on grammar failure. (α uses post-hoc
  filtering since `LLMProvider.complete()` returns whole completions;
  β would integrate Outlines/llama.cpp grammar files for token-level
  masking.)
- ✅ QuanBench evaluation harness: `qmesh.ai.quanbench` —
  10-prompt built-in suite (Bell ×2, GHZ ×2, QFT, HEA, W-state,
  Random Clifford, plus 2 hard prompts forcing LLM fallback).
  `run_quanbench()` returns a versioned JSON report with
  success_rate, mean_iterations, per-pattern + per-provider
  breakdowns, wall-time stats. CLI: `qmesh ai-eval --suite quanbench
  --provider mock`. Mock-provider headline: success_rate=1.000 across
  10 prompts in ~0.14s.
- 5 new tests + demo: `examples/transformer_decoder.py` — side-by-side
  MLP vs Transformer table (val_acc 0.976 vs 0.975, sha256-signed
  weights for both).

**Phase 5γ α — DONE (2026-05-01):**
- ✅ Token-level grammar masking: `qmesh.ai.grammar_mask` ships three
  adapters wrapping `QASMGrammarGate` — `StubAdapter` (post-hoc filter,
  α-fallback), `LogitBiasAdapter` (returns `{token_id: +inf}` masks
  when `tiktoken` is installed; β-grade, α-degrades to string-keyed
  masks), `OutlinesAdapter` (compiles a 700+ char regex covering header
  + parametric/non-parametric gates; lazy-imports `outlines>=0.1`).
  Public entry `compile_grammar_mask(gate, backend=...)`.
- ✅ Novel-prompt benchmark suite: `qmesh.ai.quanbench.NOVEL_SUITE` has
  10 prompts that bypass the rule-based intent compiler (multi-controlled
  prep `{0,3,5,7}`, [[5,1,3]] syndrome, Heisenberg Trotter,
  teleportation, QFT-inverse, GHZ X-basis, phase-kickback, Haar random,
  iterative phase estimation, VQE/UCCSD). Each has a sim-checkable
  `expected_property`. Under MockProvider, success_rate = 0.0 — proving
  the suite is genuinely hard. `quanbench_combined()` runs both BUILTIN
  and NOVEL with separate breakdowns.
- 10 new tests + demo `examples/ai_grammar_mask.py`: Bell + GHZ-3 step
  through the grammar gate with rich-panel `legal_next_tokens` per
  line, summary of adapter maturity.

**Phase 5δ — DONE (2026-05-02):**
- ✅ Real-time grammar wiring: `qmesh.ai.grammar_mask.GrammarMaskedProvider`
  is itself an `LLMProvider`, wrapping any provider with per-line gate
  validation *during* generation. Two modes: **streaming** (β when the
  underlying provider honours `stop`; one provider call per emitted line,
  with a one-shot repair on rejection) and **whole-completion fallback**
  (α; full generation + line-by-line filter + a single repair completion
  with the rejected line surfaced to the model). Stop heuristic terminates
  on classical-bit-saturation (every declared `bit[N]` measured) so the
  output doesn't run away. Provider name reads
  `grammar_masked[streaming|fallback](inner)`.
- ✅ 3 new tests: clean Bell pass-through (mock provider, fallback mode),
  scripted-bad-line repair (rejects illegal `ZAPS q[0];`, retries clean
  Bell tail; metadata reports `n_repairs=1` and `rejected_at=4`),
  streaming-mode line-count check (8 calls = 8 emitted lines, stops on
  measurement saturation).

**Phase 5δ — next (research-grade):**
- AlphaQubit-2-scale model (100M+ params, 10⁶+ Stim shots,
  multi-distance pre-training).
- Wire `GrammarMaskedProvider.streaming=True` to a real
  Outlines / llama.cpp / vLLM endpoint that honours `stop=("\n",)`.
- Import the published QuanBench suite once public; integrate with
  Metriq export from Phase 6.

**Phase 6 α — DONE (2026-05-01):**
- ✅ Metriq exporter: `qmesh.metriq` translates qmesh manifests into
  Metriq-compatible Submission/Result JSON. Per-row evidence carries
  `manifest_hash`, `signature_alg`, `signature_value`, `public_key`,
  `qmesh_version`. APIs: `export_manifest`, `export_run_dir`,
  `submit_to_metriq(..., dry_run=True)` (live mode lazy-imports
  `requests`). FT manifests pull through `logical_error_rate`,
  `estimated_logical_error_rate`, `physical_qubits_required` automatically.
- ✅ Compliance pack: `qmesh.compliance.build_pack(ledger_dir, since,
  until, out_path)` writes a tar.gz containing
  `manifests/`, `signatures.json`, `chain.json`, `README.txt`. Chain
  rule (git-style): `this_hash = sha256(prev_hash || manifest_hash ||
  signature_value)`, genesis = `"GENESIS"`.
  `qmesh.compliance.verify_pack(archive)` re-walks the chain offline
  (no private key needed), re-verifies every ed25519 signature, returns
  `ComplianceVerification(chain_valid, n_manifests, n_signatures_ok,
  broken_at, errors, notes)`. Tampering with even one manifest sets
  `broken_at` and `chain_valid=False`.
- ✅ Audit-grade FT records: `manifest.ftmode` now carries
  `decoder_audit` (decoder.identity + weights_sha256 + weights_size_bytes
  for NeuralDecoder/Transformer) and `cultivation_audit` (
  `audit_signed_at` + `cumulative_ft_chain` slot for run-to-run
  chaining via the new `chain_to: Path | None` kwarg on
  `memory_experiment` / `promote_and_run`). FT campaigns are now
  hash-chained.
- CLI: `qmesh metriq-export`, `qmesh compliance-pack`,
  `qmesh compliance-verify`.
- 9 new tests in `tests/test_compliance.py` + demo
  `examples/compliance_export.py`: 3 runs (Bell, GHZ, FT memory) →
  pack → verify chain offline (chain_valid=True, 3 sigs ok) → metriq
  dry-run produces 8 Result rows, no network.

**Phase 6β — DONE (2026-05-02):**
- ✅ Cert-chain trust model: `qmesh.compliance.trust` ships
  `Certificate`, `TrustStore`, `TrustResult`, `issue_certificate`,
  `verify_certificate_signature`, `verify_chain`,
  `generate_root_keypair`, `attach_cert_chain_to_manifest`. Each cert
  is an ed25519 signature over (subject_pubkey, subject_name,
  issuer_pubkey, issuer_name, not_before, not_after, serial); chains
  walk leaf → root with cycle detection and validity-window enforcement.
  `verify_pack(..., trust_roots=TrustStore)` extends Phase 6α with
  `trust_chain_valid`, `n_trusted`, `untrusted_signers`. ManifestSigner
  accepts an optional `cert_chain=[...]` so the chain is bound into the
  body and any post-hoc cert swap invalidates the signature.
- ✅ PennyLane / qBraid SDK plugins:
  `qmesh.frontends.pennylane.from_pennylane_tape(...)` /
  `from_qnode(qnode, *args, **kwargs)` lazy-imports `pennylane`,
  maps PL ops (Hadamard, PauliX/Y/Z, S, T, RX/RY/RZ, PhaseShift,
  CNOT/CZ/SWAP/Toffoli) to canonical qmesh gates with arbitrary-wire
  labels denseified to integers. `qmesh.frontends.qbraid.from_qbraid(...)`
  routes any qBraid-supported program through `qbraid.transpiler.transpile(
  program, "qasm3")` and then through `qmesh.frontends.qasm3.parse`,
  inheriting qBraid's full SDK coverage with no per-vendor lowering.
  Both plugins α-degrade with informative `RuntimeError` when their SDK
  is missing.
- ✅ 12 new tests across `test_compliance.py` (cert issuance + tamper
  detection, one-level + two-level chain walks, cycle detection,
  validity window, pack-with-trust round-trip, untrusted-signer
  flagging, back-compat without trust store) and `test_frontends.py`
  (PennyLane / qBraid stub-import behaviour).

**Phase 6β — next (external-API gated):**
- Live Metriq HTTP submission once Unitary Foundation publishes the
  endpoint contract (current `submit_to_metriq(..., dry_run=False)`
  scaffolds requests retry/auth but is gated on dry_run=True until the
  contract lands).
- Compliance-pack signing-key rotation + revocation list (RL); cert
  validity windows are honoured but a CRL/OCSP shape is β work.

## 6. Why this can be the framework everyone builds on

- It is the only IR that crosses modalities — the only place a CV+gate+Rydberg
  program can be expressed without leaving the toolchain.
- It is the only runtime that signs every result. As regulated industries enter
  (pharma, finance, defense — driven by Q-day mandates), this becomes essential.
- It is the only compiler stack with FT-mode as a one-flag promotion, with
  cultivation-aware lattice surgery scheduling out of the box (arXiv 2512.06484).
- It composes — does not replace — TKET, BQSKit, MQT, Mitiq, Stim. Those teams
  win when qmesh wins. Adoption-friendly.
- The hybrid DAG scheduler matches the HPC-coupled deployment direction
  (RIKEN, NERSC, ORNL, JSC) — those centers need exactly this layer.

## 7. Sources

The full per-agent research reports are committed in `docs/research/`:
- `docs/research/hardware.md` — vendor survey (May 2026)
- `docs/research/software.md` — software ecosystems
- `docs/research/simulation.md` — simulators + noise
- `docs/research/qec.md` — error correction milestones
- `docs/research/gaps.md` — cross-platform gap analysis
- `docs/research/ai_hybrid.md` — AI-assisted + hybrid orchestration

Top citations: nature.com (Willow, Helios), arxiv.org (2409.17595 cultivation,
2505.15917 RSA, 2512.07737 AlphaQubit 2), ibm.com/quantum (FTQC roadmap),
quantinuum.com (Helios), ionq.com (Walking Cat), nvidia.github.io (CUDA-Q),
psiquantum.com (Omega), spectrum.ieee.org (Magne), thequantuminsider.com
(Q-day timeline), unitary.foundation (Mitiq, Metriq).
