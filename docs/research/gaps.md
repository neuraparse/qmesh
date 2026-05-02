# Cross-Platform / Meta Quantum Frameworks — Gap Analysis (May 2026)

Source: parallel research agent run, 2026-05-01.

## 1. The Landscape

**Pytket / TKET (Quantinuum)** — Mature cross-vendor optimizing compiler with
extension packages (pytket-qiskit, pytket-cirq, pytket-pennylane,
pytket-quantinuum, pytket-braket, pytket-iqm, pytket-aqt). Strongest on gate-set
rebasing, routing, and noise-aware passes for superconducting + trapped-ion.
Doesn't natively cover photonic CV, neutral-atom analog/global-pulse, or FT
logical layouts.

**BQSKit (LBL)** — Gate-set-agnostic *synthesis* compiler ("any gate set to
any gate set"), strong for unitary resynthesis and circuit resizing. Portable
but execution depends on others' runtimes; not a runtime, scheduler, or device
manager.

**Munich Quantum Toolkit (MQT)** — Academic suite spanning simulation (DDs,
ZX), compilation, equivalence checking, QECC, physical design, and **MQT
Predictor** for ML-based device selection (top-3 device choice in 98% of
cases). Research-quality; not productionized.

**Qiskit** — Genuinely open-source/backend-agnostic, but practical reach skews
IBM. QRMI (qiskit-community) is a new vendor-agnostic resource interface, but
BackendV1→V2 inconsistencies and IBM Functions tilt the gravity well.

**PennyLane** — Strong device-plugin model: Lightning (CPU/GPU/AMD/Kokkos/
tensor), pennylane-qiskit, IonQ, AQT, Braket, Cirq. Differentiable-programming
first; not optimal as a heavy compiler.

**NVIDIA CUDA-Q** — As of v0.14 (Mar 2026) hits ~75% of public QPUs: IonQ,
Quantinuum, Anyon, IQM, OQC, QCI, TII (via Qibo), Infleqtion, Pasqal (via QRMI),
QuEra, ORCA (photonic), plus Scaleway QaaS, NVQLink/HPE. GPU-simulation lead
is unmatched; weaker on pure-vendor compiler optimization vs TKET/BQSKit.

**Mitiq** — Cross-frontend (Cirq core; Qiskit 2.0, PennyLane 0.43, Braket,
OpenQASM 3 frontends). Techniques: ZNE, PEC, DDD, LRE, CDR, REM, PT
(+ experimental PEA, Shadows, VD, TREX). Not a pipeline manager.

**IRs**: OpenQASM 3 is the de facto source IR (control flow, but most runtimes
only support `if`). QIR (LLVM-based) advancing; Zapata's QIR patent granted in
CA/EU/IL/AU Feb 2026. Adoption uneven — TUM 2024 paper showed feasibility, but
most vendors still ship custom dialects.

**Strangeworks** — Ops/billing/portfolio layer (40+ partners: IBM, Rigetti,
Quantinuum, Hitachi, NEC, Toshiba, ColdQuanta). Abstracts auth/billing, not
compilation.

**Classiq** — High-level functional model → synthesized circuits;
hardware-agnostic backend list. Closed-source synthesis IP; doesn't expose the
IR.

**Quantinuum Nexus** — Project workspace + execution for Quantinuum stack and
partner backends; not truly neutral.

**Quantum Brilliance Qristal** — C++/Python full-stack with MPI + CUDA; aimed
at HPC + room-temp NV-diamond accelerator.

**qBraid** — Graph-based transpiler covering 18+ SDKs / 34+ devices (cirq,
qiskit, pennylane, pyquil, pytket, braket, openqasm3, pyqir, cudaq, qibo,
stim, pulser, autoqasm); centralized auth + notebook env. Closest thing to a
true meta-layer.

## 2. Operator Pain Points (2026)

- **Vendor gravity**, not formal lock-in: Qiskit defaults assume IBM Runtime
  semantics; Nexus optimizes for H-series; Braket's UX is best on AWS-native
  services.
- **No unified abstraction across modalities**: photonic CV (ORCA/Xanadu),
  analog neutral-atom (QuEra/Pasqal), digital gate-based, and measurement-based
  each need their own front-end. Pulser ≠ Strawberry Fields ≠ OpenQASM3.
- **Dynamic circuits / classical feedforward** are non-portable. OpenQASM 3
  specs `if/while/for`, but Qiskit Runtime still only honors `if_test`;
  Braket exposes IQM Garnet dynamic circuits via its own dialect.
- **Error-mitigation pipelines are bespoke**. Mitiq covers techniques; nobody
  automates "given this circuit + this device, what stack of ZNE/PEC/REM/DDD
  optimizes bias/variance/$ trade-off?"
- **No "best backend" router that's a product**. MQT Predictor + Ariadne are
  research; IBM only ranks within IBM. Hardware-Agnostic Backend Adaptation
  (EPJ QUICK 2026) is academic.
- **Reproducibility is broken**. Metriq (Unitary Foundation) is the first
  schema-validated benchmarking platform, explicitly because results are
  "scattered across papers and press releases, rarely reproduced".
- **Hybrid scheduling is immature**. Qurator (arXiv 2604.05505) and
  "Three ways to share a QPU" (2604.14955) are *April 2026 preprints* — no
  production scheduler is cost-aware across providers.
- **FT-mode provenance**: no standard for capturing logical-qubit decoder
  versions, syndrome streams, magic-state factories used. IBM's Kookaburra
  qLDPC, IonQ's Walking Cat, neutral-atom Nature 2025 (448-atom) all ship
  with private tooling.

## 3. Classes Not Well-Served

- **Mixed-modality circuits** (photonic interconnect + atomic memory +
  superconducting compute) — no IR captures CV gates, Rydberg blockade, and
  CX in one program.
- **Adaptive / streaming algorithms** with mid-circuit measurement loops
  driven by external classical state.
- **Multi-tenant fractional QPU usage** (QPU "share-a-shot" scheduling).
- **Cost+fidelity Pareto routing** across providers as a paid SaaS contract.
- **Audit-grade FT execution records** (decoder graph, syndrome history,
  calibration snapshot, mitigation chain) — required for regulated industries.
- **Cross-vendor pulse-level access** (OpenPulse died; Quantinuum, IQM,
  Rigetti each have private pulse APIs).
- **Resource-estimation that compares architectures honestly** (Azure RE is
  single-architecture; Q-pragma/MQT each estimate one model).

## 4. Who's Working on Each Gap

| Gap | Movers | Status |
|---|---|---|
| Modality unification | Academia (TUM/MQT, FRQC paper Springer 2026), Pasqal (Pulser), Xanadu (SF) | Slow; no consortium |
| Dynamic-circuit IR | OpenQASM TSC, IBM, AWS Braket | Spec ahead of runtimes |
| EM autopilot | Unitary Foundation (Mitiq), Q-CTRL Fire Opal (closed) | Closed > open |
| Backend router | MQT Predictor, Ariadne, qBraid | Research, no SLA product |
| Reproducibility | Unitary Foundation (Metriq) | Early; needs vendor buy-in |
| Hybrid scheduler | Qurator (arXiv Apr'26), HPE/NVQLink, Munich MQSS | Preprints only |
| FT provenance | IBM, IonQ, Riverlane (decoders) | Each siloed |
| QIR adoption | Microsoft, Zapata (patent), TUM, qBraid | Patent risk dampens uptake |

## 5. Seven Unfilled Niches (Ranked by Impact × Tractability)

1. **Cost+fidelity backend router as a paid API.** Submit OpenQASM3/QIR + an
   objective; it picks vendor, transpiles via TKET/BQSKit, runs, returns
   results + receipt. MQT Predictor is the algorithm; nobody runs it as SLA
   SaaS. Small team, high leverage. *Differentiator*: live calibration
   ingestion + multi-vendor billing.

2. **Reproducibility-as-a-runtime.** Wraps any cross-vendor execution and
   emits a signed manifest. Plug-in to Metriq. Regulated buyers
   (pharma/finance) will pay; nobody else is shipping audit-grade.

3. **Auto-EM compiler.** Input: circuit + device + budget. Output: chosen
   mitigation stack with bias/variance/$ Pareto. Mitiq has primitives, no
   orchestrator; Q-CTRL is closed.

4. **Dynamic-circuit portability shim.** Single OpenQASM3-control-flow
   dialect compiled down to whatever each runtime actually supports.

5. **Mixed-modality IR for "QPU + photonic link + atomic memory."** A
   type-checked IR that distinguishes CV modes, Rydberg-blockade ops, and
   discrete gates, with explicit cross-modality channels. Higher risk, but
   no incumbent.

6. **FT-mode provenance + decoder swap layer.** Standard format for syndrome
   streams + decoder identity. Lets buyers swap Riverlane vs. open decoders,
   audit choices, replay runs.

7. **Cross-provider hybrid scheduler with quantum-aware DAGs.** Productize
   Qurator: entanglement-aware barriers, circuit cutting/merging, queue-time
   vs fidelity SLA.

**Recommended for a small team**: combine #1 + #2 + #3 — a single "submit,
route, mitigate, sign" runtime is the smallest product that operators would
replace existing glue code with. **qmesh's Phase-1 MVP targets exactly this
combination, plus #5 in IR shape.**

## Sources

- [pytket API documentation](https://docs.quantinuum.com/tket/api-docs/index.html)
- [BQSKit](https://bqskit.lbl.gov/)
- [Munich Quantum Toolkit](https://mqt.readthedocs.io/)
- [MQT Handbook](https://arxiv.org/pdf/2405.17543)
- [MQT Predictor paper](https://dl.acm.org/doi/10.1145/3673241)
- [PennyLane plugins](https://pennylane.ai/plugins)
- [CUDA-Q hardware backends](https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware.html)
- [Mitiq](https://github.com/unitaryfoundation/mitiq)
- [Zapata QIR patent (Feb 2026)](https://thequantuminsider.com/2026/02/03/zapata-secures-patent-for-interoperable-quantum-software-across-key-global-markets/)
- [Quantinuum Nexus](https://www.quantinuum.com/products-solutions/nexus)
- [qBraid SDK overview](https://docs.qbraid.com/sdk/user-guide/overview)
- [QRMI (qiskit-community)](https://github.com/qiskit-community/qrmi)
- [Ariadne quantum router](https://github.com/Hmbown/ariadne)
- [Hardware-Agnostic Backend Adaptation (EPJ 2026)](https://www.epj-conferences.org/articles/epjconf/abs/2026/16/epjconf_quick2026_01009/epjconf_quick2026_01009.html)
- [Metriq (Unitary Foundation)](https://unitary.foundation/posts/2026_metriq_platform/)
- [Qurator scheduler](https://arxiv.org/abs/2604.05505)
- [Three ways to share a QPU](https://arxiv.org/abs/2604.14955)
- [IBM FTQC roadmap](https://www.ibm.com/quantum/blog/large-scale-ftqc)
- [Quantum SE Roadmap (ACM TOSEM)](https://dl.acm.org/doi/10.1145/3712002)
