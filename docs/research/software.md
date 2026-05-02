# Quantum Software Stack Survey — May 2026

Source: parallel research agent run, 2026-05-01. All facts cited inline.

## Qiskit 2.x / Qiskit SDK
IBM's flagship has settled into the 2.x series (v2.2 current), built around
C-accelerated transpilation (`qk_transpile()` in the C API), Primitives V2 as
the only supported execution surface, and Qiskit Runtime / Qiskit Functions as
the cloud orchestration layer for IBM hardware [ibm.com]. Dynamic-circuit
support landed in earnest with mid-circuit measurement, classical feed-forward,
and `if/else/for/while` lowered through `BlockBasePadder` and dynamical-decoupling
padding passes [qiskit.qotlabs.org].

- API style: hybrid gate-builder (`QuantumCircuit`) + functional Primitives
  (`SamplerV2`, `EstimatorV2` accepting PUBs).
- Backend reach: IBM Heron / Flamingo QPUs; broad third-party providers via
  Qiskit Functions catalog.
- Simulation reach: local Aer + sparse Statevector reference primitives; heavy
  lifting offloaded to Runtime.
- Killer feature: Functions catalog (curated higher-level workflows: portfolio
  optimization, chemistry, error-mitigated estimation) and 10–20% transpiler
  speedups in 2.2 [ibm.com].
- Pain points: aggressive deprecations across 1.x→2.x broke many 2024 codebases;
  Runtime lock-in for non-trivial features; Estimator error-mitigation knobs feel
  like a black box.

## Cirq + Stim, Google Quantum AI
Cirq remains a moment-precise, hardware-aware circuit library and the only
client for Google's Quantum Engine API (Willow-class processors), while Stim
(v1.16 pre-release, April 2026) is the de-facto stabilizer/QEC simulator,
paired with the Crumble web-based editor for surface-code design and the new
Tsim extension that adds T-gates and 10⁻⁹–10⁻¹² error-rate analysis
[github.com/quantumlib].

- Backend reach: Google Willow + qsim + Quantum Virtual Machine; Stim is
  hardware-agnostic.
- Simulation reach: best-in-class for Clifford/QEC (Stim) and large noisy NISQ
  (qsim).
- Killer feature: Stim+Crumble+Tsim is the gold standard for surface-code
  research [quantumzeitgeist.com].
- Pain points: Cirq has slipped in mindshare vs. Qiskit and PennyLane; Google's
  hardware access remains gated; Stim's API is great for experts, hostile to
  newcomers.

## PennyLane (Xanadu)
PennyLane is the differentiable-QC leader; Catalyst is its MLIR/LLVM-based AOT
compiler with a stable `Quantum` MLIR dialect, now tied into Open Quantum
Design (OQD), Munich Quantum Toolkit (via `core-plugins-catalyst`), and ETRI's
FT-resource-estimation work [pennylane.ai, quantumzeitgeist.com]. Xanadu shipped
PennyLane on ORNL's Frontier exascale system in April 2026 [prnewswire.com].

- Backend reach: ~30 plugins (IBM, IonQ, AWS, Quantinuum, Pasqal, OQD, Rigetti,
  photonic).
- Simulation reach: `lightning.gpu`, `lightning.kokkos`, MPI-distributed
  lightning on Frontier.
- Killer feature: end-to-end differentiability + Catalyst JIT compiling hybrid
  Python to MLIR→QIR.
- Pain points: Catalyst error messages are MLIR-grade unfriendly; subtle
  semantic gaps between `@qjit` and pure-Python qnodes.

## Microsoft Q# / Azure Quantum Resource Estimator + QIR
The Modern QDK is now the default; Q# compiles to QIR, which is the lingua
franca of Azure Quantum's heterogeneous backends and the FT Resource Estimator
(free, no Azure account needed) [learn.microsoft.com]. Resource Estimator
accepts Q#, Qiskit, and raw QIR; QIR 2.0 is the active standardization target
and is being adopted by MQT.

- Killer feature: Resource Estimator — physical/logical qubit counts, runtime,
  T-factory layouts.
- Pain points: Q# adoption outside Microsoft is thin; Azure Quantum credit
  model frustrates academics; QIR 2.0 still in flux.

## AWS Braket SDK (2026 state)
Braket SDK is a thin device-abstraction layer; the real value lives in Hybrid
Jobs (priority-queueing, parametric compilation, custom CloudWatch metrics,
S3 result handoff) and the new Program Sets + Mitiq integration for native
error mitigation [aws.amazon.com].

- Backend reach: IonQ, IQM, Rigetti, QuEra Aquila, plus SV1/DM1/TN1 simulators.
- Killer feature: vendor-neutral + Hybrid Jobs containerized orchestration.
- Pain points: Braket's own gate library lags Qiskit/Cirq; "neutral" comes at
  the cost of feature parity.

## TKET / pytket (Quantinuum) + Nexus
TKET retains its reputation as the best general-purpose optimizing compiler
(architecture-aware routing, lattice-surgery passes, ZX-based rewriting). As of
pytket-quantinuum 0.56, hardware submission moved to the `qnexus` Python
client; Nexus is now the unified workflow/job/compile portal, updated through
Feb 2026 [docs.quantinuum.com, github.com/quantinuum].

- Killer feature: best-in-class compilation for trapped-ion all-to-all
  topologies; mid-circuit measurement + conditional gates first-class.
- Pain points: forced migration to qnexus annoyed long-time users; closed-source
  compiler internals limit reproducibility.

## OpenQASM 3.0 — adoption reality
Adoption is uneven. IBM hardware accepts a defined subset (gates, measurements,
`if/else`, `for`, `while`, `delay`, `barrier`); Quantinuum H-series accepts a
similar but distinct subset; Rigetti, IonQ accept basic 3.0 with limited
classical control [quantum.cloud.ibm.com, openqasm.com]. Real-time classical
compute remains hardware-restricted by design. CUDA-Q now has dedicated
OpenQASM-3 dynamic-circuit transpilation (arXiv 2604.11599, April 2026).

Pain point: each vendor's "supported subset" is documented separately;
portable dynamic-circuit code is still a fiction.

## NVIDIA CUDA-Q
The unified hybrid C++/Python kernel framework (formerly cuQuantum + CUDA
Quantum) is the default GPU layer for the field. Kernels written once run
across `nvidia` (single-GPU), `nvidia-mgpu` (state-vector distribution),
`tensornet` (multi-node tensor networks), and `mqpu` (one simulated QPU per
GPU); 300×+ scaling reported on multi-GPU [nvidia.github.io].

- Backend reach: IonQ, Quantinuum, IQM, OQC, Pasqal, ORNL/HPC clusters, NVIDIA
  Quantum Cloud.
- Killer feature: HPC-grade hybrid programming with C++ kernels; QCentroid +
  enterprise stacks adopting it [thequantuminsider.com].
- Pain points: still tied to NVIDIA hardware; abstraction over QPU semantics is
  leaky; learning curve for non-CUDA devs.

## MQT, BQSKit, Qibo
- **MQT** (TUM/MQSC) — most ambitious academic suite: circuit synthesis,
  equivalence checking, QECC tooling, MLIR/QIR-2.0 plugins, wired into Catalyst
  [mqt.readthedocs.io].
- **BQSKit** (LBNL) — excels at numerical instantiation/synthesis for arbitrary
  unitaries.
- **Qibo** (Milan) — full hardware-control + simulation stack popular in Europe.

## Mitiq, QuTiP, Bloqade, Pulser
- **Mitiq** (Unitary Foundation) — cross-framework error-mitigation library:
  ZNE, PEC, CDR, REM, integrated officially with Braket Program Sets.
- **QuTiP** — open-system / pulse simulation workhorse.
- **Bloqade** (QuEra) and **Pulser** (Pasqal) — neutral-atom analog programming
  libraries; Pulser sits on top of QuTiP for simulation [github.com/pasqal-io].

## What does NOT exist or is poorly served (gaps)

- **Unified IR across modalities**: QIR 2.0 + MLIR-Quantum + Catalyst dialect
  are converging on gate-based, but photonic CV, neutral-atom analog Rydberg
  programs, and pulse-level control still live in disjoint dialects with no
  common semantics. Infleqtion's DARPA-funded Multistaq is an early attempt
  [infleqtion.com], not yet a standard.
- **Real-time hybrid compute**: sub-microsecond classical feedback during a
  shot is hardware-locked per vendor; no portable language describes it.
- **AI-assisted circuit synthesis**: BQSKit + ML-guided transpilation are
  research one-offs; no productized "LLM-aware" synthesis service exists.
- **Reproducibility / provenance**: no standard manifest for "circuit +
  transpile config + calibration snapshot + mitigation parameters + shots";
  results are hard to replay months later.
- **FT-circuit-aware scheduling**: T-factory/magic-state distillation,
  lattice-surgery scheduling, code-switching are research code, not first-class
  compiler citizens.
- **Cross-vendor benchmarking** beyond random-circuit volumetric tests; no
  agreed application-level benchmark contract.
- **Photonic CV + measurement-based** programming has no peer to Qiskit/PennyLane
  in maturity outside Xanadu's own stack.

A new 2026 framework that combined a modality-agnostic IR, provenance-by-default,
FT-aware scheduling primitives, and AI-assisted synthesis would address the
largest gaps simultaneously. **This is qmesh's thesis.**

## Sources

- [Qiskit SDK v2.2 release blog (ibm.com)](https://www.ibm.com/quantum/blog/qiskit-2-2-release-summary)
- [Qiskit Runtime V2 primitives docs](https://quantum.cloud.ibm.com/docs/en/guides/v2-primitives)
- [PennyLane + Catalyst + OQD](https://pennylane.ai/blog/2025/12/open-source-quantum-computing-pennyLane-catalyst-open-quantum-design)
- [Xanadu/PennyLane on Frontier](https://www.prnewswire.com/news-releases/xanadu-and-oak-ridge-national-laboratory-push-the-boundaries-of-large-scale-quantum-programming-on-the-frontier-supercomputer-302756068.html)
- [CUDA-Q multi-GPU docs](https://nvidia.github.io/cuda-quantum/latest/using/examples/multi_gpu_workflows.html)
- [Cirq ecosystem](https://quantumai.google/cirq/build/ecosystem)
- [Stim repo](https://github.com/quantumlib/Stim)
- [Tsim QEC simulator](https://quantumzeitgeist.com/quantum-error-correction-tsim-simulator/)
- [Azure Resource Estimator concepts](https://quantum.microsoft.com/en-us/insights/education/concepts/resource-estimation)
- [Braket Hybrid Jobs](https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs.html)
- [Braket + Mitiq Program Sets](https://aws.amazon.com/blogs/quantum-computing/error-mitigation-on-amazon-braket-with-program-sets-and-mitiq/)
- [pytket docs](https://docs.quantinuum.com/tket/api-docs/)
- [OpenQASM 3 feature table (IBM)](https://quantum.cloud.ibm.com/docs/en/guides/qasm-feature-table)
- [OpenQASM 3 spec](https://openqasm.com/intro.html)
- [OpenQASM 3 → CUDA-Q paper](https://arxiv.org/abs/2604.11599)
- [MQT](https://mqt.readthedocs.io/)
- [Pulser](https://github.com/pasqal-io/pulser)
- [QuEra Bloqade](https://www.quera.com/bloqade)
- [Mitiq guide](https://mitiq.readthedocs.io/en/stable/guide/error-mitigation.html)
- [Infleqtion Multistaq / DARPA](https://infleqtion.com/infleqtion-selected-by-darpa-to-advance-next-generation-heterogeneous-quantum-software/)
