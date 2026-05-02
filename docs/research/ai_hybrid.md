# AI-Assisted Quantum & Hybrid Orchestration — May 2026

Source: parallel research agent run, 2026-05-01.

## 1. LLM-assisted quantum programming

The **Qiskit Code Assistant** (IBM, granite-8b-qiskit, 8B params, fine-tuned
from Granite Code on permissively-licensed Qiskit GitHub corpus) is the
production reference, benchmarked with a custom Qiskit-HumanEval [ibm.com,
arxiv.org]. Beyond IBM:

- **Classiq** continues to push intent-level synthesis (high-level functional
  spec → optimized circuit).
- **PennyLang** (PennyLane-centric dataset) — arXiv 2503.02497.
- **Agent-Q** — fine-tuned LLMs that both draft and optimize circuits
  (arXiv 2504.11109).
- **QUASAR** — QASM generation (OpenReview).
- **QCoder Benchmark** — closes the loop via simulator feedback
  (arXiv 2510.26101).
- **QuanBench** (Oct 2025, arXiv 2510.16779) is the current standard
  cross-framework eval and shows even strong frontier LLMs still under-perform
  on multi-qubit semantics, gate-count optimization, and hardware-aware
  passes.

**State of the art in May 2026**: LLMs reliably draft routine ansätze,
VQE/QAOA scaffolding, and transpiler glue, but human review is still required
for non-trivial algorithm design and noise-aware optimization.

## 2. RL / ML for circuit optimization

- **AlphaTensor-Quantum** (DeepMind, Nature Machine Intelligence) reached
  production-grade T-count reduction, often matching or beating hand-tuned
  solutions [nature.com].
- **RL transpilers** now beat Qiskit's SABRE: practical RL synthesis/routing
  reports ~50% fewer SWAP layers vs Qiskit TokenSwapper, near-optimal Clifford
  up to 11 qubits, and routing wins on devices up to 133 qubits.
- **ZX-calculus + RL** (Quantum journal, May 2025) is now a credible second
  school for T-count and 2-qubit-depth reduction.
- **Gadget RL (GRL)** learns composite gates respecting hardware constraints
  and scales chemistry/Ising problems to ~10 qubits.
- **Neural decoders**: AlphaQubit 2 (March 2026) hits real-time decoding
  <1 µs/cycle on commercial accelerators for surface code distance-11 and
  color code distance-9 — the first real-time color-code decoder. A widely-
  cited April 2026 study shows AI decoders cutting logical errors up to 17×
  over MWPM baselines.

## 3. Quantum machine learning (QML) — honest 2026 status

Confirmed: still no clean, reproducible quantum advantage on real classical
datasets. The 2026 review consensus (Springer Discover Computing; Quantum
Frontiers blog) is that QSVMs and QNNs are limited by trainability (barren
plateaus), noise, and data-loading costs that often erase any theoretical
speedup. Notable 2026 results — exponential-memory-savings claims on movie
reviews and scRNA-seq with <60 logical qubits, and a Science Advances paper
showing ~20% accuracy gain for *quantum-informed* chaos prediction — are
largely **simulation-based** and use *classical* models *informed* by quantum
structure, not full QML on hardware.

**Bottom line: useful QML is mostly "quantum-inspired tensor-network ML"
running on GPUs.**

## 4. Quantum-for-AI

QC does not yet contribute to training or inference of mainstream classical
AI in May 2026. The real bridges are:
- **Tensor networks** running on GPUs (cuTensorNet, NVIDIA) used for both
  quantum simulation and classical inference.
- Quantum-inspired ML models that borrow MPS/PEPS structure for
  interpretability.
- Sampling acceleration research — Non-Degenerate Batched Sampling reports
  up to 10⁸× speedup for specific TN contractions (arXiv 2604.08467).
- Generative chemistry (SQMG, 2026) uses TN-simulated VQCs to sample
  molecular graphs to N=40 heavy atoms (arXiv 2604.13877).

**No deployed LLM trains any layer on a QPU.**

## 5. Hybrid orchestration

- **Qiskit Runtime** — session/batch primitives (Sampler, Estimator V2) with
  HPC plugins for SLURM/PBS so transpilation and post-processing co-locate
  with QPUs.
- **Braket Hybrid Jobs** — managed classical containers spun up alongside
  QPU access; AWS + NVIDIA shipped first-class **CUDA-Q** support so a single
  kernel distributes circuit sampling and observable evaluation across GPU/QPU
  nodes.
- **CUDA-Q** — the de facto multi-vendor hybrid kernel language; runs on
  Braket, IBM, IQM, and ORNL/Jülich GPU-quantum testbeds.
- **Quantinuum NEXUS / TKET** — NEXUS batches jobs and exploits IBM batch mode
  via pytket-qiskit; TKET-Aer remains the most mature hybrid simulator path
  for trapped-ion targets.
- **Pure GPU/QPU mesh schedulers** — no clear winner. NVIDIA DGX Quantum
  (deployed at JSC) and **Multi-Programmer**-style schedulers (ORNL, RIKEN)
  are the closest things; the field is **still pre-Kubernetes**.

**HPC + QC actually deployed**:
- **RIKEN/Fugaku ↔ Reimei** (trapped-ion) ran a full scientific workflow
  Q1 2026.
- **NERSC QCAN** with IBM and QuEra has a 2026 call open.
- **ORNL** runs Infleqtion neutral-atom + GB200 NVL72 (HPE) installed early
  2026.
- **JSC** integrated a 5-qubit semiconductor prototype into JURECA DC and
  operates the world's first HPC-deployed **NVIDIA DGX Quantum**.

## 6. Real-time mid-circuit measurement → conditional gates

Production today:
- **IBM** ships utility-scale dynamic circuits across all Heron/Eagle backends.
  Recent demonstration ran a 46-site kicked-Ising on 106 qubits with 28% fewer
  2-qubit gates.
- **Quantinuum Helios** (data sheet Jan 21, 2026) supports MCM-conditioned
  branching, qubit reuse, real-time integer/float arithmetic, boolean logic,
  and arbitrary control flow inside the shot.
- **QuEra** offers MCM on Gemini-class neutral atoms.

Latency budgets: classical feedforward on superconducting hardware is hundreds
of ns to low µs (still 10–100× a 2Q gate); trapped ions tolerate it more
easily because gates are slower.

## 7. Quantum networking & distributed quantum

Early stage, but visible progress:
- **IonQ** (April 2026) photonically interconnected two independent commercial
  trapped-ion systems with verified entanglement — **first commercial-system
  link**.
- **Cisco** unveiled a **Universal Quantum Switch** routing entangled photons
  over telecom fiber at room temperature with ≤4% fidelity loss, plus a
  metro-scale entanglement-swapping demo with **Qunnect** over deployed fiber.
- **EPB Chattanooga** is opening the first commercial quantum networking hub
  with an IonQ system.

Distributed-quantum-computing optical-link benchmarks exist (Oxford, Nature
2024), but multi-node logical algorithms remain research.

## What a 2026 framework should integrate

1. **Copilot-style circuit assistant** — wrap Qiskit Code Assistant / Granite
   + Classiq-style intent compiler; expose simulator-feedback loops à la
   QCoder so generated circuits are validated before submission.
2. **RL-based transpilation pass** — plug AlphaTensor-Quantum / ZX-RL /
   SABRE-RL as optional passes; pick per-backend via learned cost model.
3. **Hybrid job-graph executor** — a DAG over (classical task, QPU primitive,
   MCM-conditioned subgraph) with checkpointing; CUDA-Q kernels as the
   portable IR.
4. **Latency-aware backend router** — score backends on queue depth, fidelity,
   MCM-feedforward latency, native-gate match, and price.
5. **Neural-decoder slot** — first-class hook for AlphaQubit-2-style decoders.
6. **Tensor-network bridge** — cuTensorNet for offline validation and for the
   (quantum-inspired) classical-ML escape hatch.

## Sources

- [Qiskit Code Assistant blog](https://www.ibm.com/quantum/blog/qiskit-code-assistant)
- [Qiskit Code Assistant paper](https://arxiv.org/abs/2405.19495)
- [QuanBench](https://www.arxiv.org/pdf/2510.16779)
- [PennyLang](https://arxiv.org/html/2503.02497v1)
- [Agent-Q](https://arxiv.org/html/2504.11109)
- [QCoder Benchmark](https://arxiv.org/html/2510.26101)
- [QUASAR](https://openreview.net/pdf?id=fKKKtEW71h)
- [Practical RL transpiling](https://arxiv.org/abs/2405.13196)
- [AlphaTensor-Quantum (Nature MI)](https://www.nature.com/articles/s42256-025-01001-1)
- [RL + ZX-calculus (Quantum)](https://quantum-journal.org/papers/q-2025-05-28-1758/)
- [Gadget RL (Comms Physics)](https://www.nature.com/articles/s42005-025-02475-6)
- [AlphaQubit](https://www.nature.com/articles/s41586-024-08148-8)
- [AlphaQubit 2](https://arxiv.org/abs/2512.07737)
- [QML 2026 review](https://link.springer.com/article/10.1007/s10791-026-10085-1)
- [Quantum-informed chaos (Sci Adv)](https://www.science.org/doi/10.1126/sciadv.aec5049)
- [Braket + CUDA-Q hybrid](https://aws.amazon.com/blogs/quantum-computing/advancing-hybrid-quantum-computing-research-with-amazon-braket-and-nvidia-cuda-q/)
- [ORNL hybrid HPC quantum](https://docs.olcf.ornl.gov/quantum/quantum_software/hybrid_hpc.html)
- [RIKEN Fugaku-Reimei workflow](https://www.riken.jp/en/news_pubs/research_news/rr/20260106_1/index.html)
- [JSC + NVIDIA DGX Quantum](https://nvidianews.nvidia.com/news/nvidia-and-julich-supercomputing-centre-to-build-quantum-computing-lab)
- [IBM utility-scale dynamic circuits](https://www.ibm.com/quantum/blog/utility-scale-dynamic-circuits)
- [Quantinuum Helios data sheet](https://docs.quantinuum.com/systems/data_sheets/Quantinuum%20Helios%20Product%20Data%20Sheet.pdf)
- [IonQ photonic interconnect](https://www.ionq.com/news/ionq-achieves-key-photonic-interconnect-milestone-demonstrating-networked-quantum-systems-using-entanglement)
- [Cisco Universal Quantum Switch](https://newsroom.cisco.com/c/r/newsroom/en/us/a/y2026/m04/cisco-introduces-universal-quantum-switch-advancing-the-path-to-a-quantum-network.html)
