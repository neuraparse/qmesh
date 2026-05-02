# Classical Simulation Backends for Quantum Circuits — May 2026 Survey

Source: parallel research agent run, 2026-05-01.

## 1. State-vector simulators
State-vector memory doubles per qubit (16 B per amplitude in c128). On a single
H100 (80 GB), the practical ceiling is ~32 qubits c128 / 33 qubits c64; on B200
(192 GB) ~33–34 c128. NVIDIA's GB200 NVL72 benchmarks show cuQuantum with B200
GPUs is roughly 3× faster than H100 at 40 qubits [nvidia.com]. A single DGX
H100 (8×80 GB) reaches ~35 c128; the cuQuantum Appliance with cusvaer scales
out to 40 qubits c128 / 41 c64 on ABCI-2 [docs.nvidia.com].

**Multi-node SOTA**: Qiskit Aer's full state-vector method has now hit
**44 qubits using 4096 GPUs across 1024 LUMI nodes** (Apr 2025) [fiqci.fi].
CINECA + Kipu Quantum reported a 43-qubit QAOA-style state-vector run on 2,048
Ampere GPUs (Mar 2026). Intel-QS scales to 42 qubits on 2048 SuperMUC-NG nodes
[intel-qs.readthedocs.io]; QuEST has been benchmarked to 38 qubits on 2048
nodes, with a 2025 NUMA/AVX-512 rework boosting CPU performance significantly
[arxiv.org]. Ceiling is set by aggregate RAM; **~45–46 qubits is the
contemporary frontier on top-tier supercomputers.**

## 2. Tensor networks
For low-entanglement / shallow circuits, qubit count is essentially unbounded —
cost scales with bond dimension and contraction width. cuTensorNet delivers
8–20× speedups over CuPy on A100 and is the de facto GPU contraction engine
[developer.nvidia.com].

**Pan Zhang and collaborators classically reproduced IBM's 127-qubit Eagle
kicked-Ising experiment**, then pushed to 433-qubit Osprey and 1121-qubit
Condor lattices using belief-propagation-contracted gPEPS with effective bond
dimension >16M [link.aps.org, arxiv.org/abs/2309.15642]. quimb + cotengra
remains the best open-source path optimizer; PennyLane's `default.tensor`
wraps quimb. 60-qubit RCS at depth 20+ is within reach of a single workstation
on low-entanglement instances; depth-24 Sycamore-class circuits remain the
genuine quantum-advantage frontier where simulation costs still hit days on
~300 GPUs.

## 3. Stabilizer / Clifford
**Stim 1.15.0** (May 2025) is the standard. It can analyze a distance-100
surface code (~20k qubits, 8M gates, 1M measurements) in 15 s and then sample
at ~1 kHz [zenodo.org]. It's the workhorse of every serious QEC paper —
decoder training (Sinter, PyMatching, Tesseract), threshold studies, magic-state
cost estimation. **STABSim** (Jul 2025) adds parallelism and modest non-stabilizer
extensions [arxiv.org/abs/2507.03092]. **Wrap Stim; don't reimplement.**

## 4. Density matrix / noisy
DM simulators square the memory cost, so the ceiling is roughly half the qubit
count of state-vector. Single H100 ≈16 qubits c128 DM; DGX H100 ~18; multi-node
~22. Qiskit Aer DM and Braket DM1 (max 17 qubits, $0.075/min) are the practical
ceilings. For larger noisy systems people now use trajectory MC on state-vector
or noisy MPS rather than full DM.

## 5. MPS / PEPS / isoTNS / NN states
cuQuantum now has a first-class **MPS API** with GPU contraction.
**isoPEPS** has emerged as a sweet spot — every tensor is an isometry, so the
network maps directly to a quantum circuit, and 2D simulation of Eagle-class
devices is now routine [arxiv.org/abs/2503.08626]. Belief-propagation PEPS
contraction (Tindall, Fishman, Stoudenmire, Sels) is the dominant 2D method.
**Neural-network quantum states**: a Jun 2025 paper seeds RBM wave functions
from MPS via canonical polyadic decomposition, giving polynomial-time warm
starts for ground-state VMC.

## 6. Photonic / CV
**Strawberry Fields** (Xanadu) ships Gaussian, Fock, Bosonic, and TF backends.
**MrMustard** (Xanadu) is the modern Gaussian + Fock differentiable engine,
used heavily for GBS / bosonic-code research. **Piquasso** (Apr 2025, Quantum
journal) is the new credible alternative. Perceval (Quandela) covers
linear-optical / discrete-photon paradigms. CV simulation is its own world —
separate from qubit backends.

## 7. GPU acceleration
**CUDA-Q** (formerly CUDA Quantum) is now NVIDIA's umbrella; the GTC'26
release runs CUDA-Q 0.12 with cuQuantum 25.9.1, exposes
cuStateVec/cuTensorNet/cuDensityMat, and supports MPI/NVLink/InfiniBand for
multi-node. **JAX**: Google qsim has a JAX path; PennyLane-Lightning offers
JAX-jit. **AMD/ROCm**: HIP backend for qsim is mature; BlueQubit reported the
largest single-GPU quantum simulation on AMD MI300 in 2026.

## 8. Cloud simulators
- **Braket SV1**: 34 qubits, $0.075/min; **DM1**: 17 qubits noisy; **TN1**:
  hundreds of qubits if low-entanglement.
- **IBM Quantum Platform**: simulators deprecated in 2024 — push toward
  Aer-on-Qiskit-Runtime; cloud focus now on hardware fleet (Heron r2, Flamingo).
- **Azure Quantum**: per-provider pricing, hosts Quantinuum H-Series emulator,
  IonQ simulator.

## 9. Noise models
2026 simulators model T1/T2 thermal relaxation, depolarizing, amplitude/phase
damping, readout error, and Pauli-Lindblad as standard. IBM's calibration-driven
`NoiseModel.from_backend` ingests live device data; a 2025 IBM report claims
98.7% noise-profile fidelity on 127-qubit Heron. Crosstalk and leakage still
require custom channels — 2025 Adv. Quantum Tech. work on shared-qubit
crosstalk and ML-fitted noise (43–46% fidelity gain over analytic models) is
the current frontier.

## 10. Recent quantum-advantage simulation results
- Tindall et al. and Begušić/Chan classically reproduced the 127-qubit IBM
  Eagle Trotterized-Ising experiment with belief-propagation MPS/PEPS.
- Computational Power of Random Quantum Circuits in Arbitrary Geometries (PRX,
  May 2025) sharpens the spoofing-cost curve.
- 56-qubit trapped-ion RCS (Quantinuum, Nature 2025) — first ion-trap RCS,
  classical spoof cost reportedly 9 orders of magnitude beyond Sycamore '19.
- Quantum Frontiers Jan 2026 retrospective: finite-fidelity RCS spoofing
  remains expensive but no longer infeasible for Sycamore-era circuits.

## What a 2026 framework should support

**Wrap (first-class):**
- **Stim** — non-negotiable for any QEC story.
- **CUDA-Q** (cuStateVec + cuTensorNet + cuDensityMat) — covers GPU SV, TN,
  DM in one MPI-aware stack.
- **Qiskit Aer** — the reference noisy/SV simulator, calibration ingestion,
  MPS method.
- **quimb + cotengra** — open-source TN with best contraction-path heuristics.

**Wrap (second tier):**
- **PennyLane Lightning** (default.qubit/lightning.gpu) — gradient/JAX story.
- **Strawberry Fields / MrMustard** — only if photonic/CV is in scope.
- **NetKet** — for NQS research users.

**Ignore / deprioritize:**
- Intel-QS, QuEST, Jet — superseded by CUDA-Q and Aer for production.
- Cirq's built-in simulator — fine for tests, but qsim is what you want.
- IBM Cloud simulators — IBM has effectively retired them; route to Aer locally.
- TensorCircuit — niche; quimb covers the same ground with more momentum.

Hard requirements: a uniform circuit IR, calibration-data ingestion
(T1/T2/readout/crosstalk JSON from real backends), an MPI/NCCL launcher
abstraction, and a "method selector" that picks SV vs MPS vs Stim vs DM from
circuit structure and noise spec.

## Sources

- [cuQuantum Appliance / cusvaer](https://docs.nvidia.com/cuda/cuquantum/latest/appliance/cusvaer.html)
- [44-qubit Aer simulation on LUMI](https://fiqci.fi/publications/2025-04-01-LUMI-quantum-simulations-qiskit-aer)
- [Qiskit Aer multi-GPU docs](https://qiskit.github.io/qiskit-aer/howtos/running_gpu.html)
- [Efficient TN simulation of IBM Eagle (PRX Quantum)](https://link.aps.org/doi/10.1103/PRXQuantum.5.010308)
- [Stim release on Zenodo](https://zenodo.org/records/15354872)
- [STABSim](https://arxiv.org/pdf/2507.03092)
- [Tensor networks for QC review](https://arxiv.org/html/2503.08626v1)
- [MPS in NVIDIA cuQuantum](https://developer.nvidia.com/blog/enabling-matrix-product-state-based-quantum-circuit-simulation-with-nvidia-cuquantum/)
- [Strawberry Fields docs](https://strawberryfields.readthedocs.io/en/stable/introduction/introduction.html)
- [Piquasso (Quantum 2025)](https://quantum-journal.org/wp-content/uploads/2025/04/q-2025-04-15-1708.pdf)
- [CUDA-Q at GTC 2026](https://nvidia.github.io/cuda-quantum/blogs/blog/2026/03/16/cudaq-GTC-26/)
- [Largest single-GPU sim on AMD by BlueQubit](https://www.amd.com/en/developer/resources/technical-articles/2026/largest-single-gpu-quantum-simulation-on-amd-by-bluequbit.html)
- [Amazon Braket pricing](https://aws.amazon.com/braket/pricing/)
- [Compare Braket simulators](https://docs.aws.amazon.com/braket/latest/developerguide/choose-a-simulator.html)
- [Qiskit Aer noise tutorials](https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html)
- [Crosstalk attacks & defence (Adv. Quantum Tech. 2025)](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/qute.202500009)
- [Quantum Frontiers: has quantum advantage been achieved?](https://quantumfrontiers.com/2026/01/25/has-quantum-advantage-been-achieved-part-2-considering-the-evidence/)
