# Quantum Error Correction: State of the Field, May 2026

Source: parallel research agent run, 2026-05-01.

## 1. Surface code progress
Google's December 2024 *Willow* result (Nature) was the first clean
"below threshold" demonstration: scaling distance 3 to 5 to 7 suppressed
logical error per cycle by Λ ≈ 2.14, reaching **0.143% per cycle at d=7** with
101 qubits, exceeding the best physical qubit lifetime by 2.4× — the first
true break-even logical memory [nature.com][research.google]. Real-time
decoding held up at ~63 µs latency at d=5 across 1M cycles. Through 2025–2026,
Google's post-Willow generations have pushed Λ further and demonstrated d=9
and d=11 patches; **AlphaQubit 2** (March 2026, arXiv 2512.07737) reports
near-optimal logical error rates for surface and color codes with
**sub-µs/cycle real-time inference up to d=11**.

## 2. qLDPC and bivariate-bicycle codes
IBM's 2024 Nature paper on the **[[144,12,12]] "gross" bivariate-bicycle (BB)
code** showed ~10× physical-qubit savings vs. the surface code at comparable
suppression [ibm.com/quantum]. Follow-ups in 2025–26 produced coprime BB
variants and neutral-atom layouts (Quantum journal, Feb 2026).

- **IBM Loon** (Nov 2025) — testbed integrating multi-layer routing and
  long-range c-couplers required for BB connectivity.
- **IBM Kookaburra** (2026) — first IBM module to store data in a qLDPC code
  and operate it with an attached **Logical Processing Unit (LPU)**.
- **IBM Cockatoo** (2027) — entangles two QEC modules.
- Full FTQC targeted for 2029 [ibm.com/quantum, quantumcomputingreport.com].
- **IonQ "Walking Cat"** blueprint (April 2026) commits to qLDPC over surface
  codes for trapped ions [ionq.com].

## 3. Color, Bacon-Shor, Floquet codes
Color codes returned to mainstream relevance now that AlphaQubit 2 decodes
them in real time at d=9, and Google has experimentally run color-code
patches. **Floquet codes** are gaining ground: hyperbolic and semi-hyperbolic
Floquet codes show ~5.6× lower overhead than surface codes and ~30× over
honeycomb codes for circuit-level depolarizing noise; one variant encodes
52 logical qubits in 400 physical at d=8. **Bacon-Shor** is largely niche, but
a Floquet-Bacon-Shor hybrid was implemented on superconducting hardware in
2025. Net: **surface codes still dominate superconducting, but Floquet/qLDPC
are now serious competitors**, and color codes are alive thanks to better
decoders.

## 4. Magic state distillation & cultivation
Gidney & Shutty's **magic state cultivation** (arXiv 2409.17595, late 2024)
replaced multi-stage distillation with in-place growth inside a surface patch,
hitting **2×10⁻⁹ logical error at 10⁻³ noise**, falling to **4×10⁻¹¹ at 5×10⁻⁴**,
with an order-of-magnitude reduction in qubit-rounds vs. prior distillation.
2025 follow-ups: high-rate cultivation directly on the surface code (arXiv
2502.01743); "magic tricycles" generalizing cultivation to qLDPC (Aug 2025);
first superconducting-hardware demonstration (arXiv 2512.13908). **Cultivation
is now baked into resource estimates as default; dedicated distillation
factories are largely deprecated.**

## 5. Logical qubit demonstrations (mid-2026)

| Vendor | System | Demonstration |
|---|---|---|
| Google QAI | Willow + post-Willow | d=7 below threshold, breakeven memory; AlphaQubit 2 real-time at d=11 |
| IBM | Heron r2 / Loon / Kookaburra | Heron r2 5,000 2Q gates; Loon Nov 2025 testbed; Kookaburra 2026 first qLDPC module + LPU |
| Quantinuum | Helios (Nov 2025) | 98 trapped-ion qubits, 2Q fidelity 99.921%; **94 logical qubits in GHZ state** Mar 2026; 12 protected logical on H2 |
| Atom Computing + MS | Magne | 50 logical from ~1,200 atoms, operational early 2027 |
| QuEra | AIST + Gemini | ~37 logical at AIST today; 2026 system targeting ~100 logical / ~10,000 physical |
| Pasqal | Orion | 250-qubit QPU 2026, 1,000+ scaling |
| IonQ | Tempo / Walking Cat | 99.99% 2Q fidelity Oct 2025; 256-qubit prototype 2026; ~1,600 logical by 2028 (qLDPC) |

## 6. Mitigation vs correction
2026 stance: **mitigation for NISQ-scale expectation values**, **correction for
any algorithm needing repeatability or non-Clifford depth**. ZNE, PEC, DD, CDR,
REM, PT are stable in **Mitiq 1.0** (public API frozen); virtual distillation
lives in `mitiq.experimental`. PEC is ground-truth-accurate but exponentially
expensive in sampling; ZNE is the workhorse default. Above ~50 qubits or for
sampling-heavy workloads, mitigation overheads exceed FT overheads — vendors
now route those workloads to logical execution.

## 7. Decoders
Real-time decoding is no longer a bottleneck at moderate distance.
- **AlphaQubit 2** (DeepMind/Google, March 2026): sub-µs/cycle on commercial
  accelerators, surface to d=11, color to d=9.
- **QUNET**: 4-bit quantized UNet on FPGA with early-exit, ~34% latency savings.
- **GNN FPGA accelerator** (March 2026): <1 µs at d=7.
- **EdenCode** (Jan 2026) productizes neural decoders.
- Sliding-window MWPM and Union-Find remain baselines for surface codes;
  **BP+OSD dominates qLDPC.**

## 8. Resource estimation
- **RSA-2048**: Gidney's May 2025 paper — **<1M noisy qubits, <1 week**
  runtime, a 20× qubit reduction vs. his 2019 20M-qubit/8-hour estimate, by
  combining approximate modular arithmetic, yoked surface codes, and magic
  state cultivation [arxiv.org/2505.15917]. Some 2026 estimates project
  sub-100k qubits under qLDPC + cultivation.
- **FeMoco** ground-state simulation now estimated in low-millions of physical
  qubits, hours of runtime.
- **ECC** factoring sits at ~500k physical qubits.

## 9. Q-day proximity
NSA CNSA 2.0 mandates **quantum-safe national security systems by Jan 2027**;
NIST PQC migration timelines published. Industry and CSA estimates for a CRQC
have compressed: previously 2030–2035, now widely cited as **late-2020s
feasible for nation-state actors** following the Gidney 2025 reduction. **2026
is the FBI/NIST/CISA-designated "Year of Quantum Security."** No CRQC
demonstrated yet; harvest-now-decrypt-later remains the operative threat
model.

## What a 2026 framework should integrate

- **Decoders**: PyMatching / sliding-window MWPM as default; **Stim** for
  circuit-level simulation; **AlphaQubit 2** or open neural decoders
  (QUNET-style) as opt-in real-time backends; BP+OSD for qLDPC; FPGA hooks
  (Riverlane Deltaflow-style) for sub-µs paths.
- **Mitigation**: **Mitiq 1.0** stable API (ZNE, PEC, DDD, LRE, CDR, REM, PT);
  experimental VD module behind a flag.
- **Codes**: surface, color, BB qLDPC, Floquet — selected per backend
  topology, not hardcoded.
- **Magic states**: cultivation primitives by default; distillation only as
  fallback.
- **User abstractions**: a "FT mode" toggle that promotes a circuit from
  physical-qubit-with-mitigation to logical-qubit-with-correction, with
  code/decoder/cultivation chosen per backend; resource-estimation pass
  (Microsoft-Estimator-compatible) before submission; transparent budget
  reporting (logical qubits, T-states, cycles, wall time).
- **Compiler layer**: lattice-surgery scheduler aware of cultivation pools
  (arXiv 2512.06484) and BB-code routing.

## Sources

- [Quantum error correction below the surface code threshold (Nature)](https://www.nature.com/articles/s41586-024-08449-y)
- [Making QEC work — Google Research](https://research.google/blog/making-quantum-error-correction-work/)
- [IBM qLDPC paper](https://www.ibm.com/quantum/blog/nature-qldpc-error-correction)
- [IBM lays out clear path to FTQC](https://www.ibm.com/quantum/blog/large-scale-ftqc)
- [IBM 2026 Roadmap](https://www.ibm.com/roadmaps/quantum/2026/)
- [Coprime BB Codes (Quantum, 2026)](https://quantum-journal.org/papers/q-2026-02-23-2009/pdf/)
- [Magic state cultivation (Gidney & Shutty)](https://arxiv.org/abs/2409.17595)
- [High Rate Magic State Cultivation](https://arxiv.org/html/2502.01743)
- [Magic state cultivation on superconducting](https://arxiv.org/html/2512.13908v1)
- [Quantinuum Helios announcement](https://www.quantinuum.com/blog/introducing-helios-the-most-accurate-quantum-computer-in-the-world)
- [Quantinuum 94 protected logical qubits](https://thequantuminsider.com/2026/03/10/quantinuum-researchers-demonstrates-quantum-computations-with-dozens-of-protected-logical-qubits/)
- [Neutral Atom QC 2026 (IEEE Spectrum)](https://spectrum.ieee.org/neutral-atom-quantum-computing)
- [Pasqal 2025 Roadmap](https://www.pasqal.com/newsroom/pasqal-releases-2025-roadmap/)
- [IonQ Walking Cat Blueprint](https://www.ionq.com/blog/blueprint-for-fault-tolerant-trapped-ion-quantum-computing-the-walking-cat-architecture)
- [IonQ 99.99% record](https://postquantum.com/quantum-research/ionq-record-2025/)
- [Hyperbolic Floquet codes](https://link.aps.org/doi/10.1103/PRXQuantum.5.040327)
- [AlphaQubit 2 / scalable real-time decoder](https://arxiv.org/abs/2512.07737)
- [FPGA-Accelerated Early-Exit Neural Decoder (IEEE)](https://ieeexplore.ieee.org/document/11272758/)
- [GNN Accelerator for QEC](https://arxiv.org/abs/2603.22149)
- [EdenCode emerges from stealth](https://thequantuminsider.com/2026/01/24/edencode-emerges-from-stealth-with-real-time-ai-decoder-for-quantum-error-correction/)
- [Gidney 2025: factor 2048-bit RSA](https://arxiv.org/abs/2505.15917)
- [Q-Day Just Got Closer](https://thequantuminsider.com/2026/03/31/q-day-just-got-closer-three-papers-in-three-months-are-rewriting-the-quantum-threat-timeline/)
- [Mitiq Changelog](https://mitiq.readthedocs.io/en/stable/changelog.html)
- [Scheduling Lattice Surgery with Cultivation](https://arxiv.org/html/2512.06484v1)
