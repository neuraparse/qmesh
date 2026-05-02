# Quantum Hardware Vendor Survey — May 2026

Compiled from cross-references in software/qec/ai_hybrid agent reports plus
targeted web searches (PsiQuantum, IQM, Origin Quantum/USTC).

## Superconducting

### IBM Quantum
- **Heron r2** — current production chip; 5,000 two-qubit gates per shot;
  utility-scale dynamic circuits across all backends.
- **Flamingo** — 462-qubit module with link.
- **Condor** — 1,121-qubit (research).
- **Loon** (Nov 2025) — testbed for qLDPC routing (multi-layer routing,
  long-range c-couplers).
- **Kookaburra** (2026) — first IBM module to encode data in the [[144,12,12]]
  bivariate-bicycle qLDPC code, with an attached Logical Processing Unit (LPU).
- **Cockatoo** (2027) — entangles two QEC modules.
- Full FTQC targeted **2029**.
- Dynamic circuits: `if/else`, `for`, `while` via Qiskit Runtime.
- Sources: [ibm.com/roadmaps](https://www.ibm.com/roadmaps/quantum/2026/),
  [ibm.com/quantum/blog](https://www.ibm.com/quantum/blog/large-scale-ftqc).

### Google Quantum AI
- **Willow** (Dec 2024, Nature) — first below-threshold demonstration:
  Λ ≈ 2.14 between distance 3/5/7, **0.143% logical error per cycle at d=7**,
  ~2.4× better than physical lifetime.
- Post-Willow generations through 2025–2026 demonstrated d=9 and d=11.
- **AlphaQubit 2** (March 2026) decodes surface d=11 + color d=9 in
  **<1 µs/cycle**.
- Hardware access remains gated to academic partners.
- Sources: [nature.com/articles/s41586-024-08449-y](https://www.nature.com/articles/s41586-024-08449-y),
  [arxiv.org/abs/2512.07737](https://arxiv.org/abs/2512.07737).

### Rigetti
- **Ankaa** + **Novera** — superconducting; 84-qubit Ankaa-3 in 2024.
  Reduced public visibility in 2026 vs IBM/Google.

### IQM (Finland)
- **Crystal** topology: two-qubit fidelity 99.9% in test systems, 99.95% target.
- **Halocene 150-qubit** (end-2026): 99.7% 2Q target; 5 logical qubits with
  Clifford-gate error correction features. On-premises modular.
- Sources: [meetiqm.com/technology/roadmap](https://meetiqm.com/technology/roadmap/),
  [thequantuminsider.com (Nov 2025)](https://thequantuminsider.com/2025/11/13/iqm-launches-a-new-quantum-computer-product-line-for-error-correction/).

### Oxford Quantum Circuits, Anyon, Alice & Bob, SEEQC, Nord Quantique
Notable mid-tier players. Alice & Bob's cat-qubit approach (autonomous bit-flip
suppression) is distinctive but small-scale in 2026.

## Trapped ion

### Quantinuum
- **Helios** (Nov 2025) — 98 trapped-ion qubits, 2Q fidelity **99.921%**;
  arbitrary mid-circuit measurement with feedforward, real-time int/float
  arithmetic, qubit reuse.
- **94 protected logical qubits in a GHZ state** demonstrated March 2026.
- 12 protected logical qubits routine on H2.
- Sources: [quantinuum.com](https://www.quantinuum.com/blog/introducing-helios-the-most-accurate-quantum-computer-in-the-world),
  [thequantuminsider.com](https://thequantuminsider.com/2026/03/10/quantinuum-researchers-demonstrates-quantum-computations-with-dozens-of-protected-logical-qubits/).

### IonQ
- **99.99% 2Q fidelity** (Oct 2025).
- 256-qubit prototype 2026.
- **"Walking Cat"** blueprint (April 2026) commits to qLDPC over surface codes
  for trapped ions; ~1,600 logical by 2028.
- **First commercial-system photonic interconnect** (April 2026): two
  independent ion-trap systems with verified entanglement.
- Sources: [ionq.com Walking Cat](https://www.ionq.com/blog/blueprint-for-fault-tolerant-trapped-ion-quantum-computing-the-walking-cat-architecture),
  [ionq.com photonic milestone](https://www.ionq.com/news/ionq-achieves-key-photonic-interconnect-milestone-demonstrating-networked-quantum-systems-using-entanglement).

## Neutral atom

### Atom Computing + Microsoft
- **Magne** machine: ~50 logical qubits from ~1,200 atoms, operational
  early 2027.
- Source: [spectrum.ieee.org](https://spectrum.ieee.org/neutral-atom-quantum-computing).

### QuEra
- ~37 logical qubits at AIST today; 2026 target ~100 logical / ~10,000
  physical (Gemini class).
- **Bloqade** programming model.
- Source: [spectrum.ieee.org](https://spectrum.ieee.org/neutral-atom-quantum-computing).

### Pasqal
- 250-qubit QPU 2026 (Orion); 1,000+ scaling.
- **Pulser** programming model.
- Source: [pasqal.com/newsroom](https://www.pasqal.com/newsroom/pasqal-releases-2025-roadmap/).

### Infleqtion
- ColdQuanta-derived; partnered with ORNL on neutral-atom + GB200 NVL72.

## Photonic

### PsiQuantum
- **Omega** chiplet (Nature 2024): SPAM 99.98%, chip-to-chip qubit
  interconnect 99.72%.
- $1B raise Sept 2025; targets **1M physical qubits late 2020s**.
- **Chicago + Brisbane** sites announced March 2026 with $1B build-out.
- **Construct** (FT algorithm tool) and Airbus QuLAB collaboration (Jan 2026).
- Source: [psiquantum.com/omega](https://www.psiquantum.com/omega),
  [businesswire 2025](https://www.businesswire.com/news/home/20250910135739/en/PsiQuantum-Raises-$1-Billion-to-Build-Million-Qubit-Scale-Fault-Tolerant-Quantum-Computers).

### Xanadu
- **Aurora** (photonic-CV); PennyLane on ORNL Frontier (April 2026).
- **Strawberry Fields** + **MrMustard** simulation stacks.

### USTC (Jiuzhang series)
- **Jiuzhang 4.0** (Aug 2025): 3,050 detected photons, world-leading boson
  sampling demonstrations.
- Source: [english.cas.cn](https://english.cas.cn/newsroom/cas_media/202310/t20231011_378680.shtml).

### Quandela / ORCA
- Linear-optical / discrete-photon systems. ORCA accessible via CUDA-Q.

## Annealing / specialty

### D-Wave
- **Advantage 2** — annealing; relevance vs gate-based has narrowed but
  optimization market persists.

### Quantum Brilliance
- Room-temp NV-diamond accelerator; **Qristal** software stack.

## China

### Origin Quantum
- **Wukong** 72-qubit superconducting; cloud accessible (30M+ visits from
  145 countries since launch despite US Entity List).
- **Origin Pilot V4.0** (Feb 2026): "world's first publicly downloadable
  quantum OS".
- IPO process at ~¥6.9B.
- Source: [postquantum.com](https://postquantum.com/quantum-computing-companies/origin-quantum/).

### Other
- USTC for photonic / academic side; Baidu has reduced public quantum
  presence.

## Networking layer (new in 2026)

### Cisco Universal Quantum Switch
- **April 2026**: routes entangled photons over telecom fiber at room
  temperature; metro-scale entanglement-swapping demo with Qunnect over
  deployed fiber; ≤4% fidelity loss.
- Source: [newsroom.cisco.com](https://newsroom.cisco.com/c/r/newsroom/en/us/a/y2026/m04/cisco-introduces-universal-quantum-switch-advancing-the-path-to-a-quantum-network.html).

### EPB Chattanooga
- First commercial quantum networking hub with an IonQ system.

## Cross-cutting trends (May 2026)

1. **Below-threshold logical memory** is now table stakes (Google Willow).
   The race has moved to (a) larger code distance with sub-µs decoders,
   (b) qLDPC adoption at scale.
2. **qLDPC is winning** the FT-architecture argument: IBM (Kookaburra),
   IonQ (Walking Cat) commit, neutral-atom variants follow. Surface-code
   "monoculture" of 2023–2024 is over.
3. **Magic-state cultivation** (Gidney 2024) eliminated multi-stage
   distillation factories; now baked into all serious resource estimates.
4. **Photonic interconnect ships commercially** for the first time (IonQ April
   2026). Mixed-modality programs become realistic — but no IR captures them.
5. **HPC-coupled quantum** is a 2026 reality: RIKEN Fugaku-Reimei, NERSC
   QCAN, ORNL+Infleqtion, JSC DGX Quantum. Hybrid orchestration is the next
   bottleneck.
6. **Q-day timeline compressed** post-Gidney 2025 — RSA-2048 in <1M qubits,
   <1 week. NSA CNSA 2.0 mandates quantum-safe NSS by Jan 2027.
7. **AI-augmented control** is mainstream: AlphaQubit 2 real-time decoder,
   AlphaTensor-Quantum T-count reduction, Qiskit Code Assistant in production.
