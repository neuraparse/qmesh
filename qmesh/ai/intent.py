"""qmesh.ai.intent — rule-based intent compiler.

Recognises common quantum-circuit-construction intents and emits qmesh.ir
Modules directly, without LLM round-trip. Faster, deterministic, and free
— used as the back-stop for the mock LLM provider and as a first-pass
shortcut for the real LLM copilot.

Patterns covered:
  - "Bell"                     → 2-qubit Bell state
  - "GHZ on N"                 → N-qubit GHZ
  - "QFT on N"                 → N-qubit Quantum Fourier Transform
  - "n-qubit hardware-efficient ansatz" → Ry-CX-Ry brick wall
  - "W state on N"             → 1-excitation symmetric superposition
  - "random Clifford on N depth D" → seeded Clifford-only circuit
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from qmesh.ir.builder import circuit
from qmesh.ir.module import Module


@dataclass(slots=True)
class IntentResult:
    pattern: str
    module: Module
    description: str


_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bbell\b", re.I),                                          "bell"),
    (re.compile(r"ghz\s*(?:state)?\s*(?:on\s+)?(\d+)", re.I),                 "ghz"),
    (re.compile(r"qft\s*(?:on\s+)?(\d+)", re.I),                              "qft"),
    (re.compile(r"hardware[-\s]efficient.*?(\d+)", re.I),                     "hea"),
    (re.compile(r"\bansatz\b.*?(\d+)", re.I),                                 "hea"),
    (re.compile(r"w\s*state\s*(?:on\s+)?(\d+)", re.I),                        "w_state"),
    (re.compile(r"random\s+clifford.*?(\d+).*?depth\s*(\d+)", re.I),          "random_clifford"),
]


def compile_intent(prompt: str) -> IntentResult | None:
    for pat, name in _PATTERNS:
        m = pat.search(prompt)
        if m:
            return _build(name, m.groups(), prompt)
    return None


def _build(name: str, groups: tuple[str, ...], prompt: str) -> IntentResult:
    if name == "bell":
        with circuit("bell", n_qubits=2, n_bits=2) as c:
            c.h(0); c.cx(0, 1)
            c.measure(0, 0); c.measure(1, 1)
        return IntentResult("bell", c.module, "Bell state |00⟩ + |11⟩")

    if name == "ghz":
        n = int(groups[0])
        with circuit(f"ghz{n}", n_qubits=n, n_bits=n) as c:
            c.h(0)
            for i in range(n - 1):
                c.cx(i, i + 1)
            for i in range(n):
                c.measure(i, i)
        return IntentResult("ghz", c.module, f"{n}-qubit GHZ state")

    if name == "qft":
        n = int(groups[0])
        with circuit(f"qft{n}", n_qubits=n, n_bits=n) as c:
            for i in range(n):
                c.h(i)
                for k in range(i + 1, n):
                    c._gate("cp", [k, i], (math.pi / (2 ** (k - i)),))
            for i in range(n // 2):
                c.swap(i, n - 1 - i)
            for i in range(n):
                c.measure(i, i)
        return IntentResult("qft", c.module, f"{n}-qubit QFT")

    if name == "hea":
        n = int(groups[0])
        # 2-layer Ry-CX-Ry hardware-efficient ansatz with theta=π/4 placeholder
        with circuit(f"hea{n}", n_qubits=n, n_bits=n) as c:
            for i in range(n):
                c.ry(math.pi / 4, i)
            for i in range(0, n - 1, 2):
                c.cx(i, i + 1)
            for i in range(1, n - 1, 2):
                c.cx(i, i + 1)
            for i in range(n):
                c.ry(math.pi / 4, i)
            for i in range(n):
                c.measure(i, i)
        return IntentResult("hardware_efficient_ansatz", c.module,
                           f"{n}-qubit hardware-efficient ansatz, θ=π/4 (placeholder)")

    if name == "w_state":
        n = int(groups[0])
        # Approximate W state via Ry-cascade construction
        with circuit(f"w{n}", n_qubits=n, n_bits=n) as c:
            for k in range(n - 1):
                theta = 2 * math.acos(math.sqrt(1 / (n - k)))
                c.ry(theta, k)
                if k + 1 < n:
                    c.cx(k, k + 1)
            c.x(n - 1)
            for i in range(n):
                c.measure(i, i)
        return IntentResult("w_state", c.module, f"{n}-qubit W state")

    if name == "random_clifford":
        import random
        n, depth = int(groups[0]), int(groups[1])
        rng = random.Random(42)
        with circuit(f"rc{n}d{depth}", n_qubits=n, n_bits=n) as c:
            for _ in range(depth):
                for q in range(n):
                    c._gate(rng.choice(["h", "s", "x", "z"]), [q])
                for a in range(0, n - 1, 2):
                    c.cx(a, a + 1)
                for a in range(1, n - 1, 2):
                    c.cx(a, a + 1)
            for q in range(n):
                c.measure(q, q)
        return IntentResult("random_clifford", c.module,
                           f"{n}-qubit random Clifford circuit, depth {depth}")

    raise ValueError(f"unknown intent pattern '{name}'")


__all__ = ["IntentResult", "compile_intent"]
