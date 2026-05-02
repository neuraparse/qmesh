"""AI-assisted workflow: prompt → IR → validation → run.

Demonstrates the qmesh.ai copilot end-to-end for several prompt patterns,
including a fall-through to the LLM provider for novel requests. Works
without API keys via the rule-based mock provider.

Run:
    PYTHONPATH=. python3 examples/ai_workflow.py
"""

from __future__ import annotations

from rich import print
from rich.panel import Panel
from rich.table import Table

import qmesh
from qmesh.ai import draft


PROMPTS = [
    "Build a Bell state",
    "GHZ state on 6 qubits",
    "QFT on 4 qubits",
    "hardware-efficient ansatz on 5 qubits",
    "W state on 4 qubits",
    "random Clifford on 8 qubits depth 4",
]


def main() -> None:
    print(Panel.fit(
        "[bold cyan]qmesh.ai copilot[/]\n\n"
        "Natural-language prompts → IR → simulator validation → counts.\n"
        "[dim]Without API keys, the rule-based mock provider matches common[/]\n"
        "[dim]patterns directly. With ANTHROPIC_API_KEY / OPENAI_API_KEY /[/]\n"
        "[dim]QMESH_OLLAMA_URL set, novel prompts route through the real LLM[/]\n"
        "[dim]and are QCoder-validated before being returned.[/]",
        title="Phase-5α",
    ))

    table = Table(title="AI-assisted drafts")
    table.add_column("prompt", overflow="fold", max_width=40)
    table.add_column("intent / provider", overflow="fold")
    table.add_column("iters", justify="right")
    table.add_column("validation", overflow="fold", max_width=30)
    table.add_column("counts (top)", overflow="fold", max_width=30)

    for prompt in PROMPTS:
        result = draft(prompt)
        # Run the produced module on Aer
        run, _ = qmesh.submit(result.module, backend="qmesh.aer", shots=1024,
                             sign=False, ledger_dir="ledger/ai_workflow")
        top = sorted(run.counts.items(), key=lambda kv: -kv[1])[:2]
        top_str = ", ".join(f"{k}={v}" for k, v in top)
        intent_or_provider = (
            f"intent:{result.intent_pattern}"
            if result.intent_pattern else f"llm:{result.provider}/{result.model}"
        )
        validation_str = (
            "passed" if result.validation.get("passed") else
            f"failed: {result.validation}"
        )
        table.add_row(
            prompt, intent_or_provider, str(result.iterations),
            validation_str, top_str,
        )
    print(table)


if __name__ == "__main__":
    main()
