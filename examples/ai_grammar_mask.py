"""Phase 5γ demo: token-level grammar masking adapters.

Walks through the StubAdapter on Bell + GHZ prompts, printing the legal
next tokens proposed by the QASMGrammarGate at each step. If `outlines` is
installed, also emits the regex compiled by OutlinesAdapter for a "QFT on
3 qubits" prompt. Closes with a Rich Panel summary.

Run:
    PYTHONPATH=. python3 examples/ai_grammar_mask.py
"""

from __future__ import annotations

from rich import print
from rich.panel import Panel
from rich.table import Table

from qmesh.ai.constrained_decoding import QASMGrammarGate
from qmesh.ai.grammar_mask import (
    LogitBiasAdapter,
    OutlinesAdapter,
    StubAdapter,
    compile_grammar_mask,
)
from qmesh.ai.llm import MockProvider


def _walk_legal_tokens(adapter: StubAdapter, qasm: str, label: str) -> None:
    """Print the gate's legal next-token set after each line of `qasm`."""
    table = Table(title=f"legal_next_tokens walk -- {label}", show_lines=False)
    table.add_column("step", justify="right")
    table.add_column("just-emitted line", overflow="fold", max_width=50)
    table.add_column("# legal next", justify="right")
    table.add_column("first 4 legal tokens", overflow="fold", max_width=60)

    accumulated = ""
    for i, line in enumerate(qasm.splitlines(), start=1):
        accumulated += line + "\n"
        legal = adapter.legal_next_tokens(accumulated)
        preview = ", ".join(legal[:4]) if legal else "(none)"
        table.add_row(str(i), line, str(len(legal)), preview)
    print(table)


def main() -> None:
    print(Panel.fit(
        "[bold cyan]qmesh.ai grammar-mask adapters[/]\n\n"
        "Walks the [bold]QASMGrammarGate[/]'s legal-next-token oracle on a Bell\n"
        "and a 3-qubit GHZ generation, then shows the [bold]LogitBiasAdapter[/]\n"
        "compiling a logit-bias dict (string-keyed when [italic]tiktoken[/] is\n"
        "missing). If [italic]outlines[/] is installed, the OutlinesAdapter\n"
        "regex is shown for a QFT prompt.",
        title="Phase 5g - grammar mask",
    ))

    gate = QASMGrammarGate()
    stub: StubAdapter = compile_grammar_mask(gate, backend="stub",
                                             provider=MockProvider())  # type: ignore[assignment]

    # --- Bell ---
    bell = stub.complete("Build a Bell state").text
    _walk_legal_tokens(stub, bell, "Bell")

    # --- GHZ ---
    gate.reset()
    ghz = stub.complete("GHZ state on 3 qubits").text
    _walk_legal_tokens(stub, ghz, "GHZ-3")

    # --- LogitBiasAdapter ---
    lb: LogitBiasAdapter = compile_grammar_mask(gate, backend="logit_bias")  # type: ignore[assignment]
    bias = lb.compile_to_logit_bias("OPENQASM 3.0;\n")
    print(Panel.fit(
        f"[bold]LogitBiasAdapter.maturity[/] = {lb.maturity}\n"
        f"tokenizer present : {lb.has_tokenizer()}\n"
        f"# legal next tokens : {len(bias)}\n"
        f"sample keys : {list(bias)[:4]}",
        title="LogitBiasAdapter",
    ))

    # --- OutlinesAdapter ---
    ol: OutlinesAdapter = compile_grammar_mask(gate, backend="outlines")  # type: ignore[assignment]
    regex = ol.compile_to_outlines()
    avail = ol.is_available()
    print(Panel.fit(
        f"[bold]OutlinesAdapter.maturity[/] = {ol.maturity}\n"
        f"outlines installed : {avail}\n"
        f"regex length       : {len(regex)} chars\n"
        f"regex prefix       : {regex[:80]} ...",
        title="OutlinesAdapter",
    ))
    if avail:
        try:
            text = ol.generate_constrained(
                provider="transformers", model="gpt2",
                prompt="QFT on 3 qubits", max_tokens=64,
            )
            print(Panel.fit(text, title="OutlinesAdapter -- QFT on 3 qubits"))
        except Exception as e:  # noqa: BLE001
            print(Panel.fit(f"outlines run skipped: {e}", title="note"))

    # --- summary ---
    summary = (
        f"[bold green]Bell QASM[/] - {len(bell.splitlines())} lines, "
        f"valid through grammar gate.\n"
        f"[bold green]GHZ-3 QASM[/] - {len(ghz.splitlines())} lines, "
        f"valid through grammar gate.\n"
        f"[bold]Adapters[/] : stub({stub.maturity}), "
        f"logit_bias({lb.maturity}), outlines({ol.maturity})"
    )
    print(Panel.fit(summary, title="result"))


if __name__ == "__main__":
    main()
