"""Phase 5α tests: intent compiler, copilot, decoder training."""

from __future__ import annotations

import os

import pytest

import qmesh
from qmesh.ai import draft, train_neural_decoder
from qmesh.ai.intent import compile_intent
from qmesh.ai.llm import MockProvider, auto_provider


# ---------- intent compiler ----------

@pytest.mark.parametrize("prompt,expected_pattern", [
    ("Build a Bell state",                              "bell"),
    ("GHZ state on 5 qubits",                            "ghz"),
    ("QFT on 4 qubits",                                  "qft"),
    ("hardware-efficient ansatz on 6 qubits",            "hardware_efficient_ansatz"),
    ("W state on 3 qubits",                              "w_state"),
    ("random Clifford on 7 qubits depth 5",              "random_clifford"),
])
def test_intent_compiler_recognises(prompt, expected_pattern):
    r = compile_intent(prompt)
    assert r is not None, f"intent compiler missed prompt: {prompt!r}"
    assert r.pattern == expected_pattern


def test_intent_compiler_misses_unknown():
    r = compile_intent("the moon is made of cheese")
    assert r is None


# ---------- mock provider ----------

def test_mock_provider_returns_qasm():
    prov = MockProvider()
    completion = prov.complete("Make a Bell state please")
    assert "OPENQASM" in completion.text
    assert completion.provider == "mock"


def test_auto_provider_returns_mock_when_no_keys(monkeypatch):
    # Clear all known env keys
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "QMESH_OLLAMA_URL"):
        monkeypatch.delenv(k, raising=False)
    prov = auto_provider()
    assert prov.name == "mock"


# ---------- copilot ----------

def test_copilot_intent_path():
    r = draft("GHZ state on 4 qubits")
    assert r.intent_pattern == "ghz"
    assert r.iterations == 0
    assert r.validation.get("passed", False)


def test_copilot_validation_runs_on_simulator():
    r = draft("Bell state")
    # The validator runs on qmesh.aer; results live in r.validation
    assert "backend" in r.validation
    assert r.validation["backend"] == "qmesh.aer"


def test_copilot_module_is_runnable(tmp_path):
    pytest.importorskip("qiskit_aer")
    r = draft("Build a Bell state")
    result, _ = qmesh.submit(r.module, backend="qmesh.aer", shots=512,
                            ledger_dir=tmp_path)
    p_even = (result.counts.get("00", 0) + result.counts.get("11", 0)) / result.shots
    assert p_even > 0.95


def test_copilot_falls_through_to_mock(monkeypatch):
    """Force the LLM path: a prompt the intent compiler can't handle."""
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "QMESH_OLLAMA_URL"):
        monkeypatch.delenv(k, raising=False)
    r = draft("a strangely-worded request that does not match any pattern")
    # MockProvider falls back to a Bell stub, validator passes
    assert r.intent_pattern is None
    assert r.provider == "mock"


# ---------- neural decoder training ----------

def test_neural_decoder_training_pipeline_runs(tmp_path):
    pytest.importorskip("torch")
    from qmesh.ftmode import SurfaceCode
    code = SurfaceCode(distance=3, rounds=2)
    decoder, tr = train_neural_decoder(
        code, physical_error_rate=1e-3,
        n_train=2_000, n_val=500, epochs=2, hidden_dim=32,
        weights_dir=tmp_path, seed=42,
    )
    assert tr.weights_path.exists()
    assert tr.weights_sha256
    assert tr.accuracy > 0.5            # better than random; not great with this little data
    assert decoder.identity()["trained"] is True
    # Manifest should include the trained=True flag from the decoder
    assert decoder.identity()["weights_path"] == str(tr.weights_path)


# ---------- Phase 5β: transformer decoder ----------

def test_transformer_decoder_trains_and_persists_weights(tmp_path):
    pytest.importorskip("torch")
    from qmesh.ai.transformer_decoder import train_transformer_decoder
    from qmesh.ftmode import SurfaceCode
    code = SurfaceCode(distance=3, rounds=3)
    _decoder, tr = train_transformer_decoder(
        code, physical_error_rate=1e-3,
        n_train=5_000, n_val=1_000, epochs=4,
        weights_dir=tmp_path, seed=42,
    )
    assert tr.weights_path.exists()
    assert tr.weights_sha256
    # AlphaQubit-2 is 100M+ params; this is ~50K, but on d=3 + 5k shots
    # it still beats the trivial >random bar comfortably.
    assert tr.accuracy > 0.9, f"val_accuracy={tr.accuracy:.3f} below 0.9"


def test_transformer_decoder_loads_into_neural_slot(tmp_path):
    pytest.importorskip("torch")
    from qmesh.ai.transformer_decoder import (
        load_transformer_decoder,
        train_transformer_decoder,
    )
    from qmesh.ftmode import SurfaceCode
    code = SurfaceCode(distance=3, rounds=3)
    _decoder, tr = train_transformer_decoder(
        code, physical_error_rate=1e-3,
        n_train=2_000, n_val=500, epochs=2,
        weights_dir=tmp_path, seed=42,
    )
    loaded = load_transformer_decoder(tr.weights_path)
    # Drop into the NeuralDecoder slot for the same circuit.
    stim_circ = code.generate_memory_circuit(physical_error_rate=1e-3)
    loaded.from_circuit(stim_circ)
    ident = loaded.identity()
    assert ident["trained"] is True
    assert ident.get("architecture") == "transformer"


# ---------- Phase 5β: constrained decoding ----------

def test_constrained_decoding_accepts_valid_qasm():
    from qmesh.ai.constrained_decoding import (
        QASMGrammarGate,
        validate_qasm,
        validate_qasm_streaming,
    )
    bell = (
        "OPENQASM 3.0;\n"
        'include "stdgates.inc";\n'
        "qubit[2] q;\nbit[2] c;\n"
        "h q[0];\ncx q[0], q[1];\n"
        "c[0] = measure q[0];\nc[1] = measure q[1];\n"
    )
    ok, errors = validate_qasm(bell)
    assert ok, f"valid Bell QASM rejected: {errors}"
    # streaming should be uniformly valid for every prefix that ends a stmt
    statuses = [v for _, v in validate_qasm_streaming(bell)]
    assert all(statuses), f"streaming flagged a step: {statuses}"
    # gate proposes the header at start
    gate = QASMGrammarGate()
    assert gate.legal_next_tokens("") == ["OPENQASM 3.0;"]


def test_constrained_decoding_rejects_invalid_qasm():
    from qmesh.ai.constrained_decoding import (
        validate_qasm,
        validate_qasm_streaming,
    )
    bad = (
        "OPENQASM 3.0;\n"
        'include "stdgates.inc";\n'
        "qubit[2] q;\nbit[2] c;\n"
        "cx q[0];\n"            # missing second qubit — clearly malformed
    )
    ok, errors = validate_qasm(bad)
    assert not ok
    assert any("cx" in e for e in errors), f"expected error mentioning cx, got {errors}"
    # streaming pinpoints the bad line
    statuses = list(validate_qasm_streaming(bad))
    bad_steps = [i for i, (_, v) in enumerate(statuses) if not v]
    assert bad_steps, "streaming did not flag any bad line"


# ---------- Phase 5β: QuanBench harness ----------

def test_quanbench_runs_built_in_suite_with_mock_provider():
    from qmesh.ai.quanbench import BUILTIN_SUITE, run_quanbench
    report = run_quanbench(provider="mock", suite=BUILTIN_SUITE)
    # required keys
    for k in ("schema", "n_prompts", "success_rate", "per_pattern",
              "per_provider", "results", "wall_time_seconds",
              "mean_iterations"):
        assert k in report, f"missing key {k!r} in report"
    assert report["n_prompts"] > 0
    assert 0.0 <= report["success_rate"] <= 1.0
    assert report["per_pattern"], "per_pattern dict should be non-empty"
    assert report["per_provider"], "per_provider dict should be non-empty"
    # results length matches n_prompts
    assert len(report["results"]) == report["n_prompts"]


# ---------- Phase 5γ: token-level grammar masking ----------

def test_grammar_mask_stub_adapter_filters_through_constrained_generator():
    """StubAdapter on a Bell prompt produces valid QASM via the existing
    ConstrainedQASMGenerator pipeline."""
    from qmesh.ai.constrained_decoding import QASMGrammarGate, validate_qasm
    from qmesh.ai.grammar_mask import StubAdapter, compile_grammar_mask
    from qmesh.ai.llm import MockProvider

    gate = QASMGrammarGate()
    adapter = compile_grammar_mask(gate, backend="stub", provider=MockProvider())
    assert isinstance(adapter, StubAdapter)
    assert adapter.maturity == "alpha"

    completion = adapter.complete("Build a Bell state")
    assert "OPENQASM" in completion.text
    ok, errors = validate_qasm(completion.text)
    assert ok, f"StubAdapter emitted invalid QASM: {errors}"

    # And legal_next_tokens at empty prefix proposes the header.
    assert adapter.legal_next_tokens("") == ["OPENQASM 3.0;"]


def test_grammar_mask_logit_bias_emits_token_dict_when_tokenizer_present():
    """If tiktoken is available, LogitBiasAdapter returns {int: float};
    otherwise the test skips. The string-keyed fallback is exercised
    indirectly through the demo and the stub-adapter test."""
    pytest.importorskip("tiktoken")
    import math
    from qmesh.ai.constrained_decoding import QASMGrammarGate
    from qmesh.ai.grammar_mask import LogitBiasAdapter, compile_grammar_mask

    gate = QASMGrammarGate()
    adapter = compile_grammar_mask(gate, backend="logit_bias")
    assert isinstance(adapter, LogitBiasAdapter)
    assert adapter.has_tokenizer()
    assert adapter.maturity == "beta"

    bias = adapter.compile_to_logit_bias("")
    assert isinstance(bias, dict)
    assert bias, "expected a non-empty bias dict"
    for key, val in bias.items():
        assert isinstance(key, int), f"want int token-id keys; got {type(key)}"
        assert val == math.inf, f"legal token bias should be +inf; got {val}"


def test_grammar_mask_logit_bias_falls_back_when_tokenizer_missing():
    """Without tiktoken installed the adapter should still produce a
    string-keyed mask — the honest α fallback."""
    import math
    import sys
    if "tiktoken" in sys.modules:
        pytest.skip("tiktoken installed; covered by the int-keyed test above")
    from qmesh.ai.constrained_decoding import QASMGrammarGate
    from qmesh.ai.grammar_mask import LogitBiasAdapter, compile_grammar_mask

    gate = QASMGrammarGate()
    adapter = compile_grammar_mask(gate, backend="logit_bias")
    assert isinstance(adapter, LogitBiasAdapter)
    assert not adapter.has_tokenizer()
    assert adapter.maturity == "alpha"

    bias = adapter.compile_to_logit_bias("")
    assert isinstance(bias, dict)
    assert all(isinstance(k, str) for k in bias), \
        "fallback mask should be keyed by legal-prefix strings"
    assert all(v == math.inf for v in bias.values())


def test_grammar_mask_outlines_compiles_regex():
    """OutlinesAdapter.compile_to_outlines() returns a non-empty string
    covering the QASM 3 header keywords. Skip if outlines missing — but
    note the regex itself doesn't require the lib (it's just a string)."""
    from qmesh.ai.constrained_decoding import QASMGrammarGate
    from qmesh.ai.grammar_mask import OutlinesAdapter, compile_grammar_mask

    gate = QASMGrammarGate()
    adapter = compile_grammar_mask(gate, backend="outlines")
    assert isinstance(adapter, OutlinesAdapter)
    regex = adapter.compile_to_outlines()
    assert isinstance(regex, str) and regex, "regex must be non-empty"
    # Header keywords should appear in the regex.
    for kw in ("OPENQASM", "stdgates", "qubit", "bit", "measure", "barrier"):
        assert kw in regex, f"regex missing keyword {kw!r}"
    # The actual generation path requires outlines; only assert availability.
    assert adapter.is_available() == ("outlines" in __import__("sys").modules
                                       or _outlines_importable())


def _outlines_importable() -> bool:
    try:
        import outlines  # noqa: F401
        return True
    except ImportError:
        return False


def test_grammar_masked_provider_fallback_passes_clean_qasm():
    """GrammarMaskedProvider in fallback (α) mode produces grammar-valid QASM
    for a Bell prompt routed through MockProvider. Metadata records mode."""
    from qmesh.ai.constrained_decoding import QASMGrammarGate, validate_qasm
    from qmesh.ai.grammar_mask import GrammarMaskedProvider
    from qmesh.ai.llm import MockProvider

    gate = QASMGrammarGate()
    masked = GrammarMaskedProvider(
        gate=gate, provider=MockProvider(), streaming=False,
    )
    assert masked.maturity == "alpha"
    assert "grammar_masked[fallback]" in masked.name

    out = masked.complete("Build a Bell state")
    assert "OPENQASM" in out.text
    ok, errors = validate_qasm(out.text)
    assert ok, f"GrammarMaskedProvider emitted invalid QASM: {errors}"
    assert out.metadata["grammar_mask"] == "fallback"
    assert out.metadata["n_lines"] >= 4
    # Mock provider returns clean QASM, so no repair was needed.
    assert out.metadata["n_repairs"] == 0
    assert out.metadata["rejected_at"] is None


def test_grammar_masked_provider_repairs_a_bad_line():
    """Drive the fallback path with a synthetic provider that emits an
    illegal first body line, then a clean repair on the retry. The wrapper
    must accept the repair and grow the kept-lines list past it."""
    import math
    from qmesh.ai.constrained_decoding import QASMGrammarGate, validate_qasm
    from qmesh.ai.grammar_mask import GrammarMaskedProvider
    from qmesh.ai.llm import CompletionResult, LLMProvider

    bad_then_good = [
        # First call: header + decls valid, then a bogus gate line.
        (
            "OPENQASM 3.0;\n"
            'include "stdgates.inc";\n'
            "qubit[2] q;\nbit[2] c;\n"
            "ZAPS q[0];\n"   # rejected: ZAPS is not a real gate
            "h q[0];\ncx q[0], q[1];\n"
        ),
        # Second call (repair): emit valid Bell tail.
        (
            "h q[0];\ncx q[0], q[1];\n"
            "c[0] = measure q[0];\nc[1] = measure q[1];\n"
        ),
    ]

    class _ScriptedProvider(LLMProvider):
        @property
        def name(self): return "scripted"
        def __init__(self): self.calls = 0
        def complete(self, prompt, *, system=None, max_tokens=1024,
                     temperature=0.0):
            text = bad_then_good[self.calls]
            self.calls += 1
            return CompletionResult(
                text=text, provider="scripted", model="x",
                metadata={"call": self.calls},
            )

    gate = QASMGrammarGate()
    masked = GrammarMaskedProvider(
        gate=gate, provider=_ScriptedProvider(), streaming=False, max_repairs=1,
    )
    out = masked.complete("Bell state please")
    ok, errs = validate_qasm(out.text)
    assert ok, f"after repair, masked output should be valid: {errs}"
    assert out.metadata["n_repairs"] == 1
    assert out.metadata["rejected_at"] == 4         # header + 3 decls + bad gate at idx 4
    # The kept output must include both Hadamard and CX from the repair.
    assert "h q[0]" in out.text
    assert "cx q[0], q[1]" in out.text


def test_grammar_masked_provider_streaming_calls_provider_per_line():
    """Streaming mode requests one line at a time and stops after measurement."""
    from qmesh.ai.constrained_decoding import QASMGrammarGate
    from qmesh.ai.grammar_mask import GrammarMaskedProvider
    from qmesh.ai.llm import CompletionResult, LLMProvider

    bell_lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "qubit[2] q;",
        "bit[2] c;",
        "h q[0];",
        "cx q[0], q[1];",
        "c[0] = measure q[0];",
        "c[1] = measure q[1];",
    ]

    class _LineProvider(LLMProvider):
        @property
        def name(self): return "lineprov"
        def __init__(self): self.calls = 0
        def complete(self, prompt, *, system=None, max_tokens=1024,
                     temperature=0.0, stop=None):
            line = bell_lines[self.calls] if self.calls < len(bell_lines) else ""
            self.calls += 1
            return CompletionResult(
                text=line + "\n", provider="lineprov", model="x", metadata={},
            )

    gate = QASMGrammarGate()
    p = _LineProvider()
    masked = GrammarMaskedProvider(
        gate=gate, provider=p, streaming=True, max_lines=20,
    )
    assert masked.maturity == "beta"
    out = masked.complete("Bell state")
    assert out.metadata["grammar_mask"] == "streaming"
    assert out.metadata["n_lines"] == 8
    # Provider must have been polled once per line emitted (no repair needed).
    assert p.calls == 8
    assert "c[0] = measure q[0]" in out.text


def test_grammar_mask_compile_grammar_mask_rejects_unknown_backend():
    from qmesh.ai.constrained_decoding import QASMGrammarGate
    from qmesh.ai.grammar_mask import compile_grammar_mask
    with pytest.raises(ValueError, match="unknown grammar-mask backend"):
        compile_grammar_mask(QASMGrammarGate(), backend="xyz")


# ---------- Phase 5γ: novel-prompt benchmark suite ----------

def test_quanbench_novel_suite_has_at_least_8_prompts():
    """NOVEL_SUITE has 8+ prompts, each with a callable expected_property."""
    from qmesh.ai.quanbench import NOVEL_SUITE
    assert len(NOVEL_SUITE) >= 8, \
        f"NOVEL_SUITE too small: {len(NOVEL_SUITE)}"
    for prompt in NOVEL_SUITE:
        assert prompt.expected_property is not None, \
            f"prompt missing expected_property: {prompt.prompt[:60]!r}"
        assert callable(prompt.expected_property), \
            f"expected_property must be callable: {prompt.prompt[:60]!r}"
        assert prompt.expected_pattern is None, \
            "novel prompts should fall through the rule-based intent compiler"


def test_quanbench_novel_suite_falls_through_intent_compiler():
    """Sanity check: the rule-based intent compiler must miss every novel
    prompt; otherwise the suite isn't actually exercising the LLM path."""
    from qmesh.ai.intent import compile_intent
    from qmesh.ai.quanbench import NOVEL_SUITE
    for p in NOVEL_SUITE:
        assert compile_intent(p.prompt) is None, \
            f"intent compiler unexpectedly matched: {p.prompt[:60]!r}"


def test_quanbench_combined_returns_both_breakdowns():
    """quanbench_combined returns a 'builtin' and a 'novel' report, each
    with success_rate and per_pattern populated."""
    from qmesh.ai.quanbench import quanbench_combined
    report = quanbench_combined(provider="mock")
    assert "builtin" in report and "novel" in report
    for key in ("builtin", "novel"):
        sub = report[key]
        for needed in ("success_rate", "per_pattern", "n_prompts", "results"):
            assert needed in sub, f"{key!r} report missing {needed!r}"
        assert sub["per_pattern"], f"{key!r} per_pattern empty"
    assert "combined" in report
    combined = report["combined"]
    assert combined["n_prompts"] == (
        report["builtin"]["n_prompts"] + report["novel"]["n_prompts"]
    )


def test_quanbench_novel_suite_with_mock_provider_has_realistic_failure_rate():
    """NOVEL_SUITE under MockProvider should score well below 100% — the
    mock falls back to a Bell stub which can't satisfy these properties.
    A real LLM should improve on this number; the assertion proves the
    suite is actually hard."""
    from qmesh.ai.quanbench import NOVEL_SUITE, run_quanbench
    report = run_quanbench(provider="mock", suite=NOVEL_SUITE)
    assert report["success_rate"] < 0.5, (
        f"NOVEL_SUITE looks too easy under mock provider: "
        f"success_rate={report['success_rate']:.2f}; "
        f"expected < 0.5 to demonstrate the suite genuinely needs an LLM."
    )


def test_quanbench_run_quanbench_default_path_uses_builtin_suite():
    """run_quanbench(provider='mock', suite=NOVEL_SUITE) is testable as a
    distinct invocation — exercises the suite= kwarg."""
    from qmesh.ai.quanbench import NOVEL_SUITE, run_quanbench
    report = run_quanbench(provider="mock", suite=NOVEL_SUITE)
    assert report["n_prompts"] == len(NOVEL_SUITE)
    # All entries should have property_checked True (or compiled=False).
    for r in report["results"]:
        if r["compiled"]:
            assert r["property_checked"], \
                "novel-suite results should report property_checked=True"
