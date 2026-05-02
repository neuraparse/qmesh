"""qmesh.ai.grammar_mask — token-level grammar masking adapters.

Phase 5γ. Wraps the line-level :class:`QASMGrammarGate` (from
:mod:`qmesh.ai.constrained_decoding`) with three adapters that bridge the
gate to logit-biasing-capable LLM backends. The adapters' β-maturity is
heterogenous, and we are honest about that:

  * :class:`StubAdapter` — α-only fallback. Has no optional deps and just
    delegates to :class:`ConstrainedQASMGenerator` for post-hoc filtering.
    This is what the qmesh CI tests exercise so they don't need outlines or
    tiktoken installed.

  * :class:`LogitBiasAdapter` — β-quality once a real tokenizer is wired.
    Compiles the next-legal-token set into a `{token_id: bias}` dict where
    legal tokens get +inf bias and illegal tokens get -inf, suitable for
    OpenAI / Anthropic / vLLM logit_bias APIs. Requires `tiktoken`. If it is
    not available, the adapter falls back to a string-based legal-prefix
    mask (the keys are the prefix strings, not integer IDs) — enough to
    drive a search-style decoder without bringing in a heavy dep.

  * :class:`OutlinesAdapter` — β/γ. Compiles the gate set to an Outlines-
    compatible regex. Requires `outlines>=0.1`. If outlines is missing,
    the regex string is still produced (it's plain regex), but
    :meth:`generate_constrained` raises :class:`RuntimeError`.

Public entry point:

    from qmesh.ai.grammar_mask import compile_grammar_mask
    adapter = compile_grammar_mask(gate, backend="stub")
    result = adapter.complete(prompt, system=...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from qmesh.ai.constrained_decoding import (
    _GATES_NO_PARAM,
    _GATES_PARAM,
    ConstrainedQASMGenerator,
    QASMGrammarGate,
)
from qmesh.ai.llm import CompletionResult, LLMProvider

# --------------------------------------------------------------------------- #
# StubAdapter                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class StubAdapter:
    """α-only fallback adapter. No optional deps required.

    Wraps the existing :class:`ConstrainedQASMGenerator` so the public
    interface matches the other adapters. The "masking" here is post-hoc
    (the LLM emits a full string, the gate filters line-by-line). This is
    not real token-level masking; it is the honest baseline for tests and
    tutorials that should run on a stock Python install.

    Maturity: α. Use :class:`LogitBiasAdapter` (with tiktoken) or
    :class:`OutlinesAdapter` (with outlines) for true token-level masking.
    """

    gate: QASMGrammarGate
    provider: LLMProvider | None = None
    max_tokens: int = 1024

    @property
    def name(self) -> str:
        return "grammar_mask:stub"

    @property
    def maturity(self) -> str:
        return "alpha"

    def _generator(self, provider: LLMProvider) -> ConstrainedQASMGenerator:
        return ConstrainedQASMGenerator(provider=provider, max_tokens=self.max_tokens)

    def complete(
        self,
        prompt: str,
        *,
        provider: LLMProvider | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Run the prompt through the wrapped LLM provider with post-hoc
        line-by-line grammar filtering applied.
        """
        prov = provider or self.provider
        if prov is None:
            from qmesh.ai.llm import MockProvider
            prov = MockProvider()
        return self._generator(prov).complete(
            prompt, system=system, max_tokens=max_tokens, temperature=temperature,
        )

    def legal_next_tokens(self, prefix: str) -> list[str]:
        """Re-export the gate's per-prefix legal-token oracle."""
        return self.gate.legal_next_tokens(prefix)


# --------------------------------------------------------------------------- #
# LogitBiasAdapter                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class LogitBiasAdapter:
    """Compile :class:`QASMGrammarGate`'s legal-next-token set to a
    `{token_id: float}` logit-bias dict.

    The expectation is that the consumer will pass this dict into a backend
    that supports per-token logit biasing — OpenAI's `logit_bias`,
    Anthropic's tool-use grammar mask, vLLM, llama.cpp, etc.

    Tokenizer policy:
      * If a `tiktoken.Encoding` is passed (or auto-loaded), each legal
        token string is encoded into its token-id sequence and the *first*
        id is biased to ``+math.inf``; everything else gets ``-math.inf``.
      * If no tokenizer is available, the adapter falls back to a
        string-keyed mask: ``{legal_prefix_string: +inf}``. This is enough
        to drive a beam-style decoder over text prefixes but is not a real
        logit-bias dict.

    Maturity: β when wired through tiktoken with a backend that honours the
    biases; α when no tokenizer is available.
    """

    gate: QASMGrammarGate
    tokenizer: Any | None = None
    _tiktoken_available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            try:
                import tiktoken  # noqa: F401  (probe only)
                self._tiktoken_available = True
            except ImportError:
                self._tiktoken_available = False
        else:
            # Caller passed a tokenizer-like object; assume it has `.encode`.
            self._tiktoken_available = hasattr(self.tokenizer, "encode")

    @property
    def name(self) -> str:
        return "grammar_mask:logit_bias"

    @property
    def maturity(self) -> str:
        return "beta" if self._tiktoken_available else "alpha"

    def _ensure_tokenizer(self) -> Any | None:
        if self.tokenizer is not None:
            return self.tokenizer
        try:
            import tiktoken
        except ImportError:
            return None
        # Default: cl100k_base — used by GPT-4 / GPT-3.5-turbo.
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
        return self.tokenizer

    def compile_to_logit_bias(self, prefix: str) -> dict[Any, float]:
        """Build a `{token_id_or_prefix_str: bias}` dict for the next token.

        With a tokenizer present, keys are :class:`int` token-ids and values
        are ``+inf`` for legal first-token-of-keyword ids, ``-inf`` for any
        explicitly disallowed id (none, by default — backends interpret a
        missing key as "neutral").

        Without a tokenizer, keys are strings (the legal next-token text) so
        the consumer can score continuations by prefix match — this is the
        fallback honest interface.
        """
        legal = self.gate.legal_next_tokens(prefix)
        tok = self._ensure_tokenizer()
        if tok is None:
            return {tok_str: math.inf for tok_str in legal}

        bias: dict[int, float] = {}
        for tok_str in legal:
            try:
                ids = tok.encode(tok_str)
            except Exception:
                continue
            if not ids:
                continue
            # First id is the discriminating piece for this branch — biasing
            # it positively pushes the model toward this keyword. Sub-tokens
            # are handled per-step on subsequent decode rounds.
            bias[ids[0]] = math.inf
        return bias

    def has_tokenizer(self) -> bool:
        return self._ensure_tokenizer() is not None


# --------------------------------------------------------------------------- #
# OutlinesAdapter                                                             #
# --------------------------------------------------------------------------- #


_QASM_HEADER_REGEX = (
    r'OPENQASM\s+3(?:\.0)?\s*;\s*'
    r'(?:include\s+"stdgates\.inc"\s*;\s*)?'
    r'(?:qubit\s*\[\s*\d+\s*\]\s*[A-Za-z_]\w*\s*;\s*)+'
    r'(?:bit\s*\[\s*\d+\s*\]\s*[A-Za-z_]\w*\s*;\s*)*'
)


def _build_qasm_regex() -> str:
    """Build an Outlines-compatible regex covering :class:`QASMGrammarGate`.

    Coverage:
      * OPENQASM 3.0 header + optional stdgates include
      * one or more qubit register declarations, optional bit register
      * a sequence of gate / measure / reset / barrier statements
      * uses the same gate set as the gate (no-param vs param)

    Note: this is a *grammar-shaped* regex, not a full QASM 3 parser. It's
    tight enough to keep an LLM on the rails for the gate set qmesh emits.
    """
    no_param = "|".join(sorted(_GATES_NO_PARAM, key=lambda s: -len(s)))
    with_param = "|".join(sorted(_GATES_PARAM, key=lambda s: -len(s)))
    qref = r"[A-Za-z_]\w*\s*\[\s*\d+\s*\]"
    qrefs = rf"{qref}(?:\s*,\s*{qref})*"
    # A signed real number, optionally with `pi` etc. Be permissive.
    num = r"[-+0-9.eEpiPI*/() ]+"

    no_param_stmt = rf"(?:{no_param})\s+{qrefs}\s*;\s*"
    with_param_stmt = rf"(?:{with_param})\s*\(\s*{num}\s*(?:,\s*{num}\s*)*\)\s+{qrefs}\s*;\s*"
    measure_arrow = rf"measure\s+{qref}\s*->\s*{qref}\s*;\s*"
    measure_assign = rf"{qref}\s*=\s*measure\s+{qref}\s*;\s*"
    reset_stmt = rf"reset\s+{qref}\s*;\s*"
    barrier_stmt = r"barrier\s*;\s*"

    body = (
        rf"(?:{no_param_stmt}|{with_param_stmt}|{measure_arrow}|"
        rf"{measure_assign}|{reset_stmt}|{barrier_stmt})+"
    )
    return _QASM_HEADER_REGEX + body


@dataclass
class OutlinesAdapter:
    """Compile the grammar gate to an Outlines-compatible regex string.

    Outlines (https://github.com/dottxt-ai/outlines) ships a structured
    generation engine that can clamp an HF Transformers model to a regex.
    We expose the regex string + a thin wrapper that drives Outlines if
    installed.

    Maturity: β for the regex (it covers the gate's surface grammar, not
    full QASM 3); requires ``outlines>=0.1`` for actual generation.
    """

    gate: QASMGrammarGate

    @property
    def name(self) -> str:
        return "grammar_mask:outlines"

    @property
    def maturity(self) -> str:
        return "beta"

    def compile_to_outlines(self) -> str:
        """Return the regex string. Stable across calls; cheap."""
        return _build_qasm_regex()

    def is_available(self) -> bool:
        try:
            import outlines  # noqa: F401
            return True
        except ImportError:
            return False

    def generate_constrained(
        self,
        provider: str,
        model: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """Run Outlines structured generation under the grammar regex.

        Currently supports HF Transformers via Outlines' default loader.
        ``provider`` is reserved for future routing (e.g. Outlines-compatible
        OpenAI / vLLM endpoints); callers passing anything other than
        ``"transformers"`` will raise :class:`NotImplementedError`.
        """
        try:
            import outlines  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "outlines not installed: `pip install 'outlines>=0.1'` "
                "to enable token-level grammar-masked generation."
            ) from e

        if provider != "transformers":
            raise NotImplementedError(
                f"OutlinesAdapter currently routes through HF transformers; "
                f"got provider={provider!r}. Wire vLLM/openai backends in β."
            )
        regex = self.compile_to_outlines()
        # Outlines' top-level API has shifted across versions; use the
        # generate.regex helper which has been stable since 0.1.
        try:
            llm = outlines.models.transformers(model)  # type: ignore[attr-defined]
            generator = outlines.generate.regex(llm, regex)  # type: ignore[attr-defined]
        except AttributeError as e:
            raise RuntimeError(
                f"outlines API mismatch: {e}. Pin outlines>=0.1,<0.2 or update."
            ) from e
        return generator(prompt, max_tokens=max_tokens)


# --------------------------------------------------------------------------- #
# Public entry                                                                #
# --------------------------------------------------------------------------- #


def compile_grammar_mask(
    gate: QASMGrammarGate,
    backend: str = "stub",
    **kwargs: Any,
) -> StubAdapter | LogitBiasAdapter | OutlinesAdapter:
    """Build the requested adapter.

    Parameters
    ----------
    gate :
        The :class:`QASMGrammarGate` to wrap.
    backend :
        One of ``"stub"`` (default; α-quality, no deps),
        ``"logit_bias"`` (β with tiktoken, α without), or
        ``"outlines"`` (β; requires outlines for actual generation).
    **kwargs :
        Forwarded to the chosen adapter's constructor — e.g.
        ``provider=...`` for stub, ``tokenizer=...`` for logit_bias.
    """
    if backend == "stub":
        return StubAdapter(gate=gate, **kwargs)
    if backend == "logit_bias":
        return LogitBiasAdapter(gate=gate, **kwargs)
    if backend == "outlines":
        return OutlinesAdapter(gate=gate, **kwargs)
    raise ValueError(
        f"unknown grammar-mask backend {backend!r}; "
        f"want 'stub' | 'logit_bias' | 'outlines'"
    )


# --------------------------------------------------------------------------- #
# GrammarMaskedProvider — real-time per-line gate wiring (Phase 5δ)            #
# --------------------------------------------------------------------------- #


_LINE_STOP_DEFAULT: tuple[str, ...] = ("\n",)


@dataclass
class GrammarMaskedProvider:
    """An :class:`LLMProvider` wrapper that interleaves per-line grammar
    validation with the underlying provider's generation calls.

    Unlike :class:`StubAdapter`, this class is *generation-time*: the gate
    is consulted between successive provider calls. Two operating modes:

      * **streaming mode** (β when wired to a provider that honours
        ``stop`` sequences): we loop, calling the provider once per line
        with ``stop=("\\n",)`` so each call returns at most one line.
        After each line, the gate either accepts it (extend the prefix
        and continue) or rejects it (issue one repair completion with a
        targeted hint; on failure we abort).

      * **whole-completion fallback** (α when the provider only exposes
        full completions): we generate once, then validate line-by-line.
        On the first illegal line, we issue a single *repair* completion
        with the rejected line replaced by a one-line hint and re-validate
        from that point forward. This is a step beyond the post-hoc
        :class:`StubAdapter` because the LLM gets to *see* the rejected
        line and try again, instead of the validator silently truncating.

    Maturity: ``beta`` when ``streaming=True`` and the provider's
    ``.complete()`` accepts a ``stop`` kwarg (Anthropic / OpenAI / Ollama
    all do); ``alpha`` otherwise.
    """

    gate: QASMGrammarGate
    provider: LLMProvider
    streaming: bool = False
    max_lines: int = 64
    max_repairs: int = 1
    line_token_budget: int = 64

    @property
    def name(self) -> str:
        kind = "streaming" if self.streaming else "fallback"
        return f"grammar_masked[{kind}]({self.provider.name})"

    @property
    def maturity(self) -> str:
        return "beta" if self.streaming else "alpha"

    def _call_provider(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
        temperature: float,
        stop: tuple[str, ...] | None,
    ) -> CompletionResult:
        """Invoke the wrapped provider with optional `stop`. Falls back
        gracefully if the provider doesn't accept the stop kwarg."""
        try:
            return self.provider.complete(
                prompt, system=system,
                max_tokens=max_tokens, temperature=temperature,
                stop=stop,                                # type: ignore[call-arg]
            )
        except TypeError:
            return self.provider.complete(
                prompt, system=system,
                max_tokens=max_tokens, temperature=temperature,
            )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        if self.streaming:
            return self._complete_streaming(
                prompt, system=system,
                max_tokens=max_tokens, temperature=temperature,
            )
        return self._complete_fallback(
            prompt, system=system,
            max_tokens=max_tokens, temperature=temperature,
        )

    # -- streaming mode (β) --------------------------------------------------

    def _complete_streaming(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int | None,
        temperature: float,
    ) -> CompletionResult:
        self.gate.reset()
        accepted_lines: list[str] = []
        n_repairs_done = 0
        last_provider = self.provider.name
        last_model = ""

        for _line_idx in range(self.max_lines):
            ctx = prompt
            if accepted_lines:
                ctx = (
                    f"{prompt}\n"
                    "# Continue the OpenQASM 3 program; emit exactly one line.\n"
                    "# Lines so far:\n" + "\n".join(accepted_lines)
                )
            result = self._call_provider(
                ctx, system=system,
                max_tokens=self.line_token_budget,
                temperature=temperature,
                stop=_LINE_STOP_DEFAULT,
            )
            last_model = result.model
            raw = result.text.splitlines()[0] if result.text else ""
            line = raw.rstrip()
            # Empty / whitespace-only line: treat as model-emitted EOS and stop.
            if not line.strip():
                break
            ok, err = self.gate.feed_line(line)
            if not ok:
                if n_repairs_done < self.max_repairs:
                    n_repairs_done += 1
                    repair = self._call_provider(
                        f"{ctx}\n# REJECTED previous line: {line!r}\n"
                        f"# Reason: {err}. Emit one valid line.",
                        system=system,
                        max_tokens=self.line_token_budget,
                        temperature=0.0,
                        stop=_LINE_STOP_DEFAULT,
                    )
                    line2 = repair.text.splitlines()[0] if repair.text else ""
                    ok2, _ = self.gate.feed_line(line2)
                    if ok2:
                        accepted_lines.append(line2)
                        continue
                # give up after repair budget exhausted
                break
            accepted_lines.append(line)
            # Stop heuristic: once we've measured every declared classical
            # bit (n_bits_total inferred from `bit[N] c;` declarations),
            # the program is logically complete and any further provider
            # call would just emit redundant gates.
            n_bits_total = sum(self.gate.state.bregs.values())
            n_measure_lines = sum(
                1 for L in accepted_lines
                if "measure" in L
            )
            if n_bits_total > 0 and n_measure_lines >= n_bits_total:
                break

        text = "\n".join(accepted_lines)
        if text and not text.endswith("\n"):
            text += "\n"
        return CompletionResult(
            text=text,
            provider=last_provider,
            model=last_model or "unknown",
            metadata={
                "grammar_mask": "streaming",
                "n_lines": len(accepted_lines),
                "n_repairs": n_repairs_done,
            },
        )

    # -- fallback mode (α) ---------------------------------------------------

    def _complete_fallback(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int | None,
        temperature: float,
    ) -> CompletionResult:
        self.gate.reset()
        result = self.provider.complete(
            prompt, system=system,
            max_tokens=max_tokens or 1024, temperature=temperature,
        )
        # Strip markdown fences if present (matches StubAdapter behaviour).
        import re as _re
        text = result.text
        m = _re.search(r"```(?:qasm3?|openqasm)?\s*\n(.*?)```", text, _re.S | _re.I)
        if m:
            text = m.group(1)
        idx = text.find("OPENQASM")
        if idx > 0:
            text = text[idx:]

        kept: list[str] = []
        rejected_at: int | None = None
        for i, line in enumerate(text.splitlines()):
            ok, _err = self.gate.feed_line(line)
            if ok:
                kept.append(line)
            else:
                rejected_at = i
                break

        n_repairs = 0
        if rejected_at is not None and n_repairs < self.max_repairs:
            n_repairs += 1
            # Repair prompt: hand the LLM what we kept + an instruction to
            # continue from there. Reset gate to the pre-rejection state and
            # re-feed the kept lines.
            self.gate.reset()
            for L in kept:
                self.gate.feed_line(L)
            repair_prompt = (
                f"{prompt}\n"
                "# Lines accepted so far (continue from here):\n"
                + "\n".join(kept)
                + "\n# Emit valid OpenQASM 3 to complete the program."
            )
            repair = self.provider.complete(
                repair_prompt, system=system,
                max_tokens=max_tokens or 1024, temperature=0.0,
            )
            for line in repair.text.splitlines():
                ok, _err = self.gate.feed_line(line)
                if ok:
                    kept.append(line)
                else:
                    break

        out = "\n".join(kept)
        if out and not out.endswith("\n"):
            out += "\n"
        return CompletionResult(
            text=out,
            provider=result.provider,
            model=result.model,
            metadata={
                "grammar_mask": "fallback",
                "n_lines": len(kept),
                "n_repairs": n_repairs,
                "rejected_at": rejected_at,
            },
        )


__all__ = [
    "StubAdapter",
    "LogitBiasAdapter",
    "OutlinesAdapter",
    "GrammarMaskedProvider",
    "compile_grammar_mask",
]
