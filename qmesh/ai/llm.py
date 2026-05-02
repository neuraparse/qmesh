"""qmesh.ai.llm — LLM provider abstraction.

Supports four providers, picked automatically based on environment:

  1. Ollama       (QMESH_OLLAMA_URL set, e.g. http://localhost:11434)
  2. Anthropic    (ANTHROPIC_API_KEY set)
  3. OpenAI       (OPENAI_API_KEY set)
  4. Mock         (no keys; deterministic rule-based fallback)

The mock provider is non-trivial: it runs the rule-based `intent` compiler
and emits the corresponding QASM 3 source. This means the AI surface is
*always* available — even on a fresh box without API keys — for
tutorials, tests, and CI.

Each provider exposes `complete(prompt: str, **kwargs) -> str`. The
provider's identity (name, model, version) is captured per-draft and
embedded in the resulting Module's metadata so generations are auditable.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CompletionResult:
    text: str
    provider: str
    model: str
    metadata: dict[str, Any]


class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> CompletionResult: ...


class MockProvider(LLMProvider):
    """Rule-based fallback. Routes through the intent compiler when the
    prompt matches a known pattern; otherwise emits a stub QASM file."""

    @property
    def name(self) -> str:
        return "mock"

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> CompletionResult:
        from qmesh.ai.intent import compile_intent
        result = compile_intent(prompt)
        if result is not None:
            from qmesh.frontends.qasm3 import emit
            qasm = emit(result.module)
            return CompletionResult(
                text=qasm, provider="mock", model="intent-rules-v0",
                metadata={"source": "intent_compiler", "pattern": result.pattern},
            )
        # Generic fallback — emit an empty Bell pair as a safe default
        return CompletionResult(
            text=(
                "OPENQASM 3.0;\n"
                'include "stdgates.inc";\n'
                "qubit[2] q;\nbit[2] c;\n"
                "h q[0];\ncx q[0], q[1];\n"
                "c[0] = measure q[0];\nc[1] = measure q[1];\n"
            ),
            provider="mock", model="intent-rules-v0",
            metadata={"source": "fallback_bell", "warning": "no pattern match"},
        )


class OllamaProvider(LLMProvider):
    def __init__(self, *, url: str | None = None, model: str = "qwen2.5-coder:7b") -> None:
        self.url = url or os.getenv("QMESH_OLLAMA_URL", "http://localhost:11434")
        self.model = model

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> CompletionResult:
        import json
        try:
            import urllib.request
        except ImportError as e:
            raise RuntimeError("urllib unavailable") from e

        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            body["system"] = system
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return CompletionResult(
            text=data.get("response", ""),
            provider="ollama", model=self.model,
            metadata={"eval_count": data.get("eval_count"),
                      "duration_ns": data.get("total_duration")},
        )


class AnthropicProvider(LLMProvider):
    def __init__(self, *, api_key: str | None = None,
                 model: str = "claude-sonnet-4-6") -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> CompletionResult:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed: `pip install anthropic`") from e
        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return CompletionResult(
            text=text, provider="anthropic", model=self.model,
            metadata={"input_tokens": resp.usage.input_tokens,
                      "output_tokens": resp.usage.output_tokens,
                      "stop_reason": resp.stop_reason},
        )


class OpenAIProvider(LLMProvider):
    def __init__(self, *, api_key: str | None = None,
                 model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> CompletionResult:
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("openai SDK not installed: `pip install openai`") from e
        client = openai.OpenAI(api_key=self.api_key)
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature, messages=msgs,
        )
        text = resp.choices[0].message.content or ""
        return CompletionResult(
            text=text, provider="openai", model=self.model,
            metadata={"input_tokens": resp.usage.prompt_tokens if resp.usage else None,
                      "output_tokens": resp.usage.completion_tokens if resp.usage else None,
                      "finish_reason": resp.choices[0].finish_reason},
        )


def auto_provider() -> LLMProvider:
    """Pick the best available provider based on environment."""
    if os.getenv("QMESH_OLLAMA_URL"):
        return OllamaProvider()
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return AnthropicProvider()
        except RuntimeError:
            pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIProvider()
        except RuntimeError:
            pass
    return MockProvider()


__all__ = [
    "CompletionResult",
    "LLMProvider",
    "MockProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "auto_provider",
]
