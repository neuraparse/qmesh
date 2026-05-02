"""qmesh.ai — LLM copilot, intent compiler, neural-decoder training.

Phase 5α public API:

    from qmesh.ai import draft, train_neural_decoder
    from qmesh.ai.intent import compile_intent
    from qmesh.ai.llm import auto_provider, MockProvider, AnthropicProvider

The mock provider routes through the rule-based intent compiler so the
copilot is *always* available — no API keys required for tutorials, tests,
or CI. With ANTHROPIC_API_KEY / OPENAI_API_KEY / QMESH_OLLAMA_URL set,
the real LLM is used and the response is QASM-3-validated against the
qmesh.aer simulator before being returned.
"""

from __future__ import annotations

from qmesh.ai.copilot import DraftResult, draft
from qmesh.ai.intent import IntentResult, compile_intent
from qmesh.ai.llm import (
    AnthropicProvider,
    CompletionResult,
    LLMProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    auto_provider,
)
from qmesh.ai.constrained_decoding import (
    ConstrainedQASMGenerator,
    QASMGrammarGate,
    validate_qasm,
    validate_qasm_streaming,
)
from qmesh.ai.grammar_mask import (
    LogitBiasAdapter,
    OutlinesAdapter,
    StubAdapter,
    compile_grammar_mask,
)
from qmesh.ai.quanbench import (
    BUILTIN_SUITE,
    NOVEL_SUITE,
    BenchPrompt,
    QuanBenchPrompt,
    quanbench_combined,
    run_quanbench,
)
from qmesh.ai.train_decoder import (
    TrainingResult,
    generate_dataset,
    train_neural_decoder,
)
from qmesh.ai.transformer_decoder import (
    load_transformer_decoder,
    train_transformer_decoder,
)

__all__ = [
    "DraftResult",
    "draft",
    "IntentResult",
    "compile_intent",
    "LLMProvider",
    "CompletionResult",
    "MockProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "auto_provider",
    "TrainingResult",
    "generate_dataset",
    "train_neural_decoder",
    "train_transformer_decoder",
    "load_transformer_decoder",
    "validate_qasm",
    "validate_qasm_streaming",
    "QASMGrammarGate",
    "ConstrainedQASMGenerator",
    "BenchPrompt",
    "QuanBenchPrompt",
    "BUILTIN_SUITE",
    "NOVEL_SUITE",
    "run_quanbench",
    "quanbench_combined",
    "StubAdapter",
    "LogitBiasAdapter",
    "OutlinesAdapter",
    "compile_grammar_mask",
]
