"""LLM decision layer: explanation and a one-way veto. Never a decision-maker."""

from app.ai_decision.llm_layer import (
    LLMDecisionLayer,
    LLMReview,
    apply_verdict,
    build_prompt,
)
from app.ai_decision.providers import (
    ClaudeProvider,
    NullVetoBaseline,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
    ProviderInfo,
    ScriptedProvider,
    build_candidates,
)
from app.ai_decision.schema import (
    LLMAction,
    LLMVerdict,
    ValidationError,
    validate_response,
)

__all__ = [
    "LLMDecisionLayer",
    "LLMReview",
    "apply_verdict",
    "build_prompt",
    "Provider",
    "ProviderInfo",
    "ProviderError",
    "ClaudeProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "NullVetoBaseline",
    "ScriptedProvider",
    "build_candidates",
    "LLMAction",
    "LLMVerdict",
    "ValidationError",
    "validate_response",
]
