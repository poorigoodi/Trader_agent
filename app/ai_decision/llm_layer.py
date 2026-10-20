"""The LLM layer: explanation, and a one-way veto. Nothing else.

Where this sits in the pipeline::

    Strategy (deterministic)  ->  [ LLM layer ]  ->  Risk (deterministic)
                                        |
                        may only: annotate, or downgrade to NO_TRADE

The layer is structurally incapable of doing more than that, and that is
enforced in code rather than by prompt instruction:

  * ``apply_verdict`` accepts a direction and a verdict and returns either
    the SAME direction or NO_TRADE. There is no branch that produces any
    other value, so a model cannot flip LONG to SHORT even if it tries.
  * A verdict on a NO_TRADE signal cannot resurrect it: NO_TRADE in, NO_TRADE
    out, whatever the model says.
  * The layer never sees or returns position size, stop loss or take profit.
    It runs BEFORE the risk module and has no access to its inputs.

Failure is always safe. A timeout, a malformed response, an unknown field, a
provider outage -- every one of them falls back to the original rule-based
signal, unchanged, and records why. The system without an LLM is a working
system; the LLM is an optional filter on top of it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.ai_decision.providers import Provider, ProviderError
from app.ai_decision.schema import (
    SCHEMA_TEXT,
    LLMAction,
    LLMVerdict,
    ValidationError,
    validate_response,
)
from app.strategies.base import Signal, SignalDirection

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are a risk reviewer for a rule-based crypto trading system.

A deterministic strategy has ALREADY produced the signal below. Your job is \
NOT to trade, and NOT to decide direction. Direction, position size, stop \
loss and take profit are fixed by code you cannot influence.

You may do exactly one of two things:
  - "CONFIRM": the signal stands as it is.
  - "VETO": downgrade this signal to NO_TRADE, because you see a clear,
    specific risk that makes taking it unwise right now.

Veto only for a concrete, identifiable risk. "The market looks uncertain" is \
not a reason. A known hack, an exchange halt or delisting affecting this \
symbol, or an obvious contradiction inside the data below, are reasons. If \
you have no specific reason, CONFIRM.

--- SIGNAL ---
Symbol:        {symbol}
Timeframe:     {timeframe}
Direction:     {direction}
Candle close:  {close}
Candle time:   {timestamp}

Strategy reasoning:
{reasons}

Invalidation condition:
{invalidation}

--- INDICATORS AT DECISION TIME ---
{indicators}

--- CONTEXT ---
{context}

--- RESPONSE FORMAT ---
Reply with ONE JSON object and nothing else. No prose, no markdown fence.
It must match this schema exactly; any extra field is a failure:

{schema}
"""


@dataclass(frozen=True)
class LLMReview:
    """The outcome of running one signal past a model."""

    original_direction: SignalDirection
    final_direction: SignalDirection
    verdict: LLMVerdict | None
    provider: str
    ok: bool
    failure_reason: str | None = None
    latency_seconds: float = 0.0
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def vetoed(self) -> bool:
        return self.original_direction is not self.final_direction

    @property
    def reasoning(self) -> str:
        if self.verdict:
            return self.verdict.reasoning
        return f"LLM layer unavailable ({self.failure_reason}); rule-based signal stands."

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "ok": self.ok,
            "failure_reason": self.failure_reason,
            "original_direction": self.original_direction.value,
            "final_direction": self.final_direction.value,
            "vetoed": self.vetoed,
            "reasoning": self.reasoning,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "latency_seconds": self.latency_seconds,
            "attempts": self.attempts,
            "cost_usd": self.cost_usd,
        }


def apply_verdict(
    direction: SignalDirection, verdict: LLMVerdict | None
) -> SignalDirection:
    """The one-way veto, in three lines and no other branches.

    Returns ``direction`` unchanged, or NO_TRADE. There is deliberately no
    code path that returns any other direction, which is why a model cannot
    flip a trade or invent one regardless of what it emits.
    """
    if direction is SignalDirection.NO_TRADE:
        return SignalDirection.NO_TRADE  # a veto cannot resurrect a signal
    if verdict is not None and verdict.action is LLMAction.VETO:
        return SignalDirection.NO_TRADE
    return direction


def build_prompt(signal: Signal, context: str = "") -> str:
    """Render the review prompt for one signal."""
    snapshot = signal.snapshot
    indicators = json.dumps(
        {
            k: v
            for k, v in snapshot.to_dict().items()
            if k not in {"symbol", "timeframe", "timestamp"}
        },
        indent=2,
        default=str,
    )
    return PROMPT_TEMPLATE.format(
        symbol=snapshot.symbol,
        timeframe=snapshot.timeframe,
        direction=signal.direction.value,
        close=snapshot.close,
        timestamp=snapshot.timestamp,
        reasons="\n".join(f"  - {r}" for r in signal.reasons) or "  (none recorded)",
        invalidation=signal.invalidation,
        indicators=indicators,
        context=context.strip() or "No external context available at this stage "
        "(the news/sentiment layer is roadmap Phase 6 and is not built yet).",
        schema=SCHEMA_TEXT,
    )


class LLMDecisionLayer:
    """Runs a signal past a provider and applies the one-way veto.

    Parameters
    ----------
    provider:
        Any :class:`~app.ai_decision.providers.Provider`.
    max_attempts:
        Retries on a malformed response. One retry is worth it (small models
        sometimes emit prose on the first pass); more is throwing latency at
        a model that cannot follow the schema.
    enabled:
        When False the layer is inert and every signal passes through
        untouched. This is the default state of the system.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        max_attempts: int = 2,
        enabled: bool = True,
        context_provider=None,
    ) -> None:
        self.provider = provider
        self.max_attempts = max(1, max_attempts)
        self.enabled = enabled and provider is not None
        self.context_provider = context_provider

    def review(self, signal: Signal) -> LLMReview:
        """Review one signal. Never raises: failure means the signal stands."""
        direction = signal.direction

        if not self.enabled or self.provider is None:
            return LLMReview(
                original_direction=direction,
                final_direction=direction,
                verdict=None,
                provider="disabled",
                ok=True,
                failure_reason=None,
            )

        # Reviewing a NO_TRADE is pure cost: the only permitted outcome is
        # NO_TRADE either way.
        if direction is SignalDirection.NO_TRADE:
            return LLMReview(
                original_direction=direction,
                final_direction=direction,
                verdict=None,
                provider=self.provider.info.name,
                ok=True,
                failure_reason="skipped: signal is already NO_TRADE",
            )

        available, why = self.provider.available()
        if not available:
            return LLMReview(
                original_direction=direction,
                final_direction=direction,
                verdict=None,
                provider=self.provider.info.name,
                ok=False,
                failure_reason=f"provider unavailable: {why}",
            )

        context = ""
        if self.context_provider is not None:
            try:
                context = self.context_provider(signal) or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("context provider failed: %s", exc)

        prompt = build_prompt(signal, context)
        last_error = "unknown"
        started = time.perf_counter()
        raw = ""

        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.provider.complete(prompt)
                verdict = validate_response(raw)
            except ValidationError as exc:
                last_error = f"schema violation: {exc}"
                logger.warning(
                    "%s attempt %d/%d rejected: %s",
                    self.provider.info.name, attempt, self.max_attempts, exc,
                )
                continue
            except ProviderError as exc:
                last_error = f"provider error: {exc}"
                logger.warning("%s attempt %d failed: %s", self.provider.info.name, attempt, exc)
                continue
            except Exception as exc:  # noqa: BLE001 - the layer must never crash the agent
                last_error = f"unexpected error: {type(exc).__name__}: {exc}"
                logger.exception("LLM layer raised unexpectedly")
                continue

            elapsed = time.perf_counter() - started
            input_tokens = self.provider.estimate_tokens(prompt)
            output_tokens = self.provider.estimate_tokens(raw)
            return LLMReview(
                original_direction=direction,
                final_direction=apply_verdict(direction, verdict),
                verdict=verdict,
                provider=self.provider.info.name,
                ok=True,
                latency_seconds=elapsed,
                attempts=attempt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=self.provider.info.cost_usd(input_tokens, output_tokens),
            )

        # Every attempt failed -> fall back to the rule-based signal.
        return LLMReview(
            original_direction=direction,
            final_direction=direction,  # unchanged, deliberately
            verdict=None,
            provider=self.provider.info.name,
            ok=False,
            failure_reason=last_error,
            latency_seconds=time.perf_counter() - started,
            attempts=self.max_attempts,
        )

    def apply(self, signal: Signal) -> tuple[Signal, LLMReview]:
        """Return (possibly downgraded signal, review).

        The returned signal keeps every original reason and gains the model's
        reasoning as an extra one, so the journal shows both the rule that
        fired and the review that overrode it.
        """
        from dataclasses import replace

        review = self.review(signal)
        if review.final_direction is signal.direction:
            if review.verdict is not None:
                return (
                    replace(
                        signal,
                        metadata={**signal.metadata, "llm": review.to_dict()},
                    ),
                    review,
                )
            return signal, review

        return (
            replace(
                signal,
                direction=review.final_direction,
                reasons=signal.reasons
                + (f"VETOED by {review.provider}: {review.reasoning}",),
                metadata={**signal.metadata, "llm": review.to_dict()},
            ),
            review,
        )
