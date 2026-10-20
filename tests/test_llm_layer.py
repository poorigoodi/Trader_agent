"""Unit tests for the LLM decision layer.

NO NETWORK. Every test uses ScriptedProvider or the deterministic baseline.

The tests fall into two groups, and the first group is the important one:

  SAFETY -- a model can only annotate or veto. It cannot flip a direction,
  cannot create a trade from NO_TRADE, and cannot touch size or levels. These
  are the project's non-negotiable rules, so they are tested as properties
  over every possible model output, not just the well-behaved ones.

  ROBUSTNESS -- malformed output, outages and timeouts must degrade to the
  rule-based signal rather than crash or, worse, be partially believed.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.ai_decision.llm_layer import (
    LLMDecisionLayer,
    apply_verdict,
    build_prompt,
)
from app.ai_decision.providers import (
    NullVetoBaseline,
    ProviderError,
    ScriptedProvider,
)
from app.ai_decision.schema import (
    MAX_REASONING_CHARS,
    LLMAction,
    LLMVerdict,
    ValidationError,
    extract_json,
    validate_response,
)
from app.indicators.engine import IndicatorSnapshot
from app.strategies.base import Signal, SignalDirection

LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT
NO_TRADE = SignalDirection.NO_TRADE


def snapshot(symbol="BTC/USDT:USDT", close=100.0) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        symbol=symbol, timeframe="15m", timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        open=close, high=close + 1, low=close - 1, close=close, volume=5.0,
        ema_fast=101.0, ema_slow=100.0, ema_fast_prev=99.0, ema_slow_prev=100.0,
        sma_slow=100.0, rsi=55.5, atr=2.0, macd=0.1, macd_signal=0.05, macd_hist=0.05,
        bb_upper=105.0, bb_middle=100.0, bb_lower=95.0, bb_percent_b=0.5,
        vwap=99.5, trend="UP",
    )


def signal(direction=LONG) -> Signal:
    return Signal(
        direction=direction,
        reasons=("EMA20 > EMA50", "RSI 55.5 within [50, 70]", "structure UP"),
        invalidation="close back below EMA50",
        snapshot=snapshot(),
        strategy_name="ema_rsi_trend_v1",
        metadata={"cross_age": 1},
    )


def response(action="CONFIRM", reasoning="Looks consistent.", **extra) -> str:
    return json.dumps({"action": action, "reasoning": reasoning, **extra})


def layer(*responses, **kwargs) -> LLMDecisionLayer:
    return LLMDecisionLayer(ScriptedProvider(list(responses)), **kwargs)


# ==========================================================================
# SAFETY: what a model is structurally unable to do
# ==========================================================================

def test_veto_downgrades_long_to_no_trade():
    result, review = layer(response("VETO", "Exchange halt reported.")).apply(signal(LONG))
    assert result.direction is NO_TRADE
    assert review.vetoed


def test_veto_downgrades_short_to_no_trade():
    result, _ = layer(response("VETO", "Listing news invalidates the short.")).apply(
        signal(SHORT)
    )
    assert result.direction is NO_TRADE


def test_confirm_leaves_the_direction_untouched():
    result, review = layer(response("CONFIRM")).apply(signal(LONG))
    assert result.direction is LONG
    assert not review.vetoed


def test_a_veto_cannot_resurrect_a_no_trade():
    """NO_TRADE in, NO_TRADE out -- whatever the model says."""
    result, review = layer(response("CONFIRM", "This looks like a great long!")).apply(
        signal(NO_TRADE)
    )
    assert result.direction is NO_TRADE
    assert not review.vetoed


@pytest.mark.parametrize("direction", [LONG, SHORT, NO_TRADE])
@pytest.mark.parametrize("action", ["CONFIRM", "VETO"])
def test_output_direction_is_always_the_original_or_no_trade(direction, action):
    """The core safety property, over the whole input space: there is no
    model output that produces any third outcome."""
    result, _ = layer(response(action)).apply(signal(direction))
    assert result.direction in (direction, NO_TRADE)


def test_apply_verdict_cannot_flip_long_to_short():
    for action in (LLMAction.CONFIRM, LLMAction.VETO):
        verdict = LLMVerdict(action=action, reasoning="x")
        assert apply_verdict(LONG, verdict) in (LONG, NO_TRADE)
        assert apply_verdict(SHORT, verdict) in (SHORT, NO_TRADE)


def test_a_model_proposing_a_direction_is_rejected_outright():
    """'action': 'LONG' is not a valid verdict; it is a model trying to
    trade. It must fail validation, not be interpreted."""
    with pytest.raises(ValidationError, match="CONFIRM or VETO"):
        validate_response(response("LONG"))
    with pytest.raises(ValidationError):
        validate_response(response("SHORT"))
    with pytest.raises(ValidationError):
        validate_response(response("BUY"))


def test_a_model_returning_size_or_levels_is_rejected():
    """Extra fields are a refusal, not something to quietly ignore: they mean
    the model answered a different question."""
    for field in ("position_size", "stop_loss", "take_profit", "leverage", "quantity"):
        with pytest.raises(ValidationError, match="unexpected field"):
            validate_response(response(**{field: 123}))


def test_the_prompt_never_exposes_size_or_levels():
    """The layer runs BEFORE risk sizing and cannot see those numbers."""
    prompt = build_prompt(signal(LONG))
    for forbidden in ("position_size", "stop_loss", "take_profit", "risk_amount", "balance"):
        assert forbidden not in prompt


def test_the_prompt_states_the_two_permitted_actions():
    prompt = build_prompt(signal(LONG))
    assert "CONFIRM" in prompt and "VETO" in prompt
    assert "NOT to decide direction" in prompt


def test_self_reported_confidence_is_recorded_but_kept_separate():
    """A model's self-assessment is not evidence; the calibrated confidence
    comes from historical outcomes instead."""
    _, review = layer(response(confidence=0.99)).apply(signal(LONG))
    assert review.verdict.self_reported_confidence == pytest.approx(0.99)
    assert "self_reported_confidence" in review.verdict.to_dict()


# ==========================================================================
# ROBUSTNESS: everything that can go wrong, degrading safely
# ==========================================================================

def test_malformed_json_falls_back_to_the_rule_based_signal():
    result, review = layer("this is not json", "still not json").apply(signal(LONG))
    assert result.direction is LONG  # unchanged
    assert not review.ok
    assert "schema violation" in review.failure_reason


def test_a_retry_can_recover_from_a_bad_first_response():
    result, review = layer("garbage", response("VETO", "Hack reported.")).apply(signal(LONG))
    assert result.direction is NO_TRADE
    assert review.ok and review.attempts == 2


def test_provider_outage_falls_back_safely():
    class Broken(ScriptedProvider):
        def complete(self, prompt):
            raise ProviderError("connection refused")

    result, review = LLMDecisionLayer(Broken([])).apply(signal(LONG))
    assert result.direction is LONG
    assert not review.ok and "provider error" in review.failure_reason


def test_an_unexpected_exception_never_reaches_the_caller():
    class Exploding(ScriptedProvider):
        def complete(self, prompt):
            raise ZeroDivisionError("boom")

    result, review = LLMDecisionLayer(Exploding([])).apply(signal(LONG))
    assert result.direction is LONG
    assert not review.ok


def test_an_unavailable_provider_is_not_called():
    class Unavailable(ScriptedProvider):
        def available(self):
            return False, "no API key"

    provider = Unavailable([response("VETO")])
    result, review = LLMDecisionLayer(provider).apply(signal(LONG))
    assert result.direction is LONG
    assert provider.calls == []
    assert "unavailable" in review.failure_reason


def test_a_disabled_layer_is_completely_inert():
    provider = ScriptedProvider([response("VETO")])
    result, review = LLMDecisionLayer(provider, enabled=False).apply(signal(LONG))
    assert result.direction is LONG
    assert provider.calls == []
    assert review.provider == "disabled"


def test_no_provider_at_all_is_inert():
    result, review = LLMDecisionLayer(None).apply(signal(LONG))
    assert result.direction is LONG and review.ok


def test_no_trade_signals_are_not_sent_to_the_model():
    """Reviewing a NO_TRADE is pure cost -- the outcome cannot differ."""
    provider = ScriptedProvider([response("VETO")])
    LLMDecisionLayer(provider).apply(signal(NO_TRADE))
    assert provider.calls == []


def test_a_failing_context_provider_does_not_break_the_review():
    def broken(_signal):
        raise RuntimeError("news feed down")

    instance = LLMDecisionLayer(
        ScriptedProvider([response("CONFIRM")]), context_provider=broken
    )
    _, review = instance.apply(signal(LONG))
    assert review.ok


# ==========================================================================
# schema validation
# ==========================================================================

def test_markdown_fence_is_tolerated():
    verdict = validate_response('```json\n{"action":"VETO","reasoning":"x"}\n```')
    assert verdict.action is LLMAction.VETO


def test_json_embedded_in_prose_is_recovered():
    verdict = validate_response(
        'Sure! Here is my answer:\n{"action": "CONFIRM", "reasoning": "fine"}\nHope that helps.'
    )
    assert verdict.action is LLMAction.CONFIRM


def test_action_is_case_insensitive_and_trimmed():
    assert validate_response(response(" confirm ")).action is LLMAction.CONFIRM


def test_empty_response_is_rejected():
    with pytest.raises(ValidationError, match="empty"):
        validate_response("")


def test_a_json_array_is_rejected():
    with pytest.raises(ValidationError, match="expected a JSON object"):
        validate_response('[{"action":"CONFIRM","reasoning":"x"}]')


def test_missing_reasoning_is_rejected():
    with pytest.raises(ValidationError, match="missing required"):
        validate_response('{"action":"CONFIRM"}')


def test_empty_reasoning_is_rejected():
    with pytest.raises(ValidationError, match="must not be empty"):
        validate_response(response(reasoning="   "))


def test_overlong_reasoning_is_rejected():
    with pytest.raises(ValidationError, match="max"):
        validate_response(response(reasoning="x" * (MAX_REASONING_CHARS + 1)))


def test_non_string_reasoning_is_rejected():
    with pytest.raises(ValidationError, match="reasoning must be a string"):
        validate_response('{"action":"CONFIRM","reasoning":42}')


def test_risk_flags_must_be_a_list_of_strings():
    with pytest.raises(ValidationError, match="must be an array"):
        validate_response(response(risk_flags="not a list"))
    with pytest.raises(ValidationError, match="entries must be strings"):
        validate_response(response(risk_flags=[1, 2]))


def test_risk_flags_are_captured():
    verdict = validate_response(response("VETO", "x", risk_flags=["exchange halt", "low liquidity"]))
    assert verdict.risk_flags == ("exchange halt", "low liquidity")


def test_out_of_range_confidence_is_rejected():
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        validate_response(response(confidence=1.5))


def test_boolean_confidence_is_rejected():
    with pytest.raises(ValidationError, match="must be a number"):
        validate_response(response(confidence=True))


def test_extract_json_reports_when_there_is_no_object():
    with pytest.raises(ValidationError, match="no JSON object"):
        extract_json("absolutely no braces here")


# ==========================================================================
# the baseline control arm
# ==========================================================================

def test_baseline_is_always_available():
    assert NullVetoBaseline().available()[0] is True


def test_baseline_never_vetoes():
    result, review = LLMDecisionLayer(NullVetoBaseline()).apply(signal(LONG))
    assert result.direction is LONG
    assert review.ok and not review.vetoed


def test_baseline_output_passes_the_same_validator():
    """The control arm is held to the same contract as the candidates."""
    assert validate_response(NullVetoBaseline().complete("x")).action is LLMAction.CONFIRM


def test_baseline_costs_nothing():
    assert NullVetoBaseline().info.cost_usd(1_000_000, 1_000_000) == 0.0


# ==========================================================================
# journal payload
# ==========================================================================

def test_review_is_serialisable_for_the_database():
    _, review = layer(response("VETO", "Reported exploit on the bridge.")).apply(signal(LONG))
    payload = review.to_dict()
    assert payload["vetoed"] is True
    assert payload["final_direction"] == "NO_TRADE"
    assert payload["original_direction"] == "LONG"
    assert "Reported exploit" in payload["reasoning"]


def test_a_vetoed_signal_keeps_its_original_reasons():
    """The journal must show both the rule that fired and the override."""
    result, _ = layer(response("VETO", "Delisting announced.")).apply(signal(LONG))
    assert any("EMA20 > EMA50" in r for r in result.reasons)
    assert any("VETOED" in r for r in result.reasons)


def test_llm_review_is_attached_to_signal_metadata():
    result, _ = layer(response("CONFIRM", "Consistent.")).apply(signal(LONG))
    assert result.metadata["llm"]["ok"] is True
    assert result.metadata["cross_age"] == 1  # original metadata preserved


def test_cost_is_zero_for_a_free_provider():
    _, review = LLMDecisionLayer(NullVetoBaseline()).apply(signal(LONG))
    assert review.cost_usd == 0.0
