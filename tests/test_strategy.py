"""Unit tests for the rule-based strategy.

Two things are being pinned down here:

1. Each of the three gates rejects independently, and names itself when it
   does. A NO_TRADE whose reason is wrong is worse than no log at all.
2. Every signal -- including NO_TRADE -- carries a reason and an
   invalidation condition, which is a stated requirement of this phase.

The fixtures use ``cross_max_age=12`` rather than the shipped default of 3.
That is not a workaround: it is because a fractal swing needs 3 confirming
bars on each side, so market structure cannot print a fresh UP read until
roughly 10-15 candles after a reversal begins, by which time a 3-candle
cross window has already expired. On real data (measured on Toobit 15m,
~1000 candles x 3 symbols) the shipped default still fires, because real
swings are irregular; on a perfectly regular synthetic staircase it cannot.
See docs/architecture.md, "Gate timing".
"""

from __future__ import annotations

import pytest

from app.config.settings import StrategySettings
from app.indicators.engine import compute_indicators
from app.strategies.base import SignalDirection
from app.strategies.ema_rsi_trend import EmaRsiTrendStrategy
from tests.conftest import make_ohlcv


PERMISSIVE = StrategySettings(cross_max_age=12, structure_lookback=30)


def staircase(sign: int, pre: int = 100, cycles: int = 6) -> list[float]:
    """Decline, then a zig-zag trend in ``sign`` direction (1 up, -1 down)."""
    values: list[float] = []
    price = 100.0
    for _ in range(pre):
        price -= 0.08 * sign
        values.append(price)
    for _ in range(cycles):
        for _ in range(5):
            price += 1.2 * sign
            values.append(price)
        for _ in range(4):
            price -= 0.7 * sign
            values.append(price)
    return values


def evaluate(values, settings=PERMISSIVE, end=117, symbol="BTC/USDT"):
    frame = compute_indicators(make_ohlcv(values, spread=0.2), settings)
    strategy = EmaRsiTrendStrategy(settings)
    return strategy.evaluate(frame.iloc[:end].reset_index(drop=True), symbol, "15m")


# --------------------------------------------------------------------------
# happy paths
# --------------------------------------------------------------------------

def test_clean_uptrend_produces_long():
    signal = evaluate(staircase(1))
    assert signal.direction is SignalDirection.LONG
    assert signal.is_actionable


def test_clean_downtrend_produces_short():
    signal = evaluate(staircase(-1))
    assert signal.direction is SignalDirection.SHORT
    assert signal.is_actionable


def test_long_reasons_name_all_three_gates():
    signal = evaluate(staircase(1))
    joined = signal.reason_text
    assert "EMA20" in joined and "EMA50" in joined
    assert "fresh EMA cross" in joined
    assert "RSI14" in joined
    assert "structure UP" in joined


def test_short_reasons_name_all_three_gates():
    signal = evaluate(staircase(-1))
    joined = signal.reason_text
    assert "fresh EMA cross" in joined
    assert "RSI14" in joined
    assert "structure DOWN" in joined


def test_long_invalidation_points_below_the_slow_ema():
    signal = evaluate(staircase(1))
    assert "below EMA50" in signal.invalidation
    assert "lower low" in signal.invalidation


def test_short_invalidation_points_above_the_slow_ema():
    signal = evaluate(staircase(-1))
    assert "above EMA50" in signal.invalidation
    assert "higher high" in signal.invalidation


# --------------------------------------------------------------------------
# gate A: the EMA cross
# --------------------------------------------------------------------------

def test_stale_cross_is_rejected_and_says_so():
    """Same data, same everything -- only the freshness window changes."""
    strict = StrategySettings(cross_max_age=3, structure_lookback=30)
    signal = evaluate(staircase(1), settings=strict)
    assert signal.direction is SignalDirection.NO_TRADE
    assert "stale" in signal.reason_text
    assert signal.metadata["cross_age"] > strict.cross_max_age


def test_no_cross_at_all_is_rejected():
    """A monotonic ramp never crosses: EMA20 is above EMA50 throughout."""
    signal = evaluate([100.0 + i * 0.5 for i in range(200)], end=180)
    assert signal.direction is SignalDirection.NO_TRADE
    assert "no EMA cross" in signal.reason_text


def test_cross_age_zero_means_the_cross_is_on_the_latest_candle():
    values = staircase(1)
    frame = compute_indicators(make_ohlcv(values, spread=0.2), PERMISSIVE)
    strategy = EmaRsiTrendStrategy(PERMISSIVE)
    age = strategy._cross_age(frame.iloc[:107].reset_index(drop=True), SignalDirection.LONG)
    assert age == 0


def test_cross_age_is_none_when_no_cross_exists():
    frame = compute_indicators(
        make_ohlcv([100.0 + i * 0.5 for i in range(200)]), PERMISSIVE
    )
    strategy = EmaRsiTrendStrategy(PERMISSIVE)
    assert strategy._cross_age(frame, SignalDirection.SHORT) is None


def test_cross_age_ignores_the_warmup_boundary():
    """The first row where both EMAs exist is not a cross."""
    frame = compute_indicators(make_ohlcv([100.0] * 60), PERMISSIVE)
    strategy = EmaRsiTrendStrategy(PERMISSIVE)
    assert strategy._cross_age(frame, SignalDirection.LONG) is None


# --------------------------------------------------------------------------
# gate B: the RSI window
# --------------------------------------------------------------------------

def test_rsi_outside_the_window_is_rejected_and_says_so():
    # Force an impossible window so the RSI gate is the one that fails.
    settings = StrategySettings(
        cross_max_age=12, structure_lookback=30,
        rsi_long_min=95.0, rsi_long_max=99.0,
    )
    signal = evaluate(staircase(1), settings=settings)
    assert signal.direction is SignalDirection.NO_TRADE
    assert "RSI14" in signal.reason_text and "outside" in signal.reason_text
    assert "would re-open this setup" in signal.invalidation


def test_rsi_gate_runs_after_the_cross_gate():
    """Reasons accumulate in evaluation order, so the cross is named first."""
    settings = StrategySettings(
        cross_max_age=12, structure_lookback=30,
        rsi_long_min=95.0, rsi_long_max=99.0,
    )
    signal = evaluate(staircase(1), settings=settings)
    assert "EMA cross" in signal.reasons[1]
    assert "RSI" in signal.reasons[-1]


# --------------------------------------------------------------------------
# gate C: market structure
# --------------------------------------------------------------------------

def test_structure_disagreement_is_rejected_and_says_so():
    """Evaluate the uptrend data while structure has not confirmed yet."""
    signal = evaluate(staircase(1), end=110)
    assert signal.direction is SignalDirection.NO_TRADE
    assert "market structure is" in signal.reason_text


def test_structure_rejection_names_the_structure_it_wanted():
    signal = evaluate(staircase(1), end=110)
    assert "needed UP" in signal.reason_text
    assert "structure printing UP" in signal.invalidation


# --------------------------------------------------------------------------
# warm-up and guards
# --------------------------------------------------------------------------

def test_insufficient_history_is_no_trade_not_a_crash():
    # min_rows = max(ema_slow + rsi_period, structure_lookback) + 5 = 69
    signal = evaluate([100.0 + i * 0.1 for i in range(65)], end=65)
    assert signal.direction is SignalDirection.NO_TRADE
    assert "insufficient history" in signal.reason_text


def test_unwarmed_indicators_are_no_trade():
    signal = evaluate([100.0 + i * 0.1 for i in range(60)], end=55)
    assert signal.direction is SignalDirection.NO_TRADE
    assert not signal.is_actionable


# --------------------------------------------------------------------------
# contract that holds for EVERY signal
# --------------------------------------------------------------------------

@pytest.mark.parametrize("end", list(range(105, 140, 5)))
def test_every_signal_has_a_reason_and_an_invalidation(end):
    signal = evaluate(staircase(1), end=end)
    assert signal.reasons, "a signal with no reason is unreviewable"
    assert signal.reason_text != "no reason recorded"
    assert signal.invalidation.strip()


@pytest.mark.parametrize("end", [110, 117, 125])
def test_snapshot_is_attached_to_every_signal(end):
    signal = evaluate(staircase(1), end=end)
    assert signal.snapshot.symbol == "BTC/USDT"
    assert signal.snapshot.close > 0
    payload = signal.to_dict()
    assert payload["snapshot"]["rsi"] is not None
    assert payload["direction"] == signal.direction.value


def test_strategy_is_deterministic():
    """Same candles in, same signal out. Twenty times."""
    values = staircase(1)
    results = {
        (evaluate(values).direction, evaluate(values).reason_text) for _ in range(20)
    }
    assert len(results) == 1


def test_metadata_records_context_that_is_not_a_gate():
    """MACD/%B/VWAP are recorded for later review but must not gate today."""
    signal = evaluate(staircase(1))
    for key in ("macd_hist", "bb_percent_b", "close_vs_vwap_pct", "atr_pct_of_price"):
        assert key in signal.metadata


def test_strategy_never_returns_an_unknown_direction():
    for end in range(105, 145):
        assert evaluate(staircase(1), end=end).direction in set(SignalDirection)
