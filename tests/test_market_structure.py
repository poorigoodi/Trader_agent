"""Unit tests for swing detection and trend classification.

The structure module is what stops the strategy from buying a downtrend just
because two EMAs crossed, so its failure modes matter: it must refuse to
guess (UNKNOWN) rather than default to a direction, and it must never use a
candle that has not been confirmed yet.
"""

from __future__ import annotations

import pytest

from app.market_structure.trend import Trend, detect_trend, find_swings
from tests.conftest import make_ohlcv


def frame_from_hl(highs: list[float], lows: list[float]):
    """Build a frame with explicit highs/lows (closes sit midway)."""
    import pandas as pd

    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    df = make_ohlcv(closes)
    df["high"] = highs
    df["low"] = lows
    return df


# --------------------------------------------------------------------------
# find_swings
# --------------------------------------------------------------------------

def test_find_swings_detects_an_obvious_peak():
    highs = [1.0, 2.0, 5.0, 2.0, 1.0]
    lows = [0.0, 0.0, 0.0, 0.0, 0.0]
    swing_highs, _ = find_swings(frame_from_hl(highs, lows), window=2)
    assert swing_highs == [5.0]


def test_find_swings_detects_an_obvious_trough():
    highs = [9.0, 9.0, 9.0, 9.0, 9.0]
    lows = [5.0, 4.0, 1.0, 4.0, 5.0]
    _, swing_lows = find_swings(frame_from_hl(highs, lows), window=2)
    assert swing_lows == [1.0]


def test_last_candles_can_never_be_a_swing():
    """The final `window` bars are unconfirmed. Treating them as swings would
    be look-ahead: we would be claiming a peak before the right side exists."""
    highs = [1.0, 2.0, 3.0, 4.0, 9.0]  # the 9.0 is the newest bar
    lows = [0.0] * 5
    swing_highs, _ = find_swings(frame_from_hl(highs, lows), window=2)
    assert 9.0 not in swing_highs


def test_flat_market_has_no_swings():
    highs = [5.0] * 20
    lows = [1.0] * 20
    swing_highs, swing_lows = find_swings(frame_from_hl(highs, lows), window=3)
    assert swing_highs == [] and swing_lows == []


def test_find_swings_rejects_bad_window():
    with pytest.raises(ValueError):
        find_swings(make_ohlcv([1.0] * 10), window=0)


# --------------------------------------------------------------------------
# detect_trend
# --------------------------------------------------------------------------

def build_staircase(direction: int, steps: int = 4, window: int = 2):
    """Zig-zag with each peak/trough higher (or lower) than the last."""
    highs: list[float] = []
    lows: list[float] = []
    base = 100.0
    for step in range(steps):
        level = base + direction * step * 10.0
        # up leg, peak, down leg, trough -- each confirmed by flat shoulders
        highs += [level, level + 5, level]
        lows += [level - 5, level, level - 5]
    return frame_from_hl(highs, lows), window


def test_uptrend_is_higher_highs_and_higher_lows():
    df, window = build_staircase(direction=1)
    reading = detect_trend(df, lookback=len(df), window=1)
    assert reading.trend is Trend.UP
    assert reading.higher_highs and reading.higher_lows
    assert "higher highs" in reading.reason


def test_downtrend_is_lower_highs_and_lower_lows():
    df, window = build_staircase(direction=-1)
    reading = detect_trend(df, lookback=len(df), window=1)
    assert reading.trend is Trend.DOWN
    assert reading.lower_highs and reading.lower_lows


def test_too_few_swings_is_unknown_not_a_guess():
    """A flat tape must not be reported as RANGE-with-confidence or as a
    direction; UNKNOWN is what the strategy treats as 'do not trade'."""
    df = frame_from_hl([5.0] * 30, [1.0] * 30)
    reading = detect_trend(df, lookback=30, window=3)
    assert reading.trend is Trend.UNKNOWN
    assert reading.reason == "no clear swing sequence"


def test_mixed_structure_is_range():
    # Higher high but lower low -> expanding range, not a trend.
    highs = [100.0, 110.0, 100.0, 100.0, 120.0, 100.0]
    lows = [90.0, 90.0, 80.0, 90.0, 90.0, 70.0]
    reading = detect_trend(frame_from_hl(highs, lows), lookback=6, window=1)
    assert reading.trend in (Trend.RANGE, Trend.UNKNOWN)


def test_detect_trend_rejects_lookback_too_small_for_window():
    with pytest.raises(ValueError):
        detect_trend(make_ohlcv([1.0] * 50), lookback=5, window=3)


def test_detect_trend_only_looks_at_the_lookback_window():
    """Ancient history must not colour the current read."""
    old = [200.0 - i for i in range(40)]      # long decline
    recent = [100.0, 105.0, 100.0, 108.0, 103.0, 112.0, 107.0, 115.0]
    df = make_ohlcv(old + recent)
    reading = detect_trend(df, lookback=8, window=1)
    assert reading.trend is not Trend.DOWN


def test_trend_reading_is_serialisable_for_the_journal():
    df, _ = build_staircase(direction=1)
    reading = detect_trend(df, lookback=len(df), window=1)
    assert isinstance(reading.trend.value, str)
    assert isinstance(reading.swing_highs, tuple)
