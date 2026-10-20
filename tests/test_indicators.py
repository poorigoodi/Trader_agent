"""Unit tests for the indicator library.

Where possible, expected values come from an INDEPENDENT source rather than
from this codebase:
  * RSI(14) is checked against Wilder's published worked example (the series
    reproduced by StockCharts), which is the canonical reference.
  * EMA, SMA, Bollinger and VWAP are checked against hand-computed values.
  * ATR is checked against both a hand-computed case and its defining
    recurrence.
A test that only compares the code to itself would pass no matter how wrong
the maths were.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.indicators.atr import atr, true_range
from app.indicators.base import validate_period, wilder_smooth
from app.indicators.bollinger import bollinger_bands
from app.indicators.macd import macd
from app.indicators.moving_averages import ema, sma
from app.indicators.rsi import rsi
from app.indicators.vwap import rolling_vwap, session_vwap, typical_price
from tests.conftest import make_ohlcv


# --------------------------------------------------------------------------
# SMA
# --------------------------------------------------------------------------

def test_sma_hand_computed():
    result = sma(pd.Series([1.0, 2, 3, 4, 5]), 3)
    assert math.isnan(result.iloc[0]) and math.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)  # (1+2+3)/3
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_sma_warmup_is_nan_not_partial():
    """A partial average would be a silently wrong number, not a missing one."""
    result = sma(pd.Series([10.0] * 10), 5)
    assert result.iloc[:4].isna().all()
    assert (result.iloc[4:] == 10.0).all()


def test_sma_reads_close_from_frame():
    df = make_ohlcv([1.0, 2, 3, 4, 5])
    assert sma(df, 3).iloc[-1] == pytest.approx(4.0)


# --------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------

def test_ema_hand_computed():
    """period=3 -> alpha=0.5, seed = SMA(1,2,3) = 2 at index 2."""
    result = ema(pd.Series([1.0, 2, 3, 4, 5]), 3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(0.5 * 4 + 0.5 * 2)  # 3.0
    assert result.iloc[4] == pytest.approx(0.5 * 5 + 0.5 * 3)  # 4.0


def test_ema_of_constant_series_is_that_constant():
    result = ema(pd.Series([7.0] * 30), 10).dropna()
    assert np.allclose(result.to_numpy(), 7.0)


def test_ema_reacts_faster_than_sma():
    """The defining property: right after a step up, EMA leads SMA.

    Measured 3 candles after the step, while the SMA window still holds
    mostly pre-step values. Once a full window of new prices has passed both
    converge, so the comparison only means something early.
    """
    series = pd.Series([10.0] * 30 + [20.0] * 3)
    assert ema(series, 10).iloc[-1] > sma(series, 10).iloc[-1]


def test_ema_shorter_than_period_is_all_nan():
    assert ema(pd.Series([1.0, 2.0]), 5).isna().all()


def test_ema_rejects_bad_period():
    with pytest.raises(ValueError):
        ema(pd.Series([1.0, 2.0]), 0)
    with pytest.raises(TypeError):
        ema(pd.Series([1.0, 2.0]), 3.5)


# --------------------------------------------------------------------------
# RSI  -- checked against Wilder's published example
# --------------------------------------------------------------------------

WILDER_CLOSES = [
    44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245,
    45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028,
    46.0328, 46.4116, 46.2222, 45.6439, 46.2122, 46.2521, 45.7137, 46.4515,
    45.7835, 45.3548, 44.0288, 44.1783, 44.2181, 44.5672, 43.4205, 42.6628,
    43.1314,
]
# Published RSI(14) values for the series above.
WILDER_RSI = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38,
    54.71, 50.42, 39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77,
]


def test_rsi_matches_wilders_published_example():
    result = rsi(pd.Series(WILDER_CLOSES), 14).dropna().tolist()
    assert len(result) == len(WILDER_RSI)
    for actual, expected in zip(result, WILDER_RSI):
        assert actual == pytest.approx(expected, abs=0.01)


def test_rsi_first_value_lands_at_index_period():
    result = rsi(pd.Series(WILDER_CLOSES), 14)
    assert result.iloc[:14].isna().all()
    assert not math.isnan(result.iloc[14])


def test_rsi_of_monotonic_rise_is_100():
    result = rsi(pd.Series([float(i) for i in range(1, 40)]), 14)
    assert result.dropna().eq(100.0).all()


def test_rsi_of_monotonic_fall_is_0():
    result = rsi(pd.Series([float(i) for i in range(40, 1, -1)]), 14)
    assert result.dropna().eq(0.0).all()


def test_rsi_stays_within_bounds(sample_closes):
    values = rsi(pd.Series(sample_closes), 14).dropna()
    assert values.between(0.0, 100.0).all()


def test_rsi_too_short_returns_all_nan():
    assert rsi(pd.Series([1.0] * 10), 14).isna().all()


# --------------------------------------------------------------------------
# ATR
# --------------------------------------------------------------------------

def test_true_range_takes_the_widest_of_three():
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 11.0],
            "low": [9.0, 11.0, 8.0],
            "close": [9.5, 11.5, 8.5],
        }
    )
    tr = true_range(df)
    assert math.isnan(tr.iloc[0])          # no previous close
    assert tr.iloc[1] == pytest.approx(2.5)  # |12 - 9.5| beats 12-11
    assert tr.iloc[2] == pytest.approx(3.5)  # |8 - 11.5| beats 11-8


def test_true_range_first_row_is_nan_not_high_minus_low():
    df = make_ohlcv([100.0, 101.0, 102.0])
    assert math.isnan(true_range(df).iloc[0])


def test_atr_of_constant_range_equals_that_range():
    """Every bar has high-low = 1.0 and no gaps, so ATR must be 1.0."""
    df = make_ohlcv([100.0] * 30, spread=0.5)
    result = atr(df, 14).dropna()
    assert result.eq(1.0).all()


def test_atr_hand_computed_seed_and_first_smoothing():
    df = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [9.5, 10.5, 11.5, 12.5],
        }
    )
    # TRs: NaN, |11-9.5|=1.5, |12-10.5|=1.5, |13-11.5|=1.5
    result = atr(df, 2)
    assert result.iloc[2] == pytest.approx(1.5)  # seed = mean(1.5, 1.5)
    # Wilder step: prev + (tr - prev)/2 = 1.5
    assert result.iloc[3] == pytest.approx(1.5)


def test_atr_warms_up_after_period_true_ranges():
    df = make_ohlcv([100.0 + i for i in range(20)])
    result = atr(df, 14)
    # TR is NaN at row 0, so the seed lands at row 14.
    assert result.iloc[:14].isna().all()
    assert not math.isnan(result.iloc[14])


def test_atr_is_positive_on_real_shaped_data(sample_ohlcv):
    assert (atr(sample_ohlcv, 14).dropna() > 0).all()


def test_wilder_smooth_recurrence():
    values = pd.Series([2.0, 4.0, 6.0, 8.0])
    result = wilder_smooth(values, 2)
    assert result.iloc[1] == pytest.approx(3.0)  # seed = (2+4)/2
    assert result.iloc[2] == pytest.approx(3.0 + (6.0 - 3.0) / 2)  # 4.5
    assert result.iloc[3] == pytest.approx(4.5 + (8.0 - 4.5) / 2)  # 6.25


# --------------------------------------------------------------------------
# MACD
# --------------------------------------------------------------------------

def test_macd_line_is_the_ema_difference():
    series = pd.Series(np.linspace(100, 140, 120))
    result = macd(series, 12, 26, 9)
    expected = ema(series, 12) - ema(series, 26)
    pd.testing.assert_series_equal(
        result.macd.dropna(), expected.dropna(), check_names=False
    )


def test_macd_histogram_is_line_minus_signal():
    series = pd.Series(np.linspace(100, 140, 120))
    result = macd(series, 12, 26, 9)
    diff = (result.macd - result.signal).dropna()
    pd.testing.assert_series_equal(
        result.histogram.dropna(), diff, check_names=False
    )


def test_macd_of_constant_series_is_zero():
    result = macd(pd.Series([50.0] * 120), 12, 26, 9)
    assert result.macd.dropna().abs().max() == pytest.approx(0.0, abs=1e-9)
    assert result.histogram.dropna().abs().max() == pytest.approx(0.0, abs=1e-9)


def test_macd_is_positive_in_a_rising_market():
    series = pd.Series(np.linspace(100, 200, 150))
    assert macd(series, 12, 26, 9).macd.dropna().iloc[-1] > 0


def test_macd_rejects_fast_not_below_slow():
    with pytest.raises(ValueError):
        macd(pd.Series([1.0] * 50), fast=26, slow=12)


# --------------------------------------------------------------------------
# Bollinger Bands
# --------------------------------------------------------------------------

def test_bollinger_hand_computed_on_1_to_20():
    """SMA(1..20) = 10.5; population std = sqrt(665/20) = 5.766281..."""
    series = pd.Series([float(i) for i in range(1, 21)])
    result = bollinger_bands(series, 20, 2.0)
    expected_std = math.sqrt(665.0 / 20.0)
    assert result.middle.iloc[-1] == pytest.approx(10.5)
    assert result.upper.iloc[-1] == pytest.approx(10.5 + 2 * expected_std)
    assert result.lower.iloc[-1] == pytest.approx(10.5 - 2 * expected_std)


def test_bollinger_uses_population_std_not_sample_std():
    """ddof=0 is the published convention; pandas defaults to ddof=1."""
    series = pd.Series([float(i) for i in range(1, 21)])
    result = bollinger_bands(series, 20, 2.0)
    sample_std = series.std(ddof=1)
    assert result.upper.iloc[-1] != pytest.approx(10.5 + 2 * sample_std)


def test_bollinger_bands_collapse_on_constant_series():
    result = bollinger_bands(pd.Series([25.0] * 30), 20, 2.0)
    assert result.upper.dropna().eq(25.0).all()
    assert result.lower.dropna().eq(25.0).all()


def test_bollinger_percent_b_is_one_at_the_upper_band():
    series = pd.Series([float(i) for i in range(1, 21)])
    result = bollinger_bands(series, 20, 2.0)
    upper, lower, close = (
        result.upper.iloc[-1], result.lower.iloc[-1], series.iloc[-1]
    )
    assert result.percent_b.iloc[-1] == pytest.approx((close - lower) / (upper - lower))


def test_bollinger_ordering_always_holds(sample_closes):
    result = bollinger_bands(pd.Series(sample_closes), 20, 2.0)
    valid = result.upper.notna()
    assert (result.upper[valid] >= result.middle[valid]).all()
    assert (result.middle[valid] >= result.lower[valid]).all()


def test_bollinger_rejects_non_positive_std():
    with pytest.raises(ValueError):
        bollinger_bands(pd.Series([1.0] * 30), 20, 0.0)


# --------------------------------------------------------------------------
# VWAP
# --------------------------------------------------------------------------

def test_typical_price_is_hlc_over_three():
    df = pd.DataFrame({"high": [12.0], "low": [6.0], "close": [9.0]})
    assert typical_price(df).iloc[0] == pytest.approx(9.0)


def test_session_vwap_hand_computed():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T00:15Z"], utc=True
            ),
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.0, 20.0],
            "volume": [1.0, 3.0],
        }
    )
    result = session_vwap(df)
    assert result.iloc[0] == pytest.approx(10.0)
    # (10*1 + 20*3) / 4 = 17.5
    assert result.iloc[1] == pytest.approx(17.5)


def test_session_vwap_resets_on_a_new_utc_day():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T23:45Z", "2026-01-02T00:00Z"], utc=True
            ),
            "high": [11.0, 51.0],
            "low": [9.0, 49.0],
            "close": [10.0, 50.0],
            "volume": [100.0, 1.0],
        }
    )
    result = session_vwap(df)
    # Without the reset, yesterday's 100 units of volume would dominate.
    assert result.iloc[1] == pytest.approx(50.0)


def test_rolling_vwap_uses_only_the_window():
    df = make_ohlcv([10.0, 10.0, 10.0, 100.0], spread=0.0)
    result = rolling_vwap(df, 2)
    assert result.iloc[2] == pytest.approx(10.0)
    assert result.iloc[3] == pytest.approx(55.0)


def test_vwap_sits_inside_the_price_range(sample_ohlcv):
    result = session_vwap(sample_ohlcv).dropna()
    assert result.min() >= sample_ohlcv["low"].min()
    assert result.max() <= sample_ohlcv["high"].max()


# --------------------------------------------------------------------------
# shared validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -14])
def test_validate_period_rejects_non_positive(bad):
    with pytest.raises(ValueError):
        validate_period(bad)


@pytest.mark.parametrize("bad", [1.5, "14", None, True])
def test_validate_period_rejects_non_int(bad):
    with pytest.raises(TypeError):
        validate_period(bad)
