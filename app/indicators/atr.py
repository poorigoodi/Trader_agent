"""True Range and Average True Range (Wilder)."""

from __future__ import annotations

import pandas as pd

from app.indicators.base import require_ohlc, validate_period, wilder_smooth


def true_range(df: pd.DataFrame) -> pd.Series:
    """TR = max(high-low, |high - prev_close|, |low - prev_close|).

    The first row has no previous close, so it is NaN (we do not fall back to
    high-low: that silently invents a value the market never printed).
    """
    require_ohlc(df)
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    prev_close = df["close"].astype("float64").shift(1)

    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    tr = ranges.max(axis=1)
    tr.iloc[0] = float("nan")
    return tr.rename("true_range")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed. NaN until ``period`` TRs exist."""
    validate_period(period)
    return wilder_smooth(true_range(df), period).rename(f"atr_{period}")
