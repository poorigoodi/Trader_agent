"""Simple and exponential moving averages."""

from __future__ import annotations

import pandas as pd

from app.indicators.base import as_series, validate_period


def sma(data: pd.Series | pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Simple moving average. NaN for the first ``period - 1`` rows."""
    validate_period(period)
    series = as_series(data, column)
    return series.rolling(window=period, min_periods=period).mean().rename(f"sma_{period}")


def ema(data: pd.Series | pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Exponential moving average, TA-Lib compatible.

    Seeded with SMA(period) at index ``period - 1``, then
    ``ema[i] = alpha * price[i] + (1 - alpha) * ema[i-1]`` with
    ``alpha = 2 / (period + 1)``. Rows before the seed are NaN, so callers
    can never accidentally act on an unwarmed average.
    """
    validate_period(period)
    series = as_series(data, column)
    out = pd.Series(float("nan"), index=series.index, dtype="float64")
    if len(series) < period:
        return out.rename(f"ema_{period}")

    alpha = 2.0 / (period + 1.0)
    prev = float(series.iloc[:period].mean())
    out.iloc[period - 1] = prev
    for i in range(period, len(series)):
        prev = alpha * float(series.iloc[i]) + (1.0 - alpha) * prev
        out.iloc[i] = prev
    return out.rename(f"ema_{period}")
