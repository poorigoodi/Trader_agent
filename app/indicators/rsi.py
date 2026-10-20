"""Relative Strength Index (Wilder, 1978)."""

from __future__ import annotations

import pandas as pd

from app.indicators.base import as_series, validate_period, wilder_smooth


def rsi(data: pd.Series | pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """RSI over ``period`` closes, on a 0-100 scale.

    First value appears at index ``period`` (the first ``period`` price
    changes are consumed by the seed average). A zero average loss yields
    100.0 rather than an infinity/NaN, matching the standard convention.
    """
    validate_period(period)
    series = as_series(data, column)
    out = pd.Series(float("nan"), index=series.index, dtype="float64")
    if len(series) <= period:
        return out.rename(f"rsi_{period}")

    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)

    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    result = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 -> pure uptrend -> RSI 100; avg_gain == 0 -> RSI 0.
    result = result.where(~((avg_loss == 0.0) & avg_gain.notna()), 100.0)
    result = result.where(~((avg_gain == 0.0) & avg_loss.notna() & (avg_loss > 0.0)), 0.0)
    return result.rename(f"rsi_{period}")
