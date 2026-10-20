"""Shared helpers and validation for the indicator library.

All indicators in this package are implemented directly on top of pandas
rather than pulled from ``pandas-ta``/TA-Lib. Reasons (see README, OD-1):
  * every number the agent trades on must be covered by our own unit tests
  * ``pandas-ta`` is unmaintained and breaks on numpy>=2 / pandas>=3
  * indicator seeding conventions differ between libraries; pinning our own
    makes historical results reproducible

Convention used throughout, matching TA-Lib:
  * an indicator with period ``n`` produces NaN until enough data exists
  * Wilder-smoothed indicators (RSI, ATR) seed with a simple average of the
    first ``n`` observations, then apply the recursive Wilder update
"""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def validate_period(period: int, name: str = "period") -> int:
    if not isinstance(period, int) or isinstance(period, bool):
        raise TypeError(f"{name} must be an int, got {type(period).__name__}")
    if period < 1:
        raise ValueError(f"{name} must be >= 1, got {period}")
    return period


def as_series(data: pd.Series | pd.DataFrame, column: str = "close") -> pd.Series:
    """Accept either a Series or an OHLCV frame and return a float Series."""
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise KeyError(f"column {column!r} not found in frame")
        series = data[column]
    elif isinstance(data, pd.Series):
        series = data
    else:
        raise TypeError(f"expected Series or DataFrame, got {type(data).__name__}")
    return series.astype("float64")


def require_ohlc(df: pd.DataFrame, columns: tuple[str, ...] = ("high", "low", "close")) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"missing required OHLC column(s): {missing}")


def wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) with a simple-average seed.

    ``values`` is expected to already have leading NaNs stripped of meaning:
    the seed is the mean of the first ``period`` non-NaN observations, placed
    at that observation's index; everything before it is NaN.
    """
    validate_period(period)
    out = pd.Series(float("nan"), index=values.index, dtype="float64")
    valid = values.dropna()
    if len(valid) < period:
        return out

    seed_idx = valid.index[period - 1]
    prev = float(valid.iloc[:period].mean())
    out.loc[seed_idx] = prev

    alpha = 1.0 / period
    for idx, value in valid.iloc[period:].items():
        prev = prev + alpha * (float(value) - prev)
        out.loc[idx] = prev
    return out
