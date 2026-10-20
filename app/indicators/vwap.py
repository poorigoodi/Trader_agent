"""Volume Weighted Average Price.

Two variants are provided because they answer different questions:
  * ``session_vwap``  — anchored to the UTC day, what an intraday desk means
    by "VWAP". Resets at 00:00 UTC (crypto has no exchange session break, so
    the UTC day is the conventional anchor).
  * ``rolling_vwap``  — a fixed-length rolling window, useful on higher
    timeframes where a daily anchor is too coarse.
"""

from __future__ import annotations

import pandas as pd

from app.indicators.base import require_ohlc, validate_period


def typical_price(df: pd.DataFrame) -> pd.Series:
    require_ohlc(df)
    return (
        (df["high"].astype("float64") + df["low"].astype("float64") + df["close"].astype("float64"))
        / 3.0
    ).rename("typical_price")


def session_vwap(df: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.Series:
    """Cumulative VWAP anchored to each UTC calendar day."""
    require_ohlc(df, ("high", "low", "close", "volume"))
    if timestamp_column not in df.columns:
        raise KeyError(f"column {timestamp_column!r} required for session VWAP")

    tp = typical_price(df)
    volume = df["volume"].astype("float64")
    day = pd.to_datetime(df[timestamp_column], utc=True).dt.floor("D")

    cum_pv = (tp * volume).groupby(day).cumsum()
    cum_vol = volume.groupby(day).cumsum()
    return (cum_pv / cum_vol.replace(0.0, float("nan"))).rename("vwap")


def rolling_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """VWAP over the last ``period`` candles."""
    validate_period(period)
    require_ohlc(df, ("high", "low", "close", "volume"))

    tp = typical_price(df)
    volume = df["volume"].astype("float64")
    pv = (tp * volume).rolling(window=period, min_periods=period).sum()
    vol = volume.rolling(window=period, min_periods=period).sum()
    return (pv / vol.replace(0.0, float("nan"))).rename(f"vwap_{period}")
