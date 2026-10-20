"""MACD — Moving Average Convergence Divergence."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.indicators.base import as_series, validate_period
from app.indicators.moving_averages import ema


@dataclass(frozen=True)
class MACDResult:
    """The three MACD series, kept together so callers cannot mismatch them."""

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"macd": self.macd, "macd_signal": self.signal, "macd_hist": self.histogram}
        )


def macd(
    data: pd.Series | pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> MACDResult:
    """MACD line = EMA(fast) - EMA(slow); signal = EMA(signal) of that line.

    The signal EMA is computed on the MACD line *after* dropping its warm-up
    NaNs, so the signal seed uses real MACD values only.
    """
    for value, name in ((fast, "fast"), (slow, "slow"), (signal, "signal")):
        validate_period(value, name)
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    series = as_series(data, column)
    macd_line = (ema(series, fast) - ema(series, slow)).rename("macd")

    valid = macd_line.dropna()
    signal_line = pd.Series(float("nan"), index=series.index, dtype="float64")
    if len(valid) >= signal:
        signal_line.loc[valid.index] = ema(valid, signal).to_numpy()
    signal_line = signal_line.rename("macd_signal")

    histogram = (macd_line - signal_line).rename("macd_hist")
    return MACDResult(macd=macd_line, signal=signal_line, histogram=histogram)
