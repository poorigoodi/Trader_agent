"""Bollinger Bands."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.indicators.base import as_series, validate_period
from app.indicators.moving_averages import sma


@dataclass(frozen=True)
class BollingerResult:
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    percent_b: pd.Series

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "bb_upper": self.upper,
                "bb_middle": self.middle,
                "bb_lower": self.lower,
                "bb_bandwidth": self.bandwidth,
                "bb_percent_b": self.percent_b,
            }
        )


def bollinger_bands(
    data: pd.Series | pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    column: str = "close",
) -> BollingerResult:
    """Bollinger Bands using the population standard deviation (ddof=0).

    ddof=0 is the convention in the original Bollinger definition and in
    TA-Lib; using the sample std (pandas' default ddof=1) would widen the
    bands slightly and silently disagree with every published chart.
    """
    validate_period(period)
    if num_std <= 0:
        raise ValueError(f"num_std must be > 0, got {num_std}")

    series = as_series(data, column)
    middle = sma(series, period).rename("bb_middle")
    std = series.rolling(window=period, min_periods=period).std(ddof=0)

    upper = (middle + num_std * std).rename("bb_upper")
    lower = (middle - num_std * std).rename("bb_lower")

    span = (upper - lower)
    bandwidth = (span / middle).rename("bb_bandwidth")
    percent_b = ((series - lower) / span.replace(0.0, float("nan"))).rename("bb_percent_b")

    return BollingerResult(
        upper=upper, middle=middle, lower=lower, bandwidth=bandwidth, percent_b=percent_b
    )
