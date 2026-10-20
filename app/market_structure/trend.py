"""Minimal, explicit market-structure detection.

"Trend" here means the classic price-action definition -- a sequence of higher
highs + higher lows (uptrend) or lower highs + lower lows (downtrend) -- not an
indicator reading. It is deliberately simple and fully deterministic; the
roadmap's richer support/resistance work belongs to a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Trend(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TrendReading:
    trend: Trend
    higher_highs: bool
    higher_lows: bool
    lower_highs: bool
    lower_lows: bool
    swing_highs: tuple[float, ...]
    swing_lows: tuple[float, ...]

    @property
    def reason(self) -> str:
        parts = []
        if self.higher_highs:
            parts.append("higher highs")
        if self.higher_lows:
            parts.append("higher lows")
        if self.lower_highs:
            parts.append("lower highs")
        if self.lower_lows:
            parts.append("lower lows")
        return ", ".join(parts) if parts else "no clear swing sequence"


def find_swings(df: pd.DataFrame, window: int = 3) -> tuple[list[float], list[float]]:
    """Fractal swing points: a high with `window` lower highs on each side.

    Only *confirmed* swings are returned -- the last `window` candles can
    never form a swing yet, which is exactly what prevents look-ahead here.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    highs = df["high"].astype("float64").to_numpy()
    lows = df["low"].astype("float64").to_numpy()

    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(window, len(df) - window):
        left_h, right_h = highs[i - window : i], highs[i + 1 : i + 1 + window]
        if highs[i] > left_h.max() and highs[i] > right_h.max():
            swing_highs.append(float(highs[i]))
        left_l, right_l = lows[i - window : i], lows[i + 1 : i + 1 + window]
        if lows[i] < left_l.min() and lows[i] < right_l.min():
            swing_lows.append(float(lows[i]))
    return swing_highs, swing_lows


def detect_trend(df: pd.DataFrame, lookback: int = 20, window: int = 3) -> TrendReading:
    """Classify the last `lookback` candles as UP / DOWN / RANGE / UNKNOWN.

    UP requires both higher highs and higher lows; DOWN requires both lower
    highs and lower lows. Anything else with enough swings is RANGE; too few
    swings is UNKNOWN (and the strategy treats UNKNOWN as "do not trade").
    """
    if lookback < 2 * window + 2:
        raise ValueError(f"lookback must be >= {2 * window + 2} for window={window}")

    recent = df.tail(lookback)
    swing_highs, swing_lows = find_swings(recent, window)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return TrendReading(
            Trend.UNKNOWN, False, False, False, False,
            tuple(swing_highs), tuple(swing_lows),
        )

    hh = swing_highs[-1] > swing_highs[-2]
    hl = swing_lows[-1] > swing_lows[-2]
    lh = swing_highs[-1] < swing_highs[-2]
    ll = swing_lows[-1] < swing_lows[-2]

    if hh and hl:
        trend = Trend.UP
    elif lh and ll:
        trend = Trend.DOWN
    else:
        trend = Trend.RANGE

    return TrendReading(trend, hh, hl, lh, ll, tuple(swing_highs), tuple(swing_lows))
