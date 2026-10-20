"""Computes the full indicator set once per candle and freezes a snapshot.

The snapshot is the *single* object the strategy is allowed to read, and the
exact same object is written to the database. That guarantees the logged
inputs are literally the inputs the decision was made from, not a
re-computation that might drift.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from app.config.settings import StrategySettings
from app.indicators.atr import atr
from app.indicators.bollinger import bollinger_bands
from app.indicators.macd import macd
from app.indicators.moving_averages import ema, sma
from app.indicators.rsi import rsi
from app.indicators.vwap import session_vwap


def _f(value: Any) -> float | None:
    """Convert a possibly-NaN numpy scalar to a JSON-safe float or None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # NaN is the only value != itself


@dataclass(frozen=True)
class IndicatorSnapshot:
    """Every indicator value at one specific candle close."""

    symbol: str
    timeframe: str
    timestamp: pd.Timestamp

    open: float
    high: float
    low: float
    close: float
    volume: float

    ema_fast: float | None
    ema_slow: float | None
    ema_fast_prev: float | None
    ema_slow_prev: float | None
    sma_slow: float | None

    rsi: float | None
    atr: float | None

    macd: float | None
    macd_signal: float | None
    macd_hist: float | None

    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    bb_percent_b: float | None

    vwap: float | None

    trend: str = "UNKNOWN"
    higher_highs: bool = False
    higher_lows: bool = False
    lower_highs: bool = False
    lower_lows: bool = False

    @property
    def is_complete(self) -> bool:
        """True when every value the strategy needs has warmed up."""
        required = (
            self.ema_fast,
            self.ema_slow,
            self.ema_fast_prev,
            self.ema_slow_prev,
            self.rsi,
            self.atr,
        )
        return all(v is not None for v in required)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


def compute_indicators(df: pd.DataFrame, settings: StrategySettings) -> pd.DataFrame:
    """Return a copy of `df` with every indicator column appended."""
    if df.empty:
        raise ValueError("cannot compute indicators on an empty frame")

    out = df.copy()
    out["ema_fast"] = ema(out, settings.ema_fast)
    out["ema_slow"] = ema(out, settings.ema_slow)
    out["sma_slow"] = sma(out, settings.ema_slow)
    out["rsi"] = rsi(out, settings.rsi_period)
    out["atr"] = atr(out, settings.atr_period)

    macd_result = macd(out, settings.macd_fast, settings.macd_slow, settings.macd_signal)
    out = out.join(macd_result.to_frame())

    bb = bollinger_bands(out, settings.bb_period, settings.bb_std)
    out = out.join(bb.to_frame())

    if "timestamp" in out.columns:
        out["vwap"] = session_vwap(out)
    else:
        out["vwap"] = float("nan")

    return out


def build_snapshot(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    index: int = -1,
) -> IndicatorSnapshot:
    """Freeze the indicator values at `index` (default: the last candle).

    `df` must already have been through compute_indicators().
    """
    if "ema_fast" not in df.columns:
        raise KeyError("frame has no indicator columns; call compute_indicators first")
    if len(df) < 2:
        raise ValueError("need at least 2 candles to build a snapshot")

    row = df.iloc[index]
    prev = df.iloc[index - 1] if index != 0 else row

    return IndicatorSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=pd.Timestamp(row["timestamp"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        ema_fast=_f(row["ema_fast"]),
        ema_slow=_f(row["ema_slow"]),
        ema_fast_prev=_f(prev["ema_fast"]),
        ema_slow_prev=_f(prev["ema_slow"]),
        sma_slow=_f(row["sma_slow"]),
        rsi=_f(row["rsi"]),
        atr=_f(row["atr"]),
        macd=_f(row["macd"]),
        macd_signal=_f(row["macd_signal"]),
        macd_hist=_f(row["macd_hist"]),
        bb_upper=_f(row["bb_upper"]),
        bb_middle=_f(row["bb_middle"]),
        bb_lower=_f(row["bb_lower"]),
        bb_percent_b=_f(row["bb_percent_b"]),
        vwap=_f(row["vwap"]),
    )
