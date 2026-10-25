"""Compute EMA, RSI and ATR on synthetic candles — no network.

Run from the repository root:

    .venv/Scripts/python.exe examples/01_indicators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.config.settings import StrategySettings
from app.indicators.engine import compute_indicators
from tests.conftest import make_ohlcv


def main() -> None:
    rng = np.random.default_rng(42)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.4, 120))
    frame = make_ohlcv(closes, spread=0.3)

    settings = StrategySettings()
    enriched = compute_indicators(frame, settings)
    last = enriched.iloc[-1]

    print("Indicator snapshot (last closed candle)")
    print(f"  close     {last['close']:.4f}")
    print(f"  EMA({settings.ema_fast})  {last['ema_fast']:.4f}")
    print(f"  EMA({settings.ema_slow})  {last['ema_slow']:.4f}")
    print(f"  RSI({settings.rsi_period})  {last['rsi']:.2f}")
    print(f"  ATR({settings.atr_period})  {last['atr']:.4f}")
    print(f"  MACD hist {last['macd_hist']:.4f}")
    print(f"  BB %B     {last['bb_percent_b']:.3f}")


if __name__ == "__main__":
    main()
