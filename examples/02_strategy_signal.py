"""Evaluate the EMA+RSI+structure strategy on a synthetic uptrend.

Run from the repository root:

    .venv/Scripts/python.exe examples/02_strategy_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import StrategySettings
from app.indicators.engine import compute_indicators
from app.strategies.ema_rsi_trend import EmaRsiTrendStrategy
from tests.conftest import make_ohlcv


def staircase_up(pre: int = 100, cycles: int = 6) -> list[float]:
    """Decline, then a zig-zag climb (same pattern the unit tests use)."""
    values: list[float] = []
    price = 100.0
    for _ in range(pre):
        price -= 0.08
        values.append(price)
    for _ in range(cycles):
        for _ in range(5):
            price += 1.2
            values.append(price)
        for _ in range(4):
            price -= 0.7
            values.append(price)
    return values


def main() -> None:
    # cross_max_age=12 so a synthetic staircase can still print a fresh cross
    # after fractal structure confirms (see tests/test_strategy.py).
    settings = StrategySettings(cross_max_age=12, structure_lookback=30)
    frame = compute_indicators(make_ohlcv(staircase_up(), spread=0.2), settings)

    strategy = EmaRsiTrendStrategy(settings)
    signal = strategy.evaluate(frame.iloc[:117].reset_index(drop=True), "BTC/USDT", "15m")

    print(f"Strategy : {signal.strategy_name}")
    print(f"Direction: {signal.direction.value}")
    print(f"Actionable: {signal.is_actionable}")
    print("Reasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
    print(f"Invalidation: {signal.invalidation}")
    snap = signal.snapshot
    print(
        f"Snapshot  : close={snap.close:.2f}  "
        f"EMA20={snap.ema_fast:.2f}  EMA50={snap.ema_slow:.2f}  "
        f"RSI={snap.rsi:.1f}"
    )


if __name__ == "__main__":
    main()
