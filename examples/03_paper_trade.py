"""Size a LONG, open a paper position, then hit the take-profit.

Run from the repository root:

    .venv/Scripts/python.exe examples/03_paper_trade.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import PaperTradingSettings, RiskSettings, StrategySettings
from app.indicators.engine import compute_indicators
from app.paper_trading.engine import Candle, PaperTradingEngine
from app.risk.levels import compute_levels
from app.risk.position_sizing import calculate_position_size
from app.strategies.ema_rsi_trend import EmaRsiTrendStrategy
from tests.conftest import make_ohlcv


def staircase_up() -> list[float]:
    values: list[float] = []
    price = 100.0
    for _ in range(100):
        price -= 0.08
        values.append(price)
    for _ in range(6):
        for _ in range(5):
            price += 1.2
            values.append(price)
        for _ in range(4):
            price -= 0.7
            values.append(price)
    return values


def main() -> None:
    strategy_settings = StrategySettings(cross_max_age=12, structure_lookback=30)
    frame = compute_indicators(make_ohlcv(staircase_up(), spread=0.2), strategy_settings)
    signal = EmaRsiTrendStrategy(strategy_settings).evaluate(
        frame.iloc[:117].reset_index(drop=True), "BTC/USDT", "15m"
    )
    if not signal.is_actionable:
        raise SystemExit(f"expected an actionable signal, got {signal.direction}")

    atr = signal.snapshot.atr
    assert atr is not None
    levels = compute_levels(signal.direction, entry=signal.snapshot.close, atr=atr)
    size = calculate_position_size(
        balance=10_000.0,
        levels=levels,
        settings=RiskSettings(),
    )
    if not size.accepted:
        raise SystemExit(f"size rejected: {size.rejection_reason}")

    engine = PaperTradingEngine(
        settings=PaperTradingSettings(starting_balance=10_000.0),
    )
    position = engine.open_position(
        signal=signal,
        levels=levels,
        size=size,
        exchange_id="example",
        is_dev_data_source=True,
        opened_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    # Candle that tags the take-profit (high clears TP; low stays above stop).
    closed = engine.update_position(
        "BTC/USDT",
        Candle(
            timestamp=datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
            high=levels.take_profit + 0.5,
            low=position.entry_price - 0.1,
            close=levels.take_profit + 0.2,
        ),
    )

    print(f"Opened   : {position.direction.value} {position.symbol}")
    print(f"  qty      {position.quantity:.6f}")
    print(f"  entry    {position.entry_price:.4f}  (ref {position.reference_price:.4f})")
    print(f"  stop     {position.stop_loss:.4f}")
    print(f"  target   {position.take_profit:.4f}")
    print(f"  risk $   {position.risk_amount:.2f}")
    if closed is None:
        print("Exit     : still open (unexpected for this demo)")
        return
    print(f"Closed   : {closed.exit_reason.value} @ {closed.exit_price:.4f}")
    print(f"  net PnL  {closed.net_pnl:+.2f} USDT")
    print(f"  R mult   {closed.r_multiple:+.2f}" if closed.r_multiple is not None else "")
    print(f"Wallet   : {engine.wallet.balance:.2f} USDT")


if __name__ == "__main__":
    main()
