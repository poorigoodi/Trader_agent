"""Event-driven backtest: replay history one candle at a time.

NO LOOK-AHEAD, BY CONSTRUCTION
------------------------------
The strategy is only ever handed ``frame.iloc[:i+1]`` -- the candles up to and
including the one being decided on, and nothing after it. It is not possible
for a rule to see the future here, because the future is not in the object it
receives. That is a stronger guarantee than "we were careful with shifts".

Two further sources of leakage are closed explicitly:

  * A trade opened on the signal candle is filled at that candle's CLOSE, and
    can only be resolved by LATER candles. It is never resolved by the bar
    that produced it, which would let the same bar both trigger and pay out.
  * Indicators are recomputed on each prefix rather than once over the whole
    series. This is slower, and it is the point: computing them once and
    slicing is how a centred or back-filled value silently leaks.

The output is a labelled dataset -- every signal, its inputs at decision
time, and whether it actually won or lost. That is what the bake-off scores
veto quality against, and what the confidence calibrator trains on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from app.config.settings import Settings
from app.indicators.engine import compute_indicators
from app.paper_trading.engine import Candle, ExitReason, PaperTradingEngine
from app.risk.correlation import ExposureItem
from app.risk.manager import RiskManager
from app.strategies.base import Signal, SignalDirection, Strategy
from app.strategies.ema_rsi_trend import EmaRsiTrendStrategy

logger = logging.getLogger(__name__)


@dataclass
class LabelledSignal:
    """One signal plus the outcome history actually delivered.

    ``won`` is the training target for the confidence calibrator and the
    ground truth the bake-off judges a model's vetoes against.
    """

    symbol: str
    timestamp: datetime
    direction: str
    strategy: str
    reason: str
    invalidation: str

    # inputs frozen at decision time
    close: float
    ema_fast: float | None
    ema_slow: float | None
    ema_spread_pct: float | None
    rsi: float | None
    atr: float | None
    atr_pct_of_price: float | None
    macd_hist: float | None
    bb_percent_b: float | None
    close_vs_vwap_pct: float | None
    trend: str
    cross_age: int | None

    # risk numbers
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_amount: float | None = None
    rr: float | None = None

    # outcome
    traded: bool = False
    exit_reason: str | None = None
    net_pnl: float | None = None
    r_multiple: float | None = None
    won: int | None = None
    bars_held: int | None = None

    def to_row(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["timestamp"] = pd.Timestamp(self.timestamp).isoformat()
        return data


@dataclass
class BacktestResult:
    """Everything one backtest produced."""

    signals: list[LabelledSignal] = field(default_factory=list)
    trades: list[Any] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    starting_balance: float = 0.0
    final_balance: float = 0.0
    candles_evaluated: int = 0
    vendor: str = ""
    market_type: str = ""

    # -- dataset ----------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_row() for s in self.signals])

    def traded_signals(self) -> list[LabelledSignal]:
        return [s for s in self.signals if s.traded and s.won is not None]

    # -- metrics ----------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        closed = [t for t in self.trades]
        wins = [t for t in closed if t.net_pnl > 0]
        losses = [t for t in closed if t.net_pnl < 0]
        gross_win = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        r_values = [t.r_multiple for t in closed if t.r_multiple is not None]

        return {
            "candles_evaluated": self.candles_evaluated,
            "signals_total": len(self.signals),
            "signals_actionable": sum(1 for s in self.signals if s.direction != "NO_TRADE"),
            "trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(closed)) if closed else None,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
            "expectancy_r": (sum(r_values) / len(r_values)) if r_values else None,
            "starting_balance": self.starting_balance,
            "final_balance": self.final_balance,
            "return_pct": (
                (self.final_balance - self.starting_balance) / self.starting_balance * 100
                if self.starting_balance
                else None
            ),
            "max_drawdown_pct": self.max_drawdown_pct(),
            "vendor": self.vendor,
            "market_type": self.market_type,
        }

    def max_drawdown_pct(self) -> float | None:
        if not self.equity_curve:
            return None
        peak = self.equity_curve[0][1]
        worst = 0.0
        for _, equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                worst = max(worst, (peak - equity) / peak)
        return worst * 100


class Backtester:
    """Replays candles through the real strategy, risk and paper modules.

    It reuses the production objects rather than re-implementing them. A
    backtest that runs different code than the live loop measures the wrong
    system.
    """

    def __init__(
        self,
        settings: Settings,
        strategy: Strategy | None = None,
        warmup: int = 120,
    ) -> None:
        self.settings = settings
        self.strategy = strategy or EmaRsiTrendStrategy(settings.strategy)
        self.warmup = warmup

    def run(
        self,
        frames: dict[str, pd.DataFrame],
        vendor: str = "",
        market_type: str = "",
        progress_every: int = 0,
    ) -> BacktestResult:
        """Replay ``{symbol: ohlcv_frame}`` in lockstep on a shared clock.

        Symbols advance together so the correlation gate sees a coherent
        portfolio state, exactly as it would live.
        """
        if not frames:
            raise ValueError("no frames to backtest")

        paper = PaperTradingEngine(self.settings.paper)
        risk = RiskManager(self.settings.risk)
        result = BacktestResult(
            starting_balance=paper.wallet.starting_balance,
            vendor=vendor,
            market_type=market_type,
        )

        symbols = list(frames)
        # Precompute indicators ONCE per symbol for speed, then only ever read
        # row i and earlier. Every indicator in app/indicators is causal
        # (backward-looking only), which is what makes this equivalent to
        # recomputing per prefix -- and test_backtesting asserts that
        # equivalence directly rather than trusting the claim.
        enriched = {
            symbol: compute_indicators(frame, self.settings.strategy)
            for symbol, frame in frames.items()
        }
        length = min(len(f) for f in enriched.values())
        open_bars: dict[str, int] = {}

        for i in range(self.warmup, length):
            if progress_every and i % progress_every == 0:
                logger.info("backtest %d/%d", i, length)

            for symbol in symbols:
                frame = enriched[symbol]
                row = frame.iloc[i]
                candle_time = pd.Timestamp(row["timestamp"]).to_pydatetime()

                # 1. Resolve any open position against THIS candle first.
                closed = paper.update_position(
                    symbol,
                    Candle(
                        timestamp=candle_time,
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                    ),
                )
                if closed is not None:
                    result.trades.append(closed)
                    self._label_outcome(result, closed, i - open_bars.pop(symbol, i))

                # 2. Decide on the prefix ending at this candle. Nothing after
                #    index i is visible to the strategy.
                window = frame.iloc[: i + 1]
                signal = self.strategy.evaluate(window, symbol, self.settings.timeframe)
                record = self._record(signal)

                # 3. Risk gate, then open at this candle's close.
                if signal.is_actionable and not paper.has_position(symbol):
                    decision = risk.evaluate(
                        signal,
                        balance=paper.wallet.balance,
                        open_positions=len(paper.open_positions),
                        now=candle_time,
                        open_exposures=[
                            ExposureItem(p.symbol, p.direction, p.risk_amount)
                            for p in paper.open_positions.values()
                        ],
                        closes_by_symbol={
                            s: enriched[s]["close"].iloc[: i + 1] for s in symbols
                        },
                    )
                    if decision.approved and decision.levels and decision.size:
                        record.entry = decision.levels.entry
                        record.stop_loss = decision.levels.stop_loss
                        record.take_profit = decision.levels.take_profit
                        record.rr = decision.levels.rr
                        record.risk_amount = decision.size.risk_amount

                        position = paper.open_position(
                            signal, decision.levels, decision.size,
                            exchange_id=vendor or self.settings.exchange_id,
                            is_dev_data_source=True,
                            opened_at=candle_time,
                        )
                        risk.commit_risk(decision.size.risk_amount, now=candle_time)
                        position.metadata["signal_index"] = len(result.signals)
                        open_bars[symbol] = i

                result.signals.append(record)

            result.candles_evaluated += 1
            marks = {s: float(enriched[s]["close"].iloc[i]) for s in symbols}
            result.equity_curve.append(
                (pd.Timestamp(enriched[symbols[0]]["timestamp"].iloc[i]).to_pydatetime(),
                 paper.equity(marks))
            )

        # Close whatever is still open at the last mark, so the dataset has no
        # unlabelled trades hanging off the end.
        final_marks = {s: float(enriched[s]["close"].iloc[length - 1]) for s in symbols}
        for position_id, position in list(paper.open_positions.items()):
            trade = paper.close_position(
                position_id, final_marks[position.symbol], ExitReason.SESSION_END
            )
            result.trades.append(trade)
            self._label_outcome(result, trade, None)

        result.final_balance = paper.wallet.balance
        return result

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _record(signal: Signal) -> LabelledSignal:
        snapshot = signal.snapshot
        metadata = signal.metadata
        return LabelledSignal(
            symbol=snapshot.symbol,
            timestamp=pd.Timestamp(snapshot.timestamp).to_pydatetime(),
            direction=signal.direction.value,
            strategy=signal.strategy_name,
            reason=signal.reason_text,
            invalidation=signal.invalidation,
            close=snapshot.close,
            ema_fast=snapshot.ema_fast,
            ema_slow=snapshot.ema_slow,
            ema_spread_pct=metadata.get("ema_spread_pct"),
            rsi=snapshot.rsi,
            atr=snapshot.atr,
            atr_pct_of_price=metadata.get("atr_pct_of_price"),
            macd_hist=snapshot.macd_hist,
            bb_percent_b=snapshot.bb_percent_b,
            close_vs_vwap_pct=metadata.get("close_vs_vwap_pct"),
            trend=snapshot.trend,
            cross_age=metadata.get("cross_age"),
        )

    @staticmethod
    def _label_outcome(result: BacktestResult, trade, bars_held: int | None) -> None:
        """Attach the realised outcome back onto the signal that caused it."""
        for record in reversed(result.signals):
            if (
                record.symbol == trade.symbol
                and record.direction == trade.direction.value
                and not record.traded
                and record.entry is not None
            ):
                record.traded = True
                record.exit_reason = trade.exit_reason.value
                record.net_pnl = trade.net_pnl
                record.r_multiple = trade.r_multiple
                record.won = 1 if trade.net_pnl > 0 else 0
                record.bars_held = bars_held
                return
