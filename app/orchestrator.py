"""Single-process orchestrator: the whole agent, in one readable loop.

Deliberately NOT a multi-agent system, not a message queue, and not a set of
services. One process, ordered function calls, explicit state. That is a
architectural decision for this phase, recorded in docs/architecture.md, and
it is what makes the behaviour reviewable at all.

One cycle, per symbol:

    fetch candles -> persist -> indicators -> signal -> persist signal
                                                  |
                     update open positions <------+ (stops/targets first)
                                                  |
                     risk gate -> size -> open paper trade -> persist

Positions are updated BEFORE new entries are considered: a stop that was hit
on this candle must be booked before the same candle is allowed to open
anything new.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.ai_decision.llm_layer import LLMDecisionLayer
from app.config.settings import Settings
from app.database.repository import Database
from app.indicators.engine import compute_indicators
from app.market.data_feed import MarketDataError, MarketDataFeed
from app.paper_trading.engine import Candle, ExitReason, PaperTradingEngine
from app.risk.correlation import ExposureItem
from app.risk.manager import RiskManager
from app.strategies.base import Signal, SignalDirection, Strategy
from app.strategies.ema_rsi_trend import EmaRsiTrendStrategy

logger = logging.getLogger(__name__)


@dataclass
class CycleStats:
    """What one pass over all symbols did. Aggregated into the run summary."""

    cycles: int = 0
    candles_fetched: int = 0
    candles_stored: int = 0
    signals_generated: int = 0
    actionable_signals: int = 0
    signals_by_direction: dict[str, int] = field(default_factory=dict)
    trades_opened: int = 0
    trades_closed: int = 0
    risk_rejections: int = 0
    llm_vetoes: int = 0
    llm_failures: int = 0
    errors: int = 0

    def record_signal(self, signal: Signal) -> None:
        self.signals_generated += 1
        key = signal.direction.value
        self.signals_by_direction[key] = self.signals_by_direction.get(key, 0) + 1
        if signal.is_actionable:
            self.actionable_signals += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles,
            "candles_fetched": self.candles_fetched,
            "candles_stored": self.candles_stored,
            "signals_generated": self.signals_generated,
            "actionable_signals": self.actionable_signals,
            "signals_by_direction": dict(self.signals_by_direction),
            "trades_opened": self.trades_opened,
            "trades_closed": self.trades_closed,
            "risk_rejections": self.risk_rejections,
            "llm_vetoes": self.llm_vetoes,
            "llm_failures": self.llm_failures,
            "errors": self.errors,
        }


class TradingAgent:
    """Wires the modules together. Owns no trading logic of its own."""

    def __init__(
        self,
        settings: Settings,
        feed: MarketDataFeed | None = None,
        database: Database | None = None,
        strategy: Strategy | None = None,
        paper_engine: PaperTradingEngine | None = None,
        risk_manager: RiskManager | None = None,
        llm_layer: LLMDecisionLayer | None = None,
    ) -> None:
        self.settings = settings
        self.feed = feed or MarketDataFeed(
            settings.exchange_id, market_type=settings.market_type
        )
        self.db = database or Database(settings.database_url)
        self.strategy = strategy or EmaRsiTrendStrategy(settings.strategy)
        self.paper = paper_engine or PaperTradingEngine(settings.paper)
        self.risk = risk_manager or RiskManager(settings.risk)
        # Off unless a provider is injected. The rule-based system is the
        # system; the LLM is an optional filter bolted onto it, never a
        # dependency.
        self.llm = llm_layer or LLMDecisionLayer(provider=None, enabled=False)

        self.stats = CycleStats()
        self.run_id: int | None = None
        self.last_marks: dict[str, float] = {}

        # Rolling close history per symbol, kept for the correlation matrix.
        self.closes: dict[str, pd.Series] = {}

        # Symbols actually tradeable on this venue. Populated by
        # verify_symbols(); until then we assume nothing.
        self.active_symbols: tuple[str, ...] = tuple(settings.symbols)
        self.dropped_symbols: dict[str, str] = {}

    # -- symbol verification ----------------------------------------------

    def verify_symbols(self) -> tuple[str, ...]:
        """Ask the venue which configured symbols it actually lists.

        Anything missing is dropped with a loud warning rather than failing
        quietly every cycle. If verification itself cannot run (LBank's perp
        host is unreachable from some networks), we keep the configured list
        and say so -- refusing to trade at all would be a worse failure than
        finding out per-symbol on the first fetch.
        """
        if not self.settings.verify_symbols:
            return self.active_symbols

        available, rejected = self.feed.verify_symbols(self.settings.symbols)
        self.dropped_symbols = rejected

        if not available and rejected:
            logger.warning(
                "symbol verification did not confirm any symbol on %s; "
                "continuing with the configured list and letting per-symbol "
                "fetches report failures. Reasons: %s",
                self.feed.exchange_id, rejected,
            )
            return self.active_symbols

        for symbol, why in rejected.items():
            logger.warning("SYMBOL DROPPED %s: %s", symbol, why)

        self.active_symbols = tuple(available)
        logger.info(
            "trading %d/%d configured symbols on %s (%s): %s",
            len(available), len(self.settings.symbols), self.feed.exchange_id,
            self.settings.market_type, ", ".join(available),
        )
        return self.active_symbols

    # -- run lifecycle ----------------------------------------------------

    def start_run(self, notes: str = "") -> int:
        self.run_id = self.db.start_run(
            exchange_id=self.feed.exchange_id,
            is_dev_data_source=self.feed.is_dev_data_source,
            symbols=self.settings.symbols,
            timeframe=self.settings.timeframe,
            config=self.settings.to_dict(),
            notes=notes,
        )
        logger.info(
            "run %s started | exchange=%s (dev_source=%s) symbols=%s tf=%s",
            self.run_id, self.feed.exchange_id, self.feed.is_dev_data_source,
            ",".join(self.settings.symbols), self.settings.timeframe,
        )
        return self.run_id

    def finish_run(self) -> None:
        if self.run_id is None:
            return
        self.db.finish_run(
            self.run_id,
            cycles_completed=self.stats.cycles,
            candles_evaluated=self.stats.candles_fetched,
            signals_generated=self.stats.signals_generated,
            actionable_signals=self.stats.actionable_signals,
            trades_opened=self.stats.trades_opened,
            trades_closed=self.stats.trades_closed,
        )

    # -- one symbol -------------------------------------------------------

    def process_symbol(self, symbol: str) -> Signal | None:
        """Run the full pipeline for one symbol. Returns the signal, if any."""
        tf = self.settings.timeframe
        try:
            frame = self.feed.fetch_ohlcv(symbol, tf, self.settings.candle_limit)
        except MarketDataError as exc:
            self.stats.errors += 1
            logger.error("skipping %s this cycle: %s", symbol, exc)
            return None

        if frame.empty or len(frame) < 2:
            self.stats.errors += 1
            logger.error("no usable candles for %s", symbol)
            return None

        self.stats.candles_fetched += len(frame)
        self.stats.candles_stored += self.db.save_candles(
            frame, symbol, tf, self.feed.exchange_id, self.feed.is_dev_data_source
        )

        last = frame.iloc[-1]
        self.last_marks[symbol] = float(last["close"])
        # Kept for the correlation matrix; indexed by timestamp so symbols
        # with different history lengths align instead of shifting silently.
        self.closes[symbol] = pd.Series(
            frame["close"].to_numpy(), index=frame["timestamp"].to_numpy()
        )

        # 1. Book stops/targets on the newest candle before anything else.
        closed = self.paper.update_position(
            symbol,
            Candle(
                timestamp=pd.Timestamp(last["timestamp"]).to_pydatetime(),
                high=float(last["high"]),
                low=float(last["low"]),
                close=float(last["close"]),
            ),
        )
        if closed is not None:
            self.db.save_closed_trade(closed)
            self.stats.trades_closed += 1

        # 2. Indicators, then the deterministic decision.
        enriched = compute_indicators(frame, self.settings.strategy)
        signal = self.strategy.evaluate(enriched, symbol, tf)

        # 2b. Optional LLM review. It sits BETWEEN strategy and risk, and can
        #     only annotate or downgrade to NO_TRADE -- never change direction,
        #     and never touch size or levels (those are computed after this).
        signal, review = self.llm.apply(signal)
        if review.vetoed:
            self.stats.llm_vetoes += 1
            logger.info("%s LLM VETO by %s: %s", symbol, review.provider, review.reasoning)
        if not review.ok and review.failure_reason:
            self.stats.llm_failures += 1

        self.stats.record_signal(signal)

        # 3. Risk gate. Runs even for NO_TRADE so the refusal reason is logged.
        risk_decision = None
        if signal.is_actionable:
            if self.paper.has_position(symbol):
                logger.info(
                    "%s %s ignored: a position is already open on this symbol",
                    symbol, signal.direction.value,
                )
            else:
                risk_decision = self.risk.evaluate(
                    signal,
                    balance=self.paper.wallet.balance,
                    open_positions=len(self.paper.open_positions),
                    open_exposures=self.open_exposures(),
                    closes_by_symbol=self.closes,
                )

        signal_id = self.db.save_signal(
            signal,
            exchange_id=self.feed.exchange_id,
            is_dev_data_source=self.feed.is_dev_data_source,
            run_id=self.run_id,
            risk_decision=risk_decision,
        )

        logger.info(
            "%s %s @ %.4f | %s",
            symbol, signal.direction.value, signal.snapshot.close, signal.reason_text,
        )

        # 4. Open the paper trade.
        if risk_decision is not None:
            if not risk_decision.approved:
                self.stats.risk_rejections += 1
                logger.info("%s risk gate refused: %s", symbol, risk_decision.reason)
            else:
                assert risk_decision.levels is not None and risk_decision.size is not None
                position = self.paper.open_position(
                    signal=signal,
                    levels=risk_decision.levels,
                    size=risk_decision.size,
                    exchange_id=self.feed.exchange_id,
                    is_dev_data_source=self.feed.is_dev_data_source,
                    opened_at=pd.Timestamp(last["timestamp"]).to_pydatetime(),
                    signal_id=signal_id,
                )
                self.risk.commit_risk(risk_decision.size.risk_amount)
                self.db.save_open_trade(position, run_id=self.run_id)
                self.stats.trades_opened += 1

        return signal

    # -- the loop ---------------------------------------------------------

    def open_exposures(self) -> list[ExposureItem]:
        """Signed risk currently open, for the correlation gate."""
        return [
            ExposureItem(p.symbol, p.direction, p.risk_amount)
            for p in self.paper.open_positions.values()
        ]

    def run_cycle(self) -> None:
        for symbol in self.active_symbols:
            self.process_symbol(symbol)
        self.stats.cycles += 1

    def run(self, max_cycles: int | None = None, sleep_fn=time.sleep) -> CycleStats:
        """Run cycles until ``max_cycles`` is reached (None = forever)."""
        limit = max_cycles if max_cycles is not None else self.settings.max_cycles
        if self.run_id is None:
            self.start_run()
        self.verify_symbols()

        cycle = 0
        try:
            while limit is None or cycle < limit:
                cycle += 1
                started = datetime.now(timezone.utc)
                logger.info("--- cycle %d ---", cycle)
                self.run_cycle()

                if limit is None or cycle < limit:
                    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                    sleep_fn(max(0.0, self.settings.cycle_interval_seconds - elapsed))
        except KeyboardInterrupt:
            logger.info("interrupted by user after %d cycle(s)", cycle)
        finally:
            self.finish_run()

        return self.stats

    # -- reporting --------------------------------------------------------

    def close_all(self, reason: ExitReason = ExitReason.SESSION_END) -> int:
        """Close every open position at its last known mark. Reporting only."""
        closed = 0
        for position_id, position in list(self.paper.open_positions.items()):
            mark = self.last_marks.get(position.symbol)
            if mark is None:
                continue
            trade = self.paper.close_position(position_id, mark, reason)
            self.db.save_closed_trade(trade)
            closed += 1
            self.stats.trades_closed += 1
        return closed

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "exchange_id": self.feed.exchange_id,
            "market_type": self.settings.market_type,
            "target_exchange_id": self.settings.target_exchange_id,
            "is_dev_data_source": self.feed.is_dev_data_source,
            "symbols_configured": list(self.settings.symbols),
            "symbols_active": list(self.active_symbols),
            "symbols_dropped": dict(self.dropped_symbols),
            "timeframe": self.settings.timeframe,
            "strategy": self.strategy.name,
            **self.stats.to_dict(),
            "portfolio": self.paper.summary(self.last_marks),
        }
