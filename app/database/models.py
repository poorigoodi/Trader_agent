"""SQLAlchemy models for the local audit trail.

Everything the agent looks at or decides is written here, because the point
of this phase is producing a record a human can review candle by candle:

  Candle      -- every OHLCV bar we evaluated, tagged with its source venue
  SignalLog   -- every decision, INCLUDING NO_TRADE, with all indicator
                 inputs frozen at decision time and the reason/invalidation
  PaperTrade  -- every simulated round trip
  RunLog      -- one row per process start, with the full effective config

Storing NO_TRADE rows is deliberate: "why did the agent not take that obvious
setup?" is unanswerable if only the trades are logged.

Every table carries ``exchange_id`` and ``is_dev_data_source`` so that rows
from different venues are never silently compared.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Candle(Base):
    """One OHLCV bar as evaluated by the agent."""

    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint(
            "exchange_id", "symbol", "timeframe", "timestamp", name="uq_candle"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange_id: Mapped[str] = mapped_column(String(32), index=True)
    is_dev_data_source: Mapped[bool] = mapped_column(Boolean, default=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SignalLog(Base):
    """One strategy decision, with every input it was made from."""

    __tablename__ = "signals"
    __table_args__ = (
        UniqueConstraint(
            "exchange_id", "symbol", "timeframe", "candle_timestamp", "strategy",
            name="uq_signal_per_candle",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)

    exchange_id: Mapped[str] = mapped_column(String(32), index=True)
    is_dev_data_source: Mapped[bool] = mapped_column(Boolean, default=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    candle_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    strategy: Mapped[str] = mapped_column(String(64))
    direction: Mapped[str] = mapped_column(String(16), index=True)
    reason: Mapped[str] = mapped_column(String(1024))
    invalidation: Mapped[str] = mapped_column(String(512))

    # Price context at decision time.
    close: Mapped[float] = mapped_column(Float)

    # --- frozen indicator inputs ---
    ema_fast: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_slow: Mapped[float | None] = mapped_column(Float, nullable=True)
    rsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd_hist: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_middle: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_percent_b: Mapped[float | None] = mapped_column(Float, nullable=True)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    trend: Mapped[str] = mapped_column(String(16), default="UNKNOWN")

    # Full snapshot + strategy metadata + risk decision, verbatim.
    snapshot_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    risk_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    trades: Mapped[list["PaperTrade"]] = relationship(back_populates="signal")


class PaperTrade(Base):
    """One simulated round trip. Never a real order."""

    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"), nullable=True)
    position_id: Mapped[str] = mapped_column(String(32), index=True)

    exchange_id: Mapped[str] = mapped_column(String(32), index=True)
    is_dev_data_source: Mapped[bool] = mapped_column(Boolean, default=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(16))
    strategy: Mapped[str] = mapped_column(String(64))

    quantity: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    risk_amount: Mapped[float] = mapped_column(Float)

    gross_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_fee: Mapped[float] = mapped_column(Float, default=0.0)
    exit_fee: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(16), default="OPEN", index=True)
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    invalidation: Mapped[str] = mapped_column(String(512), default="")

    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    signal: Mapped[SignalLog | None] = relationship(back_populates="trades")


class RunLog(Base):
    """One process start, with the exact configuration it ran under."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    exchange_id: Mapped[str] = mapped_column(String(32))
    is_dev_data_source: Mapped[bool] = mapped_column(Boolean, default=False)
    symbols: Mapped[str] = mapped_column(String(256))
    timeframe: Mapped[str] = mapped_column(String(8))
    mode: Mapped[str] = mapped_column(String(16), default="PAPER")

    cycles_completed: Mapped[int] = mapped_column(Integer, default=0)
    candles_evaluated: Mapped[int] = mapped_column(Integer, default=0)
    signals_generated: Mapped[int] = mapped_column(Integer, default=0)
    actionable_signals: Mapped[int] = mapped_column(Integer, default=0)
    trades_opened: Mapped[int] = mapped_column(Integer, default=0)
    trades_closed: Mapped[int] = mapped_column(Integer, default=0)

    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(String(512), default="")
