"""Persistence helpers. The only module that opens a database session.

"Migrations" at this stage are ``Base.metadata.create_all`` -- appropriate for
a single-developer SQLite MVP. Alembic becomes worth its weight once the
schema is being changed while a paper run is live; see README "Open
Decisions" (OD-3).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base, Candle, PaperTrade, RunLog, SignalLog
from app.paper_trading.engine import ClosedTrade, PaperPosition
from app.risk.manager import RiskDecision
from app.strategies.base import Signal

logger = logging.getLogger(__name__)


class Database:
    """Thin wrapper over a SQLAlchemy engine + session factory."""

    def __init__(self, url: str = "sqlite:///data/trading_agent.db", echo: bool = False) -> None:
        self.url = url
        if url.startswith("sqlite:///") and not url.endswith(":memory:"):
            path = Path(url.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(url, echo=echo, future=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -- runs -------------------------------------------------------------

    def start_run(
        self,
        exchange_id: str,
        is_dev_data_source: bool,
        symbols: tuple[str, ...],
        timeframe: str,
        config: dict[str, Any],
        notes: str = "",
    ) -> int:
        with self.session() as s:
            run = RunLog(
                exchange_id=exchange_id,
                is_dev_data_source=is_dev_data_source,
                symbols=",".join(symbols),
                timeframe=timeframe,
                mode="PAPER",
                config_json=_jsonable(config),
                notes=notes,
            )
            s.add(run)
            s.flush()
            return run.id

    def finish_run(self, run_id: int, **counters: Any) -> None:
        with self.session() as s:
            run = s.get(RunLog, run_id)
            if run is None:
                return
            run.finished_at = datetime.now(timezone.utc)
            for key, value in counters.items():
                if hasattr(run, key):
                    setattr(run, key, value)

    # -- candles ----------------------------------------------------------

    def save_candles(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        exchange_id: str,
        is_dev_data_source: bool,
    ) -> int:
        """Insert candles that are not already stored. Returns the new count.

        De-duplication is by (exchange, symbol, timeframe, timestamp): polling
        re-fetches overlapping windows constantly, and a duplicated bar would
        corrupt any later count of "how much data do we actually have".
        """
        if df.empty:
            return 0

        with self.session() as s:
            existing = set(
                s.scalars(
                    select(Candle.timestamp).where(
                        Candle.exchange_id == exchange_id,
                        Candle.symbol == symbol,
                        Candle.timeframe == timeframe,
                    )
                ).all()
            )
            existing = {_as_utc(ts) for ts in existing}

            rows = []
            for record in df.itertuples(index=False):
                ts = _as_utc(pd.Timestamp(record.timestamp).to_pydatetime())
                if ts in existing:
                    continue
                existing.add(ts)
                rows.append(
                    Candle(
                        exchange_id=exchange_id,
                        is_dev_data_source=is_dev_data_source,
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        open=float(record.open),
                        high=float(record.high),
                        low=float(record.low),
                        close=float(record.close),
                        volume=float(record.volume),
                    )
                )
            s.add_all(rows)
            return len(rows)

    def candle_count(self, symbol: str | None = None) -> int:
        from sqlalchemy import func

        with self.session() as s:
            stmt = select(func.count()).select_from(Candle)
            if symbol:
                stmt = stmt.where(Candle.symbol == symbol)
            return int(s.scalar(stmt) or 0)

    # -- signals ----------------------------------------------------------

    def save_signal(
        self,
        signal: Signal,
        exchange_id: str,
        is_dev_data_source: bool,
        run_id: int | None = None,
        risk_decision: RiskDecision | None = None,
    ) -> int | None:
        """Persist one decision. Returns the row id, or None if already stored.

        Re-running a cycle before a new candle closes must not create a second
        row for the same candle, hence the uniqueness check.
        """
        snap = signal.snapshot
        candle_ts = _as_utc(pd.Timestamp(snap.timestamp).to_pydatetime())

        with self.session() as s:
            duplicate = s.scalar(
                select(SignalLog.id).where(
                    SignalLog.exchange_id == exchange_id,
                    SignalLog.symbol == snap.symbol,
                    SignalLog.timeframe == snap.timeframe,
                    SignalLog.candle_timestamp == candle_ts,
                    SignalLog.strategy == signal.strategy_name,
                )
            )
            if duplicate is not None:
                return None

            row = SignalLog(
                run_id=run_id,
                exchange_id=exchange_id,
                is_dev_data_source=is_dev_data_source,
                symbol=snap.symbol,
                timeframe=snap.timeframe,
                candle_timestamp=candle_ts,
                strategy=signal.strategy_name,
                direction=signal.direction.value,
                reason=signal.reason_text[:1024],
                invalidation=signal.invalidation[:512],
                close=snap.close,
                ema_fast=snap.ema_fast,
                ema_slow=snap.ema_slow,
                rsi=snap.rsi,
                atr=snap.atr,
                macd=snap.macd,
                macd_signal=snap.macd_signal,
                macd_hist=snap.macd_hist,
                bb_upper=snap.bb_upper,
                bb_middle=snap.bb_middle,
                bb_lower=snap.bb_lower,
                bb_percent_b=snap.bb_percent_b,
                vwap=snap.vwap,
                trend=snap.trend,
                snapshot_json=_jsonable(snap.to_dict()),
                metadata_json=_jsonable(signal.metadata),
                risk_json=_jsonable(risk_decision.to_dict()) if risk_decision else None,
            )
            s.add(row)
            s.flush()
            return row.id

    # -- trades -----------------------------------------------------------

    def save_open_trade(self, position: PaperPosition, run_id: int | None = None) -> int:
        with self.session() as s:
            row = PaperTrade(
                run_id=run_id,
                signal_id=position.signal_id,
                position_id=position.position_id,
                exchange_id=position.exchange_id,
                is_dev_data_source=position.is_dev_data_source,
                symbol=position.symbol,
                direction=position.direction.value,
                strategy=position.strategy_name,
                quantity=position.quantity,
                entry_price=position.entry_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                risk_amount=position.risk_amount,
                entry_fee=position.entry_fee,
                status="OPEN",
                invalidation=position.invalidation[:512],
                opened_at=_as_utc(position.opened_at),
            )
            s.add(row)
            s.flush()
            return row.id

    def save_closed_trade(self, trade: ClosedTrade) -> None:
        with self.session() as s:
            row = s.scalar(
                select(PaperTrade).where(PaperTrade.position_id == trade.position_id)
            )
            if row is None:
                logger.warning(
                    "closing trade %s that was never recorded as open", trade.position_id
                )
                return
            row.exit_price = trade.exit_price
            row.gross_pnl = trade.gross_pnl
            row.exit_fee = trade.exit_fee
            row.net_pnl = trade.net_pnl
            row.r_multiple = trade.r_multiple
            row.status = "CLOSED"
            row.exit_reason = trade.exit_reason.value
            row.closed_at = _as_utc(trade.closed_at)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _as_utc(value: datetime) -> datetime:
    """Normalise to timezone-aware UTC.

    SQLite drops tzinfo on the way out, so values read back are naive. We
    treat those as UTC (which is what we wrote) rather than letting a naive/
    aware comparison raise mid-run.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _jsonable(value: Any) -> Any:
    """Recursively convert to something the JSON column can store."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
