"""Unit tests for the SQLite audit trail.

All of these run against an in-memory database. The behaviours that matter:
  * candles and signals de-duplicate, because polling re-reads the same bars
  * NO_TRADE decisions are stored too -- "why did it not trade?" must be
    answerable from the journal alone
  * every row records which venue it came from
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import select

from app.config.settings import RiskSettings, Settings
from app.database.models import Candle, PaperTrade, RunLog, SignalLog
from app.database.repository import Database
from app.indicators.engine import IndicatorSnapshot
from app.paper_trading.engine import ExitReason, PaperTradingEngine
from app.risk.levels import compute_levels
from app.risk.manager import RiskManager
from app.risk.position_sizing import calculate_position_size
from app.strategies.base import Signal, SignalDirection
from tests.conftest import make_ohlcv

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def db() -> Database:
    return Database("sqlite:///:memory:")


def snapshot(symbol="BTC/USDT:USDT", ts=None, close=100.0) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        symbol=symbol, timeframe="15m",
        timestamp=pd.Timestamp(ts or T0),
        open=close, high=close + 1, low=close - 1, close=close, volume=5.0,
        ema_fast=101.0, ema_slow=100.0, ema_fast_prev=99.0, ema_slow_prev=100.0,
        sma_slow=100.0, rsi=55.5, atr=2.0,
        macd=0.1, macd_signal=0.05, macd_hist=0.05,
        bb_upper=105.0, bb_middle=100.0, bb_lower=95.0, bb_percent_b=0.5,
        vwap=99.5, trend="UP",
    )


def signal(direction=SignalDirection.LONG, **kwargs) -> Signal:
    return Signal(
        direction=direction,
        reasons=("EMA20 > EMA50", "RSI 55.5 within [50, 70]"),
        invalidation="close back below EMA50",
        snapshot=snapshot(**kwargs),
        strategy_name="ema_rsi_trend_v1",
        metadata={"cross_age": 1, "macd_hist": 0.05},
    )


# --------------------------------------------------------------------------
# schema / runs
# --------------------------------------------------------------------------

def test_tables_are_created_on_construction(db):
    with db.session() as s:
        assert s.scalars(select(Candle)).all() == []
        assert s.scalars(select(SignalLog)).all() == []
        assert s.scalars(select(PaperTrade)).all() == []


def test_start_run_records_the_effective_config(db):
    settings = Settings()
    run_id = db.start_run(
        "toobit", False, settings.symbols, "15m", settings.to_dict(), notes="test"
    )
    with db.session() as s:
        run = s.get(RunLog, run_id)
        assert run.exchange_id == "toobit"
        assert run.mode == "PAPER"
        assert "BTC/USDT:USDT" in run.symbols
        assert run.config_json["risk"]["risk_per_trade_pct"] == 0.01
        assert run.finished_at is None


def test_finish_run_stamps_counters_and_time(db):
    run_id = db.start_run("toobit", False, ("BTC/USDT:USDT",), "15m", {})
    db.finish_run(run_id, cycles_completed=3, signals_generated=12, trades_opened=1)
    with db.session() as s:
        run = s.get(RunLog, run_id)
        assert run.cycles_completed == 3
        assert run.signals_generated == 12
        assert run.finished_at is not None


def test_finish_run_on_a_missing_id_is_a_no_op(db):
    db.finish_run(99_999, cycles_completed=1)  # must not raise


# --------------------------------------------------------------------------
# candles
# --------------------------------------------------------------------------

def test_candles_are_saved(db):
    frame = make_ohlcv([100.0, 101.0, 102.0])
    assert db.save_candles(frame, "BTC/USDT:USDT", "15m", "toobit", False) == 3
    assert db.candle_count() == 3


def test_re_saving_the_same_candles_inserts_nothing(db):
    """Polling re-fetches overlapping windows every cycle."""
    frame = make_ohlcv([100.0, 101.0, 102.0])
    db.save_candles(frame, "BTC/USDT:USDT", "15m", "toobit", False)
    assert db.save_candles(frame, "BTC/USDT:USDT", "15m", "toobit", False) == 0
    assert db.candle_count() == 3


def test_an_overlapping_window_inserts_only_the_new_bars(db):
    db.save_candles(make_ohlcv([100.0, 101.0, 102.0]), "BTC/USDT:USDT", "15m", "toobit", False)
    grown = make_ohlcv([100.0, 101.0, 102.0, 103.0, 104.0])
    assert db.save_candles(grown, "BTC/USDT:USDT", "15m", "toobit", False) == 2
    assert db.candle_count() == 5


def test_the_same_bar_on_two_venues_is_two_rows(db):
    """Different venues print different prices; collapsing them would be a
    silent basis error."""
    frame = make_ohlcv([100.0, 101.0])
    db.save_candles(frame, "BTC/USDT:USDT", "15m", "toobit", False)
    db.save_candles(frame, "BTC/USDT:USDT", "15m", "binance", True)
    assert db.candle_count() == 4


def test_candle_count_filters_by_symbol(db):
    db.save_candles(make_ohlcv([1.0, 2.0]), "BTC/USDT:USDT", "15m", "toobit", False)
    db.save_candles(make_ohlcv([1.0, 2.0, 3.0]), "ETH/USDT:USDT", "15m", "toobit", False)
    assert db.candle_count("BTC/USDT:USDT") == 2
    assert db.candle_count("ETH/USDT:USDT") == 3
    assert db.candle_count() == 5


def test_saving_an_empty_frame_is_a_no_op(db):
    assert db.save_candles(make_ohlcv([]), "BTC/USDT:USDT", "15m", "toobit", False) == 0


def test_candle_records_its_provenance(db):
    db.save_candles(make_ohlcv([100.0]), "BTC/USDT:USDT", "15m", "binance", True)
    with db.session() as s:
        row = s.scalars(select(Candle)).one()
        assert row.exchange_id == "binance"
        assert row.is_dev_data_source is True


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------

def test_signal_is_saved_with_its_indicator_inputs(db):
    signal_id = db.save_signal(signal(), "toobit", False)
    with db.session() as s:
        row = s.get(SignalLog, signal_id)
        assert row.direction == "LONG"
        assert row.rsi == pytest.approx(55.5)
        assert row.atr == pytest.approx(2.0)
        assert row.ema_fast == pytest.approx(101.0)
        assert row.trend == "UP"
        assert row.close == pytest.approx(100.0)


def test_no_trade_signals_are_stored_too(db):
    """The journal must be able to answer 'why did it skip that setup?'."""
    no_trade = Signal(
        direction=SignalDirection.NO_TRADE,
        reasons=("EMA cross is stale: 9 candles old, max 3",),
        invalidation="a new EMA cross would open a setup",
        snapshot=snapshot(),
        strategy_name="ema_rsi_trend_v1",
    )
    signal_id = db.save_signal(no_trade, "toobit", False)
    with db.session() as s:
        row = s.get(SignalLog, signal_id)
        assert row.direction == "NO_TRADE"
        assert "stale" in row.reason


def test_reason_and_invalidation_are_persisted(db):
    signal_id = db.save_signal(signal(), "toobit", False)
    with db.session() as s:
        row = s.get(SignalLog, signal_id)
        assert "EMA20 > EMA50" in row.reason
        assert row.invalidation == "close back below EMA50"


def test_full_snapshot_and_metadata_are_stored_as_json(db):
    signal_id = db.save_signal(signal(), "toobit", False)
    with db.session() as s:
        row = s.get(SignalLog, signal_id)
        assert row.snapshot_json["rsi"] == pytest.approx(55.5)
        assert row.snapshot_json["bb_percent_b"] == pytest.approx(0.5)
        assert row.metadata_json["cross_age"] == 1


def test_the_same_candle_is_not_logged_twice(db):
    """Cycles run more often than candles close."""
    assert db.save_signal(signal(), "toobit", False) is not None
    assert db.save_signal(signal(), "toobit", False) is None


def test_a_new_candle_is_logged_again(db):
    db.save_signal(signal(), "toobit", False)
    later = signal(ts=T0 + timedelta(minutes=15))
    assert db.save_signal(later, "toobit", False) is not None


def test_the_same_candle_on_two_venues_is_two_signal_rows(db):
    assert db.save_signal(signal(), "toobit", False) is not None
    assert db.save_signal(signal(), "binance", True) is not None


def test_risk_decision_is_stored_alongside_the_signal(db):
    decision = RiskManager().evaluate(signal(), balance=10_000.0)
    signal_id = db.save_signal(signal(), "toobit", False, risk_decision=decision)
    with db.session() as s:
        row = s.get(SignalLog, signal_id)
        assert row.risk_json["approved"] is True
        assert row.risk_json["size"]["risk_amount"] == pytest.approx(100.0)
        assert row.risk_json["levels"]["stop_loss"] == pytest.approx(97.0)


def test_a_refusal_is_stored_with_its_reason(db):
    decision = RiskManager().evaluate(signal(), balance=20.0)  # below min notional
    signal_id = db.save_signal(signal(), "toobit", False, risk_decision=decision)
    with db.session() as s:
        row = s.get(SignalLog, signal_id)
        assert row.risk_json["approved"] is False
        assert row.risk_json["reason"]


def test_signal_links_to_its_run(db):
    run_id = db.start_run("toobit", False, ("BTC/USDT:USDT",), "15m", {})
    signal_id = db.save_signal(signal(), "toobit", False, run_id=run_id)
    with db.session() as s:
        assert s.get(SignalLog, signal_id).run_id == run_id


# --------------------------------------------------------------------------
# trades
# --------------------------------------------------------------------------

def open_a_position(engine, symbol="BTC/USDT:USDT"):
    risk = RiskSettings()
    sig = signal(symbol=symbol)
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    size = calculate_position_size(10_000.0, levels, risk)
    return engine.open_position(sig, levels, size, "toobit", False, opened_at=T0)


def test_open_trade_is_recorded(db):
    engine = PaperTradingEngine()
    position = open_a_position(engine)
    db.save_open_trade(position)
    with db.session() as s:
        row = s.scalars(select(PaperTrade)).one()
        assert row.status == "OPEN"
        assert row.direction == "LONG"
        assert row.exit_price is None
        assert row.stop_loss == pytest.approx(97.0)


def test_closing_updates_the_same_row(db):
    engine = PaperTradingEngine()
    position = open_a_position(engine)
    db.save_open_trade(position)
    trade = engine.close_position(position.position_id, 106.0, ExitReason.TAKE_PROFIT, T0)
    db.save_closed_trade(trade)

    with db.session() as s:
        rows = s.scalars(select(PaperTrade)).all()
        assert len(rows) == 1  # updated, not duplicated
        assert rows[0].status == "CLOSED"
        assert rows[0].exit_reason == "TAKE_PROFIT"
        assert rows[0].net_pnl is not None
        assert rows[0].closed_at is not None


def test_closing_an_unrecorded_trade_warns_but_does_not_raise(db):
    engine = PaperTradingEngine()
    position = open_a_position(engine)
    trade = engine.close_position(position.position_id, 106.0, ExitReason.MANUAL, T0)
    db.save_closed_trade(trade)  # never saved as open; must not raise
    assert db.candle_count() == 0


def test_trade_links_to_its_signal(db):
    engine = PaperTradingEngine()
    signal_id = db.save_signal(signal(), "toobit", False)
    position = open_a_position(engine)
    position.signal_id = signal_id
    db.save_open_trade(position)
    with db.session() as s:
        assert s.scalars(select(PaperTrade)).one().signal_id == signal_id


def test_trade_records_its_provenance_and_invalidation(db):
    engine = PaperTradingEngine()
    position = open_a_position(engine)
    db.save_open_trade(position)
    with db.session() as s:
        row = s.scalars(select(PaperTrade)).one()
        assert row.exchange_id == "toobit"
        assert row.is_dev_data_source is False
        assert row.invalidation == "close back below EMA50"


# --------------------------------------------------------------------------
# serialisation edge cases
# --------------------------------------------------------------------------

def test_nan_becomes_null_rather_than_breaking_json(db):
    from dataclasses import replace

    sig = signal()
    sig = replace(sig, snapshot=replace(sig.snapshot, vwap=float("nan")))
    signal_id = db.save_signal(sig, "toobit", False)
    with db.session() as s:
        assert s.get(SignalLog, signal_id).snapshot_json["vwap"] is None


def test_timestamps_round_trip_as_utc(db):
    db.save_candles(make_ohlcv([100.0]), "BTC/USDT:USDT", "15m", "toobit", False)
    with db.session() as s:
        stored = s.scalars(select(Candle)).one().timestamp
    # SQLite drops tzinfo; the repository normalises on the way back in.
    assert stored.replace(tzinfo=timezone.utc) == T0


def test_a_failing_session_rolls_back(db):
    with pytest.raises(RuntimeError):
        with db.session() as s:
            s.add(Candle(
                exchange_id="toobit", symbol="BTC/USDT:USDT", timeframe="15m",
                timestamp=T0, open=1, high=1, low=1, close=1, volume=1,
            ))
            raise RuntimeError("boom")
    assert db.candle_count() == 0
