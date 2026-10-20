"""Unit tests for the virtual wallet and the paper trading engine.

The engine's job is to be PESSIMISTIC and consistent. These tests pin down
the two things that make paper results trustworthy or worthless:
  * slippage and fees always work against us, on both sides
  * an ambiguous candle (stop and target both inside its range) resolves as
    the stop
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.config.settings import PaperTradingSettings, RiskSettings
from app.indicators.engine import IndicatorSnapshot
from app.paper_trading.engine import Candle, ExitReason, PaperTradingEngine
from app.paper_trading.wallet import InsufficientBalance, VirtualWallet
from app.risk.levels import compute_levels
from app.risk.position_sizing import calculate_position_size
from app.strategies.base import Signal, SignalDirection

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_signal(direction=SignalDirection.LONG, close=100.0, atr=2.0, symbol="BTC/USDT"):
    snapshot = IndicatorSnapshot(
        symbol=symbol, timeframe="15m", timestamp=pd.Timestamp(T0),
        open=close, high=close + 1, low=close - 1, close=close, volume=1.0,
        ema_fast=101.0, ema_slow=100.0, ema_fast_prev=99.0, ema_slow_prev=100.0,
        sma_slow=100.0, rsi=55.0, atr=atr,
        macd=0.1, macd_signal=0.05, macd_hist=0.05,
        bb_upper=105.0, bb_middle=100.0, bb_lower=95.0, bb_percent_b=0.5,
        vwap=100.0, trend="UP",
    )
    return Signal(
        direction=direction, reasons=("test setup",),
        invalidation="close back below EMA50", snapshot=snapshot,
        strategy_name="test_strategy",
    )


def open_trade(engine, direction=SignalDirection.LONG, close=100.0, atr=2.0,
               symbol="BTC/USDT", balance=10_000.0):
    risk = RiskSettings()
    signal = make_signal(direction, close, atr, symbol)
    levels = compute_levels(direction, close, atr, risk)
    size = calculate_position_size(balance, levels, risk)
    position = engine.open_position(
        signal, levels, size, exchange_id="test", is_dev_data_source=True,
        opened_at=T0,
    )
    return position, levels, size


# --------------------------------------------------------------------------
# wallet
# --------------------------------------------------------------------------

def test_wallet_starts_at_the_configured_balance():
    wallet = VirtualWallet(starting_balance=5_000.0)
    assert wallet.balance == 5_000.0
    assert wallet.realized_pnl == 0.0
    assert wallet.win_rate is None


def test_wallet_rejects_a_non_positive_start():
    with pytest.raises(ValueError):
        VirtualWallet(starting_balance=0.0)


def test_fees_reduce_balance_and_accumulate():
    wallet = VirtualWallet(1_000.0)
    wallet.charge_fee(10.0)
    wallet.charge_fee(5.0)
    assert wallet.balance == pytest.approx(985.0)
    assert wallet.fees_paid == pytest.approx(15.0)


def test_fee_larger_than_balance_is_refused():
    wallet = VirtualWallet(10.0)
    with pytest.raises(InsufficientBalance):
        wallet.charge_fee(50.0)


def test_settle_tracks_wins_losses_and_win_rate():
    wallet = VirtualWallet(1_000.0)
    wallet.settle(100.0)
    wallet.settle(-50.0)
    wallet.settle(25.0)
    assert wallet.balance == pytest.approx(1_075.0)
    assert (wallet.wins, wallet.losses) == (2, 1)
    assert wallet.win_rate == pytest.approx(2 / 3)


def test_breakeven_trade_counts_as_neither_win_nor_loss():
    wallet = VirtualWallet(1_000.0)
    wallet.settle(0.0)
    assert wallet.wins == 0 and wallet.losses == 0
    assert wallet.trades_closed == 1


def test_equity_includes_open_pnl():
    wallet = VirtualWallet(1_000.0)
    assert wallet.equity(unrealized_pnl=250.0) == pytest.approx(1_250.0)


# --------------------------------------------------------------------------
# fill pricing
# --------------------------------------------------------------------------

def test_long_entry_fills_worse_than_the_reference_price():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.01, taker_fee_pct=0.0))
    position, levels, _ = open_trade(engine)
    assert position.entry_price == pytest.approx(levels.entry * 1.01)
    assert position.entry_price > position.reference_price


def test_short_entry_fills_worse_than_the_reference_price():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.01, taker_fee_pct=0.0))
    position, levels, _ = open_trade(engine, SignalDirection.SHORT)
    assert position.entry_price == pytest.approx(levels.entry * 0.99)
    assert position.entry_price < position.reference_price


def test_long_exit_fills_worse_than_the_reference_price():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.01, taker_fee_pct=0.0))
    position, _, _ = open_trade(engine)
    trade = engine.close_position(position.position_id, 110.0, ExitReason.MANUAL, T0)
    assert trade.exit_price == pytest.approx(110.0 * 0.99)


def test_short_exit_fills_worse_than_the_reference_price():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.01, taker_fee_pct=0.0))
    position, _, _ = open_trade(engine, SignalDirection.SHORT)
    trade = engine.close_position(position.position_id, 90.0, ExitReason.MANUAL, T0)
    assert trade.exit_price == pytest.approx(90.0 * 1.01)


def test_slippage_makes_a_round_trip_at_the_same_price_a_loss():
    """Zero price movement must still cost money, or the engine is lying."""
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0025, taker_fee_pct=0.001))
    position, _, _ = open_trade(engine)
    trade = engine.close_position(position.position_id, 100.0, ExitReason.MANUAL, T0)
    assert trade.net_pnl < 0
    assert engine.wallet.balance < 10_000.0


def test_fees_are_charged_on_both_sides():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.001))
    position, _, _ = open_trade(engine)
    assert position.entry_fee > 0
    trade = engine.close_position(position.position_id, 100.0, ExitReason.MANUAL, T0)
    assert trade.exit_fee > 0
    assert engine.wallet.fees_paid == pytest.approx(position.entry_fee + trade.exit_fee)


# --------------------------------------------------------------------------
# PnL direction
# --------------------------------------------------------------------------

def test_long_profits_when_price_rises():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, _, _ = open_trade(engine)
    trade = engine.close_position(position.position_id, 110.0, ExitReason.TAKE_PROFIT, T0)
    assert trade.net_pnl > 0


def test_short_profits_when_price_falls():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, _, _ = open_trade(engine, SignalDirection.SHORT)
    trade = engine.close_position(position.position_id, 90.0, ExitReason.TAKE_PROFIT, T0)
    assert trade.net_pnl > 0


def test_short_loses_when_price_rises():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, _, _ = open_trade(engine, SignalDirection.SHORT)
    trade = engine.close_position(position.position_id, 110.0, ExitReason.STOP_LOSS, T0)
    assert trade.net_pnl < 0


def test_r_multiple_is_roughly_minus_one_at_the_stop():
    """A stop-out with no costs should be almost exactly -1R."""
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, levels, _ = open_trade(engine)
    trade = engine.close_position(
        position.position_id, levels.stop_loss, ExitReason.STOP_LOSS, T0
    )
    assert trade.r_multiple == pytest.approx(-1.0, abs=0.01)


def test_r_multiple_is_roughly_plus_two_at_the_target():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, levels, _ = open_trade(engine)
    trade = engine.close_position(
        position.position_id, levels.take_profit, ExitReason.TAKE_PROFIT, T0
    )
    assert trade.r_multiple == pytest.approx(2.0, abs=0.01)


def test_costs_make_the_realised_r_worse_than_the_planned_r():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0025, taker_fee_pct=0.001))
    position, levels, _ = open_trade(engine)
    trade = engine.close_position(
        position.position_id, levels.take_profit, ExitReason.TAKE_PROFIT, T0
    )
    assert trade.r_multiple < 2.0


# --------------------------------------------------------------------------
# candle-driven exits
# --------------------------------------------------------------------------

def test_stop_is_taken_when_the_candle_low_reaches_it():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, levels, _ = open_trade(engine)
    candle = Candle(T0 + timedelta(minutes=15), high=101.0, low=levels.stop_loss - 0.1, close=97.0)
    trade = engine.update_position("BTC/USDT", candle)
    assert trade is not None and trade.exit_reason is ExitReason.STOP_LOSS


def test_target_is_taken_when_the_candle_high_reaches_it():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, levels, _ = open_trade(engine)
    candle = Candle(T0 + timedelta(minutes=15), high=levels.take_profit + 0.1, low=99.0, close=106.0)
    trade = engine.update_position("BTC/USDT", candle)
    assert trade is not None and trade.exit_reason is ExitReason.TAKE_PROFIT


def test_ambiguous_candle_resolves_as_the_stop():
    """OHLC cannot say which printed first, so we assume the worst. This
    single choice is the difference between an honest paper record and a
    flattering one."""
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, levels, _ = open_trade(engine)
    candle = Candle(
        T0 + timedelta(minutes=15),
        high=levels.take_profit + 1.0,
        low=levels.stop_loss - 1.0,
        close=100.0,
    )
    trade = engine.update_position("BTC/USDT", candle)
    assert trade.exit_reason is ExitReason.STOP_LOSS


def test_short_ambiguous_candle_also_resolves_as_the_stop():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, levels, _ = open_trade(engine, SignalDirection.SHORT)
    candle = Candle(
        T0 + timedelta(minutes=15),
        high=levels.stop_loss + 1.0,
        low=levels.take_profit - 1.0,
        close=100.0,
    )
    trade = engine.update_position("BTC/USDT", candle)
    assert trade.exit_reason is ExitReason.STOP_LOSS


def test_untouched_candle_leaves_the_position_open():
    engine = PaperTradingEngine()
    open_trade(engine)
    candle = Candle(T0 + timedelta(minutes=15), high=100.5, low=99.5, close=100.0)
    assert engine.update_position("BTC/USDT", candle) is None
    assert len(engine.open_positions) == 1


def test_update_on_a_symbol_with_no_position_is_a_no_op():
    engine = PaperTradingEngine()
    candle = Candle(T0, high=1.0, low=0.0, close=0.5)
    assert engine.update_position("ETH/USDT", candle) is None


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------

def test_pyramiding_is_refused():
    engine = PaperTradingEngine()
    open_trade(engine)
    with pytest.raises(ValueError, match="already open"):
        open_trade(engine)


def test_positions_on_different_symbols_coexist():
    engine = PaperTradingEngine()
    open_trade(engine, symbol="BTC/USDT")
    open_trade(engine, symbol="ETH/USDT")
    assert len(engine.open_positions) == 2


def test_no_trade_signal_cannot_open_a_position():
    engine = PaperTradingEngine()
    risk = RiskSettings()
    signal = make_signal(SignalDirection.NO_TRADE)
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    size = calculate_position_size(10_000.0, levels, risk)
    with pytest.raises(ValueError):
        engine.open_position(signal, levels, size, "test", True, T0)


def test_rejected_size_cannot_open_a_position():
    engine = PaperTradingEngine()
    risk = RiskSettings()
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    rejected = calculate_position_size(20.0, levels, risk)  # below min notional
    assert not rejected.accepted
    with pytest.raises(ValueError):
        engine.open_position(make_signal(), levels, rejected, "test", True, T0)


def test_closing_an_unknown_position_raises():
    with pytest.raises(KeyError):
        PaperTradingEngine().close_position("nope", 100.0, ExitReason.MANUAL, T0)


def test_unrealized_pnl_marks_open_positions():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, _, _ = open_trade(engine)
    assert engine.unrealized_pnl({"BTC/USDT": 110.0}) == pytest.approx(
        10.0 * position.quantity
    )


def test_summary_reports_profit_factor_and_avg_r():
    engine = PaperTradingEngine(PaperTradingSettings(slippage_pct=0.0, taker_fee_pct=0.0))
    position, levels, _ = open_trade(engine)
    engine.close_position(position.position_id, levels.take_profit, ExitReason.TAKE_PROFIT, T0)
    position, levels, _ = open_trade(engine)
    engine.close_position(position.position_id, levels.stop_loss, ExitReason.STOP_LOSS, T0)

    summary = engine.summary()
    assert summary["trades_closed"] == 2
    assert summary["wins"] == 1 and summary["losses"] == 1
    assert summary["profit_factor"] == pytest.approx(2.0, abs=0.05)
    assert summary["avg_r_multiple"] == pytest.approx(0.5, abs=0.05)


def test_closed_trade_serialises_every_number_needed_to_audit_it():
    engine = PaperTradingEngine()
    position, _, _ = open_trade(engine)
    trade = engine.close_position(position.position_id, 105.0, ExitReason.MANUAL, T0)
    data = trade.to_dict()
    for key in (
        "entry_price", "exit_price", "stop_loss", "take_profit", "risk_amount",
        "gross_pnl", "entry_fee", "exit_fee", "net_pnl", "r_multiple",
        "exit_reason", "opened_at", "closed_at",
    ):
        assert key in data
