"""Unit tests for the risk layer.

This is the module where a bug costs money, so the tests check the arithmetic
identities directly rather than just "returns something sensible".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.config.settings import RiskSettings
from app.risk.levels import compute_levels
from app.risk.manager import RiskManager
from app.risk.position_sizing import calculate_position_size
from app.strategies.base import SignalDirection


@pytest.fixture
def risk() -> RiskSettings:
    return RiskSettings()


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------

def test_long_levels_hand_computed(risk):
    levels = compute_levels(SignalDirection.LONG, entry=100.0, atr=2.0, settings=risk)
    assert levels.risk_per_unit == pytest.approx(3.0)   # 1.5 * ATR
    assert levels.stop_loss == pytest.approx(97.0)
    assert levels.take_profit == pytest.approx(106.0)   # entry + 2R
    assert levels.rr == pytest.approx(2.0)


def test_short_levels_mirror_long(risk):
    levels = compute_levels(SignalDirection.SHORT, entry=100.0, atr=2.0, settings=risk)
    assert levels.stop_loss == pytest.approx(103.0)
    assert levels.take_profit == pytest.approx(94.0)
    assert levels.rr == pytest.approx(2.0)


def test_stop_is_always_on_the_losing_side(risk):
    long_levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    short_levels = compute_levels(SignalDirection.SHORT, 100.0, 2.0, risk)
    assert long_levels.stop_loss < long_levels.entry < long_levels.take_profit
    assert short_levels.take_profit < short_levels.entry < short_levels.stop_loss


def test_wider_atr_gives_a_wider_stop(risk):
    narrow = compute_levels(SignalDirection.LONG, 100.0, 1.0, risk)
    wide = compute_levels(SignalDirection.LONG, 100.0, 5.0, risk)
    assert wide.risk_per_unit > narrow.risk_per_unit
    assert wide.rr == pytest.approx(narrow.rr)  # R:R is unchanged by volatility


def test_rr_equals_the_configured_target():
    settings = RiskSettings(rr_target=3.0)
    assert compute_levels(SignalDirection.LONG, 100.0, 2.0, settings).rr == pytest.approx(3.0)


def test_levels_reject_no_trade():
    with pytest.raises(ValueError):
        compute_levels(SignalDirection.NO_TRADE, 100.0, 2.0)


@pytest.mark.parametrize("entry,atr", [(0.0, 2.0), (-5.0, 2.0), (100.0, 0.0), (100.0, -1.0)])
def test_levels_reject_impossible_inputs(entry, atr):
    with pytest.raises(ValueError):
        compute_levels(SignalDirection.LONG, entry, atr)


def test_levels_reject_atr_large_enough_to_push_the_stop_below_zero():
    with pytest.raises(ValueError):
        compute_levels(SignalDirection.LONG, entry=10.0, atr=100.0)


# --------------------------------------------------------------------------
# position sizing
# --------------------------------------------------------------------------

def test_position_size_is_risk_divided_by_stop_distance(risk):
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    size = calculate_position_size(balance=10_000.0, levels=levels, settings=risk)

    assert size.accepted
    # 1% of 10,000 = 100 risked; stop is 3.0 away -> 33.333... units
    assert size.risk_amount == pytest.approx(100.0)
    assert size.quantity == pytest.approx(100.0 / 3.0)
    assert size.notional == pytest.approx(size.quantity * 100.0)


def test_losing_at_the_stop_costs_exactly_the_planned_risk(risk):
    """The identity the whole module exists to guarantee."""
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    size = calculate_position_size(10_000.0, levels, risk)
    loss = (levels.entry - levels.stop_loss) * size.quantity
    assert loss == pytest.approx(size.risk_amount)
    assert loss == pytest.approx(10_000.0 * risk.risk_per_trade_pct)


def test_wider_stop_buys_a_smaller_position(risk):
    """Both ATRs are chosen so the notional cap does NOT bind, which is what
    lets the risked amount stay identical while the quantity shrinks."""
    narrow = calculate_position_size(
        10_000.0, compute_levels(SignalDirection.LONG, 100.0, 2.0, risk), risk
    )
    wide = calculate_position_size(
        10_000.0, compute_levels(SignalDirection.LONG, 100.0, 4.0, risk), risk
    )
    assert not narrow.caps_applied and not wide.caps_applied
    assert wide.quantity < narrow.quantity
    assert wide.risk_amount == pytest.approx(narrow.risk_amount)


def test_size_scales_with_balance(risk):
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    small = calculate_position_size(5_000.0, levels, risk)
    large = calculate_position_size(10_000.0, levels, risk)
    assert large.quantity == pytest.approx(2 * small.quantity)


def test_notional_cap_applies_when_the_stop_is_very_tight(risk):
    """A 0.01-wide stop would otherwise imply a position 100x the account."""
    levels = compute_levels(SignalDirection.LONG, 100.0, 0.01, risk)
    size = calculate_position_size(10_000.0, levels, risk)

    assert size.accepted
    assert "max_position_notional" in size.caps_applied
    assert size.notional == pytest.approx(10_000.0 * risk.max_position_notional_pct)
    # After capping, the numbers stay internally consistent.
    assert size.risk_amount == pytest.approx(size.quantity * levels.risk_per_unit)
    assert size.risk_amount < 10_000.0 * risk.risk_per_trade_pct


def test_daily_budget_caps_the_last_trade_of_the_day(risk):
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    # Budget is 3% of 10,000 = 300; 250 already used leaves 50.
    size = calculate_position_size(10_000.0, levels, risk, daily_risk_used=250.0)
    assert size.accepted
    assert "daily_risk_budget" in size.caps_applied
    assert size.risk_amount == pytest.approx(50.0)


def test_exhausted_daily_budget_is_rejected_with_a_reason(risk):
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    size = calculate_position_size(10_000.0, levels, risk, daily_risk_used=300.0)
    assert not size.accepted
    assert size.quantity == 0.0
    assert "daily risk budget" in size.rejection_reason


def test_max_open_positions_is_enforced(risk):
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    size = calculate_position_size(
        10_000.0, levels, risk, open_positions=risk.max_open_positions
    )
    assert not size.accepted
    assert "max_open_positions" in size.rejection_reason


def test_below_minimum_rr_is_rejected():
    settings = RiskSettings(rr_target=1.0, min_rr=1.5)
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, settings)
    size = calculate_position_size(10_000.0, levels, settings)
    assert not size.accepted
    assert "reward:risk" in size.rejection_reason


def test_dust_notional_is_rejected(risk):
    """A tiny account cannot meet the venue minimum; say so, do not round up."""
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    size = calculate_position_size(20.0, levels, risk)
    assert not size.accepted
    assert "minimum" in size.rejection_reason


@pytest.mark.parametrize("balance", [0.0, -100.0])
def test_non_positive_balance_is_rejected(balance, risk):
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    size = calculate_position_size(balance, levels, risk)
    assert not size.accepted


def test_every_rejection_carries_a_reason(risk):
    levels = compute_levels(SignalDirection.LONG, 100.0, 2.0, risk)
    for kwargs in (
        {"balance": 0.0},
        {"balance": 10_000.0, "open_positions": 99},
        {"balance": 10_000.0, "daily_risk_used": 10_000.0},
        {"balance": 20.0},
    ):
        size = calculate_position_size(levels=levels, settings=risk, **kwargs)
        assert not size.accepted
        assert size.rejection_reason


# --------------------------------------------------------------------------
# risk manager / daily ledger
# --------------------------------------------------------------------------

def make_signal(direction=SignalDirection.LONG, close=100.0, atr=2.0):
    from app.indicators.engine import IndicatorSnapshot
    from app.strategies.base import Signal
    import pandas as pd

    snapshot = IndicatorSnapshot(
        symbol="BTC/USDT", timeframe="15m",
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        open=close, high=close + 1, low=close - 1, close=close, volume=1.0,
        ema_fast=101.0, ema_slow=100.0, ema_fast_prev=99.0, ema_slow_prev=100.0,
        sma_slow=100.0, rsi=55.0, atr=atr,
        macd=0.1, macd_signal=0.05, macd_hist=0.05,
        bb_upper=105.0, bb_middle=100.0, bb_lower=95.0, bb_percent_b=0.5,
        vwap=100.0, trend="UP",
    )
    return Signal(
        direction=direction, reasons=("test",), invalidation="test",
        snapshot=snapshot, strategy_name="test",
    )


def test_manager_approves_a_clean_signal():
    manager = RiskManager()
    decision = manager.evaluate(make_signal(), balance=10_000.0)
    assert decision.approved
    assert decision.size.risk_amount == pytest.approx(100.0)
    assert "risking" in decision.reason


def test_manager_refuses_no_trade():
    decision = RiskManager().evaluate(
        make_signal(SignalDirection.NO_TRADE), balance=10_000.0
    )
    assert not decision.approved
    assert decision.levels is None


def test_manager_refuses_when_atr_is_missing():
    from dataclasses import replace

    signal = make_signal()
    signal = replace(signal, snapshot=replace(signal.snapshot, atr=None))
    decision = RiskManager().evaluate(signal, balance=10_000.0)
    assert not decision.approved
    assert "ATR" in decision.reason


def test_daily_ledger_accumulates_then_blocks():
    manager = RiskManager()
    signal = make_signal()

    # 3% budget on 10,000 = 300, at 100 per trade.
    for _ in range(3):
        decision = manager.evaluate(signal, balance=10_000.0)
        assert decision.approved
        manager.commit_risk(decision.size.risk_amount)

    blocked = manager.evaluate(signal, balance=10_000.0)
    assert not blocked.approved
    assert "daily risk budget" in blocked.reason


def test_daily_ledger_resets_on_a_new_utc_day():
    manager = RiskManager()
    day_one = datetime(2026, 1, 1, 23, 0, tzinfo=timezone.utc)
    manager.commit_risk(300.0, now=day_one)
    assert manager.risk_committed_today(day_one) == pytest.approx(300.0)

    day_two = day_one + timedelta(hours=2)
    assert manager.risk_committed_today(day_two) == pytest.approx(0.0)


def test_release_risk_gives_budget_back():
    manager = RiskManager()
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    manager.commit_risk(100.0, now=now)
    manager.release_risk(40.0, now=now)
    assert manager.risk_committed_today(now) == pytest.approx(60.0)


def test_release_risk_never_goes_negative():
    manager = RiskManager()
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    manager.commit_risk(10.0, now=now)
    manager.release_risk(999.0, now=now)
    assert manager.risk_committed_today(now) == 0.0


def test_sizing_is_deterministic():
    """Same inputs, same output. Every time. No hidden state, no model."""
    signal = make_signal()
    results = [
        RiskManager().evaluate(signal, balance=10_000.0).size.quantity
        for _ in range(20)
    ]
    assert len(set(results)) == 1
