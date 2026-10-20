"""Unit tests for correlation risk.

The module's whole job is to notice that four positions across BTC, ETH, SOL
and BNB are not four independent bets. These tests check that it does, that
it fails SAFE when it cannot measure, and that a genuine spread (long one,
short a correlated other) is not punished as if it were a doubled bet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.settings import RiskSettings
from app.risk.correlation import (
    CorrelationRiskManager,
    ExposureItem,
    correlation_matrix,
    find_cluster,
    log_returns,
)
from app.risk.manager import RiskManager
from app.strategies.base import SignalDirection

LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT


@pytest.fixture
def risk() -> RiskSettings:
    return RiskSettings()


def series(values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype="float64"))


def correlated_pair(n: int = 200, noise: float = 0.0005, seed: int = 7):
    """Two price series that move together (like BTC and ETH)."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 0.01, n)
    a = 100.0 * np.exp(np.cumsum(shocks))
    b = 50.0 * np.exp(np.cumsum(shocks + rng.normal(0, noise, n)))
    return series(a), series(b)


def independent_pair(n: int = 200, seed: int = 11):
    rng = np.random.default_rng(seed)
    a = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    b = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return series(a), series(b)


# --------------------------------------------------------------------------
# returns and matrix
# --------------------------------------------------------------------------

def test_log_returns_first_value_is_nan():
    assert np.isnan(log_returns(series([100.0, 101.0, 102.0])).iloc[0])


def test_log_returns_are_additive():
    """log(c/a) == log(b/a) + log(c/b). This is why we use log returns."""
    values = series([100.0, 110.0, 121.0])
    returns = log_returns(values).dropna()
    assert returns.sum() == pytest.approx(np.log(121.0 / 100.0))


def test_identical_series_correlate_at_one():
    a, _ = correlated_pair()
    matrix = correlation_matrix({"A": a, "B": a.copy()}, lookback=150)
    assert matrix.at["A", "B"] == pytest.approx(1.0)


def test_co_moving_series_correlate_highly():
    a, b = correlated_pair()
    matrix = correlation_matrix({"A": a, "B": b}, lookback=150)
    assert matrix.at["A", "B"] > 0.9


def test_independent_series_correlate_weakly():
    a, b = independent_pair()
    matrix = correlation_matrix({"A": a, "B": b}, lookback=150)
    assert abs(matrix.at["A", "B"]) < 0.5


def test_inverse_series_correlate_negatively():
    a, _ = correlated_pair()
    inverse = series((a.iloc[0] ** 2) / a.to_numpy())
    matrix = correlation_matrix({"A": a, "B": inverse}, lookback=150)
    assert matrix.at["A", "B"] < -0.9


def test_matrix_respects_the_lookback_window():
    rng = np.random.default_rng(3)
    shared = np.cumsum(rng.normal(0, 0.01, 300))
    a = series(100 * np.exp(shared))
    # First half tracks A, second half is independent noise.
    tail = np.cumsum(rng.normal(0, 0.01, 150))
    b = series(np.concatenate([100 * np.exp(shared[:150]), 100 * np.exp(tail + shared[149])]))
    recent = correlation_matrix({"A": a, "B": b}, lookback=100).at["A", "B"]
    everything = correlation_matrix({"A": a, "B": b}, lookback=300).at["A", "B"]
    assert recent < everything


def test_too_few_observations_gives_nan_not_a_number():
    """A correlation from 5 points is noise; reporting it as fact is worse
    than reporting that we do not know."""
    a, b = correlated_pair(n=8)
    matrix = correlation_matrix({"A": a, "B": b}, lookback=8, min_observations=30)
    assert np.isnan(matrix.at["A", "B"])


def test_empty_input_gives_an_empty_matrix():
    assert correlation_matrix({}).empty


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------

def test_correlated_peer_joins_the_cluster():
    matrix = pd.DataFrame(
        [[1.0, 0.95], [0.95, 1.0]], index=["BTC", "ETH"], columns=["BTC", "ETH"]
    )
    cluster, correlations = find_cluster("BTC", ["ETH"], matrix, threshold=0.7)
    assert set(cluster) == {"BTC", "ETH"}
    assert correlations["ETH"] == pytest.approx(0.95)


def test_uncorrelated_peer_stays_out():
    matrix = pd.DataFrame(
        [[1.0, 0.10], [0.10, 1.0]], index=["BTC", "XYZ"], columns=["BTC", "XYZ"]
    )
    cluster, _ = find_cluster("BTC", ["XYZ"], matrix, threshold=0.7)
    assert cluster == ["BTC"]


def test_clustering_is_transitive():
    """A~B and B~C puts A, B and C in one cluster even though A~C is weak.
    Single linkage is the conservative reading in one asset class."""
    labels = ["A", "B", "C"]
    matrix = pd.DataFrame(
        [[1.0, 0.9, 0.3], [0.9, 1.0, 0.9], [0.3, 0.9, 1.0]],
        index=labels, columns=labels,
    )
    cluster, _ = find_cluster("A", ["B", "C"], matrix, threshold=0.7)
    assert set(cluster) == {"A", "B", "C"}


def test_strong_negative_correlation_also_clusters():
    """-0.95 is just as much a shared factor as +0.95; the direction is
    handled by signed exposure, not by ignoring the relationship."""
    matrix = pd.DataFrame(
        [[1.0, -0.95], [-0.95, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    cluster, _ = find_cluster("A", ["B"], matrix, threshold=0.7)
    assert set(cluster) == {"A", "B"}


def test_unmeasurable_correlation_fails_safe():
    """NaN must mean 'assume correlated', never 'assume independent'."""
    matrix = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    cluster, _ = find_cluster("A", ["B"], matrix, threshold=0.7)
    assert set(cluster) == {"A", "B"}


def test_symbol_missing_from_the_matrix_fails_safe():
    matrix = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    cluster, _ = find_cluster("A", ["MISSING"], matrix, threshold=0.7)
    assert "MISSING" in cluster


# --------------------------------------------------------------------------
# the budget decision
# --------------------------------------------------------------------------

def test_first_position_is_approved(risk):
    a, b = correlated_pair()
    decision = CorrelationRiskManager(risk).evaluate(
        "BTC", LONG, proposed_risk=100.0, balance=10_000.0,
        open_exposures=[], closes_by_symbol={"BTC": a, "ETH": b},
    )
    assert decision.approved
    assert decision.allowed_risk_amount == pytest.approx(100.0)


def test_second_correlated_long_is_trimmed_to_the_cluster_budget(risk):
    """Budget is 2% of 10,000 = 200. One 100 long is open, so a second
    correlated 150 long may only take 100."""
    a, b = correlated_pair()
    decision = CorrelationRiskManager(risk).evaluate(
        "ETH", LONG, proposed_risk=150.0, balance=10_000.0,
        open_exposures=[ExposureItem("BTC", LONG, 100.0)],
        closes_by_symbol={"BTC": a, "ETH": b},
    )
    assert decision.approved
    assert decision.allowed_risk_amount == pytest.approx(100.0)
    assert "trimmed" in decision.reason


def test_third_correlated_long_is_refused_outright(risk):
    a, b = correlated_pair()
    c = b * 1.01
    decision = CorrelationRiskManager(risk).evaluate(
        "SOL", LONG, proposed_risk=100.0, balance=10_000.0,
        open_exposures=[
            ExposureItem("BTC", LONG, 100.0),
            ExposureItem("ETH", LONG, 100.0),
        ],
        closes_by_symbol={"BTC": a, "ETH": b, "SOL": c},
    )
    assert not decision.approved
    assert "already holds" in decision.reason


def test_opposing_direction_on_a_correlated_pair_is_a_spread_not_a_double(risk):
    """Long BTC + short ETH is a relative-value trade. Netting to zero is the
    entire reason exposures are signed."""
    a, b = correlated_pair()
    decision = CorrelationRiskManager(risk).evaluate(
        "ETH", SHORT, proposed_risk=100.0, balance=10_000.0,
        open_exposures=[ExposureItem("BTC", LONG, 100.0)],
        closes_by_symbol={"BTC": a, "ETH": b},
    )
    assert decision.approved
    assert decision.allowed_risk_amount == pytest.approx(100.0)


def test_uncorrelated_symbol_gets_its_own_budget(risk):
    a, b = independent_pair()
    decision = CorrelationRiskManager(risk).evaluate(
        "XYZ", LONG, proposed_risk=200.0, balance=10_000.0,
        open_exposures=[ExposureItem("BTC", LONG, 200.0)],
        closes_by_symbol={"BTC": a, "XYZ": b},
    )
    assert decision.approved
    assert decision.cluster == ("XYZ",)
    assert decision.allowed_risk_amount == pytest.approx(200.0)


def test_decision_reports_the_cluster_and_the_numbers(risk):
    a, b = correlated_pair()
    decision = CorrelationRiskManager(risk).evaluate(
        "ETH", LONG, proposed_risk=150.0, balance=10_000.0,
        open_exposures=[ExposureItem("BTC", LONG, 100.0)],
        closes_by_symbol={"BTC": a, "ETH": b},
    )
    payload = decision.to_dict()
    assert set(payload["cluster"]) == {"BTC", "ETH"}
    assert payload["cluster_budget"] == pytest.approx(200.0)
    assert payload["cluster_net_risk"] == pytest.approx(100.0)
    assert "BTC" in payload["correlations"]


def test_budget_scales_with_balance(risk):
    a, b = correlated_pair()
    manager = CorrelationRiskManager(risk)
    small = manager.evaluate(
        "ETH", LONG, 1_000.0, 10_000.0,
        [ExposureItem("BTC", LONG, 0.0)], {"BTC": a, "ETH": b},
    )
    large = manager.evaluate(
        "ETH", LONG, 1_000.0, 100_000.0,
        [ExposureItem("BTC", LONG, 0.0)], {"BTC": a, "ETH": b},
    )
    assert large.cluster_budget == pytest.approx(10 * small.cluster_budget)


def test_missing_price_history_fails_safe(risk):
    """No history for a peer means we cannot rule out correlation."""
    a, _ = correlated_pair()
    decision = CorrelationRiskManager(risk).evaluate(
        "ETH", LONG, proposed_risk=100.0, balance=10_000.0,
        open_exposures=[ExposureItem("BTC", LONG, 200.0)],
        closes_by_symbol={"BTC": a},  # ETH history absent
    )
    assert not decision.approved


def test_exposure_item_signs_by_direction():
    assert ExposureItem("BTC", LONG, 100.0).signed_risk == 100.0
    assert ExposureItem("BTC", SHORT, 100.0).signed_risk == -100.0


# --------------------------------------------------------------------------
# integration with RiskManager
# --------------------------------------------------------------------------

def make_signal(direction=LONG, symbol="ETH/USDT:USDT", close=100.0, atr=2.0):
    from app.indicators.engine import IndicatorSnapshot
    from app.strategies.base import Signal

    snapshot = IndicatorSnapshot(
        symbol=symbol, timeframe="15m", timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        open=close, high=close + 1, low=close - 1, close=close, volume=1.0,
        ema_fast=101.0, ema_slow=100.0, ema_fast_prev=99.0, ema_slow_prev=100.0,
        sma_slow=100.0, rsi=55.0, atr=atr, macd=0.1, macd_signal=0.05, macd_hist=0.05,
        bb_upper=105.0, bb_middle=100.0, bb_lower=95.0, bb_percent_b=0.5,
        vwap=100.0, trend="UP",
    )
    return Signal(
        direction=direction, reasons=("test",), invalidation="test",
        snapshot=snapshot, strategy_name="test",
    )


def test_risk_manager_skips_correlation_when_no_history_is_supplied():
    """Backwards compatible: without price history the gate is inert."""
    decision = RiskManager().evaluate(make_signal(), balance=10_000.0)
    assert decision.approved
    assert decision.correlation is None


def test_risk_manager_applies_the_correlation_trim():
    a, b = correlated_pair()
    manager = RiskManager()
    decision = manager.evaluate(
        make_signal(symbol="ETH/USDT:USDT"),
        balance=10_000.0,
        open_exposures=[ExposureItem("BTC/USDT:USDT", LONG, 150.0)],
        closes_by_symbol={"BTC/USDT:USDT": a, "ETH/USDT:USDT": b},
    )
    assert decision.approved
    assert decision.correlation is not None
    # Budget 200, 150 already used -> only 50 left of the requested 100.
    assert decision.size.risk_amount == pytest.approx(50.0)
    assert "correlated_cluster" in decision.size.caps_applied


def test_trimmed_size_stays_internally_consistent():
    """quantity * stop distance must still equal the logged risk."""
    a, b = correlated_pair()
    decision = RiskManager().evaluate(
        make_signal(symbol="ETH/USDT:USDT"),
        balance=10_000.0,
        open_exposures=[ExposureItem("BTC/USDT:USDT", LONG, 150.0)],
        closes_by_symbol={"BTC/USDT:USDT": a, "ETH/USDT:USDT": b},
    )
    implied = decision.size.quantity * decision.levels.risk_per_unit
    assert implied == pytest.approx(decision.size.risk_amount)


def test_risk_manager_refuses_when_the_cluster_is_full():
    a, b = correlated_pair()
    decision = RiskManager().evaluate(
        make_signal(symbol="ETH/USDT:USDT"),
        balance=10_000.0,
        open_exposures=[ExposureItem("BTC/USDT:USDT", LONG, 200.0)],
        closes_by_symbol={"BTC/USDT:USDT": a, "ETH/USDT:USDT": b},
    )
    assert not decision.approved
    assert decision.correlation is not None and not decision.correlation.approved


def test_correlation_decision_is_serialised_into_the_risk_payload():
    a, b = correlated_pair()
    decision = RiskManager().evaluate(
        make_signal(symbol="ETH/USDT:USDT"),
        balance=10_000.0,
        open_exposures=[ExposureItem("BTC/USDT:USDT", LONG, 100.0)],
        closes_by_symbol={"BTC/USDT:USDT": a, "ETH/USDT:USDT": b},
    )
    assert decision.to_dict()["correlation"]["cluster"]
