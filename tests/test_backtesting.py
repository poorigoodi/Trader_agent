"""Unit tests for the backtester and the historical data source.

NO NETWORK. The data source is tested against a fake ccxt client.

The most important test in this file is
``test_indicators_are_causal_so_precompute_equals_per_prefix``. The backtester
precomputes indicators once over the whole series for speed and then reads
only row i and earlier. That is only valid if every indicator is strictly
backward-looking. Rather than assert that in a comment, the test recomputes
indicators on a truncated prefix and checks the values match. If anyone ever
adds a centred or forward-filled indicator, that test fails and the
look-ahead is caught at the source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtesting.data_source import (
    HistoricalDataSource,
    HistoricalWindow,
    to_spot_symbol,
)
from app.backtesting.engine import Backtester, BacktestResult, LabelledSignal
from app.config.settings import PaperTradingSettings, RiskSettings, Settings
from app.indicators.engine import compute_indicators
from tests.conftest import make_ohlcv


# --------------------------------------------------------------------------
# symbol mapping
# --------------------------------------------------------------------------

def test_perp_symbol_maps_to_spot():
    assert to_spot_symbol("BTC/USDT:USDT") == "BTC/USDT"
    assert to_spot_symbol("ETH/USDT") == "ETH/USDT"


# --------------------------------------------------------------------------
# data source
# --------------------------------------------------------------------------

class FakeHistoryClient:
    """Serves paged OHLCV like a vendor that honours `since`."""

    def __init__(self, candles: int = 5000, step_ms: int = 900_000, honour_since=True):
        import time as _time

        # Anchor the series so it actually covers the window the tests ask
        # for; a fixed past base would make every `since` land off the end.
        self.step = step_ms
        self.base = int(_time.time() * 1000) - candles * step_ms
        self.candles = candles
        self.honour_since = honour_since
        self.calls: list[int | None] = []
        self.apiKey = None
        self.secret = None

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls.append(since)
        start = 0
        if self.honour_since and since is not None:
            start = max(0, (since - self.base) // self.step)
        rows = []
        for i in range(int(start), min(int(start) + limit, self.candles)):
            price = 100.0 + i * 0.01
            rows.append([self.base + i * self.step, price, price + 0.5, price - 0.5, price, 5.0])
        return rows


def source(client=None, **kwargs) -> HistoricalDataSource:
    kwargs.setdefault("sleep_fn", lambda _: None)
    kwargs.setdefault("cache_dir", "data/_test_cache")
    return HistoricalDataSource(client=client or FakeHistoryClient(), **kwargs)


def test_unknown_vendor_is_refused():
    with pytest.raises(ValueError, match="not in verified list"):
        HistoricalDataSource(vendor="not_a_vendor")


def test_data_source_refuses_credentials():
    client = FakeHistoryClient()
    client.apiKey = "leaked"
    with pytest.raises(RuntimeError, match="read-only"):
        HistoricalDataSource(client=client)


def test_paging_collects_more_than_one_page():
    client = FakeHistoryClient(candles=5000)
    window = source(client).fetch(
        "BTC/USDT:USDT", "15m", days=60, page_limit=1000, use_cache=False
    )
    assert len(client.calls) > 1
    assert window.candles > 1000


def test_paging_stops_when_the_vendor_ignores_since():
    """Otherwise this loops forever collecting the same page."""
    client = FakeHistoryClient(candles=500, honour_since=False)
    window = source(client).fetch(
        "BTC/USDT:USDT", "15m", days=60, page_limit=500, max_pages=50, use_cache=False
    )
    assert len(client.calls) <= 2
    assert window.candles == 500


def test_fetched_frame_is_sorted_and_unique():
    window = source().fetch("BTC/USDT:USDT", "15m", days=10, use_cache=False)
    assert window.frame["timestamp"].is_monotonic_increasing
    assert window.frame["timestamp"].is_unique


def test_window_records_its_provenance():
    window = source().fetch("BTC/USDT:USDT", "15m", days=10, use_cache=False)
    assert window.vendor_symbol == "BTC/USDT"  # spot mapping applied
    assert window.market_type == "spot"
    assert "kucoin" in window.describe()


def test_unsupported_timeframe_is_refused():
    with pytest.raises(ValueError, match="unsupported timeframe"):
        source().fetch("BTC/USDT:USDT", "7s", days=1, use_cache=False)


def test_fetch_many_survives_one_bad_symbol():
    class PartlyBroken(FakeHistoryClient):
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
            if symbol == "BAD/USDT":
                raise RuntimeError("no such market")
            return super().fetch_ohlcv(symbol, timeframe, since, limit)

    result = source(PartlyBroken()).fetch_many(
        ["BTC/USDT:USDT", "BAD/USDT"], "15m", days=5, use_cache=False
    )
    assert "BTC/USDT:USDT" in result
    assert "BAD/USDT" not in result


# --------------------------------------------------------------------------
# basis measurement
# --------------------------------------------------------------------------

def test_basis_is_zero_for_identical_frames():
    frame = make_ohlcv([100.0, 101.0, 102.0])
    basis = HistoricalDataSource.measure_basis(frame, frame)
    assert basis["overlapping_candles"] == 3
    assert basis["mean_basis_pct"] == pytest.approx(0.0)


def test_basis_reports_a_real_gap():
    vendor = make_ohlcv([100.0, 100.0, 100.0])
    venue = make_ohlcv([101.0, 101.0, 101.0])
    basis = HistoricalDataSource.measure_basis(vendor, venue)
    assert basis["mean_basis_pct"] == pytest.approx(1.0)


def test_basis_with_no_overlap_reports_zero_candles():
    vendor = make_ohlcv([100.0], start="2026-01-01T00:00:00Z")
    venue = make_ohlcv([100.0], start="2027-01-01T00:00:00Z")
    assert HistoricalDataSource.measure_basis(vendor, venue)["overlapping_candles"] == 0


# --------------------------------------------------------------------------
# no look-ahead
# --------------------------------------------------------------------------

def test_indicators_are_causal_so_precompute_equals_per_prefix():
    """The guarantee the backtester's precompute optimisation rests on.

    If this fails, an indicator has become forward-looking and every
    backtest result is contaminated.
    """
    settings = Settings()
    rng = np.random.default_rng(5)
    closes = 100 + np.cumsum(rng.normal(0, 0.5, 300))
    frame = make_ohlcv(closes)

    whole = compute_indicators(frame, settings.strategy)

    for cut in (150, 200, 260):
        prefix = compute_indicators(frame.iloc[:cut].copy(), settings.strategy)
        for column in ("ema_fast", "ema_slow", "rsi", "atr", "macd", "bb_upper", "vwap"):
            left = whole[column].iloc[:cut].to_numpy()
            right = prefix[column].to_numpy()
            both_valid = ~np.isnan(left) & ~np.isnan(right)
            assert np.allclose(left[both_valid], right[both_valid]), (
                f"{column} changed when future candles were removed: it is "
                f"not causal and leaks look-ahead into the backtest"
            )


def test_strategy_only_ever_sees_the_prefix():
    """Capture what the strategy is handed and assert it ends at the
    decision candle."""
    settings = Settings(symbols=("BTC/USDT:USDT",))
    seen: list[int] = []

    class Spy:
        name = "spy"

        def evaluate(self, df, symbol, timeframe):
            seen.append(len(df))
            from app.strategies.base import Signal, SignalDirection
            from app.indicators.engine import build_snapshot

            return Signal(
                direction=SignalDirection.NO_TRADE,
                reasons=("spy",), invalidation="n/a",
                snapshot=build_snapshot(df, symbol, timeframe),
                strategy_name="spy",
            )

    frame = make_ohlcv([100.0 + i * 0.05 for i in range(200)])
    Backtester(settings, strategy=Spy(), warmup=120).run({"BTC/USDT:USDT": frame})

    # Lengths must be strictly increasing, one per candle, starting at 121.
    assert seen == list(range(121, 201))


# --------------------------------------------------------------------------
# the replay loop
# --------------------------------------------------------------------------

def trending_frames(n: int = 400) -> dict[str, pd.DataFrame]:
    values: list[float] = []
    price = 100.0
    for _ in range(120):
        price -= 0.05
        values.append(price)
    for step in range(n - 120):
        price += 0.5 if step % 6 else -0.4
        values.append(price)
    return {"BTC/USDT:USDT": make_ohlcv(values, spread=0.3)}


def test_backtest_runs_and_reports_metrics():
    settings = Settings(symbols=("BTC/USDT:USDT",))
    result = Backtester(settings, warmup=120).run(trending_frames())
    metrics = result.metrics()
    assert metrics["candles_evaluated"] > 0
    assert metrics["signals_total"] > 0
    assert metrics["starting_balance"] == 10_000.0


def test_empty_frames_are_refused():
    with pytest.raises(ValueError, match="no frames"):
        Backtester(Settings()).run({})


def test_every_candle_produces_exactly_one_signal_row_per_symbol():
    settings = Settings(symbols=("BTC/USDT:USDT",))
    result = Backtester(settings, warmup=120).run(trending_frames(300))
    assert len(result.signals) == result.candles_evaluated


def test_no_trades_are_left_open_at_the_end():
    """An unlabelled trade would silently corrupt the training dataset."""
    settings = Settings(symbols=("BTC/USDT:USDT",))
    result = Backtester(settings, warmup=120).run(trending_frames())
    for record in result.signals:
        if record.entry is not None:
            assert record.traded, "a signal opened a position that was never labelled"


def test_traded_signals_carry_a_win_loss_label():
    settings = Settings(symbols=("BTC/USDT:USDT",))
    result = Backtester(settings, warmup=120).run(trending_frames())
    for record in result.traded_signals():
        assert record.won in (0, 1)
        assert record.net_pnl is not None
        assert record.exit_reason is not None


def test_labels_agree_with_the_pnl_sign():
    settings = Settings(symbols=("BTC/USDT:USDT",))
    result = Backtester(settings, warmup=120).run(trending_frames())
    for record in result.traded_signals():
        assert record.won == (1 if record.net_pnl > 0 else 0)


def test_dataset_frame_has_the_calibration_features():
    settings = Settings(symbols=("BTC/USDT:USDT",))
    frame = Backtester(settings, warmup=120).run(trending_frames()).to_frame()
    for column in ("rsi", "atr_pct_of_price", "ema_spread_pct", "trend", "won", "timestamp"):
        assert column in frame.columns


def test_equity_curve_tracks_every_evaluated_candle():
    settings = Settings(symbols=("BTC/USDT:USDT",))
    result = Backtester(settings, warmup=120).run(trending_frames())
    assert len(result.equity_curve) == result.candles_evaluated


def test_max_drawdown_is_non_negative():
    settings = Settings(symbols=("BTC/USDT:USDT",))
    result = Backtester(settings, warmup=120).run(trending_frames())
    assert (result.max_drawdown_pct() or 0) >= 0


def test_higher_costs_never_improve_results():
    """A sanity check on the cost model's direction."""
    frames = trending_frames()
    cheap = Settings(
        symbols=("BTC/USDT:USDT",),
        paper=PaperTradingSettings(taker_fee_pct=0.0, slippage_pct=0.0),
    )
    dear = Settings(
        symbols=("BTC/USDT:USDT",),
        paper=PaperTradingSettings(taker_fee_pct=0.002, slippage_pct=0.004),
    )
    cheap_result = Backtester(cheap, warmup=120).run(frames)
    dear_result = Backtester(dear, warmup=120).run(frames)
    if cheap_result.trades and dear_result.trades:
        assert dear_result.final_balance <= cheap_result.final_balance


def test_backtest_is_deterministic():
    frames = trending_frames()
    settings = Settings(symbols=("BTC/USDT:USDT",))
    first = Backtester(settings, warmup=120).run(frames).metrics()
    second = Backtester(settings, warmup=120).run(frames).metrics()
    assert first == second


def test_result_records_the_vendor_it_used():
    settings = Settings(symbols=("BTC/USDT:USDT",))
    result = Backtester(settings, warmup=120).run(
        trending_frames(), vendor="kucoin", market_type="spot"
    )
    assert result.metrics()["vendor"] == "kucoin"
    assert result.metrics()["market_type"] == "spot"
