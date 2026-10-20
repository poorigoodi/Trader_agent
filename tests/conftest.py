"""Shared fixtures and deterministic sample data.

No test in this suite touches the network. Anything that would talk to an
exchange uses ``FakeExchange``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.config.settings import Settings, StrategySettings


@pytest.fixture
def settings() -> Settings:
    return Settings(exchange_id="lbank", symbols=("BTC/USDT",), timeframe="15m")


@pytest.fixture
def strategy_settings(settings: Settings) -> StrategySettings:
    return settings.strategy


def make_ohlcv(
    closes: list[float] | np.ndarray,
    start: str = "2026-01-01T00:00:00Z",
    freq: str = "15min",
    spread: float = 0.5,
    volume: float = 10.0,
) -> pd.DataFrame:
    """Build a well-formed OHLCV frame from a close series.

    Highs/lows are placed a fixed distance either side of the close, which
    keeps ATR predictable in tests that care about it.
    """
    closes = np.asarray(closes, dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(closes), freq=freq, tz="UTC"),
            "open": closes,
            "high": closes + spread,
            "low": closes - spread,
            "close": closes,
            "volume": np.full(len(closes), volume, dtype="float64"),
        }
    )


@pytest.fixture
def sample_closes() -> np.ndarray:
    """A fixed, reproducible random walk. Same numbers on every run."""
    rng = np.random.default_rng(42)
    return 100.0 + np.cumsum(rng.normal(0, 0.5, 300))


@pytest.fixture
def sample_ohlcv(sample_closes: np.ndarray) -> pd.DataFrame:
    return make_ohlcv(sample_closes)


@pytest.fixture
def uptrend_ohlcv() -> pd.DataFrame:
    """A clean uptrend: rising staircase with pullbacks.

    Constructed (not random) so the market-structure detector reliably sees
    higher highs and higher lows, and so an EMA cross actually occurs.
    """
    values: list[float] = []
    price = 100.0
    # Flat-to-down opening so EMA(20) starts below EMA(50) and must cross up.
    for _ in range(80):
        price -= 0.05
        values.append(price)
    # Then a steady climb with regular 3-candle pullbacks.
    for step in range(140):
        price += 0.6 if step % 5 else -0.35
        values.append(price)
    return make_ohlcv(values, spread=0.3)


class FakeExchange:
    """Stand-in for a ccxt client. Records calls, returns canned data."""

    def __init__(
        self,
        ohlcv: list[list[Any]] | None = None,
        errors: list[Exception] | None = None,
        ticker: dict[str, Any] | None = None,
        order_book: dict[str, Any] | None = None,
    ) -> None:
        self.ohlcv = ohlcv if ohlcv is not None else default_raw_ohlcv()
        self.errors = list(errors or [])
        self.ticker = ticker or {"bid": 100.0, "ask": 100.2, "last": 100.1}
        self.order_book = order_book or {"bids": [[100.0, 1.0]], "asks": [[100.2, 1.0]]}
        self.calls: list[tuple[str, tuple, dict]] = []
        # Attributes ccxt clients expose; the read-only tripwire checks these.
        self.apiKey = None
        self.secret = None
        self.password = None
        self.uid = None

    def _maybe_raise(self) -> None:
        if self.errors:
            raise self.errors.pop(0)

    def fetch_ohlcv(self, symbol, timeframe=None, limit=None, since=None):
        self.calls.append(("fetch_ohlcv", (symbol,), {"timeframe": timeframe, "limit": limit}))
        self._maybe_raise()
        return [list(row) for row in self.ohlcv]

    def fetch_ticker(self, symbol):
        self.calls.append(("fetch_ticker", (symbol,), {}))
        self._maybe_raise()
        return dict(self.ticker)

    def fetch_order_book(self, symbol, limit=None):
        self.calls.append(("fetch_order_book", (symbol,), {"limit": limit}))
        self._maybe_raise()
        return {k: list(v) for k, v in self.order_book.items()}


def default_raw_ohlcv(count: int = 120) -> list[list[float]]:
    """ccxt-shaped rows: [ms_timestamp, open, high, low, close, volume]."""
    base_ms = 1_767_225_600_000  # 2026-01-01T00:00:00Z
    step_ms = 15 * 60 * 1000
    rows = []
    for i in range(count):
        close = 100.0 + i * 0.1
        rows.append([base_ms + i * step_ms, close, close + 0.5, close - 0.5, close, 10.0])
    return rows


@pytest.fixture
def fake_exchange() -> FakeExchange:
    return FakeExchange()


@pytest.fixture
def no_sleep():
    """A sleep function that records durations instead of waiting."""
    recorded: list[float] = []
    return recorded, recorded.append
