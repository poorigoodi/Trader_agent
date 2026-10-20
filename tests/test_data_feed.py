"""Unit tests for the ccxt market data feed.

NO TEST HERE TOUCHES THE NETWORK. Every exchange interaction goes through
``FakeExchange`` from conftest. That is not just for speed: a test suite that
silently depends on an exchange being up stops being a test suite the first
time the venue has an outage.
"""

from __future__ import annotations

import ccxt
import pandas as pd
import pytest

from app.market.data_feed import (
    MarketDataError,
    MarketDataFeed,
    RetryPolicy,
)
from tests.conftest import FakeExchange, default_raw_ohlcv


def feed(client=None, **kwargs) -> MarketDataFeed:
    """A feed wired to a fake client and a sleep that never actually sleeps."""
    kwargs.setdefault("sleep_fn", lambda _: None)
    return MarketDataFeed(client=client or FakeExchange(), **kwargs)


# --------------------------------------------------------------------------
# the read-only tripwire
# --------------------------------------------------------------------------

@pytest.mark.parametrize("attribute", ["apiKey", "secret", "password", "uid"])
def test_feed_refuses_to_start_with_credentials(attribute):
    """The single most important test in this file: if anyone ever wires a
    key into this layer, the agent must refuse to run rather than quietly
    gain the ability to authenticate."""
    client = FakeExchange()
    setattr(client, attribute, "leaked-secret-value")
    with pytest.raises(RuntimeError, match="read-only"):
        MarketDataFeed(client=client)


def test_feed_accepts_a_client_with_empty_credentials():
    client = FakeExchange()
    client.apiKey = ""
    client.secret = None
    MarketDataFeed(client=client)  # must not raise


def test_built_client_has_no_credentials():
    real = MarketDataFeed._build_client("toobit")
    assert not real.apiKey and not real.secret


def test_built_client_respects_rate_limits():
    """Hammering a public endpoint is how you get IP-banned from a venue."""
    assert MarketDataFeed._build_client("toobit").enableRateLimit is True


def test_feed_has_no_order_placement_surface():
    """A structural assertion: this layer must not expose trading methods."""
    instance = feed()
    for forbidden in (
        "create_order", "create_market_order", "create_limit_order",
        "cancel_order", "fetch_balance", "fetch_my_trades", "withdraw",
    ):
        assert not hasattr(instance, forbidden)


def test_unsupported_exchange_is_refused():
    with pytest.raises(ValueError, match="not supported"):
        MarketDataFeed(exchange_id="definitely_not_an_exchange")


# --------------------------------------------------------------------------
# frame shape
# --------------------------------------------------------------------------

def test_fetch_ohlcv_returns_the_documented_columns():
    frame = feed().fetch_ohlcv("BTC/USDT", "15m", 100)
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_timestamps_are_utc_aware_datetimes():
    frame = feed().fetch_ohlcv("BTC/USDT")
    assert isinstance(frame["timestamp"].iloc[0], pd.Timestamp)
    assert str(frame["timestamp"].dt.tz) == "UTC"


def test_price_columns_are_floats():
    frame = feed().fetch_ohlcv("BTC/USDT")
    for column in ("open", "high", "low", "close", "volume"):
        assert frame[column].dtype == "float64"


def test_the_forming_candle_is_dropped_by_default():
    """Acting on a candle that has not closed produces signals that vanish
    on the next poll."""
    raw = default_raw_ohlcv(50)
    frame = feed(FakeExchange(ohlcv=raw)).fetch_ohlcv("BTC/USDT")
    assert len(frame) == 49
    assert frame["close"].iloc[-1] == raw[-2][4]


def test_the_forming_candle_can_be_kept_explicitly():
    raw = default_raw_ohlcv(50)
    frame = feed(FakeExchange(ohlcv=raw), drop_unclosed_candle=False).fetch_ohlcv("BTC/USDT")
    assert len(frame) == 50


def test_rows_are_sorted_ascending():
    raw = list(reversed(default_raw_ohlcv(30)))
    frame = feed(FakeExchange(ohlcv=raw)).fetch_ohlcv("BTC/USDT")
    assert frame["timestamp"].is_monotonic_increasing


def test_duplicate_timestamps_are_collapsed():
    """Polling re-fetches overlapping windows constantly; a duplicated bar
    would corrupt every later count and every indicator."""
    raw = default_raw_ohlcv(10)
    frame = feed(FakeExchange(ohlcv=raw + raw)).fetch_ohlcv("BTC/USDT")
    assert frame["timestamp"].is_unique


def test_duplicate_keeps_the_last_value_seen():
    raw = default_raw_ohlcv(3)
    revised = [list(raw[-1])]
    revised[0][4] = 999.0
    frame = feed(
        FakeExchange(ohlcv=raw + revised), drop_unclosed_candle=False
    ).fetch_ohlcv("BTC/USDT")
    assert frame["close"].iloc[-1] == 999.0


def test_empty_response_gives_an_empty_but_valid_frame():
    frame = feed(FakeExchange(ohlcv=[])).fetch_ohlcv("BTC/USDT")
    assert frame.empty
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_single_candle_is_not_dropped_to_nothing():
    frame = feed(FakeExchange(ohlcv=default_raw_ohlcv(1))).fetch_ohlcv("BTC/USDT")
    assert len(frame) == 1


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def test_frame_records_which_venue_it_came_from():
    """Numbers from different venues must never be silently comparable."""
    frame = MarketDataFeed(
        exchange_id="binance", client=FakeExchange(), sleep_fn=lambda _: None
    ).fetch_ohlcv("BTC/USDT", "15m")
    assert frame.attrs["exchange_id"] == "binance"
    assert frame.attrs["is_dev_data_source"] is True
    assert frame.attrs["symbol"] == "BTC/USDT"
    assert frame.attrs["timeframe"] == "15m"


def test_target_venue_is_not_flagged_as_a_dev_source():
    instance = MarketDataFeed(
        exchange_id="toobit", client=FakeExchange(), sleep_fn=lambda _: None
    )
    assert instance.is_dev_data_source is False


def test_empty_frame_still_carries_provenance():
    frame = feed(FakeExchange(ohlcv=[])).fetch_ohlcv("BTC/USDT")
    assert "exchange_id" in frame.attrs


# --------------------------------------------------------------------------
# retry / backoff
# --------------------------------------------------------------------------

def test_transient_network_error_is_retried_then_succeeds():
    client = FakeExchange(errors=[ccxt.NetworkError("blip")])
    frame = feed(client).fetch_ohlcv("BTC/USDT")
    assert not frame.empty
    assert len(client.calls) == 2


def test_retries_stop_at_max_attempts_and_raise_market_data_error():
    client = FakeExchange(errors=[ccxt.NetworkError("down")] * 10)
    instance = feed(client, retry_policy=RetryPolicy(max_attempts=3))
    with pytest.raises(MarketDataError, match="after 3 attempts"):
        instance.fetch_ohlcv("BTC/USDT")
    assert len(client.calls) == 3


def test_rate_limit_errors_are_retried():
    client = FakeExchange(errors=[ccxt.RateLimitExceeded("slow down")])
    assert not feed(client).fetch_ohlcv("BTC/USDT").empty


def test_bad_symbol_is_not_retried():
    """A wrong symbol will never start working; retrying just wastes the
    rate-limit budget."""
    client = FakeExchange(errors=[ccxt.BadSymbol("NOPE/USDT")])
    with pytest.raises(MarketDataError, match="unknown symbol"):
        feed(client).fetch_ohlcv("NOPE/USDT")
    assert len(client.calls) == 1


def test_backoff_delays_grow_exponentially_and_are_capped():
    policy = RetryPolicy(
        base_delay_seconds=1.0, backoff_factor=2.0, max_delay_seconds=5.0
    )
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(3) == 4.0
    assert policy.delay_for(4) == 5.0  # capped
    assert policy.delay_for(9) == 5.0


def test_feed_actually_sleeps_between_retries():
    slept: list[float] = []
    client = FakeExchange(errors=[ccxt.NetworkError("a"), ccxt.NetworkError("b")])
    instance = MarketDataFeed(client=client, sleep_fn=slept.append)
    instance.fetch_ohlcv("BTC/USDT")
    assert slept == [1.0, 2.0]


# --------------------------------------------------------------------------
# ticker / order book / spread
# --------------------------------------------------------------------------

def test_fetch_ticker_returns_a_plain_dict():
    assert feed().fetch_ticker("BTC/USDT")["last"] == 100.1


def test_observed_spread_uses_the_ticker_when_it_has_bid_and_ask():
    spread = feed().observed_spread_pct("BTC/USDT")
    assert spread == pytest.approx(0.2 / 100.1)


def test_observed_spread_falls_back_to_the_order_book():
    """LBank returns null bid/ask on its ticker; the book still works."""
    client = FakeExchange(
        ticker={"bid": None, "ask": None, "last": 100.0},
        order_book={"bids": [[99.0, 1.0]], "asks": [[101.0, 1.0]]},
    )
    assert feed(client).observed_spread_pct("BTC/USDT") == pytest.approx(2.0 / 100.0)


def test_observed_spread_is_none_when_nothing_is_quoted():
    client = FakeExchange(
        ticker={"bid": None, "ask": None}, order_book={"bids": [], "asks": []}
    )
    assert feed(client).observed_spread_pct("BTC/USDT") is None


def test_observed_spread_is_none_when_the_venue_is_down():
    client = FakeExchange(errors=[ccxt.NetworkError("down")] * 20)
    assert feed(client).observed_spread_pct("BTC/USDT") is None


# --------------------------------------------------------------------------
# polling
# --------------------------------------------------------------------------

def test_stream_latest_fires_once_for_one_unchanging_candle_set():
    received: list[pd.DataFrame] = []
    fired = feed().stream_latest(
        "BTC/USDT", "15m", received.append, poll_seconds=0, max_polls=3
    )
    assert fired == 1  # the newest timestamp never advanced
    assert len(received) == 1


def test_stream_latest_fires_again_when_a_new_candle_appears():
    client = FakeExchange(ohlcv=default_raw_ohlcv(20))
    instance = feed(client)
    received: list[pd.DataFrame] = []

    original = client.fetch_ohlcv
    calls = {"n": 0}

    def growing(*args, **kwargs):
        calls["n"] += 1
        client.ohlcv = default_raw_ohlcv(20 + calls["n"])
        return original(*args, **kwargs)

    client.fetch_ohlcv = growing
    fired = instance.stream_latest(
        "BTC/USDT", "15m", received.append, poll_seconds=0, max_polls=3
    )
    assert fired == 3


def test_stream_latest_survives_a_failing_poll():
    """A venue outage must not kill the loop."""
    client = FakeExchange(errors=[ccxt.NetworkError("x")] * 10)
    received: list[pd.DataFrame] = []
    fired = feed(
        client, retry_policy=RetryPolicy(max_attempts=1)
    ).stream_latest("BTC/USDT", "15m", received.append, poll_seconds=0, max_polls=2)
    assert fired == 0
    assert received == []


def test_stream_latest_hands_the_callback_a_full_frame():
    received: list[pd.DataFrame] = []
    feed().stream_latest("BTC/USDT", "15m", received.append, poll_seconds=0, max_polls=1)
    assert len(received[0]) > 50  # the whole warmed-up window, not one row
