"""Read-only market data access via ccxt.

SAFETY BOUNDARY
---------------
This module is the ONLY place in the codebase that talks to an exchange, and
it talks to public endpoints only. The ccxt client is constructed without
apiKey/secret, and ``MarketDataFeed._assert_read_only`` re-checks that on
every construction. There is deliberately no ``create_order``, no
``fetch_balance``, and no account method anywhere in this package: live
execution is a separate, later layer (roadmap Phase 8), not a flag on this
class.

DATA SOURCE vs TARGET EXCHANGE
------------------------------
The final target venues are LBank and Toobit, market type USDT-M futures.
The default is TOOBIT, chosen by measurement rather than preference: on this
dev machine (2026-09-04) Toobit serves all five target perps with full 15m
history, LBank's ccxt driver cannot complete ``load_markets`` because it
reaches for an unreachable perp host, and api.binance.com is unreachable
entirely. Binance stays selectable as a dev-only fallback where it works and
deeper history is wanted. See ``app/config/settings.py`` for the full note.

Every frame this module returns carries ``attrs["exchange_id"]``,
``attrs["market_type"]`` and ``attrs["is_dev_data_source"]`` so downstream
code and the database always record WHICH venue and WHICH market a number
came from. Prices between venues are not identical (LBank's average spread is
roughly 0.44 percent and Toobit's 0.54 percent, versus under 0.02 percent on
tier-1 venues), so this basis risk is recorded, never assumed away.

``verify_symbols`` exists because a listing must never be assumed: it asks
the venue what it actually offers and reports anything missing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import ccxt
import pandas as pd

from app.config.settings import (
    DEFAULT_EXCHANGE_ID,
    DEFAULT_MARKET_TYPE,
    SUPPORTED_EXCHANGE_IDS,
    TARGET_EXCHANGE_IDS,
)

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# ccxt errors that are worth retrying rather than crashing the agent loop.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
)


class MarketDataError(RuntimeError):
    """Raised when market data could not be retrieved after all retries."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    max_delay_seconds: float = 30.0

    def delay_for(self, attempt: int) -> float:
        """Exponential backoff delay before attempt number ``attempt`` (1-based)."""
        delay = self.base_delay_seconds * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay_seconds)


class MarketDataFeed:
    """Public OHLCV / ticker access for one exchange.

    Parameters
    ----------
    exchange_id:
        ccxt id. Defaults to the production target (``lbank``);
        ``binance`` is the dev-only fallback.
    client:
        Pre-built ccxt client, used by tests to inject a mock. When given,
        it is still checked for credentials.
    drop_unclosed_candle:
        When True (the default) the most recent candle is discarded because
        it is still forming. Acting on a forming candle is the most common
        source of phantom signals that vanish on the next poll.
    """

    def __init__(
        self,
        exchange_id: str = DEFAULT_EXCHANGE_ID,
        client: Any | None = None,
        retry_policy: RetryPolicy | None = None,
        drop_unclosed_candle: bool = True,
        sleep_fn: Callable[[float], None] = time.sleep,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> None:
        exchange_id = exchange_id.lower()
        if exchange_id not in SUPPORTED_EXCHANGE_IDS:
            raise ValueError(
                f"exchange_id {exchange_id!r} not supported; "
                f"expected one of {SUPPORTED_EXCHANGE_IDS}"
            )

        self.exchange_id = exchange_id
        self.market_type = market_type
        self.retry_policy = retry_policy or RetryPolicy()
        self.drop_unclosed_candle = drop_unclosed_candle
        self._sleep = sleep_fn
        self.client = (
            client if client is not None else self._build_client(exchange_id, market_type)
        )
        self._assert_read_only()

    # -- construction -----------------------------------------------------

    @staticmethod
    def _build_client(exchange_id: str, market_type: str = DEFAULT_MARKET_TYPE) -> Any:
        """Build a ccxt client with NO credentials attached."""
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class(
            {
                "enableRateLimit": True,  # respect the venue, avoid bans
                "timeout": 20_000,
                "options": {"defaultType": market_type},
            }
        )

    def _assert_read_only(self) -> None:
        """Fail loudly if credentials ever appear on the client.

        This is a tripwire, not decoration: if someone later wires keys in,
        the agent refuses to start rather than quietly gaining the ability
        to authenticate.
        """
        guarded = ("apiKey", "secret", "password", "uid", "privateKey", "walletAddress")
        for attribute in guarded:
            if getattr(self.client, attribute, None):
                raise RuntimeError(
                    f"MarketDataFeed refuses to run with credentials set "
                    f"({attribute!r} is populated). This layer is read-only."
                )

    @property
    def is_dev_data_source(self) -> bool:
        """True when this venue is not one we intend to trade on."""
        return self.exchange_id not in TARGET_EXCHANGE_IDS

    # -- symbol verification ----------------------------------------------

    def verify_symbols(self, symbols: Iterable[str]) -> tuple[list[str], dict[str, str]]:
        """Split ``symbols`` into (available, {rejected: why}) via load_markets.

        Never assume a listing. A symbol the venue does not actually offer
        would otherwise fail silently every cycle, or -- worse -- resolve to
        a different market than intended (spot instead of the perp).

        If ``load_markets`` itself fails (LBank's driver reaches for an
        unreachable perp host from some networks), every symbol is reported
        as unverified rather than silently accepted.
        """
        requested = list(symbols)
        try:
            markets = self._with_retry("load_markets", self.client.load_markets)
        except MarketDataError as exc:
            reason = f"could not verify: load_markets failed ({exc})"
            return [], {symbol: reason for symbol in requested}

        available: list[str] = []
        rejected: dict[str, str] = {}
        for symbol in requested:
            market = markets.get(symbol)
            if market is None:
                rejected[symbol] = f"not listed on {self.exchange_id}"
                continue
            actual = market.get("type")
            if self.market_type == "swap" and not market.get("swap"):
                rejected[symbol] = (
                    f"listed on {self.exchange_id} but as {actual!r}, "
                    f"not a USDT-M perpetual"
                )
                continue
            if self.market_type == "swap" and not market.get("linear", True):
                rejected[symbol] = "inverse contract, not USDT-margined"
                continue
            available.append(symbol)

        if rejected:
            for symbol, why in rejected.items():
                logger.warning("dropping symbol %s: %s", symbol, why)
        return available, rejected

    # -- retry ------------------------------------------------------------

    def _with_retry(self, description: str, fn: Callable[[], Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                return fn()
            except ccxt.BadSymbol as exc:
                # Not transient: a wrong symbol will never start working.
                raise MarketDataError(f"{description}: unknown symbol - {exc}") from exc
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt == self.retry_policy.max_attempts:
                    break
                delay = self.retry_policy.delay_for(attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                    description,
                    attempt,
                    self.retry_policy.max_attempts,
                    type(exc).__name__,
                    delay,
                )
                self._sleep(delay)

        raise MarketDataError(
            f"{description} failed after {self.retry_policy.max_attempts} "
            f"attempts: {last_error}"
        ) from last_error

    # -- public data ------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 300,
        since: int | None = None,
    ) -> pd.DataFrame:
        """Fetch candles as a DataFrame indexed 0..n-1.

        Columns: timestamp (UTC-aware), open, high, low, close, volume.
        The frame is sorted ascending and de-duplicated on timestamp; the
        still-forming last candle is dropped unless disabled.
        """
        raw = self._with_retry(
            f"fetch_ohlcv({symbol}, {timeframe})",
            lambda: self.client.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit, since=since
            ),
        )
        return self._to_frame(raw, symbol=symbol, timeframe=timeframe)

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch the public ticker (last/bid/ask) for a symbol."""
        ticker = self._with_retry(
            f"fetch_ticker({symbol})", lambda: self.client.fetch_ticker(symbol)
        )
        return dict(ticker)

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """Fetch the public order book. Used only to measure spread today."""
        book = self._with_retry(
            f"fetch_order_book({symbol})",
            lambda: self.client.fetch_order_book(symbol, limit=limit),
        )
        return dict(book)

    def observed_spread_pct(self, symbol: str) -> float | None:
        """Top-of-book spread as a fraction of mid price, or None.

        Recorded so that the conservative slippage assumption in paper
        trading can later be checked against what the venue actually quotes.

        LBank's ``fetch_ticker`` returns null bid/ask, so we fall back to the
        order book rather than reporting "no spread" for our target venue.
        Note this is the *top-of-book* spread only: a real fill of any size
        walks the book and costs more, which is why the paper engine's
        slippage assumption stays conservative regardless of this number.
        """
        try:
            ticker = self.fetch_ticker(symbol)
            bid, ask = ticker.get("bid"), ticker.get("ask")
            if not bid or not ask:
                book = self.fetch_order_book(symbol, limit=5)
                bids, asks = book.get("bids") or [], book.get("asks") or []
                if not bids or not asks:
                    return None
                bid, ask = bids[0][0], asks[0][0]
        except MarketDataError:
            return None

        mid = (float(bid) + float(ask)) / 2.0
        return (float(ask) - float(bid)) / mid if mid > 0 else None

    # -- polling ----------------------------------------------------------

    def stream_latest(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[pd.DataFrame], None],
        poll_seconds: int = 60,
        limit: int = 300,
        max_polls: int | None = None,
    ) -> int:
        """Poll for candles and hand each NEW closed candle to ``callback``.

        WebSocket streaming (ccxt.pro) is deliberately not used yet: polling
        is enough for a 15m timeframe and keeps the failure modes obvious.
        The callback receives the full warmed-up frame, not a single row, so
        it can compute indicators without keeping state of its own.

        Returns the number of callback invocations made.
        """
        last_seen: pd.Timestamp | None = None
        fired = 0
        polls = 0

        while max_polls is None or polls < max_polls:
            polls += 1
            try:
                frame = self.fetch_ohlcv(symbol, timeframe, limit)
            except MarketDataError as exc:
                logger.error("stream_latest(%s): %s", symbol, exc)
                self._sleep(poll_seconds)
                continue

            if not frame.empty:
                newest = pd.Timestamp(frame["timestamp"].iloc[-1])
                if last_seen is None or newest > last_seen:
                    last_seen = newest
                    callback(frame)
                    fired += 1

            if max_polls is None or polls < max_polls:
                self._sleep(poll_seconds)

        return fired

    # -- internals --------------------------------------------------------

    def _to_frame(self, raw: Iterable[Any], symbol: str, timeframe: str) -> pd.DataFrame:
        rows = list(raw or [])
        frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)

        if frame.empty:
            for column in ("open", "high", "low", "close", "volume"):
                frame[column] = frame[column].astype("float64")
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, unit="ms")
        else:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, unit="ms")
            for column in ("open", "high", "low", "close", "volume"):
                frame[column] = frame[column].astype("float64")
            frame = (
                frame.drop_duplicates(subset="timestamp", keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if self.drop_unclosed_candle and len(frame) > 1:
                frame = frame.iloc[:-1].reset_index(drop=True)

        frame.attrs["exchange_id"] = self.exchange_id
        frame.attrs["market_type"] = self.market_type
        frame.attrs["is_dev_data_source"] = self.is_dev_data_source
        frame.attrs["symbol"] = symbol
        frame.attrs["timeframe"] = timeframe
        return frame
