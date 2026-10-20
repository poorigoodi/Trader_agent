"""Historical OHLCV for backtesting, from a source separate from execution.

WHY A SEPARATE SOURCE
---------------------
The execution target (Toobit USDT-M perps) will not serve deep history: its
``fetch_ohlcv`` ignores ``since`` and always returns the newest 1000 candles
(~10 days on 15m, measured 2026-09-04). LBank's ccxt driver cannot even
complete ``load_markets`` from this network. Binance -- the obvious vendor --
is unreachable here entirely.

Reachable and paging correctly (measured, same date): KuCoin, Gate, HTX.
KuCoin is the default vendor: all five target bases are listed and 60 days of
15m candles page cleanly.

BASIS RISK IS REAL AND IS NOT HIDDEN
------------------------------------
KuCoin **spot** is not Toobit **perpetual futures**. They are different books
with different participants; perps additionally carry funding. Any backtest
built on this data therefore measures the STRATEGY, not the exact fills the
target venue would have given. Every frame and every cached file records the
vendor it came from, and :func:`measure_basis` quantifies the gap against the
execution venue rather than assuming it away.

This is the arrangement the project's ground rules call for ("Backtest data
may come from a different source than the execution venue -- this must be
managed explicitly"), not a shortcut.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# Vendors verified reachable and honouring `since` on 2026-09-04.
HISTORICAL_VENDORS = ("kucoin", "gate", "htx")
DEFAULT_VENDOR = "kucoin"

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

CACHE_DIR = Path("data/historical")


def to_spot_symbol(symbol: str) -> str:
    """Map a perp symbol to the vendor's spot equivalent.

    ``BTC/USDT:USDT`` -> ``BTC/USDT``. The settle suffix is what makes it a
    contract; spot vendors do not use it.
    """
    return symbol.split(":")[0]


@dataclass(frozen=True)
class HistoricalWindow:
    """A fetched history, with the provenance needed to interpret it."""

    symbol: str
    vendor_symbol: str
    timeframe: str
    vendor: str
    market_type: str
    frame: pd.DataFrame

    @property
    def candles(self) -> int:
        return len(self.frame)

    @property
    def span_days(self) -> float:
        if self.frame.empty:
            return 0.0
        delta = self.frame["timestamp"].iloc[-1] - self.frame["timestamp"].iloc[0]
        return delta.total_seconds() / 86_400

    def describe(self) -> str:
        return (
            f"{self.symbol} {self.timeframe}: {self.candles} candles "
            f"({self.span_days:.1f} days) from {self.vendor} {self.market_type}"
        )


class HistoricalDataSource:
    """Pages deep OHLCV history from a public vendor, with a local cache.

    Read-only and unauthenticated, like every other exchange touch in this
    codebase. There is no order path here either.
    """

    def __init__(
        self,
        vendor: str = DEFAULT_VENDOR,
        market_type: str = "spot",
        cache_dir: Path | str = CACHE_DIR,
        client: Any | None = None,
        sleep_fn=time.sleep,
    ) -> None:
        if vendor not in HISTORICAL_VENDORS and client is None:
            raise ValueError(
                f"vendor {vendor!r} not in verified list {HISTORICAL_VENDORS}"
            )
        self.vendor = vendor
        self.market_type = market_type
        self.cache_dir = Path(cache_dir)
        self._sleep = sleep_fn
        self.client = client or self._build_client(vendor, market_type)
        self._assert_read_only()

    @staticmethod
    def _build_client(vendor: str, market_type: str) -> Any:
        return getattr(ccxt, vendor)(
            {
                "enableRateLimit": True,
                "timeout": 25_000,
                "options": {"defaultType": market_type},
            }
        )

    def _assert_read_only(self) -> None:
        for attribute in ("apiKey", "secret", "password", "uid"):
            if getattr(self.client, attribute, None):
                raise RuntimeError(
                    "HistoricalDataSource refuses to run with credentials set; "
                    "this layer is read-only."
                )

    # -- fetching ---------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        timeframe: str = "15m",
        days: int = 60,
        page_limit: int = 1500,
        max_pages: int = 40,
        use_cache: bool = True,
    ) -> HistoricalWindow:
        """Page backwards ``days`` of candles, newest page last.

        Pages forward from ``since`` because that is what these vendors
        support; stops when a page returns nothing new, which is also the
        guard against an infinite loop on a vendor that ignores ``since``.
        """
        if timeframe not in TIMEFRAME_MS:
            raise ValueError(f"unsupported timeframe {timeframe!r}")

        vendor_symbol = to_spot_symbol(symbol) if self.market_type == "spot" else symbol
        cache_path = self._cache_path(vendor_symbol, timeframe, days)

        if use_cache and cache_path.exists():
            frame = pd.read_parquet(cache_path) if cache_path.suffix == ".parquet" \
                else pd.read_csv(cache_path, parse_dates=["timestamp"])
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            logger.info("loaded %d cached candles for %s", len(frame), vendor_symbol)
            return HistoricalWindow(
                symbol, vendor_symbol, timeframe, self.vendor, self.market_type, frame
            )

        step = TIMEFRAME_MS[timeframe]
        cursor = int(time.time() * 1000) - days * 86_400_000
        rows: list[list[float]] = []

        for page in range(max_pages):
            batch = self.client.fetch_ohlcv(
                vendor_symbol, timeframe, since=cursor, limit=page_limit
            )
            if not batch:
                break
            rows.extend(batch)
            advanced = batch[-1][0] + step
            if advanced <= cursor:
                # The vendor ignored `since`; one page is all we will get.
                logger.warning(
                    "%s did not advance past %s; stopping paging", self.vendor, cursor
                )
                break
            cursor = advanced
            if cursor > time.time() * 1000:
                break
            self._sleep(0.35)  # stay well inside the public rate limit

        frame = self._to_frame(rows)
        if use_cache and not frame.empty:
            self._write_cache(frame, cache_path)

        window = HistoricalWindow(
            symbol, vendor_symbol, timeframe, self.vendor, self.market_type, frame
        )
        logger.info("fetched %s", window.describe())
        return window

    def fetch_many(
        self, symbols: list[str], timeframe: str = "15m", days: int = 60, **kwargs
    ) -> dict[str, HistoricalWindow]:
        out: dict[str, HistoricalWindow] = {}
        for symbol in symbols:
            try:
                out[symbol] = self.fetch(symbol, timeframe, days, **kwargs)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the set
                logger.error("history for %s failed: %s", symbol, exc)
        return out

    # -- basis ------------------------------------------------------------

    @staticmethod
    def measure_basis(
        vendor_frame: pd.DataFrame, venue_frame: pd.DataFrame
    ) -> dict[str, float | int]:
        """Quantify the price gap between the data vendor and the venue.

        Reported, never assumed to be zero. A backtest edge smaller than this
        basis is not an edge, it is a data artefact.
        """
        left = vendor_frame[["timestamp", "close"]].rename(columns={"close": "vendor"})
        right = venue_frame[["timestamp", "close"]].rename(columns={"close": "venue"})
        merged = left.merge(right, on="timestamp", how="inner")
        if merged.empty:
            return {"overlapping_candles": 0}

        relative = (merged["venue"] - merged["vendor"]) / merged["vendor"]
        return {
            "overlapping_candles": int(len(merged)),
            "mean_basis_pct": float(relative.mean() * 100),
            "median_basis_pct": float(relative.median() * 100),
            "max_abs_basis_pct": float(relative.abs().max() * 100),
            "stdev_basis_pct": float(relative.std() * 100),
        }

    # -- internals --------------------------------------------------------

    def _cache_path(self, vendor_symbol: str, timeframe: str, days: int) -> Path:
        safe = vendor_symbol.replace("/", "-").replace(":", "_")
        return self.cache_dir / f"{self.vendor}_{safe}_{timeframe}_{days}d.csv"

    def _write_cache(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    @staticmethod
    def _to_frame(rows: list[list[float]]) -> pd.DataFrame:
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, unit="ms")
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = frame[column].astype("float64")
        return (
            frame.drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
