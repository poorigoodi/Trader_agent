"""Generate the labelled signal dataset by replaying historical candles.

This is the input for both the model bake-off and the confidence calibrator,
so it is a script rather than a notebook: the dataset must be reproducible.

    python -m evaluation.run_backtest --days 180
    python -m evaluation.run_backtest --days 60 --vendor gate --timeframe 1h

Historical data comes from a vendor SEPARATE from the execution venue (see
``app/backtesting/data_source.py`` for why), so the basis between them is
measured and printed rather than assumed to be zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from app.backtesting.data_source import (
    DEFAULT_VENDOR,
    HISTORICAL_VENDORS,
    HistoricalDataSource,
)
from app.backtesting.engine import Backtester
from app.config.settings import DEFAULT_SWAP_SYMBOLS, Settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest and emit a labelled dataset.")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--vendor", default=DEFAULT_VENDOR, choices=HISTORICAL_VENDORS)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SWAP_SYMBOLS))
    parser.add_argument("--out", type=Path, default=Path("data/labelled_signals.csv"))
    parser.add_argument("--metrics-out", type=Path, default=Path("data/backtest_metrics.json"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    source = HistoricalDataSource(args.vendor)

    logger.info("fetching %d days of %s from %s", args.days, args.timeframe, args.vendor)
    windows = source.fetch_many(
        symbols, args.timeframe, days=args.days, use_cache=not args.no_cache
    )
    if not windows:
        logger.error("no history could be fetched")
        return 1
    for window in windows.values():
        logger.info("  %s", window.describe())

    frames = {symbol: window.frame for symbol, window in windows.items()}
    settings = Settings(timeframe=args.timeframe)

    started = time.perf_counter()
    result = Backtester(settings).run(
        frames, vendor=args.vendor, market_type=source.market_type, progress_every=4000
    )
    logger.info("backtest finished in %.0fs", time.perf_counter() - started)

    metrics = result.metrics()
    metrics["days"] = args.days
    metrics["symbols"] = symbols
    metrics["timeframe"] = args.timeframe
    metrics["fee_pct"] = settings.paper.taker_fee_pct
    metrics["slippage_pct"] = settings.paper.slippage_pct

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = result.to_frame()
    frame.to_csv(args.out, index=False)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    labelled = frame[frame["won"].notna()]
    print(json.dumps(metrics, indent=2, default=str))
    print(
        f"\nwrote {len(frame)} signal rows ({len(labelled)} with outcomes) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
