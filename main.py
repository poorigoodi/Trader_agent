"""Entry point for the paper-trading agent.

    python main.py                     # run forever, 5-minute cycles
    python main.py --cycles 3          # run three cycles and stop
    python main.py --interval 60       # poll every 60 seconds
    python main.py --exchange binance  # dev fallback data source

PAPER TRADING ONLY. This process cannot place a real order: it holds no
credentials, and no module it imports has an order-placement code path. See
README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from app.config.settings import (
    FALLBACK_EXCHANGE_ID,
    PaperTradingSettings,
    Settings,
    SUPPORTED_EXCHANGE_IDS,
    TARGET_EXCHANGE_ID,
    load_settings,
)
from app.orchestrator import TradingAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rule-based crypto paper-trading agent (no live execution)."
    )
    parser.add_argument(
        "--cycles", type=int, default=None,
        help="number of cycles to run; omit to run until interrupted",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="seconds between cycles (default: 300)",
    )
    parser.add_argument(
        "--exchange", choices=SUPPORTED_EXCHANGE_IDS, default=None,
        help=f"data source (default: {TARGET_EXCHANGE_ID}; "
             f"dev fallback: {FALLBACK_EXCHANGE_ID})",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="comma-separated symbols, e.g. 'BTC/USDT,ETH/USDT'",
    )
    parser.add_argument("--timeframe", type=str, default=None, help="candle timeframe")
    parser.add_argument(
        "--balance", type=float, default=None, help="starting virtual balance",
    )
    parser.add_argument(
        "--close-at-end", action="store_true",
        help="mark open positions to market when the run ends (reporting only)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Rebuild settings with CLI overrides applied (Settings is frozen)."""
    from dataclasses import replace

    changes: dict = {}
    if args.exchange:
        changes["exchange_id"] = args.exchange
    if args.symbols:
        changes["symbols"] = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    if args.timeframe:
        changes["timeframe"] = args.timeframe
    if args.interval is not None:
        changes["cycle_interval_seconds"] = args.interval
    if args.cycles is not None:
        changes["max_cycles"] = args.cycles
    if args.balance is not None:
        changes["paper"] = PaperTradingSettings(starting_balance=args.balance)
    return replace(settings, **changes) if changes else settings


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("main")

    settings = apply_overrides(load_settings(), args)

    banner = (
        f"PAPER TRADING ONLY | data source: {settings.exchange_id} "
        f"| target venue: {settings.target_exchange_id}"
    )
    if settings.is_dev_data_source:
        banner += " | WARNING: dev data source differs from target venue (basis risk)"
    log.info(banner)

    agent = TradingAgent(settings)
    agent.start_run(notes=banner)

    try:
        agent.run(max_cycles=args.cycles)
    finally:
        if args.close_at_end:
            closed = agent.close_all()
            if closed:
                log.info("marked %d open position(s) to market for reporting", closed)
                # run() already wrote the run row; refresh it with these exits.
                agent.finish_run()
        summary = agent.summary()
        log.info("run summary:\n%s", json.dumps(summary, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
