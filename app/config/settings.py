"""Central, explicit configuration for the trading agent.

Everything the agent's behaviour depends on lives here so that a run is
reproducible from a single object. Values can be overridden from the
environment (see ``load_settings``) but the defaults are the documented ones.

SAFETY: this module intentionally has no field for an API key or secret.
The agent only ever touches public/read-only endpoints at this stage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------
# Exchange / data source
# --------------------------------------------------------------------------
# The FINAL TARGET venues are LBank and Toobit, market type USDT-M FUTURES
# (linear perpetual swaps). Both are supported by ccxt.
#
# Which of the two is the DEFAULT was decided by measurement, not preference
# (measured on this dev machine, 2026-09-04):
#
#   toobit  - load_markets() OK: 1308 markets, 756 linear swaps. All five
#             target pairs exist as USDT-M perps and return a full 1000-candle
#             15m history. Top-of-book BTC perp spread measured at ~0.0001%.
#   lbank   - ccxt's LBank driver ALWAYS calls lbkperp.lbank.com during
#             load_markets(), and that host is unreachable from this network.
#             Spot OHLCV sometimes succeeds (BTC/ETH/BNB/XRP did, SOL did not),
#             so LBank is intermittent here and its futures side is unusable.
#   binance - api.binance.com is unreachable from this network entirely.
#
# So: default to TOOBIT futures, which is a real target venue with verified
# futures data. LBank stays selectable and is still a target; Binance stays as
# a documented dev-only fallback for machines where it is reachable and deeper
# history is wanted.
#
# The distinction stays a *first-class* concept, never an implicit default:
#   TARGET_EXCHANGE_IDS  -> venues the system is ultimately meant to trade
#   FALLBACK_EXCHANGE_ID -> dev-only data source, NOT a trading target
#
# Basis risk between venues (different books, different spreads) is real and
# is recorded on every frame and every DB row, not assumed away.
# See docs/architecture.md.
TARGET_EXCHANGE_IDS = ("toobit", "lbank")
TARGET_EXCHANGE_ID = "toobit"
FALLBACK_EXCHANGE_ID = "binance"
DEFAULT_EXCHANGE_ID = TARGET_EXCHANGE_ID
SUPPORTED_EXCHANGE_IDS = ("toobit", "lbank", "binance")

# USDT-M futures. ccxt spells linear perpetuals "swap"; the unified symbol is
# "BASE/QUOTE:SETTLE", e.g. "BTC/USDT:USDT". Spot is kept selectable because
# the Binance fallback is used for deep historical spot data in backtesting.
MARKET_TYPE_SWAP = "swap"
MARKET_TYPE_SPOT = "spot"
DEFAULT_MARKET_TYPE = MARKET_TYPE_SWAP

# The five target pairs: top-5 by market cap excluding stablecoins.
# VERIFIED present as Toobit USDT-M linear perps on 2026-09-04 -- each one
# returned 300+ candles on the 15m timeframe. Never assume a listing; the
# orchestrator re-verifies with load_markets() at startup and drops, with a
# loud warning, anything the venue does not actually list.
#
# NOTE: this ranking moves. Re-check it against CoinGecko/CMC before relying
# on it -- see README "Open Decisions" (OD-6).
DEFAULT_SWAP_SYMBOLS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "SOL/USDT:USDT",
)


@dataclass(frozen=True)
class RiskSettings:
    """Deterministic risk parameters. No model, no heuristic, no LLM."""

    # Fraction of the *current* virtual balance risked on a single trade.
    risk_per_trade_pct: float = 0.01  # 1%

    # Hard ceiling on cumulative risk opened within one UTC day.
    max_daily_risk_pct: float = 0.03  # 3%

    # Refuse to open more than this many concurrent positions.
    #
    # Raised from 2 to 4 when the universe went from 2 symbols to 5. It is
    # deliberately NOT 5: with the correlation cap below doing the real work,
    # this stays a blunt backstop against a single cascade opening the whole
    # book. Four positions at 1% each is 4% of balance at risk in the worst
    # case where every stop fills.
    max_open_positions: int = 4

    # --- correlation risk -------------------------------------------------
    # With BTC/ETH only, this was a placeholder: the two are ~0.8+ correlated
    # and the position cap was the real limit. With five large-cap pairs that
    # all carry high beta to crypto as a whole, "four uncorrelated positions"
    # is a fiction -- four longs on BTC, ETH, SOL and BNB is close to one
    # levered long on the market. These two knobs make that explicit.

    # Pairs whose rolling return correlation is at or above this belong to
    # the same risk cluster.
    correlation_threshold: float = 0.70

    # Total risk allowed across any ONE correlated cluster, as a fraction of
    # balance. Deliberately below max_daily_risk_pct: the daily cap limits
    # risk taken over time, this limits risk concentrated in one direction of
    # one market at one moment.
    max_correlated_risk_pct: float = 0.02  # 2%

    # Candles of returns used for the rolling correlation matrix. 96 x 15m is
    # one day -- long enough to be stable, short enough to react to a regime
    # change within a session.
    correlation_lookback: int = 96

    # Below this many overlapping observations, correlation is noise. We then
    # fail SAFE (assume correlated) rather than assume independence.
    correlation_min_observations: int = 30

    # A single position's notional may never exceed this fraction of balance
    # (protects against a micro-stop producing an absurd size).
    max_position_notional_pct: float = 0.50  # 50%

    # Stop loss distance = atr_multiplier * ATR(14) from entry.
    atr_stop_multiplier: float = 1.5

    # Take profit distance = rr_target * stop distance.
    rr_target: float = 2.0

    # Reject any setup whose reward:risk is below this.
    min_rr: float = 1.5

    # Exchange minimum notional guard (USDT). Below this we do not trade.
    min_notional_usdt: float = 10.0


@dataclass(frozen=True)
class StrategySettings:
    """Rule-based strategy parameters (EMA cross + RSI filter + structure)."""

    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14

    # RSI gate: longs need RSI above the floor but not already exhausted.
    rsi_long_min: float = 50.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    rsi_short_max: float = 50.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    bb_period: int = 20
    bb_std: float = 2.0

    # Market-structure lookback (swing highs/lows) used for trend confirmation.
    structure_lookback: int = 20
    structure_swing_window: int = 3

    # A cross is only "fresh" for this many closed candles.
    cross_max_age: int = 3


@dataclass(frozen=True)
class PaperTradingSettings:
    """Simulated execution costs, derived from MEASURED venue data.

    These numbers were originally set from CoinGecko's whole-exchange average
    spot spreads (LBank ~0.44%, Toobit ~0.54%), which produced a round-trip
    cost of 0.70%. Measured against the actual target market -- USDT-M perps
    on the five majors -- that was wrong by one to two orders of magnitude,
    and it made every backtest lose ~2R per trade for arithmetic reasons
    rather than strategy reasons.

    Measured on Toobit USDT-M perps, 2026-09-04, cost to fill 10,000 USDT by
    walking the live order book:

        BTC  0.00038%    ETH  0.00813%    BNB  0.00487%
        XRP  0.01763%    SOL  0.00980%

    So ``slippage_pct`` is set to 0.05% -- roughly three times the WORST
    measured value, as a buffer for fast markets where books thin out and
    stops fill worse than a calm-market snapshot suggests. That is
    conservative without being fictional.

    ``taker_fee_pct`` is 0.06%, the standard USDT-M futures taker rate. Toobit
    does not publish its fee schedule through ccxt (``market['taker']`` is
    None), so this is an ASSUMPTION, not a measurement -- see README
    "Open Decisions" (OD-8).

    Round trip at these settings: 2 x (0.06% + 0.05%) = 0.22%.
    Still large next to a 1.5 x ATR stop on 15m (~0.30-0.45% on these
    symbols); see docs/backtest_results.md for what that implies.
    """

    starting_balance: float = 10_000.0
    quote_currency: str = "USDT"

    # Taker fee per side, as a fraction of notional. ASSUMED (see above).
    taker_fee_pct: float = 0.0006  # 0.06%

    # Slippage per side, as a fraction of price, applied against us on both
    # entry and exit. ~3x the worst measured book-walk cost.
    slippage_pct: float = 0.0005  # 0.05%

    # The pre-measurement defaults, kept so the sensitivity analysis in
    # docs/backtest_results.md can be reproduced exactly.
    LEGACY_WIDE_SPREAD_FEE: float = 0.001
    LEGACY_WIDE_SPREAD_SLIPPAGE: float = 0.0025


@dataclass(frozen=True)
class Settings:
    """Top-level settings object passed explicitly through the whole app."""

    # --- data source ---
    exchange_id: str = DEFAULT_EXCHANGE_ID
    target_exchange_id: str = TARGET_EXCHANGE_ID
    market_type: str = DEFAULT_MARKET_TYPE
    symbols: tuple[str, ...] = DEFAULT_SWAP_SYMBOLS
    timeframe: str = "15m"
    candle_limit: int = 300
    # Verify each symbol against load_markets() before trading it, and drop
    # (loudly) anything the venue does not actually list.
    verify_symbols: bool = True

    # --- loop ---
    cycle_interval_seconds: int = 300  # 5 minutes
    max_cycles: int | None = None  # None = run forever

    # --- storage ---
    database_url: str = "sqlite:///data/trading_agent.db"

    # --- sub-configs ---
    risk: RiskSettings = field(default_factory=RiskSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    paper: PaperTradingSettings = field(default_factory=PaperTradingSettings)

    def __post_init__(self) -> None:
        if self.exchange_id not in SUPPORTED_EXCHANGE_IDS:
            raise ValueError(
                f"exchange_id {self.exchange_id!r} not in {SUPPORTED_EXCHANGE_IDS}"
            )
        if self.market_type not in (MARKET_TYPE_SWAP, MARKET_TYPE_SPOT):
            raise ValueError(f"market_type {self.market_type!r} must be swap or spot")
        if self.candle_limit < 100:
            raise ValueError("candle_limit must be >= 100 to warm up EMA(50)/MACD")
        if not self.symbols:
            raise ValueError("at least one symbol is required")

    @property
    def is_dev_data_source(self) -> bool:
        """True when the data source is not one of the target trading venues."""
        return self.exchange_id not in TARGET_EXCHANGE_IDS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def load_settings() -> Settings:
    """Build settings, allowing a few env overrides for convenience.

    Only non-sensitive knobs are overridable. There is deliberately no
    ``API_KEY``/``API_SECRET`` path anywhere in this function.
    """
    symbols_raw = os.getenv("AGENT_SYMBOLS")
    symbols = (
        tuple(s.strip() for s in symbols_raw.split(",") if s.strip())
        if symbols_raw
        else DEFAULT_SWAP_SYMBOLS
    )
    return Settings(
        exchange_id=os.getenv("AGENT_EXCHANGE", DEFAULT_EXCHANGE_ID).lower(),
        market_type=os.getenv("AGENT_MARKET_TYPE", DEFAULT_MARKET_TYPE).lower(),
        symbols=symbols,
        timeframe=os.getenv("AGENT_TIMEFRAME", "15m"),
        candle_limit=_env_int("AGENT_CANDLE_LIMIT", 300),
        cycle_interval_seconds=_env_int("AGENT_CYCLE_SECONDS", 300),
        database_url=os.getenv("AGENT_DATABASE_URL", "sqlite:///data/trading_agent.db"),
        paper=PaperTradingSettings(
            starting_balance=_env_float("AGENT_STARTING_BALANCE", 10_000.0),
        ),
    )
