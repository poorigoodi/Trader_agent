# Claude Code Kickoff Prompt (English)

> Paste this directly into the Claude Code terminal (inside the empty project folder) to start actual
> coding. Assumes `03_roadmap.md` is available alongside the project — it's the source of truth for
> task ordering.

---

## Role & Context

You're working on an **AI Crypto Trading Agent**. The project has a full 11-phase roadmap
(read `roadmap.md` — it's the reference for task order), but at this stage we only want to implement
**Phase 0, Phase 1, and the initial MVP skeleton**. There is no real trading, no real API keys, and no
connection to real funds at this stage.

Target exchanges (final goal): **LBank** and **Toobit** (both are supported by the `ccxt` library —
use it as the base for the exchange abstraction layer, not a hand-rolled REST client).

## Hard Constraints (mandatory)

1. **No multi-agent framework.** A single-process orchestrator with separate Python modules is enough
   for now (no LangGraph, no message queue, no separate services). This is a deliberate architectural
   decision to avoid over-engineering.
2. **No LLM at this stage.** The initial signal must be 100% rule-based (technical indicators only).
   The LLM/news/sentiment layer is added in a later phase.
3. **No real API keys and no code path for real order placement.** Only read public market data
   (candles/ticker/order book) via ccxt. Even the folder structure should not "pre-wire" a live order
   path — execution is a fully separate, later layer.
4. **Everything is paper trading.** Balance, positions, and PnL are all simulated and stored in a
   local database (SQLite to start).
5. **Determinism in risk/execution.** No decision about position size or SL/TP should ever be
   delegated to a language model; these must be explicit math/code.
6. **Every component needs unit tests**, especially indicator calculations and position sizing.
7. **Full logging.** Every generated signal must be logged with all of its inputs (price, indicator
   values, timestamp) in the database so it can be reviewed later.

## What I want you to do now (in this order)

1. Scaffold the Python project (per the "Project Structure" section in `roadmap.md`; if that file
   isn't available, use the structure below):
   ```
   app/
   ├── market/          # ccxt connection, OHLCV/ticker fetching
   ├── indicators/       # RSI, EMA, SMA, MACD, ATR, Bollinger, VWAP
   ├── strategies/        # rule-based signal generation logic
   ├── risk/               # position sizing, SL/TP, R:R (deterministic)
   ├── paper_trading/       # virtual wallet, order simulation, PnL
   ├── database/             # SQLite models and migrations
   ├── config/                 # settings (symbols, timeframe, risk params)
   └── tests/
   requirements.txt
   README.md
   ```
2. Set up a virtual environment via `pip` and add `ccxt`, `pandas`, `pandas-ta` (or an equivalent),
   `pytest`, and `sqlalchemy` to `requirements.txt`.
3. Write a `market/data_feed.py` module that uses ccxt to pull historical and live candles for
   **LBank** (symbols BTC/USDT and ETH/USDT, 15-minute timeframe). If connecting to LBank is
   problematic in the dev environment, fall back to Binance public data for testing the logic — but
   clearly comment this in the code and README (the distinction between the dev data source and the
   final target exchange must always be explicit).
4. Implement EMA(20/50), RSI(14), ATR(14), MACD, and Bollinger Bands in `indicators/` — each with a
   unit test against sample data.
5. Build a simple, transparent strategy in `strategies/` (e.g. EMA cross + RSI filter + basic market
   structure trend confirmation) that outputs `LONG` / `SHORT` / `NO_TRADE` along with a reason.
6. In `risk/`, write a deterministic module for position sizing (fixed % risk of virtual balance,
   distance to stop loss, and a max daily risk cap) plus SL/TP/R:R calculation.
7. In `paper_trading/`, build a simple engine: a virtual wallet, simulated order open/close with
   configurable fees and slippage (default conservative, given LBank/Toobit's wider spreads), and PnL
   tracking.
8. Log everything to SQLite (`database/`): every candle evaluated, every signal (with all indicator
   values at decision time), and every paper trade.
9. Build a `main.py` script that runs this cycle every few minutes (configurable) for BTC/USDT and
   ETH/USDT: fetch data → compute indicators → generate signal → (if signaled) size the position →
   log the paper trade.
10. Run `pytest` and make sure everything passes. Include a summary of an actual run (how many
    candles, how many signals, how many paper trades) in the README.

## Definition of Done for this stage

- The code runs and completes several cycles on real/sample data without errors.
- No private API keys or real order-placement code paths exist anywhere in the codebase.
- All risk/indicator calculations have unit tests, and they pass.
- Every signal's output includes a readable reason and an invalidation condition, even in rule-based
  mode.
- The README explains how to run it, what is *deliberately* out of scope at this stage (LLM, news,
  live execution), and what the next phase is per `roadmap.md`.

Whenever you need to make an architectural decision that isn't specified above, don't guess — write it
down in the README as an "Open Decision" so we can review it together, especially anything that gets
close to real API keys or live execution.
