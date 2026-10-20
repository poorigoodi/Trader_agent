# AI Crypto Trading Agent

A rule-based crypto trading agent with an optional LLM review layer.
**Paper trading only.** There are no exchange credentials in this repository
and no code path that can place a real order.

Target venues: **LBank / Toobit, USDT-M futures**.
Default data source: **Toobit** (chosen by measurement — see [Data sources](#data-sources)).

---

## Status

| | |
|---|---|
| Tests | **352 passing** |
| Symbols | BTC, ETH, BNB, XRP, SOL — all verified listed as Toobit USDT-M perps |
| Live loop | working — 5 symbols, 15m candles, 0 errors over multi-cycle runs |
| Backtest | 180 days × 5 symbols, 17,160 candles, 185 trades |
| **Backtest verdict** | **the strategy has no edge — see [Results](#results)** |
| LLM layer | built, tested, **off by default**; no model evaluated yet (none installed) |
| Confidence calibration | trained; **does not beat the base-rate baseline** |

---

## Install and run

Requires Python 3.11+ (developed and tested on 3.14).

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Run the paper-trading loop:

```bash
.venv/Scripts/python.exe main.py --cycles 2 --interval 60
```

Run it continuously (5-minute cycles, Ctrl-C to stop):

```bash
.venv/Scripts/python.exe main.py
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--cycles N` | stop after N cycles (omit to run until interrupted) |
| `--interval S` | seconds between cycles (default 300) |
| `--exchange {toobit,lbank,binance}` | data source |
| `--symbols "BTC/USDT:USDT,ETH/USDT:USDT"` | override the universe |
| `--balance 10000` | starting virtual balance |
| `--close-at-end` | mark open positions to market when the run ends |

Run the tests:

```bash
.venv/Scripts/python.exe -m pytest
```

### Offline analysis

Generate the labelled dataset (fetches history, replays it, writes
`data/labelled_signals.csv`). Takes ~11 minutes for 180 days:

```bash
.venv/Scripts/python.exe -m evaluation.run_backtest --days 180
```

Train the confidence calibrator (temporal split, never random):

```bash
.venv/Scripts/python.exe -m evaluation.train_confidence --compare
```

Score candidate LLMs on real signals with real outcomes:

```bash
.venv/Scripts/python.exe -m evaluation.model_bakeoff
```

Score sentiment classifiers (the separate track):

```bash
.venv/Scripts/python.exe -m evaluation.sentiment_bakeoff
```

---

## Results

### The strategy does not currently have an edge

180 days, 5 symbols, 17,160 candles evaluated, 185 trades taken:

| Metric | Value |
|---|---|
| Win rate | 34.05% |
| Expectancy | **−0.566 R** |
| Profit factor | 0.58 |
| Return | −20.4% |
| Max drawdown | 21.7% |

This is the headline finding of this phase, and it is worth being precise
about *why*, because the first version of these numbers was wrong for a
different reason.

**First problem (fixed): the cost model was mis-specified.** Slippage was set
from CoinGecko's whole-exchange average *spot* spreads (LBank ~0.44%, Toobit
~0.54%), giving a 0.70% round trip. Measured against the actual target market
— USDT-M perps on the five majors, walking the live order book for a 10,000
USDT fill — real slippage is:

| BTC | ETH | BNB | XRP | SOL |
|---|---|---|---|---|
| 0.00038% | 0.00813% | 0.00487% | 0.01763% | 0.00980% |

The assumption was too high by one to two orders of magnitude. It is now
0.05% per side (≈3× the worst measured value, as a buffer for fast markets),
with a 0.06% taker fee — a 0.22% round trip.

**Second problem (real, not fixed): there is no edge underneath.** With the
costs corrected, a sensitivity sweep gives:

| Scenario | Trades | Win rate | Expectancy |
|---|---|---|---|
| Legacy spot-spread costs (0.10% / 0.25%) | 75 | 10.7% | −2.22 R |
| Measured perp costs (0.06% / 0.05%) | 75 | 32.0% | −0.73 R |
| Measured costs, 3.0× ATR stop | 67 | 32.8% | −0.36 R |
| Measured costs, 5.0× ATR stop | 52 | 30.8% | −0.29 R |
| **Zero cost (impossible upper bound)** | 75 | 32.0% | **−0.04 R** |

At literally zero transaction cost the strategy is still slightly negative.
A 2:1 reward-to-risk setup needs a 33.3% win rate to break even; over 180
days it produced 34.05%, a gross edge of about +0.02 R that is then entirely
consumed by roughly 0.63 R of costs (the round trip is 0.22% against a stop
distance of only 0.30–0.45%).

**Conclusion: this strategy is not ready to progress.** The roadmap's own
gate — profit factor > 1 before paper trading counts for anything — is not
met, and nothing here should be read as a reason to move toward live trading.

**What was deliberately not done:** the parameters were not tuned until the
results looked good. Widening the stop improves expectancy in the table
above, and picking the best row would be curve-fitting to 180 days of one
market regime. The cost model was corrected because it was factually wrong
against measured data; the strategy was left alone because there is no
honest basis for changing it yet.

### Confidence calibration: no learnable signal yet

Trained on the 185 labelled signals, temporal split (70% oldest / 30% newest):

| Model | Test accuracy | Test AUC | Brier | Baseline Brier | Beats baseline? |
|---|---|---|---|---|---|
| Logistic regression | 55.4% | 0.561 | 0.2587 | 0.2302 | **no** |
| Gradient boosting | 60.7% | 0.605 | 0.2357 | 0.2302 | **no** |

Neither beats simply predicting the training base rate. AUC near 0.5 is
chance. With 185 rows this is the expected outcome and is reported as such
rather than dressed up — see `data/calibration_report.json`.

### Model bake-off: not yet run

The harness works end-to-end (the control arm scored 185 signals at 100%
schema validity), but no candidate model was available on this machine.
See [`docs/model_bakeoff_results.md`](docs/model_bakeoff_results.md).
To run it for real, install Ollama and pull a small model:

```bash
ollama pull qwen3:1.7b
```

then re-run `python -m evaluation.model_bakeoff`.

### Sentiment track (bake-off candidate #4)

Scored separately, because labelling a headline POSITIVE/NEGATIVE/NEUTRAL is
a different task from deciding CONFIRM/VETO on a signal. 60 hand-written
headlines, 19 of them marked *hard* (negation, mixed signals, sarcasm,
crypto-specific vocabulary):

| Classifier | Accuracy | **Hard subset** | Macro F1 | Latency |
|---|---|---|---|---|
| `finbert` (110M, local, free) | 63.3% | **47.4%** | 0.604 | 50.7 ms |
| `lexicon-baseline` (word list) | 63.3% | **26.3%** | 0.634 | 0.3 ms |
| `ollama/qwen3:1.7b` | not installed | — | — | — |

Majority-class baseline is 40.0%.

The headline number is a tie — and that is exactly why the hard subset
exists. On easy headlines a word list does as well as a transformer. On
negation and mixed signals FinBERT is **nearly twice as accurate**, which is
where a news veto would actually have to earn its place. The lexicon's
slightly higher macro-F1 comes from being better calibrated on the easy
NEUTRAL class, not from understanding anything.

Full detail: [`docs/sentiment_track_results.md`](docs/sentiment_track_results.md).
Requires the optional `transformers` + `torch` extras (see `requirements.txt`).

---

## Data sources

Three roles, never conflated:

| Role | Source | Why |
|---|---|---|
| Execution target | Toobit USDT-M perps | all 5 pairs verified listed |
| Live market data | Toobit | same venue ⇒ no basis risk |
| Backtest history | **KuCoin spot** | Toobit serves only ~1000 candles |

Measured 2026-09-04:

- **Toobit** — works fully; 1308 markets, 756 linear swaps; all five target
  pairs present with full 15m history. But `fetch_ohlcv` **ignores `since`**,
  capping history at ~1000 candles (≈10 days on 15m).
- **LBank** — ccxt's driver always hits `lbkperp.lbank.com` during
  `load_markets()`, and that host is unreachable from this network. Spot
  OHLCV works intermittently (BTC/ETH/BNB/XRP succeeded, SOL did not). Its
  futures side is unusable here. Still supported in code.
- **Binance** — `api.binance.com` unreachable from this network entirely, so
  the original plan to develop against Binance data was not possible.
- **KuCoin / Gate / HTX** — reachable and honour `since`. KuCoin is the
  backtest vendor: all five bases listed, 180 days page cleanly.

**KuCoin spot is not Toobit perps.** Different books, and perps carry
funding. Basis is measured by `HistoricalDataSource.measure_basis()` and
recorded, never assumed to be zero. Every frame and database row carries
`exchange_id` and `is_dev_data_source`.

---

## Safety model

**No exchange credentials exist anywhere in this repository.**

- `app/config/settings.py` has no field for an API key or secret, and
  `load_settings()` reads no such environment variable.
- `MarketDataFeed._assert_read_only()` raises on construction if `apiKey`,
  `secret`, `password`, `uid`, `privateKey` or `walletAddress` is populated
  on the ccxt client. A test asserts this for each attribute.
- There is no `create_order`, `cancel_order`, `fetch_balance`, or any account
  method anywhere in `app/`. A test asserts their absence structurally.
- `app/paper_trading/` imports no exchange client at all. It cannot become
  live by flipping a flag; making the system trade would require writing a
  new execution layer.

The LLM providers do read API keys from the environment at call time
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). Those are **model** keys, not
**exchange** keys — they cannot move funds, and they are never hard-coded,
committed, or written to the database. Local models need no key at all, which
is the configured default.

### What the LLM layer can and cannot do

| | |
|---|---|
| **May** | attach a written explanation |
| **May** | downgrade LONG or SHORT to NO_TRADE (**one-way veto**) |
| **Cannot** | flip LONG ↔ SHORT |
| **Cannot** | turn NO_TRADE into a trade |
| **Cannot** | set or influence position size, stop loss, or take profit |

This is enforced in code, not by prompt instruction: `apply_verdict()` has
exactly two possible return values — the direction it was given, or
`NO_TRADE`. Extra fields in a response (`position_size`, `stop_loss`, …) fail
strict validation rather than being ignored. Any failure — bad JSON, timeout,
outage, unexpected exception — falls back to the rule-based signal and
records why. **The layer is off by default.**

---

## Project layout

```
app/
  config/           settings — no credential fields, by design
  market/           ccxt public data, symbol verification, retry/backoff
  indicators/       EMA SMA RSI ATR MACD Bollinger VWAP (in-house)
  market_structure/ fractal swings → trend classification
  strategies/       rule-based signal generation
  ai_decision/      LLM layer: strict schema + one-way veto + providers
  risk/             levels, position sizing, daily ledger, correlation
  paper_trading/    virtual wallet, simulated fills, PnL
  backtesting/      historical vendor + causal replay engine
  confidence/       calibration model (temporal split)
  news/             sentiment classifiers (Phase 6 groundwork)
  database/         SQLAlchemy models + repository
  orchestrator.py   the single-process loop
evaluation/         bake-off, backtest runner, calibration trainer
tests/              352 tests, no network access
docs/               architecture + results
main.py
```

---

## Open Decisions

Things that need a human call, or that were decided on evidence that may go
stale. Please review these.

### OD-1 — Indicators are written in-house, not `pandas-ta` / TA-Lib

**Decided.** Every number the agent trades on is covered by our own unit
tests, seeding conventions are pinned so historical results reproduce, and
`pandas-ta` is unmaintained and breaks on numpy ≥ 2 / pandas ≥ 3 (this
project runs pandas 3.0.5 and numpy 2.5.2, where it would not have worked).
RSI is validated against Wilder's published worked example to ±0.01.

### OD-2 — Target market is USDT-M futures, so margin-style accounting is correct

**Decided, no change needed.** The virtual wallet uses a single cash balance
with symmetric LONG/SHORT treatment, which is right for USDT-M perps. If the
target ever moves to spot, short trades would need a different model.

### OD-3 — `Base.metadata.create_all`, not Alembic

**Decided for now.** Appropriate for a single-developer SQLite MVP. Alembic
becomes worth its weight once the schema changes while a paper run is live —
i.e. before any multi-week continuous run that must not be reset.

### OD-4 — Default venue switched from LBank to Toobit ⚠️ needs your review

The plan named LBank first. Measurement inverted that: LBank's ccxt driver
cannot complete `load_markets()` from this network, while Toobit works fully
including futures. **This may be a local network restriction rather than a
property of LBank.** If LBank works from where you intend to deploy, this
decision should be revisited — the code supports both.

### OD-5 — Backtest history comes from KuCoin spot ⚠️ needs your review

Toobit caps history at ~1000 candles and Binance is unreachable here, so
KuCoin spot is the vendor. This introduces real basis risk (spot vs perp, and
a different venue). It is measured and reported, but if you have access to a
proper data vendor or to Binance from your network, that would be strictly
better.

### OD-6 — The top-5 ranking will change

BTC, ETH, BNB, XRP, SOL were correct at the time of writing and are verified
listed on Toobit. Market-cap rankings move. Re-check against CoinGecko/CMC
before any extended run; `verify_symbols()` will drop anything delisted, but
it cannot tell you a *better* symbol now exists.

### OD-7 — LLM pricing may be stale ⚠️

Prices in `app/ai_decision/providers.py` (Claude Haiku 4.5 at $1/$5 per Mtok;
GPT-5 mini at $0.25/$2) are as published at the time of writing and are
printed back in the bake-off report so a stale figure is visible rather than
silently believed. Verify against each provider's current documentation
before choosing on cost.

### OD-8 — Toobit's fee schedule is an assumption, not a measurement ⚠️

Toobit does not expose fees through ccxt (`market['taker']` is `None`). The
0.06% taker fee is the standard USDT-M rate, assumed. Confirm it against
Toobit's published schedule — it feeds directly into every backtest number.

### OD-9 — `cross_max_age = 3` makes the strategy very selective

A fractal swing needs ~10–15 candles to confirm, so a 3-candle cross window
rarely coincides with a fresh structure read. Measured signal rates: 0.34% at
3, 1.36% at 10, 2.54% at 25. Kept at 3 (selective is intended), but it limits
how much evaluation data the system can produce. **Not tuned**, because
tuning it to produce more signals would be fitting to one regime.

### OD-10 — The strategy has no measured edge ⚠️ needs your decision

The most important open item. At zero cost the expectancy is −0.04 R. Options
as I see them: revise the entry rules, widen stops so costs are a smaller
fraction of R, move to a higher timeframe, or accept that this particular
rule set is a scaffold for the pipeline rather than a candidate for capital.
I have deliberately not chosen one.

### OD-11 — The sentiment dataset is hand-written

`evaluation/sentiment_dataset.py` contains 60 headlines I wrote, because the
news collector (roadmap Phase 6) does not exist and there is no feed to
sample. Scores from it are a sanity check, not a production estimate.
Replace with sampled, independently-labelled real headlines when Phase 6
lands.

---

## Deliberately out of scope

| Not built | Roadmap phase |
|---|---|
| Live order placement, exchange credentials, account endpoints | 8–9 |
| News collection (RSS, CryptoPanic) | 6 |
| Sentiment in the live decision loop | 6–7 |
| Multi-agent architecture, message queue | 10 |
| WebSocket streaming (polling is sufficient at 15m) | later |
| Alembic migrations | see OD-3 |
| Fine-tuning any LLM | not planned — see below |

**On fine-tuning.** Fine-tuning a language model was considered for the
confidence layer and rejected: it needs a large labelled corpus and a GPU,
and it invites exactly the overfitting this project's ground rules warn
about. The only trained component is a logistic regression / small
gradient-boosted tree over ~10 numeric features — trains in under a second on
a CPU, and its coefficients can be read and argued with. If a fine-tuned
local model for crypto-specific news classification is wanted later, that is
a separate decision requiring a much larger labelled dataset.

---

## Next steps (roadmap)

1. **Resolve OD-10** — decide what to do about the missing edge. Nothing
   downstream is worth much until this is settled.
2. Install Ollama and run the real bake-off (`ollama pull qwen3:1.7b`).
3. Phase 6 — news collection, which would give the LLM veto the one input
   most likely to make it genuinely valuable.
4. Re-run calibration once there are meaningfully more than 185 labelled
   signals.
5. Phase 8 — exchange integration, **only** after a strategy with a real edge
   exists and the roadmap's gate criteria are met.
