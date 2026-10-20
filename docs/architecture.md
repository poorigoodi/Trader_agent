# Architecture

AI Crypto Trading Agent — paper trading only.
Target venues: **LBank / Toobit, USDT-M futures**. Default data source: **Toobit**.

---

## 1. The shape of the system

One process. One thread. Ordered function calls. No message queue, no
services, no agent framework.

That is a deliberate decision, not a stage we have not reached yet. A
multi-agent architecture would make the behaviour of a system that is
supposed to be auditable substantially harder to audit, and it buys nothing
while everything still fits in one loop. Splitting into real agents is a
later phase, and only after the single-process version has proven itself.

```
                          ┌──────────────────────────────────┐
                          │        main.py / orchestrator    │
                          │   (single process, one loop)     │
                          └───────────────┬──────────────────┘
                                          │  every N seconds, per symbol
                                          ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  1. MARKET DATA          app/market/data_feed.py                      │
 │     ccxt, PUBLIC endpoints only, no credentials ever                  │
 │     • verify_symbols() asks the venue what it actually lists          │
 │     • drops the still-forming candle                                  │
 │     • retry with exponential backoff                                  │
 │     • tags every frame: exchange_id, market_type, is_dev_data_source  │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  2. INDICATORS           app/indicators/  +  app/market_structure/    │
 │     EMA, SMA, RSI, ATR, MACD, Bollinger, VWAP — all in-house          │
 │     swing highs/lows → trend UP / DOWN / RANGE / UNKNOWN              │
 │     → freezes ONE IndicatorSnapshot, which is what gets logged        │
 │     DETERMINISTIC                                                     │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  3. STRATEGY             app/strategies/ema_rsi_trend.py              │
 │     three gates, all must agree:                                      │
 │       A. fresh EMA(20)/EMA(50) cross    B. RSI(14) window             │
 │       C. market structure confirms                                    │
 │     → LONG | SHORT | NO_TRADE  + reasons + invalidation condition     │
 │     DETERMINISTIC — no model, no fitted parameters                    │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  4. LLM REVIEW (optional, OFF by default)   app/ai_decision/          │
 │     ┌───────────────────────────────────────────────────────────┐     │
 │     │  MAY:  add a written explanation                          │     │
 │     │  MAY:  downgrade LONG/SHORT → NO_TRADE   (one-way veto)   │     │
 │     │  CANNOT: flip direction, create a trade from NO_TRADE,    │     │
 │     │          set size, set stop loss, set take profit         │     │
 │     └───────────────────────────────────────────────────────────┘     │
 │     strict JSON schema; any violation → fall back to the rule signal  │
 │     NON-DETERMINISTIC, and therefore structurally powerless           │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  5. RISK                 app/risk/                                    │
 │     levels.py       ATR stop, fixed-R target, R:R                     │
 │     position_sizing quantity = (balance × risk%) / stop distance      │
 │     manager.py      daily risk ledger (resets 00:00 UTC)              │
 │     correlation.py  rolling correlation → clusters → per-cluster cap  │
 │     DETERMINISTIC — closed-form arithmetic, never a model             │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  6. PAPER TRADING        app/paper_trading/                           │
 │     virtual wallet, taker fees + slippage against us on both sides,   │
 │     stop/target resolution (ambiguous candle ⇒ STOP), PnL, R multiple │
 │     NO EXCHANGE CLIENT. NO ORDER PATH. Cannot become live by a flag.  │
 └───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  7. DATABASE             app/database/  (SQLite via SQLAlchemy)       │
 │     candles · signals (INCLUDING every NO_TRADE) · paper_trades · runs│
 │     every row carries exchange_id + is_dev_data_source                │
 └───────────────────────────────────────────────────────────────────────┘
```

### Offline path (not in the live loop)

```
  KuCoin public history ──► app/backtesting/data_source.py
                                     │  (separate vendor — see §5)
                                     ▼
                            app/backtesting/engine.py
                            replays candles causally
                                     │
                                     ▼
                      data/labelled_signals.csv   (signal + did it win?)
                                     │
              ┌──────────────────────┴────────────────────────┐
              ▼                                               ▼
   evaluation/model_bakeoff.py                    app/confidence/calibration.py
   scores candidate models on                     logistic / GBM, TEMPORAL split
   real signals with real outcomes                → calibrated confidence score
```

---

## 2. Where the LLM sits, and why it sits there

The LLM layer is between **strategy** and **risk**. That position is chosen
so the layer is structurally unable to do damage:

- It runs **after** the direction is decided, so it cannot choose one.
- It runs **before** risk sizing, so it has never seen a position size, a
  stop, or a balance — those numbers do not exist yet when it is called.
- Its only output that affects anything is a single enum, `CONFIRM` or
  `VETO`, and `apply_verdict()` has exactly two possible return values:
  the direction it was given, or `NO_TRADE`.

```python
def apply_verdict(direction, verdict):
    if direction is NO_TRADE:      return NO_TRADE   # cannot resurrect
    if verdict and verdict.action is VETO: return NO_TRADE
    return direction                                  # unchanged
```

There is no third branch. A model that emits `{"action": "LONG"}` fails
validation; a model that emits `{"action": "VETO", "position_size": 5}` fails
validation on the unexpected field. Failure of any kind — malformed JSON,
timeout, provider outage, unexpected exception — falls back to the
rule-based signal unchanged and records why.

`tests/test_llm_layer.py` asserts this over the full cross-product of
directions and actions, so the guarantee is enforced by test, not by comment.

**The system's default state is with this layer switched off.** The
rule-based agent is the agent; the LLM is an optional filter.

---

## 3. Gate timing (why signals are rare)

The three strategy gates have different confirmation lags, and they interact:

| Gate | Confirms in | Note |
|---|---|---|
| EMA cross | immediately | but is only "fresh" for `cross_max_age` (default 3) candles |
| RSI window | immediately | |
| Market structure | ~10–15 candles | a fractal swing needs `structure_swing_window` (3) confirming bars on **each** side, and two swings are needed to compare |

After a trend reversal, structure cannot print a confirmed `UP` until well
after a 3-candle cross window has expired. On perfectly regular synthetic
data the two therefore never coincide. On real data they do, because real
swings are irregular — measured on Toobit 15m, ~1000 candles × 3 symbols:

| `cross_max_age` | actionable signals |
|---|---|
| 3 (default) | 0.34% |
| 6 | 0.80% |
| 10 | 1.36% |
| 25 | 2.54% |

The default is kept at 3. It is very selective, and that is the intent — but
it is a real constraint on how much data the system can generate for
evaluation, which is why the backtest window has to be long.

---

## 4. What is deterministic, and what is not

| Component | Deterministic? | Why it matters |
|---|---|---|
| Market data ingestion | yes | same candles in, same frame out |
| Indicators | yes | in-house, unit-tested against published reference values |
| Market structure | yes | fractal rules, confirmed swings only |
| Strategy | yes | pure function of the candle window |
| **LLM review** | **no** | which is exactly why it can only subtract |
| Position sizing | yes | closed-form arithmetic |
| Stop / target / R:R | yes | ATR multiple, fixed R |
| Daily risk ledger | yes | |
| Correlation clustering | yes | Pearson correlation + single linkage |
| Paper fills | yes | fixed fee and slippage model |
| Confidence calibration | trained, but **frozen at inference** | a fitted model, not a generative one; inspectable coefficients |

The rule the project is built around: **no non-deterministic component ever
sets a number that touches money.**

---

## 5. Data sources, and the basis risk between them

Three distinct roles, never conflated:

| Role | Source | Why |
|---|---|---|
| **Execution target** | Toobit USDT-M perps | verified: all 5 pairs listed, full 15m history |
| **Live data** | Toobit (same venue) | no basis risk between study and execution |
| **Backtest history** | **KuCoin spot** | Toobit ignores `since` (max ~1000 candles ≈ 10 days) |

Measured 2026-09-04:

- **Toobit** — `load_markets()` OK, 1308 markets / 756 linear swaps. All five
  target pairs present as USDT-M perps. `fetch_ohlcv` ignores `since`.
- **LBank** — ccxt's driver always calls `lbkperp.lbank.com` during
  `load_markets()`; that host is unreachable from this network. Spot OHLCV
  works intermittently. Its futures side is unusable here.
- **Binance** — `api.binance.com` unreachable from this network entirely.
- **KuCoin / Gate / HTX** — reachable and honour `since`. KuCoin chosen:
  all five bases listed, 180 days of 15m candles page cleanly.

**KuCoin spot is not Toobit perpetual futures.** Different books, different
participants, and perps carry funding. A backtest on this data measures the
*strategy*, not the fills the target venue would have given.
`HistoricalDataSource.measure_basis()` quantifies the gap; it is reported,
never assumed to be zero. Every frame and every database row records the
venue it came from.

---

## 6. Correlation risk

With BTC and ETH only, this was a placeholder — the position cap was the real
limit. With five large-cap perps it is not: they all carry high beta to
crypto as a whole, so four simultaneous longs is closer to one 4% bet on a
single factor than to four independent 1% bets. The daily risk cap will not
notice, because each trade looks small on its own.

```
log returns per symbol
        │
        ▼
rolling Pearson correlation matrix        (lookback 96 candles = 1 day)
        │
        ▼
single-linkage clustering at |ρ| ≥ 0.70   (transitive: A~B, B~C ⇒ {A,B,C})
        │
        ▼
net SIGNED risk per cluster ≤ 2% of balance
```

Three design choices worth naming:

- **Signed exposure.** Long BTC + short ETH is a spread, not a doubled bet,
  so they net against each other rather than adding.
- **Single linkage.** Conservative on purpose: within one asset class, an
  indirect relationship usually reasserts itself in a drawdown.
- **NaN fails safe.** An unmeasurable correlation is treated as *correlated*.
  "We could not measure it" is not evidence of independence.

The gate trims rather than refuses where it can: a trade that does not fit
whole is admitted at the size that does, and the quantity is re-derived so
the logged risk and the actual risk cannot disagree.

`max_open_positions` went from 2 to **4** (not 5) when the universe grew —
the correlation cap does the real work, and this stays a blunt backstop.

---

## 7. What is deliberately NOT here

| Not built | Roadmap phase |
|---|---|
| Live order placement, exchange credentials, account endpoints | 8–9 |
| News collection (RSS, CryptoPanic) | 6 |
| Sentiment in the live loop | 6–7 |
| Multi-agent architecture, message queue | 10 |
| WebSocket streaming (polling is enough at 15m) | later |
| Alembic migrations | when the schema changes during a live run |
| Funding-rate and open-interest data | later |

There is no code path from this repository to a real order. Not a disabled
one, not a flagged one — none. Making the system trade would require writing
a new execution layer, which is the point.
