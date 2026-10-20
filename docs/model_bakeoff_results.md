# Model Bake-off Results

- Generated: `2026-09-04T14:13:56.405628+00:00`
- Dataset: `data\labelled_signals.csv` (185 traded signals with a known outcome)
- Population win rate: **34.1%** / loss rate **65.9%**

## Result

**Winner: none selected**

No candidate model could be evaluated: none was available on this machine. The control arm (baseline-no-veto) ran, so the harness itself is verified, but no model was scored.

## Scores

| Model | Kind | Available | Calls | Schema valid (1st try) | Mean latency | p95 | USD / 1000 signals | Veto rate | Veto precision | Veto edge |
|---|---|---|---|---|---|---|---|---|---|---|
| `claude-haiku-4.5` | hosted | no - ANTHROPIC_API_KEY is not set | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `gpt-5-mini` | hosted | no - OPENAI_API_KEY is not set | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `ollama/qwen3:1.7b` | local | no - Ollama not reachable at http://localhost:11434 (<urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `baseline-no-veto` | baseline | yes | 185 | 100.0% | 0.00s | 0.00s | $0.0000 | 0.0% | n/a | n/a |

## How to read this

- **Schema valid (1st try)** - responses that parsed and passed strict
  validation without a retry. Below ~90% a model is not usable here.
- **Veto precision** - of the signals it vetoed, the share that actually
  lost. Compare against the population loss rate of 65.9%: that is
  what vetoing at random would score.
- **Veto edge** - precision minus that base rate. Zero means random.
  Negative means the model removed winners more often than losers.
- **`baseline-no-veto`** is the control: the rule-based system with no LLM.
  A candidate that does not beat it is not worth its latency or price.

## What this does NOT measure

- **Portfolio expectancy under each veto set.** Skipping a trade frees
  capital and changes which other trades can open, so a true comparison
  needs a full backtest re-run per candidate. This report deliberately
  does not print a portfolio number it did not compute.
- **News-driven vetoes.** The prompt carries no news context, because the
  news layer is roadmap Phase 6 and does not exist yet. The one veto
  reason most likely to be genuinely valuable is therefore untested here.
