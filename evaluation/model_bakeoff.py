"""Bake-off: score candidate models on the SAME fixed set of real signals.

The four metrics, and why each one is here
------------------------------------------
1. **Schema validity** -- what fraction of responses parsed and passed the
   strict validator on the FIRST attempt. The layer's output must be valid
   JSON with no extra fields; a model that needs retries costs latency on
   every call. This is the metric small local models most often fail.

2. **Latency** -- mean and p95 seconds per call. The agent evaluates a candle
   every 15 minutes across 5 symbols, so a 20-second model is usable and a
   200-second one is not.

3. **Cost** -- USD per 1000 signals at the provider's published rate. Zero for
   local models, which is itself a decisive property for something meant to
   run continuously.

4. **Veto alignment** -- of the signals a model vetoed, what fraction actually
   LOST in the backtest. This is the only metric that says whether the veto is
   worth having. A model that vetoes at random will score close to the base
   loss rate; a model that is genuinely reading risk will score above it.
   ``veto_edge`` reports the difference, and a candidate whose edge is not
   clearly positive is removing good trades along with bad ones.

The control arm
---------------
``baseline-no-veto`` never vetoes. It is what the system scores with no LLM at
all. Any candidate that does not beat it on realised expectancy is not worth
its latency, let alone its price. Without this arm the other numbers cannot be
interpreted.

Statistical honesty
-------------------
With a small number of traded signals, veto alignment has very low power.
The report prints the sample size next to every rate and refuses to declare a
winner on veto quality when the sample is too small to support one.

Usage::

    python -m evaluation.model_bakeoff --dataset data/labelled_signals.csv
    python -m evaluation.model_bakeoff --limit 50 --out docs/model_bakeoff_results.md
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from app.ai_decision.llm_layer import LLMDecisionLayer
from app.ai_decision.providers import Provider, build_candidates
from app.ai_decision.schema import LLMAction
from app.indicators.engine import IndicatorSnapshot
from app.strategies.base import Signal, SignalDirection

logger = logging.getLogger(__name__)

# Below this many traded signals, veto alignment is noise, not evidence.
MIN_SAMPLE_FOR_VETO_CLAIM = 30


@dataclass
class CandidateScore:
    """One model's results over the fixed signal set."""

    name: str
    kind: str
    model_id: str
    available: bool
    unavailable_reason: str = ""

    calls: int = 0
    valid_first_try: int = 0
    valid_eventually: int = 0
    failures: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)

    latencies: list[float] = field(default_factory=list)
    total_cost_usd: float = 0.0

    vetoed: int = 0
    vetoed_that_lost: int = 0
    vetoed_that_won: int = 0
    confirmed: int = 0
    confirmed_that_lost: int = 0
    confirmed_that_won: int = 0

    # -- derived ----------------------------------------------------------

    @property
    def validity_rate(self) -> float | None:
        return self.valid_first_try / self.calls if self.calls else None

    @property
    def mean_latency(self) -> float | None:
        return statistics.mean(self.latencies) if self.latencies else None

    @property
    def p95_latency(self) -> float | None:
        if len(self.latencies) < 2:
            return self.latencies[0] if self.latencies else None
        ordered = sorted(self.latencies)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    @property
    def cost_per_1000_signals(self) -> float | None:
        return (self.total_cost_usd / self.calls) * 1000 if self.calls else None

    @property
    def veto_rate(self) -> float | None:
        decided = self.vetoed + self.confirmed
        return self.vetoed / decided if decided else None

    @property
    def veto_precision(self) -> float | None:
        """Of what it vetoed, how much actually lost."""
        labelled = self.vetoed_that_lost + self.vetoed_that_won
        return self.vetoed_that_lost / labelled if labelled else None

    @property
    def labelled_vetoes(self) -> int:
        return self.vetoed_that_lost + self.vetoed_that_won

    def veto_edge(self, base_loss_rate: float | None) -> float | None:
        """veto_precision minus the loss rate of the whole population.

        Zero means the model is vetoing at random. Positive means it is
        preferentially removing losers.
        """
        precision = self.veto_precision
        if precision is None or base_loss_rate is None:
            return None
        return precision - base_loss_rate

    def to_dict(self, base_loss_rate: float | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "model_id": self.model_id,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "calls": self.calls,
            "validity_rate": self.validity_rate,
            "valid_first_try": self.valid_first_try,
            "failures": self.failures,
            "failure_reasons": dict(self.failure_reasons),
            "mean_latency_s": self.mean_latency,
            "p95_latency_s": self.p95_latency,
            "cost_per_1000_signals_usd": self.cost_per_1000_signals,
            "veto_rate": self.veto_rate,
            "vetoed": self.vetoed,
            "labelled_vetoes": self.labelled_vetoes,
            "veto_precision": self.veto_precision,
            "veto_edge": self.veto_edge(base_loss_rate),
        }


def signal_from_row(row: pd.Series) -> Signal:
    """Rebuild a Signal from a labelled backtest row.

    Only fields that existed at decision time are restored. The outcome
    columns are deliberately NOT passed to the model -- handing it the answer
    would make every score meaningless.
    """
    snapshot = IndicatorSnapshot(
        symbol=str(row["symbol"]),
        timeframe="15m",
        timestamp=pd.Timestamp(row["timestamp"]),
        open=float(row["close"]),
        high=float(row["close"]),
        low=float(row["close"]),
        close=float(row["close"]),
        volume=0.0,
        ema_fast=_opt(row.get("ema_fast")),
        ema_slow=_opt(row.get("ema_slow")),
        ema_fast_prev=None,
        ema_slow_prev=None,
        sma_slow=None,
        rsi=_opt(row.get("rsi")),
        atr=_opt(row.get("atr")),
        macd=None,
        macd_signal=None,
        macd_hist=_opt(row.get("macd_hist")),
        bb_upper=None,
        bb_middle=None,
        bb_lower=None,
        bb_percent_b=_opt(row.get("bb_percent_b")),
        vwap=None,
        trend=str(row.get("trend", "UNKNOWN")),
    )
    return Signal(
        direction=SignalDirection(row["direction"]),
        reasons=tuple(str(row.get("reason", "")).split("; ")),
        invalidation=str(row.get("invalidation", "")),
        snapshot=snapshot,
        strategy_name=str(row.get("strategy", "ema_rsi_trend_v1")),
        metadata={
            "cross_age": _opt(row.get("cross_age")),
            "atr_pct_of_price": _opt(row.get("atr_pct_of_price")),
            "ema_spread_pct": _opt(row.get("ema_spread_pct")),
        },
    )


def _opt(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def load_signals(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Load only TRADED signals: those are the ones with a real outcome."""
    frame = pd.read_csv(path)
    frame = frame[frame["won"].notna()].copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if limit:
        frame = frame.head(limit)
    return frame


def score_candidate(
    provider: Provider, signals: pd.DataFrame, max_attempts: int = 2
) -> CandidateScore:
    """Run one provider over every signal and tally the four metrics."""
    available, why = provider.available()
    score = CandidateScore(
        name=provider.info.name,
        kind=provider.info.kind,
        model_id=provider.info.model_id,
        available=available,
        unavailable_reason=why,
    )
    if not available:
        logger.warning("skipping %s: %s", provider.info.name, why)
        return score

    layer = LLMDecisionLayer(provider, max_attempts=max_attempts)

    for _, row in signals.iterrows():
        signal = signal_from_row(row)
        review = layer.review(signal)
        score.calls += 1

        if review.ok and review.verdict is not None:
            score.valid_eventually += 1
            if review.attempts == 1:
                score.valid_first_try += 1
            score.latencies.append(review.latency_seconds)
            score.total_cost_usd += review.cost_usd

            won = int(row["won"]) == 1
            if review.verdict.action is LLMAction.VETO:
                score.vetoed += 1
                score.vetoed_that_won += int(won)
                score.vetoed_that_lost += int(not won)
            else:
                score.confirmed += 1
                score.confirmed_that_won += int(won)
                score.confirmed_that_lost += int(not won)
        else:
            score.failures += 1
            reason = (review.failure_reason or "unknown").split(":")[0]
            score.failure_reasons[reason] = score.failure_reasons.get(reason, 0) + 1

    return score


def realised_expectancy(signals: pd.DataFrame, score: CandidateScore) -> float | None:
    """Not computed per-candidate here; see the note in the report.

    Recomputing portfolio expectancy under a veto set requires re-running the
    backtest with those trades removed, because skipping a trade frees
    capital and changes what else could open. The bake-off reports veto
    precision instead, which is measurable without that re-run, and the
    report says so rather than implying a portfolio number it did not compute.
    """
    return None


def run_bakeoff(
    signals: pd.DataFrame,
    providers: Sequence[Provider] | None = None,
    max_attempts: int = 2,
) -> tuple[list[CandidateScore], dict[str, Any]]:
    providers = list(providers or build_candidates())
    base_loss_rate = float((signals["won"] == 0).mean()) if len(signals) else None

    scores: list[CandidateScore] = []
    for provider in providers:
        logger.info("scoring %s ...", provider.info.name)
        started = time.perf_counter()
        score = score_candidate(provider, signals, max_attempts)
        logger.info(
            "  %s: %d calls in %.1fs", provider.info.name, score.calls,
            time.perf_counter() - started,
        )
        scores.append(score)

    context = {
        "signals_scored": int(len(signals)),
        "base_win_rate": float((signals["won"] == 1).mean()) if len(signals) else None,
        "base_loss_rate": base_loss_rate,
        "sample_supports_veto_claim": len(signals) >= MIN_SAMPLE_FOR_VETO_CLAIM,
        "min_sample_for_veto_claim": MIN_SAMPLE_FOR_VETO_CLAIM,
    }
    return scores, context


def pick_winner(
    scores: Sequence[CandidateScore], context: dict[str, Any]
) -> tuple[str | None, str]:
    """Choose a winner, or explain honestly why none can be chosen."""
    ran = [s for s in scores if s.available and s.calls > 0 and s.kind != "baseline"]
    if not ran:
        return None, (
            "No candidate model could be evaluated: none was available on this "
            "machine. The control arm (baseline-no-veto) ran, so the harness "
            "itself is verified, but no model was scored."
        )

    usable = [s for s in ran if (s.validity_rate or 0) >= 0.90]
    if not usable:
        best = max(ran, key=lambda s: s.validity_rate or 0)
        return None, (
            f"No candidate reached 90% first-try schema validity. Best was "
            f"{best.name} at {(best.validity_rate or 0):.1%}. A model that "
            f"cannot reliably emit the required JSON is not usable here, "
            f"whatever else it scores."
        )

    if not context.get("sample_supports_veto_claim"):
        cheapest = min(usable, key=lambda s: (s.cost_per_1000_signals or 0, s.mean_latency or 0))
        return cheapest.name, (
            f"Chosen on schema validity, latency and cost only. The signal "
            f"sample ({context['signals_scored']} traded signals) is below the "
            f"{MIN_SAMPLE_FOR_VETO_CLAIM} needed to say anything meaningful "
            f"about veto quality, so veto alignment did NOT influence this "
            f"choice and should not be read as validated."
        )

    base = context.get("base_loss_rate")
    with_edge = [s for s in usable if (s.veto_edge(base) or 0) > 0]
    if not with_edge:
        return None, (
            "Every usable candidate vetoed at or below the population loss "
            "rate, i.e. no better than chance. On this evidence the LLM veto "
            "adds nothing and the baseline (no LLM) is the correct choice."
        )

    winner = max(with_edge, key=lambda s: (s.veto_edge(base) or 0))
    return winner.name, (
        f"{winner.name} had the highest veto edge "
        f"({(winner.veto_edge(base) or 0):+.1%} over the {base:.1%} base loss "
        f"rate) among candidates meeting the validity bar."
    )


def render_markdown(
    scores: Sequence[CandidateScore], context: dict[str, Any], dataset: Path
) -> str:
    base = context.get("base_loss_rate")
    winner, rationale = pick_winner(scores, context)

    lines = [
        "# Model Bake-off Results",
        "",
        f"- Generated: `{pd.Timestamp.now(tz='UTC').isoformat()}`",
        f"- Dataset: `{dataset}` ({context['signals_scored']} traded signals "
        f"with a known outcome)",
        f"- Population win rate: **{_pct(context.get('base_win_rate'))}** "
        f"/ loss rate **{_pct(base)}**",
        "",
        "## Result",
        "",
        f"**Winner: {winner or 'none selected'}**",
        "",
        rationale,
        "",
        "## Scores",
        "",
        "| Model | Kind | Available | Calls | Schema valid (1st try) | "
        "Mean latency | p95 | USD / 1000 signals | Veto rate | Veto precision | Veto edge |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for score in scores:
        lines.append(
            "| `{name}` | {kind} | {avail} | {calls} | {validity} | {mean} | {p95} | "
            "{cost} | {vrate} | {vprec} | {vedge} |".format(
                name=score.name,
                kind=score.kind,
                avail="yes" if score.available else f"no - {score.unavailable_reason}",
                calls=score.calls,
                validity=_pct(score.validity_rate),
                mean=_secs(score.mean_latency),
                p95=_secs(score.p95_latency),
                cost=(
                    "n/a"
                    if score.cost_per_1000_signals is None
                    else f"${score.cost_per_1000_signals:.4f}"
                ),
                vrate=_pct(score.veto_rate),
                vprec=(
                    f"{_pct(score.veto_precision)} (n={score.labelled_vetoes})"
                    if score.veto_precision is not None
                    else "n/a"
                ),
                vedge=_signed_pct(score.veto_edge(base)),
            )
        )

    lines += [
        "",
        "## How to read this",
        "",
        "- **Schema valid (1st try)** - responses that parsed and passed strict",
        "  validation without a retry. Below ~90% a model is not usable here.",
        "- **Veto precision** - of the signals it vetoed, the share that actually",
        f"  lost. Compare against the population loss rate of {_pct(base)}: that is",
        "  what vetoing at random would score.",
        "- **Veto edge** - precision minus that base rate. Zero means random.",
        "  Negative means the model removed winners more often than losers.",
        "- **`baseline-no-veto`** is the control: the rule-based system with no LLM.",
        "  A candidate that does not beat it is not worth its latency or price.",
        "",
        "## What this does NOT measure",
        "",
        "- **Portfolio expectancy under each veto set.** Skipping a trade frees",
        "  capital and changes which other trades can open, so a true comparison",
        "  needs a full backtest re-run per candidate. This report deliberately",
        "  does not print a portfolio number it did not compute.",
        "- **News-driven vetoes.** The prompt carries no news context, because the",
        "  news layer is roadmap Phase 6 and does not exist yet. The one veto",
        "  reason most likely to be genuinely valuable is therefore untested here.",
        "",
    ]

    if not context.get("sample_supports_veto_claim"):
        lines += [
            "> **Sample size warning.** "
            f"{context['signals_scored']} traded signals is below the "
            f"{MIN_SAMPLE_FOR_VETO_CLAIM} this report requires before treating "
            "veto alignment as evidence. Validity, latency and cost are still "
            "meaningful; veto quality is not.",
            "",
        ]

    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _signed_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score candidate models on real signals.")
    parser.add_argument("--dataset", type=Path, default=Path("data/labelled_signals.csv"))
    parser.add_argument("--limit", type=int, default=None, help="cap signals scored")
    parser.add_argument("--out", type=Path, default=Path("docs/model_bakeoff_results.md"))
    parser.add_argument("--json-out", type=Path, default=Path("data/bakeoff_scores.json"))
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if not args.dataset.exists():
        logger.error(
            "dataset %s not found. Generate it first with:\n"
            "  python -m evaluation.run_backtest",
            args.dataset,
        )
        return 1

    signals = load_signals(args.dataset, args.limit)
    if signals.empty:
        logger.error("no traded signals with outcomes in %s", args.dataset)
        return 1

    logger.info("scoring %d signals", len(signals))
    scores, context = run_bakeoff(signals, max_attempts=args.max_attempts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(scores, context, args.dataset), encoding="utf-8")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {
                "context": context,
                "scores": [s.to_dict(context.get("base_loss_rate")) for s in scores],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    winner, rationale = pick_winner(scores, context)
    print(f"\nWinner: {winner or 'none selected'}\n{rationale}\n")
    print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
