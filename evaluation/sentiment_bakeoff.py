"""Score sentiment classifiers -- the separate track for bake-off candidate #4.

    python -m evaluation.sentiment_bakeoff

Reports accuracy and macro-F1 overall, plus a separate score on the subset
marked ``hard`` (negation, mixed signals, sarcasm, crypto-specific
vocabulary). The hard subset is where a purpose-built classifier should
justify itself against a word list; overall accuracy on easy headlines mostly
measures how obvious the easy headlines are.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.news.sentiment import (
    Sentiment,
    SentimentClassifier,
    build_sentiment_candidates,
)
from evaluation.sentiment_dataset import HEADLINES, LabelledHeadline, label_counts

logger = logging.getLogger(__name__)

LABELS = [Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL]


@dataclass
class SentimentScore:
    name: str
    kind: str
    available: bool
    unavailable_reason: str = ""

    total: int = 0
    correct: int = 0
    hard_total: int = 0
    hard_correct: int = 0
    latencies: list[float] = field(default_factory=list)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None

    @property
    def hard_accuracy(self) -> float | None:
        return self.hard_correct / self.hard_total if self.hard_total else None

    @property
    def mean_latency(self) -> float | None:
        return sum(self.latencies) / len(self.latencies) if self.latencies else None

    def per_label(self) -> dict[str, dict[str, float]]:
        """Precision / recall / F1 for each class."""
        out: dict[str, dict[str, float]] = {}
        for label in LABELS:
            key = label.value
            tp = self.confusion.get(key, {}).get(key, 0)
            predicted = sum(
                self.confusion.get(actual, {}).get(key, 0) for actual in self.confusion
            )
            actual_total = sum(self.confusion.get(key, {}).values())
            precision = tp / predicted if predicted else 0.0
            recall = tp / actual_total if actual_total else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            out[key] = {"precision": precision, "recall": recall, "f1": f1,
                        "support": actual_total}
        return out

    @property
    def macro_f1(self) -> float | None:
        scores = self.per_label()
        if not scores:
            return None
        return sum(s["f1"] for s in scores.values()) / len(scores)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "total": self.total,
            "accuracy": self.accuracy,
            "hard_accuracy": self.hard_accuracy,
            "macro_f1": self.macro_f1,
            "mean_latency_s": self.mean_latency,
            "per_label": self.per_label(),
            "confusion": self.confusion,
        }


def score_classifier(
    classifier: SentimentClassifier, headlines: Sequence[LabelledHeadline]
) -> SentimentScore:
    available, why = classifier.available()
    score = SentimentScore(
        name=classifier.name, kind=classifier.kind,
        available=available, unavailable_reason=why,
    )
    if not available:
        logger.warning("skipping %s: %s", classifier.name, why)
        return score

    for item in headlines:
        started = time.perf_counter()
        result = classifier.classify(item.text)
        score.latencies.append(time.perf_counter() - started)

        actual, predicted = item.label.value, result.label.value
        score.confusion.setdefault(actual, {})
        score.confusion[actual][predicted] = score.confusion[actual].get(predicted, 0) + 1

        score.total += 1
        hit = actual == predicted
        score.correct += int(hit)
        if item.hard:
            score.hard_total += 1
            score.hard_correct += int(hit)

    return score


def render_markdown(scores: Sequence[SentimentScore]) -> str:
    counts = label_counts()
    hard = sum(1 for h in HEADLINES if h.hard)
    majority = max(counts.values()) / len(HEADLINES)

    lines = [
        "## Sentiment track (bake-off candidate #4)",
        "",
        "Scored separately from the CONFIRM/VETO leaderboard: this is a",
        "three-class labelling task, not a veto decision, so the two are not",
        "comparable on one scale.",
        "",
        f"- Dataset: {len(HEADLINES)} hand-written headlines "
        f"({hard} marked *hard*: negation, mixed signals, sarcasm, jargon)",
        f"- Label balance: {counts}",
        f"- Always-predict-majority baseline: **{majority:.1%}** accuracy",
        "",
        "> **The dataset is hand-written, not scraped.** The news collector is",
        "> roadmap Phase 6 and does not exist yet, so there is no real feed to",
        "> sample. These scores are a sanity check on the classifier, not an",
        "> estimate of production accuracy. See `evaluation/sentiment_dataset.py`.",
        "",
        "| Classifier | Kind | Available | Accuracy | Hard-subset accuracy | Macro F1 | Mean latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for score in scores:
        lines.append(
            "| `{name}` | {kind} | {avail} | {acc} | {hard} | {f1} | {lat} |".format(
                name=score.name,
                kind=score.kind,
                avail="yes" if score.available else f"no - {score.unavailable_reason}",
                acc=_pct(score.accuracy),
                hard=_pct(score.hard_accuracy),
                f1="n/a" if score.macro_f1 is None else f"{score.macro_f1:.3f}",
                lat="n/a" if score.mean_latency is None else f"{score.mean_latency*1000:.1f}ms",
            )
        )

    ran = [s for s in scores if s.available and s.total]
    lines += ["", "### Reading this", ""]
    if ran:
        best = max(ran, key=lambda s: s.accuracy or 0)
        lines += [
            f"- Best available: **`{best.name}`** at {_pct(best.accuracy)} overall, "
            f"{_pct(best.hard_accuracy)} on the hard subset.",
            f"- Compare against the {majority:.1%} majority-class baseline: anything "
            "near that number has learned nothing.",
            "- The hard subset is the one that matters. Easy headlines are easy for",
            "  a word list too; the question is whether a learned model handles",
            "  negation and mixed signals better than keyword counting.",
        ]
    else:
        lines.append("- No classifier could be evaluated on this machine.")

    lines += [
        "",
        "### Per-class detail",
        "",
    ]
    for score in ran:
        lines += [f"**`{score.name}`**", "", "| Class | Precision | Recall | F1 | Support |",
                  "|---|---|---|---|---|"]
        for label, stats in score.per_label().items():
            lines.append(
                f"| {label} | {stats['precision']:.3f} | {stats['recall']:.3f} | "
                f"{stats['f1']:.3f} | {int(stats['support'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score sentiment classifiers.")
    parser.add_argument("--out", type=Path, default=Path("docs/sentiment_track_results.md"))
    parser.add_argument("--json-out", type=Path, default=Path("data/sentiment_scores.json"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    scores = [score_classifier(c, HEADLINES) for c in build_sentiment_candidates()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(scores), encoding="utf-8")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps([s.to_dict() for s in scores], indent=2), encoding="utf-8"
    )

    for score in scores:
        if score.available:
            print(
                f"{score.name:28s} acc={_pct(score.accuracy):>7s} "
                f"hard={_pct(score.hard_accuracy):>7s} macroF1={score.macro_f1:.3f}"
            )
        else:
            print(f"{score.name:28s} unavailable - {score.unavailable_reason}")
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
