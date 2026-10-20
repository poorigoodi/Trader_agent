"""Crypto news sentiment classification -- a SEPARATE track from the veto layer.

Why this is scored separately
-----------------------------
Candidate #4 in the model bake-off is a small specialised classifier
(FinBERT-class) rather than a generative model. It does a different job from
the other three: it labels a headline POSITIVE / NEGATIVE / NEUTRAL, it does
not decide CONFIRM / VETO on a trading signal. Scoring it on the same
leaderboard would be comparing two different tasks and calling one the
winner, so it gets its own metrics.

It also has no production input yet: the news collector is roadmap Phase 6 and
does not exist. This module is therefore the classifier half of that future
layer, evaluated on a hand-labelled sample so its accuracy is a measured
number rather than an assumption.

Three implementations
---------------------
``LexiconSentiment``  - deterministic keyword scoring. Always available, zero
                        dependencies, and the control arm: any learned model
                        that cannot beat a word list is not earning its
                        install size.
``FinBertSentiment``  - a real FinBERT-class transformer, if ``transformers``
                        and a model are installed locally. Free, offline.
``OllamaSentiment``   - a small generative model asked to classify, for
                        comparison against the purpose-built classifier.

Nothing here places trades or feeds the risk module. Even once the news layer
exists, sentiment can only ever reach a signal through the LLM layer's
one-way veto.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class Sentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class SentimentResult:
    label: Sentiment
    score: float  # -1.0 (max negative) .. +1.0 (max positive)
    detail: str = ""


class SentimentClassifier:
    """Base interface, mirroring ``ai_decision.providers.Provider``."""

    name: str = "base"
    kind: str = "local"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def classify(self, headline: str) -> SentimentResult:
        raise NotImplementedError

    def classify_many(self, headlines: Sequence[str]) -> list[SentimentResult]:
        return [self.classify(h) for h in headlines]


# --------------------------------------------------------------------------
# Lexicon baseline
# --------------------------------------------------------------------------

# Weighted terms. Chosen for crypto specifically: "listing" and "unlock" carry
# strong directional meaning here that a general finance lexicon misses.
POSITIVE_TERMS: dict[str, float] = {
    "surge": 2.0, "surges": 2.0, "rally": 2.0, "rallies": 2.0, "soar": 2.0,
    "soars": 2.0, "jump": 1.5, "jumps": 1.5, "gain": 1.2, "gains": 1.2,
    "rise": 1.0, "rises": 1.0, "climb": 1.2, "climbs": 1.2, "bullish": 2.0,
    "record high": 2.5, "all-time high": 2.5, "breakout": 1.8, "upgrade": 1.5,
    "approval": 2.0, "approved": 2.0, "approves": 2.0, "adoption": 1.8,
    "partnership": 1.5, "integration": 1.2, "launch": 1.0, "launches": 1.0,
    "listing": 2.0, "lists": 1.5, "inflow": 1.8, "inflows": 1.8,
    "accumulate": 1.5, "accumulation": 1.5, "institutional": 1.0,
    "etf approval": 3.0, "buyback": 1.5, "staking rewards": 1.2,
    "milestone": 1.2, "outperform": 1.8, "recovery": 1.5, "rebound": 1.8,
}

NEGATIVE_TERMS: dict[str, float] = {
    "hack": 3.0, "hacked": 3.0, "exploit": 3.0, "exploited": 3.0,
    "breach": 2.5, "stolen": 2.5, "theft": 2.5, "rug pull": 3.0,
    "scam": 2.5, "fraud": 2.5, "lawsuit": 2.0, "sues": 2.0, "sued": 2.0,
    "investigation": 1.8, "subpoena": 2.0, "crackdown": 2.2, "ban": 2.5,
    "bans": 2.5, "banned": 2.5, "delisting": 3.0, "delisted": 3.0,
    "halt": 2.2, "halts": 2.2, "halted": 2.2, "suspend": 2.0,
    "suspends": 2.0, "suspended": 2.0, "insolvency": 3.0, "bankrupt": 3.0,
    "bankruptcy": 3.0, "liquidation": 2.2, "liquidations": 2.2,
    "crash": 2.5, "crashes": 2.5, "plunge": 2.2, "plunges": 2.2,
    "plummet": 2.2, "slump": 1.8, "tumble": 1.8, "tumbles": 1.8,
    "drop": 1.2, "drops": 1.2, "fall": 1.2, "falls": 1.2, "decline": 1.2,
    "bearish": 2.0, "selloff": 2.0, "sell-off": 2.0, "outflow": 1.8,
    "outflows": 1.8, "downgrade": 1.5, "warning": 1.5, "risk": 0.8,
    "unlock": 1.5, "dump": 2.0, "exploit attempt": 2.5, "vulnerability": 2.0,
    "outage": 1.8, "downtime": 1.5, "rejected": 2.0, "denial": 1.8,
}

# Terms that mark a headline as reporting rather than moving.
NEUTRAL_MARKERS = (
    "unchanged", "flat", "sideways", "steady", "holds", "consolidat",
    "little changed", "mixed", "range-bound", "awaits", "ahead of",
)


class LexiconSentiment(SentimentClassifier):
    """Deterministic weighted keyword scoring. The control arm."""

    name = "lexicon-baseline"
    kind = "baseline"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def available(self) -> tuple[bool, str]:
        return True, ""

    def classify(self, headline: str) -> SentimentResult:
        text = (headline or "").lower()
        if not text.strip():
            return SentimentResult(Sentiment.NEUTRAL, 0.0, "empty headline")

        positive = self._score(text, POSITIVE_TERMS)
        negative = self._score(text, NEGATIVE_TERMS)

        # Negation flips the sense of the nearest term, crudely but usefully:
        # "no hack occurred" should not read as maximally negative.
        if re.search(r"\b(no|not|denies|denied|false|dismissed)\b", text):
            positive, negative = positive + negative * 0.5, negative * 0.3

        net = positive - negative
        if any(marker in text for marker in NEUTRAL_MARKERS) and abs(net) < 2.0:
            return SentimentResult(Sentiment.NEUTRAL, 0.0, "neutral marker present")

        if net >= self.threshold:
            label = Sentiment.POSITIVE
        elif net <= -self.threshold:
            label = Sentiment.NEGATIVE
        else:
            label = Sentiment.NEUTRAL

        scale = max(positive + negative, 1.0)
        return SentimentResult(
            label,
            max(-1.0, min(1.0, net / scale)),
            f"pos={positive:.1f} neg={negative:.1f}",
        )

    @staticmethod
    def _score(text: str, terms: dict[str, float]) -> float:
        total = 0.0
        for term, weight in terms.items():
            if " " in term:
                if term in text:
                    total += weight
            elif re.search(rf"\b{re.escape(term)}\b", text):
                total += weight
        return total


# --------------------------------------------------------------------------
# FinBERT-class transformer
# --------------------------------------------------------------------------

class FinBertSentiment(SentimentClassifier):
    """A small finance-domain transformer, run locally.

    Candidate #4 proper. Roughly 110M parameters, CPU-runnable, free. Only
    available when ``transformers`` (and its weights) are installed; the
    bake-off reports it as unavailable rather than pretending otherwise.
    """

    kind = "local"

    def __init__(self, model_id: str = "ProsusAI/finbert") -> None:
        self.model_id = model_id
        self.name = f"finbert/{model_id.split('/')[-1]}"
        self._pipeline = None

    def available(self) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, (
                "transformers is not installed "
                "(pip install transformers torch --index-url ...)"
            )
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001
            return False, f"model {self.model_id!r} could not be loaded: {exc}"
        return True, ""

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        self._pipeline = pipeline(
            "sentiment-analysis", model=self.model_id, truncation=True
        )

    def classify(self, headline: str) -> SentimentResult:
        self._load()
        raw = self._pipeline(headline)[0]
        label = str(raw.get("label", "")).upper()
        score = float(raw.get("score", 0.0))

        mapping = {
            "POSITIVE": Sentiment.POSITIVE,
            "NEGATIVE": Sentiment.NEGATIVE,
            "NEUTRAL": Sentiment.NEUTRAL,
        }
        sentiment = mapping.get(label, Sentiment.NEUTRAL)
        signed = score if sentiment is Sentiment.POSITIVE else (
            -score if sentiment is Sentiment.NEGATIVE else 0.0
        )
        return SentimentResult(sentiment, signed, f"{label} p={score:.3f}")


# --------------------------------------------------------------------------
# Small generative model via Ollama
# --------------------------------------------------------------------------

SENTIMENT_PROMPT = """Classify the sentiment of this crypto news headline for \
a trader holding the asset it mentions.

Headline: {headline}

Answer with ONE JSON object and nothing else:
{{"label": "POSITIVE" | "NEGATIVE" | "NEUTRAL"}}"""


class OllamaSentiment(SentimentClassifier):
    """A small generative model asked to do the classifier's job.

    Included so the bake-off can answer a real question: is a purpose-built
    110M classifier better than a general 1.7B model at this specific task?
    """

    kind = "local"

    def __init__(self, model_id: str = "qwen3:1.7b", host: str = "http://localhost:11434"):
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.name = f"ollama/{model_id}"

    def available(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3.0) as response:
                tags = json.loads(response.read())
        except Exception as exc:  # noqa: BLE001
            return False, f"Ollama not reachable at {self.host} ({exc})"
        installed = {m.get("name", "") for m in tags.get("models", [])}
        if not installed:
            return False, "Ollama has no models pulled"
        return True, ""

    def classify(self, headline: str) -> SentimentResult:
        body = json.dumps(
            {
                "model": self.model_id,
                "prompt": SENTIMENT_PROMPT.format(headline=headline),
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60.0) as response:
                payload = json.loads(response.read())
            parsed = json.loads(payload.get("response", "{}"))
            label = str(parsed.get("label", "NEUTRAL")).upper()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama sentiment failed: %s", exc)
            return SentimentResult(Sentiment.NEUTRAL, 0.0, f"error: {exc}")

        try:
            sentiment = Sentiment(label)
        except ValueError:
            return SentimentResult(Sentiment.NEUTRAL, 0.0, f"unparseable label {label!r}")
        signed = {Sentiment.POSITIVE: 1.0, Sentiment.NEGATIVE: -1.0, Sentiment.NEUTRAL: 0.0}
        return SentimentResult(sentiment, signed[sentiment], label)


def build_sentiment_candidates() -> list[SentimentClassifier]:
    return [FinBertSentiment(), OllamaSentiment(), LexiconSentiment()]
