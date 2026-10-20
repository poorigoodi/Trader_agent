"""Signal types shared by every strategy.

A signal is a *statement about the market*, not an order. It carries the
direction, the human-readable reasons behind it, and the condition that would
prove it wrong. Sizing, stop placement and any notion of money live in
``app/risk`` and ``app/paper_trading`` -- deliberately not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.indicators.engine import IndicatorSnapshot


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class Signal:
    """One strategy decision at one candle close.

    Attributes
    ----------
    direction:
        LONG / SHORT / NO_TRADE.
    reasons:
        Every rule that fired, in plain language, in evaluation order. For a
        NO_TRADE this explains which gate blocked the trade -- a NO_TRADE
        with no reason would be useless when reviewing the journal later.
    invalidation:
        The condition that would make this read wrong. Required for every
        direction, including NO_TRADE (where it describes what would have to
        change for a trade to appear).
    snapshot:
        The exact indicator values the decision was made from.
    """

    direction: SignalDirection
    reasons: tuple[str, ...]
    invalidation: str
    snapshot: IndicatorSnapshot
    strategy_name: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction is not SignalDirection.NO_TRADE

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no reason recorded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "direction": self.direction.value,
            "reasons": list(self.reasons),
            "reason_text": self.reason_text,
            "invalidation": self.invalidation,
            "metadata": dict(self.metadata),
            "snapshot": self.snapshot.to_dict(),
        }


class Strategy:
    """Base class. Subclasses implement :meth:`evaluate` and nothing else.

    Strategies are pure: given the same frame and settings they must return
    the same signal. No I/O, no clock reads, no randomness, and no LLM.
    """

    name: str = "base"

    def evaluate(self, df, symbol: str, timeframe: str) -> Signal:  # pragma: no cover
        raise NotImplementedError
