"""Stop loss / take profit / reward-to-risk arithmetic.

Everything in this module is closed-form arithmetic on numbers the strategy
already produced. There is no model here, and by design there never will be:
a language model must never decide where a stop goes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from app.config.settings import RiskSettings
from app.strategies.base import SignalDirection


@dataclass(frozen=True)
class TradeLevels:
    """Entry, stop and target for one prospective trade."""

    direction: SignalDirection
    entry: float
    stop_loss: float
    take_profit: float
    risk_per_unit: float  # absolute price distance entry -> stop
    reward_per_unit: float  # absolute price distance entry -> target
    rr: float  # reward_per_unit / risk_per_unit
    atr: float
    atr_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["direction"] = self.direction.value
        return data


def compute_levels(
    direction: SignalDirection,
    entry: float,
    atr: float,
    settings: RiskSettings | None = None,
) -> TradeLevels:
    """ATR-based stop, fixed-R target.

    stop distance = ``atr_stop_multiplier * ATR``
    target distance = ``rr_target * stop distance``

    An ATR-derived stop is used rather than a fixed percentage because it
    adapts to the instrument's current volatility: the same 1% stop is noise
    on one day and a real invalidation on another.
    """
    settings = settings or RiskSettings()

    if direction is SignalDirection.NO_TRADE:
        raise ValueError("cannot compute levels for a NO_TRADE signal")
    if entry <= 0:
        raise ValueError(f"entry must be > 0, got {entry}")
    if atr <= 0:
        raise ValueError(f"atr must be > 0, got {atr}")

    risk_per_unit = settings.atr_stop_multiplier * atr
    reward_per_unit = settings.rr_target * risk_per_unit

    if direction is SignalDirection.LONG:
        stop_loss = entry - risk_per_unit
        take_profit = entry + reward_per_unit
    else:
        stop_loss = entry + risk_per_unit
        take_profit = entry - reward_per_unit

    if stop_loss <= 0 or take_profit <= 0:
        raise ValueError(
            f"ATR {atr} is too large relative to entry {entry}: "
            f"stop={stop_loss}, target={take_profit}"
        )

    return TradeLevels(
        direction=direction,
        entry=float(entry),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        risk_per_unit=float(risk_per_unit),
        reward_per_unit=float(reward_per_unit),
        rr=float(reward_per_unit / risk_per_unit),
        atr=float(atr),
        atr_multiplier=settings.atr_stop_multiplier,
    )
