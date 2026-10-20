"""Deterministic position sizing.

The core identity, and the only one that matters:

    quantity = (balance * risk_per_trade_pct) / risk_per_unit

That is: risk a fixed fraction of the *current* balance, and let the distance
to the stop decide how many units that buys. A wider stop therefore buys a
smaller position, which is the whole point -- the loss when the stop is hit
is the same fraction of the account regardless of volatility.

Every rejection returns a reason. A silent zero-size would be indistinguishable
from "no signal" in the journal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

from app.config.settings import RiskSettings
from app.risk.levels import TradeLevels


@dataclass(frozen=True)
class PositionSize:
    """The sizing decision, accepted or not, with its full arithmetic."""

    accepted: bool
    quantity: float
    notional: float
    risk_amount: float  # what we lose in quote currency if the stop is hit
    risk_pct_of_balance: float
    rejection_reason: str | None = None
    caps_applied: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["caps_applied"] = list(self.caps_applied)
        return data


def _reject(reason: str) -> PositionSize:
    return PositionSize(
        accepted=False,
        quantity=0.0,
        notional=0.0,
        risk_amount=0.0,
        risk_pct_of_balance=0.0,
        rejection_reason=reason,
    )


def calculate_position_size(
    balance: float,
    levels: TradeLevels,
    settings: RiskSettings | None = None,
    daily_risk_used: float = 0.0,
    open_positions: int = 0,
) -> PositionSize:
    """Size one position, or refuse with a reason.

    Parameters
    ----------
    balance:
        Current virtual balance in quote currency.
    levels:
        Entry/stop/target from :func:`app.risk.levels.compute_levels`.
    daily_risk_used:
        Quote-currency risk already committed today. Checked against
        ``max_daily_risk_pct * balance``.
    open_positions:
        How many positions are already open.

    Checks are ordered cheapest-and-most-fundamental first so the rejection
    reason names the real blocker rather than an incidental one.
    """
    settings = settings or RiskSettings()

    if balance <= 0:
        return _reject(f"balance is {balance}, nothing to risk")
    if levels.risk_per_unit <= 0:
        return _reject("risk_per_unit is zero; entry and stop are the same price")
    if levels.rr < settings.min_rr:
        return _reject(
            f"reward:risk {levels.rr:.2f} below minimum {settings.min_rr:.2f}"
        )
    if open_positions >= settings.max_open_positions:
        return _reject(
            f"already at max_open_positions ({open_positions}/"
            f"{settings.max_open_positions})"
        )

    # --- daily risk budget ----------------------------------------------
    daily_budget = balance * settings.max_daily_risk_pct
    remaining_budget = daily_budget - daily_risk_used
    if remaining_budget <= 0:
        return _reject(
            f"daily risk budget exhausted: {daily_risk_used:.2f} of "
            f"{daily_budget:.2f} already committed today"
        )

    caps: list[str] = []

    # --- base risk amount -------------------------------------------------
    risk_amount = balance * settings.risk_per_trade_pct
    if risk_amount > remaining_budget:
        risk_amount = remaining_budget
        caps.append("daily_risk_budget")

    quantity = risk_amount / levels.risk_per_unit
    notional = quantity * levels.entry

    # --- notional ceiling -------------------------------------------------
    # A very tight stop can imply a position far larger than the account.
    # Cap the notional and re-derive quantity and risk from the cap, so the
    # numbers stay internally consistent.
    max_notional = balance * settings.max_position_notional_pct
    if notional > max_notional:
        notional = max_notional
        quantity = notional / levels.entry
        risk_amount = quantity * levels.risk_per_unit
        caps.append("max_position_notional")

    # --- venue minimum ----------------------------------------------------
    if notional < settings.min_notional_usdt:
        return _reject(
            f"notional {notional:.2f} below venue minimum "
            f"{settings.min_notional_usdt:.2f}"
        )

    return PositionSize(
        accepted=True,
        quantity=float(quantity),
        notional=float(notional),
        risk_amount=float(risk_amount),
        risk_pct_of_balance=float(risk_amount / balance),
        rejection_reason=None,
        caps_applied=tuple(caps),
    )
