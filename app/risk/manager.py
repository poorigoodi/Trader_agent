"""Risk manager: the single gate between a signal and a paper trade.

Holds the daily risk ledger (how much risk has been committed since 00:00
UTC) and combines level calculation with sizing. The orchestrator never calls
``compute_levels`` or ``calculate_position_size`` directly -- it goes through
here, so the daily cap can never be bypassed by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from app.config.settings import RiskSettings
from app.risk.correlation import (
    CorrelationDecision,
    CorrelationRiskManager,
    ExposureItem,
)
from app.risk.levels import TradeLevels, compute_levels
from app.risk.position_sizing import PositionSize, calculate_position_size
from app.strategies.base import Signal, SignalDirection


@dataclass(frozen=True)
class RiskDecision:
    """Result of running one signal through the risk gate."""

    approved: bool
    levels: TradeLevels | None
    size: PositionSize | None
    reason: str
    correlation: CorrelationDecision | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "levels": self.levels.to_dict() if self.levels else None,
            "size": self.size.to_dict() if self.size else None,
            "correlation": self.correlation.to_dict() if self.correlation else None,
        }


class RiskManager:
    """Stateful only in the daily risk ledger; the maths stays pure."""

    def __init__(self, settings: RiskSettings | None = None) -> None:
        self.settings = settings or RiskSettings()
        self._ledger_day: date | None = None
        self._risk_committed_today: float = 0.0
        self.correlation = CorrelationRiskManager(self.settings)

    # -- daily ledger -----------------------------------------------------

    def _roll_day(self, now: datetime) -> None:
        """Reset the ledger when the UTC day changes."""
        today = now.astimezone(timezone.utc).date()
        if self._ledger_day != today:
            self._ledger_day = today
            self._risk_committed_today = 0.0

    def risk_committed_today(self, now: datetime | None = None) -> float:
        self._roll_day(now or datetime.now(timezone.utc))
        return self._risk_committed_today

    def commit_risk(self, amount: float, now: datetime | None = None) -> None:
        """Record risk actually taken. Call this only when a trade opens."""
        self._roll_day(now or datetime.now(timezone.utc))
        self._risk_committed_today += float(amount)

    def release_risk(self, amount: float, now: datetime | None = None) -> None:
        """Give budget back if a trade is cancelled before it ever opened.

        Deliberately NOT called when a position closes: the daily cap limits
        risk *taken* during the day, not risk currently open. Refunding on a
        win would let a good morning fund an unlimited afternoon.
        """
        self._roll_day(now or datetime.now(timezone.utc))
        self._risk_committed_today = max(0.0, self._risk_committed_today - float(amount))

    # -- the gate ---------------------------------------------------------

    def evaluate(
        self,
        signal: Signal,
        balance: float,
        open_positions: int = 0,
        now: datetime | None = None,
        open_exposures: list[ExposureItem] | None = None,
        closes_by_symbol: dict[str, Any] | None = None,
    ) -> RiskDecision:
        """Turn an actionable signal into an approved, sized trade or a refusal.

        Gate order, cheapest and most fundamental first:
          1. direction is actionable
          2. a stop can be placed at all (ATR exists)
          3. per-trade sizing, daily budget, notional and venue minimums
          4. correlated-cluster budget across the open book

        Correlation runs LAST because it needs a candidate size to trim.
        """
        if signal.direction is SignalDirection.NO_TRADE:
            return RiskDecision(False, None, None, "signal is NO_TRADE")

        snapshot = signal.snapshot
        if snapshot.atr is None or snapshot.atr <= 0:
            return RiskDecision(False, None, None, "ATR unavailable; cannot place a stop")

        # Entry is the close of the signal candle. Actual fill price (with
        # slippage) is decided by the paper engine, not here: risk sizing
        # uses the reference price so the numbers are reproducible.
        levels = compute_levels(signal.direction, snapshot.close, snapshot.atr, self.settings)

        size = calculate_position_size(
            balance=balance,
            levels=levels,
            settings=self.settings,
            daily_risk_used=self.risk_committed_today(now),
            open_positions=open_positions,
        )

        if not size.accepted:
            return RiskDecision(False, levels, size, size.rejection_reason or "rejected")

        # --- correlated cluster budget --------------------------------
        correlation = None
        if closes_by_symbol:
            correlation = self.correlation.evaluate(
                symbol=snapshot.symbol,
                direction=signal.direction,
                proposed_risk=size.risk_amount,
                balance=balance,
                open_exposures=open_exposures or [],
                closes_by_symbol=closes_by_symbol,
            )
            if not correlation.approved:
                return RiskDecision(
                    False, levels, size, correlation.reason, correlation
                )
            if correlation.allowed_risk_amount < size.risk_amount:
                size = _rescale(size, correlation.allowed_risk_amount, levels, balance)
                if not size.accepted:
                    return RiskDecision(
                        False, levels, size,
                        size.rejection_reason or "correlated trim left dust",
                        correlation,
                    )

        caps = f" (capped by {', '.join(size.caps_applied)})" if size.caps_applied else ""
        reason = (
            f"risking {size.risk_amount:.2f} "
            f"({size.risk_pct_of_balance * 100:.2f}% of balance) "
            f"on {size.quantity:.8f} units, R:R {levels.rr:.2f}{caps}"
        )
        return RiskDecision(True, levels, size, reason, correlation)


def _rescale(
    size: PositionSize,
    allowed_risk: float,
    levels: TradeLevels,
    balance: float,
) -> PositionSize:
    """Re-derive quantity and notional from a trimmed risk amount.

    Trimming risk without re-deriving quantity would leave a position whose
    logged risk and actual risk disagree -- the exact kind of quiet
    inconsistency that makes a trade journal useless.
    """
    from dataclasses import replace

    quantity = allowed_risk / levels.risk_per_unit
    notional = quantity * levels.entry

    if notional < size.notional and notional < 10.0:
        return replace(
            size,
            accepted=False,
            quantity=0.0,
            notional=0.0,
            risk_amount=0.0,
            risk_pct_of_balance=0.0,
            rejection_reason=(
                f"correlated-cluster trim left a notional of {notional:.2f}, "
                f"below the venue minimum"
            ),
        )

    return replace(
        size,
        quantity=quantity,
        notional=notional,
        risk_amount=allowed_risk,
        risk_pct_of_balance=allowed_risk / balance if balance else 0.0,
        caps_applied=size.caps_applied + ("correlated_cluster",),
    )
