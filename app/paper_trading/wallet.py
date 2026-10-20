"""Virtual wallet. Simulated money only -- there is no custody here.

Accounting model: a single quote-currency cash balance, margin-style. Opening
a position does not move cash (only the fee does); realised PnL and the exit
fee settle on close. This is what lets LONG and SHORT be symmetrical without
pretending to model spot borrowing.

See README "Open Decisions" (OD-2): if the target market ends up being LBank
spot rather than USDT-M futures, short trades need a different treatment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class InsufficientBalance(RuntimeError):
    """Raised when a simulated debit would take the balance below zero."""


@dataclass
class VirtualWallet:
    """Tracks simulated cash, fees paid, and realised PnL."""

    starting_balance: float
    quote_currency: str = "USDT"
    balance: float = field(init=False)
    realized_pnl: float = field(init=False, default=0.0)
    fees_paid: float = field(init=False, default=0.0)
    trades_closed: int = field(init=False, default=0)
    wins: int = field(init=False, default=0)
    losses: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.starting_balance <= 0:
            raise ValueError(f"starting_balance must be > 0, got {self.starting_balance}")
        self.balance = float(self.starting_balance)

    # -- mutations --------------------------------------------------------

    def charge_fee(self, amount: float) -> None:
        if amount < 0:
            raise ValueError(f"fee must be >= 0, got {amount}")
        if amount > self.balance:
            raise InsufficientBalance(
                f"fee {amount:.2f} exceeds balance {self.balance:.2f}"
            )
        self.balance -= amount
        self.fees_paid += amount

    def settle(self, pnl: float) -> None:
        """Apply realised PnL from a closed position (gross of the exit fee)."""
        self.balance += pnl
        self.realized_pnl += pnl
        self.trades_closed += 1
        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1

    # -- reporting --------------------------------------------------------

    @property
    def total_return_pct(self) -> float:
        return (self.balance - self.starting_balance) / self.starting_balance

    @property
    def win_rate(self) -> float | None:
        decided = self.wins + self.losses
        return self.wins / decided if decided else None

    def equity(self, unrealized_pnl: float = 0.0) -> float:
        """Balance plus open-position PnL marked to the last known price."""
        return self.balance + unrealized_pnl

    def to_dict(self) -> dict[str, Any]:
        return {
            "quote_currency": self.quote_currency,
            "starting_balance": self.starting_balance,
            "balance": self.balance,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "trades_closed": self.trades_closed,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "total_return_pct": self.total_return_pct,
        }
