"""Paper trading engine: simulated order fills, positions and PnL.

NOTHING in this module touches an exchange. It has no client, no network
call, and no import from ``app.market``. That is on purpose: the only way to
turn this into live trading would be to write a new execution layer, not to
flip a flag here.

Fill model (deliberately pessimistic)
-------------------------------------
* Entry and exit are both modelled as TAKER fills.
* Slippage is applied AGAINST us on every fill: a long buys higher and sells
  lower than the reference price; a short does the reverse.
* Fees are charged on notional at both entry and exit.

Defaults come from ``PaperTradingSettings`` and are set for LBank/Toobit-class
spreads (~0.44% / ~0.54% average), not tier-1 venue spreads. Optimistic cost
assumptions are the single easiest way to build a paper record that cannot be
reproduced with real money.

Intrabar ambiguity
------------------
When one candle's range contains BOTH the stop and the target, OHLC data
cannot tell us which printed first. We always resolve it as the STOP. This
biases results downward, which is the correct direction for a system whose
purpose is to decide whether real capital is warranted.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.config.settings import PaperTradingSettings
from app.paper_trading.wallet import VirtualWallet
from app.risk.levels import TradeLevels
from app.risk.position_sizing import PositionSize
from app.strategies.base import Signal, SignalDirection

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    MANUAL = "MANUAL"
    SESSION_END = "SESSION_END"


@dataclass
class PaperPosition:
    """One open simulated position."""

    position_id: str
    symbol: str
    direction: SignalDirection
    quantity: float
    entry_price: float  # fill price, slippage included
    reference_price: float  # signal close, before slippage
    stop_loss: float
    take_profit: float
    risk_amount: float
    entry_fee: float
    opened_at: datetime
    strategy_name: str
    invalidation: str
    exchange_id: str
    is_dev_data_source: bool
    signal_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price

    def unrealized_pnl(self, price: float) -> float:
        """Mark-to-market PnL, gross of the (not yet paid) exit fee."""
        if self.direction is SignalDirection.LONG:
            return (price - self.entry_price) * self.quantity
        return (self.entry_price - price) * self.quantity


@dataclass
class ClosedTrade:
    """A completed round trip, with every number needed to audit it."""

    position_id: str
    symbol: str
    direction: SignalDirection
    quantity: float
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    gross_pnl: float
    entry_fee: float
    exit_fee: float
    net_pnl: float
    r_multiple: float | None
    exit_reason: ExitReason
    opened_at: datetime
    closed_at: datetime
    strategy_name: str
    exchange_id: str
    is_dev_data_source: bool
    signal_id: int | None = None

    @property
    def holding_period_seconds(self) -> float:
        return (self.closed_at - self.opened_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_amount": self.risk_amount,
            "gross_pnl": self.gross_pnl,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "net_pnl": self.net_pnl,
            "r_multiple": self.r_multiple,
            "exit_reason": self.exit_reason.value,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "holding_period_seconds": self.holding_period_seconds,
            "strategy_name": self.strategy_name,
            "exchange_id": self.exchange_id,
            "is_dev_data_source": self.is_dev_data_source,
            "signal_id": self.signal_id,
        }


@dataclass(frozen=True)
class Candle:
    """The minimum a position update needs to resolve stops and targets."""

    timestamp: datetime
    high: float
    low: float
    close: float


class PaperTradingEngine:
    """Simulated execution over a virtual wallet."""

    def __init__(
        self,
        settings: PaperTradingSettings | None = None,
        wallet: VirtualWallet | None = None,
    ) -> None:
        self.settings = settings or PaperTradingSettings()
        self.wallet = wallet or VirtualWallet(
            starting_balance=self.settings.starting_balance,
            quote_currency=self.settings.quote_currency,
        )
        self.open_positions: dict[str, PaperPosition] = {}
        self.closed_trades: list[ClosedTrade] = []

    # -- fill pricing -----------------------------------------------------

    def _entry_fill(self, direction: SignalDirection, price: float) -> float:
        slip = self.settings.slippage_pct
        return price * (1 + slip) if direction is SignalDirection.LONG else price * (1 - slip)

    def _exit_fill(self, direction: SignalDirection, price: float) -> float:
        slip = self.settings.slippage_pct
        return price * (1 - slip) if direction is SignalDirection.LONG else price * (1 + slip)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.settings.taker_fee_pct

    # -- lifecycle --------------------------------------------------------

    def has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.open_positions.values())

    def open_position(
        self,
        signal: Signal,
        levels: TradeLevels,
        size: PositionSize,
        exchange_id: str,
        is_dev_data_source: bool,
        opened_at: datetime | None = None,
        signal_id: int | None = None,
    ) -> PaperPosition:
        """Open a simulated position from an approved, sized signal."""
        if signal.direction is SignalDirection.NO_TRADE:
            raise ValueError("cannot open a position from a NO_TRADE signal")
        if not size.accepted or size.quantity <= 0:
            raise ValueError("cannot open a position from a rejected position size")
        if self.has_position(signal.snapshot.symbol):
            raise ValueError(
                f"a position on {signal.snapshot.symbol} is already open; "
                "pyramiding is not supported at this stage"
            )

        entry_price = self._entry_fill(signal.direction, levels.entry)
        notional = size.quantity * entry_price
        entry_fee = self._fee(notional)
        self.wallet.charge_fee(entry_fee)

        position = PaperPosition(
            position_id=uuid.uuid4().hex[:12],
            symbol=signal.snapshot.symbol,
            direction=signal.direction,
            quantity=size.quantity,
            entry_price=entry_price,
            reference_price=levels.entry,
            stop_loss=levels.stop_loss,
            take_profit=levels.take_profit,
            risk_amount=size.risk_amount,
            entry_fee=entry_fee,
            opened_at=opened_at or datetime.now(timezone.utc),
            strategy_name=signal.strategy_name,
            invalidation=signal.invalidation,
            exchange_id=exchange_id,
            is_dev_data_source=is_dev_data_source,
            signal_id=signal_id,
            metadata=dict(signal.metadata),
        )
        self.open_positions[position.position_id] = position
        logger.info(
            "OPEN %s %s qty=%.8f entry=%.4f (ref %.4f) sl=%.4f tp=%.4f fee=%.4f",
            position.direction.value, position.symbol, position.quantity,
            position.entry_price, position.reference_price,
            position.stop_loss, position.take_profit, entry_fee,
        )
        return position

    def close_position(
        self,
        position_id: str,
        exit_reference_price: float,
        reason: ExitReason,
        closed_at: datetime | None = None,
    ) -> ClosedTrade:
        """Close a position at ``exit_reference_price`` plus adverse slippage."""
        position = self.open_positions.pop(position_id, None)
        if position is None:
            raise KeyError(f"no open position with id {position_id!r}")

        exit_price = self._exit_fill(position.direction, exit_reference_price)
        gross_pnl = position.unrealized_pnl(exit_price)
        exit_fee = self._fee(position.quantity * exit_price)
        net_pnl = gross_pnl - exit_fee

        # R multiple is measured net of BOTH fees against the risk we planned
        # to take, so a "1R win" really means the account moved by 1R.
        r_multiple = (
            (net_pnl - position.entry_fee) / position.risk_amount
            if position.risk_amount > 0
            else None
        )

        self.wallet.charge_fee(exit_fee)
        self.wallet.settle(gross_pnl)

        trade = ClosedTrade(
            position_id=position.position_id,
            symbol=position.symbol,
            direction=position.direction,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            risk_amount=position.risk_amount,
            gross_pnl=gross_pnl,
            entry_fee=position.entry_fee,
            exit_fee=exit_fee,
            net_pnl=net_pnl,
            r_multiple=r_multiple,
            exit_reason=reason,
            opened_at=position.opened_at,
            closed_at=closed_at or datetime.now(timezone.utc),
            strategy_name=position.strategy_name,
            exchange_id=position.exchange_id,
            is_dev_data_source=position.is_dev_data_source,
            signal_id=position.signal_id,
        )
        self.closed_trades.append(trade)
        logger.info(
            "CLOSE %s %s exit=%.4f reason=%s net_pnl=%.4f (%.2fR)",
            trade.direction.value, trade.symbol, trade.exit_price,
            trade.exit_reason.value, trade.net_pnl,
            trade.r_multiple if trade.r_multiple is not None else float("nan"),
        )
        return trade

    # -- per-candle update ------------------------------------------------

    def update_position(self, symbol: str, candle: Candle) -> ClosedTrade | None:
        """Check an open position on ``symbol`` against one candle.

        Stop takes precedence over target when the candle's range covers
        both -- see the module docstring.
        """
        position = next(
            (p for p in self.open_positions.values() if p.symbol == symbol), None
        )
        if position is None:
            return None

        if position.direction is SignalDirection.LONG:
            stop_hit = candle.low <= position.stop_loss
            target_hit = candle.high >= position.take_profit
        else:
            stop_hit = candle.high >= position.stop_loss
            target_hit = candle.low <= position.take_profit

        if stop_hit:
            return self.close_position(
                position.position_id, position.stop_loss, ExitReason.STOP_LOSS,
                closed_at=candle.timestamp,
            )
        if target_hit:
            return self.close_position(
                position.position_id, position.take_profit, ExitReason.TAKE_PROFIT,
                closed_at=candle.timestamp,
            )
        return None

    # -- reporting --------------------------------------------------------

    def unrealized_pnl(self, marks: dict[str, float]) -> float:
        """Total open PnL given ``{symbol: last_price}``."""
        return sum(
            p.unrealized_pnl(marks[p.symbol])
            for p in self.open_positions.values()
            if p.symbol in marks
        )

    def equity(self, marks: dict[str, float] | None = None) -> float:
        return self.wallet.equity(self.unrealized_pnl(marks or {}))

    def summary(self, marks: dict[str, float] | None = None) -> dict[str, Any]:
        wins = [t for t in self.closed_trades if t.net_pnl > 0]
        losses = [t for t in self.closed_trades if t.net_pnl < 0]
        gross_win = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        r_values = [t.r_multiple for t in self.closed_trades if t.r_multiple is not None]

        return {
            **self.wallet.to_dict(),
            "open_positions": len(self.open_positions),
            "equity": self.equity(marks),
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
            "avg_r_multiple": (sum(r_values) / len(r_values)) if r_values else None,
            "fee_pct": self.settings.taker_fee_pct,
            "slippage_pct": self.settings.slippage_pct,
        }
