"""Correlation risk: stop four positions from being one bet in disguise.

Why this module exists
----------------------
With BTC/USDT and ETH/USDT only, a correlation module was a placeholder --
the two move together, and ``max_open_positions`` was the real constraint.
With five large-cap perps (BTC, ETH, BNB, XRP, SOL) it is not a placeholder
any more. All five carry high beta to crypto as a whole; in a risk-off hour
they fall together. Four simultaneous longs across them is not four 1% bets,
it is closer to one 4% bet on a single factor, and the daily risk cap will
not notice because each individual trade looks small.

What it does
------------
1. Compute a rolling correlation matrix of log returns across the symbols we
   might hold.
2. Group symbols into clusters via single-linkage on
   ``correlation >= correlation_threshold``. Single-linkage (transitive
   grouping) is the conservative choice: if A~B and B~C, then A, B and C are
   treated as one cluster even when A and C are not directly correlated,
   because in a crypto drawdown they usually are.
3. Cap the total risk allowed inside any one cluster at
   ``max_correlated_risk_pct`` of balance.

Direction matters: a LONG on BTC and a SHORT on ETH are a *spread*, not a
doubled bet, so exposures are signed and it is the absolute NET exposure per
cluster that is capped.

Everything here is closed-form arithmetic on price history. No model, no
fitting, no LLM -- it satisfies the project's "risk stays deterministic"
rule the same way position sizing does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from app.config.settings import RiskSettings
from app.strategies.base import SignalDirection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExposureItem:
    """One position's signed risk contribution."""

    symbol: str
    direction: SignalDirection
    risk_amount: float

    @property
    def signed_risk(self) -> float:
        return (
            self.risk_amount
            if self.direction is SignalDirection.LONG
            else -self.risk_amount
        )


@dataclass(frozen=True)
class CorrelationDecision:
    """Whether a proposed trade fits inside its cluster's risk budget."""

    approved: bool
    reason: str
    cluster: tuple[str, ...] = ()
    cluster_net_risk: float = 0.0
    cluster_budget: float = 0.0
    allowed_risk_amount: float = 0.0
    correlations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "cluster": list(self.cluster),
            "cluster_net_risk": self.cluster_net_risk,
            "cluster_budget": self.cluster_budget,
            "allowed_risk_amount": self.allowed_risk_amount,
            "correlations": dict(self.correlations),
        }


def log_returns(closes: pd.Series) -> pd.Series:
    """Log returns. Log rather than simple returns so they are additive."""
    values = closes.astype("float64")
    return np.log(values / values.shift(1))


def correlation_matrix(
    closes_by_symbol: Mapping[str, pd.Series],
    lookback: int = 96,
    min_observations: int = 30,
) -> pd.DataFrame:
    """Pearson correlation of log returns over the last ``lookback`` candles.

    Symbols are aligned on their index before correlating, so a shorter or
    gappy series cannot silently shift another one. Pairs with fewer than
    ``min_observations`` overlapping returns are reported as NaN, and callers
    must treat NaN as "assume correlated" -- never as "assume independent".
    """
    if not closes_by_symbol:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {symbol: log_returns(series) for symbol, series in closes_by_symbol.items()}
    )
    frame = frame.tail(lookback)
    matrix = frame.corr(min_periods=min_observations)

    # A symbol with no variance correlates with nothing; make that explicit
    # rather than leaving a silent NaN that reads like missing data.
    return matrix


def find_cluster(
    symbol: str,
    peers: Iterable[str],
    matrix: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], dict[str, float]]:
    """Single-linkage cluster containing ``symbol``.

    Returns the cluster members (including ``symbol``) and the correlation of
    each peer to ``symbol`` for logging. Missing/NaN correlations are treated
    as correlated: with five assets in one asset class, "we could not measure
    it" is not evidence of independence.
    """
    peers = [p for p in peers if p != symbol]
    cluster = [symbol]
    correlations: dict[str, float] = {}

    frontier = [symbol]
    seen = {symbol}
    while frontier:
        current = frontier.pop()
        for peer in peers:
            if peer in seen:
                continue
            value = _lookup(matrix, current, peer)
            if peer not in correlations and current == symbol:
                correlations[peer] = value if value is not None else float("nan")
            # None (unmeasurable) fails safe: treat as correlated.
            if value is None or abs(value) >= threshold:
                seen.add(peer)
                cluster.append(peer)
                frontier.append(peer)

    return cluster, correlations


def _lookup(matrix: pd.DataFrame, a: str, b: str) -> float | None:
    if matrix.empty or a not in matrix.index or b not in matrix.columns:
        return None
    value = matrix.at[a, b]
    if value is None or (isinstance(value, float) and value != value):
        return None
    return float(value)


class CorrelationRiskManager:
    """Caps net risk per correlated cluster. Deterministic, no fitting."""

    def __init__(self, settings: RiskSettings | None = None) -> None:
        self.settings = settings or RiskSettings()

    def evaluate(
        self,
        symbol: str,
        direction: SignalDirection,
        proposed_risk: float,
        balance: float,
        open_exposures: Iterable[ExposureItem],
        closes_by_symbol: Mapping[str, pd.Series],
    ) -> CorrelationDecision:
        """Decide whether ``proposed_risk`` on ``symbol`` fits its cluster.

        Returns an approval with ``allowed_risk_amount`` possibly reduced
        below ``proposed_risk``: partial admission is better than a binary
        refusal, and mirrors how the daily budget already behaves.
        """
        settings = self.settings
        exposures = list(open_exposures)
        budget = balance * settings.max_correlated_risk_pct

        if budget <= 0:
            return CorrelationDecision(
                False, "correlated-risk budget is zero", allowed_risk_amount=0.0
            )

        # No open positions -> nothing to be correlated WITH.
        if not exposures:
            allowed = min(proposed_risk, budget)
            if allowed < proposed_risk:
                return CorrelationDecision(
                    True,
                    f"first position, trimmed to the per-cluster budget "
                    f"{budget:.2f}",
                    cluster=(symbol,),
                    cluster_budget=budget,
                    allowed_risk_amount=allowed,
                )
            return CorrelationDecision(
                True, "no open positions to correlate against",
                cluster=(symbol,), cluster_budget=budget,
                allowed_risk_amount=allowed,
            )

        universe = [symbol] + [e.symbol for e in exposures]
        matrix = correlation_matrix(
            {s: closes_by_symbol[s] for s in universe if s in closes_by_symbol},
            lookback=settings.correlation_lookback,
            min_observations=settings.correlation_min_observations,
        )

        cluster, correlations = find_cluster(
            symbol, [e.symbol for e in exposures], matrix, settings.correlation_threshold
        )

        # Signed net risk already committed inside this cluster.
        proposed_sign = 1.0 if direction is SignalDirection.LONG else -1.0
        cluster_net = sum(e.signed_risk for e in exposures if e.symbol in cluster)

        # How much of the proposed risk keeps |net| within budget?
        projected = cluster_net + proposed_sign * proposed_risk
        if abs(projected) <= budget:
            return CorrelationDecision(
                True,
                f"cluster {sorted(cluster)} net risk "
                f"{abs(projected):.2f} within budget {budget:.2f}",
                cluster=tuple(cluster),
                cluster_net_risk=cluster_net,
                cluster_budget=budget,
                allowed_risk_amount=proposed_risk,
                correlations=correlations,
            )

        # Trim to whatever still fits in the same direction.
        headroom = budget - proposed_sign * cluster_net
        allowed = max(0.0, min(proposed_risk, headroom))

        if allowed <= 0:
            return CorrelationDecision(
                False,
                f"correlated cluster {sorted(cluster)} already holds "
                f"{abs(cluster_net):.2f} of its {budget:.2f} budget",
                cluster=tuple(cluster),
                cluster_net_risk=cluster_net,
                cluster_budget=budget,
                allowed_risk_amount=0.0,
                correlations=correlations,
            )

        return CorrelationDecision(
            True,
            f"trimmed to {allowed:.2f} to stay inside correlated cluster "
            f"{sorted(cluster)} budget {budget:.2f}",
            cluster=tuple(cluster),
            cluster_net_risk=cluster_net,
            cluster_budget=budget,
            allowed_risk_amount=allowed,
            correlations=correlations,
        )
