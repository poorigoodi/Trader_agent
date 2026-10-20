"""EMA cross + RSI filter + market-structure confirmation.

Design goals, in priority order:
  1. transparency  -- every branch produces a sentence a human can check
  2. determinism   -- same candles in, same signal out, always
  3. selectivity   -- three independent gates must agree, so NO_TRADE is the
                      normal outcome and a signal means something

The rules
---------
LONG requires all of:
  A. EMA(20) is above EMA(50) AND the cross happened within the last
     ``cross_max_age`` closed candles (a stale cross is not an entry).
  B. RSI(14) sits in [rsi_long_min, rsi_long_max] -- momentum confirms, but
     the move is not already exhausted.
  C. Market structure over the last ``structure_lookback`` candles reads UP
     (higher highs AND higher lows).

SHORT is the exact mirror. Anything else is NO_TRADE, with the failing gate
named in the reasons.

Not used as a gate, but recorded on every signal for later review: MACD
histogram sign, Bollinger %B, VWAP position. These are candidates for future
rules; promoting one to a gate is a deliberate change, not a silent tweak.
"""

from __future__ import annotations

import pandas as pd

from app.config.settings import StrategySettings
from app.indicators.engine import IndicatorSnapshot, build_snapshot
from app.market_structure.trend import Trend, TrendReading, detect_trend
from app.strategies.base import Signal, SignalDirection, Strategy


class EmaRsiTrendStrategy(Strategy):
    """The MVP rule-based strategy. No LLM, no fitted parameters."""

    name = "ema_rsi_trend_v1"

    def __init__(self, settings: StrategySettings | None = None) -> None:
        self.settings = settings or StrategySettings()

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _cross_age(df: pd.DataFrame, direction: SignalDirection) -> int | None:
        """How many closed candles ago EMA fast crossed EMA slow, or None.

        Age 0 means the cross printed on the most recent closed candle.
        Returns None when no cross of that direction exists in the visible
        window. Rows where either EMA is still NaN are skipped, so the
        warm-up boundary is never mistaken for a cross.
        """
        fast = df["ema_fast"].to_numpy(dtype="float64")
        slow = df["ema_slow"].to_numpy(dtype="float64")

        # Positions where both EMAs have warmed up (NaN != NaN).
        valid = [i for i in range(len(fast)) if fast[i] == fast[i] and slow[i] == slow[i]]
        if len(valid) < 2:
            return None

        for age, pos in enumerate(range(len(valid) - 1, 0, -1)):
            now, prev = valid[pos], valid[pos - 1]
            crossed_up = fast[now] > slow[now] and fast[prev] <= slow[prev]
            crossed_down = fast[now] < slow[now] and fast[prev] >= slow[prev]
            if direction is SignalDirection.LONG and crossed_up:
                return age
            if direction is SignalDirection.SHORT and crossed_down:
                return age
        return None

    def _no_trade(
        self,
        snapshot: IndicatorSnapshot,
        reasons: list[str],
        invalidation: str,
        metadata: dict | None = None,
    ) -> Signal:
        return Signal(
            direction=SignalDirection.NO_TRADE,
            reasons=tuple(reasons),
            invalidation=invalidation,
            snapshot=snapshot,
            strategy_name=self.name,
            metadata=metadata or {},
        )

    # -- main -------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Signal:
        """Evaluate the most recent CLOSED candle in ``df``.

        ``df`` must already carry indicator columns (see
        ``app.indicators.engine.compute_indicators``) and must not include a
        still-forming candle -- the data feed strips that.
        """
        s = self.settings
        min_rows = max(s.ema_slow + s.rsi_period, s.structure_lookback) + 5
        snapshot = build_snapshot(df, symbol, timeframe)

        if len(df) < min_rows:
            return self._no_trade(
                snapshot,
                [f"insufficient history: {len(df)} candles, need >= {min_rows}"],
                f"a signal becomes possible once {min_rows} candles are available",
            )

        if not snapshot.is_complete:
            return self._no_trade(
                snapshot,
                ["indicators not warmed up (EMA/RSI/ATR still NaN)"],
                "a signal becomes possible once all indicators have warmed up",
            )

        trend: TrendReading = detect_trend(
            df, lookback=s.structure_lookback, window=s.structure_swing_window
        )
        snapshot = replace_structure(snapshot, trend)

        ema_fast = float(snapshot.ema_fast)
        ema_slow = float(snapshot.ema_slow)
        rsi_value = float(snapshot.rsi)
        close = snapshot.close

        # Context recorded on every signal, gate or not.
        metadata = {
            "ema_spread_pct": (ema_fast - ema_slow) / ema_slow if ema_slow else None,
            "macd_hist": snapshot.macd_hist,
            "macd_hist_sign": _sign(snapshot.macd_hist),
            "bb_percent_b": snapshot.bb_percent_b,
            "close_vs_vwap_pct": (
                (close - snapshot.vwap) / snapshot.vwap if snapshot.vwap else None
            ),
            "structure_reason": trend.reason,
            "atr_pct_of_price": snapshot.atr / close if close else None,
        }

        reasons: list[str] = []

        # --- Gate A: fresh EMA cross ------------------------------------
        bullish_stack = ema_fast > ema_slow
        long_age = self._cross_age(df, SignalDirection.LONG)
        short_age = self._cross_age(df, SignalDirection.SHORT)

        if bullish_stack:
            candidate = SignalDirection.LONG
            cross_age = long_age
        else:
            candidate = SignalDirection.SHORT
            cross_age = short_age
        metadata["cross_age"] = cross_age

        stack_text = (
            f"EMA{s.ema_fast} {ema_fast:.2f} "
            f"{'>' if bullish_stack else '<'} EMA{s.ema_slow} {ema_slow:.2f}"
        )

        if cross_age is None:
            return self._no_trade(
                snapshot,
                [stack_text, "no EMA cross found in the visible window"],
                f"a fresh EMA{s.ema_fast}/{s.ema_slow} cross would open a setup",
                metadata,
            )
        if cross_age > s.cross_max_age:
            return self._no_trade(
                snapshot,
                [
                    stack_text,
                    f"EMA cross is stale: {cross_age} candles old, "
                    f"max {s.cross_max_age}",
                ],
                "a new EMA cross would open a setup; this one is already priced in",
                metadata,
            )
        reasons.append(stack_text)
        reasons.append(f"fresh EMA cross {cross_age} candle(s) ago")

        # --- Gate B: RSI window -----------------------------------------
        if candidate is SignalDirection.LONG:
            rsi_ok = s.rsi_long_min <= rsi_value <= s.rsi_long_max
            rsi_window = f"[{s.rsi_long_min:.0f}, {s.rsi_long_max:.0f}]"
        else:
            rsi_ok = s.rsi_short_min <= rsi_value <= s.rsi_short_max
            rsi_window = f"[{s.rsi_short_min:.0f}, {s.rsi_short_max:.0f}]"

        if not rsi_ok:
            return self._no_trade(
                snapshot,
                [
                    *reasons,
                    f"RSI{s.rsi_period} {rsi_value:.1f} outside {candidate.value} "
                    f"window {rsi_window}",
                ],
                f"RSI returning into {rsi_window} would re-open this setup",
                metadata,
            )
        reasons.append(f"RSI{s.rsi_period} {rsi_value:.1f} within {rsi_window}")

        # --- Gate C: market structure -----------------------------------
        wanted = Trend.UP if candidate is SignalDirection.LONG else Trend.DOWN
        if trend.trend is not wanted:
            return self._no_trade(
                snapshot,
                [
                    *reasons,
                    f"market structure is {trend.trend.value} "
                    f"({trend.reason}), needed {wanted.value}",
                ],
                f"structure printing {wanted.value} would confirm this setup",
                metadata,
            )
        reasons.append(f"structure {trend.trend.value} ({trend.reason})")

        # --- all gates passed -------------------------------------------
        if candidate is SignalDirection.LONG:
            invalidation = (
                f"close back below EMA{s.ema_slow} ({ema_slow:.2f}), "
                f"or a lower low below the last swing low"
            )
        else:
            invalidation = (
                f"close back above EMA{s.ema_slow} ({ema_slow:.2f}), "
                f"or a higher high above the last swing high"
            )

        return Signal(
            direction=candidate,
            reasons=tuple(reasons),
            invalidation=invalidation,
            snapshot=snapshot,
            strategy_name=self.name,
            metadata=metadata,
        )


def _sign(value: float | None) -> int | None:
    if value is None:
        return None
    return (value > 0) - (value < 0)


def replace_structure(snapshot: IndicatorSnapshot, trend: TrendReading) -> IndicatorSnapshot:
    """Return a copy of ``snapshot`` with the market-structure fields filled."""
    from dataclasses import replace

    return replace(
        snapshot,
        trend=trend.trend.value,
        higher_highs=trend.higher_highs,
        higher_lows=trend.higher_lows,
        lower_highs=trend.lower_highs,
        lower_lows=trend.lower_lows,
    )
