"""Calibrated confidence: a small classical model, never an LLM's self-report.

WHAT THIS IS AND IS NOT
-----------------------
This is the ONLY thing in the project that is "trained", and it is
deliberately not a language model. It is a logistic regression (optionally a
small gradient-boosted tree) over a few dozen numeric features, whose target
is the one thing that actually matters: did this signal win or lose?

Fine-tuning an LLM was considered and rejected for this phase. It needs a
large labelled corpus, a GPU, and it invites exactly the overfitting this
project's ground rules warn about. A logistic regression on a few hundred
rows trains in under a second on a CPU and, crucially, is inspectable: you
can read its coefficients and see what it believes.

The output replaces the "confidence" number a model might report about
itself. A model's stated confidence is a fact about the model; this is an
estimate derived from what similar setups actually did.

TEMPORAL SPLIT, NEVER RANDOM
----------------------------
:func:`temporal_split` orders by timestamp and cuts at a fraction. A random
split would put a signal from Tuesday in training and its neighbour from
Monday in test, letting the model learn from the future about the past -- the
same look-ahead bias the backtester is built to prevent, reintroduced at the
last step. ``train_test_split(shuffle=True)`` is never used here, and a test
asserts the split is ordered.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features available at decision time. Every one of these is computed from
# candles at or before the signal candle, so none of them leaks.
FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi",
    "atr_pct_of_price",
    "ema_spread_pct",
    "macd_hist_norm",
    "bb_percent_b",
    "close_vs_vwap_pct",
    "cross_age",
    "is_long",
    "trend_agrees",
    "rr",
)

TARGET_COLUMN = "won"


class NotEnoughData(RuntimeError):
    """Raised when there are too few labelled rows to train honestly."""


@dataclass
class CalibrationReport:
    """What the trained model scored, and on what."""

    model_type: str
    n_total: int
    n_train: int
    n_test: int
    train_positive_rate: float
    test_positive_rate: float
    split_timestamp: str
    features: list[str]

    train_accuracy: float | None = None
    test_accuracy: float | None = None
    test_auc: float | None = None
    test_brier: float | None = None
    baseline_brier: float | None = None
    log_loss: float | None = None
    coefficients: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def beats_baseline(self) -> bool:
        """True when the model's probabilities are better than always
        predicting the base rate. Brier: lower is better."""
        if self.test_brier is None or self.baseline_brier is None:
            return False
        return self.test_brier < self.baseline_brier

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["beats_baseline"] = self.beats_baseline
        return data

    def summary(self) -> str:
        lines = [
            f"model            : {self.model_type}",
            f"rows             : {self.n_total} "
            f"({self.n_train} train / {self.n_test} test)",
            f"split at         : {self.split_timestamp}",
            f"win rate         : {self.train_positive_rate:.1%} train / "
            f"{self.test_positive_rate:.1%} test",
            f"test accuracy    : {_fmt(self.test_accuracy)}",
            f"test AUC         : {_fmt(self.test_auc)}",
            f"test Brier       : {_fmt(self.test_brier)} "
            f"(baseline {_fmt(self.baseline_brier)})",
            f"beats baseline   : {self.beats_baseline}",
        ]
        if self.warnings:
            lines.append("warnings:")
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn labelled backtest signals into a numeric feature matrix.

    Only rows that were actually traded (and therefore have an outcome) can
    be used: an untaken signal has no ground truth.
    """
    rows = frame[frame[TARGET_COLUMN].notna()].copy()
    if rows.empty:
        return rows

    rows["is_long"] = (rows["direction"] == "LONG").astype(float)
    rows["trend_agrees"] = (
        ((rows["direction"] == "LONG") & (rows["trend"] == "UP"))
        | ((rows["direction"] == "SHORT") & (rows["trend"] == "DOWN"))
    ).astype(float)

    # MACD histogram scales with price, so normalise it or BTC dominates the
    # coefficient purely by being a bigger number.
    rows["macd_hist_norm"] = rows["macd_hist"] / rows["close"].replace(0.0, np.nan)

    for column in FEATURE_COLUMNS:
        if column not in rows.columns:
            rows[column] = np.nan
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    # Median-fill rather than drop: dropping rows for one missing feature
    # throws away scarce labelled data.
    for column in FEATURE_COLUMNS:
        median = rows[column].median()
        rows[column] = rows[column].fillna(0.0 if pd.isna(median) else median)

    return rows


def temporal_split(
    frame: pd.DataFrame, train_fraction: float = 0.70, time_column: str = "timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split by TIME, oldest for training. Never random.

    Returns (train, test, split_timestamp).
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    ordered = frame.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], utc=True, format="mixed")
    ordered = ordered.sort_values(time_column).reset_index(drop=True)

    cut = int(len(ordered) * train_fraction)
    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    split_at = (
        str(test[time_column].iloc[0]) if not test.empty else str(ordered[time_column].iloc[-1])
    )
    return train, test, split_at


class ConfidenceCalibrator:
    """Estimates P(this signal wins) from features available at signal time."""

    def __init__(self, model_type: str = "logistic", random_state: int = 0) -> None:
        if model_type not in ("logistic", "gradient_boosting"):
            raise ValueError(f"unknown model_type {model_type!r}")
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.features: list[str] = list(FEATURE_COLUMNS)
        self.report: CalibrationReport | None = None

    # -- training ---------------------------------------------------------

    def fit(
        self,
        frame: pd.DataFrame,
        train_fraction: float = 0.70,
        min_rows: int = 40,
    ) -> CalibrationReport:
        """Train on the oldest ``train_fraction`` and evaluate on the newest."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )
        from sklearn.preprocessing import StandardScaler

        prepared = build_features(frame)
        warnings: list[str] = []

        if len(prepared) < min_rows:
            raise NotEnoughData(
                f"{len(prepared)} labelled rows, need at least {min_rows}. "
                "Widen the backtest window or loosen the strategy before "
                "trusting a calibration model."
            )

        train, test, split_at = temporal_split(prepared, train_fraction)
        x_train = train[self.features].to_numpy(dtype="float64")
        y_train = train[TARGET_COLUMN].to_numpy(dtype="int")
        x_test = test[self.features].to_numpy(dtype="float64")
        y_test = test[TARGET_COLUMN].to_numpy(dtype="int")

        if len(np.unique(y_train)) < 2:
            raise NotEnoughData(
                "the training window contains only one outcome class; "
                "a classifier cannot learn anything from it"
            )

        if len(prepared) < 200:
            warnings.append(
                f"only {len(prepared)} labelled rows: coefficients are indicative, "
                "not reliable. Treat the confidence score as experimental."
            )
        if len(test) and len(np.unique(y_test)) < 2:
            warnings.append(
                "the test window contains only one outcome class; AUC is undefined"
            )

        self.scaler = StandardScaler().fit(x_train)
        x_train_s = self.scaler.transform(x_train)
        x_test_s = self.scaler.transform(x_test) if len(x_test) else x_test

        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=2000, random_state=self.random_state, class_weight="balanced"
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=60, max_depth=2, random_state=self.random_state
            )
        self.model.fit(x_train_s, y_train)

        train_accuracy = float(accuracy_score(y_train, self.model.predict(x_train_s)))
        test_accuracy = test_auc = test_brier = baseline_brier = model_log_loss = None

        if len(x_test):
            probabilities = self.model.predict_proba(x_test_s)[:, 1]
            test_accuracy = float(accuracy_score(y_test, self.model.predict(x_test_s)))
            test_brier = float(brier_score_loss(y_test, probabilities))
            # The honest control: always predict the training base rate.
            base_rate = float(y_train.mean())
            baseline_brier = float(
                brier_score_loss(y_test, np.full_like(probabilities, base_rate))
            )
            if len(np.unique(y_test)) > 1:
                test_auc = float(roc_auc_score(y_test, probabilities))
                model_log_loss = float(log_loss(y_test, probabilities, labels=[0, 1]))

        coefficients: dict[str, float] = {}
        if self.model_type == "logistic":
            coefficients = {
                name: float(value)
                for name, value in zip(self.features, self.model.coef_[0])
            }
        else:
            coefficients = {
                name: float(value)
                for name, value in zip(self.features, self.model.feature_importances_)
            }

        self.report = CalibrationReport(
            model_type=self.model_type,
            n_total=len(prepared),
            n_train=len(train),
            n_test=len(test),
            train_positive_rate=float(y_train.mean()),
            test_positive_rate=float(y_test.mean()) if len(y_test) else 0.0,
            split_timestamp=split_at,
            features=list(self.features),
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            test_auc=test_auc,
            test_brier=test_brier,
            baseline_brier=baseline_brier,
            log_loss=model_log_loss,
            coefficients=coefficients,
            warnings=warnings,
        )
        return self.report

    # -- inference --------------------------------------------------------

    def predict_proba(self, features: dict[str, float] | pd.DataFrame) -> float | np.ndarray:
        """P(win) for one signal (dict) or many (frame)."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("calibrator is not trained")

        if isinstance(features, dict):
            vector = np.array(
                [[float(features.get(name, 0.0) or 0.0) for name in self.features]]
            )
            return float(self.model.predict_proba(self.scaler.transform(vector))[0, 1])

        matrix = features[self.features].to_numpy(dtype="float64")
        return self.model.predict_proba(self.scaler.transform(matrix))[:, 1]

    # -- persistence ------------------------------------------------------

    def save_report(self, path: Path | str) -> None:
        if self.report is None:
            raise RuntimeError("nothing to save; train first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report.to_dict(), indent=2), encoding="utf-8")
