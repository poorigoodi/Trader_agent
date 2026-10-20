"""Unit tests for the confidence calibrator.

The property that matters most here is the TEMPORAL split. A random split
would let the model learn from the future about the past -- reintroducing, at
the very last step, exactly the look-ahead bias the backtester exists to
prevent. Several tests below exist only to make that impossible to
reintroduce quietly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.confidence.calibration import (
    FEATURE_COLUMNS,
    ConfidenceCalibrator,
    NotEnoughData,
    build_features,
    temporal_split,
)


def dataset(n: int = 300, seed: int = 3, signal_strength: float = 0.0) -> pd.DataFrame:
    """Synthetic labelled signals.

    With ``signal_strength`` at 0 the target is pure noise, so a model that
    reports skill on it is overfitting. Above 0, RSI genuinely predicts the
    outcome, so a working model should find it.
    """
    rng = np.random.default_rng(seed)
    rsi = rng.uniform(30, 70, n)
    logit = signal_strength * (rsi - 50) / 10 - 0.5
    probability = 1 / (1 + np.exp(-logit))
    won = (rng.uniform(0, 1, n) < probability).astype(int)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
            "symbol": "BTC/USDT:USDT",
            "direction": rng.choice(["LONG", "SHORT"], n),
            "trend": rng.choice(["UP", "DOWN", "RANGE"], n),
            "close": rng.uniform(60_000, 80_000, n),
            "rsi": rsi,
            "atr": rng.uniform(50, 300, n),
            "atr_pct_of_price": rng.uniform(0.001, 0.005, n),
            "ema_spread_pct": rng.normal(0, 0.01, n),
            "macd_hist": rng.normal(0, 50, n),
            "bb_percent_b": rng.uniform(0, 1, n),
            "close_vs_vwap_pct": rng.normal(0, 0.01, n),
            "cross_age": rng.integers(0, 4, n),
            "rr": 2.0,
            "won": won,
        }
    )


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------

def test_build_features_keeps_only_labelled_rows():
    frame = dataset(100)
    frame.loc[frame.index[:40], "won"] = np.nan
    assert len(build_features(frame)) == 60


def test_build_features_creates_every_expected_column():
    prepared = build_features(dataset(60))
    for column in FEATURE_COLUMNS:
        assert column in prepared.columns


def test_trend_agrees_encodes_direction_and_structure():
    frame = dataset(20)
    frame["direction"] = "LONG"
    frame["trend"] = "UP"
    assert build_features(frame)["trend_agrees"].eq(1.0).all()

    frame["trend"] = "DOWN"
    assert build_features(frame)["trend_agrees"].eq(0.0).all()


def test_macd_is_normalised_by_price():
    """Otherwise BTC dominates the coefficient purely by being a big number."""
    frame = dataset(40)
    prepared = build_features(frame)
    expected = frame["macd_hist"] / frame["close"]
    assert prepared["macd_hist_norm"].iloc[0] == pytest.approx(expected.iloc[0])


def test_missing_features_are_filled_not_dropped():
    """Labelled rows are scarce; dropping one for a single NaN wastes data."""
    frame = dataset(80)
    frame.loc[frame.index[:20], "rsi"] = np.nan
    prepared = build_features(frame)
    assert len(prepared) == 80
    assert prepared["rsi"].notna().all()


def test_empty_input_gives_empty_output():
    frame = dataset(10)
    frame["won"] = np.nan
    assert build_features(frame).empty


# --------------------------------------------------------------------------
# the temporal split
# --------------------------------------------------------------------------

def test_split_is_ordered_in_time_not_random():
    train, test, _ = temporal_split(dataset(100), 0.70)
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_split_sizes_follow_the_fraction():
    train, test, _ = temporal_split(dataset(100), 0.70)
    assert len(train) == 70 and len(test) == 30


def test_split_reorders_unsorted_input():
    """A shuffled CSV must not produce a leaky split."""
    frame = dataset(100).sample(frac=1.0, random_state=1).reset_index(drop=True)
    train, test, _ = temporal_split(frame, 0.70)
    assert train["timestamp"].is_monotonic_increasing
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_no_row_appears_in_both_halves():
    frame = dataset(100)
    train, test, _ = temporal_split(frame, 0.70)
    assert set(train["timestamp"]) & set(test["timestamp"]) == set()


def test_split_timestamp_marks_the_boundary():
    train, test, split_at = temporal_split(dataset(100), 0.70)
    assert str(test["timestamp"].iloc[0]) == split_at


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_invalid_train_fraction_is_refused(bad):
    with pytest.raises(ValueError):
        temporal_split(dataset(50), bad)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def test_too_few_rows_is_refused_loudly():
    with pytest.raises(NotEnoughData, match="need at least"):
        ConfidenceCalibrator().fit(dataset(20), min_rows=40)


def test_single_class_training_window_is_refused():
    """A classifier cannot learn from one outcome class; say so."""
    frame = dataset(120)
    frame.loc[frame.index[:84], "won"] = 0  # entire training window loses
    with pytest.raises(NotEnoughData, match="one outcome class"):
        ConfidenceCalibrator().fit(frame, train_fraction=0.70)


def test_logistic_model_trains_and_reports():
    report = ConfidenceCalibrator("logistic").fit(dataset(300, signal_strength=1.5))
    assert report.model_type == "logistic"
    assert report.n_train + report.n_test == report.n_total
    assert report.test_auc is not None
    assert set(report.coefficients) == set(FEATURE_COLUMNS)


def test_gradient_boosting_model_trains():
    report = ConfidenceCalibrator("gradient_boosting").fit(
        dataset(300, signal_strength=1.5)
    )
    assert report.model_type == "gradient_boosting"
    assert all(v >= 0 for v in report.coefficients.values())  # importances


def test_unknown_model_type_is_refused():
    with pytest.raises(ValueError, match="unknown model_type"):
        ConfidenceCalibrator("magic")


def test_a_real_signal_is_found():
    """When RSI genuinely predicts the outcome, the model should beat the
    base rate and give RSI a large coefficient."""
    report = ConfidenceCalibrator("logistic").fit(dataset(600, signal_strength=2.5))
    assert report.test_auc > 0.65
    assert abs(report.coefficients["rsi"]) > 0.3


def test_pure_noise_does_not_produce_a_confident_model():
    """The honesty check: on random labels the model must NOT look skilful."""
    report = ConfidenceCalibrator("logistic").fit(dataset(600, signal_strength=0.0))
    assert report.test_auc is None or report.test_auc < 0.65


def test_small_sample_raises_a_warning_in_the_report():
    report = ConfidenceCalibrator().fit(dataset(120, signal_strength=1.5))
    assert any("indicative" in w for w in report.warnings)


def test_baseline_brier_is_computed_for_comparison():
    """Without the always-predict-base-rate control, Brier means nothing."""
    report = ConfidenceCalibrator().fit(dataset(300, signal_strength=1.5))
    assert report.baseline_brier is not None
    assert report.test_brier is not None


def test_beats_baseline_is_false_on_noise():
    report = ConfidenceCalibrator().fit(dataset(400, signal_strength=0.0))
    assert report.beats_baseline is False


def test_report_summary_is_readable():
    summary = ConfidenceCalibrator().fit(dataset(300, signal_strength=1.5)).summary()
    for fragment in ("model", "rows", "split at", "test AUC", "beats baseline"):
        assert fragment in summary


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

def test_predict_before_training_is_refused():
    with pytest.raises(RuntimeError, match="not trained"):
        ConfidenceCalibrator().predict_proba({"rsi": 55.0})


def test_predict_returns_a_probability():
    calibrator = ConfidenceCalibrator()
    calibrator.fit(dataset(300, signal_strength=1.5))
    probability = calibrator.predict_proba({name: 0.0 for name in FEATURE_COLUMNS})
    assert 0.0 <= probability <= 1.0


def test_predict_handles_missing_features_without_raising():
    calibrator = ConfidenceCalibrator()
    calibrator.fit(dataset(300, signal_strength=1.5))
    assert 0.0 <= calibrator.predict_proba({"rsi": 60.0}) <= 1.0


def test_batch_prediction_matches_row_count():
    calibrator = ConfidenceCalibrator()
    calibrator.fit(dataset(300, signal_strength=1.5))
    prepared = build_features(dataset(50, seed=9, signal_strength=1.5))
    probabilities = calibrator.predict_proba(prepared)
    assert len(probabilities) == len(prepared)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_higher_rsi_raises_predicted_probability_when_that_is_the_signal():
    calibrator = ConfidenceCalibrator()
    calibrator.fit(dataset(600, signal_strength=2.5))
    base = {name: 0.0 for name in FEATURE_COLUMNS}
    low = calibrator.predict_proba({**base, "rsi": 35.0})
    high = calibrator.predict_proba({**base, "rsi": 65.0})
    assert high > low


def test_report_is_saved_as_json(tmp_path):
    calibrator = ConfidenceCalibrator()
    calibrator.fit(dataset(300, signal_strength=1.5))
    path = tmp_path / "report.json"
    calibrator.save_report(path)

    import json

    saved = json.loads(path.read_text())
    assert saved["model_type"] == "logistic"
    assert "beats_baseline" in saved


def test_saving_before_training_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="train first"):
        ConfidenceCalibrator().save_report(tmp_path / "x.json")
