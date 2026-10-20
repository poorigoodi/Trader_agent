"""Train the confidence calibrator on labelled backtest signals.

    python -m evaluation.train_confidence
    python -m evaluation.train_confidence --model gradient_boosting --train-fraction 0.7

The split is TEMPORAL, always. See ``app/confidence/calibration.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from app.confidence.calibration import (
    ConfidenceCalibrator,
    NotEnoughData,
    build_features,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the confidence calibrator.")
    parser.add_argument("--dataset", type=Path, default=Path("data/labelled_signals.csv"))
    parser.add_argument(
        "--model", default="logistic", choices=["logistic", "gradient_boosting"]
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("data/calibration_report.json"))
    parser.add_argument("--compare", action="store_true", help="train both model types")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if not args.dataset.exists():
        logger.error(
            "dataset %s not found. Generate it with:\n"
            "  python -m evaluation.run_backtest --days 180",
            args.dataset,
        )
        return 1

    frame = pd.read_csv(args.dataset)
    prepared = build_features(frame)
    logger.info(
        "%d rows, %d with an outcome, win rate %.1f%%",
        len(frame), len(prepared),
        100 * prepared["won"].mean() if len(prepared) else 0.0,
    )

    models = ["logistic", "gradient_boosting"] if args.compare else [args.model]
    reports = {}

    for model_type in models:
        calibrator = ConfidenceCalibrator(model_type)
        try:
            report = calibrator.fit(
                frame, train_fraction=args.train_fraction, min_rows=args.min_rows
            )
        except NotEnoughData as exc:
            logger.error("%s: %s", model_type, exc)
            continue

        print("=" * 64)
        print(report.summary())
        if report.model_type == "logistic":
            print("\ncoefficients (standardised features, sorted by magnitude):")
            for name, value in sorted(
                report.coefficients.items(), key=lambda kv: -abs(kv[1])
            ):
                print(f"  {name:22s} {value:+.4f}")
        reports[model_type] = report.to_dict()

        if model_type == args.model:
            calibrator.save_report(args.out)

    if not reports:
        logger.error("no model could be trained")
        return 1

    if args.compare:
        combined = args.out.with_name("calibration_comparison.json")
        combined.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\ncomparison written to {combined}")

    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
