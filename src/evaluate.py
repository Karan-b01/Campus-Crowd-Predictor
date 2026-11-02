"""Metrics, chronological cross-validation, experiments, and baselines."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    BASELINE_REPORT_PATH,
    DATETIME_COLUMN,
    EXPERIMENTS_PATH,
    PREDICTION_MINIMUM,
    TARGET_COLUMN,
    TIME_SERIES_CV_GAP,
    TIME_SERIES_CV_SPLITS,
)
from src.data import split_chronologically, write_split_summary
from src.features import build_feature_set, build_model_pipeline
from src.utils import get_logger

LOGGER = get_logger(__name__)
METRIC_NAMES = ("MAE", "RMSE", "R2", "SMAPE")
EXPERIMENT_FIELDS = ("model_name", "scenario", "parameters", "metrics", "timestamp")


@dataclass(frozen=True)
class CrossValidationScores:
    """Per-fold validation metrics and their aggregate summary."""

    fold_scores: pd.DataFrame
    summary: pd.DataFrame


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute count-safe regression metrics.

    Predictions are clipped to the non-negative range because the target is a
    people count. SMAPE is used instead of MAPE: it remains defined when the
    true count is zero, and two zero values contribute zero percentage error.

    Args:
        y_true: True target values.
        y_pred: Model or baseline predictions.

    Returns:
        MAE, RMSE, R2, and SMAPE (as a percentage).

    Raises:
        ValueError: If arrays are empty, have different lengths, contain
            non-finite values, or have fewer than two rows for R2.

    """
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.size == 0 or truth.size != prediction.size:
        raise ValueError("Metric inputs must be non-empty and have matching lengths.")
    if truth.size < 2:
        raise ValueError("At least two rows are required to calculate R2.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must contain only finite numeric values.")
    prediction = np.maximum(prediction, PREDICTION_MINIMUM)

    error = truth - prediction
    absolute_error = np.abs(error)
    denominator = np.abs(truth) + np.abs(prediction)
    smape_terms = np.divide(
        2 * absolute_error,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    total_sum_squares = np.square(truth - truth.mean()).sum()
    r2 = (
        1.0 - np.square(error).sum() / total_sum_squares
        if total_sum_squares > 0
        else float("nan")
    )
    return {
        "MAE": float(absolute_error.mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "R2": float(r2),
        "SMAPE": float(100 * smape_terms.mean()),
    }


def log_experiment(
    model_name: str,
    scenario: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    path: Path = EXPERIMENTS_PATH,
) -> Path:
    """Append one run to the experiment CSV.

    Args:
        model_name: Human-readable baseline or model name.
        scenario: ``forecast`` or ``nowcast``.
        parameters: JSON-serializable estimator or baseline settings.
        metrics: Measured regression metrics.
        path: Optional path for isolated tests or alternate experiment logs.

    Returns:
        The CSV path containing the appended record.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    record = {
        "model_name": model_name,
        "scenario": scenario,
        "parameters": json.dumps(parameters, sort_keys=True, default=str),
        "metrics": json.dumps(metrics, sort_keys=True, allow_nan=True),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", newline="", encoding="utf-8") as experiment_file:
        writer = csv.DictWriter(experiment_file, fieldnames=EXPERIMENT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    LOGGER.info("Logged %s (%s) experiment to %s.", model_name, scenario, path)
    return path


def time_series_cross_validate(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_splits: int = TIME_SERIES_CV_SPLITS,
    gap: int = TIME_SERIES_CV_GAP,
) -> CrossValidationScores:
    """Evaluate an estimator using expanding-window chronological folds.

    ``gap`` is measured in rows and should equal the forecast horizon in rows
    when a multi-step target horizon is used. For the current one-row-ahead
    features, no extra gap is required: every lag is already strictly past-only.
    Each fold gets a fresh clone, so fitted preprocessing is isolated to that
    fold's training segment.

    Args:
        estimator: An sklearn estimator or preprocessing/model Pipeline.
        features: Feature rows already in chronological order.
        target: Target values aligned row-for-row with ``features``.
        n_splits: Number of expanding-window validation folds.
        gap: Rows omitted between a fold's train and validation segments.

    Returns:
        Per-fold metrics and a mean/sample-standard-deviation summary.

    Raises:
        ValueError: If inputs are misaligned or the fold settings are invalid.

    """
    if len(features) != len(target):
        raise ValueError("Features and target must have matching row counts.")
    if n_splits < 2 or gap < 0:
        raise ValueError("n_splits must be at least 2 and gap cannot be negative.")
    if len(features) <= n_splits + gap:
        raise ValueError("Not enough observations for the requested CV splits and gap.")

    y_values = np.asarray(target, dtype=float).reshape(-1)
    if not np.isfinite(y_values).all():
        raise ValueError("Cross-validation target must contain only finite values.")
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows: list[dict[str, float | int]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features), start=1
    ):
        fold_estimator = clone(estimator)
        fold_estimator.fit(features.iloc[train_indices], y_values[train_indices])
        predictions = fold_estimator.predict(features.iloc[validation_indices])
        rows.append(
            {"fold": fold, **compute_metrics(y_values[validation_indices], predictions)}
        )

    fold_scores = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "mean": fold_scores[list(METRIC_NAMES)].mean(),
            "std": fold_scores[list(METRIC_NAMES)].std(ddof=1),
        }
    )
    return CrossValidationScores(fold_scores=fold_scores, summary=summary)


def _append_baseline_result(
    result_rows: list[dict[str, Any]],
    *,
    name: str,
    scenario: str,
    parameters: dict[str, Any],
    truth: pd.Series,
    prediction: np.ndarray,
) -> None:
    """Measure and log one validation baseline run."""
    metrics = compute_metrics(truth, prediction)
    result_rows.append(
        {"scenario": scenario, "model": name, **metrics, "parameters": parameters}
    )
    log_experiment(name, scenario, parameters, metrics)


def _write_baseline_report(
    results: pd.DataFrame,
    *,
    validation: pd.DataFrame,
    missing_pair_count: int,
    missing_week_lag_count: int,
    path: Path,
) -> Path:
    """Write a readable validation-only baseline comparison."""
    validation_start = validation[DATETIME_COLUMN].iloc[0].isoformat()
    validation_end = validation[DATETIME_COLUMN].iloc[-1].isoformat()
    forecast_baseline = results.loc[
        results["scenario"].eq("forecast")
        & results["model"].eq("Training day/hour average"),
        "MAE",
    ].iloc[0]
    nowcast_baseline = results.loc[
        results["scenario"].eq("nowcast") & results["model"].eq("Same time last week"),
        "MAE",
    ].iloc[0]
    table_lines = [
        "| Scenario | Baseline | MAE | RMSE | R2 | SMAPE (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        table_lines.append(
            f"| {row.scenario.title()} | {row.model} | {row.MAE:.3f} | "
            f"{row.RMSE:.3f} | {row.R2:.3f} | {row.SMAPE:.2f} |"
        )
    content = "\n".join(
        [
            "# Validation Baselines",
            "",
            "All scores below are validation-only. Validation covers "
            f"{validation_start} through {validation_end} (UTC). The held-out "
            "test partition was not scored or used for baseline choices.",
            "",
            "## Results",
            "",
            *table_lines,
            "",
            "MAE is the primary metric, so lower is better. **For Forecast, "
            f"the score to beat is the train-only `(day_of_week, hour)` "
            f"average (MAE {forecast_baseline:.3f}). For Nowcast, it is "
            f"same-time-last-week (MAE {nowcast_baseline:.3f}).** The "
            "day/hour average is a strong, honest calendar baseline because "
            "the EDA showed daily and weekday/weekend patterns, and each "
            "pair's value is estimated only from training rows; unseen pairs "
            "fall back to the training mean.",
            "",
            f"For the day/hour baseline, {missing_pair_count:,} validation rows "
            "had a pair not observed in training and used the training-mean "
            "fallback. For the Scenario B same-time-last-week baseline, "
            f"{missing_week_lag_count:,} validation rows lacked a usable lag "
            "and used the training-mean fallback.",
            "",
            "The mean and day/hour baselines are repeated under both scenario "
            "labels because neither uses crowd lags. Same-time-last-week "
            "applies only to Nowcast. Linear Regression applies only to "
            "Forecast and is fitted with train-only preprocessing. All "
            "baseline predictions are clipped at zero before scoring.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    LOGGER.info("Saved baseline comparison to %s.", path)
    return path


def run_baseline_comparison(
    df: pd.DataFrame, *, report_path: Path = BASELINE_REPORT_PATH
) -> pd.DataFrame:
    """Fit simple baselines on train and compare them on validation only.

    Features are constructed only through the validation boundary. This lets
    Nowcast validation rows use earlier observed counts while never reading
    held-out test targets to construct evaluation features.

    Args:
        df: Raw dataset with target and expected columns.
        report_path: Optional output path for the Markdown comparison.

    Returns:
        Validation results with one row per scenario/model combination.

    """
    splits = split_chronologically(df)
    write_split_summary(splits)
    train = splits["train"]
    validation = splits["validation"]
    train_validation = pd.concat([train, validation])
    train_mean = float(train[TARGET_COLUMN].mean())
    truth = validation[TARGET_COLUMN]
    result_rows: list[dict[str, Any]] = []

    mean_prediction = np.full(len(validation), train_mean)
    for scenario in ("forecast", "nowcast"):
        _append_baseline_result(
            result_rows,
            name="Training mean",
            scenario=scenario,
            parameters={"training_mean": train_mean},
            truth=truth,
            prediction=mean_prediction,
        )

    pair_means = train.groupby(["day_of_week", "hour"])[TARGET_COLUMN].mean()
    validation_pairs = pd.MultiIndex.from_frame(validation[["day_of_week", "hour"]])
    pair_prediction = pair_means.reindex(validation_pairs)
    missing_pair = pair_prediction.isna()
    missing_pair_count = int(missing_pair.sum())
    pair_prediction = pair_prediction.fillna(train_mean).to_numpy()
    for scenario in ("forecast", "nowcast"):
        _append_baseline_result(
            result_rows,
            name="Training day/hour average",
            scenario=scenario,
            parameters={
                "group_columns": ["day_of_week", "hour"],
                "unseen_pair_fallback": "training_mean",
                "unseen_pair_count": missing_pair_count,
            },
            truth=truth,
            prediction=pair_prediction,
        )

    features = build_feature_set(train_validation, scenario="nowcast")
    validation_features = features.loc[validation.index]
    week_prediction = validation_features["crowd_lag_1w"].copy()
    missing_week_lag = week_prediction.isna()
    missing_week_lag_count = int(missing_week_lag.sum())
    week_prediction = week_prediction.fillna(train_mean).to_numpy()
    _append_baseline_result(
        result_rows,
        name="Same time last week",
        scenario="nowcast",
        parameters={
            "feature": "crowd_lag_1w",
            "unavailable_lag_fallback": "training_mean",
            "unavailable_lag_count": missing_week_lag_count,
        },
        truth=truth,
        prediction=week_prediction,
    )

    forecast_features = build_feature_set(train_validation, scenario="forecast")
    linear_pipeline = build_model_pipeline(
        LinearRegression(), forecast_features.loc[train.index], scale_numeric=True
    )
    linear_pipeline.fit(
        forecast_features.loc[train.index], train[TARGET_COLUMN].to_numpy()
    )
    linear_prediction = linear_pipeline.predict(forecast_features.loc[validation.index])
    _append_baseline_result(
        result_rows,
        name="Linear Regression",
        scenario="forecast",
        parameters={"scale_numeric": True, "fit_intercept": True},
        truth=truth,
        prediction=linear_prediction,
    )

    results = pd.DataFrame(result_rows)
    _write_baseline_report(
        results,
        validation=validation,
        missing_pair_count=missing_pair_count,
        missing_week_lag_count=missing_week_lag_count,
        path=report_path,
    )
    LOGGER.info("Completed %s validation baseline runs.", len(results))
    return results
