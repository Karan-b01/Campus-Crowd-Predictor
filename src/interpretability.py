"""Reusable error, crowd-level, interval, and permutation analyses."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.config import (
    CROWD_LEVEL_LABELS,
    CROWD_LEVEL_QUANTILES,
    EVALUATION_PERMUTATION_REPEATS,
    MODEL_N_JOBS,
    RANDOM_SEED,
    TIME_SERIES_CV_GAP,
    TIME_SERIES_CV_SPLITS,
)
from src.models import predict_features
from src.utils import get_logger

LOGGER = get_logger(__name__)


def fit_crowd_level_thresholds(target: pd.Series) -> dict[str, float]:
    """Compute two crowd cut-points from training-set target terciles.

    Args:
        target: Training-only non-negative crowd counts.

    Returns:
        A dictionary containing the Low/Medium and Medium/High cut-points.

    Raises:
        ValueError: If target is empty or has non-finite values.

    """
    values = np.asarray(target, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Training targets must be non-empty and finite.")
    low_boundary, high_boundary = np.quantile(values, CROWD_LEVEL_QUANTILES)
    thresholds = {
        "low_max": float(low_boundary),
        "medium_max": float(high_boundary),
    }
    LOGGER.info("Derived training-only crowd level cut-points: %s", thresholds)
    return thresholds


def assign_crowd_levels(
    values: pd.Series | np.ndarray, thresholds: Mapping[str, float]
) -> pd.Series:
    """Assign counts to Low, Medium, and High using fixed training cut-points.

    Values equal to a cut-point remain in the lower of the adjacent classes.
    This makes class boundaries explicit and stable for app and report use.

    Args:
        values: Crowd counts to classify.
        thresholds: ``low_max`` and ``medium_max`` values from training data.

    Returns:
        Ordered categorical labels preserving the input index.

    Raises:
        ValueError: If thresholds are invalid or counts are non-finite.

    """
    low_max = float(thresholds["low_max"])
    medium_max = float(thresholds["medium_max"])
    numeric = pd.to_numeric(pd.Series(values), errors="raise").astype(float)
    if low_max > medium_max or not np.isfinite([low_max, medium_max]).all():
        raise ValueError("Crowd-level thresholds must be finite and ordered.")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Crowd counts must be finite values.")
    labels = np.select(
        [numeric <= low_max, numeric <= medium_max],
        CROWD_LEVEL_LABELS[:2],
        default=CROWD_LEVEL_LABELS[2],
    )
    return pd.Series(
        pd.Categorical(labels, categories=CROWD_LEVEL_LABELS, ordered=True),
        index=numeric.index,
        name="crowd_level",
    )


def grouped_error_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    *,
    group_name: str,
) -> pd.DataFrame:
    """Summarize count, MAE, bias, and means for each supplied group.

    Args:
        actual: Observed crowd counts.
        predicted: Non-negative model predictions.
        groups: Group label for each observation.
        group_name: Name to use for the output's group column.

    Returns:
        One row per group, sorted from largest to smallest MAE.

    Raises:
        ValueError: If inputs are empty, non-finite, or have different lengths.

    """
    truth = np.asarray(actual, dtype=float).reshape(-1)
    prediction = np.asarray(predicted, dtype=float).reshape(-1)
    labels = np.asarray(groups).reshape(-1)
    if truth.size == 0 or not (truth.size == prediction.size == labels.size):
        raise ValueError(
            "Actuals, predictions, and groups must have equal nonzero size."
        )
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Actuals and predictions must be finite.")
    frame = pd.DataFrame({group_name: labels, "actual": truth, "predicted": prediction})
    frame["absolute_error"] = np.abs(frame["predicted"] - frame["actual"])
    summary = (
        frame.groupby(group_name, observed=True)
        .agg(
            rows=("actual", "size"),
            MAE=("absolute_error", "mean"),
            mean_actual=("actual", "mean"),
            mean_prediction=("predicted", "mean"),
        )
        .reset_index()
    )
    summary["prediction_bias"] = summary["mean_prediction"] - summary["mean_actual"]
    return summary.sort_values("MAE", ascending=False).reset_index(drop=True)


def crowd_level_accuracy(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    thresholds: Mapping[str, float],
) -> float:
    """Return exact three-level classification accuracy using training cuts."""
    true_levels = assign_crowd_levels(actual, thresholds)
    predicted_levels = assign_crowd_levels(predicted, thresholds)
    return float(np.mean(true_levels.to_numpy() == predicted_levels.to_numpy()))


def interval_coverage(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    lower_residual: float,
    upper_residual: float,
) -> dict[str, float]:
    """Measure empirical coverage and width for residual-shifted intervals."""
    truth = np.asarray(actual, dtype=float).reshape(-1)
    prediction = np.asarray(predicted, dtype=float).reshape(-1)
    if truth.size == 0 or truth.size != prediction.size:
        raise ValueError("Interval inputs must be non-empty and have matching lengths.")
    if lower_residual > upper_residual:
        raise ValueError("Lower residual bound must not exceed the upper bound.")
    lower = np.maximum(0.0, prediction + lower_residual)
    upper = np.maximum(prediction, prediction + upper_residual)
    covered = (truth >= lower) & (truth <= upper)
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(upper - lower)),
        "covered_rows": int(covered.sum()),
        "rows": int(truth.size),
    }


def training_oof_residual_quantiles(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    lower_quantile: float,
    upper_quantile: float,
    n_splits: int = TIME_SERIES_CV_SPLITS,
) -> dict[str, float]:
    """Calibrate empirical residual bounds from chronological training OOF fits.

    Every prediction is made by a clone fit only on earlier training rows.
    This helper does not consume validation or test targets.

    Args:
        pipeline: Unfitted pipeline with the selected fixed hyperparameters.
        features: Chronologically ordered training feature rows.
        target: Training target values.
        lower_quantile: Lower residual quantile, such as 0.10 for an 80% band.
        upper_quantile: Upper residual quantile, such as 0.90 for an 80% band.
        n_splits: Number of expanding folds.

    Returns:
        Lower and upper empirical residuals and number of out-of-fold rows.

    """
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Residual quantiles must be ordered within [0, 1].")
    residuals: list[np.ndarray] = []
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=TIME_SERIES_CV_GAP)
    for fit_indices, validation_indices in splitter.split(features):
        fold_model = clone(pipeline)
        with threadpool_limits(limits=MODEL_N_JOBS):
            fold_model.fit(features.iloc[fit_indices], target.iloc[fit_indices])
            predictions = predict_features(
                fold_model, features.iloc[validation_indices]
            )
        residuals.append(
            target.iloc[validation_indices].to_numpy(dtype=float) - predictions
        )
    combined = np.concatenate(residuals)
    return {
        "lower_residual": float(np.quantile(combined, lower_quantile)),
        "upper_residual": float(np.quantile(combined, upper_quantile)),
        "residual_count": int(combined.size),
    }


def permutation_mae_importance(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_repeats: int = EVALUATION_PERMUTATION_REPEATS,
) -> pd.DataFrame:
    """Estimate validation permutation importance as the increase in MAE.

    A feature matters when shuffling its values damages validation accuracy.
    This evaluates raw engineered columns through the full fitted Pipeline,
    preserving categorical preprocessing and ensuring the held-out partition
    is not used.

    Args:
        pipeline: Fitted sklearn pipeline.
        features: Validation features in original engineered columns.
        target: Validation target values.
        n_repeats: Number of independent shuffles per feature.

    Returns:
        Feature-level mean and standard deviation of MAE increase.

    """
    result = sklearn_permutation_importance(
        pipeline,
        features,
        target,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        n_jobs=MODEL_N_JOBS,
    )
    return (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
