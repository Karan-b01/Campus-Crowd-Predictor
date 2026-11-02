"""Leakage-safe sklearn regression model builders and prediction helper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime as DateTime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

from src.config import (
    DATE_COLUMN,
    MODEL_BOOSTING_ITERATIONS,
    MODEL_BOOSTING_L2_REGULARIZATION,
    MODEL_BOOSTING_LEARNING_RATE,
    MODEL_BOOSTING_MAX_LEAF_NODES,
    MODEL_ELASTICNET_ALPHA,
    MODEL_ELASTICNET_L1_RATIO,
    MODEL_ENSEMBLE_ESTIMATORS,
    MODEL_KNN_NEIGHBORS,
    MODEL_LINEAR_MAX_ITERATIONS,
    MODEL_N_JOBS,
    MODEL_RIDGE_ALPHA,
    MODEL_TREE_MAX_DEPTH,
    MODEL_TREE_MIN_SAMPLES_LEAF,
    MODEL_XGBOOST_COLUMN_SAMPLE,
    MODEL_XGBOOST_MAX_DEPTH,
    MODEL_XGBOOST_SUBSAMPLE,
    PREDICTION_INTERVAL_PATH,
    PREDICTION_MINIMUM,
    RANDOM_SEED,
    TUNED_BEST_MODEL_PATH,
)
from src.features import build_feature_set, build_model_pipeline
from src.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpecification:
    """Estimator and preprocessing settings for one model family."""

    estimator: BaseEstimator
    scale_numeric: bool
    count_aware: bool = False


def get_model_specifications() -> dict[str, ModelSpecification]:
    """Return the fixed, documented model comparison set.

    The HistGradientBoosting Poisson-loss variant is the count-aware option.
    Scikit-learn's default early stopping uses a random internal validation
    split on larger datasets, so both histogram boosting variants explicitly
    disable early stopping to preserve chronological evaluation.

    Returns:
        Named estimators and their numeric-scaling requirements.


    """
    return {
        "Ridge": ModelSpecification(Ridge(alpha=MODEL_RIDGE_ALPHA), scale_numeric=True),
        "ElasticNet": ModelSpecification(
            ElasticNet(
                alpha=MODEL_ELASTICNET_ALPHA,
                l1_ratio=MODEL_ELASTICNET_L1_RATIO,
                max_iter=MODEL_LINEAR_MAX_ITERATIONS,
                random_state=RANDOM_SEED,
            ),
            scale_numeric=True,
        ),
        "KNN": ModelSpecification(
            KNeighborsRegressor(
                n_neighbors=MODEL_KNN_NEIGHBORS,
                weights="distance",
                n_jobs=MODEL_N_JOBS,
            ),
            scale_numeric=True,
        ),
        "Decision Tree": ModelSpecification(
            DecisionTreeRegressor(
                max_depth=MODEL_TREE_MAX_DEPTH,
                min_samples_leaf=MODEL_TREE_MIN_SAMPLES_LEAF,
                random_state=RANDOM_SEED,
            ),
            scale_numeric=False,
        ),
        "Random Forest": ModelSpecification(
            RandomForestRegressor(
                n_estimators=MODEL_ENSEMBLE_ESTIMATORS,
                min_samples_leaf=MODEL_TREE_MIN_SAMPLES_LEAF,
                n_jobs=MODEL_N_JOBS,
                random_state=RANDOM_SEED,
            ),
            scale_numeric=False,
        ),
        "ExtraTrees": ModelSpecification(
            ExtraTreesRegressor(
                n_estimators=MODEL_ENSEMBLE_ESTIMATORS,
                min_samples_leaf=MODEL_TREE_MIN_SAMPLES_LEAF,
                n_jobs=MODEL_N_JOBS,
                random_state=RANDOM_SEED,
            ),
            scale_numeric=False,
        ),
        "HistGradientBoosting": ModelSpecification(
            HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=MODEL_BOOSTING_ITERATIONS,
                learning_rate=MODEL_BOOSTING_LEARNING_RATE,
                max_leaf_nodes=MODEL_BOOSTING_MAX_LEAF_NODES,
                l2_regularization=MODEL_BOOSTING_L2_REGULARIZATION,
                early_stopping=False,
                random_state=RANDOM_SEED,
            ),
            scale_numeric=False,
        ),
        "XGBoost": ModelSpecification(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=MODEL_BOOSTING_ITERATIONS,
                max_depth=MODEL_XGBOOST_MAX_DEPTH,
                learning_rate=MODEL_BOOSTING_LEARNING_RATE,
                subsample=MODEL_XGBOOST_SUBSAMPLE,
                colsample_bytree=MODEL_XGBOOST_COLUMN_SAMPLE,
                tree_method="hist",
                n_jobs=MODEL_N_JOBS,
                random_state=RANDOM_SEED,
                verbosity=0,
            ),
            scale_numeric=False,
        ),
        "HistGradientBoosting Poisson": ModelSpecification(
            HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=MODEL_BOOSTING_ITERATIONS,
                learning_rate=MODEL_BOOSTING_LEARNING_RATE,
                max_leaf_nodes=MODEL_BOOSTING_MAX_LEAF_NODES,
                l2_regularization=MODEL_BOOSTING_L2_REGULARIZATION,
                early_stopping=False,
                random_state=RANDOM_SEED,
            ),
            scale_numeric=False,
            count_aware=True,
        ),
    }


def build_model_zoo(
    feature_frame: pd.DataFrame, model_names: list[str] | None = None
) -> dict[str, Pipeline]:
    """Build the requested model pipelines around train-fitted preprocessing.

    Args:
        feature_frame: Training feature rows used to identify column types.
        model_names: Optional subset of names from the fixed model set.

    Returns:
        A mapping from display name to an unfitted sklearn Pipeline.

    Raises:
        ValueError: If a requested model name is not in the model set.


    """
    specifications = get_model_specifications()
    selected_names = model_names or list(specifications)
    unknown_names = sorted(set(selected_names) - specifications.keys())
    if unknown_names:
        raise ValueError(f"Unknown model names requested: {unknown_names}.")

    pipelines = {
        name: build_model_pipeline(
            specifications[name].estimator,
            feature_frame,
            scale_numeric=specifications[name].scale_numeric,
        )
        for name in selected_names
    }
    LOGGER.info("Built %s sklearn model pipelines.", len(pipelines))
    return pipelines


def predict_features(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Predict non-negative crowd counts with a fitted model Pipeline.

    Args:
        pipeline: A fitted pipeline returned by ``build_model_zoo``.
        features: Feature frame matching the pipeline's training scenario.

    Returns:
        One non-negative prediction per input row.

    Raises:
        sklearn.exceptions.NotFittedError: If the pipeline has not been fit.


    """
    check_is_fitted(pipeline)
    predictions = np.asarray(pipeline.predict(features), dtype=float).reshape(-1)
    return np.maximum(predictions, PREDICTION_MINIMUM)


@dataclass(frozen=True)
class PredictionResult:
    """A single non-negative crowd estimate and its empirical prediction band."""

    count: float
    lower: float
    upper: float


def _prediction_input(
    prediction_datetime: DateTime | pd.Timestamp,
    temperature: float,
    is_holiday: bool,
    is_during_semester: bool,
    is_start_of_semester: bool,
) -> pd.DataFrame:
    """Turn app inputs into the raw calendar row expected by feature code."""
    moment = pd.Timestamp(prediction_datetime)
    if moment.tzinfo is not None:
        # Preserve the wall-clock time entered by the user; gym calendar fields
        # and the dataset's date column are defined in local campus time.
        moment = moment.tz_localize(None)
    hour = moment.hour
    return pd.DataFrame(
        [
            {
                DATE_COLUMN: moment.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": hour * 3600 + moment.minute * 60 + moment.second,
                "day_of_week": moment.day_of_week,
                "is_weekend": int(moment.day_of_week >= 5),
                "is_holiday": int(is_holiday),
                "temperature": float(temperature),
                "is_start_of_semester": int(is_start_of_semester),
                "is_during_semester": int(is_during_semester),
                "month": moment.month,
                "hour": hour,
            }
        ]
    )


def predict(
    datetime: DateTime | pd.Timestamp,
    temperature: float,
    is_holiday: bool,
    is_during_semester: bool,
    is_start_of_semester: bool,
) -> PredictionResult:
    """Load the saved Forecast pipeline and predict with an empirical interval.

    Args:
        datetime: Local campus date and time to forecast.
        temperature: Expected temperature in the dataset's recorded unit.
        is_holiday: Whether the date is marked as a holiday.
        is_during_semester: Whether classes are in session.
        is_start_of_semester: Whether the date marks a semester start.

    Returns:
        A prediction count with lower and upper empirical residual bounds.

    Raises:
        FileNotFoundError: If final trained model or interval artifact is absent.
        ValueError: If inputs are invalid or artifacts contain invalid bounds.

    """
    if not np.isfinite(temperature):
        raise ValueError("Temperature must be a finite number.")
    pipeline_path = Path(TUNED_BEST_MODEL_PATH)
    interval_path = Path(PREDICTION_INTERVAL_PATH)
    if not pipeline_path.is_file() or not interval_path.is_file():
        raise FileNotFoundError(
            "Tuned Forecast artifacts are missing; run `make tune` first."
        )
    pipeline = joblib.load(pipeline_path)
    with interval_path.open(encoding="utf-8") as interval_file:
        residual_bounds: dict[str, Any] = json.load(interval_file)
    features = build_feature_set(
        _prediction_input(
            datetime,
            temperature,
            is_holiday,
            is_during_semester,
            is_start_of_semester,
        ),
        scenario="forecast",
    )
    count = float(predict_features(pipeline, features)[0])
    lower = min(
        count,
        max(PREDICTION_MINIMUM, count + float(residual_bounds["lower_residual"])),
    )
    upper = max(
        count,
        max(PREDICTION_MINIMUM, count + float(residual_bounds["upper_residual"])),
    )
    if lower > upper:
        raise ValueError("Prediction interval artifact has reversed residual bounds.")
    return PredictionResult(count=count, lower=lower, upper=upper)
