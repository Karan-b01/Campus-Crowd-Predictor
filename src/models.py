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
