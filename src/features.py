"""Leakage-safe calendar and historical crowd feature builders."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATE_COLUMN,
    EXPECTED_RAW_COLUMNS,
    FEATURE_CALENDAR_LAG_MATCH_TOLERANCE_SECONDS,
    FEATURE_CYCLE_HOURS,
    FEATURE_CYCLE_MONTHS,
    FEATURE_CYCLE_WEEKDAYS,
    FEATURE_DAYS_PER_WEEK,
    FEATURE_HISTORY_DAYS,
    FEATURE_HISTORY_WEEKS,
    FEATURE_LAG_MATCH_TOLERANCE_SECONDS,
    FEATURE_LAG_MINUTES,
    FEATURE_ROLLING_MIN_PERIODS,
    FEATURE_ROLLING_WINDOWS_MINUTES,
    FEATURE_RUSH_HOURS,
    FEATURE_SEASON_BY_MONTH,
    FEATURE_SEMESTER_PHASE_ACTIVE,
    FEATURE_SEMESTER_PHASE_BREAK,
    FEATURE_SEMESTER_PHASE_START,
    FEATURE_TIME_BUCKET_EDGES,
    FEATURE_TIME_BUCKET_LABELS,
    FORECAST_FEATURE_COLUMNS,
    NOWCAST_LAG_FEATURE_COLUMNS,
    SECONDS_PER_MINUTE,
    SOURCE_LOCAL_DATETIME_LENGTH,
    TARGET_COLUMN,
)
from src.utils import get_logger

LOGGER = get_logger(__name__)
SCENARIOS = Literal["forecast", "nowcast"]
_RAW_FEATURE_INPUT_COLUMNS = [
    column for column in EXPECTED_RAW_COLUMNS if column != TARGET_COLUMN
]
_CATEGORICAL_FEATURES = ["time_of_day_bucket", "season", "semester_phase"]


def _validate_feature_input(df: pd.DataFrame, *, require_target: bool) -> None:
    """Check that feature engineering received the columns it needs."""
    missing = [
        column for column in _RAW_FEATURE_INPUT_COLUMNS if column not in df.columns
    ]
    if require_target and TARGET_COLUMN not in df.columns:
        missing.append(TARGET_COLUMN)
    if missing:
        raise ValueError(f"Feature input is missing required columns: {missing}.")
    if df.empty:
        raise ValueError("Feature input must contain at least one row.")


def _local_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Parse the source-local wall clock used for calendar and lag alignment."""
    local_dates = pd.to_datetime(
        df[DATE_COLUMN].astype("string").str.slice(stop=SOURCE_LOCAL_DATETIME_LENGTH),
        errors="coerce",
    )
    if local_dates.isna().any():
        bad_count = int(local_dates.isna().sum())
        raise ValueError(f"Found {bad_count} invalid local date values.")
    index = pd.DatetimeIndex(local_dates)
    if not index.is_unique:
        raise ValueError(
            "Source-local timestamps must be unique to build time-based features."
        )
    return index


def _encode_cycle(values: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode a repeating integer cycle as sine and cosine coordinates."""
    angle = 2 * np.pi * values.to_numpy(dtype=float) / period
    return np.sin(angle), np.cos(angle)


def _nearest_past_values(
    sorted_times: pd.DatetimeIndex,
    sorted_target: pd.Series,
    expected_times: pd.DatetimeIndex,
    tolerance: pd.Timedelta,
) -> np.ndarray:
    """Match shifted timestamps to nearby observations within a fixed tolerance."""
    positions = sorted_times.get_indexer(
        expected_times, method="nearest", tolerance=tolerance
    )
    values = np.full(len(expected_times), np.nan, dtype=float)
    matched = positions >= 0
    values[matched] = sorted_target.to_numpy()[positions[matched]]
    return values


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build Scenario A features from calendar and weather inputs only.

    Cyclical pairs place values at the ends of a repeating scale next to each
    other. For example, 23:00 and 00:00 are one hour apart, even though their
    raw integer codes are far apart. Buckets and the rush-hour flag follow the
    observed EDA pattern: counts are near zero overnight and weekday/weekend
    peaks occur at 18:00 and 17:00 respectively.

    Args:
        df: Raw rows with the expected calendar and weather columns.

    Returns:
        A feature DataFrame in the same row order and with the same index as
        ``df``. The target, raw date string, and raw timestamp are excluded.

    Raises:
        ValueError: If required input columns or local timestamps are invalid.

    """
    _validate_feature_input(df, require_target=False)
    features = pd.DataFrame(index=df.index)

    for source, sine_name, cosine_name, period, offset in (
        ("hour", "hour_sin", "hour_cos", FEATURE_CYCLE_HOURS, 0),
        ("day_of_week", "dow_sin", "dow_cos", FEATURE_CYCLE_WEEKDAYS, 0),
        ("month", "month_sin", "month_cos", FEATURE_CYCLE_MONTHS, 1),
    ):
        values = pd.to_numeric(df[source], errors="raise") - offset
        sine, cosine = _encode_cycle(values, period)
        features[sine_name] = sine
        features[cosine_name] = cosine

    features["hour"] = df["hour"].to_numpy()
    features["day_of_week"] = df["day_of_week"].to_numpy()
    features["month"] = df["month"].to_numpy()
    features["minutes_since_midnight"] = (
        pd.to_numeric(df["timestamp"], errors="raise").to_numpy(dtype=float)
        / SECONDS_PER_MINUTE
    )
    features["time_of_day_bucket"] = pd.cut(
        pd.to_numeric(df["hour"], errors="raise"),
        bins=FEATURE_TIME_BUCKET_EDGES,
        labels=FEATURE_TIME_BUCKET_LABELS,
        ordered=True,
    )
    features["is_rush_hour"] = df["hour"].isin(FEATURE_RUSH_HOURS).astype("int8")
    features["season"] = df["month"].map(FEATURE_SEASON_BY_MONTH).astype("category")

    for column in (
        "is_weekend",
        "is_holiday",
        "is_start_of_semester",
        "is_during_semester",
        "temperature",
    ):
        features[column] = df[column].to_numpy()

    semester_phase = np.select(
        [
            df["is_start_of_semester"].eq(1),
            df["is_during_semester"].eq(1),
        ],
        [FEATURE_SEMESTER_PHASE_START, FEATURE_SEMESTER_PHASE_ACTIVE],
        default=FEATURE_SEMESTER_PHASE_BREAK,
    )
    features["semester_phase"] = pd.Categorical(semester_phase)
    features = features[FORECAST_FEATURE_COLUMNS]
    LOGGER.info("Built %s Forecast features for %s rows.", features.shape[1], len(df))
    return features


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build historical target features for Scenario B (Nowcast) only.

    Short lags choose the nearest observation to the requested local-clock
    time, within the configured one-minute tolerance. This accommodates the
    seconds-level jitter visible in the source cadence without substituting a
    much older reading when an interval is missing. Daily and weekly lags use
    a separate five-minute tolerance around the previous day's or week's local
    clock time because cadence drift accumulates over those longer periods.
    Rolling windows include earlier
    observations in the window and exclude the current row (``closed="left"``).
    Unavailable history stays NaN; the preprocessing imputer later learns
    replacement values from the training partition only.

    Args:
        df: Raw rows with source-local dates and the crowd target.

    Returns:
        Only the nine lag and rolling columns, in the original row order.

    Raises:
        ValueError: If required target/date values or source-local timestamps
            are invalid.

    """
    _validate_feature_input(df, require_target=True)
    times = _local_datetime_index(df)
    target = pd.to_numeric(df[TARGET_COLUMN], errors="raise").to_numpy(dtype=float)
    order = np.argsort(times.asi8, kind="stable")
    sorted_times = times.take(order)
    sorted_target = pd.Series(target[order], index=sorted_times)
    sorted_features = pd.DataFrame(index=sorted_times)
    tolerance = pd.Timedelta(seconds=FEATURE_LAG_MATCH_TOLERANCE_SECONDS)

    for minutes in FEATURE_LAG_MINUTES:
        expected_times = sorted_times - pd.Timedelta(minutes=minutes)
        sorted_features[f"crowd_lag_{minutes}m"] = _nearest_past_values(
            sorted_times, sorted_target, expected_times, tolerance
        )

    for name, day_offset in (
        ("1d", FEATURE_HISTORY_DAYS),
        ("1w", FEATURE_HISTORY_WEEKS * FEATURE_DAYS_PER_WEEK),
    ):
        expected_times = sorted_times - pd.Timedelta(days=day_offset)
        sorted_features[f"crowd_lag_{name}"] = _nearest_past_values(
            sorted_times,
            sorted_target,
            expected_times,
            pd.Timedelta(seconds=FEATURE_CALENDAR_LAG_MATCH_TOLERANCE_SECONDS),
        )

    for label, minutes in FEATURE_ROLLING_WINDOWS_MINUTES.items():
        window = pd.Timedelta(minutes=minutes)
        rolling = sorted_target.rolling(
            window=window,
            closed="left",
            min_periods=FEATURE_ROLLING_MIN_PERIODS,
        )
        sorted_features[f"crowd_rolling_mean_{label}"] = rolling.mean().to_numpy()
        sorted_features[f"crowd_rolling_std_{label}"] = rolling.std().to_numpy()

    inverse_order = np.argsort(order, kind="stable")
    features = sorted_features.iloc[inverse_order].copy()
    features.index = df.index
    features = features[NOWCAST_LAG_FEATURE_COLUMNS]
    LOGGER.info("Built %s Nowcast features for %s rows.", features.shape[1], len(df))
    return features


def build_feature_set(
    df: pd.DataFrame, scenario: SCENARIOS = "forecast"
) -> pd.DataFrame:
    """Build a clearly separated Forecast or Nowcast feature matrix.

    Forecast excludes all target-derived features and works without a target
    column. Nowcast appends only shifted historical features to Forecast.

    Args:
        df: Raw observations in any row order.
        scenario: ``"forecast"`` for known calendar/weather inputs or
            ``"nowcast"`` to additionally include lagged crowd readings.

    Returns:
        The selected feature matrix, preserving input order and index.

    Raises:
        ValueError: If the scenario is unsupported or required inputs are
            missing.

    """
    if scenario not in ("forecast", "nowcast"):
        raise ValueError(
            f"Unsupported scenario {scenario!r}; choose 'forecast' or 'nowcast'."
        )
    calendar_features = build_calendar_features(df)
    if scenario == "forecast":
        return calendar_features
    return pd.concat([calendar_features, add_lag_features(df)], axis="columns")


def build_preprocessor(
    feature_frame: pd.DataFrame, *, scale_numeric: bool = False
) -> ColumnTransformer:
    """Create train-fitted imputation/encoding, with optional numeric scaling.

    Tree models generally split on thresholds and do not benefit from feature
    scaling, so leave ``scale_numeric`` false for them. Linear or distance-based
    models can enable scaling. The returned transformer belongs inside an
    sklearn ``Pipeline`` so its medians, categories, and scales are learned
    only when the pipeline is fit on training data.

    Args:
        feature_frame: A Forecast or Nowcast matrix from ``build_feature_set``.
        scale_numeric: Whether to standardize numeric columns after imputation.

    Returns:
        An unfitted ColumnTransformer ready for a model Pipeline.

    Raises:
        ValueError: If the frame is empty or has no supported feature columns.

    """
    if feature_frame.empty or feature_frame.shape[1] == 0:
        raise ValueError("Cannot build a preprocessor from an empty feature frame.")
    categorical = [
        column
        for column in feature_frame.columns
        if column in _CATEGORICAL_FEATURES
        or isinstance(feature_frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(feature_frame[column].dtype)
        or pd.api.types.is_string_dtype(feature_frame[column].dtype)
    ]
    numeric = [column for column in feature_frame.columns if column not in categorical]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        numeric_steps: list[tuple[str, BaseEstimator]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    estimator: BaseEstimator,
    feature_frame: pd.DataFrame,
    *,
    scale_numeric: bool = False,
) -> Pipeline:
    """Wrap preprocessing and an estimator in one sklearn Pipeline.

    Args:
        estimator: An sklearn-compatible regressor.
        feature_frame: Training feature frame used only to select column types.
        scale_numeric: Whether numeric features should be scaled.

    Returns:
        An unfitted Pipeline whose preprocessing is learned during ``fit``.

    """
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(feature_frame, scale_numeric=scale_numeric),
            ),
            ("model", estimator),
        ]
    )
