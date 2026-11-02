"""Tests for calendar, lag, and preprocessing features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    EXPECTED_RAW_COLUMNS,
    FEATURE_CYCLE_HOURS,
    FEATURE_CYCLE_MONTHS,
    FEATURE_CYCLE_WEEKDAYS,
    NOWCAST_LAG_FEATURE_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from src.features import (
    add_lag_features,
    build_calendar_features,
    build_feature_set,
    build_model_pipeline,
    build_preprocessor,
)


def _synthetic_rows(
    times: pd.DatetimeIndex, *, index: pd.Index | None = None
) -> pd.DataFrame:
    """Create realistic raw-schema rows at supplied source-local times."""
    if index is None:
        index = pd.RangeIndex(len(times))
    seconds = times.hour * 3_600 + times.minute * 60 + times.second
    day_of_week = times.day_of_week
    return pd.DataFrame(
        {
            "number_people": np.arange(len(times), dtype=int),
            "date": [f"{time:%Y-%m-%d %H:%M:%S}-07:00" for time in times],
            "timestamp": seconds,
            "day_of_week": day_of_week,
            "is_weekend": (day_of_week >= 5).astype(int),
            "is_holiday": np.zeros(len(times), dtype=int),
            "temperature": np.linspace(50, 60, len(times)),
            "is_start_of_semester": np.zeros(len(times), dtype=int),
            "is_during_semester": np.ones(len(times), dtype=int),
            "month": times.month,
            "hour": times.hour,
        },
        columns=EXPECTED_RAW_COLUMNS,
        index=index,
    )


@pytest.fixture
def ten_minute_rows() -> pd.DataFrame:
    """Return one day of observations at ten-minute intervals."""
    times = pd.date_range("2024-01-08 06:00:00", periods=36, freq="10min")
    return _synthetic_rows(times)


def test_forecast_shape_and_no_target_dependency(
    ten_minute_rows: pd.DataFrame,
) -> None:
    """Forecast works without the target and returns calendar features only."""
    without_target = ten_minute_rows.drop(columns=TARGET_COLUMN)
    features = build_feature_set(without_target, scenario="forecast")

    assert features.shape == (len(ten_minute_rows), 19)
    assert TARGET_COLUMN not in features.columns
    assert not any("crowd_" in column for column in features.columns)
    changed_target = ten_minute_rows.copy()
    changed_target[TARGET_COLUMN] += 10_000
    pd.testing.assert_frame_equal(
        build_feature_set(changed_target, scenario="forecast"),
        build_feature_set(ten_minute_rows, scenario="forecast"),
    )
    nowcast = build_feature_set(ten_minute_rows, scenario="nowcast")
    assert nowcast.shape == (
        len(ten_minute_rows),
        features.shape[1] + len(NOWCAST_LAG_FEATURE_COLUMNS),
    )
    assert TARGET_COLUMN not in nowcast.columns


def test_cyclical_encodings_stay_in_unit_range(
    ten_minute_rows: pd.DataFrame,
) -> None:
    """Sine/cosine pairs lie in [-1, 1] and month starts at cycle zero."""
    features = build_calendar_features(ten_minute_rows)
    for sine_name, cosine_name in (
        ("hour_sin", "hour_cos"),
        ("dow_sin", "dow_cos"),
        ("month_sin", "month_cos"),
    ):
        assert features[sine_name].between(-1, 1).all()
        assert features[cosine_name].between(-1, 1).all()
    assert features.loc[0, "month_sin"] == pytest.approx(0)
    assert features.loc[0, "month_cos"] == pytest.approx(1)
    assert FEATURE_CYCLE_HOURS == 24
    assert FEATURE_CYCLE_WEEKDAYS == 7
    assert FEATURE_CYCLE_MONTHS == 12


def test_time_buckets_and_eda_rush_hours(ten_minute_rows: pd.DataFrame) -> None:
    """Overnight and evening buckets follow the observed EDA patterns."""
    rows = ten_minute_rows.copy()
    rows.loc[12, "hour"] = 18
    features = build_calendar_features(rows)

    assert features.loc[0, "time_of_day_bucket"] == "morning"
    assert features.loc[12, "time_of_day_bucket"] == "evening"
    assert features.loc[12, "is_rush_hour"] == 1
    assert features.loc[0, "is_rush_hour"] == 0


def test_lags_and_rolling_windows_use_only_earlier_rows(
    ten_minute_rows: pd.DataFrame,
) -> None:
    """Expected exact earlier readings and rolling statistics are produced."""
    features = add_lag_features(ten_minute_rows)
    row = 6

    assert features.loc[row, "crowd_lag_10m"] == 5
    assert features.loc[row, "crowd_lag_30m"] == 3
    assert features.loc[row, "crowd_lag_60m"] == 0
    assert features.loc[row, "crowd_rolling_mean_1h"] == pytest.approx(2.5)
    assert features.loc[row, "crowd_rolling_std_1h"] == pytest.approx(
        pd.Series(range(6), dtype=float).std()
    )
    assert features.loc[row, "crowd_rolling_mean_3h"] == pytest.approx(2.5)
    assert features.loc[0, NOWCAST_LAG_FEATURE_COLUMNS].isna().all()


def test_day_and_week_lags_match_same_local_clock_time() -> None:
    """Daily and weekly lags tolerate small sampling-time jitter."""
    times = pd.date_range("2024-01-01 12:00:00", periods=8, freq="D")
    times = times + pd.to_timedelta([0, 0, 0, 0, 0, 0, 15, 0], unit="s")
    rows = _synthetic_rows(times)
    features = add_lag_features(rows)

    assert features.loc[7, "crowd_lag_1d"] == 6
    assert features.loc[7, "crowd_lag_1w"] == 0


def test_missing_intervals_remain_nan_in_lag_features() -> None:
    """A missing ten-minute match is not replaced by a stale crowd reading."""
    times = pd.DatetimeIndex(
        ["2024-01-08 06:00:00", "2024-01-08 06:10:00", "2024-01-08 06:40:00"]
    )
    features = add_lag_features(_synthetic_rows(times))

    assert pd.isna(features.loc[2, "crowd_lag_10m"])


def test_changing_a_future_target_cannot_change_earlier_features(
    ten_minute_rows: pd.DataFrame,
) -> None:
    """Changing target at t affects later lags only, never features through t."""
    changed = ten_minute_rows.copy()
    changed.loc[12, TARGET_COLUMN] = 999_999

    original_features = build_feature_set(ten_minute_rows, scenario="nowcast")
    changed_features = build_feature_set(changed, scenario="nowcast")
    pd.testing.assert_frame_equal(
        original_features.iloc[:13], changed_features.iloc[:13]
    )
    assert (
        original_features.loc[13, "crowd_lag_10m"]
        != changed_features.loc[13, "crowd_lag_10m"]
    )


def test_features_restore_original_order_and_index() -> None:
    """Chronological lag calculations preserve caller row order and labels."""
    times = pd.date_range("2024-01-08 06:00:00", periods=8, freq="10min")
    rows = _synthetic_rows(times, index=pd.Index(list("abcdefgh")))
    shuffled = rows.iloc[::-1]
    features = build_feature_set(shuffled, scenario="nowcast")

    assert features.index.equals(shuffled.index)
    assert features.loc["b", "crowd_lag_10m"] == 0


def test_preprocessor_imputes_lag_nans_and_scales_only_when_requested(
    ten_minute_rows: pd.DataFrame,
) -> None:
    """Preprocessing handles early missing lags and optional scaling cleanly."""
    features = build_feature_set(ten_minute_rows, scenario="nowcast")
    unscaled = build_preprocessor(features)
    scaled_pipeline = build_model_pipeline(
        DecisionTreeRegressor(random_state=RANDOM_SEED),
        features,
        scale_numeric=True,
    )
    transformed = unscaled.fit_transform(features.iloc[12:])
    fitted = scaled_pipeline.fit(
        features.iloc[12:], ten_minute_rows[TARGET_COLUMN].iloc[12:]
    )
    predictions = fitted.predict(features.iloc[:12])

    assert not np.isnan(transformed).any()
    assert len(predictions) == 12
    assert (
        "scaler"
        in scaled_pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps
    )


def test_invalid_scenario_is_rejected(ten_minute_rows: pd.DataFrame) -> None:
    """Callers cannot silently fall into an unintended feature scenario."""
    with pytest.raises(ValueError, match="choose 'forecast' or 'nowcast'"):
        build_feature_set(ten_minute_rows, scenario="unknown")  # type: ignore[arg-type]
