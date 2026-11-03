"""Smoke tests for the Streamlit app utility and prediction helpers."""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime as DateTime

import pytest

from app.utils import (
    classify_crowd_level,
    compute_prediction_drivers,
    find_visiting_recommendation,
    load_cached_artifacts,
    load_cached_eda_data,
    predict_day_curve,
    predict_single,
    predict_weekly_matrix,
)


def test_load_cached_artifacts() -> None:
    """Verify that model pipeline and metadata load successfully."""
    pipeline, metadata, intervals = load_cached_artifacts()
    assert pipeline is not None
    assert isinstance(metadata, dict)
    assert isinstance(intervals, dict)
    assert "lower_residual" in intervals
    assert "upper_residual" in intervals


def test_classify_crowd_level() -> None:
    """Verify crowd level categorization and badge formatting."""
    lbl_low, cls_low, clr_low, _ = classify_crowd_level(15.0)
    assert lbl_low == "Low"
    assert cls_low == "badge-low"
    assert clr_low == "#10B981"

    lbl_med, cls_med, clr_med, _ = classify_crowd_level(30.0)
    assert lbl_med == "Medium"
    assert cls_med == "badge-medium"
    assert clr_med == "#F59E0B"

    lbl_high, cls_high, clr_high, _ = classify_crowd_level(55.0)
    assert lbl_high == "High"
    assert cls_high == "badge-high"
    assert clr_high == "#EF4444"


@pytest.mark.parametrize(
    ("dt", "temp", "holiday", "during_sem", "start_sem"),
    [
        (
            DateTime(2026, 11, 3, 17, 30),
            68.0,
            False,
            True,
            False,
        ),  # Normal weekday peak
        (DateTime(2026, 11, 8, 14, 0), 65.0, False, True, False),  # Weekend (Sunday)
        (
            DateTime(2026, 12, 25, 12, 0),
            45.0,
            True,
            False,
            False,
        ),  # Holiday during break
        (DateTime(2026, 11, 4, 3, 0), 55.0, False, True, False),  # Off-hours overnight
    ],
)
def test_predict_single_scenarios(
    dt: DateTime, temp: float, holiday: bool, during_sem: bool, start_sem: bool
) -> None:
    """Verify single prediction outputs across varied calendar scenarios."""
    res = predict_single(dt, temp, holiday, during_sem, start_sem)
    assert "count" in res
    assert "lower" in res
    assert "upper" in res
    assert "level" in res
    assert res["count"] >= 0.0
    assert res["lower"] >= 0.0
    assert res["lower"] <= res["count"]
    assert res["upper"] >= res["count"]
    assert res["level"] in ("Low", "Medium", "High")


def test_predict_day_curve() -> None:
    """Verify 24-hour day curve generation with 10-minute intervals."""
    test_date = Date(2026, 11, 3)
    df_curve = predict_day_curve(test_date, 68.0, False, True, False, step_minutes=10)
    assert len(df_curve) == 144
    assert (df_curve["count"] >= 0.0).all()
    assert (df_curve["lower"] <= df_curve["count"]).all()
    assert (df_curve["upper"] >= df_curve["count"]).all()


def test_find_visiting_recommendation() -> None:
    """Verify optimal visiting window and peak identification."""
    test_date = Date(2026, 11, 3)
    df_curve = predict_day_curve(test_date, 68.0, False, True, False, step_minutes=10)
    rec = find_visiting_recommendation(df_curve)
    assert "best_window" in rec
    assert "best_avg_crowd" in rec
    assert "peak_window" in rec
    assert "peak_count" in rec
    assert rec["best_avg_crowd"] <= rec["peak_count"]


def test_predict_weekly_matrix() -> None:
    """Verify 24x7 weekly heatmap grid."""
    test_date = Date(2026, 11, 3)
    matrix = predict_weekly_matrix(test_date, 68.0, True, False)
    assert matrix.shape == (24, 7)
    assert list(matrix.columns) == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    assert (matrix.values >= 0.0).all()


def test_compute_prediction_drivers() -> None:
    """Verify local sensitivity deltas and top SHAP ranking."""
    dt = DateTime(2026, 11, 3, 17, 30)
    drivers = compute_prediction_drivers(dt, 68.0, False, True, False)
    assert "current_pred" in drivers
    assert "daily_mean" in drivers
    assert "time_delta" in drivers
    assert "sem_delta" in drivers
    assert "top_shap" in drivers
    assert isinstance(drivers["top_shap"], list)


def test_load_cached_eda_data() -> None:
    """Verify that EDA aggregates are properly formed."""
    eda_data = load_cached_eda_data()
    assert "hourly_split" in eda_data
    assert "dow_summary" in eda_data
    assert "sem_summary" in eda_data
    assert "temp_sample" in eda_data
    assert not eda_data["hourly_split"].empty
