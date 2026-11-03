"""Utilities for the Campus Gym Crowd Forecaster Streamlit dashboard.

Provides cached model loading, vectorized day and week predictions,
crowd level categorization, sensitivity driver explanations, and
EDA dataset aggregation helpers.
"""

from __future__ import annotations

import json
from datetime import date as Date
from datetime import datetime as DateTime
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    DATE_COLUMN,
    EVALUATION_SHAP_IMPORTANCE_PATH,
    PREDICTION_INTERVAL_PATH,
    PREDICTION_MINIMUM,
    TUNED_BEST_MODEL_PATH,
    TUNED_MODEL_METADATA_PATH,
)
from src.features import build_feature_set
from src.models import predict_features

# Reference constants
DEFAULT_LOW_THRESHOLD = 21.0
DEFAULT_MEDIUM_THRESHOLD = 43.0
GYM_TYPICAL_PEAK = 55.0
GYM_RECORD_CAPACITY = 85.0
OPEN_HOUR = 6
CLOSE_HOUR = 23


# -----------------------------------------------------------------------------
# Cached Model and Artifact Loaders
# -----------------------------------------------------------------------------


def _load_model_artifacts() -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Load the trained Forecast pipeline and interval metadata from disk."""
    model_path = Path(TUNED_BEST_MODEL_PATH)
    meta_path = Path(TUNED_MODEL_METADATA_PATH)
    interval_path = Path(PREDICTION_INTERVAL_PATH)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Forecast model file missing at {model_path}. Please run `make tune`."
        )
    if not interval_path.is_file():
        raise FileNotFoundError(f"Prediction interval file missing at {interval_path}.")

    pipeline: Pipeline = joblib.load(model_path)

    metadata: dict[str, Any] = {}
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

    with interval_path.open(encoding="utf-8") as f:
        interval_data = json.load(f)

    return pipeline, metadata, interval_data


def load_cached_artifacts() -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Safe wrapper supporting both Streamlit runtime and pure unit test imports."""
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _cached_loader():
            return _load_model_artifacts()

        return _cached_loader()
    except Exception:
        return _load_model_artifacts()


# -----------------------------------------------------------------------------
# Crowd Level & Thresholds
# -----------------------------------------------------------------------------


def get_crowd_thresholds(metadata: dict[str, Any] | None = None) -> tuple[float, float]:
    """Extract Low and Medium threshold boundaries from metadata or fallback."""
    if metadata and "crowd_levels" in metadata:
        levels = metadata["crowd_levels"]
        low_max = float(levels.get("low_max", DEFAULT_LOW_THRESHOLD))
        med_max = float(levels.get("medium_max", DEFAULT_MEDIUM_THRESHOLD))
        return low_max, med_max
    return DEFAULT_LOW_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD


def classify_crowd_level(
    count: float,
    low_max: float = DEFAULT_LOW_THRESHOLD,
    med_max: float = DEFAULT_MEDIUM_THRESHOLD,
) -> tuple[str, str, str, str]:
    """Classify crowd into Low, Medium, High with badge classes and hex codes.

    Returns:
        tuple of (label, badge_class, hex_color, description)

    """
    if count <= low_max:
        return (
            "Low",
            "badge-low",
            "#10B981",
            "Plenty of open equipment and quiet training space.",
        )
    if count <= med_max:
        return (
            "Medium",
            "badge-medium",
            "#F59E0B",
            "Moderate activity; popular stations may have brief waits.",
        )
    return (
        "High",
        "badge-high",
        "#EF4444",
        "Peak crowding; expect waiting times for benches and squat racks.",
    )


# -----------------------------------------------------------------------------
# Vectorized Batch Prediction Helpers
# -----------------------------------------------------------------------------


def make_calendar_input_frame(
    datetimes: list[DateTime | pd.Timestamp],
    temperatures: list[float] | float,
    is_holiday: bool,
    is_during_semester: bool,
    is_start_of_semester: bool,
) -> pd.DataFrame:
    """Build raw calendar DataFrame matching features.py expectations."""
    if isinstance(temperatures, (int, float)):
        temp_list = [float(temperatures)] * len(datetimes)
    else:
        temp_list = [float(t) for t in temperatures]

    records = []
    for dt, temp in zip(datetimes, temp_list, strict=True):
        moment = pd.Timestamp(dt)
        if moment.tzinfo is not None:
            moment = moment.tz_localize(None)
        hour = moment.hour
        records.append(
            {
                DATE_COLUMN: moment.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": hour * 3600 + moment.minute * 60 + moment.second,
                "day_of_week": moment.day_of_week,
                "is_weekend": int(moment.day_of_week >= 5),
                "is_holiday": int(is_holiday),
                "temperature": float(temp),
                "is_start_of_semester": int(is_start_of_semester),
                "is_during_semester": int(is_during_semester),
                "month": moment.month,
                "hour": hour,
            }
        )
    return pd.DataFrame(records)


def predict_single(
    dt: DateTime | pd.Timestamp,
    temperature: float,
    is_holiday: bool,
    is_during_semester: bool,
    is_start_of_semester: bool,
) -> dict[str, Any]:
    """Generate prediction and empirical bounds for a single moment."""
    pipeline, metadata, intervals = load_cached_artifacts()
    low_max, med_max = get_crowd_thresholds(metadata)

    df_input = make_calendar_input_frame(
        [dt], temperature, is_holiday, is_during_semester, is_start_of_semester
    )
    features = build_feature_set(df_input, scenario="forecast")
    pred_count = float(predict_features(pipeline, features)[0])

    lower_res = float(intervals.get("lower_residual", -25.0))
    upper_res = float(intervals.get("upper_residual", 10.0))

    lower_bound = min(pred_count, max(PREDICTION_MINIMUM, pred_count + lower_res))
    upper_bound = max(pred_count, max(PREDICTION_MINIMUM, pred_count + upper_res))

    label, badge_class, color, desc = classify_crowd_level(pred_count, low_max, med_max)

    return {
        "datetime": dt,
        "count": round(pred_count, 1),
        "lower": round(lower_bound, 1),
        "upper": round(upper_bound, 1),
        "level": label,
        "badge_class": badge_class,
        "color": color,
        "description": desc,
        "pct_of_peak": min(100.0, round((pred_count / GYM_TYPICAL_PEAK) * 100, 1)),
    }


def predict_day_curve(
    selected_date: Date | DateTime,
    temperature: float,
    is_holiday: bool,
    is_during_semester: bool,
    is_start_of_semester: bool,
    step_minutes: int = 10,
) -> pd.DataFrame:
    """Predict crowd across every interval of the selected day.

    Returns DataFrame with:
        time, time_str, count, lower, upper, crowd_level, color
    """
    pipeline, metadata, intervals = load_cached_artifacts()
    low_max, med_max = get_crowd_thresholds(metadata)

    base_dt = DateTime(selected_date.year, selected_date.month, selected_date.day, 0, 0)
    minutes_in_day = 24 * 60
    timestamps = [
        base_dt + timedelta(minutes=m) for m in range(0, minutes_in_day, step_minutes)
    ]

    df_input = make_calendar_input_frame(
        timestamps, temperature, is_holiday, is_during_semester, is_start_of_semester
    )
    features = build_feature_set(df_input, scenario="forecast")
    preds = predict_features(pipeline, features)

    lower_res = float(intervals.get("lower_residual", -25.0))
    upper_res = float(intervals.get("upper_residual", 10.0))

    results = []
    for dt, count in zip(timestamps, preds, strict=True):
        count_val = float(count)
        low = min(count_val, max(PREDICTION_MINIMUM, count_val + lower_res))
        high = max(count_val, max(PREDICTION_MINIMUM, count_val + upper_res))
        lbl, cls, clr, _ = classify_crowd_level(count_val, low_max, med_max)
        results.append(
            {
                "datetime": dt,
                "time": dt.time(),
                "time_str": dt.strftime("%H:%M"),
                "time_ampm": dt.strftime("%I:%M %p").lstrip("0"),
                "hour": dt.hour + dt.minute / 60.0,
                "count": round(count_val, 1),
                "lower": round(low, 1),
                "upper": round(high, 1),
                "level": lbl,
                "badge_class": cls,
                "color": clr,
            }
        )

    return pd.DataFrame(results)


def find_visiting_recommendation(day_df: pd.DataFrame) -> dict[str, Any]:
    """Find the quietest visiting window and peak rush hour during facility hours."""
    # Filter for standard operating hours (e.g. 06:00 to 22:00)
    operating_df = day_df[
        (day_df["hour"] >= OPEN_HOUR) & (day_df["hour"] <= CLOSE_HOUR)
    ].copy()
    if operating_df.empty:
        operating_df = day_df.copy()

    # Rolling window of 90 minutes (9 ten-minute steps)
    window_steps = 9
    rolling_counts = (
        operating_df["count"]
        .rolling(window=window_steps, min_periods=window_steps)
        .mean()
    )

    if rolling_counts.isna().all():
        min_idx = operating_df["count"].idxmin()
        best_start = operating_df.loc[min_idx, "time_ampm"]
        best_end = (
            (operating_df.loc[min_idx, "datetime"] + timedelta(hours=1.5))
            .strftime("%I:%M %p")
            .lstrip("0")
        )
        best_avg = operating_df.loc[min_idx, "count"]
    else:
        best_end_idx = rolling_counts.idxmin()
        best_start_idx = best_end_idx - (window_steps - 1)
        best_start = operating_df.loc[best_start_idx, "time_ampm"]
        best_end = operating_df.loc[best_end_idx, "time_ampm"]
        best_avg = round(float(rolling_counts.loc[best_end_idx]), 1)
        best_start_hour = operating_df.loc[best_start_idx, "hour"]
        best_end_hour = operating_df.loc[best_end_idx, "hour"]

    # Peak rush
    max_idx = operating_df["count"].idxmax()
    peak_time = operating_df.loc[max_idx, "time_ampm"]
    peak_count = round(float(operating_df.loc[max_idx, "count"]), 1)
    peak_start = (
        (operating_df.loc[max_idx, "datetime"] - timedelta(hours=1))
        .strftime("%I:%M %p")
        .lstrip("0")
    )
    peak_end = (
        (operating_df.loc[max_idx, "datetime"] + timedelta(hours=1))
        .strftime("%I:%M %p")
        .lstrip("0")
    )

    return {
        "best_window": f"{best_start} – {best_end}",
        "best_avg_crowd": best_avg,
        "best_start_hour": locals().get("best_start_hour", 14.0),
        "best_end_hour": locals().get("best_end_hour", 15.5),
        "peak_window": f"{peak_start} – {peak_end}",
        "peak_time": peak_time,
        "peak_count": peak_count,
    }


def predict_weekly_matrix(
    reference_date: Date | DateTime,
    temperature: float,
    is_during_semester: bool,
    is_start_of_semester: bool,
) -> pd.DataFrame:
    """Predict crowd hourly for Monday through Sunday of the selected week."""
    pipeline, _, _ = load_cached_artifacts()

    # Find the Monday of this week
    ref_dt = pd.Timestamp(reference_date)
    monday = ref_dt - timedelta(days=ref_dt.day_of_week)

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    records = []

    for day_idx in range(7):
        current_day = monday + timedelta(days=day_idx)
        for hour in range(24):
            dt = DateTime(current_day.year, current_day.month, current_day.day, hour, 0)
            records.append((day_names[day_idx], hour, dt))

    df_input = make_calendar_input_frame(
        [r[2] for r in records],
        temperature,
        is_holiday=False,
        is_during_semester=is_during_semester,
        is_start_of_semester=is_start_of_semester,
    )
    features = build_feature_set(df_input, scenario="forecast")
    preds = predict_features(pipeline, features)

    result_df = pd.DataFrame(
        {
            "Day": [r[0] for r in records],
            "Hour": [r[1] for r in records],
            "Count": np.round(preds, 1),
        }
    )

    pivot = result_df.pivot(index="Hour", columns="Day", values="Count")
    pivot = pivot[day_names]
    return pivot


def compute_prediction_drivers(
    dt: DateTime | pd.Timestamp,
    temperature: float,
    is_holiday: bool,
    is_during_semester: bool,
    is_start_of_semester: bool,
) -> dict[str, Any]:
    """Compute local factor sensitivity and global SHAP drivers for explanation."""
    pipeline, metadata, _ = load_cached_artifacts()

    # Current prediction
    curr_input = make_calendar_input_frame(
        [dt], temperature, is_holiday, is_during_semester, is_start_of_semester
    )
    curr_features = build_feature_set(curr_input, scenario="forecast")
    curr_pred = float(predict_features(pipeline, curr_features)[0])

    # 1. Daily average benchmark on this exact day
    day_curve = predict_day_curve(
        dt.date() if hasattr(dt, "date") else dt,
        temperature,
        is_holiday,
        is_during_semester,
        is_start_of_semester,
        step_minutes=30,
    )
    daily_mean = float(day_curve["count"].mean())
    time_delta = curr_pred - daily_mean

    # 2. Semester effect (compare with outside semester)
    sem_off_input = make_calendar_input_frame(
        [dt],
        temperature,
        is_holiday,
        is_during_semester=False,
        is_start_of_semester=False,
    )
    sem_off_pred = float(
        predict_features(
            pipeline, build_feature_set(sem_off_input, scenario="forecast")
        )[0]
    )
    sem_delta = curr_pred - sem_off_pred if is_during_semester else 0.0

    # 3. Temperature sensitivity (compare with 65°F standard reference)
    temp_ref_input = make_calendar_input_frame(
        [dt], 65.0, is_holiday, is_during_semester, is_start_of_semester
    )
    temp_ref_pred = float(
        predict_features(
            pipeline, build_feature_set(temp_ref_input, scenario="forecast")
        )[0]
    )
    temp_delta = curr_pred - temp_ref_pred

    # 4. Day of week effect vs Sunday baseline
    sunday_dt = pd.Timestamp(dt) + timedelta(days=(6 - dt.weekday()))
    dow_ref_input = make_calendar_input_frame(
        [sunday_dt], temperature, is_holiday, is_during_semester, is_start_of_semester
    )
    dow_ref_pred = float(
        predict_features(
            pipeline, build_feature_set(dow_ref_input, scenario="forecast")
        )[0]
    )
    dow_delta = curr_pred - dow_ref_pred if dt.weekday() < 5 else 0.0

    # Load top global SHAP features if file exists
    shap_path = Path(EVALUATION_SHAP_IMPORTANCE_PATH)
    top_shap = []
    if shap_path.is_file():
        try:
            shap_df = pd.read_csv(shap_path).head(5)
            # Make feature names human-friendly
            name_map = {
                "numeric__minutes_since_midnight": "Time of Day (Minutes into day)",
                "numeric__hour": "Clock Hour",
                "numeric__is_during_semester": "Academic Semester Active",
                "categorical__semester_phase_outside_semester": "Break / Vacation Period",
                "numeric__hour_sin": "Hourly Harmonic Cycle (Sin)",
                "numeric__month_sin": "Seasonal Harmonic Cycle",
                "numeric__day_of_week": "Day of the Week",
                "numeric__dow_sin": "Weekly Cycle (Sin)",
                "categorical__time_of_day_bucket_evening": "Evening Rush Window",
                "numeric__temperature": "Outdoor Temperature",
                "numeric__is_weekend": "Weekend Indicator",
            }
            for _, row in shap_df.iterrows():
                feat_raw = str(row["feature"])
                clean_name = name_map.get(
                    feat_raw,
                    feat_raw.replace("numeric__", "")
                    .replace("categorical__", "")
                    .replace("_", " ")
                    .title(),
                )
                top_shap.append(
                    {
                        "feature": clean_name,
                        "importance": round(float(row["mean_absolute_shap"]), 2),
                    }
                )
        except Exception:
            pass

    return {
        "current_pred": round(curr_pred, 1),
        "daily_mean": round(daily_mean, 1),
        "time_delta": round(time_delta, 1),
        "sem_delta": round(sem_delta, 1),
        "temp_delta": round(temp_delta, 1),
        "dow_delta": round(dow_delta, 1),
        "top_shap": top_shap,
    }


# -----------------------------------------------------------------------------
# EDA & Evaluation Data Loaders
# -----------------------------------------------------------------------------


def _load_eda_aggregates():
    """Load and aggregate patterns from raw dataset for instant visualization."""
    from src.data import load_raw

    df = load_raw()

    # 1. Hourly weekday vs weekend
    hourly_split = (
        df.groupby(["hour", "is_weekend"])["number_people"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # 2. Day of week profile
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_summary = (
        df.groupby("day_of_week")["number_people"]
        .agg(["mean", "median", lambda x: np.percentile(x, 75) - np.percentile(x, 25)])
        .reset_index()
    )
    dow_summary.columns = ["day_of_week", "mean", "median", "iqr"]
    dow_summary["day_name"] = dow_summary["day_of_week"].map(lambda idx: dow_names[idx])

    # 3. Semester comparison
    sem_summary = (
        df.groupby("is_during_semester")["number_people"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    # 4. Temperature scatter sample (deterministic 600 rows for fast Plotly rendering)
    sample_df = df.sample(n=min(800, len(df)), random_state=42)[
        ["temperature", "number_people", "hour", "is_weekend", "month"]
    ].copy()

    return {
        "hourly_split": hourly_split,
        "dow_summary": dow_summary,
        "sem_summary": sem_summary,
        "temp_sample": sample_df,
    }


def load_cached_eda_data():
    """Safely return cached EDA aggregates."""
    try:
        import streamlit as st

        @st.cache_data(show_spinner=False)
        def _cached():
            return _load_eda_aggregates()

        return _cached()
    except Exception:
        return _load_eda_aggregates()
