"""Tests for the artifact-backed calendar Forecast API."""

import json
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.features import build_feature_set, build_model_pipeline
from src.models import PredictionResult, predict


def test_predict_builds_calendar_row_and_returns_interval(
    monkeypatch,
) -> None:
    """The public API derives calendar inputs and returns bounded counts."""
    raw = pd.DataFrame(
        {
            "number_people": [10, 20, 30],
            "date": [
                "2019-01-07 12:00:00",
                "2019-01-07 12:10:00",
                "2019-01-07 12:20:00",
            ],
            "timestamp": [43200, 43800, 44400],
            "day_of_week": [0, 0, 0],
            "is_weekend": [0, 0, 0],
            "is_holiday": [0, 0, 0],
            "temperature": [65.0, 66.0, 67.0],
            "is_start_of_semester": [0, 0, 0],
            "is_during_semester": [1, 1, 1],
            "month": [1, 1, 1],
            "hour": [12, 12, 12],
        }
    )
    features = build_feature_set(raw, scenario="forecast")
    pipeline = build_model_pipeline(DummyRegressor(strategy="mean"), features)
    pipeline.fit(features, raw["number_people"])
    pipeline_path = Path("forecast.joblib")
    interval_path = Path("interval.json")
    monkeypatch.setattr("src.models.TUNED_BEST_MODEL_PATH", pipeline_path)
    monkeypatch.setattr("src.models.PREDICTION_INTERVAL_PATH", interval_path)
    monkeypatch.setattr("src.models.joblib.load", lambda _: pipeline)
    original_is_file = Path.is_file
    original_open = Path.open
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: path in (pipeline_path, interval_path) or original_is_file(path),
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda path, *args, **kwargs: (
            StringIO(json.dumps({"lower_residual": -5.0, "upper_residual": 7.0}))
            if path == interval_path
            else original_open(path, *args, **kwargs)
        ),
    )

    result = predict(
        datetime=datetime(2019, 1, 7, 18, 30),
        temperature=68.0,
        is_holiday=False,
        is_during_semester=True,
        is_start_of_semester=False,
    )

    assert isinstance(result, PredictionResult)
    assert np.isfinite([result.count, result.lower, result.upper]).all()
    assert result.count >= 0
    assert result.lower == max(0.0, result.count - 5.0)
    assert result.upper == result.count + 7.0


def test_predict_rejects_non_finite_temperature() -> None:
    """Invalid numeric user input is rejected before loading model artifacts."""
    import pytest

    with pytest.raises(ValueError, match="finite number"):
        predict(datetime(2019, 1, 7, 18, 30), np.nan, False, True, False)
