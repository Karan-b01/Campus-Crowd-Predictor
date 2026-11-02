"""Tests for metrics, experiment logging, and temporal cross-validation."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import src.evaluate as evaluate_module
from src.config import EXPECTED_RAW_COLUMNS, TARGET_COLUMN, TEST_TEMP_DIR
from src.evaluate import compute_metrics, log_experiment, time_series_cross_validate


def test_metrics_are_zero_safe_and_clip_negative_count_predictions() -> None:
    """SMAPE handles zero targets and the reported count prediction is clipped."""
    metrics = compute_metrics([0, 2, 4], [0, -1, 5])

    assert metrics["MAE"] == pytest.approx(1.0)
    assert metrics["RMSE"] == pytest.approx((5 / 3) ** 0.5)
    assert metrics["SMAPE"] == pytest.approx(100 * (0 + 2 + 2 / 9) / 3)
    assert metrics["R2"] == pytest.approx(1 - 5 / 8)


def test_metrics_reject_misaligned_inputs() -> None:
    """Mismatched target and prediction lengths produce a clear error."""
    with pytest.raises(ValueError, match="matching lengths"):
        compute_metrics([1, 2], [1])


def test_experiment_logger_appends_runs() -> None:
    """Each call appends a timestamped row instead of replacing prior runs."""
    TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    file_descriptor, file_name = tempfile.mkstemp(suffix=".csv", dir=TEST_TEMP_DIR)
    os.close(file_descriptor)
    path = Path(file_name)
    try:
        log_experiment("Mean", "forecast", {"value": 3}, {"MAE": 1.5}, path)
        log_experiment("Mean", "nowcast", {"value": 3}, {"MAE": 1.2}, path)
        rows = pd.read_csv(path)

        assert rows["model_name"].tolist() == ["Mean", "Mean"]
        assert rows["scenario"].tolist() == ["forecast", "nowcast"]
        assert rows["timestamp"].notna().all()
        assert rows["metrics"].str.contains("MAE").all()
    finally:
        path.unlink(missing_ok=True)


def test_time_series_cross_validation_reports_folds_and_summary() -> None:
    """CV uses five forward folds, a row gap, and returns mean/std metrics."""
    features = pd.DataFrame({"step": range(60)})
    target = pd.Series(range(60), dtype=float)
    result = time_series_cross_validate(
        LinearRegression(), features, target, n_splits=5, gap=2
    )

    assert len(result.fold_scores) == 5
    assert result.fold_scores["fold"].tolist() == [1, 2, 3, 4, 5]
    assert set(result.summary.columns) == {"mean", "std"}
    assert set(result.summary.index) == {"MAE", "RMSE", "R2", "SMAPE"}
    assert result.summary.loc["MAE", "mean"] == pytest.approx(0)


def test_baseline_validation_scores_do_not_depend_on_test_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing held-out targets cannot affect validation baseline results."""
    TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    file_descriptor, file_name = tempfile.mkstemp(suffix=".md", dir=TEST_TEMP_DIR)
    os.close(file_descriptor)
    report_path = Path(file_name)
    monkeypatch.setattr(evaluate_module, "log_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluate_module,
        "write_split_summary",
        lambda splits: report_path,
    )

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=200, freq="10min")
    rows = []
    for index, timestamp in enumerate(timestamps):
        rows.append(
            {
                "number_people": index % 45,
                "date": f"{timestamp:%Y-%m-%d %H:%M:%S}+00:00",
                "timestamp": timestamp.hour * 3_600 + timestamp.minute * 60,
                "day_of_week": timestamp.day_of_week,
                "is_weekend": int(timestamp.day_of_week >= 5),
                "is_holiday": 0,
                "temperature": 50.0 + timestamp.month,
                "is_start_of_semester": 0,
                "is_during_semester": 1,
                "month": timestamp.month,
                "hour": timestamp.hour,
            }
        )
    data = pd.DataFrame(rows, columns=EXPECTED_RAW_COLUMNS)
    changed_test = data.copy()
    changed_test.loc[170:, TARGET_COLUMN] = 1_000_000
    try:
        original_results = evaluate_module.run_baseline_comparison(
            data, report_path=report_path
        )
        changed_results = evaluate_module.run_baseline_comparison(
            changed_test, report_path=report_path
        )
    finally:
        report_path.unlink(missing_ok=True)

    pd.testing.assert_frame_equal(
        original_results[["scenario", "model", "MAE", "RMSE", "R2", "SMAPE"]],
        changed_results[["scenario", "model", "MAE", "RMSE", "R2", "SMAPE"]],
    )
