"""Unit tests for reusable model diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.interpretability import (
    assign_crowd_levels,
    crowd_level_accuracy,
    fit_crowd_level_thresholds,
    grouped_error_metrics,
    interval_coverage,
)


def test_crowd_thresholds_are_training_terciles_and_labels_are_stable() -> None:
    """Level boundaries are learned once then applied consistently."""
    thresholds = fit_crowd_level_thresholds(pd.Series(range(1, 10)))

    assert thresholds == {
        "low_max": 3.6666666666666665,
        "medium_max": 6.333333333333333,
    }
    assert assign_crowd_levels(pd.Series([1, 4, 9]), thresholds).tolist() == [
        "Low",
        "Medium",
        "High",
    ]
    assert crowd_level_accuracy([1, 5, 9], [2, 4, 7], thresholds) == pytest.approx(1.0)


def test_grouped_error_metrics_exposes_count_mae_and_prediction_bias() -> None:
    """Group summaries preserve support and error direction."""
    summary = grouped_error_metrics(
        actual=[1, 3, 10],
        predicted=[2, 1, 8],
        groups=["a", "a", "b"],
        group_name="segment",
    ).set_index("segment")

    assert summary.loc["a", "rows"] == 2
    assert summary.loc["a", "MAE"] == pytest.approx(1.5)
    assert summary.loc["a", "prediction_bias"] == pytest.approx(-0.5)


def test_interval_coverage_uses_nonnegative_empirical_bounds() -> None:
    """Coverage and width reflect the supplied residual quantiles."""
    result = interval_coverage([0, 10, 20], [1, 10, 10], -2, 5)

    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["rows"] == 3
    assert result["mean_width"] == pytest.approx(20 / 3)


def test_crowd_cutpoints_reject_invalid_targets() -> None:
    """Missing or non-finite training labels cannot define stable levels."""
    with pytest.raises(ValueError, match="non-empty and finite"):
        fit_crowd_level_thresholds(pd.Series([1.0, np.nan]))
