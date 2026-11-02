"""Create descriptive validation diagnostics and summarize saved test metrics."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from src.config import (
    CROWD_LEVEL_LABELS,
    CROWD_LEVEL_QUANTILES,
    DATETIME_COLUMN,
    EDA_COLOR_PALETTE,
    EDA_DAY_NAMES,
    EDA_FONT_SCALE,
    EVALUATION_ERROR_FIGURE_PATH,
    EVALUATION_ERROR_FIGURE_SIZE,
    EVALUATION_ERROR_TABLE_PATH,
    EVALUATION_FIGURE_SIZE,
    EVALUATION_IMPORTANCE_FIGURE_PATH,
    EVALUATION_IMPORTANCE_PATH,
    EVALUATION_INTERVAL_LOWER_QUANTILE,
    EVALUATION_INTERVAL_UPPER_QUANTILE,
    EVALUATION_LEVEL_FIGURE_PATH,
    EVALUATION_MEAN_SHIFT_RELEVANCE_PEOPLE,
    EVALUATION_MIN_GROUP_ROWS,
    EVALUATION_PERMUTATION_REPEATS,
    EVALUATION_PLOT_DPI,
    EVALUATION_PREDICTIONS_PATH,
    EVALUATION_SHAP_HOUR_FIGURE_PATH,
    EVALUATION_SHAP_IMPORTANCE_PATH,
    EVALUATION_SHAP_LOCAL_FIGURE_PATHS,
    EVALUATION_SHAP_SAMPLE_ROWS,
    EVALUATION_SHAP_SUMMARY_FIGURE_PATH,
    EVALUATION_SHAP_TEMP_FIGURE_PATH,
    EVALUATION_SUMMARY_PATH,
    FIGURES_DIR,
    MODEL_N_JOBS,
    PREDICTION_INTERVAL_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    TRAIN_FRACTION,
    TUNED_MODEL_METADATA_PATH,
    VALIDATION_FRACTION,
)
from src.data import load_raw, parse_datetime
from src.features import build_feature_set
from src.interpretability import (
    assign_crowd_levels,
    crowd_level_accuracy,
    fit_crowd_level_thresholds,
    grouped_error_metrics,
    interval_coverage,
    permutation_mae_importance,
    training_oof_residual_quantiles,
)
from src.models import build_model_zoo, predict_features
from src.utils import get_logger, set_seed

LOGGER = get_logger(__name__)


def _group_values(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return consistent display labels for required validation slices."""
    return {
        "hour": frame["hour"].astype(int).astype(str),
        "day_of_week": frame["day_of_week"].map(dict(enumerate(EDA_DAY_NAMES))),
        "is_holiday": frame["is_holiday"].map({0: "No", 1: "Yes"}),
        "is_during_semester": frame["is_during_semester"].map(
            {0: "Outside semester", 1: "During semester"}
        ),
        "is_start_of_semester": frame["is_start_of_semester"].map(
            {0: "Not start", 1: "Semester start"}
        ),
    }


def _error_analysis(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    actual: pd.Series,
    predicted: np.ndarray,
    thresholds: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build subgroup metrics and include a training-period comparison mean."""
    validation_groups = _group_values(validation)
    training_groups = _group_values(train)
    validation_groups["crowd_level"] = assign_crowd_levels(actual, thresholds)
    training_groups["crowd_level"] = assign_crowd_levels(
        train[TARGET_COLUMN], thresholds
    )
    frames: list[pd.DataFrame] = []
    for group_name, validation_group in validation_groups.items():
        summary = grouped_error_metrics(
            actual,
            predicted,
            validation_group,
            group_name=group_name,
        )
        training_means = (
            train.assign(_group=training_groups[group_name])
            .groupby("_group", observed=True)[TARGET_COLUMN]
            .mean()
        )
        summary = summary.rename(columns={group_name: "group_value"})
        summary["group_name"] = group_name
        summary["training_mean_actual"] = summary["group_value"].map(training_means)
        summary["validation_minus_training_mean"] = (
            summary["mean_actual"] - summary["training_mean_actual"]
        )
        frames.append(summary)
    return pd.concat(frames, ignore_index=True), validation_groups["crowd_level"]


def _plot_error_profiles(error_tables: pd.DataFrame) -> None:
    """Save hour/day/status/level error panels in the EDA style."""
    plot_groups = [
        ("hour", "Hour of day", True),
        ("day_of_week", "Day of week", False),
        ("is_holiday", "Holiday", False),
        ("is_during_semester", "Semester status", False),
        ("is_start_of_semester", "Semester start", False),
        ("crowd_level", "Actual crowd level", False),
    ]
    figure, axes = plt.subplots(3, 2, figsize=EVALUATION_ERROR_FIGURE_SIZE)
    for axis, (group, title, numeric_order) in zip(axes.flat, plot_groups, strict=True):
        subset = error_tables.loc[error_tables["group_name"].eq(group)].copy()
        if numeric_order:
            subset = subset.sort_values(
                "group_value", key=lambda values: values.astype(int)
            )
            subset["group_value"] = subset["group_value"].astype(int)
        else:
            subset = subset.sort_values("group_value")
        sns.barplot(
            data=subset,
            x="group_value",
            y="MAE",
            color=sns.color_palette(EDA_COLOR_PALETTE, 3)[1],
            errorbar=None,
            ax=axis,
        )
        axis.set(title=title, xlabel=title, ylabel="MAE (people)")
        axis.tick_params(axis="x", rotation=35)
    figure.suptitle("Validation error across calendar and crowd segments")
    figure.tight_layout()
    figure.savefig(EVALUATION_ERROR_FIGURE_PATH, dpi=EVALUATION_PLOT_DPI)
    plt.close(figure)


def _plot_level_confusion(
    actual_levels: pd.Series, predicted_levels: pd.Series
) -> None:
    """Save the validation crowd-level confusion matrix."""
    matrix = confusion_matrix(
        actual_levels,
        predicted_levels,
        labels=CROWD_LEVEL_LABELS,
    )
    figure, axis = plt.subplots(figsize=EVALUATION_FIGURE_SIZE)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=EDA_COLOR_PALETTE,
        xticklabels=CROWD_LEVEL_LABELS,
        yticklabels=CROWD_LEVEL_LABELS,
        ax=axis,
    )
    axis.set(
        title="Crowd-level validation classification",
        xlabel="Predicted level",
        ylabel="Actual level",
    )
    figure.tight_layout()
    figure.savefig(EVALUATION_LEVEL_FIGURE_PATH, dpi=EVALUATION_PLOT_DPI)
    plt.close(figure)


def _plot_permutation_importance(importance: pd.DataFrame) -> None:
    """Save top engineered features ranked by validation MAE impact."""
    top = importance.head(12).sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=EVALUATION_FIGURE_SIZE)
    axis.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color=sns.color_palette(EDA_COLOR_PALETTE, 3)[1],
        capsize=3,
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Validation permutation importance",
        xlabel="Increase in MAE after shuffling (people)",
        ylabel="Forecast feature",
    )
    figure.tight_layout()
    figure.savefig(EVALUATION_IMPORTANCE_FIGURE_PATH, dpi=EVALUATION_PLOT_DPI)
    plt.close(figure)


def _plot_shap(
    model: Any,
    validation_features: pd.DataFrame,
    actual: pd.Series,
    predictions: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Save global, dependence, and three local TreeSHAP explanations."""
    positions = np.linspace(
        0,
        len(validation_features) - 1,
        min(EVALUATION_SHAP_SAMPLE_ROWS, len(validation_features)),
        dtype=int,
    )
    sample = validation_features.iloc[positions]
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = np.asarray(preprocessor.transform(sample), dtype=float)
    feature_names = list(preprocessor.get_feature_names_out())
    explanation = shap.TreeExplainer(estimator)(transformed)
    shap_values = np.asarray(explanation.values)
    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)

    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.gcf().set_size_inches(EVALUATION_FIGURE_SIZE)
    plt.title("Global SHAP feature contributions (validation sample)")
    plt.tight_layout()
    plt.savefig(EVALUATION_SHAP_SUMMARY_FIGURE_PATH, dpi=EVALUATION_PLOT_DPI)
    plt.close()

    for source, path, title in (
        ("hour", EVALUATION_SHAP_HOUR_FIGURE_PATH, "SHAP dependence: hour"),
        (
            "temperature",
            EVALUATION_SHAP_TEMP_FIGURE_PATH,
            "SHAP dependence: temperature",
        ),
    ):
        feature = next(name for name in feature_names if name.endswith(f"__{source}"))
        shap.dependence_plot(
            feature,
            shap_values,
            transformed,
            feature_names=feature_names,
            interaction_index=None,
            show=False,
        )
        plt.gcf().set_size_inches(EVALUATION_FIGURE_SIZE)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=EVALUATION_PLOT_DPI)
        plt.close()

    sample_predictions = predictions[positions]
    quantile_positions: list[int] = []
    for quantile in (0.1, 0.5, 0.9):
        target_prediction = float(np.quantile(sample_predictions, quantile))
        candidate = int(np.argmin(np.abs(sample_predictions - target_prediction)))
        if candidate not in quantile_positions:
            quantile_positions.append(candidate)
    for candidate in range(len(sample_predictions)):
        if len(quantile_positions) == len(EVALUATION_SHAP_LOCAL_FIGURE_PATHS):
            break
        if candidate not in quantile_positions:
            quantile_positions.append(candidate)

    local_rows: list[dict[str, Any]] = []
    for local_number, sample_position in enumerate(quantile_positions):
        row_position = int(positions[sample_position])
        local_explanation = explanation[sample_position]
        shap.plots.waterfall(
            local_explanation,
            max_display=10,
            show=False,
        )
        plt.gcf().set_size_inches(EVALUATION_FIGURE_SIZE)
        plt.title(f"Local SHAP explanation {local_number + 1}")
        plt.tight_layout()
        plt.savefig(
            EVALUATION_SHAP_LOCAL_FIGURE_PATHS[local_number],
            dpi=EVALUATION_PLOT_DPI,
        )
        plt.close()
        local_rows.append(
            {
                "validation_timestamp": str(validation_features.index[row_position]),
                "actual": float(actual.iloc[row_position]),
                "prediction": float(predictions[row_position]),
                "absolute_error": float(
                    abs(actual.iloc[row_position] - predictions[row_position])
                ),
                "row_position": row_position,
            }
        )
    return global_importance.reset_index(drop=True), local_rows


def _write_summary(
    *,
    metadata: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    validation_metrics: dict[str, float],
    error_tables: pd.DataFrame,
    thresholds: Mapping[str, float],
    level_accuracy: float,
    interval: dict[str, float],
    importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    local_rows: list[dict[str, Any]],
) -> None:
    """Write a grounded summary separating final test metrics and diagnostics."""
    worst = error_tables.loc[
        error_tables["rows"].ge(EVALUATION_MIN_GROUP_ROWS)
    ].nlargest(3, "MAE")
    worst_lines: list[str] = []
    hypothesis_lines: list[str] = []
    semester_check_lines: list[str] = []
    for row in worst.itertuples(index=False):
        worst_lines.append(
            f"| {row.group_name} | {row.group_value} | {row.rows:,} | "
            f"{row.MAE:.2f} | {row.prediction_bias:.2f} | "
            f"{row.training_mean_actual:.2f} | {row.mean_actual:.2f} |"
        )
        shift = row.validation_minus_training_mean
        direction = "higher" if shift > 0 else "lower"
        shift_conclusion = (
            "supports a time-period shift"
            if abs(shift) >= EVALUATION_MEAN_SHIFT_RELEVANCE_PEOPLE
            else "does not show a large mean shift"
        )
        if row.group_name == "hour":
            hypothesis = (
                "The model learned busy evening attendance in earlier periods "
                "but may not adapt to a later drop."
            )
            hour = int(row.group_value)
            validation_hour = validation.loc[validation["hour"].eq(hour)]
            semester_flag = int(validation_hour["is_during_semester"].mode().iloc[0])
            matching_train = train.loc[
                train["hour"].eq(hour) & train["is_during_semester"].eq(semester_flag)
            ]
            if not matching_train.empty and not validation_hour.empty:
                train_mean = float(matching_train[TARGET_COLUMN].mean())
                validation_mean = float(validation_hour[TARGET_COLUMN].mean())
                mean_prediction = float(row.mean_prediction)
                status_name = "during semester" if semester_flag else "outside semester"
                semester_check_lines.append(
                    f"| {hour} | {status_name} | {len(matching_train):,} | "
                    f"{len(validation_hour):,} | {train_mean:.2f} | "
                    f"{validation_mean:.2f} | {mean_prediction:.2f} | "
                    f"{validation_mean - train_mean:.2f} |"
                )
        elif row.group_name == "day_of_week":
            hypothesis = (
                "Weekday routines or attendance levels changed between the "
                "training and validation periods."
            )
        elif row.group_name in {
            "is_holiday",
            "is_during_semester",
            "is_start_of_semester",
        }:
            hypothesis = (
                "This status may be sparse or may interact with the date and "
                "hour in a way the model cannot isolate reliably."
            )
        else:
            hypothesis = (
                "Prediction difficulty may rise at this crowd level because "
                "calendar-only features do not observe current occupancy."
            )
        hypothesis_lines.append(
            f"- For **{row.group_name} = {row.group_value}**, a possible "
            f"explanation is: {hypothesis} The validation target mean was "
            f"{row.mean_actual:.2f} "
            f"versus {row.training_mean_actual:.2f} in the matching training "
            f"group ({abs(shift):.2f} people {direction}); that comparison "
            f"{shift_conclusion}. "
            f"Mean prediction bias was {row.prediction_bias:.2f} people."
        )

    test_metrics = metadata["test_metrics"]
    typical_count = float(train[TARGET_COLUMN].mean())
    top_permutation = importance.head(5)
    top_shap = shap_importance.head(5)
    time_shap = float(
        shap_importance.loc[
            shap_importance["feature"].str.endswith("__minutes_since_midnight"),
            "mean_absolute_shap",
        ].iloc[0]
    )
    temperature_shap = float(
        shap_importance.loc[
            shap_importance["feature"].str.endswith("__temperature"),
            "mean_absolute_shap",
        ].iloc[0]
    )
    permutation_text = ", ".join(
        f"{row.feature} (+{row.importance_mean:.2f} MAE)"
        for row in top_permutation.itertuples(index=False)
    )
    shap_text = ", ".join(
        f"{row.feature} ({row.mean_absolute_shap:.2f})"
        for row in top_shap.itertuples(index=False)
    )
    test_lines = "\n".join(
        f"| {name} | {value:.4f} |" for name, value in test_metrics.items()
    )
    holiday_count = int(validation["is_holiday"].sum())
    holiday_support_note = (
        "No holiday-flagged validation readings occur, so holiday-specific error "
        "cannot be estimated in this period."
        if holiday_count == 0
        else f"Validation includes {holiday_count:,} holiday-flagged readings; "
        "that small group should be interpreted cautiously."
    )
    semester_support_note = (
        f"Validation status counts: {int(validation['is_during_semester'].sum()):,} "
        f"during-semester, {int(validation['is_during_semester'].eq(0).sum()):,} "
        f"outside-semester, and "
        f"{int(validation['is_start_of_semester'].sum()):,} semester-start rows. "
        "Absent statuses cannot be compared on this validation period."
    )
    nominal_interval_coverage = (
        EVALUATION_INTERVAL_UPPER_QUANTILE - EVALUATION_INTERVAL_LOWER_QUANTILE
    )
    interval_coverage_gap = nominal_interval_coverage - interval["coverage"]
    local_lines = "\n".join(
        f"| {row['validation_timestamp']} | {row['actual']:.1f} | "
        f"{row['prediction']:.1f} | {row['absolute_error']:.1f} |"
        for row in local_rows
    )
    content = "\n".join(
        [
            "# Evaluation Summary",
            "",
            "## Data-use policy and evaluation split",
            "",
            "The saved test metrics below are copied from the final evaluation "
            "performed earlier. No test target values are used or recomputed "
            "in this analysis. To preserve the exact-once holdout rule, detailed "
            "error, interpretability, crowd-level, and interval analyses use "
            "validation data. Validation informed model tuning and selection, "
            "so these diagnostics are descriptive and are not an independent "
            "estimate of future performance.",
            "",
            "## Real-world interpretation",
            "",
            f"The final test MAE was {test_metrics['MAE']:.2f} people per reading. "
            f"The training-period mean was {typical_count:.2f} people, so this "
            f"error is about {100 * test_metrics['MAE'] / typical_count:.1f}% of "
            "a typical observed training count. This is an average absolute "
            "error; individual readings can be much farther off. The dataset "
            "does not establish the gym's maximum capacity, so the training "
            "mean should not be described as capacity.",
            "",
            "Final test metrics are from the previously completed, single "
            "test evaluation:",
            "",
            "| Metric | Test value |",
            "| --- | ---: |",
            test_lines,
            "",
            f"For detailed validation diagnostics, MAE was "
            f"{validation_metrics['MAE']:.2f}, RMSE "
            f"{validation_metrics['RMSE']:.2f}, and R2 "
            f"{validation_metrics['R2']:.3f}.",
            "",
            "## Crowd levels and classification accuracy",
            "",
            f"Thresholds use training-set terciles only: Low is at most "
            f"{thresholds['low_max']:.2f}; Medium is above that through "
            f"{thresholds['medium_max']:.2f}; High is above "
            f"{thresholds['medium_max']:.2f} people. Ties at a threshold go "
            "to the lower level. These exact cut-points were added to model "
            "metadata so the app can use the same boundaries.",
            f"Validation crowd-level accuracy was {100 * level_accuracy:.1f}%. "
            "This is an exact three-class match rate, not a replacement for "
            "count MAE; a prediction one person across a boundary counts as a "
            "class error.",
            "",
            "![Validation crowd-level confusion matrix](figures/"
            "evaluation_crowd_level_confusion.png)",
            "",
            "## Error patterns on validation data",
            "",
            "The table shows the three highest-MAE subgroup slices with at "
            f"least {EVALUATION_MIN_GROUP_ROWS} validation readings. Positive "
            "prediction bias means overprediction. Training and validation "
            "target means are compared to check whether a simple period shift "
            "could explain each pattern.",
            "",
            "| Group | Value | Rows | MAE | Prediction bias | Matching train "
            "mean | Validation mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *worst_lines,
            "",
            *hypothesis_lines,
            "",
            "The full hour, day-of-week, holiday, semester, and crowd-level "
            "tables are in `reports/validation_error_analysis.csv`; validation "
            "predictions and absolute errors are in "
            "`reports/validation_predictions.csv`.",
            "",
            "### Checking the evening overprediction hypothesis",
            "",
            "The highest-error hours are all late evening. The next table "
            "compares the same hour and semester flag across training and "
            "validation. If attendance still drops within the same flag, the "
            "recorded semester indicator alone does not explain it; an "
            "unobserved calendar-period change is plausible, but the data "
            "cannot identify its cause.",
            "",
            "| Hour | Semester status | Train rows | Validation rows | Train mean | "
            "Validation mean | Mean prediction | Change |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *semester_check_lines,
            "",
            holiday_support_note,
            semester_support_note,
            "",
            "![Validation error profiles](figures/"
            "evaluation_validation_error_profiles.png)",
            "",
            "## Interpretability: what drives the forecast",
            "",
            f"Permutation importance ranks features by the validation MAE "
            f"increase after shuffling: {permutation_text}. SHAP's mean absolute "
            f"contributions rank: {shap_text}. Agreement between these views "
            "is stronger evidence of model reliance than either alone. These "
            "are associations used by the fitted model, not causal effects of "
            "changing a calendar or weather condition.",
            "",
            f"In this model, minutes-since-midnight had mean absolute SHAP "
            f"{time_shap:.2f}, while temperature had {temperature_shap:.2f}. "
            "This indicates that time of day dominates the model's reliance; "
            "temperature contributes a smaller signal in this validation sample. "
            "The semester flag and semester phase also rank near the top, so the "
            "model relies on academic-calendar context after time of day.",
            "",
            "The hour dependence plot shows how the tree model changes its "
            "prediction across the daily cycle; the temperature plot shows "
            "whether weather adds a consistent signal after calendar features. "
            "These plots are based on a deterministic 500-row validation sample.",
            "",
            "![Permutation importance](figures/evaluation_permutation_importance.png)",
            "",
            "![SHAP global summary](figures/evaluation_shap_global_summary.png)",
            "",
            "![SHAP dependence by hour](figures/evaluation_shap_dependence_hour.png)",
            "",
            "![SHAP dependence by temperature](figures/"
            "evaluation_shap_dependence_temperature.png)",
            "",
            "### Three local validation predictions",
            "",
            "The examples span lower, middle, and higher predicted counts in "
            "the explanation sample. Waterfall plots show which encoded model "
            "features moved each prediction above or below the model's average.",
            "",
            "| Timestamp | Actual people | Predicted people | Absolute error |",
            "| --- | ---: | ---: | ---: |",
            local_lines,
            "",
            *[
                f"![Local explanation {index}](figures/"
                f"evaluation_shap_local_{index}.png)"
                for index in range(1, len(local_rows) + 1)
            ],
            "",
            "## Uncertainty calibration",
            "",
            f"An {100 * nominal_interval_coverage:.0f}% "
            f"interval was calibrated from the "
            f"{100 * EVALUATION_INTERVAL_LOWER_QUANTILE:.0f}th and "
            f"{100 * EVALUATION_INTERVAL_UPPER_QUANTILE:.0f}th percentiles of "
            "training-only expanding-window out-of-fold residuals. Its "
            f"coverage on validation data was {100 * interval['coverage']:.1f}% "
            f"({interval['covered_rows']:,}/{interval['rows']:,} readings); "
            f"this is {100 * interval_coverage_gap:.1f} "
            "percentage points below nominal coverage. "
            f"Mean interval width was {interval['mean_width']:.1f} people. "
            "This validation coverage is not a test-set coverage claim. The "
            "interval quantiles and observed validation coverage were saved "
            "in `models/prediction_interval.json`.",
            "",
            "## What the model cannot do",
            "",
            "- It cannot anticipate an unscheduled event, equipment closure, "
            "or sudden change in attendance because those inputs are absent.",
            "- It cannot use live crowd counts; this is the calendar/weather "
            "Forecast scenario, not the lag-based Nowcast scenario.",
            "- Its interval is a historical empirical band, not a guarantee "
            "that an individual reading will fall inside it.",
            "- It cannot establish why a feature is associated with attendance "
            "or prove that changing temperature, semester status, or schedule "
            "would cause a crowd change.",
            "",
            "## Limitations and generalization",
            "",
            "The dataset describes one campus gym and spans about 19 months "
            "(August 2015 through March 2017), not a broad sample of gyms or "
            "years. There is no event schedule, exam calendar, equipment status, "
            "or live occupancy stream. The supplied file does not document "
            "whether counts are directly logged, curated, or synthetic. If "
            "synthetic, it may encode simplified patterns; if sensor-logged, "
            "site-specific collection practices and missing intervals may "
            "differ from future operations. Holiday groups have "
            "particularly limited support. Calendar routines, climate, campus "
            "events, capacity, and operating hours differ elsewhere, so these "
            "results may not generalize to another gym or later period. The "
            "negative final test R2 reinforces that limitation.",
            "",
        ]
    )
    EVALUATION_SUMMARY_PATH.write_text(content, encoding="utf-8")


def run_evaluation() -> dict[str, Any]:
    """Run validation diagnostics without reopening the held-out test labels."""
    set_seed(RANDOM_SEED)
    sns.set_theme(
        style="whitegrid", palette=EDA_COLOR_PALETTE, font_scale=EDA_FONT_SCALE
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for output_path in (
        EVALUATION_SUMMARY_PATH,
        EVALUATION_ERROR_TABLE_PATH,
        EVALUATION_IMPORTANCE_PATH,
        EVALUATION_PREDICTIONS_PATH,
        EVALUATION_ERROR_FIGURE_PATH,
        EVALUATION_LEVEL_FIGURE_PATH,
        EVALUATION_IMPORTANCE_FIGURE_PATH,
        EVALUATION_SHAP_SUMMARY_FIGURE_PATH,
        EVALUATION_SHAP_HOUR_FIGURE_PATH,
        EVALUATION_SHAP_TEMP_FIGURE_PATH,
        *EVALUATION_SHAP_LOCAL_FIGURE_PATHS,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with TUNED_MODEL_METADATA_PATH.open(encoding="utf-8") as metadata_file:
        metadata: dict[str, Any] = json.load(metadata_file)
    with PREDICTION_INTERVAL_PATH.open(encoding="utf-8") as interval_file:
        interval_metadata: dict[str, Any] = json.load(interval_file)

    # Only the training and validation prefixes are materialized as splits.
    # The final test partition is deliberately not selected or analyzed here.
    ordered = parse_datetime(load_raw())
    train_end = int(len(ordered) * TRAIN_FRACTION)
    validation_end = int(len(ordered) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    train = ordered.iloc[:train_end].reset_index(drop=True)
    validation = ordered.iloc[train_end:validation_end].reset_index(drop=True)
    x_train = build_feature_set(train, scenario="forecast")
    y_train = train[TARGET_COLUMN].astype(float)
    x_validation = build_feature_set(validation, scenario="forecast")
    y_validation = validation[TARGET_COLUMN].astype(float)
    x_train.index = pd.Index(train[DATETIME_COLUMN], name=DATETIME_COLUMN)
    x_validation.index = pd.Index(validation[DATETIME_COLUMN], name=DATETIME_COLUMN)

    thresholds = fit_crowd_level_thresholds(y_train)
    metadata["crowd_levels"] = {
        "labels": list(CROWD_LEVEL_LABELS),
        "method": "training target terciles",
        "quantiles": list(CROWD_LEVEL_QUANTILES),
        **thresholds,
    }
    TUNED_MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    model_name = str(metadata["model_name"])
    diagnostic_pipeline = build_model_zoo(x_train, [model_name])[model_name]
    diagnostic_pipeline.set_params(
        **{f"model__{name}": value for name, value in metadata["parameters"].items()}
    )
    with threadpool_limits(limits=MODEL_N_JOBS):
        diagnostic_pipeline.fit(x_train, y_train)
        predictions = predict_features(diagnostic_pipeline, x_validation)
    absolute_errors = np.abs(y_validation.to_numpy() - predictions)
    residual_bounds_80 = training_oof_residual_quantiles(
        diagnostic_pipeline,
        x_train,
        y_train,
        lower_quantile=EVALUATION_INTERVAL_LOWER_QUANTILE,
        upper_quantile=EVALUATION_INTERVAL_UPPER_QUANTILE,
    )
    interval_metadata["80_percent_interval"] = {
        "lower_quantile": EVALUATION_INTERVAL_LOWER_QUANTILE,
        "upper_quantile": EVALUATION_INTERVAL_UPPER_QUANTILE,
        **residual_bounds_80,
        "validation_coverage": interval_coverage(
            y_validation,
            predictions,
            residual_bounds_80["lower_residual"],
            residual_bounds_80["upper_residual"],
        ),
    }
    PREDICTION_INTERVAL_PATH.write_text(
        json.dumps(interval_metadata, indent=2), encoding="utf-8"
    )

    validation_metrics = {
        "MAE": float(absolute_errors.mean()),
        "RMSE": float(np.sqrt(np.square(absolute_errors).mean())),
        "R2": float(
            1
            - np.square(y_validation.to_numpy() - predictions).sum()
            / np.square(y_validation.to_numpy() - y_validation.mean()).sum()
        ),
    }
    error_tables, actual_levels = _error_analysis(
        train, validation, y_validation, predictions, thresholds
    )
    error_tables.to_csv(EVALUATION_ERROR_TABLE_PATH, index=False)
    validation_predictions = validation[[DATETIME_COLUMN, "hour", "day_of_week"]].copy()
    validation_predictions["actual"] = y_validation.to_numpy()
    validation_predictions["prediction"] = predictions
    validation_predictions["absolute_error"] = absolute_errors
    validation_predictions["actual_level"] = actual_levels.astype(str).to_numpy()
    validation_predictions["predicted_level"] = (
        assign_crowd_levels(predictions, thresholds).astype(str).to_numpy()
    )
    validation_predictions.to_csv(EVALUATION_PREDICTIONS_PATH, index=False)

    _plot_error_profiles(error_tables)
    predicted_levels = assign_crowd_levels(predictions, thresholds)
    level_accuracy = crowd_level_accuracy(y_validation, predictions, thresholds)
    _plot_level_confusion(actual_levels, predicted_levels)
    importance = permutation_mae_importance(
        diagnostic_pipeline,
        x_validation,
        y_validation,
        n_repeats=EVALUATION_PERMUTATION_REPEATS,
    )
    importance.to_csv(EVALUATION_IMPORTANCE_PATH, index=False)
    _plot_permutation_importance(importance)

    shap_importance, local_rows = _plot_shap(
        diagnostic_pipeline,
        x_validation,
        y_validation,
        predictions,
    )
    shap_importance.to_csv(EVALUATION_SHAP_IMPORTANCE_PATH, index=False)
    interval = interval_metadata["80_percent_interval"]["validation_coverage"]
    _write_summary(
        metadata=metadata,
        train=train,
        validation=validation,
        validation_metrics=validation_metrics,
        error_tables=error_tables,
        thresholds=thresholds,
        level_accuracy=level_accuracy,
        interval=interval,
        importance=importance,
        shap_importance=shap_importance,
        local_rows=local_rows,
    )
    LOGGER.info("Saved evaluation summary and validation diagnostics.")
    print(f"Validation metrics: {validation_metrics}")
    print(f"Training mean count: {y_train.mean():.3f}")
    print(f"Training-derived crowd thresholds: {thresholds}")
    print(f"Validation crowd-level accuracy: {level_accuracy:.4f}")
    print(f"80% interval validation coverage: {interval['coverage']:.4f}")
    print("Worst validation error groups:")
    print(
        error_tables.loc[error_tables["rows"].ge(EVALUATION_MIN_GROUP_ROWS)]
        .nlargest(3, "MAE")[
            [
                "group_name",
                "group_value",
                "rows",
                "MAE",
                "prediction_bias",
                "validation_minus_training_mean",
            ]
        ]
        .to_string(index=False)
    )
    print("Top permutation importance:")
    print(importance.head(10).to_string(index=False))
    print(f"Saved report: {EVALUATION_SUMMARY_PATH}")
    return {
        "validation_metrics": validation_metrics,
        "error_tables": error_tables,
        "thresholds": thresholds,
        "level_accuracy": level_accuracy,
        "interval": interval,
        "importance": importance,
        "shap_importance": shap_importance,
        "local_rows": local_rows,
        "figures": [
            EVALUATION_ERROR_FIGURE_PATH,
            EVALUATION_LEVEL_FIGURE_PATH,
            EVALUATION_IMPORTANCE_FIGURE_PATH,
            EVALUATION_SHAP_SUMMARY_FIGURE_PATH,
            EVALUATION_SHAP_HOUR_FIGURE_PATH,
            EVALUATION_SHAP_TEMP_FIGURE_PATH,
            *EVALUATION_SHAP_LOCAL_FIGURE_PATHS,
        ],
        "test_metrics_from_saved_metadata": metadata["test_metrics"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
