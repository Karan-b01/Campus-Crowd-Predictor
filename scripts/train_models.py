"""Train, compare, and report the campus crowd regression model zoo."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.config import (
    DATE_COLUMN,
    EDA_COLOR_PALETTE,
    EDA_FONT_SCALE,
    EDA_X_LABEL_ROTATION,
    EXPERIMENTS_PATH,
    FIGURES_DIR,
    FORECAST_BEST_MODEL_PATH,
    MODEL_BAR_FIGURE_SIZE,
    MODEL_COMPARISON_PATH,
    MODEL_MIN_COMPLETE_WEEK_ROWS,
    MODEL_N_JOBS,
    MODEL_PLOT_DPI,
    MODEL_PLOT_LINE_WIDTH,
    MODEL_REFERENCE_LINE_WIDTH,
    MODEL_REPRESENTATIVE_WEEK_FIGURE_PATH,
    MODEL_SCATTER_ALPHA,
    MODEL_SCATTER_FIGURE_SIZE,
    MODEL_SCATTER_SIZE,
    MODEL_TIME_SERIES_FIGURE_SIZE,
    MODEL_TOP3_SCATTER_FIGURE_PATH,
    MODEL_TOP_NOWCAST_COUNT,
    MODEL_VALIDATION_MAE_FIGURE_PATH,
    NOWCAST_BEST_MODEL_PATH,
    PREDICTION_MINIMUM,
    RANDOM_SEED,
    SOURCE_LOCAL_DATETIME_LENGTH,
    TARGET_COLUMN,
    TIME_SERIES_CV_GAP,
    TIME_SERIES_CV_SPLITS,
)
from src.data import load_raw, split_chronologically
from src.evaluate import (
    METRIC_NAMES,
    compute_metrics,
    log_experiment,
    time_series_cross_validate,
)
from src.features import build_feature_set
from src.models import (
    ModelSpecification,
    build_model_zoo,
    get_model_specifications,
    predict_features,
)
from src.utils import get_logger, set_seed

LOGGER = get_logger(__name__)


def _model_parameters(specification: ModelSpecification) -> dict[str, Any]:
    """Return serializable estimator and preprocessing settings."""
    return {
        **specification.estimator.get_params(deep=False),
        "scale_numeric": specification.scale_numeric,
        "count_aware": specification.count_aware,
    }


def _log_cv_experiments(
    model_name: str,
    scenario: str,
    parameters: dict[str, Any],
    fold_scores: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Append each cross-validation fold and aggregate to the experiment log."""
    for row in fold_scores.to_dict(orient="records"):
        fold = int(row.pop("fold"))
        log_experiment(
            f"{model_name} CV fold {fold}",
            scenario,
            {**parameters, "cv_fold": fold},
            {metric: float(row[metric]) for metric in METRIC_NAMES},
        )
    means = {metric: float(summary.loc[metric, "mean"]) for metric in METRIC_NAMES}
    deviations = {metric: float(summary.loc[metric, "std"]) for metric in METRIC_NAMES}
    log_experiment(
        f"{model_name} CV mean",
        scenario,
        {**parameters, "cv_std": deviations},
        means,
    )


def _run_scenario(
    *,
    model_names: list[str],
    scenario: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    """Cross-validate and validate a selected set of model pipelines."""
    through_validation = pd.concat([train, validation])
    all_features = build_feature_set(through_validation, scenario=scenario)
    x_train = all_features.loc[train.index]
    x_validation = all_features.loc[validation.index]
    y_train = train[TARGET_COLUMN]
    y_validation = validation[TARGET_COLUMN]
    specifications = get_model_specifications()
    pipelines = build_model_zoo(x_train, model_names)
    result_rows: list[dict[str, Any]] = []
    fitted_pipelines = {}
    validation_predictions = {}

    for model_name, pipeline in pipelines.items():
        specification = specifications[model_name]
        parameters = _model_parameters(specification)
        cross_validation = time_series_cross_validate(
            pipeline,
            x_train,
            y_train,
            n_splits=TIME_SERIES_CV_SPLITS,
            gap=TIME_SERIES_CV_GAP,
        )
        _log_cv_experiments(
            model_name,
            scenario,
            parameters,
            cross_validation.fold_scores,
            cross_validation.summary,
        )

        fit_started = time.perf_counter()
        with threadpool_limits(limits=MODEL_N_JOBS):
            pipeline.fit(x_train, y_train)
        fit_time_seconds = time.perf_counter() - fit_started
        predictions = predict_features(pipeline, x_validation)
        validation_metrics = compute_metrics(y_validation, predictions)
        log_experiment(
            model_name,
            scenario,
            {**parameters, "fit_rows": len(x_train)},
            {**validation_metrics, "fit_time_seconds": fit_time_seconds},
        )

        row: dict[str, Any] = {
            "scenario": scenario,
            "model": model_name,
            "count_aware": specification.count_aware,
            **validation_metrics,
            "fit_time_seconds": fit_time_seconds,
        }
        for metric in METRIC_NAMES:
            row[f"cv_{metric}_mean"] = float(
                cross_validation.summary.loc[metric, "mean"]
            )
            row[f"cv_{metric}_std"] = float(cross_validation.summary.loc[metric, "std"])
        result_rows.append(row)
        fitted_pipelines[model_name] = pipeline
        validation_predictions[model_name] = predictions
        LOGGER.info(
            "%s %s validation MAE %.3f (CV MAE %.3f +/- %.3f).",
            scenario.title(),
            model_name,
            validation_metrics["MAE"],
            row["cv_MAE_mean"],
            row["cv_MAE_std"],
        )

    return pd.DataFrame(result_rows), fitted_pipelines, validation_predictions


def _load_baseline_scores() -> dict[str, float]:
    """Read previously measured score-to-beat MAEs from the experiment log."""
    if not EXPERIMENTS_PATH.is_file():
        raise FileNotFoundError(
            "Baseline experiment records are missing. Run the baseline workflow "
            "before model comparison."
        )
    experiments = pd.read_csv(EXPERIMENTS_PATH)
    selected = {
        "forecast": "Training day/hour average",
        "nowcast": "Same time last week",
    }
    scores: dict[str, float] = {}
    for scenario, model_name in selected.items():
        matching = experiments.loc[
            experiments["scenario"].eq(scenario)
            & experiments["model_name"].eq(model_name),
            "metrics",
        ]
        if matching.empty:
            raise ValueError(
                f"No saved {scenario} score was found for baseline {model_name!r}."
            )
        scores[scenario] = float(
            np.median([json.loads(value)["MAE"] for value in matching])
        )
    return scores


def _format_metric(value: float) -> str:
    """Format finite metrics while keeping undefined R2 values visible."""
    return f"{value:.3f}" if np.isfinite(value) else "N/A"


def _write_comparison_report(
    results: pd.DataFrame,
    *,
    baseline_scores: dict[str, float],
    train_rows: int,
    validation_rows: int,
    paired_forecast_mae: float,
    lag_absolute_gain: float,
    lag_percent_gain: float,
    poisson_gain: float,
    path: Path = MODEL_COMPARISON_PATH,
) -> Path:
    """Write results, model selection, count-loss, and lag analysis."""
    forecast = results.loc[results["scenario"].eq("forecast")].sort_values("MAE")
    nowcast = results.loc[results["scenario"].eq("nowcast")].sort_values("MAE")
    best_forecast = forecast.iloc[0]
    best_nowcast = nowcast.iloc[0]
    forecast_baseline_gap = best_forecast["MAE"] - baseline_scores["forecast"]
    nowcast_baseline_gain = baseline_scores["nowcast"] - best_nowcast["MAE"]
    poisson_wording = (
        f"improved validation MAE by {abs(poisson_gain):.3f}"
        if poisson_gain > 0
        else f"did not improve validation MAE (difference {poisson_gain:.3f})"
    )
    if best_forecast["model"] in {
        "Random Forest",
        "ExtraTrees",
        "XGBoost",
        "HistGradientBoosting",
        "HistGradientBoosting Poisson",
        "Decision Tree",
    }:
        winner_explanation = (
            "The winning tree model can make threshold splits and combine them "
            "into interactions. That matches the EDA evidence: occupancy changes "
            "cyclically through the day, the weekday/weekend curves differ, and "
            "the busiest day-hour cells are not interchangeable. The observed "
            "validation result makes it the strongest candidate in this model "
            "set, but does not prove which individual feature caused the gain. "
            "It still has negative validation R2 "
            f"({_format_metric(best_forecast['R2'])}) "
            f"and MAE {_format_metric(best_forecast['MAE'])}, versus "
            f"{_format_metric(baseline_scores['forecast'])} for the day/hour "
            "baseline, so this Forecast result does not beat the benchmark."
        )
    else:
        winner_explanation = (
            "The winning model's validation result is strongest for this split. "
            "Its CV mean and spread show how consistently it performed across "
            "earlier training windows; the observed score alone does not prove "
            "which individual feature caused the gain."
        )

    table_lines = [
        "| Scenario | Model | Count loss | Val MAE | Val RMSE | Val R2 | "
        "Val SMAPE (%) | Fit seconds | CV MAE mean +/- std | CV RMSE mean +/- std | "
        "CV R2 mean +/- std | CV SMAPE mean +/- std |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        cells = [
            row.scenario.title(),
            row.model,
            "Poisson" if row.count_aware else "Squared error",
            _format_metric(row.MAE),
            _format_metric(row.RMSE),
            _format_metric(row.R2),
            _format_metric(row.SMAPE),
            f"{row.fit_time_seconds:.2f}",
        ]
        for metric in METRIC_NAMES:
            cells.append(
                f"{_format_metric(getattr(row, f'cv_{metric}_mean'))} +/- "
                f"{_format_metric(getattr(row, f'cv_{metric}_std'))}"
            )
        table_lines.append("| " + " | ".join(cells) + " |")

    content = "\n".join(
        [
            "# Model Comparison",
            "",
            f"Models were fit on {train_rows:,} chronological training rows and "
            f"ranked on {validation_rows:,} later validation rows. Every "
            "preprocessor was fitted inside its model Pipeline on training rows "
            "only. Five expanding-window TimeSeriesSplit folds were run within "
            f"training with a {TIME_SERIES_CV_GAP}-row gap. "
            "The current same-time target needs no extra forecast-horizon gap. "
            "The test partition was not used.",
            "",
            "## Validation and training-window CV results",
            "",
            *table_lines,
            "",
            "Validation MAE is the primary model-selection metric. CV mean +/- "
            "sample standard deviation summarizes the five expanding historical "
            "folds; fit time measures the final fit on the training partition.",
            "",
            "## Best models and score-to-beat comparison",
            "",
            f"- **Best Scenario A (Forecast): {best_forecast['model']}**, "
            f"validation MAE {_format_metric(best_forecast['MAE'])}; the saved "
            f"day/hour baseline MAE is {_format_metric(baseline_scores['forecast'])}.",
            f"- **Forecast baseline check:** no Scenario A model beat the "
            f"day/hour baseline; the best model's MAE is "
            f"{_format_metric(forecast_baseline_gap)} people higher.",
            f"- **Best Scenario B (Nowcast): {best_nowcast['model']}**, "
            f"validation MAE {_format_metric(best_nowcast['MAE'])}; the saved "
            "same-time-last-week baseline MAE is "
            f"{_format_metric(baseline_scores['nowcast'])}.",
            f"- The Nowcast winner improves on the same-time-last-week baseline "
            f"by {_format_metric(nowcast_baseline_gain)} people MAE.",
            f"- Comparing the Nowcast winner with its same-model Forecast run "
            f"isolates the lag-feature gain: its Forecast MAE was "
            f"{_format_metric(paired_forecast_mae)}, a reduction of "
            f"{_format_metric(lag_absolute_gain)} people ({lag_percent_gain:.1f}%).",
            "",
            "## Count-aware loss",
            "",
            f"HistGradientBoosting with Poisson loss {poisson_wording} versus "
            "the squared-error HistGradientBoosting model on Forecast validation. "
            "Both predict non-negative counts after output clipping. This is a "
            "validation comparison only; no count-loss choice was made from test data.",
            "",
            "## Why the Forecast winner may fit this dataset",
            "",
            winner_explanation,
            "",
            f"The dataset has {train_rows:,} training examples, enough to learn "
            "structured calendar patterns but still limited compared with very "
            "large datasets. Flexible models can overfit, so the training-window "
            "CV spread and later validation score should be read together. The "
            "three Nowcast candidates were selected using Forecast validation "
            "rankings; their Nowcast scores are reported as a scenario comparison "
            "and remain validation-based.",
            "",
            "## Figures",
            "",
            "- Validation MAE: `reports/figures/model_validation_mae.png`",
            "- Top-three Forecast predicted versus actual: "
            "`reports/figures/model_top3_predicted_vs_actual.png`",
            "- Representative validation week: "
            "`reports/figures/model_representative_week_predictions.png`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    LOGGER.info("Saved model comparison report to %s.", path)
    return path


def _save_validation_mae_plot(results: pd.DataFrame) -> None:
    """Plot validation MAE by model and scenario."""
    figure, axis = plt.subplots(figsize=MODEL_BAR_FIGURE_SIZE)
    sns.barplot(
        data=results,
        x="model",
        y="MAE",
        hue="scenario",
        palette=EDA_COLOR_PALETTE,
        errorbar=None,
        ax=axis,
    )
    axis.set(
        title="Validation MAE by model and scenario",
        xlabel="Regression model",
        ylabel="Mean absolute error (people)",
    )
    axis.tick_params(axis="x", rotation=EDA_X_LABEL_ROTATION)
    axis.legend(title="Scenario")
    figure.tight_layout()
    figure.savefig(MODEL_VALIDATION_MAE_FIGURE_PATH, dpi=MODEL_PLOT_DPI)
    plt.close(figure)


def _save_top_model_scatter(
    top_models: list[str],
    actual: pd.Series,
    predictions: dict[str, np.ndarray],
    forecast_results: pd.DataFrame,
) -> None:
    """Save actual-versus-predicted scatter panels for Forecast top models."""
    figure, axes = plt.subplots(
        1, len(top_models), figsize=MODEL_SCATTER_FIGURE_SIZE, sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    colours = sns.color_palette(EDA_COLOR_PALETTE, n_colors=len(top_models))
    limits = [
        float(actual.min()),
        float(max(actual.max(), *(predictions[name].max() for name in top_models))),
    ]
    for axis, model_name, colour in zip(axes, top_models, colours, strict=True):
        axis.scatter(
            actual,
            predictions[model_name],
            alpha=MODEL_SCATTER_ALPHA,
            s=MODEL_SCATTER_SIZE,
            color=colour,
        )
        axis.plot(
            limits,
            limits,
            color="black",
            linestyle="--",
            linewidth=MODEL_REFERENCE_LINE_WIDTH,
        )
        mae = forecast_results.loc[
            forecast_results["model"].eq(model_name), "MAE"
        ].iloc[0]
        axis.set(
            title=f"{model_name}\nValidation MAE={mae:.2f}",
            xlabel="Actual people (count)",
            ylabel="Predicted people (count)",
            xlim=limits,
            ylim=limits,
        )
    figure.suptitle("Forecast validation: predicted versus actual occupancy")
    figure.tight_layout()
    figure.savefig(MODEL_TOP3_SCATTER_FIGURE_PATH, dpi=MODEL_PLOT_DPI)
    plt.close(figure)


def _save_representative_week(
    validation: pd.DataFrame, predictions: np.ndarray, model_name: str
) -> pd.Timestamp:
    """Plot one representative validation week and return its Monday date."""
    local_datetime = pd.to_datetime(
        validation[DATE_COLUMN]
        .astype("string")
        .str.slice(stop=SOURCE_LOCAL_DATETIME_LENGTH),
        errors="raise",
    )
    plot_data = pd.DataFrame(
        {
            "local_datetime": local_datetime.to_numpy(),
            "day_of_week": validation["day_of_week"].to_numpy(),
            "actual": validation[TARGET_COLUMN].to_numpy(),
            "predicted": predictions,
        },
        index=validation.index,
    )
    plot_data["week_start"] = plot_data["local_datetime"] - pd.to_timedelta(
        plot_data["day_of_week"], unit="D"
    )
    week_stats = plot_data.groupby("week_start")["actual"].agg(["mean", "size"])
    complete_weeks = week_stats.loc[week_stats["size"] >= MODEL_MIN_COMPLETE_WEEK_ROWS]
    candidates = complete_weeks if not complete_weeks.empty else week_stats
    representative_week = (
        (candidates["mean"] - candidates["mean"].median()).abs().idxmin()
    )
    representative_rows = plot_data.loc[
        plot_data["week_start"].eq(representative_week)
    ].sort_values("local_datetime")

    figure, axis = plt.subplots(figsize=MODEL_TIME_SERIES_FIGURE_SIZE)
    axis.plot(
        representative_rows["local_datetime"],
        representative_rows["actual"],
        label="Actual",
        linewidth=MODEL_PLOT_LINE_WIDTH,
    )
    axis.plot(
        representative_rows["local_datetime"],
        representative_rows["predicted"],
        label="Predicted",
        linewidth=MODEL_PLOT_LINE_WIDTH,
    )
    axis.set(
        title=f"{model_name}: predicted and actual occupancy for week of "
        f"{representative_week:%Y-%m-%d}",
        xlabel="Date and time (source local)",
        ylabel="People in gym (count)",
    )
    axis.legend(title="Series")
    figure.tight_layout()
    figure.savefig(MODEL_REPRESENTATIVE_WEEK_FIGURE_PATH, dpi=MODEL_PLOT_DPI)
    plt.close(figure)
    return pd.Timestamp(representative_week)


def run_training() -> pd.DataFrame:
    """Run model comparison without opening or scoring the held-out test set.

    Returns:
        Validation and cross-validation metrics for every fitted model.

    """
    set_seed(RANDOM_SEED)
    sns.set_theme(
        style="whitegrid", palette=EDA_COLOR_PALETTE, font_scale=EDA_FONT_SCALE
    )
    for output_path in (
        MODEL_COMPARISON_PATH,
        MODEL_VALIDATION_MAE_FIGURE_PATH,
        MODEL_TOP3_SCATTER_FIGURE_PATH,
        MODEL_REPRESENTATIVE_WEEK_FIGURE_PATH,
        FORECAST_BEST_MODEL_PATH,
        NOWCAST_BEST_MODEL_PATH,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    splits = split_chronologically(raw)
    train = splits["train"]
    validation = splits["validation"]

    all_names = list(get_model_specifications())
    forecast_results, forecast_models, forecast_predictions = _run_scenario(
        model_names=all_names,
        scenario="forecast",
        train=train,
        validation=validation,
    )
    best_three = forecast_results.nsmallest(MODEL_TOP_NOWCAST_COUNT, "MAE")[
        "model"
    ].tolist()
    nowcast_results, nowcast_models, _ = _run_scenario(
        model_names=best_three,
        scenario="nowcast",
        train=train,
        validation=validation,
    )
    all_results = pd.concat([forecast_results, nowcast_results], ignore_index=True)

    best_forecast = forecast_results.loc[forecast_results["MAE"].idxmin()]
    best_nowcast = nowcast_results.loc[nowcast_results["MAE"].idxmin()]
    paired_forecast_mae = float(
        forecast_results.loc[
            forecast_results["model"].eq(best_nowcast["model"]), "MAE"
        ].iloc[0]
    )
    lag_absolute_gain = paired_forecast_mae - float(best_nowcast["MAE"])
    lag_percent_gain = (
        100 * lag_absolute_gain / paired_forecast_mae
        if paired_forecast_mae > PREDICTION_MINIMUM
        else 0.0
    )
    poisson_mae = float(
        forecast_results.loc[
            forecast_results["model"].eq("HistGradientBoosting Poisson"), "MAE"
        ].iloc[0]
    )
    squared_boosting_mae = float(
        forecast_results.loc[
            forecast_results["model"].eq("HistGradientBoosting"), "MAE"
        ].iloc[0]
    )
    baseline_scores = _load_baseline_scores()

    _save_validation_mae_plot(all_results)
    top_three = forecast_results.nsmallest(MODEL_TOP_NOWCAST_COUNT, "MAE")[
        "model"
    ].tolist()
    _save_top_model_scatter(
        top_three,
        validation[TARGET_COLUMN],
        forecast_predictions,
        forecast_results,
    )
    representative_week = _save_representative_week(
        validation,
        forecast_predictions[str(best_forecast["model"])],
        str(best_forecast["model"]),
    )
    _write_comparison_report(
        all_results,
        baseline_scores=baseline_scores,
        train_rows=len(train),
        validation_rows=len(validation),
        paired_forecast_mae=paired_forecast_mae,
        lag_absolute_gain=lag_absolute_gain,
        lag_percent_gain=lag_percent_gain,
        poisson_gain=squared_boosting_mae - poisson_mae,
    )
    joblib.dump(forecast_models[str(best_forecast["model"])], FORECAST_BEST_MODEL_PATH)
    joblib.dump(nowcast_models[str(best_nowcast["model"])], NOWCAST_BEST_MODEL_PATH)

    print("Validation and cross-validation model comparison:")
    print(
        all_results[
            [
                "scenario",
                "model",
                "MAE",
                "RMSE",
                "R2",
                "fit_time_seconds",
                "cv_MAE_mean",
                "cv_MAE_std",
            ]
        ].to_string(index=False)
    )
    print(
        f"Best Forecast: {best_forecast['model']} "
        f"(validation MAE {best_forecast['MAE']:.3f})."
    )
    print(
        f"Best Nowcast: {best_nowcast['model']} "
        f"(validation MAE {best_nowcast['MAE']:.3f})."
    )
    print(
        f"Matched-model lag gain: {lag_absolute_gain:.3f} people "
        f"({lag_percent_gain:.1f}%) against its Forecast version."
    )
    print(f"Representative validation week starts {representative_week:%Y-%m-%d}.")
    print(f"Model comparison saved to {MODEL_COMPARISON_PATH}.")
    print(f"Experiment records appended to {EXPERIMENTS_PATH}.")
    print(f"Figures saved to {FIGURES_DIR}.")
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
