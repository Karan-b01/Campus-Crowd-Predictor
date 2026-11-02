"""Tune top Forecast models, evaluate once on test, and save serving artifacts."""

from __future__ import annotations

import json
import logging
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from src.config import (
    DATETIME_COLUMN,
    EDA_COLOR_PALETTE,
    EDA_FONT_SCALE,
    EXPERIMENTS_PATH,
    FINAL_TEST_RESULTS_PATH,
    FORECAST_FEATURE_COLUMNS,
    MODEL_N_JOBS,
    PREDICTION_INTERVAL_LOWER_QUANTILE,
    PREDICTION_INTERVAL_PATH,
    PREDICTION_INTERVAL_UPPER_QUANTILE,
    RANDOM_SEED,
    TARGET_COLUMN,
    TIME_SERIES_CV_GAP,
    TIME_SERIES_CV_SPLITS,
    TUNED_BEST_MODEL_PATH,
    TUNED_MODEL_METADATA_PATH,
    TUNING_CV_SPLITS,
    TUNING_FIGURE_SIZE,
    TUNING_HISTORY_FIGURE_PATH,
    TUNING_IMPORTANCE_FIGURE_PATH,
    TUNING_LEARNING_CURVE_FIGURE_PATH,
    TUNING_LEARNING_CURVE_FRACTIONS,
    TUNING_MIN_TRAINING_ROWS,
    TUNING_MODEL_NAMES,
    TUNING_PARAMETER_SPACES,
    TUNING_PLOT_DPI,
    TUNING_REPORT_PATH,
    TUNING_TRIALS_PER_MODEL,
)
from src.data import load_raw, split_chronologically
from src.evaluate import compute_metrics, log_experiment
from src.features import build_feature_set
from src.models import build_model_zoo, predict_features
from src.utils import get_logger, set_seed

LOGGER = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_parameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Sample a small, model-specific search space for a trial."""
    space = TUNING_PARAMETER_SPACES[model_name]
    if model_name == "Random Forest":
        # Tree count trades fitting stability against runtime. Depth controls
        # interaction complexity, leaf size regularizes rare calendar cells,
        # and max_features controls predictor diversity between component trees.
        return {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", *space["min_samples_leaf"]
            ),
            "max_features": trial.suggest_float("max_features", *space["max_features"]),
        }

    # Iterations and learning rate set the number and size of boosting updates;
    # leaf count and minimum leaf rows control tree complexity; L2 discourages
    # overly large leaf predictions.
    return {
        "max_iter": trial.suggest_int("max_iter", *space["max_iter"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *space["learning_rate"], log=True
        ),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", *space["max_leaf_nodes"]),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", *space["min_samples_leaf"]
        ),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", *space["l2_regularization"], log=True
        ),
    }


def _cv_mae(
    pipeline: Any, features: pd.DataFrame, target: pd.Series
) -> tuple[float, list[float]]:
    """Return mean fold MAE and fold scores on expanding chronological windows."""
    splitter = TimeSeriesSplit(n_splits=TUNING_CV_SPLITS, gap=TIME_SERIES_CV_GAP)
    scores: list[float] = []
    for fit_indices, validation_indices in splitter.split(features):
        fold_model = clone(pipeline)
        with threadpool_limits(limits=MODEL_N_JOBS):
            fold_model.fit(features.iloc[fit_indices], target.iloc[fit_indices])
            prediction = predict_features(fold_model, features.iloc[validation_indices])
        scores.append(
            compute_metrics(target.iloc[validation_indices], prediction)["MAE"]
        )
    return float(np.mean(scores)), scores


def _tune_one_model(
    model_name: str, x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[optuna.Study, Any]:
    """Optimize a single model using training rows and chronological CV only."""
    base_pipeline = build_model_zoo(x_train, [model_name])[model_name]

    def objective(trial: optuna.Trial) -> float:
        parameters = _suggest_parameters(trial, model_name)
        candidate = clone(base_pipeline).set_params(
            **{f"model__{name}": value for name, value in parameters.items()}
        )
        mean_mae, fold_scores = _cv_mae(candidate, x_train, y_train)
        log_experiment(
            f"{model_name} tuning trial {trial.number + 1}",
            "forecast",
            {**parameters, "cv_fold_MAE": fold_scores},
            {"CV_MAE": mean_mae},
        )
        return mean_mae

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=model_name,
    )
    study.optimize(objective, n_trials=TUNING_TRIALS_PER_MODEL, show_progress_bar=False)
    best_pipeline = clone(base_pipeline).set_params(
        **{f"model__{name}": value for name, value in study.best_params.items()}
    )
    LOGGER.info(
        "%s tuning complete: best CV MAE %.3f after %s trials.",
        model_name,
        study.best_value,
        len(study.trials),
    )
    return study, best_pipeline


def _best_cv_details(
    pipeline: Any, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Score the chosen parameters with five-fold CV and collect OOF residuals."""
    splitter = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS, gap=TIME_SERIES_CV_GAP)
    fold_metrics: list[dict[str, float]] = []
    residuals: list[np.ndarray] = []
    out_of_fold_actual: list[np.ndarray] = []
    for fit_indices, validation_indices in splitter.split(features):
        fold_model = clone(pipeline)
        with threadpool_limits(limits=MODEL_N_JOBS):
            fold_model.fit(features.iloc[fit_indices], target.iloc[fit_indices])
            prediction = predict_features(fold_model, features.iloc[validation_indices])
        actual = target.iloc[validation_indices].to_numpy(dtype=float)
        fold_metrics.append(compute_metrics(actual, prediction))
        residuals.append(actual - prediction)
        out_of_fold_actual.append(actual)
    summary: dict[str, float] = {}
    for metric in ("MAE", "RMSE", "R2", "SMAPE"):
        values = [fold[metric] for fold in fold_metrics]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=1))
    return summary, np.concatenate(residuals), np.concatenate(out_of_fold_actual)


def _save_study_figures(studies: dict[str, optuna.Study]) -> None:
    """Save Optuna optimization history and parameter importance panels."""
    sns.set_theme(
        style="whitegrid", palette=EDA_COLOR_PALETTE, font_scale=EDA_FONT_SCALE
    )
    figure, axes = plt.subplots(1, len(studies), figsize=TUNING_FIGURE_SIZE)
    for axis, (model_name, study) in zip(
        np.atleast_1d(axes), studies.items(), strict=True
    ):
        plt.sca(axis)
        plot_optimization_history(study, target_name=model_name)
        axis.set_title(f"Optimization history: {model_name}")
        axis.set_xlabel("Trial")
        axis.set_ylabel("Best CV MAE (people)")
    figure.tight_layout()
    figure.savefig(TUNING_HISTORY_FIGURE_PATH, dpi=TUNING_PLOT_DPI)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(studies), figsize=TUNING_FIGURE_SIZE)
    for axis, (model_name, study) in zip(
        np.atleast_1d(axes), studies.items(), strict=True
    ):
        plt.sca(axis)
        plot_param_importances(study, target_name=model_name)
        axis.set_title(f"Parameter importance: {model_name}")
        axis.set_xlabel("Relative importance")
    figure.tight_layout()
    figure.savefig(TUNING_IMPORTANCE_FIGURE_PATH, dpi=TUNING_PLOT_DPI)
    plt.close(figure)


def _save_learning_curve(
    pipeline: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> list[dict[str, float]]:
    """Plot chronological prefix learning scores without using test data."""
    rows: list[dict[str, float]] = []
    for fraction in TUNING_LEARNING_CURVE_FRACTIONS:
        row_count = max(TUNING_MIN_TRAINING_ROWS, int(len(x_train) * fraction))
        candidate = clone(pipeline)
        with threadpool_limits(limits=MODEL_N_JOBS):
            candidate.fit(x_train.iloc[:row_count], y_train.iloc[:row_count])
            train_predictions = predict_features(candidate, x_train.iloc[:row_count])
            validation_predictions = predict_features(candidate, x_validation)
        rows.append(
            {
                "training_rows": row_count,
                "training_MAE": compute_metrics(
                    y_train.iloc[:row_count], train_predictions
                )["MAE"],
                "validation_MAE": compute_metrics(y_validation, validation_predictions)[
                    "MAE"
                ],
            }
        )
    curve = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=TUNING_FIGURE_SIZE)
    axis.plot(
        curve["training_rows"], curve["training_MAE"], marker="o", label="Train MAE"
    )
    axis.plot(
        curve["training_rows"],
        curve["validation_MAE"],
        marker="o",
        label="Validation MAE",
    )
    axis.set(
        title="Chronological learning curve for tuned Forecast model",
        xlabel="Training rows (chronological prefix)",
        ylabel="Mean absolute error (people)",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(TUNING_LEARNING_CURVE_FIGURE_PATH, dpi=TUNING_PLOT_DPI)
    plt.close(figure)
    return rows


def _package_versions() -> dict[str, str]:
    """Record the Python and relevant installed package versions."""
    packages = ("numpy", "pandas", "scikit-learn", "optuna", "xgboost", "joblib")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _forecast_baseline_mae() -> float:
    """Read the previously recorded day/hour validation baseline."""
    experiments = pd.read_csv(EXPERIMENTS_PATH)
    baseline_rows = experiments.loc[
        experiments["model_name"].eq("Training day/hour average")
        & experiments["scenario"].eq("forecast")
    ]
    if baseline_rows.empty:
        raise ValueError("The recorded Forecast day/hour baseline is missing.")
    metrics = json.loads(baseline_rows.iloc[-1]["metrics"])
    return float(metrics["MAE"])


def _write_tuning_report(
    results: pd.DataFrame,
    studies: dict[str, optuna.Study],
    learning_curve: list[dict[str, float]],
    elapsed_seconds: float,
    winner_name: str,
    baseline_mae: float,
) -> None:
    """Write tuning choices, validation comparison, and overfit evidence."""
    lines = [
        "# Scenario A Hyperparameter Tuning",
        "",
        f"Optuna used {TUNING_TRIALS_PER_MODEL} trials per model and "
        f"{TUNING_CV_SPLITS}-fold expanding TimeSeriesSplit on training data only. "
        f"The default-fit comparison plus Optuna search took "
        f"{elapsed_seconds:.1f} seconds. Validation data "
        "was not used to choose trial parameters; the tuned candidates were then "
        "compared on validation to select the final model. Test data was not used "
        "in tuning or model selection.",
        "",
        "## Validation before and after tuning",
        "",
        "| Model | Default MAE | Tuned MAE | Default RMSE | Tuned RMSE | "
        "Default R2 | Tuned R2 | CV MAE mean +/- std | Training MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.default_MAE:.3f} | {row.MAE:.3f} | "
            f"{row.default_RMSE:.3f} | {row.RMSE:.3f} | "
            f"{row.default_R2:.3f} | {row.R2:.3f} | "
            f"{row.cv_MAE_mean:.3f} +/- {row.cv_MAE_std:.3f} | "
            f"{row.train_MAE:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Chosen model: **{winner_name}**, based on the lowest validation MAE "
            "among the two tuned candidates. The final model is fit on train plus "
            "validation after this choice.",
            f"Neither tuned candidate beats the established training-only "
            f"day/hour average (validation MAE {baseline_mae:.3f}); the selected "
            f"model is {float(results.iloc[0]['MAE']) - baseline_mae:.3f} people "
            "higher in MAE. It remains a limitation of the calendar-only model.",
            "",
            "## Search spaces and selected parameters",
            "",
        ]
    )
    for model_name, study in studies.items():
        lines.extend(
            [
                f"### {model_name}",
                "",
                f"Best training-CV MAE: {study.best_value:.3f} people.",
                "",
                "Selected parameters: `"
                + json.dumps(study.best_params, sort_keys=True)
                + "`",
                "",
            ]
        )
    lines.extend(
        [
            "## Overfitting check and learning curve",
            "",
            "Training MAE is measured on the same training rows used to fit the "
            "reported model and is therefore optimistic. Read it beside the "
            "later validation MAE and the expanding-window CV mean and spread. "
            f"The table's CV summary is from a separate {TIME_SERIES_CV_SPLITS}-fold "
            "expanding TimeSeriesSplit on training only. "
            "The learning curve grows the training prefix in time order and "
            "scores each fit on the same later validation block.",
            "",
            "| Training rows | Train MAE | Validation MAE |",
            "| ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {row['training_rows']:,} | {row['training_MAE']:.3f} | "
        f"{row['validation_MAE']:.3f} |"
        for row in learning_curve
    )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `reports/figures/optuna_optimization_history.png`",
            "- `reports/figures/optuna_parameter_importance.png`",
            "- `reports/figures/tuned_model_learning_curve.png`",
            "",
        ]
    )
    TUNING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _final_test_report(
    model_name: str,
    metrics: dict[str, float],
    training_start: str,
    training_end: str,
    test_start: str,
    test_end: str,
    test_rows: int,
) -> None:
    """Save the single final test evaluation and clearly record its use."""
    metric_lines = [f"| {name} | {value:.4f} |" for name, value in metrics.items()]
    content = "\n".join(
        [
            "# Final Test Results",
            "",
            f"Selected Forecast model: **{model_name}**. It was chosen using "
            "training-only chronological Optuna CV followed by validation MAE. "
            "The final pipeline was then retrained on the combined training and "
            "validation partitions.",
            "",
            f"Final fit dates: {training_start} through {training_end}.",
            f"Held-out test dates: {test_start} through {test_end} "
            f"({test_rows:,} rows).",
            "",
            "## Test metrics",
            "",
            "| Metric | Test score |",
            "| --- | ---: |",
            *metric_lines,
            "",
            "**Test-set use:** the held-out target was evaluated exactly once, "
            "after all trial parameters and the model choice were fixed. It was "
            "not used for fitting, feature decisions, or tuning, and no changes "
            "were made after viewing these test metrics.",
            "",
        ]
    )
    FINAL_TEST_RESULTS_PATH.write_text(content, encoding="utf-8")


def run_tuning() -> pd.DataFrame:
    """Tune the top two Forecast models, select, then test one final pipeline."""
    set_seed(RANDOM_SEED)
    sns.set_theme(
        style="whitegrid", palette=EDA_COLOR_PALETTE, font_scale=EDA_FONT_SCALE
    )
    for output_path in (
        TUNING_REPORT_PATH,
        FINAL_TEST_RESULTS_PATH,
        TUNING_HISTORY_FIGURE_PATH,
        TUNING_IMPORTANCE_FIGURE_PATH,
        TUNING_LEARNING_CURVE_FIGURE_PATH,
        TUNED_BEST_MODEL_PATH,
        TUNED_MODEL_METADATA_PATH,
        PREDICTION_INTERVAL_PATH,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    splits = split_chronologically(raw)
    train = splits["train"].reset_index(drop=True)
    validation = splits["validation"].reset_index(drop=True)
    test = splits["test"].reset_index(drop=True)
    x_train = build_feature_set(train, scenario="forecast")
    y_train = train[TARGET_COLUMN].astype(float)
    x_validation = build_feature_set(validation, scenario="forecast")
    y_validation = validation[TARGET_COLUMN].astype(float)

    studies: dict[str, optuna.Study] = {}
    tuned_pipelines: dict[str, Any] = {}
    defaults: dict[str, dict[str, float]] = {}
    tuning_started = time.perf_counter()
    for model_name in TUNING_MODEL_NAMES:
        default_pipeline = build_model_zoo(x_train, [model_name])[model_name]
        with threadpool_limits(limits=MODEL_N_JOBS):
            default_pipeline.fit(x_train, y_train)
            default_prediction = predict_features(default_pipeline, x_validation)
        defaults[model_name] = compute_metrics(y_validation, default_prediction)
        study, tuned_pipeline = _tune_one_model(model_name, x_train, y_train)
        studies[model_name] = study
        tuned_pipelines[model_name] = tuned_pipeline
    elapsed_seconds = time.perf_counter() - tuning_started

    rows: list[dict[str, float | str | dict[str, float]]] = []
    for model_name, pipeline in tuned_pipelines.items():
        cv_summary, _, _ = _best_cv_details(pipeline, x_train, y_train)
        fitted_for_score = clone(pipeline)
        with threadpool_limits(limits=MODEL_N_JOBS):
            fitted_for_score.fit(x_train, y_train)
            train_prediction = predict_features(fitted_for_score, x_train)
            validation_prediction = predict_features(fitted_for_score, x_validation)
        training_metrics = compute_metrics(y_train, train_prediction)
        validation_metrics = compute_metrics(y_validation, validation_prediction)
        log_experiment(
            f"{model_name} tuned validation",
            "forecast",
            pipeline.named_steps["model"].get_params(deep=False),
            {**validation_metrics, "fit_on": "training partition"},
        )
        rows.append(
            {
                "model": model_name,
                "default_MAE": defaults[model_name]["MAE"],
                "default_RMSE": defaults[model_name]["RMSE"],
                "default_R2": defaults[model_name]["R2"],
                **validation_metrics,
                "train_MAE": training_metrics["MAE"],
                **{f"cv_{key}": value for key, value in cv_summary.items()},
            }
        )
    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    winner_name = str(results.iloc[0]["model"])
    winner_pipeline = tuned_pipelines[winner_name]
    _save_study_figures(studies)
    learning_curve = _save_learning_curve(
        winner_pipeline, x_train, y_train, x_validation, y_validation
    )
    _write_tuning_report(
        results,
        studies,
        learning_curve,
        elapsed_seconds,
        winner_name,
        _forecast_baseline_mae(),
    )
    print("Tuned validation comparison:")
    print(
        results[
            [
                "model",
                "default_MAE",
                "default_RMSE",
                "default_R2",
                "MAE",
                "RMSE",
                "R2",
                "train_MAE",
                "cv_MAE_mean",
                "cv_MAE_std",
            ]
        ].to_string(index=False)
    )
    print(f"Optuna runtime: {elapsed_seconds:.1f} seconds.")
    print(f"Selected model: {winner_name}.")

    # Form empirical interval residuals from training-only out-of-fold forecasts.
    _, oof_residuals, _ = _best_cv_details(winner_pipeline, x_train, y_train)
    lower_residual = float(
        np.quantile(oof_residuals, PREDICTION_INTERVAL_LOWER_QUANTILE)
    )
    upper_residual = float(
        np.quantile(oof_residuals, PREDICTION_INTERVAL_UPPER_QUANTILE)
    )
    interval_metadata = {
        "method": "training-only expanding-window out-of-fold residual quantiles",
        "lower_quantile": PREDICTION_INTERVAL_LOWER_QUANTILE,
        "upper_quantile": PREDICTION_INTERVAL_UPPER_QUANTILE,
        "lower_residual": lower_residual,
        "upper_residual": upper_residual,
        "residual_count": int(len(oof_residuals)),
    }
    PREDICTION_INTERVAL_PATH.write_text(
        json.dumps(interval_metadata, indent=2), encoding="utf-8"
    )

    train_validation = pd.concat([train, validation], ignore_index=True)
    x_train_validation = build_feature_set(train_validation, scenario="forecast")
    y_train_validation = train_validation[TARGET_COLUMN].astype(float)
    final_pipeline = clone(winner_pipeline)
    with threadpool_limits(limits=MODEL_N_JOBS):
        final_pipeline.fit(x_train_validation, y_train_validation)
    joblib.dump(final_pipeline, TUNED_BEST_MODEL_PATH)

    # Test predictors are calendar-only. The test target is first read below,
    # once, after the model, parameters, and interval have all been finalized.
    test_features = build_feature_set(test, scenario="forecast")
    test_prediction = predict_features(final_pipeline, test_features)
    test_metrics = compute_metrics(test[TARGET_COLUMN], test_prediction)
    log_experiment(
        f"{winner_name} final test",
        "forecast",
        studies[winner_name].best_params,
        test_metrics,
    )
    _final_test_report(
        winner_name,
        test_metrics,
        str(train_validation[DATETIME_COLUMN].iloc[0]),
        str(train_validation[DATETIME_COLUMN].iloc[-1]),
        str(test[DATETIME_COLUMN].iloc[0]),
        str(test[DATETIME_COLUMN].iloc[-1]),
        len(test),
    )
    metadata = {
        "model_name": winner_name,
        "scenario": "forecast",
        "features": FORECAST_FEATURE_COLUMNS,
        "training_start": str(train_validation[DATETIME_COLUMN].iloc[0]),
        "training_end": str(train_validation[DATETIME_COLUMN].iloc[-1]),
        "training_rows": len(train_validation),
        "parameters": studies[winner_name].best_params,
        "library_versions": _package_versions(),
        "test_metrics": test_metrics,
        "prediction_interval_artifact": str(PREDICTION_INTERVAL_PATH.name),
    }
    TUNED_MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Final untouched test metrics (single evaluation):")
    print(pd.Series(test_metrics).to_string())
    print(f"Final test report: {FINAL_TEST_RESULTS_PATH}.")
    print(f"Saved pipeline: {TUNED_BEST_MODEL_PATH}.")
    print(f"Saved metadata: {TUNED_MODEL_METADATA_PATH}.")
    print(f"Saved prediction interval: {PREDICTION_INTERVAL_PATH}.")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tuning()
