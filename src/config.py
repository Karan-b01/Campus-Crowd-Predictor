"""Central project configuration and filesystem paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SUPPORTED_RAW_EXTENSIONS = (".csv", ".arff")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EDA_NOTEBOOK_PATH = NOTEBOOKS_DIR / "01_eda.ipynb"
EVALUATION_NOTEBOOK_PATH = NOTEBOOKS_DIR / "02_evaluation.ipynb"
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_AUDIT_PATH = REPORTS_DIR / "data_audit.md"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.md"
FEATURE_DICTIONARY_PATH = REPORTS_DIR / "feature_dictionary.md"
SPLIT_SUMMARY_PATH = REPORTS_DIR / "split_summary.md"
BASELINE_REPORT_PATH = REPORTS_DIR / "baselines.md"
EXPERIMENTS_PATH = REPORTS_DIR / "experiments.csv"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.md"
TUNING_REPORT_PATH = REPORTS_DIR / "tuning_results.md"
FINAL_TEST_RESULTS_PATH = REPORTS_DIR / "final_test_results.md"
EVALUATION_SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.md"
EVALUATION_ERROR_TABLE_PATH = REPORTS_DIR / "validation_error_analysis.csv"
EVALUATION_IMPORTANCE_PATH = REPORTS_DIR / "validation_permutation_importance.csv"
EVALUATION_SHAP_IMPORTANCE_PATH = MODELS_DIR / "validation_shap_importance.csv"
EVALUATION_PREDICTIONS_PATH = REPORTS_DIR / "validation_predictions.csv"
TUNING_HISTORY_FIGURE_PATH = FIGURES_DIR / "optuna_optimization_history.png"
TUNING_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "optuna_parameter_importance.png"
TUNING_LEARNING_CURVE_FIGURE_PATH = FIGURES_DIR / "tuned_model_learning_curve.png"
EVALUATION_ERROR_FIGURE_PATH = FIGURES_DIR / "evaluation_validation_error_profiles.png"
EVALUATION_LEVEL_FIGURE_PATH = FIGURES_DIR / "evaluation_crowd_level_confusion.png"
EVALUATION_IMPORTANCE_FIGURE_PATH = (
    FIGURES_DIR / "evaluation_permutation_importance.png"
)
EVALUATION_SHAP_SUMMARY_FIGURE_PATH = FIGURES_DIR / "evaluation_shap_global_summary.png"
EVALUATION_SHAP_HOUR_FIGURE_PATH = FIGURES_DIR / "evaluation_shap_dependence_hour.png"
EVALUATION_SHAP_TEMP_FIGURE_PATH = (
    FIGURES_DIR / "evaluation_shap_dependence_temperature.png"
)
EVALUATION_SHAP_LOCAL_FIGURE_PATHS = [
    FIGURES_DIR / f"evaluation_shap_local_{index}.png" for index in (1, 2, 3)
]
MODEL_VALIDATION_MAE_FIGURE_PATH = FIGURES_DIR / "model_validation_mae.png"
MODEL_TOP3_SCATTER_FIGURE_PATH = FIGURES_DIR / "model_top3_predicted_vs_actual.png"
MODEL_REPRESENTATIVE_WEEK_FIGURE_PATH = (
    FIGURES_DIR / "model_representative_week_predictions.png"
)
TESTS_DIR = PROJECT_ROOT / "tests"
TEST_TEMP_DIR = PROJECT_ROOT / ".tmp"

RANDOM_SEED = 42
TARGET_COLUMN = "number_people"
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3_600
DATE_PREFIX_LENGTH = 10
WEEKEND_START_DAY = 5
BINARY_TRUE_VALUE = 1
ZERO_OCCUPANCY_VALUE = 0
AUDIT_DISPLAY_PRECISION = 3
AUDIT_EXAMPLE_LIMIT = 5
AUDIT_ROW_EXAMPLE_LIMIT = 10
GAP_INTERVAL_THRESHOLD_MULTIPLIER = 1.5
TARGET_OUTLIER_IQR_MULTIPLIER = 1.5
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
TIME_SERIES_CV_SPLITS = 5
TIME_SERIES_CV_GAP = 0
PREDICTION_MINIMUM = 0.0
EXPECTED_RAW_COLUMNS = [
    TARGET_COLUMN,
    "date",
    "timestamp",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "temperature",
    "is_start_of_semester",
    "is_during_semester",
    "month",
    "hour",
]
DATE_COLUMN = "date"
DATETIME_COLUMN = "datetime"
INTEGER_COLUMNS = [
    "number_people",
    "timestamp",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "is_start_of_semester",
    "is_during_semester",
    "month",
    "hour",
]
EXPECTED_INTEGER_COLUMNS = INTEGER_COLUMNS
BINARY_COLUMNS = [
    "is_weekend",
    "is_holiday",
    "is_start_of_semester",
    "is_during_semester",
]
CATEGORICAL_AUDIT_COLUMNS = [
    *BINARY_COLUMNS,
    "day_of_week",
    "month",
    "hour",
]
VALUE_RANGES = {
    "number_people": (0, None),
    "timestamp": (0, 86_399),
    "day_of_week": (0, 6),
    "month": (1, 12),
    "hour": (0, 23),
    **{column: (0, 1) for column in BINARY_COLUMNS},
}
SOURCE_LOCAL_DATETIME_LENGTH = 19
BINARY_VALUE_LEVELS = [0, 1]
EDA_COLOR_PALETTE = "vlag"
EDA_FIGURE_DPI = 200
EDA_HISTOGRAM_BINS = 50
EDA_FONT_SCALE = 1.05
EDA_LINE_WIDTH = 1.8
EDA_FIGURE_SIZE = (12, 6)
EDA_WIDE_FIGURE_SIZE = (16, 6)
EDA_DISTRIBUTION_FIGURE_SIZE = (12, 7)
EDA_FLAG_FIGURE_SIZE = (15, 5)
EDA_SCATTER_ALPHA = 0.12
EDA_SCATTER_SIZE = 12
EDA_ROLLING_WINDOW = "28D"
EDA_ROLLING_MIN_PERIODS = 14
EDA_STRUCTURAL_LOW_QUANTILE = 0.10
EDA_WEEK_DAYS = 7
EDA_DAILY_FREQUENCY = "D"
EDA_X_LABEL_ROTATION = 45
EDA_READINGS_PER_DAY = 144
EDA_DAILY_LINE_ALPHA = 0.35
EDA_FLAG_COLUMNS = {
    "is_holiday": "Federal holiday flag",
    "is_during_semester": "During semester",
    "is_start_of_semester": "Start of semester",
}
EDA_BINARY_LABELS = {0: "No", 1: "Yes"}
EDA_WEEKEND_LABELS = {0: "Weekday", 1: "Weekend"}
EDA_DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
EDA_SEASONS = ["Winter", "Spring", "Summer", "Fall"]
EDA_MONTH_TO_SEASON = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
}
EDA_FIGURE_PATHS = {
    "target_distribution": FIGURES_DIR / "eda_target_distribution.png",
    "full_time_series": FIGURES_DIR / "eda_full_time_series.png",
    "representative_week": FIGURES_DIR / "eda_representative_week.png",
    "representative_day": FIGURES_DIR / "eda_representative_day.png",
    "hourly_weekday_weekend": FIGURES_DIR / "eda_hourly_weekday_vs_weekend.png",
    "day_hour_heatmap": FIGURES_DIR / "eda_day_of_week_by_hour_heatmap.png",
    "calendar_flags": FIGURES_DIR / "eda_calendar_flag_effects.png",
    "temperature_scatter": FIGURES_DIR / "eda_temperature_vs_crowd.png",
    "temperature_season": FIGURES_DIR / "eda_temperature_by_season.png",
    "monthly_trend": FIGURES_DIR / "eda_monthly_trend_and_seasonality.png",
    "correlation": FIGURES_DIR / "eda_numeric_correlation.png",
    "structural_break": FIGURES_DIR / "eda_structural_break_check.png",
}

# Feature engineering settings are centralized so feature boundaries are
# documented and do not drift between training, evaluation, and the app.
FEATURE_TIME_BUCKET_EDGES = [-1, 5, 9, 13, 16, 21, 23]
FEATURE_TIME_BUCKET_LABELS = [
    "early_morning",
    "morning",
    "midday",
    "afternoon",
    "evening",
    "night",
]
FEATURE_RUSH_HOURS = [17, 18]
FEATURE_CYCLE_HOURS = 24
FEATURE_CYCLE_WEEKDAYS = 7
FEATURE_CYCLE_MONTHS = 12
FEATURE_LAG_MINUTES = [10, 30, 60]
FEATURE_HISTORY_DAYS = 1
FEATURE_HISTORY_WEEKS = 1
FEATURE_DAYS_PER_WEEK = 7
FEATURE_ROLLING_WINDOWS_MINUTES = {"1h": 60, "3h": 180}
FEATURE_LAG_MATCH_TOLERANCE_SECONDS = 60
FEATURE_CALENDAR_LAG_MATCH_TOLERANCE_SECONDS = 300
FEATURE_ROLLING_MIN_PERIODS = 1
FEATURE_SEMESTER_PHASE_START = "start_of_semester"
FEATURE_SEMESTER_PHASE_ACTIVE = "during_semester"
FEATURE_SEMESTER_PHASE_BREAK = "outside_semester"
FEATURE_SEASON_BY_MONTH = {
    month: season.lower() for month, season in EDA_MONTH_TO_SEASON.items()
}
FORECAST_FEATURE_COLUMNS = [
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "dow_sin",
    "dow_cos",
    "month",
    "month_sin",
    "month_cos",
    "minutes_since_midnight",
    "time_of_day_bucket",
    "is_rush_hour",
    "season",
    "is_weekend",
    "is_holiday",
    "is_start_of_semester",
    "is_during_semester",
    "semester_phase",
    "temperature",
]
NOWCAST_LAG_FEATURE_COLUMNS = [
    *(f"crowd_lag_{minutes}m" for minutes in FEATURE_LAG_MINUTES),
    "crowd_lag_1d",
    "crowd_lag_1w",
    "crowd_rolling_mean_1h",
    "crowd_rolling_std_1h",
    "crowd_rolling_mean_3h",
    "crowd_rolling_std_3h",
]

MODEL_RIDGE_ALPHA = 10.0
MODEL_ELASTICNET_ALPHA = 0.05
MODEL_ELASTICNET_L1_RATIO = 0.5
MODEL_LINEAR_MAX_ITERATIONS = 10_000
MODEL_KNN_NEIGHBORS = 15
MODEL_TREE_MAX_DEPTH = 16
MODEL_TREE_MIN_SAMPLES_LEAF = 3
MODEL_ENSEMBLE_ESTIMATORS = 120
MODEL_BOOSTING_ITERATIONS = 120
MODEL_BOOSTING_LEARNING_RATE = 0.08
MODEL_BOOSTING_MAX_LEAF_NODES = 31
MODEL_BOOSTING_L2_REGULARIZATION = 1.0
MODEL_XGBOOST_MAX_DEPTH = 6
MODEL_XGBOOST_SUBSAMPLE = 0.8
MODEL_XGBOOST_COLUMN_SAMPLE = 0.8
# Single-worker fits avoid restricted multiprocessing on Windows and keep
# resource use predictable across machines.
MODEL_N_JOBS = 1
MODEL_PLOT_DPI = 200
MODEL_SCATTER_ALPHA = 0.2
MODEL_SCATTER_SIZE = 14
MODEL_TOP_NOWCAST_COUNT = 3
MODEL_BAR_FIGURE_SIZE = (13, 7)
MODEL_SCATTER_FIGURE_SIZE = (15, 5)
MODEL_TIME_SERIES_FIGURE_SIZE = (16, 6)
MODEL_PLOT_LINE_WIDTH = 1.5
MODEL_REFERENCE_LINE_WIDTH = 1.0
MODEL_MIN_COMPLETE_WEEK_ROWS = EDA_READINGS_PER_DAY * EDA_WEEK_DAYS
FORECAST_BEST_MODEL_PATH = MODELS_DIR / "best_forecast_pipeline.joblib"
NOWCAST_BEST_MODEL_PATH = MODELS_DIR / "best_nowcast_pipeline.joblib"
TUNED_BEST_MODEL_PATH = MODELS_DIR / "best_tuned_forecast_pipeline.joblib"
TUNED_MODEL_METADATA_PATH = MODELS_DIR / "best_tuned_forecast_metadata.json"
PREDICTION_INTERVAL_PATH = MODELS_DIR / "prediction_interval.json"

# Tuning uses chronological folds within training only. Ten trials per model
# keeps the search broad enough to compare useful settings without an excessive
# runtime for this dataset and its tree ensembles.
TUNING_MODEL_NAMES = ["Random Forest", "HistGradientBoosting Poisson"]
TUNING_TRIALS_PER_MODEL = 10
TUNING_CV_SPLITS = 3
TUNING_MIN_TRAINING_ROWS = 2
TUNING_PARAMETER_SPACES = {
    "Random Forest": {
        "n_estimators": (100, 240),
        "max_depth": (8, 28),
        "min_samples_leaf": (1, 10),
        "max_features": (0.5, 1.0),
    },
    "HistGradientBoosting Poisson": {
        "max_iter": (80, 240),
        "learning_rate": (0.02, 0.15),
        "max_leaf_nodes": (15, 63),
        "min_samples_leaf": (10, 60),
        "l2_regularization": (0.01, 10.0),
    },
}
TUNING_FIGURE_SIZE = (10, 6)
TUNING_PLOT_DPI = 200
TUNING_LEARNING_CURVE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
PREDICTION_INTERVAL_LOWER_QUANTILE = 0.05
PREDICTION_INTERVAL_UPPER_QUANTILE = 0.95
PREDICTION_INTERVAL_CV_SPLITS = TIME_SERIES_CV_SPLITS
CROWD_LEVEL_QUANTILES = (1 / 3, 2 / 3)
CROWD_LEVEL_LABELS = ("Low", "Medium", "High")
EVALUATION_PERMUTATION_REPEATS = 5
EVALUATION_SHAP_SAMPLE_ROWS = 500
EVALUATION_MIN_GROUP_ROWS = 50
EVALUATION_MEAN_SHIFT_RELEVANCE_PEOPLE = 3.0
EVALUATION_ERROR_FIGURE_SIZE = (16, 12)
EVALUATION_FIGURE_SIZE = (12, 7)
EVALUATION_PLOT_DPI = EDA_FIGURE_DPI
EVALUATION_INTERVAL_LOWER_QUANTILE = 0.10
EVALUATION_INTERVAL_UPPER_QUANTILE = 0.90

if abs(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION - 1.0) > 1e-9:
    raise ValueError("Chronological split fractions must sum to 1.0.")
