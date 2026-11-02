"""Central project configuration and filesystem paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SUPPORTED_RAW_EXTENSIONS = (".csv", ".arff")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_AUDIT_PATH = REPORTS_DIR / "data_audit.md"
FIGURES_DIR = REPORTS_DIR / "figures"
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

if abs(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION - 1.0) > 1e-9:
    raise ValueError("Chronological split fractions must sum to 1.0.")
