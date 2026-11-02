"""Central project configuration and filesystem paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TESTS_DIR = PROJECT_ROOT / "tests"

RANDOM_SEED = 42
TARGET_COLUMN = "number_people"
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

if abs(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION - 1.0) > 1e-9:
    raise ValueError("Chronological split fractions must sum to 1.0.")
