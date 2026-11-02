"""Loading, schema validation, and datetime parsing for raw gym data."""

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from src.config import (
    DATE_COLUMN,
    DATETIME_COLUMN,
    EXPECTED_RAW_COLUMNS,
    INTEGER_COLUMNS,
    RAW_DATA_DIR,
    SPLIT_SUMMARY_PATH,
    SUPPORTED_RAW_EXTENSIONS,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    VALUE_RANGES,
)
from src.utils import get_logger

LOGGER = get_logger(__name__)
_ATTRIBUTE_PATTERN = re.compile(
    r"^@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))\s+(.+)$",
    re.IGNORECASE,
)


def _resolve_raw_path(path: Path | None) -> Path:
    """Choose the supplied raw file or the only supported file in the raw folder."""
    if path is not None:
        raw_path = Path(path)
        if raw_path.suffix.lower() not in SUPPORTED_RAW_EXTENSIONS:
            raise ValueError(
                f"Unsupported raw-data format {raw_path.suffix!r}; expected one of "
                f"{', '.join(SUPPORTED_RAW_EXTENSIONS)}."
            )
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw data file does not exist: {raw_path}")
        return raw_path

    candidates = sorted(
        file_path
        for extension in SUPPORTED_RAW_EXTENSIONS
        for file_path in RAW_DATA_DIR.glob(f"*{extension}")
        if file_path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No supported raw data file found in {RAW_DATA_DIR}. "
            "Place the dataset CSV or ARFF file there."
        )
    if len(candidates) > 1:
        candidate_names = ", ".join(file_path.name for file_path in candidates)
        raise ValueError(
            f"Found multiple raw data files in {RAW_DATA_DIR}: {candidate_names}. "
            "Pass the intended file path explicitly."
        )
    return candidates[0]


def _read_arff(path: Path) -> pd.DataFrame:
    """Read a dense ARFF file, including string attributes unsupported by SciPy."""
    attribute_names: list[str] = []
    data_line_number: int | None = None
    with path.open(encoding="utf-8") as raw_file:
        for line_number, line in enumerate(raw_file):
            stripped_line = line.strip()
            if stripped_line.lower() == "@data":
                data_line_number = line_number
                break
            match = _ATTRIBUTE_PATTERN.match(stripped_line)
            if match:
                attribute_name = next(
                    value for value in match.groups()[:3] if value is not None
                )
                attribute_names.append(attribute_name)

    if data_line_number is None:
        raise ValueError(f"ARFF file has no @DATA section: {path}")
    if not attribute_names:
        raise ValueError(f"ARFF file has no @ATTRIBUTE declarations: {path}")

    LOGGER.info("Reading ARFF data from %s", path.name)
    return pd.read_csv(
        path,
        skiprows=data_line_number + 1,
        header=None,
        names=attribute_names,
        na_values=["?"],
        quotechar="'",
        skipinitialspace=True,
    )


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load one CSV or ARFF dataset without modifying the raw file.

    Args:
        path: Optional explicit file path. If omitted, the raw-data folder must
            contain exactly one supported dataset file.

    Returns:
        A DataFrame containing the raw observations.

    Raises:
        FileNotFoundError: If no raw file is available.
        ValueError: If the file type is unsupported or raw-file selection is
            ambiguous.

    """
    raw_path = _resolve_raw_path(path)
    LOGGER.info("Loading raw dataset from %s", raw_path.name)
    if raw_path.suffix.lower() == ".arff":
        return _read_arff(raw_path)
    return pd.read_csv(raw_path)


def validate_schema(df: pd.DataFrame) -> None:
    """Check expected columns, data types, and allowed value ranges.

    All discovered violations are collected into one error so data issues can
    be fixed together instead of appearing one at a time.

    Args:
        df: Raw observations to validate.

    Raises:
        ValueError: If one or more schema or value checks fail.

    """
    violations: list[str] = []
    missing_columns = [column for column in EXPECTED_RAW_COLUMNS if column not in df]
    unexpected_columns = [column for column in df if column not in EXPECTED_RAW_COLUMNS]
    if missing_columns:
        violations.append(f"Missing expected columns: {missing_columns}")
    if unexpected_columns:
        violations.append(f"Unexpected columns: {unexpected_columns}")
    if df.empty:
        violations.append("The dataset contains no rows.")

    if DATE_COLUMN in df:
        non_null_dates = df[DATE_COLUMN].dropna()
        if not non_null_dates.map(lambda value: isinstance(value, str)).all():
            violations.append(
                f"Column '{DATE_COLUMN}' must contain strings; found dtype "
                f"{df[DATE_COLUMN].dtype}."
            )

    for column in INTEGER_COLUMNS:
        if column not in df:
            continue
        series = df[column]
        if not is_numeric_dtype(series.dtype):
            violations.append(
                f"Column '{column}' must be numeric; found dtype {series.dtype}."
            )
            continue
        if not is_integer_dtype(series.dtype):
            violations.append(
                f"Column '{column}' must use an integer dtype; found {series.dtype}."
            )
            non_null = series.dropna()
            if not np.equal(non_null, np.floor(non_null)).all():
                violations.append(f"Column '{column}' must contain integer values.")

    if "temperature" in df and not is_numeric_dtype(df["temperature"].dtype):
        violations.append(
            "Column 'temperature' must be numeric; found dtype "
            f"{df['temperature'].dtype}."
        )

    for column in [*INTEGER_COLUMNS, "temperature"]:
        if column not in df or not is_numeric_dtype(df[column].dtype):
            continue
        numeric_values = pd.to_numeric(df[column], errors="coerce").dropna()
        if not np.isfinite(numeric_values).all():
            violations.append(f"Column '{column}' contains non-finite values.")

    for column, (minimum, maximum) in VALUE_RANGES.items():
        if column not in df or not is_numeric_dtype(df[column].dtype):
            continue
        values = df[column].dropna()
        outside_range = pd.Series(False, index=values.index)
        if minimum is not None:
            outside_range |= values < minimum
        if maximum is not None:
            outside_range |= values > maximum
        if outside_range.any():
            observed_min = values.min()
            observed_max = values.max()
            violations.append(
                f"Column '{column}' must be in [{minimum}, {maximum}]; "
                f"found values from {observed_min} to {observed_max}."
            )

    if violations:
        message = "Schema validation failed:\n- " + "\n- ".join(violations)
        LOGGER.error(message)
        raise ValueError(message)
    LOGGER.info("Schema validation passed for %s rows.", len(df))


def parse_datetime(
    df: pd.DataFrame, *, allow_duplicate_timestamps: bool = False
) -> pd.DataFrame:
    """Parse dates as UTC timestamps, sort chronologically, and check duplicates.

    The input date values may carry different UTC offsets, as they do across
    daylight-saving changes. Normalizing to UTC gives a stable ordering while
    preserving the source date column unchanged.

    Args:
        df: Raw observations containing the date column.
        allow_duplicate_timestamps: Set only when an audit must retain duplicate
            rows long enough to report them. The default rejects duplicates.

    Returns:
        A sorted copy with a timezone-aware ``datetime`` column.

    Raises:
        ValueError: If dates cannot be parsed or duplicate instants exist and
            ``allow_duplicate_timestamps`` is false.

    """
    if DATE_COLUMN not in df:
        raise ValueError(f"Cannot parse datetimes: missing '{DATE_COLUMN}' column.")

    parsed_dates = pd.to_datetime(
        df[DATE_COLUMN], errors="coerce", format="mixed", utc=True
    )
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        invalid_examples = df.loc[invalid_dates, DATE_COLUMN].head(5).tolist()
        raise ValueError(
            f"Column '{DATE_COLUMN}' contains {int(invalid_dates.sum())} invalid or "
            f"missing date values; examples: {invalid_examples}."
        )

    result = df.copy()
    result[DATETIME_COLUMN] = parsed_dates
    result = result.sort_values(DATETIME_COLUMN, kind="stable").reset_index(drop=True)
    duplicate_rows = result[DATETIME_COLUMN].duplicated(keep=False)
    if duplicate_rows.any() and not allow_duplicate_timestamps:
        duplicate_values = (
            result.loc[duplicate_rows, DATETIME_COLUMN]
            .drop_duplicates()
            .head(5)
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Found {int(result.loc[duplicate_rows, DATETIME_COLUMN].nunique())} "
            f"duplicate timestamp values across {int(duplicate_rows.sum())} rows; "
            f"examples: {duplicate_values}."
        )
    LOGGER.info(
        "Parsed and chronologically sorted %s rows; duplicate timestamps: %s.",
        len(result),
        int(duplicate_rows.sum()),
    )
    return result


def split_chronologically(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split observations 70/15/15 after sorting them by timestamp.

    The chronological order is essential: later observations must not appear
    in the training partition when we evaluate future predictions. The
    timestamp column is retained in each returned partition so exact date
    ranges can be reported without consulting target values.

    Args:
        df: Raw rows containing the date column.

    Returns:
        A mapping with ``train``, ``validation``, and ``test`` DataFrames.

    Raises:
        ValueError: If fewer than three rows are available for non-empty
            partitions.

    """
    if len(df) < 3:
        raise ValueError("At least three rows are required for chronological splits.")
    ordered = parse_datetime(df)
    train_end = int(len(ordered) * TRAIN_FRACTION)
    validation_end = int(len(ordered) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    if train_end == 0 or validation_end <= train_end or validation_end >= len(ordered):
        raise ValueError(
            "The dataset is too small to create non-empty train, validation, "
            "and test partitions with the configured fractions."
        )

    splits = {
        "train": ordered.iloc[:train_end].copy(),
        "validation": ordered.iloc[train_end:validation_end].copy(),
        "test": ordered.iloc[validation_end:].copy(),
    }
    for name, split in splits.items():
        LOGGER.info(
            "%s split: %s rows, %s through %s.",
            name.title(),
            len(split),
            split[DATETIME_COLUMN].iloc[0].isoformat(),
            split[DATETIME_COLUMN].iloc[-1].isoformat(),
        )
    return splits


def write_split_summary(
    splits: Mapping[str, pd.DataFrame], path: Path = SPLIT_SUMMARY_PATH
) -> Path:
    """Write exact chronological ranges and row counts for each partition.

    Args:
        splits: Mapping returned by ``split_chronologically``.
        path: Optional output path, primarily useful for isolated tests.

    Returns:
        The path where the Markdown summary was saved.

    Raises:
        ValueError: If a split is empty or lacks parsed timestamps.

    """
    expected_names = ("train", "validation", "test")
    rows: list[str] = []
    for name in expected_names:
        if name not in splits:
            raise ValueError(f"Split summary is missing the '{name}' partition.")
        split = splits[name]
        if split.empty or DATETIME_COLUMN not in split:
            raise ValueError(
                f"The '{name}' partition must have rows and a "
                f"'{DATETIME_COLUMN}' column."
            )
        start = pd.Timestamp(split[DATETIME_COLUMN].iloc[0]).isoformat()
        end = pd.Timestamp(split[DATETIME_COLUMN].iloc[-1]).isoformat()
        rows.append(f"| {name.title()} | {len(split):,} | {start} | {end} |")

    content = "\n".join(
        [
            "# Chronological Data Split Summary",
            "",
            f"Rows are sorted by parsed UTC timestamp before splitting. The "
            f"configured fractions are {TRAIN_FRACTION:.0%} train, "
            f"{VALIDATION_FRACTION:.0%} validation, and {TEST_FRACTION:.0%} "
            "test; integer boundaries use floor rounding, with any remainder "
            "assigned to test. The test partition is reserved for final "
            "reporting and is not used in baseline selection.",
            "",
            "| Partition | Rows | First timestamp (UTC) | Last timestamp (UTC) |",
            "| --- | ---: | --- | --- |",
            *rows,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    LOGGER.info("Saved chronological split ranges to %s.", path)
    return path
