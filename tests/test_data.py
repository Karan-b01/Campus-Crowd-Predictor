"""Tests for raw-data schema checks and datetime handling."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import DATETIME_COLUMN, EXPECTED_RAW_COLUMNS, TEST_TEMP_DIR
from src.data import (
    load_raw,
    parse_datetime,
    split_chronologically,
    validate_schema,
    write_split_summary,
)


def _valid_data() -> pd.DataFrame:
    """Build a small valid dataset matching the expected raw schema."""
    return pd.DataFrame(
        {
            "number_people": [0, 12],
            "date": [
                "2017-03-18 18:42:28-07:00",
                "2017-03-18 18:52:35-07:00",
            ],
            "timestamp": [67348, 67955],
            "day_of_week": [5, 5],
            "is_weekend": [1, 1],
            "is_holiday": [0, 0],
            "temperature": [61.07, 61.07],
            "is_start_of_semester": [0, 0],
            "is_during_semester": [1, 1],
            "month": [3, 3],
            "hour": [18, 18],
        },
        columns=EXPECTED_RAW_COLUMNS,
    )


def _write_temporary_file(suffix: str, content: str) -> Path:
    """Write a fixture directly beneath the configured project temp path."""
    TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    file_descriptor, file_name = tempfile.mkstemp(suffix=suffix, dir=TEST_TEMP_DIR)
    os.close(file_descriptor)
    file_path = Path(file_name)
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_validate_schema_accepts_valid_rows() -> None:
    """Accept values with the expected columns, types, and ranges."""
    validate_schema(_valid_data())


def test_validate_schema_reports_missing_and_unexpected_columns() -> None:
    """List missing and extra columns in the validation error."""
    data = _valid_data().drop(columns=["hour"]).assign(extra_feature=[1, 2])

    with pytest.raises(ValueError) as error:
        validate_schema(data)

    assert "Missing expected columns: ['hour']" in str(error.value)
    assert "Unexpected columns: ['extra_feature']" in str(error.value)


def test_validate_schema_reports_dtype_violations() -> None:
    """Reject dates that are not strings and numeric features stored as text."""
    data = _valid_data()
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data["temperature"] = ["warm", "cold"]

    with pytest.raises(ValueError) as error:
        validate_schema(data)

    assert "Column 'date' must contain strings" in str(error.value)
    assert "Column 'temperature' must be numeric" in str(error.value)


def test_validate_schema_rejects_integer_column_with_float_dtype() -> None:
    """Reject integer-like values stored under a floating-point dtype."""
    data = _valid_data()
    data["hour"] = data["hour"].astype(float)

    with pytest.raises(ValueError, match="must use an integer dtype"):
        validate_schema(data)


def test_validate_schema_lists_all_out_of_range_values() -> None:
    """Aggregate range failures rather than stopping at the first bad column."""
    data = _valid_data()
    data.loc[0, "number_people"] = -1
    data.loc[0, "hour"] = 24
    data.loc[1, "day_of_week"] = 7
    data.loc[1, "is_holiday"] = 2

    with pytest.raises(ValueError) as error:
        validate_schema(data)

    message = str(error.value)
    assert "'number_people' must be in [0, None]" in message
    assert "'hour' must be in [0, 23]" in message
    assert "'day_of_week' must be in [0, 6]" in message
    assert "'is_holiday' must be in [0, 1]" in message


def test_parse_datetime_sorts_and_rejects_duplicate_instants() -> None:
    """Sort valid timestamps and reject two local times for one UTC instant."""
    data = _valid_data().iloc[::-1].reset_index(drop=True)
    data.loc[1, "date"] = "2017-03-19 01:52:35+00:00"

    with pytest.raises(ValueError, match="duplicate timestamp values"):
        parse_datetime(data)

    sorted_data = parse_datetime(data, allow_duplicate_timestamps=True)
    assert sorted_data["datetime"].is_monotonic_increasing
    assert sorted_data["datetime"].duplicated(keep=False).all()


def test_load_raw_reads_arff_string_attributes() -> None:
    """Read the supplied ARFF format, including its string date field."""
    raw_path = _write_temporary_file(
        ".arff",
        """@RELATION sample
@ATTRIBUTE date STRING
@ATTRIBUTE count INTEGER
@DATA
'2024-01-01 08:00:00-08:00',3
'2024-01-01 08:10:00-08:00',4
""",
    )
    try:
        loaded = load_raw(raw_path)
        assert loaded["date"].tolist() == [
            "2024-01-01 08:00:00-08:00",
            "2024-01-01 08:10:00-08:00",
        ]
        assert loaded["count"].tolist() == [3, 4]
    finally:
        raw_path.unlink(missing_ok=True)


def test_load_raw_reads_csv() -> None:
    """Keep CSV input supported as described in the repository notes."""
    raw_path = _write_temporary_file(".csv", "date,count\n2024-01-01,3\n")
    try:
        loaded = load_raw(raw_path)
        assert loaded.to_dict(orient="records") == [{"date": "2024-01-01", "count": 3}]
    finally:
        raw_path.unlink(missing_ok=True)


def test_chronological_split_has_no_overlap_and_preserves_order() -> None:
    """Create 70/15/15 non-overlapping partitions in timestamp order."""
    base_row = _valid_data().iloc[0].to_dict()
    times = pd.date_range("2024-01-01 08:00:00", periods=20, freq="10min", tz="UTC")
    rows = []
    for timestamp in times:
        row = base_row.copy()
        row["date"] = timestamp.isoformat()
        rows.append(row)

    splits = split_chronologically(pd.DataFrame(rows, columns=EXPECTED_RAW_COLUMNS))
    train, validation, test = (
        splits["train"],
        splits["validation"],
        splits["test"],
    )

    assert [len(train), len(validation), len(test)] == [14, 3, 3]
    assert train[DATETIME_COLUMN].max() < validation[DATETIME_COLUMN].min()
    assert validation[DATETIME_COLUMN].max() < test[DATETIME_COLUMN].min()
    assert train[DATETIME_COLUMN].is_monotonic_increasing
    assert validation[DATETIME_COLUMN].is_monotonic_increasing
    assert test[DATETIME_COLUMN].is_monotonic_increasing
    assert not set(train[DATETIME_COLUMN]) & set(validation[DATETIME_COLUMN])
    assert not set(validation[DATETIME_COLUMN]) & set(test[DATETIME_COLUMN])

    summary_path = _write_temporary_file(".md", "")
    try:
        write_split_summary(splits, summary_path)
        summary = summary_path.read_text(encoding="utf-8")
        assert times[0].isoformat() in summary
        assert times[-1].isoformat() in summary
    finally:
        summary_path.unlink(missing_ok=True)
