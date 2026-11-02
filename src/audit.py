"""Generate a reproducible, descriptive audit of the raw gym dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    USFederalHolidayCalendar,
)

from src.config import (
    AUDIT_DISPLAY_PRECISION,
    AUDIT_EXAMPLE_LIMIT,
    AUDIT_ROW_EXAMPLE_LIMIT,
    BINARY_TRUE_VALUE,
    CATEGORICAL_AUDIT_COLUMNS,
    DATA_AUDIT_PATH,
    DATE_COLUMN,
    DATE_PREFIX_LENGTH,
    DATETIME_COLUMN,
    EXPECTED_INTEGER_COLUMNS,
    EXPECTED_RAW_COLUMNS,
    GAP_INTERVAL_THRESHOLD_MULTIPLIER,
    RAW_DATA_DIR,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    SUPPORTED_RAW_EXTENSIONS,
    TARGET_COLUMN,
    TARGET_OUTLIER_IQR_MULTIPLIER,
    WEEKEND_START_DAY,
    ZERO_OCCUPANCY_VALUE,
)
from src.data import load_raw, parse_datetime, validate_schema
from src.utils import get_logger

LOGGER = get_logger(__name__)


def _format_value(value: object) -> str:
    """Format scalar values consistently for Markdown tables."""
    if pd.isna(value):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{AUDIT_DISPLAY_PRECISION}f}"
    return str(value).replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Build a small Markdown table without adding a formatting dependency."""
    formatted_rows = [[_format_value(value) for value in row] for row in rows]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in formatted_rows]
    return "\n".join([header_line, separator_line, *body_lines])


def _source_path() -> Path:
    """Return the one supported source file selected by the data loader."""
    candidates = sorted(
        file_path
        for extension in SUPPORTED_RAW_EXTENSIONS
        for file_path in RAW_DATA_DIR.glob(f"*{extension}")
        if file_path.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            "The audit needs exactly one supported raw dataset file; found "
            f"{len(candidates)} in {RAW_DATA_DIR}."
        )
    return candidates[0]


def _frequency_summary(chronological: pd.DataFrame) -> tuple[float, pd.Series]:
    """Return the modal cadence in whole minutes and all positive time gaps."""
    intervals = chronological[DATETIME_COLUMN].diff().dt.total_seconds().dropna()
    positive_intervals = intervals[intervals > 0]
    if positive_intervals.empty:
        return float("nan"), positive_intervals
    rounded_minutes = (positive_intervals / SECONDS_PER_MINUTE).round()
    modal_minutes = float(rounded_minutes.mode().iloc[0])
    return modal_minutes * SECONDS_PER_MINUTE, positive_intervals


def _gap_rows(
    chronological: pd.DataFrame, nominal_interval_seconds: float
) -> tuple[list[list[object]], int]:
    """Describe intervals that exceed the expected sampling cadence."""
    if not np.isfinite(nominal_interval_seconds) or nominal_interval_seconds <= 0:
        return [], 0

    ordered_times = chronological[DATETIME_COLUMN]
    interval_seconds = ordered_times.diff().dt.total_seconds()
    gap_mask = (
        interval_seconds > nominal_interval_seconds * GAP_INTERVAL_THRESHOLD_MULTIPLIER
    )
    rows: list[list[object]] = []
    missing_readings_total = 0
    for row_index in np.flatnonzero(gap_mask.to_numpy()):
        elapsed_seconds = float(interval_seconds.iloc[row_index])
        missing_readings = max(
            int(round(elapsed_seconds / nominal_interval_seconds)) - 1, 1
        )
        missing_readings_total += missing_readings
        rows.append(
            [
                ordered_times.iloc[row_index - 1].isoformat(),
                ordered_times.iloc[row_index].isoformat(),
                elapsed_seconds,
                missing_readings,
            ]
        )
    return rows, missing_readings_total


def _category_sections(df: pd.DataFrame) -> list[str]:
    """Return value-count tables for categorical and binary features."""
    sections: list[str] = []
    for column in CATEGORICAL_AUDIT_COLUMNS:
        if column not in df:
            continue
        counts = df[column].value_counts(dropna=False).sort_index()
        rows = [[value, count] for value, count in counts.items()]
        sections.append(
            f"### `{column}`\n\n{_markdown_table(['Value', 'Count'], rows)}"
        )
    return sections


def _source_local_dates(df: pd.DataFrame) -> pd.Series:
    """Get calendar dates from source strings before UTC offset conversion."""
    date_strings = df[DATE_COLUMN].astype("string").str.slice(stop=DATE_PREFIX_LENGTH)
    return pd.to_datetime(date_strings, errors="coerce")


def _holiday_consistency(
    df: pd.DataFrame,
) -> tuple[int, list[str], list[str], list[str]]:
    """Compare holiday flags with the observed US federal holiday calendar."""
    local_dates = _source_local_dates(df)
    valid_dates = local_dates.dropna()
    if valid_dates.empty or "is_holiday" not in df:
        return 0, [], [], []

    calendar = USFederalHolidayCalendar()
    rules_without_observance = [
        Holiday(
            rule.name,
            month=rule.month,
            day=rule.day,
            offset=rule.offset,
            start_date=rule.start_date,
            end_date=rule.end_date,
        )
        for rule in calendar.rules
    ]
    actual_calendar_type = type(
        "ActualFederalHolidayCalendar",
        (AbstractHolidayCalendar,),
        {"rules": rules_without_observance},
    )
    actual_dates = set(
        actual_calendar_type()
        .holidays(start=valid_dates.min(), end=valid_dates.max())
        .date
    )
    observed_holidays = set(
        calendar.holidays(start=valid_dates.min(), end=valid_dates.max()).date
    )
    calendar_dates = actual_dates | observed_holidays
    flag_values = pd.to_numeric(df["is_holiday"], errors="coerce")
    flagged_dates = set(local_dates[flag_values.eq(BINARY_TRUE_VALUE)].dropna().dt.date)
    observed_dates = set(local_dates.dropna().dt.date)
    unsupported_flags = sorted(flagged_dates - calendar_dates)
    unflagged_federal_dates = sorted((calendar_dates & observed_dates) - flagged_dates)
    matched_flag_dates = sorted(flagged_dates & calendar_dates)
    return (
        len(matched_flag_dates),
        [str(day) for day in matched_flag_dates],
        [str(day) for day in unsupported_flags],
        [str(day) for day in unflagged_federal_dates],
    )


def generate_data_audit(
    output_path: Path = DATA_AUDIT_PATH,
) -> Path:
    """Write the full raw-data audit to a Markdown report.

    The audit never removes or writes back to observations. Rows with invalid
    dates, if any, remain in the report's shape and missing-value counts; they
    are excluded only from calculations that require chronological timestamps.

    Args:
        output_path: Destination for the generated Markdown report.

    Returns:
        The path to the saved report.

    """
    source_path = _source_path()
    df = load_raw(source_path)
    schema_error = ""
    try:
        validate_schema(df)
    except ValueError as error:
        schema_error = str(error)

    try:
        chronological = parse_datetime(df, allow_duplicate_timestamps=True)
    except ValueError as error:
        schema_error = "\n".join(part for part in [schema_error, str(error)] if part)
        parsed_dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce", utc=True)
        chronological = df.copy()
        chronological[DATETIME_COLUMN] = parsed_dates
        chronological = chronological.dropna(subset=[DATETIME_COLUMN]).sort_values(
            DATETIME_COLUMN, kind="stable"
        )
        chronological = chronological.reset_index(drop=True)

    actual_columns = list(df.columns)
    missing_columns = [column for column in EXPECTED_RAW_COLUMNS if column not in df]
    extra_columns = [
        column for column in actual_columns if column not in EXPECTED_RAW_COLUMNS
    ]
    column_match_note = (
        "Column names match the expected schema."
        if not missing_columns and not extra_columns
        else (
            f"Missing expected columns: {missing_columns}; "
            f"extra columns: {extra_columns}."
        )
    )

    nominal_interval_seconds, positive_intervals = _frequency_summary(chronological)
    gap_rows, estimated_missing_readings = _gap_rows(
        chronological, nominal_interval_seconds
    )
    duplicate_timestamp_rows = int(
        chronological[DATETIME_COLUMN].duplicated(keep=False).sum()
    )
    duplicate_timestamp_groups = int(
        chronological.loc[
            chronological[DATETIME_COLUMN].duplicated(keep=False), DATETIME_COLUMN
        ].nunique()
    )
    duplicate_full_rows = int(df.duplicated().sum())
    duplicate_full_mask = df.duplicated(keep=False)
    duplicate_full_examples = [
        [index, df.loc[index, DATE_COLUMN]]
        for index in df.index[duplicate_full_mask][:AUDIT_ROW_EXAMPLE_LIMIT]
    ]
    duplicate_timestamp_examples = [
        value.isoformat()
        for value in chronological.loc[
            chronological[DATETIME_COLUMN].duplicated(keep=False), DATETIME_COLUMN
        ].drop_duplicates()
    ]
    missing_counts = df.isna().sum()
    missing_row_mask = df.isna().any(axis=1)
    missing_row_examples = [
        [index, df.loc[index, DATE_COLUMN]]
        for index in df.index[missing_row_mask][:AUDIT_ROW_EXAMPLE_LIMIT]
    ]
    invalid_dates = pd.to_datetime(
        df[DATE_COLUMN], errors="coerce", format="mixed", utc=True
    ).isna()
    source_local_dates = _source_local_dates(df)
    numeric_summary = df.select_dtypes(include="number").describe().T
    summary_rows = [
        [index, *[numeric_summary.loc[index, key] for key in numeric_summary.columns]]
        for index in numeric_summary.index
    ]
    summary_headers = ["Column", *list(numeric_summary.columns)]

    if TARGET_COLUMN in df and pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        target = df[TARGET_COLUMN].dropna()
        target_skew = target.skew()
        zero_count = int(target.eq(ZERO_OCCUPANCY_VALUE).sum())
        first_quartile = target.quantile(0.25)
        third_quartile = target.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_fence = (
            first_quartile - TARGET_OUTLIER_IQR_MULTIPLIER * interquartile_range
        )
        upper_fence = (
            third_quartile + TARGET_OUTLIER_IQR_MULTIPLIER * interquartile_range
        )
        outlier_mask = target.lt(lower_fence) | target.gt(upper_fence)
        target_summary = [
            ["Non-missing target rows", len(target)],
            ["Mean", target.mean()],
            ["Median", target.median()],
            ["Standard deviation", target.std()],
            ["Skewness", target_skew],
            ["Zero counts", zero_count],
            [
                "Zero-count share (%)",
                zero_count / len(target) * 100 if len(target) else np.nan,
            ],
            ["IQR lower fence", lower_fence],
            ["IQR upper fence", upper_fence],
            ["IQR outlier rows", int(outlier_mask.sum())],
            ["Observed minimum", target.min()],
            ["Observed maximum", target.max()],
        ]
    else:
        target_summary = [
            ["Target summary", "Unavailable: target is missing or non-numeric."]
        ]

    timestamp_hour_mismatches = 0
    timestamp_hour_examples: list[list[object]] = []
    if {"timestamp", "hour"}.issubset(df.columns):
        expected_hour = (
            pd.to_numeric(df["timestamp"], errors="coerce") // SECONDS_PER_HOUR
        )
        observed_hour = pd.to_numeric(df["hour"], errors="coerce")
        mismatch_mask = (
            expected_hour.ne(observed_hour)
            & expected_hour.notna()
            & observed_hour.notna()
        )
        timestamp_hour_mismatches = int(mismatch_mask.sum())
        timestamp_hour_examples = [
            [
                df.index[index],
                df.loc[index, "timestamp"],
                df.loc[index, "hour"],
                expected_hour.loc[index],
            ]
            for index in df.index[mismatch_mask][:AUDIT_EXAMPLE_LIMIT]
        ]

    weekend_mismatches = 0
    weekend_examples: list[list[object]] = []
    if {"day_of_week", "is_weekend"}.issubset(df.columns):
        weekday = pd.to_numeric(df["day_of_week"], errors="coerce")
        weekend_flag = pd.to_numeric(df["is_weekend"], errors="coerce")
        expected_weekend = weekday.ge(WEEKEND_START_DAY).astype("float")
        mismatch_mask = (
            weekend_flag.ne(expected_weekend) & weekday.notna() & weekend_flag.notna()
        )
        weekend_mismatches = int(mismatch_mask.sum())
        weekend_examples = [
            [df.index[index], weekday.loc[index], weekend_flag.loc[index]]
            for index in df.index[mismatch_mask][:AUDIT_EXAMPLE_LIMIT]
        ]

    (
        holiday_match_count,
        matched_holiday_dates,
        unsupported_holidays,
        unflagged_holidays,
    ) = _holiday_consistency(df)
    min_datetime = (
        chronological[DATETIME_COLUMN].min() if not chronological.empty else pd.NaT
    )
    max_datetime = (
        chronological[DATETIME_COLUMN].max() if not chronological.empty else pd.NaT
    )
    local_date_range = (
        f"{source_local_dates.min().date()} to {source_local_dates.max().date()}"
        if source_local_dates.notna().any()
        else "unavailable"
    )
    schema_status = "Passed" if not schema_error else f"Issues found:\n\n{schema_error}"
    missing_columns_with_values = [
        f"`{column}` ({int(count):,})"
        for column, count in missing_counts.items()
        if count
    ]
    duplicate_decision = (
        "No duplicate timestamps or full duplicate rows were found."
        if not duplicate_timestamp_rows and not duplicate_full_rows
        else "Duplicate rows are retained and flagged; review their dates above before "
        "deciding whether they represent repeated sensor records."
    )
    gap_decision = (
        "No intervals exceeded the gap threshold."
        if not gap_rows
        else f"{len(gap_rows):,} long intervals are listed above; retain "
        "observations and "
        "do not fabricate missing measurements."
    )
    weekend_decision = (
        "Weekday and weekend flags agree for all rows."
        if not weekend_mismatches
        else f"{weekend_mismatches:,} weekday/weekend mismatches are flagged above; "
        "retain source values until their origin is checked."
    )
    hour_decision = (
        "Hour and seconds-since-midnight agree for all rows."
        if not timestamp_hour_mismatches
        else f"{timestamp_hour_mismatches:,} hour/timestamp mismatches are "
        "flagged above; "
        "do not silently recompute either column."
    )
    holiday_decision = (
        f"{len(unsupported_holidays):,} flagged dates are not federal holidays; "
        "keep the source flag unchanged and confirm the campus calendar."
        if unsupported_holidays
        else f"{len(unflagged_holidays):,} observed federal dates lack a flag; "
        "keep source flags and confirm whether they use the federal or campus "
        "calendar."
        if unflagged_holidays
        else "All observed federal dates have matching holiday flags."
    )

    report_sections = [
        "# Campus Gym Raw Data Audit",
        "",
        f"- **Source:** `{source_path.name}` ({source_path.suffix.lower()} format).",
        f"- **Actual shape:** {df.shape[0]:,} rows x {df.shape[1]} columns.",
        f"- **Datetime range:** {_format_value(min_datetime)} to "
        f"{_format_value(max_datetime)} (UTC).",
        f"- **Source-local date range:** {local_date_range}.",
        "- **Column check against `AGENTS.md`:** "
        + (
            "all expected column names are present"
            if not missing_columns and not extra_columns
            else column_match_note
        ),
        "- **File-format difference from `AGENTS.md`:** repository rules describe CSV; "
        f"but the supplied file is {source_path.suffix.lower()}. `load_raw()` "
        "supports this "
        "ARFF directly; the raw file was not converted or modified.",
        f"- **Sampling cadence:** modal interval "
        f"{nominal_interval_seconds / SECONDS_PER_MINUTE:.0f} minutes; "
        + (
            f"median positive interval: {positive_intervals.median():.1f} seconds."
            if not positive_intervals.empty
            else "no positive intervals could be calculated."
        ),
        "",
        "## Columns and dtypes",
        "",
        _markdown_table(
            ["Column", "Observed pandas dtype", "Expected in `AGENTS.md`"],
            [
                [
                    column,
                    str(df[column].dtype),
                    "string"
                    if column == DATE_COLUMN
                    else "integer"
                    if column in EXPECTED_INTEGER_COLUMNS
                    else "float",
                ]
                for column in actual_columns
            ],
        ),
        "",
        "## Schema validation",
        "",
        schema_status,
        "",
        "## Missing values and duplicates",
        "",
        _markdown_table(
            ["Column", "Missing rows"],
            [[column, count] for column, count in missing_counts.items()],
        ),
        "",
        f"- Exact duplicate rows: **{duplicate_full_rows:,}**.",
        f"- Rows in duplicate timestamp groups: **{duplicate_timestamp_rows:,}** "
        f"across **{duplicate_timestamp_groups:,}** distinct timestamps.",
        f"- Full duplicate examples (row index, date; up to "
        f"{AUDIT_ROW_EXAMPLE_LIMIT}): "
        f"{duplicate_full_examples or 'none'}.",
        f"- Duplicate timestamps in UTC: {duplicate_timestamp_examples or 'none'}.",
        f"- Rows with missing values: **{int(missing_row_mask.sum()):,}**; "
        f"examples (row index, date; up to {AUDIT_ROW_EXAMPLE_LIMIT}): "
        f"{missing_row_examples or 'none'}.",
        f"- Rows excluded from time calculations for invalid or missing dates: "
        f"**{int(invalid_dates.sum()):,}**.",
        "",
        "## Sampling gaps",
        "",
        f"Detected **{len(gap_rows):,}** intervals longer than "
        f"{GAP_INTERVAL_THRESHOLD_MULTIPLIER} times the modal "
        f"cadence; estimated missing readings: **{estimated_missing_readings:,}**. "
        "The estimate uses the modal cadence; source timestamps vary slightly.",
        "",
        _markdown_table(
            [
                "Previous timestamp (UTC)",
                "Next timestamp (UTC)",
                "Elapsed seconds",
                "Estimated missing readings",
            ],
            gap_rows or [["None", "None", 0, 0]],
        ),
        "",
        "## Numeric summary statistics",
        "",
        _markdown_table(summary_headers, summary_rows),
        "",
        "## Target distribution",
        "",
        _markdown_table(["Statistic", "Value"], target_summary),
        "",
        "## Categorical and binary value counts",
        "",
        *sum(([section, ""] for section in _category_sections(df)), []),
        "## Consistency checks",
        "",
        f"- `hour` vs. `timestamp // {SECONDS_PER_HOUR}`: "
        f"**{timestamp_hour_mismatches:,}** "
        f"mismatches. Examples (row index, timestamp, hour, expected): "
        f"{timestamp_hour_examples or 'none'}.",
        f"- `is_weekend` vs. `day_of_week in "
        f"{{{WEEKEND_START_DAY}, {WEEKEND_START_DAY + 1}}}`: "
        f"**{weekend_mismatches:,}** mismatches. Examples (row index, weekday, flag): "
        f"{weekend_examples or 'none'}.",
        f"- Distinct flagged dates matching a US federal holiday: "
        f"**{holiday_match_count:,}**.",
        f"- Flagged dates matching the calendar: {matched_holiday_dates or 'none'}.",
        f"- Flagged dates absent from the US federal calendar: "
        f"{unsupported_holidays or 'none'}.",
        f"- Observed federal dates without a holiday flag: "
        f"{unflagged_holidays or 'none'}.",
        "  This uses the U.S. federal calendar as a plausibility check; "
        "campus closures can differ from federal observances.",
        "",
        "## Findings and decisions",
        "",
        f"- **Input format:** source is {source_path.suffix.lower()}, whereas "
        "repository notes say CSV. The loader selects a parser by extension and "
        "preserves the raw file.",
        "- **Missing values:** "
        + (
            f"{', '.join(missing_columns_with_values)} contain missing cells. "
            "Keep rows for this audit; fit modeling imputation on training data only."
            if missing_columns_with_values
            else "none were found; no missing-value action is needed."
        ),
        f"- **Duplicates:** {duplicate_decision}",
        f"- **Sampling gaps:** {gap_decision}",
        "- **Zero occupancy:** retain zero counts; they may represent real gym-closed "
        "periods and are valid targets unless domain review proves otherwise.",
        f"- **Target outliers:** the {TARGET_OUTLIER_IQR_MULTIPLIER}xIQR rule flags "
        "values for review; it does not "
        "justify deletion. Keep outliers until a data error is documented.",
        f"- **Hour consistency:** {hour_decision}",
        f"- **Weekend consistency:** {weekend_decision}",
        f"- **Holiday plausibility:** {holiday_decision} Campus closures can differ "
        "from the federal calendar.",
        "",
        "No raw data or observation rows were modified by this audit.",
        "",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_sections), encoding="utf-8")
    LOGGER.info("Saved data audit to %s", output_path)
    return output_path


if __name__ == "__main__":
    generate_data_audit()
