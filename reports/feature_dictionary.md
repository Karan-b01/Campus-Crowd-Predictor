# Feature Dictionary

`build_feature_set` in `src/features.py` creates two deliberately separate
matrices. Forecast is the app-ready scenario: it uses only calendar and weather
information. Nowcast adds historical gym counts and is an upper-bound scenario
when recent readings are available. The target column is never returned as a
feature.

## Forecast (Scenario A)

| Feature name | Scenario | How it is computed | Why it may help |
| --- | --- | --- | --- |
| `hour` | Forecast | Source hour, 0–23. | Lets tree models split directly at observed busy hours. |
| `hour_sin`, `hour_cos` | Forecast | Sine and cosine of `hour / 24` around a circle. | Places 23:00 near 00:00 instead of treating them as far apart. |
| `day_of_week` | Forecast | Source weekday index, Monday=0 through Sunday=6. | Preserves weekday identity for models that learn interactions. |
| `dow_sin`, `dow_cos` | Forecast | Sine and cosine of `day_of_week / 7` around a circle. | Represents Sunday and Monday as neighboring points in the weekly cycle. |
| `month` | Forecast | Source calendar month, 1–12. | Keeps a direct month signal for models that learn nonlinear splits. |
| `month_sin`, `month_cos` | Forecast | Sine and cosine of `(month - 1) / 12` around a circle. | Makes December and January adjacent in the annual cycle. The short dataset history means this signal should be validated cautiously. |
| `minutes_since_midnight` | Forecast | Source `timestamp` divided by seconds per minute; seconds are retained as a fraction of a minute. | Captures smooth within-hour timing in addition to the hour category. |
| `time_of_day_bucket` | Forecast | Hour grouped into early morning (0–5), morning (6–9), midday (10–13), afternoon (14–16), evening (17–21), and night (22–23). | Summarizes the overnight quiet period and daytime rise visible in the EDA. |
| `is_rush_hour` | Forecast | 1 at 17:00 or 18:00, otherwise 0. | The EDA shows the weekday peak at 18:00 and weekend peak at 17:00. |
| `season` | Forecast | Meteorological season mapped from source month. | Provides a coarse seasonal context alongside temperature. The data cover less than two full years, so the effect may not generalize. |
| `is_weekend` | Forecast | Source weekend indicator. | EDA showed different weekday and weekend hourly profiles. |
| `is_holiday` | Forecast | Source holiday indicator. | Adds known calendar context; EDA showed only a small number of flagged dates, so this feature may be uncertain. |
| `is_start_of_semester` | Forecast | Source semester-start indicator. | EDA showed a distinct occupancy distribution around semester starts. |
| `is_during_semester` | Forecast | Source in-semester indicator. | EDA showed higher average occupancy during the semester. |
| `semester_phase` | Forecast | `start_of_semester` when the start flag is 1; otherwise `during_semester` when the semester flag is 1; otherwise `outside_semester`. | Gives categorical models a simple phase representation while retaining the original flags. |
| `temperature` | Forecast | Source temperature in degrees Fahrenheit. | EDA showed a positive association with occupancy, though substantial variation remains at similar temperatures. |

The EDA did not provide enough support for a holiday-by-weekend interaction:
the holiday flag applies to few readings. It also did not show a clear
nonlinear temperature pattern, so no temperature bins are added. Week-of-year
effects are omitted because fewer than two full years do not establish a
repeating annual pattern. These omissions avoid adding complexity without
evidence.

## Nowcast (Scenario B)

Every lag below is aligned on the source-local clock and reads strictly earlier
target values. The short lags select the nearest recorded observation to the
requested time, but only when it is within the configured one-minute
tolerance; this accommodates seconds-level timing jitter without substituting
a stale value across a missing interval. Daily and weekly lags select the
nearest observation to the same local clock time on the previous day or week
within a separate five-minute tolerance, allowing measured sampling drift to
accumulate over longer periods. Rolling windows use only earlier readings and exclude
the current row. Missing history remains NaN here; the sklearn preprocessing
pipeline imputes it using values learned from the training partition only.

| Feature name | Scenario | How it is computed | Why it may help |
| --- | --- | --- | --- |
| `crowd_lag_10m` | Nowcast | Closest target reading to 10 minutes before the current local timestamp, within tolerance. | Captures the most recent short-term change in gym attendance. |
| `crowd_lag_30m` | Nowcast | Closest target reading to 30 minutes before the current local timestamp, within tolerance. | Captures a half-hour trend. |
| `crowd_lag_60m` | Nowcast | Closest target reading to 60 minutes before the current local timestamp, within tolerance. | Captures the recent hourly level. |
| `crowd_lag_1d` | Nowcast | Nearest target reading to the same local clock time on the previous day, within five minutes. | Compares with the previous day's corresponding time. |
| `crowd_lag_1w` | Nowcast | Nearest target reading to the same local clock time seven calendar days earlier, within five minutes. | Compares with the corresponding weekday and time in the previous week. |
| `crowd_rolling_mean_1h` | Nowcast | Mean target in the hour before the current row; current row excluded. | Smooths recent fluctuations into a short-term level. |
| `crowd_rolling_std_1h` | Nowcast | Sample standard deviation in the preceding hour; current row excluded. | Measures recent short-term variability. It remains NaN when there are too few observations to calculate it. |
| `crowd_rolling_mean_3h` | Nowcast | Mean target in the three hours before the current row; current row excluded. | Captures a broader recent level. |
| `crowd_rolling_std_3h` | Nowcast | Sample standard deviation in the preceding three hours; current row excluded. | Measures variability over a broader time window. It remains NaN when history is insufficient. |

## Preprocessing

`build_preprocessor` returns a `ColumnTransformer` with median imputation for
numeric fields and most-frequent imputation plus one-hot encoding for
categorical fields. Numeric standardization is optional: enable it for linear
or distance-based estimators, and leave it off for tree models. Put this
transformer and the estimator in the `Pipeline` returned by
`build_model_pipeline`; fit that pipeline on training data only so imputation,
category levels, and scaling statistics cannot leak from validation or test
rows.
