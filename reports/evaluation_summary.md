# Evaluation Summary

## Data-use policy and evaluation split

The saved test metrics below are copied from the final evaluation performed earlier. No test target values are used or recomputed in this analysis. To preserve the exact-once holdout rule, detailed error, interpretability, crowd-level, and interval analyses use validation data. Validation informed model tuning and selection, so these diagnostics are descriptive and are not an independent estimate of future performance.

## Real-world interpretation

The final test MAE was 11.98 people per reading. The training-period mean was 33.12 people, so this error is about 36.2% of a typical observed training count. This is an average absolute error; individual readings can be much farther off. The dataset does not establish the gym's maximum capacity, so the training mean should not be described as capacity.

Final test metrics are from the previously completed, single test evaluation:

| Metric | Test value |
| --- | ---: |
| MAE | 11.9816 |
| RMSE | 16.6492 |
| R2 | -0.3563 |
| SMAPE | 93.0045 |

For detailed validation diagnostics, MAE was 16.38, RMSE 20.92, and R2 -0.625.

## Crowd levels and classification accuracy

Thresholds use training-set terciles only: Low is at most 21.00; Medium is above that through 43.00; High is above 43.00 people. Ties at a threshold go to the lower level. These exact cut-points were added to model metadata so the app can use the same boundaries.
Validation crowd-level accuracy was 43.6%. This is an exact three-class match rate, not a replacement for count MAE; a prediction one person across a boundary counts as a class error.

![Validation crowd-level confusion matrix](figures/evaluation_crowd_level_confusion.png)

## Error patterns on validation data

The table shows the three highest-MAE subgroup slices with at least 50 validation readings. Positive prediction bias means overprediction. Training and validation target means are compared to check whether a simple period shift could explain each pattern.

| Group | Value | Rows | MAE | Prediction bias | Matching train mean | Validation mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hour | 20 | 399 | 32.13 | 32.13 | 46.99 | 34.93 |
| hour | 19 | 397 | 30.44 | 30.44 | 50.25 | 37.17 |
| hour | 21 | 400 | 30.03 | 30.03 | 48.26 | 36.99 |

- For **hour = 20**, a possible explanation is: The model learned busy evening attendance in earlier periods but may not adapt to a later drop. The validation target mean was 34.93 versus 46.99 in the matching training group (12.06 people lower); that comparison supports a time-period shift. Mean prediction bias was 32.13 people.
- For **hour = 19**, a possible explanation is: The model learned busy evening attendance in earlier periods but may not adapt to a later drop. The validation target mean was 37.17 versus 50.25 in the matching training group (13.08 people lower); that comparison supports a time-period shift. Mean prediction bias was 30.44 people.
- For **hour = 21**, a possible explanation is: The model learned busy evening attendance in earlier periods but may not adapt to a later drop. The validation target mean was 36.99 versus 48.26 in the matching training group (11.27 people lower); that comparison supports a time-period shift. Mean prediction bias was 30.03 people.

The full hour, day-of-week, holiday, semester, and crowd-level tables are in `reports/validation_error_analysis.csv`; validation predictions and absolute errors are in `reports/validation_predictions.csv`.

### Checking the evening overprediction hypothesis

The highest-error hours are all late evening. The next table compares the same hour and semester flag across training and validation. If attendance still drops within the same flag, the recorded semester indicator alone does not explain it; an unobserved calendar-period change is plausible, but the data cannot identify its cause.

| Hour | Semester status | Train rows | Validation rows | Train mean | Validation mean | Mean prediction | Change |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | during semester | 1,240 | 399 | 56.81 | 34.93 | 67.07 | -21.87 |
| 19 | during semester | 1,240 | 397 | 60.30 | 37.17 | 67.61 | -23.13 |
| 21 | during semester | 1,247 | 400 | 59.24 | 36.99 | 67.02 | -22.25 |

No holiday-flagged validation readings occur, so holiday-specific error cannot be estimated in this period.
Validation status counts: 9,328 during-semester, 0 outside-semester, and 0 semester-start rows. Absent statuses cannot be compared on this validation period.

![Validation error profiles](figures/evaluation_validation_error_profiles.png)

## Interpretability: what drives the forecast

Permutation importance ranks features by the validation MAE increase after shuffling: minutes_since_midnight (+3.37 MAE), hour_cos (+0.23 MAE), hour (+0.23 MAE), dow_sin (+0.15 MAE), is_weekend (+0.11 MAE). SHAP's mean absolute contributions rank: numeric__minutes_since_midnight (11.26), numeric__hour (3.61), numeric__is_during_semester (3.58), categorical__semester_phase_outside_semester (3.12), numeric__hour_sin (2.55). Agreement between these views is stronger evidence of model reliance than either alone. These are associations used by the fitted model, not causal effects of changing a calendar or weather condition.

In this model, minutes-since-midnight had mean absolute SHAP 11.26, while temperature had 0.71. This indicates that time of day dominates the model's reliance; temperature contributes a smaller signal in this validation sample. The semester flag and semester phase also rank near the top, so the model relies on academic-calendar context after time of day.

The hour dependence plot shows how the tree model changes its prediction across the daily cycle; the temperature plot shows whether weather adds a consistent signal after calendar features. These plots are based on a deterministic 500-row validation sample.

![Permutation importance](figures/evaluation_permutation_importance.png)

![SHAP global summary](figures/evaluation_shap_global_summary.png)

![SHAP dependence by hour](figures/evaluation_shap_dependence_hour.png)

![SHAP dependence by temperature](figures/evaluation_shap_dependence_temperature.png)

### Three local validation predictions

The examples span lower, middle, and higher predicted counts in the explanation sample. Waterfall plots show which encoded model features moved each prediction above or below the model's average.

| Timestamp | Actual people | Predicted people | Absolute error |
| --- | ---: | ---: | ---: |
| 2016-11-07 12:04:40+00:00 | 0.0 | 0.6 | 0.6 |
| 2016-11-16 23:12:00+00:00 | 35.0 | 49.2 | 14.2 |
| 2016-11-17 02:35:07+00:00 | 38.0 | 71.6 | 33.6 |

![Local explanation 1](figures/evaluation_shap_local_1.png)
![Local explanation 2](figures/evaluation_shap_local_2.png)
![Local explanation 3](figures/evaluation_shap_local_3.png)

## Uncertainty calibration

An 80% interval was calibrated from the 10th and 90th percentiles of training-only expanding-window out-of-fold residuals. Its coverage on validation data was 72.3% (6,742/9,328 readings); this is 7.7 percentage points below nominal coverage. Mean interval width was 23.2 people. This validation coverage is not a test-set coverage claim. The interval quantiles and observed validation coverage were saved in `models/prediction_interval.json`.

## What the model cannot do

- It cannot anticipate an unscheduled event, equipment closure, or sudden change in attendance because those inputs are absent.
- It cannot use live crowd counts; this is the calendar/weather Forecast scenario, not the lag-based Nowcast scenario.
- Its interval is a historical empirical band, not a guarantee that an individual reading will fall inside it.
- It cannot establish why a feature is associated with attendance or prove that changing temperature, semester status, or schedule would cause a crowd change.

## Limitations and generalization

The dataset describes one campus gym and spans about 19 months (August 2015 through March 2017), not a broad sample of gyms or years. There is no event schedule, exam calendar, equipment status, or live occupancy stream. The supplied file does not document whether counts are directly logged, curated, or synthetic. If synthetic, it may encode simplified patterns; if sensor-logged, site-specific collection practices and missing intervals may differ from future operations. Holiday groups have particularly limited support. Calendar routines, climate, campus events, capacity, and operating hours differ elsewhere, so these results may not generalize to another gym or later period. The negative final test R2 reinforces that limitation.
