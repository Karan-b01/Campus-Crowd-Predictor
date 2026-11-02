# Validation Baselines

All scores below are validation-only. Validation covers 2016-09-11T02:36:51+00:00 through 2016-11-18T03:47:16+00:00 (UTC). The held-out test partition was not scored or used for baseline choices.

## Results

| Scenario | Baseline | MAE | RMSE | R2 | SMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Forecast | Training mean | 14.935 | 19.142 | -0.361 | 72.10 |
| Nowcast | Training mean | 14.935 | 19.142 | -0.361 | 72.10 |
| Forecast | Training day/hour average | 7.901 | 11.033 | 0.548 | 64.04 |
| Nowcast | Training day/hour average | 7.901 | 11.033 | 0.548 | 64.04 |
| Nowcast | Same time last week | 6.419 | 9.895 | 0.636 | 40.24 |
| Forecast | Linear Regression | 19.421 | 21.497 | -0.717 | 85.97 |

MAE is the primary metric, so lower is better. **For Forecast, the score to beat is the train-only `(day_of_week, hour)` average (MAE 7.901). For Nowcast, it is same-time-last-week (MAE 6.419).** The day/hour average is a strong, honest calendar baseline because the EDA showed daily and weekday/weekend patterns, and each pair's value is estimated only from training rows; unseen pairs fall back to the training mean.

For the day/hour baseline, 0 validation rows had a pair not observed in training and used the training-mean fallback. For the Scenario B same-time-last-week baseline, 472 validation rows lacked a usable lag and used the training-mean fallback.

The mean and day/hour baselines are repeated under both scenario labels because neither uses crowd lags. Same-time-last-week applies only to Nowcast. Linear Regression applies only to Forecast and is fitted with train-only preprocessing. All baseline predictions are clipped at zero before scoring.
