# Final Test Results

Selected Forecast model: **Random Forest**. It was chosen using training-only chronological Optuna CV followed by validation MAE. The final pipeline was then retrained on the combined training and validation partitions.

Final fit dates: 2015-08-15 00:00:11+00:00 through 2016-11-18 03:47:16+00:00.
Held-out test dates: 2016-11-18 03:57:22+00:00 through 2017-03-19 02:22:51+00:00 (9,328 rows).

## Test metrics

| Metric | Test score |
| --- | ---: |
| MAE | 11.9816 |
| RMSE | 16.6492 |
| R2 | -0.3563 |
| SMAPE | 93.0045 |

**Test-set use:** the held-out target was evaluated exactly once, after all trial parameters and the model choice were fixed. It was not used for fitting, feature decisions, or tuning. No model, feature, or hyperparameter changes were made after viewing these test metrics.
