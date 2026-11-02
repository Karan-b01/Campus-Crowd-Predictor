# Model Comparison

Models were fit on 43,528 chronological training rows and ranked on 9,328 later validation rows. Every preprocessor was fitted inside its model Pipeline on training rows only. Five expanding-window TimeSeriesSplit folds were run within training with a 0-row gap. The current same-time target needs no extra forecast-horizon gap. The test partition was not used.

## Validation and training-window CV results

| Scenario | Model | Count loss | Val MAE | Val RMSE | Val R2 | Val SMAPE (%) | Fit seconds | CV MAE mean +/- std | CV RMSE mean +/- std | CV R2 mean +/- std | CV SMAPE mean +/- std |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Forecast | Ridge | Squared error | 19.428 | 21.503 | -0.717 | 85.974 | 0.10 | 11.125 +/- 4.061 | 14.883 +/- 4.920 | 0.434 +/- 0.254 | 53.308 +/- 5.422 |
| Forecast | ElasticNet | Squared error | 19.360 | 21.425 | -0.705 | 85.452 | 0.14 | 10.635 +/- 3.422 | 14.311 +/- 4.291 | 0.476 +/- 0.223 | 52.105 +/- 6.503 |
| Forecast | KNN | Squared error | 19.598 | 24.588 | -1.246 | 81.991 | 0.09 | 14.033 +/- 4.605 | 19.361 +/- 5.813 | 0.075 +/- 0.219 | 64.759 +/- 4.696 |
| Forecast | Decision Tree | Squared error | 17.565 | 23.680 | -1.083 | 77.797 | 0.29 | 13.185 +/- 3.368 | 18.748 +/- 4.162 | 0.062 +/- 0.411 | 56.258 +/- 10.913 |
| Forecast | Random Forest | Squared error | 17.284 | 23.053 | -0.974 | 78.692 | 14.18 | 12.590 +/- 2.781 | 17.648 +/- 3.479 | 0.178 +/- 0.296 | 60.754 +/- 13.018 |
| Forecast | ExtraTrees | Squared error | 17.938 | 23.627 | -1.074 | 80.182 | 9.63 | 12.724 +/- 1.856 | 18.153 +/- 2.814 | 0.091 +/- 0.398 | 60.911 +/- 14.918 |
| Forecast | HistGradientBoosting | Squared error | 17.907 | 22.768 | -0.926 | 77.410 | 0.90 | 11.977 +/- 2.938 | 16.626 +/- 4.176 | 0.299 +/- 0.184 | 57.108 +/- 11.626 |
| Forecast | XGBoost | Squared error | 18.079 | 22.838 | -0.937 | 78.583 | 0.63 | 11.135 +/- 2.936 | 15.670 +/- 4.221 | 0.378 +/- 0.196 | 53.616 +/- 10.159 |
| Forecast | HistGradientBoosting Poisson | Poisson | 17.459 | 22.572 | -0.892 | 78.917 | 1.02 | 11.199 +/- 3.090 | 15.851 +/- 4.393 | 0.362 +/- 0.212 | 57.293 +/- 11.451 |
| Nowcast | Random Forest | Squared error | 3.522 | 5.263 | 0.897 | 51.102 | 38.31 | 4.189 +/- 1.149 | 5.956 +/- 1.595 | 0.913 +/- 0.011 | 42.507 +/- 10.370 |
| Nowcast | HistGradientBoosting Poisson | Poisson | 3.470 | 5.167 | 0.901 | 51.534 | 1.31 | 4.344 +/- 1.463 | 6.515 +/- 2.734 | 0.896 +/- 0.050 | 43.958 +/- 11.174 |
| Nowcast | Decision Tree | Squared error | 4.291 | 6.408 | 0.847 | 49.000 | 0.54 | 5.354 +/- 1.479 | 7.589 +/- 1.837 | 0.858 +/- 0.010 | 39.422 +/- 7.893 |

Validation MAE is the primary model-selection metric. CV mean +/- sample standard deviation summarizes the five expanding historical folds; fit time measures the final fit on the training partition.

## Best models and score-to-beat comparison

- **Best Scenario A (Forecast): Random Forest**, validation MAE 17.284; the saved day/hour baseline MAE is 7.901.
- **Forecast baseline check:** no Scenario A model beat the day/hour baseline; the best model's MAE is 9.383 people higher.
- **Best Scenario B (Nowcast): HistGradientBoosting Poisson**, validation MAE 3.470; the saved same-time-last-week baseline MAE is 6.419.
- The Nowcast winner improves on the same-time-last-week baseline by 2.949 people MAE.
- Comparing the Nowcast winner with its same-model Forecast run isolates the lag-feature gain: its Forecast MAE was 17.459, a reduction of 13.989 people (80.1%).

## Count-aware loss

HistGradientBoosting with Poisson loss improved validation MAE by 0.447 versus the squared-error HistGradientBoosting model on Forecast validation. Both predict non-negative counts after output clipping. This is a validation comparison only; no count-loss choice was made from test data.

## Why the Forecast winner may fit this dataset

The winning tree model can make threshold splits and combine them into interactions. That matches the EDA evidence: occupancy changes cyclically through the day, the weekday/weekend curves differ, and the busiest day-hour cells are not interchangeable. It is the strongest candidate in this model set, but the result does not prove which individual feature caused the gain. It still has negative validation R2 (-0.974) and MAE 17.284, versus 7.901 for the day/hour baseline, so this Forecast result does not beat the benchmark. The candidate models have not yet provided a useful calendar-only forecast against this simple benchmark.

The dataset has 43,528 training examples, enough to learn structured calendar patterns but still limited compared with very large datasets. Flexible models can overfit, so the training-window CV spread and later validation score should be read together. The three Nowcast candidates were selected using Forecast validation rankings; their Nowcast scores are reported as a scenario comparison and remain validation-based.

## Figures

- Validation MAE: `reports/figures/model_validation_mae.png`
- Top-three Forecast predicted versus actual: `reports/figures/model_top3_predicted_vs_actual.png`
- Representative validation week: `reports/figures/model_representative_week_predictions.png`
