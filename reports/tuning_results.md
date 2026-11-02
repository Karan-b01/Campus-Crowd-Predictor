# Scenario A Hyperparameter Tuning

Optuna used 10 trials per model and 3-fold expanding TimeSeriesSplit on training data only. The default-fit comparison plus Optuna search took 208.5 seconds. Validation data was not used to choose trial parameters; the tuned candidates were then compared on validation to select the final model. Test data was not used in tuning or model selection.

## Validation before and after tuning

| Model | Default MAE | Tuned MAE | Default RMSE | Tuned RMSE | Default R2 | Tuned R2 | CV MAE mean +/- std | Training MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 17.284 | 16.383 | 23.053 | 20.917 | -0.974 | -0.625 | 11.312 +/- 2.662 | 6.838 |
| HistGradientBoosting Poisson | 17.459 | 16.872 | 22.572 | 21.472 | -0.892 | -0.713 | 10.998 +/- 2.897 | 6.903 |

Chosen model: **Random Forest**, based on the lowest validation MAE among the two tuned candidates. The final model is fit on train plus validation after this choice.

Neither tuned candidate beats the established training-only day/hour average (validation MAE 7.901); the selected model is 8.482 people higher in MAE. It remains a limitation of the calendar-only model.

## Search spaces and selected parameters

### Random Forest

Best training-CV MAE: 11.063 people.

Selected parameters: `{"max_depth": 8, "max_features": 0.5852620618436457, "min_samples_leaf": 7, "n_estimators": 183}`

### HistGradientBoosting Poisson

Best training-CV MAE: 10.984 people.

Selected parameters: `{"l2_regularization": 7.886714129990489, "learning_rate": 0.028199996263578355, "max_iter": 177, "max_leaf_nodes": 18, "min_samples_leaf": 58}`

## Overfitting check and learning curve

Training MAE is measured on the same training rows used to fit the reported model and is therefore optimistic. Read it beside the later validation MAE and the expanding-window CV mean and spread. The table's CV summary is from a separate 5-fold expanding TimeSeriesSplit on training only. The learning curve grows the training prefix in time order and scores each fit on the same later validation block.

| Training rows | Train MAE | Validation MAE |
| ---: | ---: | ---: |
| 8,705 | 6.578 | 21.182 |
| 17,411 | 6.885 | 20.917 |
| 26,116 | 7.549 | 18.728 |
| 34,822 | 6.700 | 18.342 |
| 43,528 | 6.838 | 16.383 |

## Figures

- `reports/figures/optuna_optimization_history.png`
- `reports/figures/optuna_parameter_importance.png`
- `reports/figures/tuned_model_learning_curve.png`
