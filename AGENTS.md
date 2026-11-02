# AGENTS.md: Campus Crowd Prediction

> This file is the standing rulebook for every AI coding agent working in this
> repository. Read it fully before doing anything. If a task prompt conflicts
> with this file, stop and ask instead of guessing.

---

## 1. Mission

Build a professional, reproducible machine learning project that predicts how
crowded a campus gym is, using the OpenML dataset **"Crowdedness at the Campus
Gym" (ID 43752)**.

Final deliverables:

1. A clean, tested ML pipeline (data audit, features, models, tuning).
2. Rigorous, honest evaluation with error analysis and interpretability.
3. A polished Streamlit web app that looks like a real product.
4. Documentation: README, technical report, presentation outline, viva prep.

This is an academic project. The student must be able to **explain and defend
every decision in a viva**, so clarity and correctness matter more than
cleverness or the last decimal of accuracy.

---

## 2. Dataset notes (VERIFY against the real file, never assume)

Expected columns (confirm the real names, dtypes, and row count first):

| Column | Meaning |
|---|---|
| `number_people` | **Target.** People in the gym at that moment |
| `date` | Date-time string of the reading |
| `timestamp` | Seconds since midnight |
| `day_of_week` | 0-6 |
| `is_weekend` | 0/1 |
| `is_holiday` | 0/1 |
| `temperature` | Degrees Fahrenheit (verify the unit) |
| `is_start_of_semester` | 0/1 |
| `is_during_semester` | 0/1 |
| `month` | 1-12 |
| `hour` | 0-23 |

Rules:
- The raw CSV lives in `data/raw/`. **Never modify or overwrite raw data.**
  Write all derived data to `data/processed/`.
- If the real file differs from this table, report the difference and update
  `src/config.py`. Do not silently adapt.
- You may not have internet access. Do not assume you can download the data.
  If the file is missing, tell the user where to put it.

---

## 3. Repository structure

```
campus-crowd-prediction/
├── AGENTS.md
├── README.md
├── Makefile
├── pyproject.toml            # ruff + pytest config
├── requirements.txt
├── .streamlit/config.toml    # app theme
├── data/
│   ├── raw/                  # original CSV (read-only)
│   └── processed/
├── notebooks/                # 01_eda.ipynb, 02_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py             # paths, constants, seed, split fractions
│   ├── utils.py              # logger, seed helper
│   ├── data.py               # load, validate, parse, split
│   ├── features.py           # calendar + lag features, pipelines
│   ├── models.py             # model zoo, training, predict()
│   └── evaluate.py           # metrics, CV helpers, experiment logger
├── scripts/                  # train_models.py, tune.py, evaluate.py
├── app/                      # Streamlit app, components/, assets/
├── models/                   # saved pipeline + metadata JSON
├── reports/
│   ├── figures/              # all PNG figures (200 dpi)
│   └── *.md, experiments.csv
└── tests/
```

Business logic lives in `src/`. Notebooks and the app **import from `src/`**
and never duplicate logic.

---

## 4. Commands

Keep these working at all times (defined in the `Makefile`):

| Command | Purpose |
|---|---|
| `make setup` | Create venv and install dependencies |
| `make lint` | Run `ruff check` and `ruff format --check` |
| `make test` | Run `pytest` |
| `make train` | Run training and tuning scripts |
| `make app` | Launch the Streamlit app |
| `make all` | Lint, test, train, evaluate end to end |

Before declaring any task finished, run `make lint` and `make test` and show
the real output.

---

## 5. Engineering standards

- Python 3.10+. Type hints on all public functions. Google-style docstrings.
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, shap,
  xgboost or lightgbm, optuna, joblib, streamlit, pytest, ruff.
- One global seed, `RANDOM_SEED = 42`, defined in `src/config.py` and used
  everywhere (splits, models, Optuna samplers, numpy).
- All paths use `pathlib` and are defined in `src/config.py`. No hard-coded
  paths or magic numbers elsewhere.
- Use `logging` inside `src/`, not `print`. Notebooks and scripts may print.
- Functions are small and single-purpose. No copy-pasted logic.
- Every model is an **sklearn `Pipeline`**, so preprocessing is fitted on
  training data only.
- Pin dependency versions in `requirements.txt`.
- Do not add a new dependency without saying why.
- Keep commits small and focused. Write clear commit messages
  (`feat:`, `fix:`, `docs:`, `test:`).

---

## 5b. Visual standards (all figures and the app)

- One consistent seaborn theme and one colour palette across every figure.
- Every chart has a title, labelled axes with units, readable fonts, and a
  legend only when needed.
- Save figures at 200 dpi to `reports/figures/`, with descriptive names
  (`eda_hourly_pattern_weekday_vs_weekend.png`, not `fig3.png`).
- Never rely on colour alone to carry meaning (accessibility).

---

## 6. Data science rules (non-negotiable)

1. **This is a time series.** Never use random train/test splits or random
   K-fold. Use a chronological split (70% train / 15% validation / 15% test)
   and `TimeSeriesSplit` for cross-validation.
2. **The test set is used exactly once**, at the very end, for final
   reporting. No tuning, feature decisions, or model selection after seeing
   test results.
3. **No leakage.** Any feature built from past values (lags, rolling
   statistics) must be shifted so it uses only information available at
   prediction time. The current row's own target must never be included.
4. **Two scenarios, kept strictly separate:**
   - **Scenario A, Forecast (primary):** only information known in advance:
     calendar, holiday, semester, temperature. This is what the app uses,
     because users pick future dates where recent crowd counts do not exist.
   - **Scenario B, Nowcast (secondary):** Scenario A plus lagged crowd
     features. Reported as an accuracy upper bound if live data were available.
5. **Metrics:** report MAE (primary), RMSE, and R2 for every model. Because
   the target can be near zero, also use SMAPE or median absolute error
   instead of plain MAPE.
6. **Baselines first.** Every model must beat: the training mean, and the
   average per `(day_of_week, hour)` pair. State clearly which is the score
   to beat.
7. The target is a non-negative count. Clip predictions at zero and consider
   count-aware losses.
8. **Never fabricate results.** Every number in any document must come from
   code that was actually run and saved under `reports/`. If a result is not
   available, say so instead of estimating.
9. If a result looks too good (for example R2 above 0.99), assume leakage
   until proven otherwise and investigate.

---

## 7. Testing requirements

- Unit tests for: schema validation, splitting (no overlap, correct order),
  every feature function, `predict()`, and metrics.
- **Leakage tests**: changing a future target value must not change any
  earlier row's features.
- Tests use small synthetic DataFrames, not the full dataset, where possible.
- Aim for 80%+ coverage on `src/` core modules.

---

## 8. Definition of done (for every task)

A task is finished only when **all** of these are true:

- [ ] The code runs from a clean environment with the documented commands.
- [ ] `make lint` and `make test` pass, with output shown.
- [ ] Real outputs were produced and saved (tables, figures, reports).
- [ ] Important decisions are explained in plain language (comments or
      markdown), including *why*, not just *what*.
- [ ] Nothing was assumed silently. Assumptions and shortcuts are listed.
- [ ] Changes are committed with a clear message.

---

## 9. How to work

1. **Plan first.** Briefly state your plan before writing code.
2. **Make small, reviewable changes.** Do not refactor unrelated files.
3. **Run everything.** Execute the code and tests yourself, read the output,
   and fix errors before reporting back.
4. **Explain for a student.** Use clear language. Define any term a
   second-year student might not know.
5. **Stop and ask** when the data contradicts an assumption, a requirement is
   ambiguous, or a rule above would be violated.
6. **Report honestly.** If something failed, was skipped, or is weak, say so
   plainly. A flagged weakness is better than a hidden one.

---

## 10. Things you must NOT do

- Do not use random splits or random K-fold CV.
- Do not touch the test set before the final evaluation step.
- Do not tune hyperparameters on validation-plus-test data.
- Do not modify raw data.
- Do not hard-code results, metrics, or file paths.
- Do not invent datasets, citations, or numbers.
- Do not leave placeholder code (`pass`, `TODO`, fake data) in finished work.
- Do not install or run anything that sends the dataset to external services.
- Do not delete files you did not create without asking.

---

## 11. Final response format

End every task with a short report in this format:

```
SUMMARY: what was done (2-4 lines)
FILES: created or changed files
RESULTS: key real numbers or outputs, with file paths
CHECKS: lint / tests / run status
ASSUMPTIONS & RISKS: anything uncertain
NEXT STEP: what should happen next
```
