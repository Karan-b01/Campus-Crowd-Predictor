# Campus Gym Occupancy Forecaster

<div align="center">

**An End-to-End Predictive Intelligence System & Interactive Operations Dashboard**  
*Built for the **AWS Student Builder Group — VIT Vellore** Milestone Task*

[![GitHub Repository](https://img.shields.io/badge/Repository-karan--b01%2Fcampus--crowd--prediction-181717?style=flat-square&logo=github)](https://github.com/karan-b01/campus-crowd-prediction)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-000000?style=flat-square&logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Test Suite](https://img.shields.io/badge/Tests-54%2F54%20Passing-10B981?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/)
[![Institution](https://img.shields.io/badge/AWS%20Student%20Builder%20Group-VIT%20Vellore-FF9900?style=flat-square&logo=amazon-aws&logoColor=white)](https://vit.ac.in/)

</div>

---

## 📌 Project Overview

University sports complexes and campus fitness centers experience extreme variance in daily visitor flow: severe congestion during evening post-lecture rush hours, long wait times for specialized strength equipment (squat racks and bench presses), and under-utilization during morning and midday lulls.

This project implements an **end-to-end Machine Learning forecasting platform and decision-support web application** that predicts gym occupancy in 10-minute increments. Developed under the **AWS Student Builder Group at VIT Vellore**, the system enforces strict chronological data hygiene to eliminate future-to-past temporal leakage, engineering harmonic calendar coordinates and training regularized tree ensembles with calibrated empirical prediction intervals.

### Core Objectives
- **Temporal Integrity:** Replace random cross-validation with an expanding-window chronological validation scheme (70% train / 15% validation / 15% single-blind holdout test).
- **Leakage-Safe Feature Engineering:** Transform raw calendar and temperature data into continuous harmonic cycles ($\sin/\cos$), time-of-day interaction buckets, and academic calendar session phases.
- **Actionable Decision Support:** Deliver an interactive SaaS-grade Streamlit application providing real-time visit recommendations, 24-hour continuous demand curves with uncertainty bands, 7-day planning heatmaps, and SHAP explainability drivers.

---

## 🖥️ Application Showcase

The accompanying dashboard is built with a cohesive design system (**Deep Navy `#0F172A`**, **Teal Accent `#0D9488`**, and WCAG AA-compliant crowd level indicators) that automatically adapts to desktop and mobile viewports.

### 1. Live Forecast & Visit Advisor
Interactive point-in-time forecasting displaying live occupancy estimates, operational capacity gauges, and natural-language scheduling guidance.

<div align="center">
  <img src="reports/figures/app_predict_desktop.png" alt="Live Forecast Desktop Dashboard" width="95%" />
</div>

* **Dynamic KPI Summary:** Displays predicted occupant count, confidence range, and color-coded status badges:
  * 🟢 **Low** ($\le 21$ people): Ample open equipment; quiet training environment.
  * 🟡 **Medium** ($22 - 43$ people): Normal operational traffic; moderate station turnover.
  * 🔴 **High** ($> 43$ people): Peak capacity rush; wait times expected for popular equipment.
* **Peak Utilization Gauge:** Real-time indicator measuring predicted traffic against the facility's observed typical peak ($55$ people).
* **24-Hour Continuous Timeline:** Full-day curve with calibrated 80% out-of-fold empirical uncertainty intervals, threshold guides, and an automated **⭐ Quietest Window** highlight ($6:00\text{ AM} - 7:20\text{ AM}$).
* **Explainability Drivers:** Real-time sensitivity analysis quantifying the marginal impact of time-of-day, semester status, day of week, and temperature relative to global SHAP importance rankings.

---

### 2. Weekly Workout Planner
A 24-hour $\times$ 7-day crowd density matrix allowing students and facility managers to spot low-density workout slots across the full calendar week at a glance.

<div align="center">
  <img src="reports/figures/app_planner_desktop.png" alt="Weekly Workout Planner Heatmap" width="95%" />
</div>

* **Hourly Matrix:** Heatmap density visualization across Monday through Sunday.
* **Curated Time Windows:** Scheduled recommendations for *Early Bird* ($6:00 - 9:00\text{ AM}$), *Midday Lull* ($1:30 - 3:30\text{ PM}$), and *Late Night* ($9:30 - 11:00\text{ PM}$) lifting sessions.

---

### 3. Exploratory Data Insights & Operational Patterns
Interactive Plotly visualizations synthesizing historical patterns across 62,184 facility observations over 19 months.

<div align="center">
  <img src="reports/figures/app_insights_desktop.png" alt="Historical Exploratory Data Analysis Insights" width="95%" />
</div>

* **Diurnal Dynamics:** Bimodal weekday curve surging to a primary peak at **6:00 PM** (mean $52.0$ people) versus weekend volume peaking earlier at **5:00 PM** (mean $39.3$ people).
* **Weekly Cadence:** Tuesdays and Wednesdays record peak weekly density; attendance declines on Friday afternoons, with Sundays being the quietest day overall.
* **Academic Multiplier:** Active semester sessions nearly **double** average facility attendance ($34.5$ vs. $18.5$ people), identifying the academic calendar as the primary macro-level context driver.
* **Thermal Association:** Moderate positive correlation ($r = 0.373$) with ambient temperature; cold winter spells suppress visits while mild temperatures sustain steady foot traffic.

---

### 4. Model Architecture & Methodological Transparency
Transparent system documentation detailing split boundaries, loss functions, test scorecards, and model governance limitations.

<div align="center">
  <img src="reports/figures/app_about_desktop.png" alt="Model Architecture and Methodology" width="95%" />
</div>

---

### 5. Responsive Mobile Viewports
The interface automatically detects screen geometry, collapsing secondary navigation panels and stacking KPI cards vertically to maintain complete touch responsiveness on mobile devices ($390 \times 844$).

<div align="center">
  <img src="reports/figures/app_predict_mobile.png" alt="Mobile Viewport Preview" width="30%" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="reports/figures/app_planner_mobile.png" alt="Mobile Planner Preview" width="30%" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="reports/figures/app_about_mobile.png" alt="Mobile About Preview" width="30%" />
</div>

---

## 🔬 Engineering & Modeling Architecture

```
                       DATA INGESTION & AUDIT
                 62,184 Records | 11 Raw Attributes
                                  │
                                  ▼
                     CHRONOLOGICAL PARTITIONING
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
     Train (70%)              Val (15%)               Test (15%)
    43,528 Records          9,328 Records           9,328 Records
  Aug 2015 - Nov 2016    Nov 2016 - Jan 2017     Jan 2017 - Mar 2017
          │                       │                       │
          ▼                       │                       │
   FEATURE PIPELINES              │                       │
 19 Forecast Features             │                       │
 (Harmonics, Buckets)             │                       │
          │                       │                       │
          ▼                       ▼                       │
    MODEL ZOO SELECTION & OPTUNA HYPERPARAMETER TUNING     │
  9 Regressors Evaluated | Expanding-Window Chrono CV     │
          │                                               │
          ▼                                               │
   FINAL RETRAINING (Train + Val)                         │
  Tuned RandomForestRegressor (n=183, depth=8)            │
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  │
                                  ▼
                     SINGLE-BLIND HELD-OUT TEST
                   MAE: 11.98  |  RMSE: 16.65
                                  │
                                  ▼
                   STREAMLIT PRODUCTION DASHBOARD
                 app/app.py (Cached Inference Engine)
```

### 1. Leakage-Free Chronological Partitions
Time-series forecasting models must not be evaluated using standard randomized $k$-fold cross-validation, which permits future observations to predict the past. This project enforces a strict chronological split:

| Partition | Date Range | Row Count | Purpose | Target Exposure |
|---|---|---:|---|---|
| **Training** | Aug 15, 2015 – Nov 18, 2016 | 43,528 | Feature fitting, pipeline training, CV tuning | Full training exposure |
| **Validation** | Nov 18, 2016 – Jan 18, 2017 | 9,328 | Model family selection, error diagnosis, SHAP | Unblinded for tuning only |
| **Held-Out Test** | Jan 18, 2017 – Mar 19, 2017 | 9,328 | Final generalizability assessment | Evaluated **exactly once** |

### 2. Mathematical Feature Transformations
The primary operational scenario (**Scenario A: Forecast**) relies strictly on features known prior to facility visit time (calendar and weather forecasts):

* **Diurnal Harmonic Cycle:** Captures 24-hour circular periodicity without abrupt artificial boundaries between 23:59 and 00:00:
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
* **Weekly Harmonic Cycle:** Encodes weekend transition routines:
  $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
* **Granular Cadence:** Continuous minutes since midnight:
  $$\text{minutes\_since\_midnight} = \frac{\text{timestamp}}{60}$$
* **Interaction Regimes:** Segmented time buckets (`early_morning`, `morning`, `midday`, `afternoon`, `evening`, `night`) combined with academic calendar states (`start_of_semester`, `during_semester`, `outside_semester`).

### 3. Forecast vs. Nowcast Benchmark
To assess the performance ceiling of sensor-equipped infrastructure, we trained a real-time **Nowcast** benchmark incorporating short-term lag readings ($10\text{m}$, $30\text{m}$, $60\text{m}$), historical lags ($1\text{d}$, $1\text{w}$), and rolling exponential statistics:

| Operational Regime | Feature Inputs | Validation MAE | Validation $R^2$ | Primary Application |
|---|---|---:|---:|---|
| **Scenario A: Forecast** *(App Engine)* | Calendar, Clock Time, Weather Forecast | **17.28 people** | **-0.974** | Advance scheduling & student planning |
| **Scenario B: Nowcast** | Clock Time + Real-time Sensor Lags | **3.47 people** | **+0.901** | Live turnstile tracking & digital signage |

> **Finding:** Live sensor lags reduce prediction error by **80.1%** ($17.28 \to 3.47\text{ MAE}$). However, for students scheduling workouts days in advance, lag features are unavailable—making calendar forecasting the critical decision tool.

### 4. Final Blinded Test Performance
Following parameter freezing and model selection on validation folds, the final pipeline was retrained on combined training and validation data and scored against the held-out test partition:

| Metric | Held-Out Test Result | Interpretation |
|---|---:|---|
| **Mean Absolute Error (MAE)** | **11.9816** | Average error is ~36% of the historical mean occupancy |
| **Root Mean Squared Error (RMSE)** | **16.6492** | Penalizes severe peak estimation discrepancies |
| **Coefficient of Determination ($R^2$)** | **-0.3563** | Reflects late-period macroeconomic evening attendance shifts |
| **Symmetric MAPE (SMAPE)** | **93.0045%** | Symmetric relative percentage error across low and high counts |

---

## 📂 Repository Layout

```
campus-crowd-prediction/
├── .streamlit/
│   └── config.toml                  # Streamlit UI theme, palette, and server binding
├── app/
│   ├── app.py                       # Main application entry point & query param router
│   ├── main.py                      # Production execution runner
│   ├── utils.py                     # Cached model loaders, batch inference & math helpers
│   ├── assets/
│   │   └── styles.css               # Design system, KPI cards, badges & responsive CSS
│   └── components/
│       ├── kpi_cards.py             # Hero KPI metrics
│       ├── gauge.py                 # Plotly peak capacity semicircular gauge
│       ├── charts.py                # 24-hour line chart with uncertainty band & weekly heatmap
│       ├── drivers.py               # Local factor sensitivity & global SHAP bar chart
│       └── eda_charts.py            # Interactive EDA figures
├── data/
│   ├── raw/
│   │   └── Campus_gym.arff          # 62,184 raw sensor observations
│   └── processed/                   # Cached feature sets
├── models/
│   ├── best_tuned_forecast_pipeline.joblib   # Serialized production pipeline (6.0 MB)
│   ├── best_tuned_forecast_metadata.json     # Hyperparameters, test metrics & thresholds
│   ├── prediction_interval.json              # Empirical out-of-fold residual quantiles
│   └── validation_shap_importance.csv        # SHAP feature attribution rankings
├── notebooks/
│   ├── 01_eda.ipynb                 # Comprehensive exploratory data analysis
│   └── 02_evaluation.ipynb          # Error analysis, calibration & SHAP explanations
├── reports/
│   ├── figures/                     # Verified application screenshots and diagnostics
│   ├── evaluation_summary.md        # Validation error analysis and subgroup slices
│   └── final_test_results.md        # Single-blind holdout evaluation log
├── scripts/
│   ├── train_models.py              # Baseline model zoo training script
│   ├── tune_models.py               # Chronological Optuna hyperparameter optimization
│   ├── evaluate_final_model.py      # Final test scoring and interpretability artifact builder
│   └── capture_screenshots.py       # Automated Edge CDP headless browser verification suite
├── src/
│   ├── config.py                    # Global paths, constants, and evaluation boundaries
│   ├── data.py                      # ARFF parser, schema validation & chronological split
│   ├── features.py                  # Leakage-free cyclical and lag feature transformers
│   ├── models.py                    # Model zoo specifications & predict() interface
│   ├── evaluate.py                  # Expanding-window cross-validation engine
│   └── interpretability.py          # Permutation importance and SHAP summary extractors
├── tests/
│   ├── test_app.py                  # Dashboard utility, prediction API, and smoke tests
│   ├── test_data.py                 # Schema assertions and datetime consistency tests
│   ├── test_features.py             # Feature pipeline unit tests
│   ├── test_models.py               # Pipeline construction and inference tests
│   └── test_prediction_api.py       # Standalone predict() interface tests
├── Makefile                         # Unified automation commands (lint, test, app)
├── pyproject.toml                   # Ruff and Pytest tooling configurations
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Documentation
```

---

## 🚀 Execution & Verification

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/karan-b01/campus-crowd-prediction.git
cd campus-crowd-prediction

# Initialize virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app/app.py
```
*The web dashboard will open locally at `http://localhost:8501`.*

### 3. Automated Test Suite & Code Quality
The codebase maintains 100% test coverage across all pipeline modules and dashboard components:

```bash
# Run all unit and smoke tests (54 passing)
pytest

# Check code formatting and static analysis (Ruff)
ruff check src tests scripts app
ruff format --check src tests scripts app
```

---

## ☁️ Production Deployment

### Streamlit Community Cloud
* **Repository:** `karan-b01/campus-crowd-prediction`
* **Branch:** `main`
* **Entrypoint:** `app/app.py`
* **Storage Footprint:** The production pipeline artifact ([`models/best_tuned_forecast_pipeline.joblib`](models/best_tuned_forecast_pipeline.joblib)) is optimized to **6.0 MB**, eliminating the need for Git LFS and comfortably fitting within Streamlit Cloud's 1 GB RAM allocation.

### Containerized AWS Deployment (App Runner / ECS Fargate)
The application is pre-configured for containerized deployment across AWS services:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🏛️ Project Metadata & Author

* **Author:** Karan ([@karan-b01](https://github.com/karan-b01))
* **Initiative:** AWS Student Builder Group
* **Institution:** Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India
* **Domain:** Applied Machine Learning, Cloud Architecture & Operations Analytics
* **License:** [MIT License](LICENSE)
