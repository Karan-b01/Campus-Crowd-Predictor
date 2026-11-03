# 🏋️ Campus Gym Crowd Forecaster

> **An End-to-End Machine Learning System & Interactive Occupancy Intelligence Dashboard**  
> *Developed for the **AWS Student Builder Group — VIT Vellore** Task*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1%2B-EB5424)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-4.1%2B-1E88E5)](https://optuna.org/)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-000000?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/Tests-54%2F54%20Passing-10B981?logo=pytest&logoColor=white)](https://pytest.org/)
[![Affiliation](https://img.shields.io/badge/AWS%20Student%20Builder%20Group-VIT%20Vellore-FF9900?logo=amazon-aws&logoColor=white)](https://vit.ac.in/)

---

## 📌 Executive Summary

University fitness facilities face severe congestion challenges: unpredictable crowd surges, extreme wait times for strength and cardio equipment during post-lecture rush hours, and resource under-utilization during morning lulls. 

This project delivers a **production-grade Machine Learning pipeline and SaaS-style Streamlit dashboard** that predicts gym occupancy in 10-minute increments. Built with **strict chronological data integrity** to prevent temporal leakage, the system translates raw campus calendar and weather signals into actionable workout recommendations, 24-hour demand curves, weekly planning heatmaps, and empirical prediction uncertainty intervals.

```
                    ┌───────────────────────────┐
                    │  Raw Campus Observations  │
                    │   (62,184 readings / 19m)  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Chronological Split & ETL │
                    │   70% Train | 15% Val     │
                    │   15% Single-Blind Test   │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Harmonic Feature Builder  │
                    │ (Sin/Cos, Cycles, Buckets)│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Tuned Random Forest     │
                    │ (Optuna Chrono CV + Bands)│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
      ┌───────────────────────────────────────────────────────┐
      │          Streamlit Occupancy Intelligence App         │
      │  ⏱️ Forecast  |  📅 Planner  |  📊 EDA  |  ℹ️ About   │
      └───────────────────────────────────────────────────────┘
```

---

## 🎯 Key Capabilities & Dashboard Pages

The web application is designed with a modern, calm aesthetic utilizing **Deep Navy (`#0F172A`)**, **Accent Teal (`#0D9488`)**, and accessible traffic-light crowd indicators with graceful light/dark mode support.

### 1. ⏱️ Live Forecast & Advisor
* **Interactive Parameters:** Date picker, 10-minute step time slider ($00:00 - 23:50$), temperature slider ($30^\circ\text{F} - 105^\circ\text{F}$), academic semester status, and official holiday toggles.
* **Hero KPI Row:**
  * **Predicted Crowd:** Expected number of people on the gym floor (with % of typical peak capacity).
  * **Crowd Level Badge:** Color-coded status badge with real-world equipment wait descriptions:
    * 🟢 **Low** ($\le 21$ people): Plenty of open equipment and quiet training space.
    * 🟡 **Medium** ($22 - 43$ people): Moderate activity; short waits for popular stations.
    * 🔴 **High** ($> 43$ people): Peak rush; expect queues for squat racks and benches.
  * **80% Empirical Prediction Range:** Asymmetric out-of-fold residual bounds calibrated on training history.
* **Gym Peak Gauge:** Interactive Plotly semicircular gauge benchmarking the prediction against the gym's typical operational peak ($55$ people).
* **24-Hour Timeline Chart:** Continuous full-day occupancy curve with shaded 80% confidence bands, selected moment marker, Low/Medium threshold guides, and a highlighted **⭐ Quietest Window**.
* **Actionable Recommendation Banner:** Natural-language advice highlighting the optimal time window to visit and peak rush hours to avoid.
* **Sensitivity & Driver Analysis:** Real-time breakdown of local factors ($\Delta$ people from time of day, academic session, day of week, temperature) paired with a global SHAP feature importance bar chart.

### 2. 📅 Weekly Planner
* **7-Day Hourly Heatmap:** 24-hour $\times$ 7-day density matrix across Monday through Sunday.
* **Quick-Schedule Cards:** Actionable recommendations for *Early Bird* ($6:00 - 9:00\text{ AM}$), *Midday Lull* ($1:30 - 3:30\text{ PM}$), and *Late Night* ($9:30 - 11:00\text{ PM}$) sessions.
* **Scenario Planning:** Test forward planning across differing weather forecasts and semester breaks.

### 3. 📊 Facility Insights & EDA
* **Hourly Dynamics:** Bimodal weekday curve (surging to an intense peak at **6:00 PM**, mean $52.0$ people) versus weekend curve (peaking earlier at **5:00 PM**, mean $39.3$ people).
* **Day-of-Week Profile:** Tuesday and Wednesday record the highest sustained volume; Sundays remain the quietest days.
* **Semester Impact:** In-session campus periods nearly **double** average attendance ($34.5$ vs. $18.5$ people).
* **Weather Relationship:** Moderate positive correlation ($r = 0.373$) with ambient temperature.
* **Feature Importance:** Agreement between Permutation Importance and SHAP attribution confirming time-of-day and semester context as primary drivers.

### 4. ℹ️ Model Architecture & About
* **Methodology Disclosures:** Full dataset documentation, preprocessing pipelines, and hyperparameters.
* **Final Test Scorecards:** $\text{MAE} = 11.98$, $\text{RMSE} = 16.65$, $R^2 = -0.356$, $\text{SMAPE} = 93.0\%$.
* **Forecast vs. Nowcast Comparison:** Detailed operational analysis comparing calendar-only forecasting with real-time sensor lag models.
* **Honest Guardrails:** Clear explanations of single-facility transferability, unobserved campus events, evening period drift, and empirical residual guarantees.

---

## 📸 Verified Application Screenshots

Automated headless browser testing was performed across desktop ($1440 \times 1150$) and mobile ($390 \times 844$) viewports. Output figures are stored in [`reports/figures/`](reports/figures):

| Viewport / Page | Preview / File Path | Description |
|---|---|---|
| **Live Forecast (Desktop)** | [`reports/figures/app_predict_desktop.png`](reports/figures/app_predict_desktop.png) | Hero KPIs, 24h curve, peak gauge, advice banner, local drivers |
| **Weekly Planner (Desktop)** | [`reports/figures/app_planner_desktop.png`](reports/figures/app_planner_desktop.png) | 24x7 hourly heatmap matrix and time slot cards |
| **Facility Insights (Desktop)** | [`reports/figures/app_insights_desktop.png`](reports/figures/app_insights_desktop.png) | Interactive Plotly EDA tabs, weekday/weekend curves, day volume |
| **Model & About (Desktop)** | [`reports/figures/app_about_desktop.png`](reports/figures/app_about_desktop.png) | Test scorecards, Forecast vs Nowcast comparison, guardrails |
| **Sunday Scenario (Desktop)** | [`reports/figures/app_predict_sunday_desktop.png`](reports/figures/app_predict_sunday_desktop.png) | Weekend curve shift and earlier 5:00 PM peak validation |
| **Off-Hours (3:00 AM)** | [`reports/figures/app_predict_offhours_desktop.png`](reports/figures/app_predict_offhours_desktop.png) | Closed facility alert and 0-count baseline estimation |
| **Holiday Mode (Desktop)** | [`reports/figures/app_predict_holiday_desktop.png`](reports/figures/app_predict_holiday_desktop.png) | Holiday alert notice and density adjustments |
| **Mobile Viewports** | [`reports/figures/app_predict_mobile.png`](reports/figures/app_predict_mobile.png) | Responsive card stacking and auto-collapsed navigation |

---

## 🔬 Modeling Methodology & Leakage Prevention

### 1. Chronological Splitting Discipline
Standard random cross-validation ($k$-fold) causes catastrophic temporal leakage in time-series data: past observations interpolate from future observations. We strictly partition the **62,184 observations** chronologically:
- **Training (70%):** August 15, 2015 – November 18, 2016 ($43,528$ rows)
- **Validation (15%):** Used for model ranking, error diagnosis, and hyperparameter tuning ($9,328$ rows)
- **Held-Out Test (15%):** November 18, 2016 – March 19, 2017 ($9,328$ rows) evaluated **exactly once** after all decisions were frozen.

### 2. Feature Engineering (Scenario A: Forecast)
To predict occupancy days or weeks in advance without assuming real-time turnstile counts:
- **Harmonic Cyclical Coordinates:** Encodes continuous diurnal and seasonal cycles without discontinuous boundary jumps:
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
  $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
- **Granular Cadence:** Minutes since midnight ($0 - 1439$).
- **Time-of-Day Buckets:** Categorical grouping (`early_morning`, `morning`, `midday`, `afternoon`, `evening`, `night`).
- **Academic Context:** Start-of-semester and active-semester interaction indicators.

### 3. Forecast vs. Nowcast Benchmark
In addition to the primary calendar Forecast model, we trained a real-time **Nowcast** benchmark incorporating lag signals ($10\text{m}$, $30\text{m}$, $60\text{m}$, $1\text{d}$, $1\text{w}$) and rolling historical statistics:

| Evaluation Scenario | Features Used | Validation MAE | Validation $R^2$ | Primary Purpose |
|---|---|---:|---:|---|
| **Scenario A: Forecast** *(App Model)* | Calendar, Clock Time, Weather Forecast | **17.28 people** | **-0.974** | Forward planning & scheduling |
| **Scenario B: Nowcast** | Clock Time + Real-time sensor lags | **3.47 people** | **+0.901** | Live facility occupancy tracking |

> **Key Takeaway:** Incorporating live turnstile lag features reduces error by **80.1%** ($17.28 \to 3.47\text{ MAE}$). However, for future scheduling, lag observations do not exist, making calendar forecasting essential.

---

## 🛠️ Repository Structure

```
campus-crowd-prediction/
├── .streamlit/
│   └── config.toml                  # Streamlit theme (navy/teal) & server settings
├── app/
│   ├── app.py                       # Main application entry point & page router
│   ├── main.py                      # Streamlit forwarding script
│   ├── utils.py                     # Cached loaders, batch predictions, drivers & math
│   ├── assets/
│   │   └── styles.css               # Design system, KPI cards, badges & responsive styles
│   └── components/
│       ├── kpi_cards.py             # Hero metric cards
│       ├── gauge.py                 # Semicircular peak capacity gauge
│       ├── charts.py                # 24-hour Plotly curve & weekly heatmap
│       ├── drivers.py               # Local factor sensitivity & global SHAP bar chart
│       └── eda_charts.py            # Interactive EDA figures
├── data/
│   ├── raw/
│   │   └── Campus_gym.arff          # Raw gym observations (62,184 rows)
│   └── processed/                   # Prepared data partitions
├── models/
│   ├── best_tuned_forecast_pipeline.joblib   # Tuned Random Forest model (6.0 MB)
│   ├── best_tuned_forecast_metadata.json     # Hyperparameters, test metrics & cut-points
│   ├── prediction_interval.json              # Calibrated empirical residual bounds
│   └── validation_shap_importance.csv        # SHAP feature importance attribution
├── notebooks/
│   ├── 01_eda.ipynb                 # Comprehensive exploratory data analysis
│   └── 02_evaluation.ipynb          # Error analysis, calibration & SHAP explanations
├── reports/
│   ├── figures/                     # Generated charts and app verification screenshots
│   ├── evaluation_summary.md        # Comprehensive validation diagnostic report
│   └── final_test_results.md        # Blinded test-set metrics
├── scripts/
│   ├── train_models.py              # Model zoo training script
│   ├── tune_models.py               # Optuna hyperparameter tuning
│   ├── evaluate_final_model.py      # Test set evaluation & SHAP generation
│   └── capture_screenshots.py       # Automated Edge CDP verification suite
├── src/
│   ├── config.py                    # Centralized paths, thresholds, and parameters
│   ├── data.py                      # ARFF loader, schema validator & chronological split
│   ├── features.py                  # Leakage-free feature pipelines
│   ├── models.py                    # Model zoo builders & single prediction helper
│   ├── evaluate.py                  # Time-series cross-validation & metric calculators
│   └── interpretability.py          # Permutation importance & SHAP explainers
├── tests/
│   ├── test_app.py                  # Streamlit app utility and smoke tests
│   ├── test_data.py                 # Schema validation & datetime parsing tests
│   ├── test_features.py             # Feature engineering tests
│   ├── test_models.py               # Model zoo pipeline tests
│   └── test_prediction_api.py       # Prediction interface tests
├── Makefile                         # Automation targets (lint, test, train, app)
├── pyproject.toml                   # Ruff and Pytest configurations
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python `3.10`, `3.11`, or `3.12`
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/campus-crowd-prediction.git
cd campus-crowd-prediction

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing, Quality & Automation

The repository enforces strict code quality and test coverage via `Makefile` commands:

| Command | Action | Status |
|---|---|---|
| `make lint` | Run Ruff linter and code formatter checks | **Pass (0 warnings)** |
| `make test` | Run full Pytest test suite | **Pass (54/54 passing)** |
| `make audit` | Validate raw dataset integrity and schema | **Pass** |
| `make train` | Fit model zoo and run chronological Optuna tuning | **Reproducible** |
| `make evaluate`| Compute out-of-fold intervals and SHAP attributions | **Reproducible** |
| `make app` | Launch the Streamlit dashboard | **Ready** |

---

## ☁️ Deployment Guidelines

### Option A: Streamlit Community Cloud
1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, branch (`main`), and set the main file path to:
   ```text
   app/app.py
   ```
4. **Model Artifact Note:** The production pipeline [`models/best_tuned_forecast_pipeline.joblib`](models/best_tuned_forecast_pipeline.joblib) is only **6.0 MB**, well within GitHub's 100 MB limit (no Git LFS required) and within Streamlit Cloud's 1 GB RAM tier.

### Option B: AWS Deployment (EC2 / ECS / App Runner)
For deployment within the AWS ecosystem (AWS Student Builder Group):

#### Dockerfile Example
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

#### AWS App Runner / ECS Deployment
1. Build and push the container image to **Amazon ECR** (Elastic Container Registry).
2. Deploy to **AWS App Runner** or **Amazon ECS (Fargate)** with 1 vCPU and 2 GB memory.
3. Configure the health check endpoint to `/_stcore/health`.

---

## 🏫 Institutional Affiliation

* **Organization:** AWS Student Builder Group
* **Institution:** Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India
* **Domain:** Cloud Computing, Machine Learning Systems & Applied AI

---

## 📄 License & Citations

This project is open-source under the [MIT License](LICENSE).

If you use this work for academic research, coursework, or demonstrations, please cite:
```bibtex
@misc{vit_vellore_campus_crowd_forecaster,
  author = {AWS Student Builder Group, VIT Vellore},
  title = {Campus Gym Crowd Forecaster: Leakage-Safe Time-Series Occupancy Prediction},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/your-username/campus-crowd-prediction}}
}
```
