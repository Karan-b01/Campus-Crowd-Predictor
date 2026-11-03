"""Campus Gym Crowd Forecaster - Main Application Entrypoint.

A professional, production-grade Streamlit application for predicting and
visualizing campus gym occupancy, planning workout times, exploring
historical attendance patterns, and inspecting model performance.
"""

from __future__ import annotations

import sys
from datetime import date as Date
from datetime import datetime as DateTime
from datetime import time as Time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

try:
    from app.components.charts import (
        render_daily_forecast_chart,
        render_weekly_heatmap_chart,
    )
    from app.components.drivers import render_prediction_drivers
    from app.components.eda_charts import (
        render_day_of_week_chart,
        render_hourly_weekday_weekend_chart,
        render_semester_comparison_chart,
        render_temperature_scatter_chart,
    )
    from app.components.gauge import render_peak_gauge
    from app.components.kpi_cards import render_kpi_hero
    from app.utils import (
        GYM_TYPICAL_PEAK,
        compute_prediction_drivers,
        find_visiting_recommendation,
        get_crowd_thresholds,
        load_cached_artifacts,
        load_cached_eda_data,
        predict_day_curve,
        predict_single,
        predict_weekly_matrix,
    )
except (ImportError, ModuleNotFoundError):
    from components.charts import (
        render_daily_forecast_chart,
        render_weekly_heatmap_chart,
    )
    from components.drivers import render_prediction_drivers
    from components.eda_charts import (
        render_day_of_week_chart,
        render_hourly_weekday_weekend_chart,
        render_semester_comparison_chart,
        render_temperature_scatter_chart,
    )
    from components.gauge import render_peak_gauge
    from components.kpi_cards import render_kpi_hero

    from utils import (
        GYM_TYPICAL_PEAK,
        compute_prediction_drivers,
        find_visiting_recommendation,
        get_crowd_thresholds,
        load_cached_artifacts,
        load_cached_eda_data,
        predict_day_curve,
        predict_single,
        predict_weekly_matrix,
    )

# Set page configuration
st.set_page_config(
    page_title="Campus Gym Forecaster",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="auto",
)

# -----------------------------------------------------------------------------
# Style Injection
# -----------------------------------------------------------------------------
CSS_FILE = Path(__file__).resolve().parent / "assets" / "styles.css"
if CSS_FILE.is_file():
    with CSS_FILE.open(encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Context Inputs
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color); margin-bottom: 1rem;">
      <h2 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #0D9488; display: flex; align-items: center; gap: 0.5rem;">
        🏋️ GymForecaster
      </h2>
      <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #64748B;">Campus Occupancy Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

query_page = st.query_params.get("page", "").lower()
page_index_map = {
    "forecast": 0,
    "planner": 1,
    "insights": 2,
    "about": 3,
}
default_nav_idx = page_index_map.get(query_page, 0)

nav_page = st.sidebar.radio(
    "Navigation",
    options=[
        "⏱️ Live Forecast",
        "📅 Weekly Planner",
        "📊 Insights & Trends",
        "ℹ️ Model & About",
    ],
    index=default_nav_idx,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# Page 1: Live Forecast & Advisor
# -----------------------------------------------------------------------------
if nav_page == "⏱️ Live Forecast":
    st.sidebar.subheader("⚙️ Forecast Parameters")

    # Check query params for automated testing & deep links
    init_date = Date(2026, 11, 3)
    if (
        st.query_params.get("scenario") == "sunday"
        or st.query_params.get("sunday") == "1"
    ):
        init_date = Date(2026, 11, 8)  # Sunday

    init_minutes = 1050  # 17:30 (5:30 PM)
    if (
        st.query_params.get("scenario") == "offhours"
        or st.query_params.get("offhours") == "1"
    ):
        init_minutes = 180  # 3:00 AM

    init_holiday = False
    if (
        st.query_params.get("scenario") == "holiday"
        or st.query_params.get("holiday") == "1"
    ):
        init_holiday = True

    # 1. Date selection
    selected_date = st.sidebar.date_input(
        "Forecast Date",
        value=init_date,
        help="Choose any date to forecast expected gym crowd.",
    )

    # 2. Time selection (10-minute steps)
    time_minutes = st.sidebar.slider(
        "Time of Day",
        min_value=0,
        max_value=1430,
        value=init_minutes,
        step=10,
        help="Adjust target time in 10-minute increments.",
    )
    hour = time_minutes // 60
    minute = time_minutes % 60
    selected_time = Time(hour, minute)
    selected_dt = DateTime(
        selected_date.year, selected_date.month, selected_date.day, hour, minute
    )
    time_label = selected_dt.strftime("%I:%M %p").lstrip("0")

    st.sidebar.caption(f"🕒 Selected Time: **{time_label}** ({hour:02d}:{minute:02d})")

    # 3. Temperature input
    temp_val = st.sidebar.slider(
        "Expected Temperature (°F)",
        min_value=30.0,
        max_value=105.0,
        value=68.0,
        step=1.0,
        help="Local forecast temperature in Fahrenheit.",
    )

    # 4. Semester status
    semester_option = st.sidebar.selectbox(
        "Semester Status",
        options=[
            "Active Semester (Classes in Session)",
            "Semester Start (Orientation / Welcome Week)",
            "Academic Break (Summer / Winter Break)",
        ],
        index=0,
        help="Campus academic calendar phase strongly affects student attendance.",
    )
    is_during_sem = semester_option != "Academic Break (Summer / Winter Break)"
    is_start_sem = semester_option == "Semester Start (Orientation / Welcome Week)"

    # 5. Holiday flag
    is_holiday_flag = st.sidebar.checkbox(
        "Federal / Campus Holiday",
        value=init_holiday,
        help="Flag if campus facilities operate on holiday schedules.",
    )

    # Friendly input validation & notices
    if hour < 6 or hour >= 23:
        st.sidebar.warning(
            "⚠️ **Off-Hours Selected**: Standard facility operating hours are 6:00 AM – 11:00 PM. Forecast displays after-hours baseline activity."
        )
    if is_holiday_flag:
        st.sidebar.info(
            "ℹ️ **Holiday Mode**: Campus density is generally lower on holidays. Check local facility hours."
        )

    # Main Page Header
    st.markdown(
        f"""
        <div class="app-header">
          <h1 class="app-title">⏱️ Gym Occupancy Forecast</h1>
          <p class="app-subtitle">
            Forecasting crowd levels for <b>{selected_date.strftime("%A, %B %d, %Y")}</b> at <b>{time_label}</b>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Execute predictions with loading feedback
    try:
        with st.spinner("Calculating forecast..."):
            pred = predict_single(
                selected_dt,
                temp_val,
                is_holiday_flag,
                is_during_sem,
                is_start_sem,
            )
            day_curve = predict_day_curve(
                selected_date,
                temp_val,
                is_holiday_flag,
                is_during_sem,
                is_start_sem,
                step_minutes=10,
            )
            recommendations = find_visiting_recommendation(day_curve)
            driver_info = compute_prediction_drivers(
                selected_dt,
                temp_val,
                is_holiday_flag,
                is_during_sem,
                is_start_sem,
            )
            _, meta, _ = load_cached_artifacts()
            low_max, med_max = get_crowd_thresholds(meta)

        # 1. Hero KPI Cards
        render_kpi_hero(
            count=pred["count"],
            lower=pred["lower"],
            upper=pred["upper"],
            level=pred["level"],
            badge_class=pred["badge_class"],
            color=pred["color"],
            description=pred["description"],
            pct_of_peak=pred["pct_of_peak"],
        )

        # 2. Main Visuals Grid (Line Chart + Peak Gauge)
        col_chart, col_gauge = st.columns([2.3, 1.0])

        with col_chart:
            st.markdown("### 📈 24-Hour Day Timeline")
            render_daily_forecast_chart(day_curve, selected_dt, recommendations)

        with col_gauge:
            st.markdown("### 🎯 Relative to Peak")
            render_peak_gauge(
                count=pred["count"],
                color=pred["color"],
                level=pred["level"],
                low_max=low_max,
                med_max=med_max,
            )
            st.markdown(
                f"""
                <div style="font-size: 0.82rem; color: #64748B; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.6rem 0.8rem; margin-top: -0.5rem;">
                  <b>Thresholds:</b><br>
                  🟢 <b>Low:</b> 0 – {int(low_max)} people<br>
                  🟡 <b>Medium:</b> {int(low_max) + 1} – {int(med_max)} people<br>
                  🔴 <b>High:</b> > {int(med_max)} people<br>
                  📍 <b>Typical Peak:</b> ~{int(GYM_TYPICAL_PEAK)} people
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. Actionable Text Recommendation Banner
        st.markdown(
            f"""
            <div class="recommendation-banner">
              <div class="recommendation-title">
                ✨ Optimal Workout Advice for {selected_date.strftime("%A")}
              </div>
              <p class="recommendation-text">
                <b>Best time to visit:</b> <b>{recommendations["best_window"]}</b> (Expected crowd: <b>~{recommendations["best_avg_crowd"]} people</b>, Low level).<br>
                <b>Avoid peak rush:</b> <b>{recommendations["peak_window"]}</b> (Expected crowd: <b>~{recommendations["peak_count"]} people</b>, High level).
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. What's driving this prediction
        render_prediction_drivers(driver_info)

    except Exception as e:
        st.error(f"⚠️ Unable to generate prediction: {e}")
        st.info(
            "Please verify that model artifacts exist in `models/` by running `make tune`."
        )


# -----------------------------------------------------------------------------
# Page 2: Weekly Planner
# -----------------------------------------------------------------------------
elif nav_page == "📅 Weekly Planner":
    st.sidebar.subheader("⚙️ Planner Settings")
    week_ref_date = st.sidebar.date_input(
        "Reference Week Date",
        value=Date(2026, 11, 3),
        help="Select any day within your intended week.",
    )
    plan_temp = st.sidebar.slider(
        "Average Forecast Temp (°F)",
        min_value=30.0,
        max_value=105.0,
        value=68.0,
        step=1.0,
    )
    plan_semester = st.sidebar.selectbox(
        "Campus Semester State",
        options=["Active Semester", "Academic Break"],
        index=0,
    )
    is_during_plan = plan_semester == "Active Semester"

    st.markdown(
        """
        <div class="app-header">
          <h1 class="app-title">📅 Weekly Workout Planner</h1>
          <p class="app-subtitle">
            Hourly crowd density matrix across all 7 days. Spot quiet slots at a glance to schedule your week.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Generating weekly forecast matrix..."):
            matrix = predict_weekly_matrix(
                reference_date=week_ref_date,
                temperature=plan_temp,
                is_during_semester=is_during_plan,
                is_start_of_semester=False,
            )

        render_weekly_heatmap_chart(matrix)

        st.markdown("### 📋 Recommended Workout Windows by Day")

        col_w1, col_w2, col_w3 = st.columns(3)
        days = list(matrix.columns)

        with col_w1:
            st.markdown(
                """
                <div class="custom-card">
                  <h4 style="margin: 0 0 0.5rem 0; color: #0D9488;">🌅 Early Bird (6:00 – 9:00 AM)</h4>
                  <p style="font-size: 0.85rem; color: #64748B;">Ideal for focused lifting with zero wait times.</p>
                  <ul style="font-size: 0.85rem; padding-left: 1.2rem; margin-bottom: 0;">
                    <li><b>Mon – Wed:</b> ~12 – 18 people (Quiet)</li>
                    <li><b>Thu – Fri:</b> ~10 – 16 people (Very Quiet)</li>
                    <li><b>Weekends:</b> ~8 – 14 people (Empty)</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_w2:
            st.markdown(
                """
                <div class="custom-card">
                  <h4 style="margin: 0 0 0.5rem 0; color: #F59E0B;">☀️ Midday Lull (1:30 – 3:30 PM)</h4>
                  <p style="font-size: 0.85rem; color: #64748B;">Between afternoon lectures and post-work rush.</p>
                  <ul style="font-size: 0.85rem; padding-left: 1.2rem; margin-bottom: 0;">
                    <li><b>Monday:</b> ~22 people (Moderate)</li>
                    <li><b>Tue – Thu:</b> ~24 – 28 people (Moderate)</li>
                    <li><b>Friday:</b> ~18 people (Light)</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_w3:
            st.markdown(
                """
                <div class="custom-card">
                  <h4 style="margin: 0 0 0.5rem 0; color: #EF4444;">🌙 Late Night (9:30 – 11:00 PM)</h4>
                  <p style="font-size: 0.85rem; color: #64748B;">Evening crowd tapers off sharply before close.</p>
                  <ul style="font-size: 0.85rem; padding-left: 1.2rem; margin-bottom: 0;">
                    <li><b>Mon – Thu:</b> ~26 people (Down from peak)</li>
                    <li><b>Friday:</b> ~12 people (Very Quiet)</li>
                    <li><b>Saturday:</b> ~10 people (Quiet)</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Error computing weekly planner: {e}")


# -----------------------------------------------------------------------------
# Page 3: Insights & Trends
# -----------------------------------------------------------------------------
elif nav_page == "📊 Insights & Trends":
    st.markdown(
        """
        <div class="app-header">
          <h1 class="app-title">📊 Historical Insights & Patterns</h1>
          <p class="app-subtitle">
            Interactive exploratory findings from 62,184 historical gym observations across 19 months.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading historical aggregates..."):
            eda_data = load_cached_eda_data()

        tab1, tab2, tab3 = st.tabs(
            [
                "⏰ Hourly & Weekly Cycles",
                "🎓 Academic & Seasonal Factors",
                "🔬 Feature Importance",
            ]
        )

        with tab1:
            st.subheader("Hourly Dynamics: Weekdays vs Weekends")
            st.markdown(
                "Weekday attendance follows a pronounced bimodal routine, surging into an intense evening peak at **6:00 PM** (mean 52.0 people). Weekend attendance peaks earlier at **5:00 PM** (mean 39.3 people) with significantly lower overall density."
            )
            render_hourly_weekday_weekend_chart(eda_data["hourly_split"])

            st.markdown("---")
            st.subheader("Day-of-Week Attendance Profile")
            st.markdown(
                "Tuesdays and Wednesdays record the highest consistent gym traffic. Friday afternoons drop noticeably, and Sundays remain the quietest days across the academic year."
            )
            render_day_of_week_chart(eda_data["dow_summary"])

        with tab2:
            col_sem, col_temp = st.columns(2)

            with col_sem:
                st.subheader("Semester In-Session Effect")
                st.markdown(
                    "When classes are in session, average gym crowd nearly **doubles** from **18.5 to 34.5 people**, making the academic calendar the single strongest contextual macro-driver."
                )
                render_semester_comparison_chart(eda_data["sem_summary"])

            with col_temp:
                st.subheader("Temperature vs Occupancy")
                st.markdown(
                    "Observed crowd shows a moderate positive relationship with outdoor temperature (Pearson r = 0.373). Cold winter days experience suppressed visits, while mild weather correlates with steady campus activity."
                )
                render_temperature_scatter_chart(eda_data["temp_sample"])

        with tab3:
            st.subheader("What the Model Relies On: SHAP vs Permutation Importance")
            st.markdown(
                """
                Agreement between Permutation Importance (drop in validation MAE) and SHAP (attribution magnitude) confirms that **time-of-day** dominates the model's decisions, followed by **academic semester context** and **day of week**.
                """
            )
            shap_path = Path(PROJECT_ROOT / "models" / "validation_shap_importance.csv")
            if shap_path.is_file():
                shap_df = pd.read_csv(shap_path).head(10)
                st.dataframe(
                    shap_df,
                    column_config={
                        "feature": "Encoded Feature Name",
                        "mean_absolute_shap": st.column_config.NumberColumn(
                            "Mean |SHAP| Impact", format="%.3f"
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Feature importance artifact not found.")

    except Exception as e:
        st.error(f"Error loading EDA insights: {e}")


# -----------------------------------------------------------------------------
# Page 4: Model & About
# -----------------------------------------------------------------------------
elif nav_page == "ℹ️ Model & About":
    st.markdown(
        """
        <div class="app-header">
          <h1 class="app-title">ℹ️ Model Architecture & Methodology</h1>
          <p class="app-subtitle">
            System documentation, evaluation metrics, and engineering disclosures for Campus Gym Forecaster.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overview Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            """
            <div class="kpi-card">
              <div class="kpi-label">Final Test MAE</div>
              <div class="kpi-value" style="font-size: 1.8rem;">11.98<span class="kpi-unit">people</span></div>
              <div class="kpi-subtext">≈ 36% of typical count</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="kpi-card">
              <div class="kpi-label">Test RMSE</div>
              <div class="kpi-value" style="font-size: 1.8rem;">16.65<span class="kpi-unit">people</span></div>
              <div class="kpi-subtext">Penalizes large errors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="kpi-card">
              <div class="kpi-label">Test R² Score</div>
              <div class="kpi-value" style="font-size: 1.8rem;">-0.356</div>
              <div class="kpi-subtext">Reflects late-period shift</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="kpi-card">
              <div class="kpi-label">Test SMAPE</div>
              <div class="kpi-value" style="font-size: 1.8rem;">93.0<span class="kpi-unit">%</span></div>
              <div class="kpi-subtext">Symmetric percentage error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📚 Dataset & System Summary")
    st.markdown(
        """
        - **Source & Cadence:** 62,184 crowd readings logged at 10-minute intervals from August 2015 through March 2017 (~19 months) at a single major university fitness center.
        - **Raw Variables:** Datetime, timestamp, day of week, weekend indicator, federal holiday indicator, temperature (°F), semester start flag, during-semester flag, month, clock hour, and turnstile crowd count (`number_people`).
        - **Chronological Discipline:** Strict chronological splitting (70% Train, 15% Validation, 15% Test) without data leakage. The held-out test target was unblinded exactly once after all tuning was finalized.
        - **Core Model:** Tuned `RandomForestRegressor` with 183 trees, max depth of 8, cyclical harmonic transforms (`sin`/`cos` for hour, day, and month), and empirical residual interval calibration.
        """
    )

    st.markdown("---")
    st.markdown("### 🔄 Forecast vs. Nowcast Comparison")
    st.markdown(
        """
        In production machine learning, distinguishing between **planning ahead (Forecast)** and **monitoring in real time (Nowcast)** is paramount:
        """
    )

    comparison_data = {
        "Scenario": [
            "Scenario A: Forecast (This App)",
            "Scenario B: Nowcast (Live Sensor Lags)",
        ],
        "Inputs Available": [
            "Calendar, Clock Time, Weather Forecast only",
            "Clock Time + 10m/30m/60m lags + 1-day/1-week lags + rolling means",
        ],
        "Validation MAE": ["17.28 people", "3.47 people"],
        "Validation R²": ["-0.974", "+0.901"],
        "Primary Use Case": [
            "Future planning (e.g. 'What will the gym be like tomorrow at 5 PM?')",
            "Real-time facility monitoring (e.g. 'How busy is it right now?')",
        ],
    }
    st.table(pd.DataFrame(comparison_data))

    st.info(
        "💡 **Key Takeaway:** Incorporating recent sensor history reduces error by **80.1%** (from 17.28 down to 3.47 MAE). However, for workout scheduling days in advance, lag observations are physically unavailable, making the calendar Forecast essential."
    )

    st.markdown("---")
    st.markdown("### ⚠️ Honest Limitations & Operational Guardrails")
    st.markdown(
        """
        1. **Single Facility Specificity:** The model is trained exclusively on one campus facility. It cannot generalize to other gyms without local historical calibration.
        2. **Unobserved Events:** The model lacks visibility into athletic games, campus concerts, exam schedules, or sudden equipment closures.
        3. **Late-Period Attendance Shift:** As documented in validation diagnostics, evening attendance softened in later periods, leading to systematic evening overprediction.
        4. **Empirical Intervals:** The 80% prediction range is derived from historical out-of-fold residuals; it provides a realistic operational spread rather than a strict theoretical guarantee.
        """
    )
