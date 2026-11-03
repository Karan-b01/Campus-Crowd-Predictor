"""Hero KPI card component for the prediction dashboard."""

from __future__ import annotations

import streamlit as st


def render_kpi_hero(
    count: float,
    lower: float,
    upper: float,
    level: str,
    badge_class: str,
    color: str,
    description: str,
    pct_of_peak: float,
) -> None:
    """Render the hero row containing the 3 primary metric cards."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Predicted Crowd</div>
              <div class="kpi-value">{int(round(count))}<span class="kpi-unit">people</span></div>
              <div class="kpi-subtext">
                <span>≈ {pct_of_peak}% of typical peak capacity</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Crowd Level</div>
              <div style="margin: 0.25rem 0;">
                <span class="badge {badge_class}">
                  <span class="badge-dot"></span>
                  {level.upper()} CROWD
                </span>
              </div>
              <div class="kpi-subtext">
                <span>{description}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Empirical Range (80% Band)</div>
              <div class="kpi-value" style="font-size: 1.85rem;">
                {int(round(lower))} – {int(round(upper))}<span class="kpi-unit">people</span>
              </div>
              <div class="kpi-subtext">
                <span>Calibrated out-of-fold residual bounds</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
