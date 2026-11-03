"""Component explaining the drivers behind a single prediction."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_prediction_drivers(driver_info: dict[str, Any]) -> None:
    """Render the compact driver breakdown card and global importance ranking."""
    st.markdown("### 🔍 What's Driving This Prediction?")

    col_local, col_global = st.columns([1.1, 0.9])

    with col_local:
        st.markdown(
            """
            <div style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 0.8rem;">
              Local factor sensitivity relative to baseline conditions:
            </div>
            """,
            unsafe_allow_html=True,
        )

        time_delta = driver_info.get("time_delta", 0.0)
        sem_delta = driver_info.get("sem_delta", 0.0)
        dow_delta = driver_info.get("dow_delta", 0.0)
        temp_delta = driver_info.get("temp_delta", 0.0)

        def _format_impact(val: float) -> str:
            sign = "+" if val > 0 else ""
            cls = (
                "driver-impact-pos"
                if val > 0.5
                else ("driver-impact-neg" if val < -0.5 else "driver-impact-neutral")
            )
            return f'<span class="driver-impact {cls}">{sign}{val:.1f} people</span>'

        st.markdown(
            f"""
            <div class="drivers-container">
              <div class="driver-row">
                <span class="driver-name">🕒 <b>Time of Day Effect</b> (vs daily mean)</span>
                {_format_impact(time_delta)}
              </div>
              <div class="driver-row">
                <span class="driver-name">🎓 <b>Semester Status</b> (classes in session)</span>
                {_format_impact(sem_delta)}
              </div>
              <div class="driver-row">
                <span class="driver-name">📅 <b>Day of Week</b> (routine attendance)</span>
                {_format_impact(dow_delta)}
              </div>
              <div class="driver-row">
                <span class="driver-name">🌡️ <b>Temperature Effect</b> (vs 65°F standard)</span>
                {_format_impact(temp_delta)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_global:
        top_shap = driver_info.get("top_shap", [])
        if top_shap:
            st.markdown(
                """
                <div style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                  Model reliance ranking (Mean Absolute SHAP):
                </div>
                """,
                unsafe_allow_html=True,
            )

            features = [item["feature"] for item in reversed(top_shap)]
            importances = [item["importance"] for item in reversed(top_shap)]

            fig = go.Figure(
                go.Bar(
                    x=importances,
                    y=features,
                    orientation="h",
                    marker=dict(color="#0D9488"),
                    text=[f"{v:.1f}" for v in importances],
                    textposition="auto",
                    hoverinfo="x+y",
                )
            )
            fig.update_layout(
                height=150,
                margin=dict(l=10, r=20, t=5, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=11, color="#334155"),
                xaxis=dict(
                    title="Impact (SHAP)",
                    showgrid=True,
                    gridcolor="rgba(226, 232, 240, 0.7)",
                ),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )
        else:
            st.info("Global SHAP summary metrics loaded from validation artifacts.")
