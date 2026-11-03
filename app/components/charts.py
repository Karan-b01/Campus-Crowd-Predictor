"""Plotly visualization components for the daily timeline and weekly planner."""

from __future__ import annotations

from datetime import datetime as DateTime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from app.utils import DEFAULT_LOW_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD
except (ImportError, ModuleNotFoundError):
    from utils import DEFAULT_LOW_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD


def render_daily_forecast_chart(
    day_df: pd.DataFrame,
    selected_dt: DateTime | pd.Timestamp,
    recommendations: dict[str, Any],
) -> None:
    """Render the 24-hour crowd forecast line chart with shaded uncertainty band."""
    fig = go.Figure()

    # 1. Shaded uncertainty band (Lower to Upper)
    fig.add_trace(
        go.Scatter(
            x=day_df["datetime"],
            y=day_df["upper"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name="Upper Bound",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=day_df["datetime"],
            y=day_df["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(13, 148, 136, 0.12)",
            hoverinfo="skip",
            name="80% Prediction Band",
            showlegend=True,
        )
    )

    # 2. Main predicted line
    fig.add_trace(
        go.Scatter(
            x=day_df["datetime"],
            y=day_df["count"],
            mode="lines",
            line=dict(color="#0D9488", width=3.2),
            name="Predicted Crowd",
            hovertemplate=(
                "<b>Time:</b> %{customdata[0]}<br>"
                "<b>Predicted Crowd:</b> %{y:.1f} people<br>"
                "<b>Expected Range:</b> %{customdata[1]:.0f} – %{customdata[2]:.0f}<br>"
                "<b>Level:</b> %{customdata[3]}<extra></extra>"
            ),
            customdata=np.stack(
                (
                    day_df["time_ampm"],
                    day_df["lower"],
                    day_df["upper"],
                    day_df["level"],
                ),
                axis=-1,
            ),
        )
    )

    # 3. Selected time point marker
    # Find row closest to selected_dt
    time_diffs = (day_df["datetime"] - pd.Timestamp(selected_dt)).abs()
    closest_idx = time_diffs.idxmin()
    selected_row = day_df.loc[closest_idx]

    fig.add_trace(
        go.Scatter(
            x=[selected_row["datetime"]],
            y=[selected_row["count"]],
            mode="markers+text",
            marker=dict(
                color=selected_row["color"],
                size=14,
                line=dict(color="#FFFFFF", width=3),
                symbol="circle",
            ),
            text=[
                f"  <b>{selected_row['count']:.0f}</b> ({selected_row['time_ampm']})"
            ],
            textposition="top right",
            textfont=dict(size=12, color="#0F172A"),
            name="Selected Moment",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # 4. Highlight quietest recommended window
    best_start_hour = recommendations.get("best_start_hour", 6.0)
    best_end_hour = recommendations.get("best_end_hour", 7.5)
    best_window_text = recommendations.get("best_window", "Quiet Window")

    day_start = day_df["datetime"].iloc[0].replace(hour=0, minute=0, second=0)
    rect_x0 = day_start + pd.Timedelta(hours=best_start_hour)
    rect_x1 = day_start + pd.Timedelta(hours=best_end_hour)

    fig.add_vrect(
        x0=rect_x0,
        x1=rect_x1,
        fillcolor="rgba(16, 185, 129, 0.15)",
        layer="below",
        line_width=1,
        line_dash="dot",
        line_color="#10B981",
        annotation_text=f"⭐ Quietest Window ({best_window_text})",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#065F46"),
    )

    # 5. Threshold lines (Low & Medium)
    fig.add_hline(
        y=DEFAULT_LOW_THRESHOLD,
        line_dash="dash",
        line_width=1,
        line_color="#10B981",
        annotation_text="Low limit (21)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#10B981"),
    )
    fig.add_hline(
        y=DEFAULT_MEDIUM_THRESHOLD,
        line_dash="dash",
        line_width=1,
        line_color="#F59E0B",
        annotation_text="Medium limit (43)",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#F59E0B"),
    )

    # Styling layout
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=30, t=35, b=40),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        font=dict(family="sans-serif", color="#334155"),
        xaxis=dict(
            title="Time of Day",
            tickformat="%I %p",
            dtick=3 * 3600 * 1000,  # every 3 hours
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.7)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Predicted Crowd (People)",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.7)",
            zeroline=True,
            zerolinecolor="#CBD5E1",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_weekly_heatmap_chart(pivot_df: pd.DataFrame) -> None:
    """Render the weekly hour-by-day heatmap for workout scheduling."""
    # Custom color palette: Calm teal to coral
    colors = [
        [0.0, "#ECFDF5"],  # very low (soft mint)
        [0.25, "#A7F3D0"],  # low
        [0.50, "#FDE68A"],  # medium (amber)
        [0.75, "#FB923C"],  # orange
        [1.0, "#EF4444"],  # peak (coral/red)
    ]

    # Format hours for y-axis
    y_labels = [f"{h:02d}:00" for h in pivot_df.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=y_labels,
            colorscale=colors,
            colorbar=dict(
                title=dict(text="People", side="top"),
                thickness=14,
                len=0.85,
                tickfont=dict(size=11, color="#64748B"),
            ),
            hovertemplate="<b>%{x}</b> at <b>%{y}</b><br>Predicted Crowd: <b>%{z:.1f} people</b><extra></extra>",
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=50, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#334155"),
        xaxis=dict(
            title="Day of Week",
            side="top",
            tickfont=dict(size=12, weight="bold"),
            showgrid=False,
        ),
        yaxis=dict(
            title="Hour of Day",
            autorange="reversed",  # 00:00 at top to 23:00 at bottom
            showgrid=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
