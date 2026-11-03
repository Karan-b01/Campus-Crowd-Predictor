"""Interactive Plotly charts for the Insights & EDA dashboard page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_hourly_weekday_weekend_chart(hourly_df: pd.DataFrame) -> None:
    """Render weekday vs weekend hourly curves with shaded intervals."""
    fig = go.Figure()

    # Weekday (is_weekend == 0)
    weekday_data = hourly_df[hourly_df["is_weekend"] == 0].sort_values("hour")
    # Weekend (is_weekend == 1)
    weekend_data = hourly_df[hourly_df["is_weekend"] == 1].sort_values("hour")

    fig.add_trace(
        go.Scatter(
            x=weekday_data["hour"],
            y=weekday_data["mean"],
            mode="lines+markers",
            name="Weekday (Mon–Fri)",
            line=dict(color="#0D9488", width=3),
            marker=dict(size=6),
            hovertemplate="<b>Weekday</b> %{x}:00<br>Average: <b>%{y:.1f} people</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=weekend_data["hour"],
            y=weekend_data["mean"],
            mode="lines+markers",
            name="Weekend (Sat–Sun)",
            line=dict(color="#F59E0B", width=3, dash="dot"),
            marker=dict(size=6),
            hovertemplate="<b>Weekend</b> %{x}:00<br>Average: <b>%{y:.1f} people</b><extra></extra>",
        )
    )

    fig.add_annotation(
        x=18,
        y=weekday_data[weekday_data["hour"] == 18]["mean"].values[0],
        text="Weekday Peak: 6:00 PM (~52 people)",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#0D9488",
        font=dict(size=11, color="#0F172A"),
        bgcolor="#F0FDFA",
        bordercolor="#0D9488",
        borderwidth=1,
    )

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=30, t=25, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        font=dict(family="sans-serif", color="#334155"),
        xaxis=dict(
            title="Hour of Day (24-Hour)",
            dtick=2,
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.7)",
        ),
        yaxis=dict(
            title="Average Crowd (People)",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.7)",
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


def render_day_of_week_chart(dow_df: pd.DataFrame) -> None:
    """Render average crowd count across days of the week."""
    colors = ["#0D9488"] * 5 + ["#64748B", "#64748B"]

    fig = go.Figure(
        go.Bar(
            x=dow_df["day_name"],
            y=dow_df["mean"],
            marker_color=colors,
            text=[f"{v:.1f}" for v in dow_df["mean"]],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Average Crowd: <b>%{y:.1f} people</b><extra></extra>",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=20, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#334155"),
        xaxis=dict(title="Day of Week", showgrid=False),
        yaxis=dict(
            title="Mean People", showgrid=True, gridcolor="rgba(226, 232, 240, 0.7)"
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_semester_comparison_chart(sem_df: pd.DataFrame) -> None:
    """Render bar chart comparing during-semester vs outside-semester crowd."""
    labels = ["Academic Break / Outside", "During Semester"]
    values = sem_df["mean"].tolist()

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#94A3B8", "#0D9488"],
            text=[f"{v:.1f} people" for v in values],
            textposition="auto",
            width=0.45,
            hovertemplate="<b>%{x}</b>: <b>%{y:.1f} people</b><extra></extra>",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=20, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#334155"),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title="Mean People", showgrid=True, gridcolor="rgba(226, 232, 240, 0.7)"
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_temperature_scatter_chart(sample_df: pd.DataFrame) -> None:
    """Render scatter plot of temperature versus occupancy."""
    fig = px.scatter(
        sample_df,
        x="temperature",
        y="number_people",
        color="hour",
        color_continuous_scale="Tealgrn",
        labels={
            "temperature": "Recorded Temperature (°F)",
            "number_people": "Observed Crowd (People)",
            "hour": "Hour",
        },
        opacity=0.6,
    )

    fig.update_layout(
        height=340,
        margin=dict(l=30, r=20, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        font=dict(family="sans-serif", color="#334155"),
        xaxis=dict(showgrid=True, gridcolor="rgba(226, 232, 240, 0.7)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(226, 232, 240, 0.7)"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
