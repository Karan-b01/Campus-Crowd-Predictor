"""Interactive Plotly gauge comparing forecast crowd to typical peak."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

try:
    from app.utils import (
        DEFAULT_LOW_THRESHOLD,
        DEFAULT_MEDIUM_THRESHOLD,
        GYM_TYPICAL_PEAK,
    )
except (ImportError, ModuleNotFoundError):
    from utils import (
        DEFAULT_LOW_THRESHOLD,
        DEFAULT_MEDIUM_THRESHOLD,
        GYM_TYPICAL_PEAK,
    )


def render_peak_gauge(
    count: float,
    color: str,
    level: str,
    low_max: float = DEFAULT_LOW_THRESHOLD,
    med_max: float = DEFAULT_MEDIUM_THRESHOLD,
) -> None:
    """Render a clean Plotly gauge comparing predicted occupancy to typical peak."""
    max_gauge = 80.0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=count,
            number={"suffix": " people", "font": {"size": 26, "color": "#0F172A"}},
            gauge={
                "axis": {
                    "range": [0, max_gauge],
                    "tickwidth": 1,
                    "tickcolor": "#64748B",
                    "tickvals": [0, 20, 40, 55, 75],
                    "ticktext": ["0", "20", "40", "Peak 55", "75+"],
                    "tickfont": {"size": 11, "color": "#64748B"},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(241, 245, 249, 0.8)",
                "borderwidth": 1,
                "bordercolor": "rgba(226, 232, 240, 0.8)",
                "steps": [
                    {"range": [0, low_max], "color": "rgba(16, 185, 129, 0.20)"},
                    {"range": [low_max, med_max], "color": "rgba(245, 158, 11, 0.20)"},
                    {"range": [med_max, max_gauge], "color": "rgba(239, 68, 68, 0.20)"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 3},
                    "thickness": 0.8,
                    "value": GYM_TYPICAL_PEAK,
                },
            },
        )
    )

    fig.update_layout(
        height=220,
        margin=dict(l=25, r=25, t=30, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
