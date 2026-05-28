from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_athlete_scores(scores: pd.Series, title: str = "Athlete Scores") -> go.Figure:
    df = scores.reset_index()
    df.columns = ["athlete", "score"]
    df = df.sort_values("score", ascending=True)
    return px.bar(
        df,
        x="score",
        y="athlete",
        orientation="h",
        title=title,
        color="score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
    )


def plot_radar_chart(metrics: dict[str, float], athlete_name: str) -> go.Figure:
    categories = list(metrics.keys())
    values = list(metrics.values())
    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=athlete_name,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"{athlete_name} — Metric Radar",
    )
    return fig


def plot_score_breakdown(breakdown: pd.DataFrame, top_n: int = 10) -> go.Figure:
    top = breakdown.nlargest(top_n, "total").drop(columns="total")
    fig = px.bar(
        top.reset_index().melt(id_vars="index"),
        x="value",
        y="index",
        color="variable",
        orientation="h",
        title=f"Top {top_n} Athletes — Score Breakdown",
        barmode="stack",
    )
    return fig
