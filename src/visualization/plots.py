from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_ERA_COLORS: dict[str, str] = {
    "Open Era": "#7986CB",
    "Modern Era": "#FF8A65",
    "Big 3 Era": "#66BB6A",
}

_NBA_ERA_COLORS: dict[str, str] = {
    "pre_advanced": "#90A4AE",
    "pre_3pt":      "#FFA726",
    "modern":       "#5C6BC0",
    "analytics":    "#26A69A",
}

_SURFACE_COLORS: dict[str, str] = {
    "Hard": "#2196F3",
    "Clay": "#FF5722",
    "Grass": "#4CAF50",
}



def plot_goat_rankings(
    df: pd.DataFrame,
    top_n: int = 20,
    score_col: str = "goat_score",
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    _color_map = color_map if color_map is not None else _ERA_COLORS
    top = (
        df.nlargest(top_n, score_col)
        .sort_values(score_col, ascending=True)
    )
    fig = px.bar(
        top,
        x=score_col,
        y="name",
        color="era",
        orientation="h",
        color_discrete_map=_color_map,
        title=f"GOAT Rankings — Top {top_n}",
        labels={score_col: "GOAT Score (0–100)", "name": "", "era": "Era"},
        text=score_col,
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    for trace in fig.data:
        trace.update(textfont_color=_color_map.get(trace.name, "#333333"))
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(range=[0, 105], gridcolor="#eeeeee"),
        legend=dict(title="Era", orientation="h", yanchor="bottom", y=1.02),
        height=max(400, top_n * 28),
    )
    return fig


def plot_surface_map(df: pd.DataFrame, score_col: str = "goat_score") -> go.Figure:
    top10 = df.nlargest(10, score_col)["name"].tolist()
    plot_df = df.copy()
    plot_df["label"] = plot_df["name"].where(plot_df["name"].isin(top10), "")

    fig = px.scatter(
        plot_df,
        x="clay_win_pct",
        y="grass_win_pct",
        size=score_col,
        color="era",
        text="label",
        hover_name="name",
        hover_data={
            score_col: ":.1f",
            "clay_win_pct": ":.3f",
            "grass_win_pct": ":.3f",
            "hard_win_pct": ":.3f",
            "label": False,
        },
        color_discrete_map=_ERA_COLORS,
        size_max=50,
        title="Surface Map — Clay vs Grass Win Rate",
        labels={
            "clay_win_pct": "Clay Win Rate",
            "grass_win_pct": "Grass Win Rate",
            "era": "Era",
        },
    )
    fig.update_traces(textposition="top center", textfont_size=11)
    fig.update_layout(
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eeeeee"),
        yaxis=dict(gridcolor="#eeeeee"),
    )
    return fig


def plot_surface_distributions(df: pd.DataFrame) -> go.Figure:
    melted = df.melt(
        id_vars=["name", "era"],
        value_vars=["hard_win_pct", "clay_win_pct", "grass_win_pct"],
        var_name="surface",
        value_name="win_pct",
    )
    melted["surface"] = melted["surface"].str.replace("_win_pct", "").str.capitalize()

    fig = px.box(
        melted,
        x="surface",
        y="win_pct",
        color="surface",
        hover_name="name",
        hover_data={"era": True, "win_pct": ":.3f", "surface": False},
        points="all",
        color_discrete_map=_SURFACE_COLORS,
        category_orders={"surface": ["Hard", "Clay", "Grass"]},
        title="Surface Win Rate Distributions",
        labels={"win_pct": "Win Rate", "surface": "Surface"},
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        showlegend=False,
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#eeeeee"),
    )
    return fig


def plot_player_radar(metrics: dict[str, float], player_name: str) -> go.Figure:
    categories = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line_color="#1565C0",
            fillcolor="rgba(21, 101, 192, 0.2)",
            name=player_name,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"{player_name} — Performance Profile",
        showlegend=False,
        height=450,
    )
    return fig


def plot_feature_importance(importance: dict[str, float], title: str = "Feature Importance") -> go.Figure:
    df = pd.DataFrame(
        sorted(importance.items(), key=lambda x: abs(x[1])),
        columns=["feature", "value"],
    )
    df["abs_value"] = df["value"].abs()
    df["color"] = df["value"].apply(lambda v: "#43A047" if v >= 0 else "#E53935")

    fig = go.Figure(
        go.Bar(
            x=df["abs_value"],
            y=df["feature"],
            orientation="h",
            marker_color=df["color"],
            text=df["value"].round(4),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Importance / |Coefficient|",
        yaxis_title="",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eeeeee"),
        height=max(350, len(importance) * 30),
    )
    return fig


def plot_predicted_vs_actual(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
) -> go.Figure:
    fig = px.scatter(
        df,
        x=actual_col,
        y=predicted_col,
        color="era",
        hover_name="name",
        hover_data={actual_col: ":.2f", predicted_col: ":.2f"},
        color_discrete_map=_ERA_COLORS,
        title="Model Predictions vs Actual Composite Score",
        labels={
            actual_col: "Actual Score",
            predicted_col: "Predicted Score",
            "era": "Era",
        },
    )
    score_min = df[actual_col].min()
    score_max = df[actual_col].max()
    fig.add_shape(
        type="line",
        x0=score_min, y0=score_min,
        x1=score_max, y1=score_max,
        line=dict(color="gray", dash="dash", width=1),
    )
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eeeeee"),
        yaxis=dict(gridcolor="#eeeeee"),
    )
    return fig
