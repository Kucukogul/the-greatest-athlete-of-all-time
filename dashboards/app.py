from __future__ import annotations

import streamlit as st
import pandas as pd

from src.scoring.scorer import AthleteScorer, ScoringConfig
from src.visualization.plots import plot_athlete_scores, plot_radar_chart, plot_score_breakdown

st.set_page_config(
    page_title="The Greatest Athlete of All Time",
    layout="wide",
)

st.title("The Greatest Athlete of All Time")
st.caption("A data-driven ranking system for cross-sport athletic greatness")

st.sidebar.header("Scoring Weights")
ppg_w = st.sidebar.slider("Points Per Game", 0.0, 1.0, 0.30, 0.05)
rpg_w = st.sidebar.slider("Rebounds Per Game", 0.0, 1.0, 0.20, 0.05)
apg_w = st.sidebar.slider("Assists Per Game", 0.0, 1.0, 0.20, 0.05)
per_w = st.sidebar.slider("PER", 0.0, 1.0, 0.30, 0.05)

weights_sum = ppg_w + rpg_w + apg_w + per_w
weights_valid = abs(weights_sum - 1.0) <= 0.01

if not weights_valid:
    st.sidebar.warning(f"Weights sum to {weights_sum:.2f} — must equal 1.0")

uploaded = st.file_uploader("Upload processed athlete CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
    with col2:
        st.subheader("Dataset Info")
        st.metric("Athletes", len(df))
        st.metric("Columns", len(df.columns))
        if "sport" in df.columns:
            st.metric("Sports", df["sport"].nunique())

    if weights_valid:
        config = ScoringConfig(
            weights={
                "ppg_normalized": ppg_w,
                "rpg_normalized": rpg_w,
                "apg_normalized": apg_w,
                "per_normalized": per_w,
            }
        )
        scorer = AthleteScorer(config)
        try:
            scores = scorer.score(df)
            breakdown = scorer.score_breakdown(df)

            st.subheader("Athlete Rankings")
            st.plotly_chart(
                plot_athlete_scores(scores, "Composite Athlete Score (0–100)"),
                use_container_width=True,
            )

            st.subheader("Score Breakdown — Top 10")
            st.plotly_chart(
                plot_score_breakdown(breakdown, top_n=10),
                use_container_width=True,
            )

            top_athlete_idx = scores.idxmax()
            athlete_metrics = {
                "PPG": df.loc[top_athlete_idx, "ppg_normalized"],
                "RPG": df.loc[top_athlete_idx, "rpg_normalized"],
                "APG": df.loc[top_athlete_idx, "apg_normalized"],
                "PER": df.loc[top_athlete_idx, "per_normalized"],
            }
            athlete_name = df.loc[top_athlete_idx, "name"] if "name" in df.columns else str(top_athlete_idx)
            st.subheader(f"Top Athlete Radar — {athlete_name}")
            st.plotly_chart(plot_radar_chart(athlete_metrics, athlete_name), use_container_width=True)

        except ValueError as e:
            st.error(f"Scoring error: {e}")
else:
    st.info("Upload a processed CSV with normalized columns (ppg_normalized, rpg_normalized, apg_normalized, per_normalized) to see rankings.")
