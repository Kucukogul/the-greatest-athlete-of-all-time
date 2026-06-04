from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when Streamlit runs from any cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.features.engineering import TennisSurfaceFeatures
from src.models.tennis_model import (
    TARGET_COL,
    TennisForestModel,
    TennisRidgeModel,
    cross_validate_model,
    prepare_features,
)
from src.visualization.plots import (
    plot_feature_importance,
    plot_goat_rankings,
    plot_player_radar,
    plot_predicted_vs_actual,
    plot_surface_distributions,
    plot_surface_map,
)

_DATA_PATH = Path(__file__).parent.parent / "data/processed/tennis_all_v2.csv"

_RADAR_METRICS: list[str] = [
    "grand_slam_normalized",
    "weeks_no1_normalized",
    "masters_titles_normalized",
    "career_win_rate_normalized",
    "finals_win_rate_normalized",
    "h2h_top10_normalized",
    "surface_versatility_normalized",
    "longevity_normalized",
]

_RADAR_LABELS: dict[str, str] = {
    "grand_slam_normalized": "Grand Slams",
    "weeks_no1_normalized": "Weeks No.1",
    "masters_titles_normalized": "Masters",
    "career_win_rate_normalized": "Win Rate",
    "finals_win_rate_normalized": "Finals Win%",
    "h2h_top10_normalized": "H2H Top-10",
    "surface_versatility_normalized": "Versatility",
    "longevity_normalized": "Longevity",
}

st.set_page_config(
    page_title="The Greatest Athlete of All Time",
    page_icon="🎾",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH)
    return TennisSurfaceFeatures.build_all(df)


@st.cache_resource
def fit_models(cache_key: int):  # cache_key ties to data version
    df = load_data()
    X = prepare_features(df)
    y = df[TARGET_COL]

    ridge = TennisRidgeModel().fit(X, y)
    forest = TennisForestModel(n_estimators=300, random_state=42).fit(X, y)

    ridge_cv = cross_validate_model(TennisRidgeModel, X, y, cv=5)
    forest_cv = cross_validate_model(
        lambda: TennisForestModel(n_estimators=300, random_state=42), X, y, cv=5
    )

    df["ridge_pred"] = ridge.predict(X).values
    df["forest_pred"] = forest.predict(X).values

    return ridge, forest, ridge_cv, forest_cv, df


def main() -> None:
    st.title("The Greatest Athlete of All Time")
    st.caption("Surface-based GOAT analysis · ATP Tennis · 617 Players · 1968–2026")

    df = load_data()
    ridge, forest, ridge_cv, forest_cv, df_with_preds = fit_models(cache_key=2)

    # ── Sidebar ──────────────────────────────────────────────
    st.sidebar.header("Filters")
    all_eras = sorted(df["era"].unique().tolist())
    selected_eras = st.sidebar.multiselect("Era", all_eras, default=all_eras)
    top_n = st.sidebar.slider("Top N players to show", 10, 50, 20, 5)
    gs_only = st.sidebar.checkbox("Grand Slam winners only", value=False)

    filtered = df[df["era"].isin(selected_eras)].copy()
    if gs_only:
        filtered = filtered[filtered["grand_slams"] >= 1]

    # ── Tabs ─────────────────────────────────────────────────
    tab_rank, tab_surface, tab_player, tab_model = st.tabs([
        "🏆 Rankings",
        "🌍 Surface Map",
        "🔍 Player Deep-Dive",
        "🤖 Model Insights",
    ])

    # ── Tab 1: Rankings ──────────────────────────────────────
    with tab_rank:
        col_left, col_right = st.columns([3, 1])

        with col_left:
            st.plotly_chart(
                plot_goat_rankings(filtered, top_n=top_n),
                use_container_width=True,
            )

        with col_right:
            st.subheader("Top 3")
            top3 = filtered.nlargest(3, "composite_score")
            for rank, (_, row) in enumerate(top3.iterrows(), 1):
                medal = ["🥇", "🥈", "🥉"][rank - 1]
                st.metric(
                    label=f"{medal} #{rank}",
                    value=row["name"],
                    delta=f"{row['composite_score']:.1f} pts",
                )

            st.divider()
            st.subheader("Era breakdown")
            era_counts = filtered["era"].value_counts()
            for era, count in era_counts.items():
                st.write(f"**{era}:** {count} players")

        st.subheader("Full Rankings Table")
        table_cols = ["name", "era", "composite_score", "grand_slams",
                      "hard_win_pct", "clay_win_pct", "grass_win_pct",
                      "surface_versatility_normalized"]
        display = (
            filtered[table_cols]
            .sort_values("composite_score", ascending=False)
            .reset_index(drop=True)
        )
        display.index += 1
        st.dataframe(
            display.style.format({
                "composite_score": "{:.2f}",
                "grand_slams": "{:.1f}",
                "hard_win_pct": "{:.1%}",
                "clay_win_pct": "{:.1%}",
                "grass_win_pct": "{:.1%}",
                "surface_versatility_normalized": "{:.1f}",
            }),
            use_container_width=True,
            height=400,
        )

    # ── Tab 2: Surface Map ───────────────────────────────────
    with tab_surface:
        col_a, col_b = st.columns(2)

        with col_a:
            st.plotly_chart(
                plot_surface_map(filtered),
                use_container_width=True,
            )

        with col_b:
            st.plotly_chart(
                plot_surface_distributions(filtered),
                use_container_width=True,
            )

        st.subheader("Era × Surface Average Win Rates")
        era_surface = (
            filtered
            .groupby("era")[["hard_win_pct", "clay_win_pct", "grass_win_pct"]]
            .mean()
            .round(3)
            .rename(columns={
                "hard_win_pct": "Hard",
                "clay_win_pct": "Clay",
                "grass_win_pct": "Grass",
            })
        )
        era_surface["Spread (pp)"] = (
            (era_surface.max(axis=1) - era_surface.min(axis=1)) * 100
        ).round(1)
        st.dataframe(
            era_surface.style.format({
                "Hard": "{:.1%}", "Clay": "{:.1%}", "Grass": "{:.1%}",
            }),
            use_container_width=True,
        )

    # ── Tab 3: Player Deep-Dive ──────────────────────────────
    with tab_player:
        player_names = filtered.sort_values("composite_score", ascending=False)["name"].tolist()
        selected = st.selectbox("Select player", player_names)

        player_row = filtered[filtered["name"] == selected].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Composite Score", f"{player_row['composite_score']:.2f}")
        col2.metric("Era", player_row["era"])
        col3.metric("Grand Slams", f"{player_row['grand_slams']:.1f}")
        col4.metric("Surface Versatility", f"{player_row['surface_versatility_normalized']:.1f}")

        st.divider()
        col_radar, col_surface = st.columns([1, 1])

        with col_radar:
            radar_metrics = {
                _RADAR_LABELS[col]: float(player_row[col])
                for col in _RADAR_METRICS
                if col in player_row.index
            }
            st.plotly_chart(
                plot_player_radar(radar_metrics, selected),
                use_container_width=True,
            )

        with col_surface:
            st.subheader("Surface Performance")
            surface_data = {
                "Hard": player_row["hard_win_pct"],
                "Clay": player_row["clay_win_pct"],
                "Grass": player_row["grass_win_pct"],
            }
            for surface, rate in surface_data.items():
                st.metric(f"{surface} Win Rate", f"{rate:.1%}")

            st.divider()
            st.subheader("Surface Consistency (Year-over-Year Std)")
            st.metric("Clay Std", f"{player_row['clay_win_rate_std']:.3f}")
            st.metric("Grass Std", f"{player_row['grass_win_rate_std']:.3f}")

            st.divider()
            st.subheader("Engineered Features")
            st.metric("Surface Flexibility (clay × grass)",
                       f"{player_row['surface_flexibility']:.3f}")
            st.metric("Surface Gap |clay − grass|",
                       f"{player_row['surface_gap']:.3f}")
            st.metric("Surface Floor (worst surface)",
                       f"{player_row['surface_floor']:.3f}")

    # ── Tab 4: Model Insights ────────────────────────────────
    with tab_model:
        st.subheader("Cross-Validation Performance (5-Fold)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ridge R²", f"{ridge_cv.r2_cv_mean:.3f}", f"± {ridge_cv.r2_cv_std:.3f}")
        m2.metric("Ridge RMSE", f"{ridge_cv.rmse_cv_mean:.3f}")
        m3.metric("Forest R²", f"{forest_cv.r2_cv_mean:.3f}", f"± {forest_cv.r2_cv_std:.3f}")
        m4.metric("Forest RMSE", f"{forest_cv.rmse_cv_mean:.3f}")

        st.caption(
            "R² = variance explained by surface+era features alone (no grand_slams, no weeks_at_no1). "
            "Forest > Ridge confirms non-linear era × surface thresholds exist in the data."
        )

        st.divider()
        col_imp, col_coef = st.columns(2)

        with col_imp:
            st.plotly_chart(
                plot_feature_importance(
                    forest.feature_importance(),
                    title="Random Forest — Feature Importance",
                ),
                use_container_width=True,
            )

        with col_coef:
            st.plotly_chart(
                plot_feature_importance(
                    ridge.coefficients(),
                    title="Ridge — Scaled Coefficients (green=+, red=−)",
                ),
                use_container_width=True,
            )

        st.divider()
        model_choice = st.radio(
            "Predicted vs Actual — select model",
            ["Random Forest", "Ridge"],
            horizontal=True,
        )
        pred_col = "forest_pred" if model_choice == "Random Forest" else "ridge_pred"
        plot_df = df_with_preds[df_with_preds["era"].isin(selected_eras)].copy()
        st.plotly_chart(
            plot_predicted_vs_actual(plot_df, TARGET_COL, pred_col),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
