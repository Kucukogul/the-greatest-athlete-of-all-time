from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.features.engineering import TennisSurfaceFeatures
from src.models.nba_model import NBAForestModel, load_nba_model_config
from src.models.tennis_model import (
    TARGET_COL,
    TennisForestModel,
    TennisRidgeModel,
    cross_validate_model,
    prepare_features,
)
from src.scoring.tennis_scorer import TennisScorer
from src.visualization.plots import (
    _ERA_COLORS,
    _NBA_ERA_COLORS,
    _SWIMMING_ERA_COLORS,
    plot_feature_importance,
    plot_goat_rankings,
    plot_player_radar,
    plot_predicted_vs_actual,
    plot_surface_distributions,
    plot_surface_map,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
_TENNIS_DATA_PATH   = _ROOT / "data/processed/tennis_all_v2.csv"
_NBA_DATA_PATH      = _ROOT / "data/processed/nba_goat_v1.csv"
_SWIMMING_DATA_PATH = _ROOT / "data/processed/swimming_goat_v1.csv"
_TENNIS_SCORE_CFG   = _ROOT / "configs/scoring_tennis.yaml"
_NBA_CONFIG_DIR     = _ROOT / "configs"

# ── Tennis constants ──────────────────────────────────────────────────────────

_TENNIS_RADAR_METRICS: list[str] = [
    "grand_slam_normalized",
    "weeks_no1_normalized",
    "masters_titles_normalized",
    "career_win_rate_normalized",
    "finals_win_rate_normalized",
    "h2h_top10_normalized",
    "surface_versatility_normalized",
    "longevity_normalized",
]

_TENNIS_RADAR_LABELS: dict[str, str] = {
    "grand_slam_normalized":          "Grand Slams",
    "weeks_no1_normalized":           "Weeks No.1",
    "masters_titles_normalized":      "Masters",
    "career_win_rate_normalized":     "Win Rate",
    "finals_win_rate_normalized":     "Finals Win%",
    "h2h_top10_normalized":           "H2H Top-10",
    "surface_versatility_normalized": "Versatility",
    "longevity_normalized":           "Longevity",
}

# ── NBA constants ─────────────────────────────────────────────────────────────

_NBA_RADAR_METRICS: list[str] = [
    "peak_score",
    "career_score",
    "achievement_score",
    "efficiency_score",
    "longevity_score",
]

_NBA_RADAR_LABELS: dict[str, str] = {
    "peak_score":         "Peak",
    "career_score":       "Career",
    "achievement_score":  "Achievement",
    "efficiency_score":   "Efficiency",
    "longevity_score":    "Longevity",
}

# ── Swimming constants ────────────────────────────────────────────────────────

_SWIMMING_RADAR_METRICS: list[str] = [
    "achievement_score",
    "peak_score",
    "versatility_score",
    "longevity_score",
]

_SWIMMING_RADAR_LABELS: dict[str, str] = {
    "achievement_score": "Achievement",
    "peak_score":        "Peak",
    "versatility_score": "Versatility",
    "longevity_score":   "Longevity",
}

_SWIMMING_TABLE_COLS: list[str] = [
    "name", "era", "goat_score",
    "olympic_gold_individual", "wc_gold_individual",
    "world_records", "events_dominated", "career_years",
]

_SWIMMING_TABLE_HEADERS: dict[str, str] = {
    "name": "Athlete",
    "era": "Era",
    "goat_score": "GOAT Score",
    "olympic_gold_individual": "OG Gold (Ind.)",
    "wc_gold_individual": "WC Gold (Ind.)",
    "world_records": "World Records",
    "events_dominated": "Events Dom.",
    "career_years": "Career Years",
}

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_tennis_data() -> pd.DataFrame:
    df = pd.read_csv(_TENNIS_DATA_PATH)
    df = TennisSurfaceFeatures.build_all(df)
    scorer = TennisScorer(_TENNIS_SCORE_CFG)
    df["goat_score"] = scorer.score(df)
    layer_df = scorer.layer_scores(df)
    return pd.concat([df, layer_df], axis=1)


@st.cache_resource
def fit_tennis_models(cache_key: int):
    df = load_tennis_data()
    X = prepare_features(df)
    y = df[TARGET_COL]
    ridge  = TennisRidgeModel().fit(X, y)
    forest = TennisForestModel(n_estimators=300, random_state=42).fit(X, y)
    ridge_cv  = cross_validate_model(TennisRidgeModel, X, y, cv=5)
    forest_cv = cross_validate_model(
        lambda: TennisForestModel(n_estimators=300, random_state=42), X, y, cv=5
    )
    df["ridge_pred"]  = ridge.predict(X).values
    df["forest_pred"] = forest.predict(X).values
    return ridge, forest, ridge_cv, forest_cv, df


@st.cache_data
def load_nba_data() -> pd.DataFrame:
    return pd.read_csv(_NBA_DATA_PATH)


@st.cache_resource
def fit_nba_forest():
    df = load_nba_data()
    cfg = load_nba_model_config(_NBA_CONFIG_DIR / "model_nba.yaml")
    model = NBAForestModel(cfg)
    model.fit(df, df["goat_score"])
    return model


@st.cache_data
def load_swimming_data() -> pd.DataFrame:
    return pd.read_csv(_SWIMMING_DATA_PATH)


# ── App entry ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="The Greatest Athlete of All Time",
    page_icon="🏆",
    layout="wide",
)


def main() -> None:
    st.sidebar.title("The GOAT Project")
    sport = st.sidebar.selectbox("Sport", ["🎾 Tennis", "🏀 NBA", "🏊 Swimming"], index=0)

    if sport == "🎾 Tennis":
        _tennis_page()
    elif sport == "🏀 NBA":
        _nba_page()
    else:
        _swimming_page()


# ── Tennis page ───────────────────────────────────────────────────────────────

def _tennis_page() -> None:
    st.title("Greatest of All Time — Tennis")
    st.caption("Rule-based GOAT scoring · ATP · 617 Players · 1968–2026")

    df = load_tennis_data()
    ridge, forest, ridge_cv, forest_cv, df_preds = fit_tennis_models(cache_key=2)

    st.sidebar.header("Filters")
    all_eras = sorted(df["era"].unique().tolist())
    selected_eras = st.sidebar.multiselect("Era", all_eras, default=all_eras)
    top_n    = st.sidebar.slider("Top N players", 10, 50, 20, 5)
    gs_only  = st.sidebar.checkbox("Grand Slam winners only", value=False)

    filtered = df[df["era"].isin(selected_eras)].copy()
    if gs_only:
        filtered = filtered[filtered["grand_slams"] >= 1]

    tab_rank, tab_surface, tab_player, tab_model = st.tabs([
        "🏆 Rankings", "🌍 Surface Map", "🔍 Player Deep-Dive", "🤖 Model Insights",
    ])

    with tab_rank:
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.plotly_chart(
                plot_goat_rankings(filtered, top_n=top_n, color_map=_ERA_COLORS),
                use_container_width=True,
            )
        with col_right:
            st.subheader("Top 3")
            top3 = filtered.nlargest(3, "goat_score")
            for rank, (_, row) in enumerate(top3.iterrows(), 1):
                medal = ["🥇", "🥈", "🥉"][rank - 1]
                st.metric(label=f"{medal} #{rank}", value=row["name"],
                          delta=f"{row['goat_score']:.1f} pts")
            st.divider()
            st.subheader("Era breakdown")
            for era, count in filtered["era"].value_counts().items():
                st.write(f"**{era}:** {count} players")

        st.subheader("Full Rankings Table")
        table_cols = ["name", "era", "goat_score", "grand_slams",
                      "hard_win_pct", "clay_win_pct", "grass_win_pct",
                      "surface_versatility_normalized"]
        display = (
            filtered[table_cols]
            .sort_values("goat_score", ascending=False)
            .reset_index(drop=True)
        )
        display.index += 1
        st.dataframe(
            display.style.format({
                "goat_score": "{:.2f}",
                "grand_slams": "{:.1f}",
                "hard_win_pct": "{:.1%}",
                "clay_win_pct": "{:.1%}",
                "grass_win_pct": "{:.1%}",
                "surface_versatility_normalized": "{:.1f}",
            }),
            use_container_width=True,
            height=400,
        )

    with tab_surface:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_surface_map(filtered, score_col="goat_score"),
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
            .rename(columns={"hard_win_pct": "Hard", "clay_win_pct": "Clay",
                              "grass_win_pct": "Grass"})
        )
        era_surface["Spread (pp)"] = (
            (era_surface.max(axis=1) - era_surface.min(axis=1)) * 100
        ).round(1)
        st.dataframe(
            era_surface.style.format({"Hard": "{:.1%}", "Clay": "{:.1%}", "Grass": "{:.1%}"}),
            use_container_width=True,
        )

    with tab_player:
        player_names = filtered.sort_values("goat_score", ascending=False)["name"].tolist()
        selected = st.selectbox("Select player", player_names)
        row = filtered[filtered["name"] == selected].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("GOAT Score",          f"{row['goat_score']:.2f}")
        col2.metric("Era",                 row["era"])
        col3.metric("Grand Slams",         f"{row['grand_slams']:.1f}")
        col4.metric("Surface Versatility", f"{row['surface_versatility_normalized']:.1f}")

        st.divider()
        col_radar, col_surface = st.columns(2)
        with col_radar:
            radar_metrics = {
                _TENNIS_RADAR_LABELS[col]: float(row[col])
                for col in _TENNIS_RADAR_METRICS
                if col in row.index
            }
            st.plotly_chart(plot_player_radar(radar_metrics, selected), use_container_width=True)
        with col_surface:
            st.subheader("Surface Performance")
            for surface, col_name in [("Hard", "hard_win_pct"), ("Clay", "clay_win_pct"),
                                       ("Grass", "grass_win_pct")]:
                st.metric(f"{surface} Win Rate", f"{row[col_name]:.1%}")
            st.divider()
            st.subheader("Surface Consistency (Year-over-Year Std)")
            st.metric("Clay Std",  f"{row['clay_win_rate_std']:.3f}")
            st.metric("Grass Std", f"{row['grass_win_rate_std']:.3f}")
            st.divider()
            st.subheader("Engineered Features")
            st.metric("Surface Flexibility (clay × grass)", f"{row['surface_flexibility']:.3f}")
            st.metric("Surface Gap |clay − grass|",         f"{row['surface_gap']:.3f}")
            st.metric("Surface Floor (worst surface)",      f"{row['surface_floor']:.3f}")

    with tab_model:
        st.subheader("Cross-Validation Performance (5-Fold)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ridge R²",    f"{ridge_cv.r2_cv_mean:.3f}",  f"± {ridge_cv.r2_cv_std:.3f}")
        m2.metric("Ridge RMSE",  f"{ridge_cv.rmse_cv_mean:.3f}")
        m3.metric("Forest R²",   f"{forest_cv.r2_cv_mean:.3f}", f"± {forest_cv.r2_cv_std:.3f}")
        m4.metric("Forest RMSE", f"{forest_cv.rmse_cv_mean:.3f}")
        st.caption(
            "R² = variance explained by surface + era features alone (no grand_slams, no weeks_at_no1). "
            "Forest > Ridge confirms non-linear era × surface thresholds exist in the data."
        )
        st.divider()
        col_imp, col_coef = st.columns(2)
        with col_imp:
            st.plotly_chart(
                plot_feature_importance(forest.feature_importance(),
                                        title="Random Forest — Feature Importance"),
                use_container_width=True,
            )
        with col_coef:
            st.plotly_chart(
                plot_feature_importance(ridge.coefficients(),
                                        title="Ridge — Scaled Coefficients (green=+, red=−)"),
                use_container_width=True,
            )
        st.divider()
        model_choice = st.radio("Predicted vs Actual — select model",
                                ["Random Forest", "Ridge"], horizontal=True)
        pred_col = "forest_pred" if model_choice == "Random Forest" else "ridge_pred"
        plot_df = df_preds[df_preds["era"].isin(selected_eras)].copy()
        st.plotly_chart(
            plot_predicted_vs_actual(plot_df, TARGET_COL, pred_col),
            use_container_width=True,
        )


# ── NBA page ──────────────────────────────────────────────────────────────────

def _nba_page() -> None:
    st.title("Greatest of All Time — NBA")
    st.caption("Rule-based GOAT scoring · NBA · 1481 Players · 1947–2026")

    df = load_nba_data()

    st.sidebar.header("Filters")
    all_eras  = sorted(df["era"].unique().tolist())
    era_labels = {
        "pre_advanced": "Pre-Advanced (≤1973)",
        "pre_3pt":      "Pre-3pt (1974–1979)",
        "modern":       "Modern (1980–2010)",
        "analytics":    "Analytics (2011–)",
    }
    era_display      = [era_labels.get(e, e) for e in all_eras]
    selected_display = st.sidebar.multiselect("Era", era_display, default=era_display)
    selected_eras    = [all_eras[era_display.index(d)] for d in selected_display]

    top_n      = st.sidebar.slider("Top N players", 10, 50, 20, 5)
    champ_only = st.sidebar.checkbox("Champions only", value=False)
    min_games  = st.sidebar.slider("Min career games", 200, 800, 400, 50)

    filtered = df[df["era"].isin(selected_eras)].copy()
    if champ_only:
        filtered = filtered[filtered["championships"] >= 1]
    filtered = filtered[filtered["total_games"] >= min_games]

    tab_rank, tab_player, tab_model = st.tabs([
        "🏆 Rankings", "🔍 Player Deep-Dive", "🤖 Model Insights",
    ])

    with tab_rank:
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.plotly_chart(
                plot_goat_rankings(filtered, top_n=top_n, color_map=_NBA_ERA_COLORS),
                use_container_width=True,
            )
        with col_right:
            st.subheader("Top 3")
            top3 = filtered.nlargest(3, "goat_score")
            for rank, (_, row) in enumerate(top3.iterrows(), 1):
                medal = ["🥇", "🥈", "🥉"][rank - 1]
                st.metric(label=f"{medal} #{rank}", value=row["name"],
                          delta=f"{row['goat_score']:.1f} pts")
            st.divider()
            st.subheader("Era breakdown")
            for era, count in filtered["era"].value_counts().items():
                st.write(f"**{era_labels.get(era, era)}:** {count} players")

        st.subheader("Full Rankings Table")
        table_cols = ["name", "era", "goat_score", "championships",
                      "mvp_awards", "career_ws", "peak_bpm", "years_active"]
        display = (
            filtered[table_cols]
            .sort_values("goat_score", ascending=False)
            .reset_index(drop=True)
        )
        display.index += 1
        display["era"] = display["era"].map(era_labels).fillna(display["era"])
        st.dataframe(
            display.style.format({
                "goat_score": "{:.2f}",
                "career_ws":  "{:.1f}",
                "peak_bpm":   "{:.2f}",
            }),
            use_container_width=True,
            height=400,
        )

    with tab_player:
        player_names = filtered.sort_values("goat_score", ascending=False)["name"].tolist()
        selected = st.selectbox("Select player", player_names, key="nba_player")
        row = filtered[filtered["name"] == selected].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("GOAT Score",    f"{row['goat_score']:.2f}")
        col2.metric("Era",           era_labels.get(str(row["era"]), str(row["era"])))
        col3.metric("Championships", int(row["championships"]))
        col4.metric("Years Active",  int(row["years_active"]))

        st.divider()
        col_radar, col_stats = st.columns(2)
        with col_radar:
            radar_metrics = {
                _NBA_RADAR_LABELS[col]: float(row[col])
                for col in _NBA_RADAR_METRICS
                if col in row.index
            }
            st.plotly_chart(plot_player_radar(radar_metrics, selected), use_container_width=True)
        with col_stats:
            st.subheader("Career Stats")
            st.metric("Win Shares",          f"{row['career_ws']:.1f}")
            st.metric("Peak BPM",            f"{row['peak_bpm']:.2f}")
            st.metric("MVP Awards",          int(row["mvp_awards"]))
            st.metric("All-Star Selections", int(row["all_star_selections"]))
            st.metric("Finals MVP",          int(row["finals_mvp"]))
            st.divider()
            st.subheader("Layer Scores")
            for metric, label in _NBA_RADAR_LABELS.items():
                if metric in row.index:
                    st.metric(label, f"{row[metric]:.1f}")

    with tab_model:
        st.subheader("Random Forest — Feature Importance")
        st.caption(
            "Forest predicts goat_score from normalized stat columns (R²=0.976). "
            "Feature importance validates that YAML scorer weights are non-linear consistent: "
            "career_ws dominates (49%), followed by career_vorp (21%) and peak_bpm (13%)."
        )
        with st.spinner("Fitting NBA model (first run ~5s)…"):
            forest = fit_nba_forest()
        importance_dict = forest.feature_importance().to_dict()
        st.plotly_chart(
            plot_feature_importance(importance_dict,
                                    title="NBA Forest — Normalized Feature Importance"),
            use_container_width=True,
        )


# ── Swimming page ─────────────────────────────────────────────────────────────

def _swimming_page() -> None:
    st.title("Greatest of All Time — Swimming")
    st.caption(
        "Rule-based GOAT scoring · World Aquatics · 35 Elite Swimmers · All-Time GOAT Candidates"
    )

    df = load_swimming_data()

    st.sidebar.header("Filters")
    all_eras      = ["Pre-Modern", "Amateur", "Modern"]
    selected_eras = st.sidebar.multiselect("Era", all_eras, default=all_eras)
    top_n         = st.sidebar.slider("Top N swimmers", 5, 35, 20, 5)
    st.sidebar.info(
        "**Era adjustment ON** — Pre-Modern athletes (before 1973) receive estimated WC "
        "gold medals proportional to their Olympic dominance. World Championships did "
        "not exist until 1973."
    )

    filtered = df[df["era"].isin(selected_eras)].copy()

    tab_rank, tab_player = st.tabs(["🏆 Rankings", "🔍 Athlete Deep-Dive"])

    # ── Rankings ──────────────────────────────────────────────────────────────
    with tab_rank:
        col_left, col_right = st.columns([3, 1])

        with col_left:
            st.plotly_chart(
                plot_goat_rankings(filtered, top_n=top_n, color_map=_SWIMMING_ERA_COLORS),
                use_container_width=True,
            )

        with col_right:
            st.subheader("Top 3")
            top3 = filtered.nlargest(3, "goat_score")
            for rank, (_, row) in enumerate(top3.iterrows(), 1):
                medal = ["🥇", "🥈", "🥉"][rank - 1]
                st.metric(
                    label=f"{medal} #{rank}",
                    value=row["name"],
                    delta=f"{row['goat_score']:.1f} pts",
                )
            st.divider()
            st.subheader("Era breakdown")
            for era in all_eras:
                count = (filtered["era"] == era).sum()
                if count > 0:
                    st.write(f"**{era}:** {count} swimmer{'s' if count != 1 else ''}")

        st.subheader("Full Rankings Table")
        display_cols = [c for c in _SWIMMING_TABLE_COLS if c in filtered.columns]
        display = (
            filtered[display_cols]
            .sort_values("goat_score", ascending=False)
            .reset_index(drop=True)
            .rename(columns=_SWIMMING_TABLE_HEADERS)
        )
        display.index += 1
        st.dataframe(
            display.style.format({
                "GOAT Score":    "{:.2f}",
                "Career Years":  "{:.0f}",
                "World Records": "{:.0f}",
                "Events Dom.":   "{:.0f}",
            }),
            use_container_width=True,
            height=500,
        )

        st.divider()
        st.subheader("Era × Layer Score Averages")
        layer_cols = ["achievement_score", "peak_score", "versatility_score", "longevity_score"]
        existing_layers = [c for c in layer_cols if c in filtered.columns]
        if existing_layers:
            era_avg = (
                filtered.groupby("era")[existing_layers]
                .mean()
                .round(1)
                .rename(columns={c: c.replace("_score", "").capitalize() for c in existing_layers})
            )
            st.dataframe(era_avg, use_container_width=True)

    # ── Athlete Deep-Dive ─────────────────────────────────────────────────────
    with tab_player:
        athlete_names = filtered.sort_values("goat_score", ascending=False)["name"].tolist()
        selected = st.selectbox("Select athlete", athlete_names, key="sw_athlete")
        row = filtered[filtered["name"] == selected].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("GOAT Score",   f"{row['goat_score']:.2f}")
        col2.metric("Era",          row["era"])
        col3.metric("OG Gold (Ind.)", int(row["olympic_gold_individual"]))
        col4.metric("Career Years", int(row["career_years"]))

        st.divider()
        col_radar, col_stats = st.columns(2)

        with col_radar:
            radar_metrics = {
                _SWIMMING_RADAR_LABELS[col]: float(row[col])
                for col in _SWIMMING_RADAR_METRICS
                if col in row.index
            }
            st.plotly_chart(
                plot_player_radar(radar_metrics, selected),
                use_container_width=True,
            )

        with col_stats:
            st.subheader("Olympic Medals")
            st.metric("Gold — Individual",  int(row["olympic_gold_individual"]))
            st.metric("Gold — Relay",       int(row["olympic_gold_relay"]))
            st.metric("Silver — Individual",int(row["olympic_silver_individual"]))
            st.metric("Bronze — Individual",int(row["olympic_bronze_individual"]))
            st.metric("Games Attended",     int(row["olympic_games_count"]))

            st.divider()
            st.subheader("World Championships")
            wc_ind = row["wc_gold_individual"]
            if row["era"] == "Pre-Modern":
                st.metric(
                    "Gold — Individual (estimated)",
                    f"{wc_ind:.1f}",
                    help="Estimated via era adjustment — WC did not exist before 1973.",
                )
            else:
                st.metric("Gold — Individual", int(wc_ind))
            st.metric("Gold — Relay",       int(row["wc_gold_relay"]))

            st.divider()
            st.subheader("Signature Stats")
            st.metric("World Records",    int(row["world_records"]))
            st.metric("Events Dominated", int(row["events_dominated"]))

            st.divider()
            st.subheader("Layer Scores")
            for metric, label in _SWIMMING_RADAR_LABELS.items():
                if metric in row.index:
                    st.metric(label, f"{row[metric]:.1f}")


if __name__ == "__main__":
    main()
