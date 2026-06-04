from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.scoring.normalizer import normalize_series

_CONFIG_PATH = Path("configs/scoring_tennis.yaml")

_REQUIRED_RAW_COLUMNS: frozenset[str] = frozenset({
    "athlete_id",
    "name",
    "sport",
    "era",
    "grand_slams",
    "weeks_at_no1",
    "masters_titles",
    "career_wins",
    "career_matches",
    "yearend_no1_count",
    "finals_won",
    "finals_played",
    "h2h_top10_wins",
    "h2h_top10_total",
    "hard_win_pct",
    "clay_win_pct",
    "grass_win_pct",
    "years_in_top5",
})

_METRIC_MAP: dict[str, str] = {
    "grand_slam_normalized":          "grand_slams",
    "weeks_no1_normalized":           "weeks_at_no1",
    "masters_titles_normalized":      "masters_titles",
    "career_win_rate_normalized":     "career_win_rate",
    "yearend_no1_normalized":         "yearend_no1_count",
    "finals_win_rate_normalized":     "finals_win_rate",
    "h2h_top10_normalized":           "h2h_top10",
    "surface_versatility_normalized": "surface_versatility",
    "longevity_normalized":           "years_in_top5",
}


class TennisNormalizer:
    """Raw tennis career stats → normalized [0–100] columns.

    Pipeline order (must not change):
      1. validate
      2. compute_derived
      3. era_adjust  (optional)
      4. normalize_metrics
    """

    def __init__(self, config_path: str | Path = _CONFIG_PATH) -> None:
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        self._era_defs: dict = self._config.get("era_definitions", {})
        self._surface_cfg: dict = self._config.get("surface_versatility", {})

    def normalize(self, df: pd.DataFrame, *, era_adjust: bool = True) -> pd.DataFrame:
        """Return a new DataFrame with all _normalized columns appended."""
        self._validate(df)
        result = df.copy()
        result = self._compute_derived_metrics(result)
        if era_adjust:
            result = self._apply_era_adjustment(result)
        result = self._normalize_metrics(result)
        return result

    @staticmethod
    def output_columns() -> list[str]:
        return list(_METRIC_MAP.keys())

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        missing = _REQUIRED_RAW_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required raw columns: {sorted(missing)}")
        if df["athlete_id"].duplicated().any():
            raise ValueError("Duplicate athlete_id values found — one row per career.")

    def _compute_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df["career_win_rate"] = _safe_ratio(df["career_wins"], df["career_matches"])
        df["finals_win_rate"] = _safe_ratio(df["finals_won"], df["finals_played"])
        df["h2h_top10"] = _safe_ratio(df["h2h_top10_wins"], df["h2h_top10_total"])
        df["surface_versatility"] = self._compute_surface_versatility(df)
        return df

    def _compute_surface_versatility(self, df: pd.DataFrame) -> pd.Series:
        """Inverse std across hard/clay/grass win rates — lower spread = higher score."""
        surfaces = self._surface_cfg.get("surfaces", ["hard", "clay", "grass"])
        pct_cols = [f"{s}_win_pct" for s in surfaces]
        stds = df[pct_cols].std(axis=1)
        max_std = stds.max()
        if max_std == 0:
            return pd.Series(100.0, index=df.index)
        return (1.0 - stds / max_std) * 100.0

    def _apply_era_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale grand_slams and weeks_at_no1 to Big 3 Era equivalents.

        Big 3 Era is used as the reference (factor = 1.0). Open/Modern Era players
        get their counts scaled up proportionally to their era's competitive depth.
        """
        ref = self._era_defs.get("big3_era", {})
        ref_gs = ref.get("gs_per_year_mean", 1.0)
        ref_weeks = ref.get("weeks_no1_mean", 1.0)

        df["grand_slams"] = df.apply(
            lambda row: row["grand_slams"] * self._gs_factor(str(row["era"]), ref_gs),
            axis=1,
        )
        df["weeks_at_no1"] = df.apply(
            lambda row: row["weeks_at_no1"] * self._weeks_factor(str(row["era"]), ref_weeks),
            axis=1,
        )
        return df

    def _gs_factor(self, era_label: str, ref_mean: float) -> float:
        for era_data in self._era_defs.values():
            if era_data.get("label") == era_label:
                era_mean = era_data.get("gs_per_year_mean", ref_mean)
                return ref_mean / era_mean if era_mean > 0 else 1.0
        return 1.0

    def _weeks_factor(self, era_label: str, ref_mean: float) -> float:
        for era_data in self._era_defs.values():
            if era_data.get("label") == era_label:
                era_mean = era_data.get("weeks_no1_mean", ref_mean)
                return ref_mean / era_mean if era_mean > 0 else 1.0
        return 1.0

    def _normalize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        for output_col, source_col in _METRIC_MAP.items():
            df[output_col] = normalize_series(df[source_col], method="minmax")
        return df


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, float("nan"))).fillna(0.0)
