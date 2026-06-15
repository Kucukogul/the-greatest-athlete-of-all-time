from __future__ import annotations

import logging

import pandas as pd

from src.scoring.normalizer import normalize_series

logger = logging.getLogger(__name__)

# output_col → source_col
# BPM and WS/48 are league-average-relative by construction — no era scaling applied.
# BPM/VORP columns are NaN for pre-advanced era players (pre-1974); NaN propagates through.

_PEAK_MAP: dict[str, str] = {
    "peak_bpm_normalized":  "peak_bpm",
    "peak_ws48_normalized": "peak_ws48",
    "peak_vorp_normalized": "peak_vorp_season",
}

_CAREER_MAP: dict[str, str] = {
    "career_vorp_normalized": "career_vorp",
    "career_ws_normalized":   "career_ws",
    "career_bpm_normalized":  "career_bpm",
}

_EFFICIENCY_MAP: dict[str, str] = {
    "career_ws48_normalized": "career_ws48",
    "career_ts_normalized":   "career_ts_pct",
}

_ACHIEVEMENT_MAP: dict[str, str] = {
    "championships_normalized": "championships",
    "mvp_normalized":           "mvp_awards",
    "finals_mvp_normalized":    "finals_mvp",
    "all_nba_1st_normalized":   "all_nba_1st_team",
    "all_nba_2nd_normalized":   "all_nba_2nd_team",
    "all_star_normalized":      "all_star_selections",
    "dpoy_normalized":          "dpoy_awards",
    "def_1st_normalized":       "all_defensive_1st_team",
}

_LONGEVITY_MAP: dict[str, str] = {
    "longevity_normalized": "years_active",
}

_LAYER_MAPS: dict[str, dict[str, str]] = {
    "peak":        _PEAK_MAP,
    "career":      _CAREER_MAP,
    "efficiency":  _EFFICIENCY_MAP,
    "achievement": _ACHIEVEMENT_MAP,
    "longevity":   _LONGEVITY_MAP,
}

_REQUIRED_COLS: frozenset[str] = frozenset(
    {src for m in _LAYER_MAPS.values() for src in m.values()}
    | {"player_id", "era"}
)


class NBANormalizer:
    """NBAPipeline output → normalized [0–100] columns, one per source metric.

    Pre-advanced era players (pre-1974) have NaN for BPM/VORP columns.
    NaN propagates through normalization; the scorer rescales by available weights.
    """

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with all _normalized columns appended.

        MinMax is applied across the full player pool. NaN values are excluded
        from min/max computation and propagate as NaN in the output.
        """
        self._validate(df)
        result = df.copy()

        for layer, layer_map in _LAYER_MAPS.items():
            for out_col, src_col in layer_map.items():
                result[out_col] = normalize_series(result[src_col])

        logger.debug(
            "%d players have NaN peak_bpm_normalized (pre-1974, BPM unavailable)",
            result["peak_bpm_normalized"].isna().sum(),
        )
        return result

    @staticmethod
    def output_columns() -> list[str]:
        """All normalized column names in layer order."""
        return [col for m in _LAYER_MAPS.values() for col in m.keys()]

    @staticmethod
    def layer_columns(layer: str) -> list[str]:
        """Normalized column names for a specific scoring layer."""
        if layer not in _LAYER_MAPS:
            raise ValueError(
                f"Unknown layer '{layer}'. Valid layers: {sorted(_LAYER_MAPS)}"
            )
        return list(_LAYER_MAPS[layer].keys())

    @staticmethod
    def layers() -> list[str]:
        """Ordered list of scoring layer names."""
        return list(_LAYER_MAPS.keys())

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if df["player_id"].duplicated().any():
            raise ValueError(
                "Duplicate player_id values found — NBANormalizer expects one row per player."
            )
