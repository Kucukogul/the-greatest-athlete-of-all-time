from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_ERA_BINS: list[int] = [0, 1973, 1979, 2010, 9999]
_ERA_LABELS: list[str] = ["pre_advanced", "pre_3pt", "modern", "analytics"]
_ERA_ORDER: dict[str, int] = {
    "pre_advanced": 0,
    "pre_3pt": 1,
    "modern": 2,
    "analytics": 3,
}

_ACHIEVEMENT_DEFAULTS: dict[str, int] = {
    "championships": 0,
    "mvp_awards": 0,
    "finals_mvp": 0,
    "all_nba_1st_team": 0,
    "all_nba_2nd_team": 0,
    "all_star_selections": 0,
    "dpoy_awards": 0,
    "all_defensive_1st_team": 0,
}

_SEASONS_USECOLS: list[str] = [
    "Year", "Player", "player_id", "Tm",
    "G", "MP",
    "TS%", "FTA", "FGA",
    "WS", "WS/48",
    "BPM", "VORP",
]


class NBAPipeline:
    """nba_all_seasons_raw.csv → one career-stats row per player. Pure transformation, no file writes."""

    _MULTI_TEAM_LABELS: frozenset[str] = frozenset({"TOT", "2TM", "3TM"})
    _MIN_GAMES_PEAK: int = 40

    def __init__(self, data_dir: str | Path, config_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._config_dir = Path(config_dir)

    def run(
        self,
        min_career_games: int = 400,
        min_qualifying_seasons: int = 5,
    ) -> pd.DataFrame:
        """Return one row per qualifying player, ready for NBANormalizer."""
        logger.info("Loading NBA seasons from %s", self._data_dir)
        seasons = self._load_seasons()
        seasons = self._dedup_multi_team(seasons)
        self._detect_name_collisions(seasons)

        achievements = self._load_achievements()

        career = self._aggregate_careers(seasons)
        career = self._apply_career_threshold(career, min_career_games, min_qualifying_seasons)

        peak = self._compute_peak_metrics(seasons)
        result = career.merge(peak, on="player_id", how="left")

        result = self._join_achievements(result, achievements)
        result = self._assign_era(result, seasons)

        logger.info("Pipeline complete — %d player records produced", len(result))
        return result

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_seasons(self) -> pd.DataFrame:
        path = self._data_dir / "nba_all_seasons_raw.csv"
        if not path.exists():
            raise FileNotFoundError(f"Canonical seasons file not found: {path}")

        df = pd.read_csv(path, usecols=_SEASONS_USECOLS, low_memory=False)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
        df["G"] = pd.to_numeric(df["G"], errors="coerce")
        df["MP"] = pd.to_numeric(df["MP"], errors="coerce")
        return df[df["Year"] > 0].reset_index(drop=True)

    def _load_achievements(self) -> dict[str, dict]:
        path = self._config_dir / "nba_achievements.yaml"
        if not path.exists():
            logger.warning("Achievements file not found: %s — all achievement metrics default to 0", path)
            return {}
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return raw.get("players", {})

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _dedup_multi_team(self, seasons: pd.DataFrame) -> pd.DataFrame:
        """Keep aggregate row (2TM/3TM/TOT) per (player_id, Year); drop individual team rows.

        Players who never changed teams are unaffected.
        """
        is_agg = seasons["Tm"].isin(self._MULTI_TEAM_LABELS)

        agg_idx = pd.MultiIndex.from_arrays(
            [seasons.loc[is_agg, "player_id"], seasons.loc[is_agg, "Year"]]
        )
        row_idx = pd.MultiIndex.from_arrays([seasons["player_id"], seasons["Year"]])

        # keep if NOT in an agg combo, OR IS the aggregate row itself
        in_agg_combo = row_idx.isin(agg_idx)
        keep = ~in_agg_combo | is_agg.values

        deduped = seasons[keep].reset_index(drop=True)
        dropped = len(seasons) - len(deduped)
        logger.info("Dedup: removed %d per-team rows for multi-team seasons", dropped)
        return deduped

    # ── Validation ────────────────────────────────────────────────────────────

    def _detect_name_collisions(self, seasons: pd.DataFrame) -> None:
        """Log WARNING for player_ids linked to multiple distinct names (encoding variants only)."""
        name_counts = (
            seasons.dropna(subset=["player_id", "Player"])
            .groupby("player_id")["Player"]
            .nunique()
        )
        collisions = name_counts[name_counts > 1]
        for pid in collisions.index:
            names = seasons[seasons["player_id"] == pid]["Player"].unique().tolist()
            logger.warning("Possible name collision — player_id=%s names=%s", pid, names)

    # ── Career aggregation ────────────────────────────────────────────────────

    def _aggregate_careers(self, seasons: pd.DataFrame) -> pd.DataFrame:
        records = [
            rec
            for pid, group in seasons.groupby("player_id")
            if (rec := self._career_record(pid, group)) is not None
        ]
        return pd.DataFrame(records)

    def _career_record(self, player_id: str, group: pd.DataFrame) -> dict | None:
        if group.empty:
            return None

        name = (
            group["Player"].dropna().value_counts().index[0]
            if group["Player"].notna().any()
            else str(player_id)
        )

        g_valid = group.dropna(subset=["G"])
        total_games = int(g_valid["G"].sum())
        total_mp = float(group["MP"].sum(skipna=True))

        years = group.loc[group["Year"] > 0, "Year"]
        year_start = int(years.min()) if not years.empty else 0
        year_end = int(years.max()) if not years.empty else 0
        years_active = (year_end - year_start + 1) if year_start > 0 else 0

        qualifying_seasons = int((group["G"] >= self._MIN_GAMES_PEAK).sum())

        vorp_valid = group["VORP"].dropna()
        career_vorp = round(float(vorp_valid.sum()), 2) if not vorp_valid.empty else None

        ws_valid = group["WS"].dropna()
        career_ws = round(float(ws_valid.sum()), 2) if not ws_valid.empty else None

        career_bpm = self._mp_weighted_mean(group, "BPM")
        career_ws48 = self._mp_weighted_mean(group, "WS/48")
        career_ts_pct = self._shot_weighted_ts(group)

        return {
            "player_id":           player_id,
            "name":                name,
            "total_games":         total_games,
            "total_mp":            total_mp,
            "year_start":          year_start,
            "year_end":            year_end,
            "years_active":        years_active,
            "qualifying_seasons":  qualifying_seasons,
            "career_vorp":         career_vorp,
            "career_ws":           career_ws,
            "career_bpm":          round(career_bpm, 3) if career_bpm is not None else None,
            "career_ws48":         round(career_ws48, 4) if career_ws48 is not None else None,
            "career_ts_pct":       round(career_ts_pct, 4) if career_ts_pct is not None else None,
        }

    def _apply_career_threshold(
        self,
        career: pd.DataFrame,
        min_career_games: int,
        min_qualifying_seasons: int,
    ) -> pd.DataFrame:
        """Remove players below threshold. Fringe players with name/id collisions are
        naturally eliminated here without requiring explicit disambiguation."""
        before = len(career)
        mask = (
            (career["total_games"] >= min_career_games)
            & (career["qualifying_seasons"] >= min_qualifying_seasons)
        )
        result = career[mask].reset_index(drop=True)
        logger.info(
            "Career threshold (min_games=%d, min_seasons=%d): %d → %d players",
            min_career_games,
            min_qualifying_seasons,
            before,
            len(result),
        )
        return result

    # ── Peak metrics ──────────────────────────────────────────────────────────

    def _compute_peak_metrics(self, seasons: pd.DataFrame) -> pd.DataFrame:
        """Best 3-consecutive qualifying-season window for BPM and WS/48 per player."""
        quality = seasons[seasons["G"] >= self._MIN_GAMES_PEAK]
        records = [
            self._peak_record(pid, group)
            for pid, group in quality.groupby("player_id")
        ]
        return pd.DataFrame(records)

    def _peak_record(self, player_id: str, group: pd.DataFrame) -> dict:
        g = group.sort_values("Year")
        peak_bpm, peak_bpm_year = self._peak_3_consecutive(g, "BPM")
        peak_ws48, peak_ws48_year = self._peak_3_consecutive(g, "WS/48")

        vorp_valid = g["VORP"].dropna()
        peak_vorp_season = round(float(vorp_valid.max()), 2) if not vorp_valid.empty else None

        return {
            "player_id":          player_id,
            "peak_bpm":           round(peak_bpm, 3) if peak_bpm is not None else None,
            "peak_bpm_start_year": peak_bpm_year,
            "peak_ws48":          round(peak_ws48, 4) if peak_ws48 is not None else None,
            "peak_ws48_start_year": peak_ws48_year,
            "peak_vorp_season":   peak_vorp_season,
        }

    # ── Achievements join ──────────────────────────────────────────────────────

    def _join_achievements(
        self,
        career: pd.DataFrame,
        achievements: dict[str, dict],
    ) -> pd.DataFrame:
        """Left join achievements onto career frame. Players absent from YAML default to 0."""
        ach_cols = list(_ACHIEVEMENT_DEFAULTS.keys())

        ach_df = pd.DataFrame(
            [
                {"player_id": pid, **{col: info.get(col, 0) for col in ach_cols}}
                for pid, info in achievements.items()
            ]
        ) if achievements else pd.DataFrame(columns=["player_id"] + ach_cols)

        result = career.merge(ach_df, on="player_id", how="left")
        for col in ach_cols:
            result[col] = result[col].fillna(0).astype(int)
        return result

    # ── Era assignment ────────────────────────────────────────────────────────

    def _assign_era(
        self,
        career: pd.DataFrame,
        seasons: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign era from the median year of a player's qualifying seasons.

        Median year anchors to the player's prime rather than career endpoints.
        """
        quality = seasons[seasons["G"] >= self._MIN_GAMES_PEAK]
        median_year = (
            quality.groupby("player_id")["Year"].median().rename("_median_year")
        )
        result = career.merge(median_year, on="player_id", how="left")
        result["era"] = pd.cut(
            result["_median_year"],
            bins=_ERA_BINS,
            labels=_ERA_LABELS,
        ).astype(str)
        result["era_encoded"] = result["era"].map(_ERA_ORDER).fillna(-1).astype(int)
        return result.drop(columns=["_median_year"])

    # ── Statistical helpers ───────────────────────────────────────────────────

    @staticmethod
    def _mp_weighted_mean(df: pd.DataFrame, metric: str) -> float | None:
        """MP-weighted average of metric; returns None when metric is entirely absent."""
        valid = df.dropna(subset=[metric, "MP"])
        if valid.empty or valid["MP"].sum() == 0:
            return None
        return float(np.average(valid[metric], weights=valid["MP"]))

    @staticmethod
    def _shot_weighted_ts(df: pd.DataFrame) -> float | None:
        """(FTA + FGA)-weighted average TS%; returns None when TS% is entirely absent."""
        valid = df.dropna(subset=["TS%", "FTA", "FGA"])
        if valid.empty:
            return None
        weights = valid["FTA"] + valid["FGA"]
        total = weights.sum()
        if total == 0:
            return None
        return float(np.average(valid["TS%"], weights=weights))

    @staticmethod
    def _peak_3_consecutive(
        group: pd.DataFrame,
        metric: str,
    ) -> tuple[float | None, int | None]:
        """Best 3-consecutive-qualifying-season window average for metric.

        'Consecutive' means consecutive qualifying seasons (G >= 40), not calendar years.
        Seasons where metric is NaN are excluded before windowing.
        Falls back to available-season mean when fewer than 3 valid seasons exist.
        """
        valid = group.dropna(subset=[metric]).sort_values("Year")
        s = valid[metric].reset_index(drop=True)
        years = valid["Year"].reset_index(drop=True)

        if s.empty:
            return None, None

        if len(s) < 3:
            return float(s.mean()), int(years.iloc[0])

        best_avg = -np.inf
        best_year: int | None = None
        for i in range(len(s) - 2):
            avg = float(s.iloc[i: i + 3].mean())
            if avg > best_avg:
                best_avg = avg
                best_year = int(years.iloc[i])

        return best_avg, best_year
