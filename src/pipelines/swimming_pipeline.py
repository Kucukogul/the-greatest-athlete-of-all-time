from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RAW_FILE = "swimming_athletes_raw.csv"

_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "athlete_id",
    "name",
    "nationality",
    "era",
    "birth_year",
    "career_start",
    "career_end",
    "olympic_gold_individual",
    "olympic_gold_relay",
    "olympic_silver_individual",
    "olympic_silver_relay",
    "olympic_bronze_individual",
    "olympic_bronze_relay",
    "olympic_games_count",
    "wc_gold_individual",
    "wc_gold_relay",
    "wc_silver_individual",
    "wc_bronze_individual",
    "world_records",
    "events_dominated",
})

_VALID_ERAS: frozenset[str] = frozenset({"Pre-Modern", "Amateur", "Modern"})

_NUMERIC_COLUMNS: tuple[str, ...] = (
    "birth_year", "career_start", "career_end",
    "olympic_gold_individual", "olympic_gold_relay",
    "olympic_silver_individual", "olympic_silver_relay",
    "olympic_bronze_individual", "olympic_bronze_relay",
    "olympic_games_count",
    "wc_gold_individual", "wc_gold_relay",
    "wc_silver_individual", "wc_bronze_individual",
    "world_records", "events_dominated",
)


class SwimmingPipeline:
    """Raw swimming CSV → one career-stats row per athlete. Pure transformation, no file writes.

    Pipeline order (must not change):
      1. load      — read CSV, skip comment lines
      2. validate  — required columns, era labels, no duplicates, non-negative values
      3. derive    — compute career_years and olympic_gold_total
      4. reorder   — stable column order for downstream normalizer
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)

    def run(self) -> pd.DataFrame:
        """Return a clean DataFrame ready for SwimmingNormalizer. One row per athlete."""
        logger.info("Loading swimming data from %s", self._data_dir / _RAW_FILE)
        df = self._load()
        self._validate(df)
        df = self._derive(df)
        df = self._reorder(df)
        logger.info("Pipeline complete — %d athlete records produced", len(df))
        return df

    # ── Private steps ─────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        path = self._data_dir / _RAW_FILE
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")
        df = pd.read_csv(path, comment="#")
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Raw swimming data is empty.")

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if df["athlete_id"].duplicated().any():
            dupes = df[df["athlete_id"].duplicated(keep=False)]["athlete_id"].tolist()
            raise ValueError(f"Duplicate athlete_id values found: {dupes}")

        invalid_eras = set(df["era"].unique()) - _VALID_ERAS
        if invalid_eras:
            raise ValueError(
                f"Unknown era labels: {invalid_eras}. "
                f"Allowed: {sorted(_VALID_ERAS)}"
            )

        for col in _NUMERIC_COLUMNS:
            if (df[col] < 0).any():
                offenders = df[df[col] < 0]["name"].tolist()
                raise ValueError(f"Negative values in '{col}': {offenders}")

        invalid_career = df[df["career_end"] < df["career_start"]]
        if not invalid_career.empty:
            raise ValueError(
                f"career_end < career_start for: {invalid_career['name'].tolist()}"
            )

        invalid_events = df[df["events_dominated"] > df["olympic_gold_individual"]]
        if not invalid_events.empty:
            raise ValueError(
                f"events_dominated exceeds olympic_gold_individual for: "
                f"{invalid_events['name'].tolist()}"
            )

    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["career_years"] = out["career_end"] - out["career_start"] + 1
        out["olympic_gold_total"] = (
            out["olympic_gold_individual"] + out["olympic_gold_relay"]
        )
        return out

    def _reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        identity = ["athlete_id", "name", "nationality", "era", "birth_year"]
        career = ["career_start", "career_end", "career_years", "olympic_games_count"]
        og = [
            "olympic_gold_individual", "olympic_gold_relay", "olympic_gold_total",
            "olympic_silver_individual", "olympic_silver_relay",
            "olympic_bronze_individual", "olympic_bronze_relay",
        ]
        wc = [
            "wc_gold_individual", "wc_gold_relay",
            "wc_silver_individual", "wc_bronze_individual",
        ]
        other = ["world_records", "events_dominated"]
        ordered = identity + career + og + wc + other
        remaining = [c for c in df.columns if c not in ordered]
        return df[ordered + remaining]
