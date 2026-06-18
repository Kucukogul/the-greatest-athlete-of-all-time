from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.scoring.normalizer import normalize_series

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("configs/scoring_swimming.yaml")

_REQUIRED_RAW_COLUMNS: frozenset[str] = frozenset({
    "athlete_id",
    "era",
    "career_years",
    "olympic_gold_individual",
    "olympic_gold_relay",
    "wc_gold_individual",
    "wc_gold_relay",
    "world_records",
    "events_dominated",
})

# normalized output column → source column (after era adjustment)
_METRIC_MAP: dict[str, str] = {
    "olympic_gold_individual_normalized": "olympic_gold_individual",
    "wc_gold_individual_normalized":      "wc_gold_individual",
    "olympic_gold_relay_normalized":      "olympic_gold_relay",
    "wc_gold_relay_normalized":           "wc_gold_relay",
    "world_records_normalized":           "world_records",
    "events_dominated_normalized":        "events_dominated",
    "career_years_normalized":            "career_years",
}


class SwimmingNormalizer:
    """SwimmingPipeline output → normalized [0–100] columns.

    Pipeline order (must not change):
      1. validate          — required columns, no duplicates
      2. era_adjust        — Pre-Modern wc_gold_individual imputation (optional, default on)
      3. normalize_metrics — MinMax [0–100] across full athlete population

    Era adjustment note:
      World Championships did not exist before 1973. Pre-Modern athletes have
      wc_gold_individual = 0 by structural absence. We impute:
          estimated_wc = olympic_gold_individual × wc_per_og_ratio
      using the ratio derived from Modern era athletes in EDA (≈ 2.89).
      Adjustment modifies the working copy only — raw data is never changed.
    """

    def __init__(self, config_path: str | Path = _CONFIG_PATH) -> None:
        with open(config_path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        self._era_defs: dict = self._config.get("era_definitions", {})

    def normalize(self, df: pd.DataFrame, *, era_adjust: bool = True) -> pd.DataFrame:
        """Return a new DataFrame with all _normalized columns appended."""
        self._validate(df)
        result = df.copy()
        if era_adjust:
            result = self._apply_era_adjustment(result)
        result = self._normalize_metrics(result)
        logger.info(
            "SwimmingNormalizer complete — %d athletes, era_adjust=%s",
            len(result),
            era_adjust,
        )
        return result

    @staticmethod
    def output_columns() -> list[str]:
        return list(_METRIC_MAP.keys())

    # ── Private steps ─────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        missing = _REQUIRED_RAW_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required raw columns: {sorted(missing)}")
        if df["athlete_id"].duplicated().any():
            raise ValueError("Duplicate athlete_id values found — one row per career.")

    def _apply_era_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        pre_modern_label = self._era_defs.get("pre_modern", {}).get("label", "Pre-Modern")
        ratio = float(self._era_defs.get("pre_modern", {}).get("wc_per_og_ratio", 2.89))

        mask = df["era"] == pre_modern_label
        n_adjusted = mask.sum()
        if n_adjusted > 0:
            df["wc_gold_individual"] = df["wc_gold_individual"].astype(float)
            df.loc[mask, "wc_gold_individual"] = (
                df.loc[mask, "olympic_gold_individual"] * ratio
            ).round(2)
            logger.info(
                "Era adjustment applied to %d Pre-Modern athlete(s) — wc_per_og_ratio=%.2f",
                n_adjusted,
                ratio,
            )
        return df

    def _normalize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        for output_col, source_col in _METRIC_MAP.items():
            df[output_col] = normalize_series(df[source_col], method="minmax")
        return df
