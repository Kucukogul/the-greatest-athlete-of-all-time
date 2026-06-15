from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.normalize.nba_normalizer import NBANormalizer
from src.pipelines.nba_pipeline import NBAPipeline
from src.scoring.nba_scorer import NBAScorer

logger = logging.getLogger(__name__)


class NBARunner:
    """End-to-end NBA GOAT ranking: Pipeline → Normalizer → Scorer.

    Chains the three NBA modules into a single call and returns a fully ranked
    DataFrame ready for the model layer, API, or visualisation.

    Output columns (in addition to all pipeline columns):
        - All *_normalized columns from NBANormalizer
        - goat_score     : composite GOAT score [0–100], NaN-safe
        - peak_score     : per-layer [0–100] for radar charts
        - career_score
        - efficiency_score
        - achievement_score
        - longevity_score
        - rank           : integer rank (1 = best), sorted by goat_score desc

    Usage:
        runner = NBARunner("data/raw", "configs")
        df = runner.run()  # 1481 rows × ~50 columns, ~3s on first call
    """

    def __init__(self, data_dir: str | Path, config_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._config_dir = Path(config_dir)
        self._pipeline = NBAPipeline(data_dir, config_dir)
        self._normalizer = NBANormalizer()
        self._scorer = NBAScorer(self._config_dir / "scoring_nba.yaml")

    def run(
        self,
        min_career_games: int = 400,
        min_qualifying_seasons: int = 5,
    ) -> pd.DataFrame:
        """Return one row per qualifying player, ranked by goat_score descending."""
        logger.info("NBARunner — step 1/3: pipeline")
        raw = self._pipeline.run(min_career_games, min_qualifying_seasons)

        logger.info("NBARunner — step 2/3: normalization")
        normalized = self._normalizer.normalize(raw)

        logger.info("NBARunner — step 3/3: scoring")
        normalized["goat_score"] = self._scorer.score(normalized)
        layer_df = self._scorer.layer_scores(normalized)
        result = pd.concat([normalized, layer_df], axis=1)

        result = result.sort_values("goat_score", ascending=False).reset_index(drop=True)
        result.insert(0, "rank", range(1, len(result) + 1))

        logger.info(
            "NBARunner — done. %d players ranked. Top: %s (%.1f)",
            len(result),
            result["name"].iloc[0],
            result["goat_score"].iloc[0],
        )
        return result
