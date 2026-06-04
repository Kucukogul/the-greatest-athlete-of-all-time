from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data/external/tennis_atp")
_OUTPUT_V2 = Path("data/processed/tennis_all_v2.csv")


def run_tennis_pipeline(
    data_dir: Path = _DATA_DIR,
    output_path: Path = _OUTPUT_V2,
    min_matches: int = 200,
) -> None:
    from src.pipelines.tennis_pipeline import TennisPipeline
    from src.normalize.tennis_normalizer import TennisNormalizer
    from src.scoring.scorer import AthleteScorer, load_scoring_config

    logger.info("Step 1/3 — Running TennisPipeline (min_matches=%d)", min_matches)
    raw_df = TennisPipeline(data_dir).run(min_matches=min_matches)
    logger.info("Pipeline produced %d player records", len(raw_df))

    logger.info("Step 2/3 — Normalizing with TennisNormalizer")
    normalizer = TennisNormalizer()
    normalized_df = normalizer.normalize(raw_df, era_adjust=True)

    logger.info("Step 3/3 — Computing composite scores")
    config = load_scoring_config("configs/scoring_tennis.yaml")
    scorer = AthleteScorer(config)
    normalized_df["composite_score"] = scorer.score(normalized_df).round(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(normalized_df), output_path)


def main() -> None:
    logger.info("The Greatest Athlete of All Time — pipeline entry point")
    run_tennis_pipeline()


if __name__ == "__main__":
    main()
