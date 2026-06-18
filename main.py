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

_DATA_DIR       = Path("data/external/tennis_atp")
_OUTPUT_V2      = Path("data/processed/tennis_all_v2.csv")
_NBA_DATA_DIR   = Path("data/raw")
_NBA_CONFIG_DIR = Path("configs")
_NBA_OUTPUT     = Path("data/processed/nba_goat_v1.csv")
_SW_RAW_DIR     = Path("data/raw")
_SW_OUTPUT      = Path("data/processed/swimming_goat_v1.csv")


def run_tennis_pipeline(
    data_dir: Path = _DATA_DIR,
    output_path: Path = _OUTPUT_V2,
    min_matches: int = 200,
) -> None:
    from src.pipelines.tennis_pipeline import TennisPipeline
    from src.normalize.tennis_normalizer import TennisNormalizer
    from src.scoring.scorer import AthleteScorer, load_scoring_config

    logger.info("=== Tennis pipeline ===")
    logger.info("Step 1/3 — TennisPipeline (min_matches=%d)", min_matches)
    raw_df = TennisPipeline(data_dir).run(min_matches=min_matches)
    logger.info("Pipeline produced %d player records", len(raw_df))

    logger.info("Step 2/3 — TennisNormalizer")
    normalized_df = TennisNormalizer().normalize(raw_df, era_adjust=True)

    logger.info("Step 3/3 — Computing composite scores")
    config = load_scoring_config("configs/scoring_tennis.yaml")
    scorer = AthleteScorer(config)
    normalized_df["composite_score"] = scorer.score(normalized_df).round(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(normalized_df), output_path)


def run_nba_pipeline(
    data_dir: Path = _NBA_DATA_DIR,
    config_dir: Path = _NBA_CONFIG_DIR,
    output_path: Path = _NBA_OUTPUT,
    min_games: int = 400,
) -> None:
    from src.pipelines.nba_runner import NBARunner

    logger.info("=== NBA pipeline ===")
    logger.info("Step 1/1 — NBARunner (min_games=%d)", min_games)
    runner = NBARunner(data_dir, config_dir)
    result_df = runner.run()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(result_df), output_path)


def run_swimming_pipeline(
    raw_dir: Path = _SW_RAW_DIR,
    output_path: Path = _SW_OUTPUT,
) -> None:
    from src.pipelines.swimming_pipeline import SwimmingPipeline
    from src.normalize.swimming_normalizer import SwimmingNormalizer
    from src.scoring.swimming_scorer import SwimmingScorer

    logger.info("=== Swimming pipeline ===")
    logger.info("Step 1/3 — SwimmingPipeline")
    raw_df = SwimmingPipeline(raw_dir).run()
    logger.info("Pipeline produced %d athlete records", len(raw_df))

    logger.info("Step 2/3 — SwimmingNormalizer (era_adjust=True)")
    normalized_df = SwimmingNormalizer().normalize(raw_df, era_adjust=True)

    logger.info("Step 3/3 — SwimmingScorer")
    scorer = SwimmingScorer("configs/scoring_swimming.yaml")
    normalized_df["goat_score"] = scorer.score(normalized_df).round(2)
    layers = scorer.layer_scores(normalized_df)
    result_df = normalized_df.join(layers)

    top5 = (
        result_df[["name", "era", "goat_score"]]
        .sort_values("goat_score", ascending=False)
        .head(5)
    )
    logger.info("Top 5 swimmers:\n%s", top5.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.info("Saved %d rows → %s", len(result_df), output_path)


def main() -> None:
    logger.info("The Greatest Athlete of All Time — pipeline entry point")

    tennis_raw = Path("data/external/tennis_atp")
    if tennis_raw.exists():
        try:
            run_tennis_pipeline()
        except Exception as exc:
            logger.warning("Tennis pipeline skipped: %s", exc)
    else:
        logger.warning("Tennis raw data not found at %s — skipping", tennis_raw)

    nba_raw = Path("data/raw/nba_all_seasons_raw.csv")
    if nba_raw.exists():
        try:
            run_nba_pipeline()
        except Exception as exc:
            logger.warning("NBA pipeline skipped: %s", exc)
    else:
        logger.warning("NBA raw data not found at %s — skipping", nba_raw)

    sw_raw = Path("data/raw/swimming_athletes_raw.csv")
    if sw_raw.exists():
        try:
            run_swimming_pipeline()
        except Exception as exc:
            logger.warning("Swimming pipeline skipped: %s", exc)
    else:
        logger.warning("Swimming raw data not found at %s — skipping", sw_raw)


if __name__ == "__main__":
    main()
