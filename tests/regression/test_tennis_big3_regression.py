"""
Tennis GOAT Regression Baseline — Federer / Nadal / Djokovic

These tests define the expected ordering and structural contracts
for the tennis normalization pipeline using the Big 3 as a reference dataset.

Rules:
- Never change expected ordering without a documented, intentional scoring decision.
- Config changes (weights, era params) must not silently break these assertions.
- Numeric snapshots use rel=0.02 tolerance to absorb float precision differences.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.normalize.tennis_normalizer import TennisNormalizer
from src.scoring.scorer import AthleteScorer, load_scoring_config


# ── Ordering contracts ────────────────────────────────────────────────────────

class TestBig3Ordering:
    """
    Assert directional facts about the Big 3 that should hold regardless
    of minor config tuning. These encode historical ground truth.
    """

    def test_djokovic_has_highest_grand_slam_normalized(
        self, tennis_config_path, big3_raw_df
    ):
        result = TennisNormalizer(tennis_config_path).normalize(big3_raw_df)
        gs = result.set_index("name")["grand_slam_normalized"]
        assert gs["Novak Djokovic"] > gs["Rafael Nadal"]
        assert gs["Novak Djokovic"] > gs["Roger Federer"]

    def test_nadal_is_second_in_grand_slams(self, tennis_config_path, big3_raw_df):
        result = TennisNormalizer(tennis_config_path).normalize(big3_raw_df)
        gs = result.set_index("name")["grand_slam_normalized"]
        assert gs["Rafael Nadal"] > gs["Roger Federer"]

    def test_djokovic_has_highest_weeks_at_no1(self, tennis_config_path, big3_raw_df):
        result = TennisNormalizer(tennis_config_path).normalize(big3_raw_df)
        weeks = result.set_index("name")["weeks_no1_normalized"]
        assert weeks["Novak Djokovic"] > weeks["Roger Federer"]
        assert weeks["Novak Djokovic"] > weeks["Rafael Nadal"]

    def test_djokovic_leads_surface_versatility(self, tennis_config_path, big3_raw_df):
        # Djokovic: hard=0.86, clay=0.83, grass=0.84 → lowest std → highest versatility
        result = TennisNormalizer(tennis_config_path).normalize(big3_raw_df)
        sv = result.set_index("name")["surface_versatility_normalized"]
        assert sv["Novak Djokovic"] > sv["Roger Federer"]
        assert sv["Novak Djokovic"] > sv["Rafael Nadal"]

    def test_nadal_has_lowest_surface_versatility(self, tennis_config_path, big3_raw_df):
        # Nadal: clay=0.918 vs hard=0.800, grass=0.760 → highest std → lowest versatility
        result = TennisNormalizer(tennis_config_path).normalize(big3_raw_df)
        sv = result.set_index("name")["surface_versatility_normalized"]
        assert sv["Rafael Nadal"] < sv["Roger Federer"]
        assert sv["Rafael Nadal"] < sv["Novak Djokovic"]

    def test_djokovic_leads_overall_composite_score(
        self, tennis_config_path, big3_raw_df
    ):
        normalizer = TennisNormalizer(tennis_config_path)
        normalized = normalizer.normalize(big3_raw_df)
        scorer = AthleteScorer(load_scoring_config(tennis_config_path))
        scores = scorer.score(normalized)
        scores.index = big3_raw_df["name"].values
        assert scores["Novak Djokovic"] > scores["Roger Federer"]
        assert scores["Novak Djokovic"] > scores["Rafael Nadal"]


# ── Score range contracts ─────────────────────────────────────────────────────

class TestBig3ScoreRanges:
    def test_all_normalized_columns_in_zero_to_100(
        self, tennis_config_path, big3_raw_df
    ):
        result = TennisNormalizer(tennis_config_path).normalize(big3_raw_df)
        normalized_cols = [c for c in result.columns if c.endswith("_normalized")]
        for col in normalized_cols:
            out_of_range = result[col].loc[~result[col].between(0.0, 100.0)]
            assert out_of_range.empty, f"{col} has out-of-range values: {out_of_range.tolist()}"

    def test_composite_scores_in_zero_to_100(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        normalized = normalizer.normalize(big3_raw_df)
        scorer = AthleteScorer(load_scoring_config(tennis_config_path))
        scores = scorer.score(normalized)
        assert scores.between(0.0, 100.0).all(), f"Scores out of range: {scores.tolist()}"

    def test_leader_normalized_grand_slam_is_100(self, tennis_config_path, big3_raw_df):
        # Minmax guarantees the maximum value normalizes to 100.
        result = TennisNormalizer(tennis_config_path).normalize(big3_raw_df)
        assert result["grand_slam_normalized"].max() == pytest.approx(100.0)

    def test_trailing_normalized_grand_slam_is_0(self, tennis_config_path, big3_raw_df):
        result = TennisNormalizer(tennis_config_path).normalize(big3_raw_df)
        assert result["grand_slam_normalized"].min() == pytest.approx(0.0)


# ── Determinism contracts ─────────────────────────────────────────────────────

class TestBig3Determinism:
    def test_same_input_produces_identical_output(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        result_a = normalizer.normalize(big3_raw_df.copy())
        result_b = normalizer.normalize(big3_raw_df.copy())
        normalized_cols = [c for c in result_a.columns if c.endswith("_normalized")]
        pd.testing.assert_frame_equal(result_a[normalized_cols], result_b[normalized_cols])

    def test_row_order_does_not_affect_individual_scores(
        self, tennis_config_path, big3_raw_df
    ):
        normalizer = TennisNormalizer(tennis_config_path)
        result_original = normalizer.normalize(big3_raw_df.copy())
        shuffled = big3_raw_df.sample(frac=1, random_state=42).reset_index(drop=True)
        result_shuffled = normalizer.normalize(shuffled)

        for athlete_id in ["fed_001", "nad_001", "djk_001"]:
            orig = result_original.loc[
                result_original["athlete_id"] == athlete_id, "grand_slam_normalized"
            ].iloc[0]
            shuf = result_shuffled.loc[
                result_shuffled["athlete_id"] == athlete_id, "grand_slam_normalized"
            ].iloc[0]
            assert orig == pytest.approx(shuf, rel=1e-6), (
                f"{athlete_id}: grand_slam_normalized differs after shuffle"
            )

    def test_era_adjust_false_is_reproducible(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        r1 = normalizer.normalize(big3_raw_df.copy(), era_adjust=False)
        r2 = normalizer.normalize(big3_raw_df.copy(), era_adjust=False)
        normalized_cols = [c for c in r1.columns if c.endswith("_normalized")]
        pd.testing.assert_frame_equal(r1[normalized_cols], r2[normalized_cols])


# ── Config contract ───────────────────────────────────────────────────────────

class TestConfigContract:
    def test_output_columns_match_config_weight_keys(
        self, tennis_config_path, big3_raw_df
    ):
        import yaml
        with open(tennis_config_path) as f:
            config = yaml.safe_load(f)
        expected_keys = set(config["weights"].keys())

        result = TennisNormalizer(tennis_config_path).normalize(big3_raw_df)
        actual_normalized = {c for c in result.columns if c.endswith("_normalized")}
        assert expected_keys == actual_normalized, (
            f"Config weight keys and output columns diverged.\n"
            f"Config only: {expected_keys - actual_normalized}\n"
            f"Output only: {actual_normalized - expected_keys}"
        )

    def test_scorer_consumes_normalizer_output_without_error(
        self, tennis_config_path, big3_raw_df
    ):
        normalizer = TennisNormalizer(tennis_config_path)
        normalized = normalizer.normalize(big3_raw_df)
        scorer = AthleteScorer(load_scoring_config(tennis_config_path))
        scores = scorer.score(normalized)
        assert len(scores) == 3
