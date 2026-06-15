"""
NBA GOAT Regression Baseline

Tests the full NBARunner pipeline (Pipeline → Normalizer → Scorer) on real data
and asserts ordering, range, NaN-handling, and determinism contracts.

Rules:
- Never change expected ordering without a documented, intentional weight decision.
- Snapshot tolerances are rel=0.02 (2%) to absorb float precision differences.
- These tests skip automatically when nba_all_seasons_raw.csv is absent.
- Do NOT mock the runner — these are integration/regression tests, not unit tests.

Key findings encoded as ordering facts (from EDA + first pipeline run):
  - LeBron #1 overall (career volume dominates with current weights)
  - Jordan #2 (highest peak, but shorter career than LeBron)
  - Kareem #3 (all-time peak WS/48 leader → peak_ws48_normalized = 100.0)
  - Jokić has all-time highest BPM → peak_bpm_normalized = 100.0
  - Russell & Chamberlain have NaN peak_bpm (pre-advanced era) but valid GOAT scores
"""
from __future__ import annotations

import math

import pytest


# ── Ordering contracts ────────────────────────────────────────────────────────

class TestGoatOrdering:
    """Directional ordering facts that must hold regardless of minor weight tuning."""

    def test_lebron_ranks_above_jordan_overall(self, nba_result):
        by_id = nba_result.set_index("player_id")
        assert by_id.loc["jamesle01", "goat_score"] > by_id.loc["jordami01", "goat_score"]

    def test_jordan_leads_lebron_in_peak_bpm(self, nba_result):
        # EDA Finding 3: Jordan 3-window BPM = 12.033, LeBron = 11.967
        by_id = nba_result.set_index("player_id")
        assert by_id.loc["jordami01", "peak_bpm_normalized"] > by_id.loc["jamesle01", "peak_bpm_normalized"]

    def test_kareem_has_all_time_peak_ws48(self, nba_result):
        # Kareem's 3-consecutive WS/48 is the highest ever recorded
        assert nba_result.set_index("player_id").loc["abdulka01", "peak_ws48_normalized"] == pytest.approx(100.0, abs=0.01)

    def test_jokic_has_all_time_peak_bpm(self, nba_result):
        # Jokić's 3-window BPM is the highest in dataset history
        assert nba_result.set_index("player_id").loc["jokicni01", "peak_bpm_normalized"] == pytest.approx(100.0, abs=0.01)

    def test_jordan_leads_lebron_in_achievement_score(self, nba_result):
        # Jordan: 6 rings + 5 MVPs + 6 Finals MVPs vs LeBron: 4 rings + 4 MVPs + 4 Finals MVPs
        by_id = nba_result.set_index("player_id")
        assert by_id.loc["jordami01", "achievement_score"] > by_id.loc["jamesle01", "achievement_score"]

    def test_lebron_leads_jordan_in_career_score(self, nba_result):
        # LeBron's career volume (seasons × quality) significantly exceeds Jordan's
        by_id = nba_result.set_index("player_id")
        assert by_id.loc["jamesle01", "career_score"] > by_id.loc["jordami01", "career_score"]

    def test_kareem_ranks_in_top_3(self, nba_result):
        kareem_rank = nba_result.set_index("player_id").loc["abdulka01", "rank"]
        assert kareem_rank <= 3

    def test_top_3_contains_jordan_lebron_kareem(self, nba_result):
        top3_ids = set(nba_result.head(3)["player_id"])
        assert {"jamesle01", "jordami01", "abdulka01"} == top3_ids


# ── Pre-advanced era contracts ────────────────────────────────────────────────

class TestPreAdvancedEra:
    """Pre-advanced era players (1947–73) have NaN BPM/VORP but must receive valid GOAT scores."""

    def test_russell_peak_bpm_is_nan(self, nba_result):
        row = nba_result.set_index("player_id").loc["russebi01"]
        assert math.isnan(row["peak_bpm_normalized"])

    def test_chamberlain_peak_bpm_is_nan(self, nba_result):
        row = nba_result.set_index("player_id").loc["chambwi01"]
        assert math.isnan(row["peak_bpm_normalized"])

    def test_russell_goat_score_is_valid(self, nba_result):
        score = nba_result.set_index("player_id").loc["russebi01", "goat_score"]
        assert not math.isnan(score)
        assert 0.0 <= score <= 100.0

    def test_chamberlain_goat_score_is_valid(self, nba_result):
        score = nba_result.set_index("player_id").loc["chambwi01", "goat_score"]
        assert not math.isnan(score)
        assert 0.0 <= score <= 100.0

    def test_pre_advanced_era_label(self, nba_result):
        by_id = nba_result.set_index("player_id")
        assert by_id.loc["russebi01", "era"] == "pre_advanced"
        assert by_id.loc["chambwi01", "era"] == "pre_advanced"

    def test_russell_peak_score_uses_ws48_only(self, nba_result):
        # With BPM and VORP NaN, peak_score = NaN-rescaled average of ws48 only
        row = nba_result.set_index("player_id").loc["russebi01"]
        assert row["peak_score"] == pytest.approx(row["peak_ws48_normalized"], abs=0.01)


# ── Score range contracts ─────────────────────────────────────────────────────

class TestScoreRanges:
    def test_all_goat_scores_in_0_to_100(self, nba_result):
        valid = nba_result["goat_score"].dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0

    def test_all_normalized_cols_in_0_to_100(self, nba_result):
        norm_cols = [c for c in nba_result.columns if c.endswith("_normalized")]
        for col in norm_cols:
            valid = nba_result[col].dropna()
            if valid.empty:
                continue
            assert valid.min() >= -1e-9, f"{col}: min {valid.min():.4f} < 0"
            assert valid.max() <= 100.0 + 1e-9, f"{col}: max {valid.max():.4f} > 100"

    def test_top_player_goat_score_above_80(self, nba_result):
        # With current weights the best player must exceed 80 — sanity check on scale
        assert nba_result["goat_score"].max() > 80.0

    def test_goat_score_spread_is_meaningful(self, nba_result):
        # Top player score vs median must differ by at least 20 points
        top = nba_result["goat_score"].max()
        median = nba_result["goat_score"].median()
        assert top - median > 20.0

    def test_rank_column_is_contiguous_from_1(self, nba_result):
        ranks = nba_result["rank"].tolist()
        assert ranks == list(range(1, len(nba_result) + 1))

    def test_result_is_sorted_by_goat_score_descending(self, nba_result):
        scores = nba_result["goat_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ── Snapshot contracts ────────────────────────────────────────────────────────

class TestSnapshotScores:
    """Numeric score snapshots for the GOAT tier — 2% relative tolerance.

    If a test fails here after an intentional config change:
      1. Run NBARunner and print scores to verify the new values are correct.
      2. Update tests/fixtures/nba_goat_snapshot_v1.json with the new scores.
      3. Document the weight change in the config file.
    Never adjust the fixture to make tests pass without understanding why they changed.
    """

    def test_lebron_goat_score_matches_snapshot(self, nba_result, nba_snapshot):
        expected = nba_snapshot["players"]["jamesle01"]["goat_score"]
        actual = nba_result.set_index("player_id").loc["jamesle01", "goat_score"]
        assert actual == pytest.approx(expected, rel=0.02)

    def test_jordan_goat_score_matches_snapshot(self, nba_result, nba_snapshot):
        expected = nba_snapshot["players"]["jordami01"]["goat_score"]
        actual = nba_result.set_index("player_id").loc["jordami01", "goat_score"]
        assert actual == pytest.approx(expected, rel=0.02)

    def test_kareem_goat_score_matches_snapshot(self, nba_result, nba_snapshot):
        expected = nba_snapshot["players"]["abdulka01"]["goat_score"]
        actual = nba_result.set_index("player_id").loc["abdulka01", "goat_score"]
        assert actual == pytest.approx(expected, rel=0.02)

    def test_russell_goat_score_matches_snapshot(self, nba_result, nba_snapshot):
        expected = nba_snapshot["players"]["russebi01"]["goat_score"]
        actual = nba_result.set_index("player_id").loc["russebi01", "goat_score"]
        assert actual == pytest.approx(expected, rel=0.02)

    def test_chamberlain_goat_score_matches_snapshot(self, nba_result, nba_snapshot):
        expected = nba_snapshot["players"]["chambwi01"]["goat_score"]
        actual = nba_result.set_index("player_id").loc["chambwi01", "goat_score"]
        assert actual == pytest.approx(expected, rel=0.02)

    def test_snapshot_ranks_match(self, nba_result, nba_snapshot):
        by_id = nba_result.set_index("player_id")
        for pid, info in nba_snapshot["players"].items():
            if pid not in by_id.index:
                continue
            actual_rank = int(by_id.loc[pid, "rank"])
            expected_rank = info["rank"]
            assert actual_rank == expected_rank, (
                f"{info['name']}: rank {actual_rank} ≠ expected {expected_rank}"
            )


# ── Determinism contracts ─────────────────────────────────────────────────────

class TestDeterminism:
    def test_runner_is_deterministic(self, nba_data_dir, nba_config_dir):
        import pandas as pd
        from src.pipelines.nba_runner import NBARunner

        runner = NBARunner(nba_data_dir, nba_config_dir)
        result_a = runner.run()
        result_b = runner.run()

        score_cols = ["goat_score", "peak_score", "career_score",
                      "efficiency_score", "achievement_score", "longevity_score"]
        pd.testing.assert_frame_equal(
            result_a[score_cols].reset_index(drop=True),
            result_b[score_cols].reset_index(drop=True),
            check_exact=False,
            rtol=1e-10,
        )

    def test_no_duplicate_player_ids_in_result(self, nba_result):
        assert not nba_result["player_id"].duplicated().any()


# ── Config contract ───────────────────────────────────────────────────────────

class TestConfigContract:
    def test_scorer_weight_keys_match_normalizer_output(self, nba_config_dir):
        """Config weight keys must be a subset of NBANormalizer.output_columns()."""
        import yaml
        from src.normalize.nba_normalizer import NBANormalizer

        with open(nba_config_dir / "scoring_nba.yaml") as f:
            cfg = yaml.safe_load(f)

        weight_keys = set(cfg["weights"].keys())
        normalizer_outputs = set(NBANormalizer.output_columns())
        orphan_keys = weight_keys - normalizer_outputs
        assert not orphan_keys, (
            f"Weight keys in scoring_nba.yaml not produced by NBANormalizer: {orphan_keys}"
        )

    def test_runner_output_contains_all_weight_columns(self, nba_result, nba_config_dir):
        import yaml
        with open(nba_config_dir / "scoring_nba.yaml") as f:
            cfg = yaml.safe_load(f)
        for col in cfg["weights"]:
            assert col in nba_result.columns, f"Missing column: {col}"
