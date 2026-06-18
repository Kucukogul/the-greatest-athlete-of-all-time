"""Swimming GOAT regression baseline — any ordering or score change requires a documented scoring decision."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipelines.swimming_pipeline import SwimmingPipeline
from src.normalize.swimming_normalizer import SwimmingNormalizer
from src.scoring.swimming_scorer import SwimmingScorer

_RAW_PATH = Path(__file__).parents[2] / "data" / "raw"
_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "scoring_swimming.yaml"


def _skip_if_absent():
    if not (_RAW_PATH / "swimming_athletes_raw.csv").exists():
        pytest.skip("swimming_athletes_raw.csv not present")


@pytest.fixture(scope="module")
def full_df_adjusted():
    """Full pipeline + normalizer + scorer output — era_adjust=True. Module-scoped."""
    _skip_if_absent()
    raw = SwimmingPipeline(_RAW_PATH).run()
    norm = SwimmingNormalizer(_CONFIG_PATH).normalize(raw, era_adjust=True)
    scorer = SwimmingScorer(_CONFIG_PATH)
    norm["goat_score"] = scorer.score(norm)
    layers = scorer.layer_scores(norm)
    return norm.join(layers)


@pytest.fixture(scope="module")
def full_df_unadjusted():
    """Full pipeline + normalizer + scorer — era_adjust=False. For era contrast tests."""
    _skip_if_absent()
    raw = SwimmingPipeline(_RAW_PATH).run()
    norm = SwimmingNormalizer(_CONFIG_PATH).normalize(raw, era_adjust=False)
    scorer = SwimmingScorer(_CONFIG_PATH)
    norm["goat_score"] = scorer.score(norm)
    return norm


# ── TestPhelpsLeads ───────────────────────────────────────────────────────────

class TestPhelpsLeads:
    """Phelps dominance is a structural fact — he holds absolute records in every metric."""

    def test_phelps_is_ranked_first_overall(self, full_df_adjusted):
        top = full_df_adjusted.sort_values("goat_score", ascending=False).iloc[0]
        assert top["name"] == "Michael Phelps"

    def test_phelps_score_is_above_90(self, full_df_adjusted):
        phelps = full_df_adjusted[full_df_adjusted["name"] == "Michael Phelps"].iloc[0]
        assert phelps["goat_score"] > 90.0

    def test_phelps_achievement_score_is_100(self, full_df_adjusted):
        phelps = full_df_adjusted[full_df_adjusted["name"] == "Michael Phelps"].iloc[0]
        assert phelps["achievement_score"] == pytest.approx(100.0, abs=1e-6)

    def test_phelps_peak_score_is_100(self, full_df_adjusted):
        phelps = full_df_adjusted[full_df_adjusted["name"] == "Michael Phelps"].iloc[0]
        assert phelps["peak_score"] == pytest.approx(100.0, abs=1e-6)

    def test_phelps_leads_ledecky_by_significant_margin(self, full_df_adjusted):
        scores = full_df_adjusted.set_index("name")["goat_score"]
        assert scores["Michael Phelps"] - scores["Katie Ledecky"] > 30.0


# ── TestOrdering ──────────────────────────────────────────────────────────────

class TestOrdering:
    """Cross-athlete ordering contracts that should hold under any reasonable config."""

    def test_ledecky_ranks_above_lochte(self, full_df_adjusted):
        scores = full_df_adjusted.set_index("name")["goat_score"]
        assert scores["Katie Ledecky"] > scores["Ryan Lochte"]

    def test_ledecky_is_top_female_swimmer(self, full_df_adjusted):
        females = ["Katie Ledecky", "Sarah Sjostrom", "Katinka Hosszu",
                   "Federica Pellegrini", "Janet Evans"]
        scores = full_df_adjusted.set_index("name")["goat_score"]
        ledecky_score = scores["Katie Ledecky"]
        for name in females:
            if name != "Katie Ledecky":
                assert ledecky_score > scores[name], (
                    f"Ledecky should lead {name}"
                )

    def test_pre_modern_athletes_are_all_ranked(self, full_df_adjusted):
        pre_modern_names = ["Mark Spitz", "Roland Matthes", "Dawn Fraser", "Shane Gould"]
        ranked_names = full_df_adjusted.sort_values("goat_score", ascending=False)["name"].tolist()
        for name in pre_modern_names:
            assert name in ranked_names


# ── TestEraAdjustment ─────────────────────────────────────────────────────────

class TestEraAdjustment:
    """Era adjustment must materially lift Pre-Modern athletes."""

    def test_spitz_ranks_higher_with_era_adjustment(
        self, full_df_adjusted, full_df_unadjusted
    ):
        adj_scores = full_df_adjusted.set_index("name")["goat_score"]
        unadj_scores = full_df_unadjusted.set_index("name")["goat_score"]
        assert adj_scores["Mark Spitz"] > unadj_scores["Mark Spitz"]

    def test_matthes_ranks_higher_with_era_adjustment(
        self, full_df_adjusted, full_df_unadjusted
    ):
        adj_scores = full_df_adjusted.set_index("name")["goat_score"]
        unadj_scores = full_df_unadjusted.set_index("name")["goat_score"]
        assert adj_scores["Roland Matthes"] > unadj_scores["Roland Matthes"]

    def test_modern_athletes_unaffected_by_era_adjustment(
        self, full_df_adjusted, full_df_unadjusted
    ):
        adj = full_df_adjusted.set_index("name")["goat_score"]
        unadj = full_df_unadjusted.set_index("name")["goat_score"]
        for name in ["Michael Phelps", "Katie Ledecky", "Ryan Lochte"]:
            assert adj[name] == pytest.approx(unadj[name], rel=1e-6), (
                f"{name} should be unaffected by era adjustment"
            )

    def test_spitz_wc_gold_individual_is_imputed_not_zero(self, full_df_adjusted):
        spitz = full_df_adjusted[full_df_adjusted["name"] == "Mark Spitz"].iloc[0]
        assert spitz["wc_gold_individual"] > 0.0


# ── TestScoreRange ────────────────────────────────────────────────────────────

class TestScoreRange:
    def test_all_goat_scores_in_zero_to_100(self, full_df_adjusted):
        scores = full_df_adjusted["goat_score"]
        assert scores.between(0.0, 100.0).all(), (
            f"Scores out of [0,100]: {scores[~scores.between(0.0, 100.0)].tolist()}"
        )

    def test_all_normalized_columns_in_zero_to_100(self, full_df_adjusted):
        norm_cols = [c for c in full_df_adjusted.columns if c.endswith("_normalized")]
        for col in norm_cols:
            assert full_df_adjusted[col].between(0.0, 100.0).all(), (
                f"{col} has out-of-range values"
            )

    def test_phelps_og_individual_normalized_is_100(self, full_df_adjusted):
        phelps = full_df_adjusted[full_df_adjusted["name"] == "Michael Phelps"].iloc[0]
        assert phelps["olympic_gold_individual_normalized"] == pytest.approx(100.0)

    def test_minimum_og_individual_normalized_is_0(self, full_df_adjusted):
        assert full_df_adjusted["olympic_gold_individual_normalized"].min() == pytest.approx(0.0)


# ── TestDeterminism ───────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_produces_identical_scores(self):
        _skip_if_absent()
        raw = SwimmingPipeline(_RAW_PATH).run()
        scorer = SwimmingScorer(_CONFIG_PATH)
        norm_a = SwimmingNormalizer(_CONFIG_PATH).normalize(raw.copy(), era_adjust=True)
        norm_b = SwimmingNormalizer(_CONFIG_PATH).normalize(raw.copy(), era_adjust=True)
        scores_a = scorer.score(norm_a)
        scores_b = scorer.score(norm_b)
        pd.testing.assert_series_equal(scores_a, scores_b)

    def test_row_shuffle_does_not_change_individual_scores(self):
        _skip_if_absent()
        raw = SwimmingPipeline(_RAW_PATH).run()
        norm = SwimmingNormalizer(_CONFIG_PATH).normalize(raw.copy(), era_adjust=True)
        shuffled_norm = SwimmingNormalizer(_CONFIG_PATH).normalize(
            raw.sample(frac=1, random_state=99).reset_index(drop=True),
            era_adjust=True,
        )
        scorer = SwimmingScorer(_CONFIG_PATH)
        scores_orig = scorer.score(norm).rename("orig")
        scores_shuf = scorer.score(shuffled_norm).rename("shuf")

        orig_indexed = norm[["athlete_id"]].copy()
        orig_indexed["score"] = scores_orig.values
        shuf_indexed = shuffled_norm[["athlete_id"]].copy()
        shuf_indexed["score"] = scores_shuf.values

        merged = orig_indexed.merge(shuf_indexed, on="athlete_id", suffixes=("_orig", "_shuf"))
        diff = (merged["score_orig"] - merged["score_shuf"]).abs()
        assert diff.max() < 1e-9, f"Shuffle changed scores by up to {diff.max()}"


# ── TestConfigContract ────────────────────────────────────────────────────────

class TestConfigContract:
    def test_normalizer_output_matches_scorer_weight_keys(self):
        _skip_if_absent()
        import yaml
        with open(_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        expected_keys = set(config["weights"].keys())

        raw = SwimmingPipeline(_RAW_PATH).run()
        result = SwimmingNormalizer(_CONFIG_PATH).normalize(raw, era_adjust=True)
        actual_normalized = {c for c in result.columns if c.endswith("_normalized")}
        assert expected_keys == actual_normalized, (
            f"Config weight keys and output columns diverged.\n"
            f"Config only: {expected_keys - actual_normalized}\n"
            f"Output only: {actual_normalized - expected_keys}"
        )

    def test_scorer_consumes_normalizer_output_without_error(self):
        _skip_if_absent()
        raw = SwimmingPipeline(_RAW_PATH).run()
        norm = SwimmingNormalizer(_CONFIG_PATH).normalize(raw, era_adjust=True)
        scorer = SwimmingScorer(_CONFIG_PATH)
        scores = scorer.score(norm)
        assert len(scores) == 35
