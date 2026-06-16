"""Unit tests for TennisScorer.

All tests use synthetic DataFrames — no file I/O except for integration tests
that load the real configs/scoring_tennis.yaml via the tmp_path fixture.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.scoring.tennis_scorer import TennisScorer

# ── Helpers ───────────────────────────────────────────────────────────────────

_ALL_COLS = [
    "grand_slam_normalized",
    "weeks_no1_normalized",
    "masters_titles_normalized",
    "career_win_rate_normalized",
    "yearend_no1_normalized",
    "finals_win_rate_normalized",
    "h2h_top10_normalized",
    "surface_versatility_normalized",
    "longevity_normalized",
]

_EQUAL_WEIGHTS = {c: round(1.0 / len(_ALL_COLS), 10) for c in _ALL_COLS}

_LAYERS = {
    "achievement": ["grand_slam_normalized", "weeks_no1_normalized",
                    "masters_titles_normalized", "career_win_rate_normalized"],
    "quality":     ["yearend_no1_normalized", "finals_win_rate_normalized",
                    "h2h_top10_normalized"],
    "context":     ["surface_versatility_normalized", "longevity_normalized"],
}


def _minimal_yaml(
    weights: dict[str, float],
    layers: dict | None = None,
    penalty: float = 0.0,
) -> str:
    cfg: dict = {
        "weights": weights,
        "strategy": "weighted_average",
        "layers": {
            name: {"metrics": metrics}
            for name, metrics in (layers or _LAYERS).items()
        },
        "penalties": {"tournaments_played_pct": penalty},
        "validation": {"score_min": 0.0, "score_max": 100.0, "weight_sum_tolerance": 1e-6},
    }
    return yaml.dump(cfg)


def _scorer(
    tmp_path: Path,
    weights: dict[str, float] | None = None,
    penalty: float = 0.0,
) -> TennisScorer:
    w = weights if weights is not None else _EQUAL_WEIGHTS
    cfg = tmp_path / "scoring_tennis.yaml"
    cfg.write_text(_minimal_yaml(w, penalty=penalty))
    return TennisScorer(cfg)


def _row(**overrides) -> dict:
    base = {c: 50.0 for c in _ALL_COLS}
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── TestConfigLoading ─────────────────────────────────────────────────────────

class TestConfigLoading:
    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            TennisScorer("/nonexistent/path/scoring_tennis.yaml")

    def test_weights_not_summing_to_1_raises(self, tmp_path):
        bad_w = {c: 0.05 for c in _ALL_COLS}  # 9 × 0.05 = 0.45 ≠ 1.0
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(_minimal_yaml(bad_w))
        with pytest.raises(ValueError, match="sum to 1.0"):
            TennisScorer(cfg)

    def test_valid_config_loads(self, tmp_path):
        scorer = _scorer(tmp_path)
        assert isinstance(scorer, TennisScorer)

    def test_layers_loaded_from_config(self, tmp_path):
        scorer = _scorer(tmp_path)
        assert set(scorer._layers.keys()) == {"achievement", "quality", "context"}

    def test_real_config_loads_and_weights_sum_to_1(self):
        config_path = Path(__file__).parents[2] / "configs" / "scoring_tennis.yaml"
        if not config_path.exists():
            pytest.skip("configs/scoring_tennis.yaml not present")
        scorer = TennisScorer(config_path)
        total = sum(scorer._weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_real_config_has_three_layers(self):
        config_path = Path(__file__).parents[2] / "configs" / "scoring_tennis.yaml"
        if not config_path.exists():
            pytest.skip("configs/scoring_tennis.yaml not present")
        scorer = TennisScorer(config_path)
        assert set(scorer._layers.keys()) == {"achievement", "quality", "context"}


# ── TestValidation ────────────────────────────────────────────────────────────

class TestValidation:
    def test_empty_df_raises_on_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            scorer.score(pd.DataFrame())

    def test_empty_df_raises_on_layer_scores(self, tmp_path):
        scorer = _scorer(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            scorer.layer_scores(pd.DataFrame())

    def test_missing_column_raises(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row())
        df = df.drop(columns=["longevity_normalized"])
        with pytest.raises(ValueError, match="Missing required columns"):
            scorer.score(df)

    def test_breakdown_validates_too(self, tmp_path):
        scorer = _scorer(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            scorer.score_breakdown(pd.DataFrame())


# ── TestScore ─────────────────────────────────────────────────────────────────

class TestScore:
    def test_all_zeros_gives_zero(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 0.0 for c in _ALL_COLS}))
        assert scorer.score(df).iloc[0] == pytest.approx(0.0)

    def test_all_100_gives_100(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 100.0 for c in _ALL_COLS}))
        assert scorer.score(df).iloc[0] == pytest.approx(100.0)

    def test_all_50_gives_50(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 50.0 for c in _ALL_COLS}))
        assert scorer.score(df).iloc[0] == pytest.approx(50.0)

    def test_higher_metrics_yield_higher_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(
            _row(**{c: 30.0 for c in _ALL_COLS}),
            _row(**{c: 70.0 for c in _ALL_COLS}),
        )
        scores = scorer.score(df)
        assert scores.iloc[1] > scores.iloc[0]

    def test_score_returns_named_series(self, tmp_path):
        scorer = _scorer(tmp_path)
        assert scorer.score(_df(_row())).name == "goat_score"

    def test_score_in_0_to_100_range(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(
            _row(**{c: 0.0 for c in _ALL_COLS}),
            _row(**{c: 50.0 for c in _ALL_COLS}),
            _row(**{c: 100.0 for c in _ALL_COLS}),
        )
        scores = scorer.score(df)
        assert scores.min() >= 0.0
        assert scores.max() <= 100.0

    def test_weighted_score_reflects_weights(self, tmp_path):
        """Player A scores higher only in the highest-weight column → should rank first."""
        heavy_col = "grand_slam_normalized"
        light_col = "longevity_normalized"
        remaining = [c for c in _ALL_COLS if c not in (heavy_col, light_col)]
        w = {c: (1.0 - 0.60) / (len(_ALL_COLS) - 1) for c in _ALL_COLS}
        w[heavy_col] = 0.60
        scorer = _scorer(tmp_path, weights=w)
        df = _df(
            _row(**{heavy_col: 100.0, light_col: 0.0, **{c: 50.0 for c in remaining}}),
            _row(**{heavy_col: 0.0,   light_col: 100.0, **{c: 50.0 for c in remaining}}),
        )
        scores = scorer.score(df)
        assert scores.iloc[0] > scores.iloc[1]


# ── TestLayerScores ───────────────────────────────────────────────────────────

class TestLayerScores:
    def test_returns_dataframe(self, tmp_path):
        scorer = _scorer(tmp_path)
        assert isinstance(scorer.layer_scores(_df(_row())), pd.DataFrame)

    def test_layer_columns_present(self, tmp_path):
        scorer = _scorer(tmp_path)
        result = scorer.layer_scores(_df(_row()))
        for layer in ("achievement", "quality", "context"):
            assert f"{layer}_score" in result.columns

    def test_all_50_gives_50_per_layer(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 50.0 for c in _ALL_COLS}))
        result = scorer.layer_scores(df)
        for col in result.columns:
            assert result[col].iloc[0] == pytest.approx(50.0), f"{col} should be 50.0"

    def test_all_0_gives_0_per_layer(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 0.0 for c in _ALL_COLS}))
        result = scorer.layer_scores(df)
        for col in result.columns:
            assert result[col].iloc[0] == pytest.approx(0.0)

    def test_all_100_gives_100_per_layer(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 100.0 for c in _ALL_COLS}))
        result = scorer.layer_scores(df)
        for col in result.columns:
            assert result[col].iloc[0] == pytest.approx(100.0)

    def test_layer_scores_in_0_to_100(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(
            _row(**{c: 0.0 for c in _ALL_COLS}),
            _row(**{c: 100.0 for c in _ALL_COLS}),
        )
        result = scorer.layer_scores(df)
        for col in result.columns:
            assert result[col].dropna().min() >= 0.0
            assert result[col].dropna().max() <= 100.0

    def test_layer_uses_equal_weights(self, tmp_path):
        """Achievement layer has 4 metrics — equal-weight average of 100+0+0+0 = 25.0."""
        scorer = _scorer(tmp_path)
        achievement_cols = _LAYERS["achievement"]
        overrides = {achievement_cols[0]: 100.0, **{c: 0.0 for c in achievement_cols[1:]}}
        df = _df(_row(**overrides))
        result = scorer.layer_scores(df)
        assert result["achievement_score"].iloc[0] == pytest.approx(25.0)



# ── TestScoreBreakdown ────────────────────────────────────────────────────────

class TestScoreBreakdown:
    def test_breakdown_has_contribution_and_total_columns(self, tmp_path):
        scorer = _scorer(tmp_path)
        bd = scorer.score_breakdown(_df(_row()))
        assert "goat_score" in bd.columns
        for col in _ALL_COLS:
            assert f"{col}_contribution" in bd.columns

    def test_breakdown_total_matches_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(), _row(**{c: 80.0 for c in _ALL_COLS}))
        bd = scorer.score_breakdown(df)
        direct = scorer.score(df)
        pd.testing.assert_series_equal(
            bd["goat_score"].reset_index(drop=True),
            direct.reset_index(drop=True),
            check_names=False,
        )

    def test_contributions_sum_to_goat_score(self, tmp_path):
        """For complete data (no NaN), contribution columns must sum to goat_score."""
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 60.0 for c in _ALL_COLS}))
        bd = scorer.score_breakdown(df)
        contrib_cols = [f"{c}_contribution" for c in _ALL_COLS]
        contrib_sum = bd[contrib_cols].sum(axis=1).iloc[0]
        assert contrib_sum == pytest.approx(bd["goat_score"].iloc[0], abs=1e-9)


# ── TestPenalty ───────────────────────────────────────────────────────────────

class TestPenalty:
    def test_penalty_skipped_when_column_absent(self, tmp_path):
        scorer = _scorer(tmp_path, penalty=0.05)
        df = _df(_row(**{c: 50.0 for c in _ALL_COLS}))
        # No tournaments_played_pct column → penalty not applied
        assert scorer.score(df).iloc[0] == pytest.approx(50.0)

    def test_penalty_skipped_when_zero(self, tmp_path):
        scorer = _scorer(tmp_path, penalty=0.0)
        df = _df(_row(**{c: 50.0 for c in _ALL_COLS}))
        df["tournaments_played_pct"] = 0.0  # worst participation — but penalty disabled
        assert scorer.score(df).iloc[0] == pytest.approx(50.0)

    def test_penalty_applied_for_low_participation(self, tmp_path):
        scorer = _scorer(tmp_path, penalty=0.05)
        df = _df(_row(**{c: 50.0 for c in _ALL_COLS}))
        df["tournaments_played_pct"] = 0.0  # 0% participation → max deduction = 5.0
        assert scorer.score(df).iloc[0] == pytest.approx(45.0)

    def test_full_participation_no_deduction(self, tmp_path):
        scorer = _scorer(tmp_path, penalty=0.05)
        df = _df(_row(**{c: 50.0 for c in _ALL_COLS}))
        df["tournaments_played_pct"] = 1.0  # 100% → no deduction
        assert scorer.score(df).iloc[0] == pytest.approx(50.0)

    def test_penalty_does_not_push_below_zero(self, tmp_path):
        scorer = _scorer(tmp_path, penalty=0.05)
        df = _df(_row(**{c: 0.0 for c in _ALL_COLS}))
        df["tournaments_played_pct"] = 0.0  # score=0.0, deduction=5.0 → clamped to 0.0
        assert scorer.score(df).iloc[0] == pytest.approx(0.0)
