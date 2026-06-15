"""Unit tests for NBAScorer.

All tests use synthetic DataFrames — no file I/O except for integration tests
that load the real configs/scoring_nba.yaml via the tmp_path fixture.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.scoring.nba_scorer import NBAScorer

# ── Helpers ───────────────────────────────────────────────────────────────────

_ALL_COLS = [
    "peak_bpm_normalized", "peak_ws48_normalized", "peak_vorp_normalized",
    "career_vorp_normalized", "career_ws_normalized", "career_bpm_normalized",
    "career_ws48_normalized", "career_ts_normalized",
    "championships_normalized", "mvp_normalized", "finals_mvp_normalized",
    "all_nba_1st_normalized", "all_nba_2nd_normalized", "all_star_normalized",
    "dpoy_normalized", "def_1st_normalized",
    "longevity_normalized",
]

_EQUAL_WEIGHTS = {c: 1.0 / len(_ALL_COLS) for c in _ALL_COLS}


def _minimal_yaml(weights: dict[str, float]) -> str:
    return yaml.dump({"weights": weights, "strategy": "nan_aware_weighted_average"})


def _scorer(tmp_path: Path, weights: dict[str, float] | None = None) -> NBAScorer:
    w = weights if weights is not None else _EQUAL_WEIGHTS
    cfg = tmp_path / "scoring_nba.yaml"
    cfg.write_text(_minimal_yaml(w))
    return NBAScorer(cfg)


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
            NBAScorer("/nonexistent/path/scoring_nba.yaml")

    def test_weights_not_summing_to_1_raises(self, tmp_path):
        bad_w = {c: 0.05 for c in _ALL_COLS}  # 17 × 0.05 = 0.85 ≠ 1.0
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(_minimal_yaml(bad_w))
        with pytest.raises(ValueError, match="sum to 1.0"):
            NBAScorer(cfg)

    def test_valid_config_loads(self, tmp_path):
        scorer = _scorer(tmp_path)
        assert isinstance(scorer, NBAScorer)

    def test_real_config_loads_and_weights_sum_to_1(self):
        config_path = Path(__file__).parents[2] / "configs" / "scoring_nba.yaml"
        if not config_path.exists():
            pytest.skip("configs/scoring_nba.yaml not present")
        scorer = NBAScorer(config_path)
        total = sum(scorer._weights.values())
        assert abs(total - 1.0) < 1e-6


# ── TestValidation ─────────────────────────────────────────────────────────────

class TestValidation:
    def test_empty_df_raises(self, tmp_path):
        scorer = _scorer(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            scorer.score(pd.DataFrame())

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
        scores = scorer.score(df)
        assert scores.iloc[0] == pytest.approx(0.0)

    def test_all_100_gives_100(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 100.0 for c in _ALL_COLS}))
        scores = scorer.score(df)
        assert scores.iloc[0] == pytest.approx(100.0)

    def test_all_50_gives_50(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 50.0 for c in _ALL_COLS}))
        scores = scorer.score(df)
        assert scores.iloc[0] == pytest.approx(50.0)

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
        df = _df(_row())
        result = scorer.score(df)
        assert result.name == "goat_score"

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
        """Player B exceeds only in the highest-weight column → should rank higher."""
        heavy_col = "peak_bpm_normalized"
        light_col = "longevity_normalized"
        # Equal weights except heavy_col gets much more weight
        w = {c: (1.0 - 0.60) / (len(_ALL_COLS) - 1) for c in _ALL_COLS}
        w[heavy_col] = 0.60
        scorer = _scorer(tmp_path, weights=w)
        df = _df(
            _row(**{heavy_col: 100.0, light_col: 0.0}),  # strong in heavy metric
            _row(**{heavy_col: 0.0,   light_col: 100.0}), # strong in light metric
        )
        scores = scorer.score(df)
        assert scores.iloc[0] > scores.iloc[1]


# ── TestNanAwareness ──────────────────────────────────────────────────────────

class TestNanAwareness:
    def test_all_nan_metrics_give_nan_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: float("nan") for c in _ALL_COLS}))
        scores = scorer.score(df)
        assert math.isnan(scores.iloc[0])

    def test_single_nan_column_does_not_give_nan(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(peak_bpm_normalized=float("nan")))
        scores = scorer.score(df)
        assert not math.isnan(scores.iloc[0])

    def test_nan_columns_excluded_from_average(self, tmp_path):
        """Player with NaN in some cols gets same score as player without NaN
        when the non-NaN metrics have equal value (rescaling restores scale)."""
        n = len(_ALL_COLS)
        # All metrics = 60.0 for player A
        row_a = _row(**{c: 60.0 for c in _ALL_COLS})
        # Player B: same value for available cols, NaN for first half
        nan_cols = _ALL_COLS[: n // 2]
        valid_cols = _ALL_COLS[n // 2 :]
        row_b = _row(**{c: float("nan") for c in nan_cols}, **{c: 60.0 for c in valid_cols})
        scorer = _scorer(tmp_path)
        df = _df(row_a, row_b)
        scores = scorer.score(df)
        assert scores.iloc[0] == pytest.approx(scores.iloc[1], abs=1e-9)

    def test_pre_advanced_era_nan_pattern(self, tmp_path):
        """Simulate pre-advanced era player: BPM/VORP NaN, rest available."""
        nan_cols = ["peak_bpm_normalized", "peak_vorp_normalized",
                    "career_vorp_normalized", "career_bpm_normalized"]
        row = _row(**{c: float("nan") for c in nan_cols})
        scorer = _scorer(tmp_path)
        df = _df(row)
        score = scorer.score(df).iloc[0]
        assert not math.isnan(score)
        assert 0.0 <= score <= 100.0

    def test_nan_score_stays_in_range(self, tmp_path):
        """Rescaled NaN-aware score stays in [0, 100]."""
        nan_cols = ["peak_bpm_normalized", "peak_vorp_normalized"]
        row_low  = _row(**{c: float("nan") for c in nan_cols},
                        **{c: 0.0 for c in _ALL_COLS if c not in nan_cols})
        row_high = _row(**{c: float("nan") for c in nan_cols},
                        **{c: 100.0 for c in _ALL_COLS if c not in nan_cols})
        scorer = _scorer(tmp_path)
        df = _df(row_low, row_high)
        scores = scorer.score(df)
        assert scores.iloc[0] == pytest.approx(0.0)
        assert scores.iloc[1] == pytest.approx(100.0)


# ── TestScoreBreakdown ────────────────────────────────────────────────────────

class TestScoreBreakdown:
    def test_breakdown_has_contribution_and_total_columns(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row())
        bd = scorer.score_breakdown(df)
        assert "goat_score" in bd.columns
        for col in _ALL_COLS:
            assert f"{col}_contribution" in bd.columns

    def test_breakdown_total_matches_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(), _row(**{c: 80.0 for c in _ALL_COLS}))
        bd = scorer.score_breakdown(df)
        direct_scores = scorer.score(df)
        pd.testing.assert_series_equal(
            bd["goat_score"].reset_index(drop=True),
            direct_scores.reset_index(drop=True),
            check_names=False,
        )

    def test_nan_contribution_for_nan_metric(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(peak_bpm_normalized=float("nan")))
        bd = scorer.score_breakdown(df)
        assert math.isnan(bd["peak_bpm_normalized_contribution"].iloc[0])


# ── TestLayerScores ───────────────────────────────────────────────────────────

class TestLayerScores:
    def test_layer_scores_returns_dataframe(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row())
        result = scorer.layer_scores(df)
        assert isinstance(result, pd.DataFrame)

    def test_layer_score_columns_present(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row())
        result = scorer.layer_scores(df)
        for layer in ["peak", "career", "efficiency", "achievement", "longevity"]:
            assert f"{layer}_score" in result.columns

    def test_all_50_gives_50_per_layer(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 50.0 for c in _ALL_COLS}))
        result = scorer.layer_scores(df)
        for col in result.columns:
            assert result[col].iloc[0] == pytest.approx(50.0), f"{col} should be 50.0"

    def test_nan_bpm_preserved_in_peak_score(self, tmp_path):
        """peak_bpm NaN should propagate to peak_score only if ALL peak cols are NaN."""
        scorer = _scorer(tmp_path)
        # Only BPM is NaN — ws48 and vorp_season available → peak_score should exist
        df = _df(_row(peak_bpm_normalized=float("nan")))
        result = scorer.layer_scores(df)
        assert not math.isnan(result["peak_score"].iloc[0])

    def test_all_peak_nan_gives_nan_peak_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row(
            peak_bpm_normalized=float("nan"),
            peak_ws48_normalized=float("nan"),
            peak_vorp_normalized=float("nan"),
        ))
        result = scorer.layer_scores(df)
        assert math.isnan(result["peak_score"].iloc[0])

    def test_layer_scores_are_in_0_to_100(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(
            _row(**{c: 0.0 for c in _ALL_COLS}),
            _row(**{c: 100.0 for c in _ALL_COLS}),
        )
        result = scorer.layer_scores(df)
        for col in result.columns:
            valid = result[col].dropna()
            assert valid.min() >= 0.0
            assert valid.max() <= 100.0


# ── TestWeightedAvgNan ────────────────────────────────────────────────────────

class TestWeightedAvgNan:
    """Direct unit tests for the core static helper."""

    def test_basic_weighted_avg(self):
        values  = np.array([[0.0, 100.0]])
        weights = np.array([0.3, 0.7])
        result  = NBAScorer._weighted_avg_nan(values, weights)
        assert result[0] == pytest.approx(70.0)

    def test_single_nan_excluded(self):
        values  = np.array([[np.nan, 80.0]])
        weights = np.array([0.5, 0.5])
        result  = NBAScorer._weighted_avg_nan(values, weights)
        assert result[0] == pytest.approx(80.0)

    def test_all_nan_returns_nan(self):
        values  = np.array([[np.nan, np.nan]])
        weights = np.array([0.5, 0.5])
        result  = NBAScorer._weighted_avg_nan(values, weights)
        assert math.isnan(result[0])

    def test_multiple_rows(self):
        values  = np.array([[0.0, 0.0], [100.0, 100.0]])
        weights = np.array([0.4, 0.6])
        result  = NBAScorer._weighted_avg_nan(values, weights)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(100.0)

    def test_nan_rescales_correctly(self):
        """With w=[0.4, 0.6], only col-1 available (80.0) → 80.0 rescaled by 0.6/0.6 = 80.0."""
        values  = np.array([[np.nan, 80.0]])
        weights = np.array([0.4, 0.6])
        result  = NBAScorer._weighted_avg_nan(values, weights)
        assert result[0] == pytest.approx(80.0)
