"""Unit tests for SwimmingScorer — synthetic configs written to tmp_path; real-config integration tests at end of each class."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.scoring.swimming_scorer import SwimmingScorer

# ── Constants & helpers ───────────────────────────────────────────────────────

_ALL_COLS = [
    "olympic_gold_individual_normalized",
    "wc_gold_individual_normalized",
    "olympic_gold_relay_normalized",
    "wc_gold_relay_normalized",
    "world_records_normalized",
    "events_dominated_normalized",
    "career_years_normalized",
]

_WEIGHTS = {
    "olympic_gold_individual_normalized": 0.30,
    "wc_gold_individual_normalized":      0.20,
    "olympic_gold_relay_normalized":      0.10,
    "wc_gold_relay_normalized":           0.05,
    "world_records_normalized":           0.15,
    "events_dominated_normalized":        0.10,
    "career_years_normalized":            0.10,
}

_LAYERS = {
    "achievement": [
        "olympic_gold_individual_normalized",
        "wc_gold_individual_normalized",
        "olympic_gold_relay_normalized",
        "wc_gold_relay_normalized",
    ],
    "peak":        ["world_records_normalized"],
    "versatility": ["events_dominated_normalized"],
    "longevity":   ["career_years_normalized"],
}


def _minimal_yaml(weights: dict[str, float], layers: dict | None = None) -> str:
    cfg: dict = {
        "weights": weights,
        "strategy": "weighted_average",
        "layers": {
            name: {"metrics": metrics}
            for name, metrics in (layers or _LAYERS).items()
        },
        "penalties": {"relay_dependency_penalty": 0.0},
        "validation": {"score_min": 0.0, "score_max": 100.0, "weight_sum_tolerance": 1e-6},
    }
    return yaml.dump(cfg)


def _scorer(tmp_path: Path, weights: dict[str, float] | None = None) -> SwimmingScorer:
    w = weights if weights is not None else _WEIGHTS
    cfg = tmp_path / "scoring_swimming.yaml"
    cfg.write_text(_minimal_yaml(w))
    return SwimmingScorer(cfg)


def _row(**overrides) -> dict:
    base = {c: 50.0 for c in _ALL_COLS}
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── TestConfigLoading ─────────────────────────────────────────────────────────

class TestConfigLoading:
    def test_missing_config_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            SwimmingScorer("/nonexistent/path/scoring_swimming.yaml")

    def test_weights_not_summing_to_1_raises_value_error(self, tmp_path):
        bad_w = {c: 0.05 for c in _ALL_COLS}  # 7 × 0.05 = 0.35 ≠ 1.0
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(_minimal_yaml(bad_w))
        with pytest.raises(ValueError, match="sum to 1.0"):
            SwimmingScorer(cfg)

    def test_valid_config_loads_without_error(self, tmp_path):
        scorer = _scorer(tmp_path)
        assert isinstance(scorer, SwimmingScorer)

    def test_layers_loaded_from_config(self, tmp_path):
        scorer = _scorer(tmp_path)
        assert set(scorer._layers.keys()) == {"achievement", "peak", "versatility", "longevity"}

    def test_real_config_weights_sum_to_1(self, swimming_config_path):
        scorer = SwimmingScorer(swimming_config_path)
        assert abs(sum(scorer._weights.values()) - 1.0) < 1e-6

    def test_real_config_has_four_layers(self, swimming_config_path):
        scorer = SwimmingScorer(swimming_config_path)
        assert set(scorer._layers.keys()) == {"achievement", "peak", "versatility", "longevity"}


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

    def test_missing_column_raises_on_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(_row()).drop(columns=["career_years_normalized"])
        with pytest.raises(ValueError, match="Missing required normalized columns"):
            scorer.score(df)

    def test_empty_df_raises_on_score_breakdown(self, tmp_path):
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

    def test_score_is_named_goat_score(self, tmp_path):
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

    def test_higher_metrics_yield_higher_score(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(
            _row(**{c: 30.0 for c in _ALL_COLS}),
            _row(**{c: 70.0 for c in _ALL_COLS}),
        )
        scores = scorer.score(df)
        assert scores.iloc[1] > scores.iloc[0]

    def test_highest_weight_column_dominates_ranking(self, tmp_path):
        """Athlete A is higher only in olympic_gold_individual (weight=0.30) → wins."""
        high_col = "olympic_gold_individual_normalized"
        low_col = "career_years_normalized"
        rest = [c for c in _ALL_COLS if c not in (high_col, low_col)]
        scorer = _scorer(tmp_path)
        df = _df(
            _row(**{high_col: 100.0, low_col: 0.0, **{c: 50.0 for c in rest}}),
            _row(**{high_col: 0.0,   low_col: 100.0, **{c: 50.0 for c in rest}}),
        )
        scores = scorer.score(df)
        assert scores.iloc[0] > scores.iloc[1]


# ── TestLayerScores ───────────────────────────────────────────────────────────

class TestLayerScores:
    def test_returns_dataframe(self, tmp_path):
        scorer = _scorer(tmp_path)
        assert isinstance(scorer.layer_scores(_df(_row())), pd.DataFrame)

    def test_all_four_layer_columns_present(self, tmp_path):
        scorer = _scorer(tmp_path)
        result = scorer.layer_scores(_df(_row()))
        for layer in ("achievement", "peak", "versatility", "longevity"):
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

    def test_achievement_layer_uses_equal_weights(self, tmp_path):
        """Achievement has 4 metrics — only first at 100, rest at 0 → layer = 25."""
        scorer = _scorer(tmp_path)
        achievement_cols = _LAYERS["achievement"]
        overrides = {achievement_cols[0]: 100.0, **{c: 0.0 for c in achievement_cols[1:]}}
        df = _df(_row(**overrides))
        result = scorer.layer_scores(df)
        assert result["achievement_score"].iloc[0] == pytest.approx(25.0)

    def test_peak_layer_single_metric_equals_metric_value(self, tmp_path):
        """Peak layer has exactly 1 metric — its score must equal the raw metric value."""
        scorer = _scorer(tmp_path)
        df = _df(_row(**{"world_records_normalized": 73.0}))
        result = scorer.layer_scores(df)
        assert result["peak_score"].iloc[0] == pytest.approx(73.0)

    def test_layer_scores_in_0_to_100(self, tmp_path):
        scorer = _scorer(tmp_path)
        df = _df(
            _row(**{c: 0.0 for c in _ALL_COLS}),
            _row(**{c: 100.0 for c in _ALL_COLS}),
        )
        result = scorer.layer_scores(df)
        for col in result.columns:
            assert result[col].min() >= 0.0
            assert result[col].max() <= 100.0


# ── TestScoreBreakdown ────────────────────────────────────────────────────────

class TestScoreBreakdown:
    def test_breakdown_has_goat_score_column(self, tmp_path):
        scorer = _scorer(tmp_path)
        bd = scorer.score_breakdown(_df(_row()))
        assert "goat_score" in bd.columns

    def test_breakdown_has_contribution_columns(self, tmp_path):
        scorer = _scorer(tmp_path)
        bd = scorer.score_breakdown(_df(_row()))
        for col in _ALL_COLS:
            assert f"{col}_contribution" in bd.columns

    def test_breakdown_goat_score_matches_score(self, tmp_path):
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
        scorer = _scorer(tmp_path)
        df = _df(_row(**{c: 60.0 for c in _ALL_COLS}))
        bd = scorer.score_breakdown(df)
        contrib_cols = [f"{c}_contribution" for c in _ALL_COLS]
        contrib_sum = bd[contrib_cols].sum(axis=1).iloc[0]
        assert contrib_sum == pytest.approx(bd["goat_score"].iloc[0], abs=1e-9)
