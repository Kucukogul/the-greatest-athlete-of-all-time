"""Unit tests for NBAForestModel and NBAClusterModel.

All tests use synthetic DataFrames — no file I/O, no real NBA data.
Config dicts are built inline to avoid depending on the YAML file.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.nba_model import (
    NBAClusterModel,
    NBAEvalResult,
    NBAForestModel,
    cross_validate_forest,
    load_nba_model_config,
)

# ── Synthetic data ─────────────────────────────────────────────────────────────

_FOREST_FEATURES = [
    "peak_bpm_normalized", "peak_ws48_normalized", "peak_vorp_normalized",
    "career_vorp_normalized", "career_ws_normalized", "career_bpm_normalized",
    "career_ws48_normalized", "career_ts_normalized",
    "championships_normalized", "mvp_normalized", "finals_mvp_normalized",
    "all_nba_1st_normalized", "all_star_normalized", "longevity_normalized",
]

_CLUSTER_FEATURES = [
    "peak_ws48_normalized", "career_ws_normalized", "career_ws48_normalized",
    "career_ts_normalized", "championships_normalized", "mvp_normalized",
    "all_nba_1st_normalized", "all_star_normalized", "longevity_normalized",
    "peak_bpm_normalized", "career_bpm_normalized", "career_vorp_normalized",
]

_ALL_FEATURES = list(set(_FOREST_FEATURES) | set(_CLUSTER_FEATURES))


_BASE_FOREST = {
    "target": "goat_score",
    "n_estimators": 10,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "random_state": 42,
    "nan_fill_strategy": "median",
    "features": _FOREST_FEATURES,
}

_BASE_CLUSTER = {
    "n_clusters": 3,
    "n_init": 5,
    "random_state": 42,
    "nan_fill_strategy": "median",
    "tier_labels": ["GOAT", "Elite", "Starter"],
    "features": _CLUSTER_FEATURES,
}


def _forest_cfg(**overrides) -> dict:
    return {"forest": {**_BASE_FOREST, **overrides}, "cluster": _BASE_CLUSTER}


def _cluster_cfg(**overrides) -> dict:
    return {"forest": _BASE_FOREST, "cluster": {**_BASE_CLUSTER, **overrides}}


def _make_df(n: int = 50, seed: int = 0, nan_bpm: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Generate n synthetic player rows with all required columns."""
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 100, n) for col in _ALL_FEATURES}
    if nan_bpm:
        # Simulate pre-advanced era: first 10 players have NaN BPM/VORP
        for col in ["peak_bpm_normalized", "career_bpm_normalized", "career_vorp_normalized"]:
            data[col][:10] = np.nan
    df = pd.DataFrame(data)
    df["name"] = [f"Player_{i}" for i in range(n)]
    goat_score = pd.Series(rng.uniform(20, 100, n), name="goat_score")
    return df, goat_score


# ── TestConfigLoading ──────────────────────────────────────────────────────────

class TestConfigLoading:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_nba_model_config("/nonexistent/model_nba.yaml")

    def test_real_config_loads(self):
        path = Path(__file__).parents[2] / "configs" / "model_nba.yaml"
        if not path.exists():
            pytest.skip("configs/model_nba.yaml not present")
        cfg = load_nba_model_config(path)
        assert "forest" in cfg
        assert "cluster" in cfg


# ── TestNBAForestModel ────────────────────────────────────────────────────────

class TestNBAForestModelFit:
    def test_fit_returns_self(self):
        df, y = _make_df()
        model = NBAForestModel(_forest_cfg())
        result = model.fit(df, y)
        assert result is model

    def test_predict_returns_series(self):
        df, y = _make_df()
        model = NBAForestModel(_forest_cfg()).fit(df, y)
        preds = model.predict(df)
        assert isinstance(preds, pd.Series)
        assert preds.name == "predicted_goat_score"
        assert len(preds) == len(df)

    def test_predict_index_matches_input(self):
        df, y = _make_df(n=30)
        df.index = range(100, 130)
        model = NBAForestModel(_forest_cfg()).fit(df, y)
        preds = model.predict(df)
        assert list(preds.index) == list(df.index)

    def test_missing_feature_column_raises(self):
        df, y = _make_df()
        df = df.drop(columns=["peak_bpm_normalized"])
        with pytest.raises(ValueError, match="missing features"):
            NBAForestModel(_forest_cfg()).fit(df, y)

    def test_predict_before_fit_raises(self):
        df, _ = _make_df()
        model = NBAForestModel(_forest_cfg())
        # sklearn raises AttributeError on predict before fit
        with pytest.raises(Exception):
            model.predict(df)


class TestNBAForestModelNan:
    def test_nan_bpm_handled_without_error(self):
        df, y = _make_df(nan_bpm=True)
        model = NBAForestModel(_forest_cfg())
        model.fit(df, y)
        preds = model.predict(df)
        assert preds.notna().all()

    def test_imputation_does_not_leak_future_data(self):
        """Imputer fitted on train must be used on test — no refitting on test."""
        df_train, y_train = _make_df(n=40, seed=1, nan_bpm=True)
        df_test, _ = _make_df(n=10, seed=2, nan_bpm=True)
        model = NBAForestModel(_forest_cfg())
        model.fit(df_train, y_train)
        preds = model.predict(df_test)
        assert len(preds) == 10
        assert preds.notna().all()


class TestNBAForestModelFeatureImportance:
    def test_feature_importance_returns_series(self):
        df, y = _make_df()
        model = NBAForestModel(_forest_cfg()).fit(df, y)
        fi = model.feature_importance()
        assert isinstance(fi, pd.Series)
        assert fi.name == "importance"

    def test_feature_importance_sums_to_1(self):
        df, y = _make_df()
        model = NBAForestModel(_forest_cfg()).fit(df, y)
        fi = model.feature_importance()
        assert fi.sum() == pytest.approx(1.0, abs=1e-6)

    def test_feature_importance_index_matches_features(self):
        df, y = _make_df()
        model = NBAForestModel(_forest_cfg()).fit(df, y)
        fi = model.feature_importance()
        assert set(fi.index) == set(_FOREST_FEATURES)

    def test_feature_importance_sorted_descending(self):
        df, y = _make_df()
        model = NBAForestModel(_forest_cfg()).fit(df, y)
        fi = model.feature_importance()
        assert fi.is_monotonic_decreasing


class TestNBAForestModelSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        df, y = _make_df()
        model = NBAForestModel(_forest_cfg()).fit(df, y)
        preds_before = model.predict(df)

        path = tmp_path / "forest.pkl"
        model.save(path)
        loaded = NBAForestModel.load(path)
        preds_after = loaded.predict(df)

        pd.testing.assert_series_equal(preds_before, preds_after)

    def test_saved_file_exists(self, tmp_path):
        df, y = _make_df()
        model = NBAForestModel(_forest_cfg()).fit(df, y)
        path = tmp_path / "forest.pkl"
        model.save(path)
        assert path.exists()


# ── TestNBAClusterModel ───────────────────────────────────────────────────────

class TestNBAClusterModelFit:
    def test_fit_returns_self(self):
        df, y = _make_df()
        model = NBAClusterModel(_cluster_cfg())
        assert model.fit(df, y) is model

    def test_predict_returns_series(self):
        df, y = _make_df()
        model = NBAClusterModel(_cluster_cfg()).fit(df, y)
        tiers = model.predict(df)
        assert isinstance(tiers, pd.Series)
        assert tiers.name == "tier"
        assert len(tiers) == len(df)

    def test_predict_only_valid_tier_labels(self):
        df, y = _make_df()
        model = NBAClusterModel(_cluster_cfg()).fit(df, y)
        tiers = model.predict(df)
        valid = {"GOAT", "Elite", "Starter"}
        assert set(tiers.unique()).issubset(valid)

    def test_all_n_clusters_are_populated(self):
        df, y = _make_df(n=60)
        model = NBAClusterModel(_cluster_cfg()).fit(df, y)
        tiers = model.predict(df)
        assert len(tiers.unique()) == 3

    def test_predict_before_fit_raises(self):
        df, _ = _make_df()
        model = NBAClusterModel(_cluster_cfg())
        with pytest.raises(RuntimeError, match="fit()"):
            model.predict(df)

    def test_too_few_tier_labels_raises(self):
        cfg = _cluster_cfg(n_clusters=5, tier_labels=["A", "B"])
        with pytest.raises(ValueError, match="tier labels"):
            NBAClusterModel(cfg)

    def test_missing_feature_column_raises(self):
        df, y = _make_df()
        df = df.drop(columns=["peak_ws48_normalized"])
        with pytest.raises(ValueError, match="missing features"):
            NBAClusterModel(_cluster_cfg()).fit(df, y)


class TestNBAClusterModelNan:
    def test_nan_bpm_handled(self):
        df, y = _make_df(nan_bpm=True)
        model = NBAClusterModel(_cluster_cfg()).fit(df, y)
        tiers = model.predict(df)
        assert tiers.notna().all()

    def test_nan_predict_without_refit(self):
        """Imputer from train must apply to new rows with NaN."""
        df_train, y_train = _make_df(n=40, nan_bpm=True)
        df_test, _ = _make_df(n=10, nan_bpm=True, seed=99)
        model = NBAClusterModel(_cluster_cfg()).fit(df_train, y_train)
        tiers = model.predict(df_test)
        assert tiers.notna().all()


class TestNBAClusterModelTierRanking:
    def test_goat_cluster_has_highest_mean_score(self):
        """GOAT tier should contain the player with the highest goat_score."""
        df, y = _make_df(n=120, seed=5)
        model = NBAClusterModel(_cluster_cfg()).fit(df, y)
        tiers = model.predict(df)
        df_result = df.copy()
        df_result["goat_score"] = y.values
        df_result["tier"] = tiers.values
        tier_means = df_result.groupby("tier")["goat_score"].mean()
        assert tier_means["GOAT"] == tier_means.max()


class TestNBAClusterModelSummary:
    def test_cluster_summary_has_expected_columns(self):
        df, y = _make_df(n=60)
        df["goat_score"] = y.values
        model = NBAClusterModel(_cluster_cfg()).fit(df, y)
        tiers = model.predict(df)
        summary = model.cluster_summary(df, tiers)
        for col in ["count", "goat_score_mean", "goat_score_min", "goat_score_max", "top_player"]:
            assert col in summary.columns

    def test_cluster_summary_counts_sum_to_total(self):
        df, y = _make_df(n=60)
        df["goat_score"] = y.values
        model = NBAClusterModel(_cluster_cfg()).fit(df, y)
        tiers = model.predict(df)
        summary = model.cluster_summary(df, tiers)
        assert summary["count"].sum() == len(df)

    def test_cluster_summary_indexed_by_tier(self):
        df, y = _make_df(n=60)
        df["goat_score"] = y.values
        model = NBAClusterModel(_cluster_cfg()).fit(df, y)
        tiers = model.predict(df)
        summary = model.cluster_summary(df, tiers)
        assert summary.index.name == "tier"


class TestNBAClusterModelSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        df, y = _make_df(n=60)
        model = NBAClusterModel(_cluster_cfg()).fit(df, y)
        tiers_before = model.predict(df)

        path = tmp_path / "cluster.pkl"
        model.save(path)
        loaded = NBAClusterModel.load(path)
        tiers_after = loaded.predict(df)

        pd.testing.assert_series_equal(tiers_before, tiers_after)


# ── TestCrossValidateForest ───────────────────────────────────────────────────

class TestCrossValidateForest:
    def test_returns_eval_result(self):
        df, y = _make_df(n=50)
        cfg = _forest_cfg()
        result = cross_validate_forest(df, y, cfg, cv=3)
        assert isinstance(result, NBAEvalResult)

    def test_r2_in_valid_range(self):
        df, y = _make_df(n=50)
        result = cross_validate_forest(df, y, _forest_cfg(), cv=3)
        assert result.r2_cv_mean <= 1.0

    def test_rmse_is_positive(self):
        df, y = _make_df(n=50)
        result = cross_validate_forest(df, y, _forest_cfg(), cv=3)
        assert result.rmse_cv_mean >= 0.0

    def test_model_name_is_correct(self):
        df, y = _make_df(n=50)
        result = cross_validate_forest(df, y, _forest_cfg(), cv=3)
        assert result.model_name == "NBAForestModel"

    def test_str_representation(self):
        result = NBAEvalResult("NBAForestModel", 0.95, 0.02, 3.5)
        s = str(result)
        assert "R²=0.9500" in s
        assert "RMSE=3.5000" in s
