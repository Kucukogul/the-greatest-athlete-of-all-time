from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tennis_model import (
    FEATURE_COLS,
    TARGET_COL,
    EvalResult,
    TennisForestModel,
    TennisRidgeModel,
    cross_validate_model,
    prepare_features,
)


def make_tennis_df(n: int = 30) -> pd.DataFrame:
    """Synthetic career-level DataFrame mimicking tennis_all_v2.csv schema."""
    rng = np.random.default_rng(42)
    era_choices = ["Open Era", "Modern Era", "Big 3 Era"]
    return pd.DataFrame({
        "name": [f"Player{i}" for i in range(n)],
        "era": rng.choice(era_choices, size=n),
        "hard_win_pct": rng.uniform(0.45, 0.88, n),
        "clay_win_pct": rng.uniform(0.40, 0.91, n),
        "grass_win_pct": rng.uniform(0.42, 0.88, n),
        "clay_win_rate_std": rng.uniform(0.05, 0.35, n),
        "grass_win_rate_std": rng.uniform(0.04, 0.38, n),
        "surface_versatility_normalized": rng.uniform(50.0, 98.0, n),
        TARGET_COL: rng.uniform(5.0, 95.0, n),
    })


def make_X_y(n: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    df = make_tennis_df(n)
    X = prepare_features(df)
    y = df[TARGET_COL]
    return X, y


class TestPrepareFeatures:
    def test_returns_exactly_feature_cols(self):
        df = make_tennis_df()
        X = prepare_features(df)
        assert list(X.columns) == FEATURE_COLS

    def test_index_preserved(self):
        df = make_tennis_df().set_index("name")
        X = prepare_features(df.reset_index())
        assert len(X) == len(df)

    def test_no_nulls_in_output(self):
        df = make_tennis_df()
        X = prepare_features(df)
        assert X.isnull().sum().sum() == 0

    def test_raises_on_missing_column(self):
        df = make_tennis_df().drop(columns=["hard_win_pct"])
        with pytest.raises((ValueError, KeyError)):
            prepare_features(df)


class TestTennisRidgeModel:
    def test_fit_returns_self(self):
        X, y = make_X_y()
        model = TennisRidgeModel()
        result = model.fit(X, y)
        assert result is model

    def test_predict_returns_series_with_correct_length(self):
        X, y = make_X_y()
        model = TennisRidgeModel().fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X)

    def test_predict_index_matches_input(self):
        X, y = make_X_y()
        model = TennisRidgeModel().fit(X, y)
        preds = model.predict(X)
        pd.testing.assert_index_equal(preds.index, X.index)

    def test_coefficients_has_all_features(self):
        X, y = make_X_y()
        model = TennisRidgeModel().fit(X, y)
        coefs = model.coefficients()
        assert set(coefs.keys()) == set(FEATURE_COLS)

    def test_coefficients_sorted_by_magnitude(self):
        X, y = make_X_y()
        coefs = TennisRidgeModel().fit(X, y).coefficients()
        magnitudes = [abs(v) for v in coefs.values()]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_higher_alpha_shrinks_coefficients(self):
        X, y = make_X_y()
        coefs_low = TennisRidgeModel(alpha=0.01).fit(X, y).coefficients()
        coefs_high = TennisRidgeModel(alpha=100.0).fit(X, y).coefficients()
        sum_low = sum(abs(v) for v in coefs_low.values())
        sum_high = sum(abs(v) for v in coefs_high.values())
        assert sum_high < sum_low


class TestTennisForestModel:
    def test_fit_returns_self(self):
        X, y = make_X_y()
        model = TennisForestModel(n_estimators=10)
        result = model.fit(X, y)
        assert result is model

    def test_predict_returns_series_with_correct_length(self):
        X, y = make_X_y()
        model = TennisForestModel(n_estimators=10).fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X)

    def test_predict_index_matches_input(self):
        X, y = make_X_y()
        model = TennisForestModel(n_estimators=10).fit(X, y)
        preds = model.predict(X)
        pd.testing.assert_index_equal(preds.index, X.index)

    def test_feature_importance_has_all_features(self):
        X, y = make_X_y()
        model = TennisForestModel(n_estimators=10).fit(X, y)
        importance = model.feature_importance()
        assert set(importance.keys()) == set(FEATURE_COLS)

    def test_feature_importance_sums_to_one(self):
        X, y = make_X_y()
        importance = TennisForestModel(n_estimators=10).fit(X, y).feature_importance()
        assert sum(importance.values()) == pytest.approx(1.0, abs=1e-3)

    def test_feature_importance_sorted_descending(self):
        X, y = make_X_y()
        importance = TennisForestModel(n_estimators=10).fit(X, y).feature_importance()
        values = list(importance.values())
        assert values == sorted(values, reverse=True)


class TestCrossValidateModel:
    def test_returns_eval_result(self):
        X, y = make_X_y(n=50)
        result = cross_validate_model(TennisRidgeModel, X, y, cv=3)
        assert isinstance(result, EvalResult)

    def test_model_name_set_correctly(self):
        X, y = make_X_y(n=50)
        result = cross_validate_model(TennisRidgeModel, X, y, cv=3)
        assert result.model_name == "TennisRidgeModel"

    def test_rmse_is_positive(self):
        X, y = make_X_y(n=50)
        result = cross_validate_model(TennisRidgeModel, X, y, cv=3)
        assert result.rmse_cv_mean > 0

    def test_r2_std_is_non_negative(self):
        X, y = make_X_y(n=50)
        result = cross_validate_model(TennisRidgeModel, X, y, cv=3)
        assert result.r2_cv_std >= 0

    def test_forest_factory_lambda(self):
        X, y = make_X_y(n=50)
        result = cross_validate_model(
            lambda: TennisForestModel(n_estimators=10), X, y, cv=3
        )
        assert result.model_name == "TennisForestModel"
