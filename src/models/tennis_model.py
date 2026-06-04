from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineering import TennisSurfaceFeatures
from src.models.base import BaseAthleteModel

FEATURE_COLS: list[str] = [
    "hard_win_pct",
    "clay_win_pct",
    "grass_win_pct",
    "clay_win_rate_std",
    "grass_win_rate_std",
    "surface_flexibility",
    "surface_gap",
    "surface_floor",
    "era_encoded",
    "surface_versatility_normalized",
    "hard_era_interaction",
    "clay_era_interaction",
    "grass_era_interaction",
    "versatility_era_interaction",
]

TARGET_COL: str = "composite_score"


@dataclass
class EvalResult:
    model_name: str
    r2_cv_mean: float
    r2_cv_std: float
    rmse_cv_mean: float


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run surface feature engineering and return a matrix with exactly FEATURE_COLS."""
    engineered = TennisSurfaceFeatures.build_all(df)
    missing = [c for c in FEATURE_COLS if c not in engineered.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")
    return engineered[FEATURE_COLS].copy()


def cross_validate_model(
    model_factory: Callable[[], BaseAthleteModel],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> EvalResult:
    """K-fold CV returning mean R² and RMSE. Pass the class directly for default params."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    r2_scores: list[float] = []
    rmse_scores: list[float] = []

    for train_idx, val_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        ss_res = float(((y_val.values - preds.values) ** 2).sum())
        ss_tot = float(((y_val.values - y_val.values.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(((y_val.values - preds.values) ** 2).mean()))

        r2_scores.append(r2)
        rmse_scores.append(rmse)

    return EvalResult(
        model_name=model_factory().__class__.__name__,
        r2_cv_mean=round(float(np.mean(r2_scores)), 4),
        r2_cv_std=round(float(np.std(r2_scores)), 4),
        rmse_cv_mean=round(float(np.mean(rmse_scores)), 4),
    )


class TennisRidgeModel(BaseAthleteModel):
    """Ridge regression with internal StandardScaler so alpha acts uniformly across features."""

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TennisRidgeModel":
        self._pipeline.fit(X[FEATURE_COLS], y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self._pipeline.predict(X[FEATURE_COLS])
        return pd.Series(preds, index=X.index, name="predicted_score")

    def coefficients(self) -> dict[str, float]:
        """Scaled coefficients sorted by absolute magnitude."""
        coefs = self._pipeline.named_steps["ridge"].coef_
        return {
            feat: round(float(coef), 4)
            for feat, coef in sorted(
                zip(FEATURE_COLS, coefs),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        }


class TennisForestModel(BaseAthleteModel):
    """Random Forest over surface + era features. No scaling needed — splits are scale-invariant."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TennisForestModel":
        self._model.fit(X[FEATURE_COLS], y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(X[FEATURE_COLS])
        return pd.Series(preds, index=X.index, name="predicted_score")

    def feature_importance(self) -> dict[str, float]:
        """Mean decrease in impurity, sorted descending."""
        importances = self._model.feature_importances_
        return {
            feat: round(float(imp), 4)
            for feat, imp in sorted(
                zip(FEATURE_COLS, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        }
