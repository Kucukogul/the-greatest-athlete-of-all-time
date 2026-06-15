from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from src.models.base import BaseAthleteModel

logger = logging.getLogger(__name__)


# ── Config loader ─────────────────────────────────────────────────────────────

def load_nba_model_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"NBA model config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Evaluation result ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NBAEvalResult:
    model_name: str
    r2_cv_mean: float
    r2_cv_std: float
    rmse_cv_mean: float

    def __str__(self) -> str:
        return (
            f"{self.model_name}: R²={self.r2_cv_mean:.4f}±{self.r2_cv_std:.4f}  "
            f"RMSE={self.rmse_cv_mean:.4f}"
        )


# ── NBAForestModel ────────────────────────────────────────────────────────────

class NBAForestModel(BaseAthleteModel):
    """Random Forest regressor: normalized features → predicted goat_score.

    Primary use is feature_importance() — a non-linear sanity-check on the
    rule-based scorer weights. BPM/VORP NaN values are median-imputed on the
    training set; the fitted imputer is reused for predictions (no leakage).
    """

    def __init__(self, cfg: dict) -> None:
        forest_cfg = cfg["forest"]
        self._features: list[str] = forest_cfg["features"]
        self._target: str = forest_cfg.get("target", "goat_score")
        self._imputer = SimpleImputer(
            strategy=forest_cfg.get("nan_fill_strategy", "median")
        )
        self._model = RandomForestRegressor(
            n_estimators=forest_cfg.get("n_estimators", 300),
            max_depth=forest_cfg.get("max_depth") or None,
            min_samples_leaf=forest_cfg.get("min_samples_leaf", 3),
            random_state=forest_cfg.get("random_state", 42),
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NBAForestModel":
        self._validate(X)
        X_imp = self._imputer.fit_transform(X[self._features])
        self._model.fit(X_imp, y)
        logger.info(
            "NBAForestModel fitted — %d players, %d features, target=%s",
            len(X),
            len(self._features),
            self._target,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._validate(X)
        X_imp = self._imputer.transform(X[self._features])
        preds = self._model.predict(X_imp)
        return pd.Series(preds, index=X.index, name="predicted_goat_score")

    def feature_importance(self) -> pd.Series:
        """Mean decrease in impurity per feature, sorted descending. Sums to 1.0."""
        return (
            pd.Series(
                self._model.feature_importances_,
                index=self._features,
                name="importance",
            )
            .sort_values(ascending=False)
        )

    def _validate(self, X: pd.DataFrame) -> None:
        missing = set(self._features) - set(X.columns)
        if missing:
            raise ValueError(f"NBAForestModel missing features: {sorted(missing)}")


# ── NBAClusterModel ───────────────────────────────────────────────────────────

class NBAClusterModel(BaseAthleteModel):
    """K-Means tier clustering: assigns each player to a GOAT tier label.

    Tier assignment: after fitting, clusters are ranked by their mean goat_score.
    The highest-scoring cluster is labelled with tier_labels[0] (e.g. "GOAT"),
    the next with tier_labels[1] (e.g. "Elite"), and so on.

    NaN handling: BPM/VORP columns are median-imputed (fit once on training data).

    Usage:
        model = NBAClusterModel(cfg)
        model.fit(df_normalized, df_normalized["goat_score"])
        tiers = model.predict(df_normalized)   # pd.Series of tier strings
        summary = model.cluster_summary(df_normalized, tiers)
    """

    def __init__(self, cfg: dict) -> None:
        cluster_cfg = cfg["cluster"]
        self._features: list[str] = cluster_cfg["features"]
        self._n_clusters: int = cluster_cfg.get("n_clusters", 5)
        self._tier_labels: list[str] = cluster_cfg.get(
            "tier_labels", [f"Tier{i}" for i in range(self._n_clusters)]
        )
        if len(self._tier_labels) < self._n_clusters:
            raise ValueError(
                f"Need {self._n_clusters} tier labels, "
                f"got {len(self._tier_labels)}: {self._tier_labels}"
            )
        self._imputer = SimpleImputer(
            strategy=cluster_cfg.get("nan_fill_strategy", "median")
        )
        self._kmeans = KMeans(
            n_clusters=self._n_clusters,
            n_init=cluster_cfg.get("n_init", 20),
            random_state=cluster_cfg.get("random_state", 42),
        )
        self._cluster_to_tier: dict[int, str] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NBAClusterModel":
        """Fit K-Means; y (goat_score) is used only to rank clusters post-fit, not for training."""
        self._validate(X)
        X_imp = self._imputer.fit_transform(X[self._features])
        cluster_ids = self._kmeans.fit_predict(X_imp)

        cluster_means = (
            pd.Series(y.values, index=cluster_ids, name="goat_score")
            .groupby(level=0)
            .mean()
            .sort_values(ascending=False)
        )
        self._cluster_to_tier = {
            int(cid): self._tier_labels[rank]
            for rank, cid in enumerate(cluster_means.index)
        }
        logger.info(
            "NBAClusterModel fitted — %d players → %d clusters. Tier map: %s",
            len(X),
            self._n_clusters,
            self._cluster_to_tier,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._cluster_to_tier:
            raise RuntimeError("NBAClusterModel.fit() must be called before predict().")
        self._validate(X)
        X_imp = self._imputer.transform(X[self._features])
        cluster_ids = self._kmeans.predict(X_imp)
        tiers = [self._cluster_to_tier[int(c)] for c in cluster_ids]
        return pd.Series(tiers, index=X.index, name="tier")

    def cluster_summary(self, df: pd.DataFrame, tiers: pd.Series) -> pd.DataFrame:
        """Per-tier summary: count, mean/min/max goat_score, top player. Indexed by tier."""
        summary_rows = []
        for tier in self._tier_labels:
            mask = tiers == tier
            if not mask.any():
                continue
            group = df[mask]
            best_row = group.loc[group["goat_score"].idxmax()]
            summary_rows.append({
                "tier":             tier,
                "count":            int(mask.sum()),
                "goat_score_mean":  round(float(group["goat_score"].mean()), 2),
                "goat_score_min":   round(float(group["goat_score"].min()), 2),
                "goat_score_max":   round(float(group["goat_score"].max()), 2),
                "top_player":       best_row["name"],
                "top_player_score": round(float(best_row["goat_score"]), 2),
            })
        return pd.DataFrame(summary_rows).set_index("tier")

    def _validate(self, X: pd.DataFrame) -> None:
        missing = set(self._features) - set(X.columns)
        if missing:
            raise ValueError(f"NBAClusterModel missing features: {sorted(missing)}")


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_forest(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    cv: int = 5,
) -> NBAEvalResult:
    """K-fold cross-validation for NBAForestModel. Imputer is fit per fold to prevent leakage."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=cfg["forest"].get("random_state", 42))
    r2_scores: list[float] = []
    rmse_scores: list[float] = []

    for train_idx, val_idx in kf.split(X):
        model = NBAForestModel(cfg)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])

        y_val = y.iloc[val_idx].values
        y_pred = preds.values
        ss_res = float(((y_val - y_pred) ** 2).sum())
        ss_tot = float(((y_val - y_val.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(((y_val - y_pred) ** 2).mean()))

        r2_scores.append(r2)
        rmse_scores.append(rmse)

    result = NBAEvalResult(
        model_name="NBAForestModel",
        r2_cv_mean=round(float(np.mean(r2_scores)), 4),
        r2_cv_std=round(float(np.std(r2_scores)), 4),
        rmse_cv_mean=round(float(np.mean(rmse_scores)), 4),
    )
    logger.info("CV (%d-fold): %s", cv, result)
    return result
