from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.normalize.nba_normalizer import NBANormalizer as _NBANorm

logger = logging.getLogger(__name__)


class NBAScorer:
    """NBANormalizer output → composite GOAT score [0–100] per player.

    Pre-advanced era players (pre-1974) have NaN for BPM/VORP columns.
    Their score is computed from available metrics only and rescaled by the
    sum of available weights — output stays in [0, 100] regardless of missing data.

    Usage:
        scorer = NBAScorer("configs/scoring_nba.yaml")
        scores = scorer.score(df_norm)         # pd.Series "goat_score"
        layers = scorer.layer_scores(df_norm)  # per-layer breakdown DataFrame
    """

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        cfg = self._load_config()
        self._weights: dict[str, float] = cfg["weights"]
        self._validate_weights()
        logger.info(
            "NBAScorer loaded — %d metrics, config=%s",
            len(self._weights),
            self._config_path.name,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> pd.Series:
        """NaN-safe composite GOAT score [0–100] for each player row."""
        self._validate_columns(df)
        metric_cols = list(self._weights.keys())
        weight_vals = np.array([self._weights[c] for c in metric_cols], dtype=float)
        values = df[metric_cols].to_numpy(dtype=float)
        scores = self._weighted_avg_nan(values, weight_vals)
        return pd.Series(scores, index=df.index, name="goat_score")

    def score_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-metric weighted contribution columns plus final composite score.

        For pre-advanced era players, NaN-column contributions are NaN. The
        'goat_score' column is NaN-rescaled and will not equal the arithmetic
        sum of contribution columns for those players.
        """
        result = pd.DataFrame(index=df.index)
        result["goat_score"] = self.score(df)  # validates internally
        for col, w in self._weights.items():
            result[f"{col}_contribution"] = df[col] * w
        return result

    def layer_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-layer GOAT scores [0–100] using equal weights within each layer.

        Useful for radar-chart visualisation and per-dimension comparisons.
        Each layer score is independent of inter-layer weights.
        """
        self._validate_columns(df)
        result = pd.DataFrame(index=df.index)
        for layer in _NBANorm.layers():
            layer_cols = [c for c in _NBANorm.layer_columns(layer) if c in df.columns]
            if not layer_cols:
                result[f"{layer}_score"] = np.nan
                continue
            values = df[layer_cols].to_numpy(dtype=float)
            equal_weights = np.ones(len(layer_cols), dtype=float) / len(layer_cols)
            result[f"{layer}_score"] = self._weighted_avg_nan(values, equal_weights)
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _weighted_avg_nan(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """NaN-safe weighted average over columns. Returns NaN only when all values are NaN."""
        valid_mask = ~np.isnan(values)
        available_w = np.where(valid_mask, weights, 0.0)
        weight_sums = available_w.sum(axis=1)
        raw_sums = np.where(valid_mask, values * weights, 0.0).sum(axis=1)
        # np.where evaluates both branches; errstate suppresses the 0/0 warning
        # for all-NaN rows before np.where selects np.nan for them.
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight_sums > 0, raw_sums / weight_sums, np.nan)

    def _load_config(self) -> dict:
        if not self._config_path.exists():
            raise FileNotFoundError(f"Scoring config not found: {self._config_path}")
        with open(self._config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate_weights(self) -> None:
        total = sum(self._weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights in {self._config_path.name} must sum to 1.0, got {total:.8f}"
            )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        missing = set(self._weights) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
