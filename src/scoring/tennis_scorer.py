from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class TennisScorer:
    """TennisNormalizer output → composite GOAT score [0–100] per player.

    Tennis data carries no structural NaN: TennisNormalizer fills missing
    ratios (finals_win_rate, h2h_top10) with 0.0, so a plain weighted average
    is used — no weight-rescaling is needed.

    Layers are read from the yaml `layers:` block, not from the normalizer,
    because TennisNormalizer has no layer-metadata API.

    Usage:
        scorer = TennisScorer("configs/scoring_tennis.yaml")
        scores = scorer.score(df_norm)         # pd.Series "goat_score"
        layers = scorer.layer_scores(df_norm)  # per-layer breakdown DataFrame
    """

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        cfg = self._load_config()
        self._weights: dict[str, float] = cfg["weights"]
        self._layers: dict[str, list[str]] = {
            name: block["metrics"] for name, block in cfg.get("layers", {}).items()
        }
        self._penalty_pct: float = cfg.get("penalties", {}).get(
            "tournaments_played_pct", 0.0
        )
        self._validate_weights()
        logger.info(
            "TennisScorer loaded — %d metrics, %d layers, config=%s",
            len(self._weights),
            len(self._layers),
            self._config_path.name,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Composite GOAT score [0–100] for each player row."""
        self._validate_columns(df)
        metric_cols = list(self._weights.keys())
        weight_vals = np.array([self._weights[c] for c in metric_cols], dtype=float)
        values = df[metric_cols].to_numpy(dtype=float)
        raw = (values * weight_vals).sum(axis=1)
        scores = pd.Series(np.clip(raw, 0.0, 100.0), index=df.index, name="goat_score")
        return self._apply_penalty(scores, df)

    def layer_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-layer GOAT scores [0–100] using equal weights within each layer.

        Each layer score is independent of inter-layer weights — suitable for
        radar-chart axes where all dimensions must be on the same [0–100] scale.
        """
        self._validate_columns(df)
        result = pd.DataFrame(index=df.index)
        for layer_name, metrics in self._layers.items():
            layer_cols = [c for c in metrics if c in df.columns]
            if not layer_cols:
                result[f"{layer_name}_score"] = np.nan
                continue
            values = df[layer_cols].to_numpy(dtype=float)
            equal_w = np.ones(len(layer_cols), dtype=float) / len(layer_cols)
            result[f"{layer_name}_score"] = np.clip(
                (values * equal_w).sum(axis=1), 0.0, 100.0
            )
        return result

    def score_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-metric weighted contribution columns plus final composite score."""
        result = pd.DataFrame(index=df.index)
        result["goat_score"] = self.score(df)
        for col, w in self._weights.items():
            result[f"{col}_contribution"] = df[col] * w
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_penalty(self, scores: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Deduct up to (penalty_pct × 100) pts for low participation; no-op when column absent."""
        col = "tournaments_played_pct"
        if self._penalty_pct == 0.0 or col not in df.columns:
            return scores
        max_deduction = self._penalty_pct * 100.0
        deduction = (1.0 - df[col].clip(0.0, 1.0)) * max_deduction
        return pd.Series(
            np.clip(scores.values - deduction.values, 0.0, 100.0),
            index=scores.index,
            name="goat_score",
        )

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
