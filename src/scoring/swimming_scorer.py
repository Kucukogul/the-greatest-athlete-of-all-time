from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class SwimmingScorer:
    """SwimmingNormalizer output → composite GOAT score [0–100] per athlete.

    Swimming data has no structural NaN after normalization — Pre-Modern era
    athletes receive an estimated wc_gold_individual in SwimmingNormalizer,
    and relay_dependency_penalty is disabled (set to 0.0 in config). A plain
    weighted average is used; no weight-rescaling is needed.
    """

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        cfg = self._load_config()
        self._weights: dict[str, float] = cfg["weights"]
        self._layers: dict[str, list[str]] = {
            name: block["metrics"] for name, block in cfg.get("layers", {}).items()
        }
        self._validate_weights()
        logger.info(
            "SwimmingScorer loaded — %d metrics, %d layers, config=%s",
            len(self._weights),
            len(self._layers),
            self._config_path.name,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Composite GOAT score [0–100] for each athlete row."""
        self._validate_columns(df)
        metric_cols = list(self._weights.keys())
        weight_vals = np.array([self._weights[c] for c in metric_cols], dtype=float)
        values = df[metric_cols].to_numpy(dtype=float)
        raw = (values * weight_vals).sum(axis=1)
        return pd.Series(np.clip(raw, 0.0, 100.0), index=df.index, name="goat_score")

    def layer_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-layer GOAT scores [0–100] using equal weights within each layer.

        Each layer score is independent of inter-layer weights — suitable for
        radar-chart axes where all four dimensions share the same [0–100] scale.
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
        self._validate_columns(df)
        result = pd.DataFrame(index=df.index)
        result["goat_score"] = self.score(df)
        for col, w in self._weights.items():
            result[f"{col}_contribution"] = df[col] * w
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

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
            raise ValueError(f"Missing required normalized columns: {sorted(missing)}")
