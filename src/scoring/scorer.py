from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml


@dataclass
class ScoringConfig:
    weights: dict[str, float]
    strategy: Literal["weighted_average", "rank_based"] = "weighted_average"
    penalties: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


class AthleteScorer:
    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def score(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        self._validate_columns(df)
        if self.config.strategy == "weighted_average":
            return self._weighted_average(df)
        return self._rank_based(df)

    def score_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        self._validate_columns(df)
        breakdown = pd.DataFrame(index=df.index)
        for metric, weight in self.config.weights.items():
            breakdown[metric] = df[metric] * weight
        breakdown["total"] = self.score(df)
        return breakdown

    def _weighted_average(self, df: pd.DataFrame) -> pd.Series:
        scores = pd.Series(0.0, index=df.index)
        for metric, weight in self.config.weights.items():
            scores += df[metric] * weight
        for metric, penalty in self.config.penalties.items():
            if metric in df.columns:
                scores *= 1 - (1 - df[metric]) * penalty
        return scores

    def _rank_based(self, df: pd.DataFrame) -> pd.Series:
        ranks = pd.DataFrame(index=df.index)
        for metric in self.config.weights:
            ranks[metric] = df[metric].rank(pct=True) * 100
        return self._weighted_average(ranks)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = set(self.config.weights) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required metric columns: {missing}")


def load_scoring_config(path: str | Path) -> ScoringConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScoringConfig(
        weights=data["weights"],
        strategy=data.get("strategy", "weighted_average"),
        penalties=data.get("penalties", {}),
    )