from __future__ import annotations

import pandas as pd


def normalize_series(s: pd.Series, method: str = "minmax") -> pd.Series:
    if method == "minmax":
        min_val, max_val = s.min(), s.max()
        if max_val == min_val:
            return pd.Series(50.0, index=s.index)
        return (s - min_val) / (max_val - min_val) * 100
    if method == "zscore":
        return (s - s.mean()) / s.std() * 15 + 50
    raise ValueError(f"Unknown normalization method: {method}")


def percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100
