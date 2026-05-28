from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: set[str] = {"athlete_id", "name", "sport", "season"}


def validate_athlete_dataframe(
    df: pd.DataFrame,
    required: set[str] = REQUIRED_COLUMNS,
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    if df["athlete_id"].duplicated().any():
        raise ValueError("Duplicate athlete_id values found")
