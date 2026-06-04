from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self._scaler = MinMaxScaler(feature_range=(0, 100))

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result[self._normalized_names()] = self._scaler.fit_transform(
            df[self.feature_columns]
        )
        return result

    def normalize_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result[self._normalized_names()] = self._scaler.transform(
            df[self.feature_columns]
        )
        return result

    def era_adjust(
        self,
        df: pd.DataFrame,
        era_col: str,
        era_means: dict[str, float],
    ) -> pd.DataFrame:
        result = df.copy()
        for col in self.feature_columns:
            result[col] = result.apply(
                lambda row, c=col: row[c] / era_means.get(str(row[era_col]), 1.0),
                axis=1,
            )
        return result

    def _normalized_names(self) -> list[str]:
        return [f"{col}_normalized" for col in self.feature_columns]


_ERA_ORDER: dict[str, int] = {
    "Open Era": 0,
    "Modern Era": 1,
    "Big 3 Era": 2,
}


class TennisSurfaceFeatures:
    """Surface feature derivations for the GOAT model. Input → output, no mutation."""

    @staticmethod
    def add_surface_flexibility(df: pd.DataFrame) -> pd.DataFrame:
        """clay × grass product — both must be high for the score to be high."""
        result = df.copy()
        result["surface_flexibility"] = result["clay_win_pct"] * result["grass_win_pct"]
        return result

    @staticmethod
    def add_surface_gap(df: pd.DataFrame) -> pd.DataFrame:
        """|clay − grass| — low gap means the player performs evenly across surfaces."""
        result = df.copy()
        result["surface_gap"] = (result["clay_win_pct"] - result["grass_win_pct"]).abs()
        return result

    @staticmethod
    def add_surface_floor(df: pd.DataFrame) -> pd.DataFrame:
        """Worst surface win rate — GOAT candidates stay high on every surface."""
        result = df.copy()
        result["surface_floor"] = df[["hard_win_pct", "clay_win_pct", "grass_win_pct"]].min(axis=1)
        return result

    @staticmethod
    def add_era_encoding(df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal: Open Era=0, Modern Era=1, Big 3 Era=2."""
        result = df.copy()
        result["era_encoded"] = result["era"].map(_ERA_ORDER)
        return result

    @staticmethod
    def add_era_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """Requires era_encoded — call add_era_encoding first."""
        if "era_encoded" not in df.columns:
            raise ValueError("era_encoded column missing — call add_era_encoding first")
        result = df.copy()
        era = result["era_encoded"]
        result["hard_era_interaction"] = result["hard_win_pct"] * era
        result["clay_era_interaction"] = result["clay_win_pct"] * era
        result["grass_era_interaction"] = result["grass_win_pct"] * era
        result["versatility_era_interaction"] = result["surface_versatility_normalized"] * era
        return result

    @classmethod
    def build_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.pipe(cls.add_surface_flexibility)
              .pipe(cls.add_surface_gap)
              .pipe(cls.add_surface_floor)
              .pipe(cls.add_era_encoding)
              .pipe(cls.add_era_interactions)
        )
