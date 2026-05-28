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
