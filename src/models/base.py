from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd


class BaseAthleteModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseAthleteModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        ...

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseAthleteModel":
        return joblib.load(path)
