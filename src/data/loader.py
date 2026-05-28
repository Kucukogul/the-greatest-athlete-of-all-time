from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EXTERNAL_DIR = Path("data/external")


class DataLoader:
    def __init__(self, data_dir: Path = RAW_DIR) -> None:
        self.data_dir = data_dir

    def load_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        logger.info("Loading %s", path)
        return pd.read_csv(path)

    def load_parquet(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        logger.info("Loading %s", path)
        return pd.read_parquet(path)

    def save_processed(self, df: pd.DataFrame, filename: str) -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PROCESSED_DIR / filename
        df.to_parquet(path, index=False)
        logger.info("Saved processed data to %s", path)
