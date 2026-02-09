"""Data ingestion and preprocessing utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    fill_method: str = "ffill"
    drop_na: bool = True


class DataPreprocessor:
    """Clean and prepare raw market data for modeling."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_index().loc[~df.index.duplicated(keep="last")]

        if self.config.fill_method == "ffill":
            df = df.ffill()
        elif self.config.fill_method == "bfill":
            df = df.bfill()

        if self.config.drop_na:
            df = df.dropna()

        logger.info("Preprocessed data: %d rows", len(df))
        return df
