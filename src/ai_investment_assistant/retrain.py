"""Continuous learning utilities."""

from __future__ import annotations

import logging

import pandas as pd

from .ml_model import MLSignalModel

logger = logging.getLogger(__name__)


class ReTrainer:
    """Simple retraining hook based on drift signals."""

    def __init__(self, model: MLSignalModel) -> None:
        self.model = model

    def retrain(self, data: pd.DataFrame) -> bool:
        before = self.model.is_trained
        self.model.fit(data)
        after = self.model.is_trained
        logger.info("Retrain invoked. was_trained=%s now_trained=%s", before, after)
        return after
