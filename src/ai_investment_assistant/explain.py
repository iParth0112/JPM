"""Explainability helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ml_model import MLResult


@dataclass
class Explanation:
    headline: str
    details: dict


class Explainer:
    """Generates human-readable explanations for signals."""

    def explain(self, row: pd.Series, signal: str, ml: MLResult | None, sentiment: float | None) -> Explanation:
        reasons = {}
        if "sma_fast" in row and "sma_slow" in row:
            reasons["trend"] = "Up" if row["sma_fast"] > row["sma_slow"] else "Down"
        if "rsi" in row:
            reasons["rsi"] = float(row["rsi"])
        if "macd" in row and "macd_signal" in row:
            reasons["macd"] = float(row["macd"] - row["macd_signal"])
        if sentiment is not None:
            reasons["sentiment"] = sentiment
        if ml is not None:
            reasons["ml_prediction"] = ml.label
            reasons["ml_probability_up"] = ml.probability_up

        headline = f"Signal: {signal}"
        return Explanation(headline=headline, details=reasons)
