"""Technical indicator pipeline wrapper."""

from __future__ import annotations

import pandas as pd

from ai_investment_assistant.technicals import TechnicalIndicators


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    engine = TechnicalIndicators()
    return engine.add_all(df)
