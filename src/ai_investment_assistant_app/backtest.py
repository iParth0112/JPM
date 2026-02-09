"""Backtest helper module with configurable dates."""

from __future__ import annotations

import pandas as pd

from ai_investment_assistant.backtest import BacktestEngine


def run_backtest(df: pd.DataFrame) -> dict:
    engine = BacktestEngine()
    result = engine.run(df)
    return {"metrics": result.metrics, "data": result.data}
