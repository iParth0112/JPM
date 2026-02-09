"""Simple backtesting engine."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import RiskConfig, DEFAULT_RISK


@dataclass
class BacktestResult:
    data: pd.DataFrame
    metrics: dict


class BacktestEngine:
    """Vectorized backtest for signal-driven strategies."""

    def __init__(self, risk: RiskConfig | None = None) -> None:
        self.risk = risk or DEFAULT_RISK

    def run(self, data: pd.DataFrame) -> BacktestResult:
        df = data.copy()
        df["returns"] = df["close"].pct_change().fillna(0)
        df["position"] = df["signal"].shift().fillna(0)
        df["strategy_returns"] = df["position"] * df["returns"]
        df["strategy_returns"] -= self.risk.transaction_cost_pct * df["position"].abs()

        df["equity_curve"] = (1 + df["strategy_returns"]).cumprod()
        metrics = self._metrics(df)
        return BacktestResult(data=df, metrics=metrics)

    @staticmethod
    def _metrics(df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        total_return = df["equity_curve"].iloc[-1] - 1
        periods = len(df)
        cagr = (df["equity_curve"].iloc[-1]) ** (252 / periods) - 1 if periods > 0 else 0
        sharpe = 0.0
        if df["strategy_returns"].std() != 0:
            sharpe = math.sqrt(252) * df["strategy_returns"].mean() / df["strategy_returns"].std()
        drawdown = (df["equity_curve"] / df["equity_curve"].cummax()) - 1
        max_drawdown = drawdown.min()
        win_rate = (df["strategy_returns"] > 0).mean()
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
        }
