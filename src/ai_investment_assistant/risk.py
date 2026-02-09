"""Risk management utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import RiskConfig, DEFAULT_RISK


@dataclass
class RiskDecision:
    position_size: float
    stop_loss: float
    take_profit: float


class RiskManager:
    """Applies position sizing and risk thresholds."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or DEFAULT_RISK

    def size_position(self, price: float, atr: float | None = None) -> float:
        try:
            price_val = float(price)
        except Exception:
            return 0.0
        if price_val <= 0:
            return 0.0
        atr_val = None
        try:
            atr_val = float(atr) if atr is not None else None
        except Exception:
            atr_val = None
        volatility = atr_val if atr_val and atr_val > 0 else price_val * self.config.stop_loss_pct
        position = self.config.risk_per_trade * price_val / volatility
        return min(position, self.config.max_position)

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["position_size"] = df.apply(
            lambda row: self.size_position(row["close"], row.get("atr")), axis=1
        )
        df["stop_loss"] = df["close"] * (1 - self.config.stop_loss_pct)
        df["take_profit"] = df["close"] * (1 + self.config.take_profit_pct)
        return df

    def decision(self, price: float, atr: float | None = None) -> RiskDecision:
        size = self.size_position(price, atr)
        return RiskDecision(
            position_size=size,
            stop_loss=price * (1 - self.config.stop_loss_pct),
            take_profit=price * (1 + self.config.take_profit_pct),
        )
