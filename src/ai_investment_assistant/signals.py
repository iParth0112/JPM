"""Signal generation logic."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    timestamp: pd.Timestamp
    action: str
    confidence: float
    rationale: str


class SignalGenerator:
    """Generates trade signals from indicators."""

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        trend_up = df["sma_fast"] > df["sma_slow"]
        trend_down = df["sma_fast"] < df["sma_slow"]
        ema_up = df["ema_fast"] > df["ema_slow"]
        ema_down = df["ema_fast"] < df["ema_slow"]

        strong_trend = df["adx"] > 20
        healthy_volume = df["volume_ratio"] > 1.0

        buy = trend_up & ema_up & (df["rsi"] < 70) & strong_trend & healthy_volume
        sell = trend_down & ema_down & ((df["rsi"] > 70) | (df["macd"] < df["macd_signal"]))

        df["signal"] = 0
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        macd_strength = (df["macd"] - df["macd_signal"]).abs().fillna(0)
        adx_strength = df["adx"].fillna(0)
        df["signal_confidence"] = (macd_strength + adx_strength / 50).clip(0, 1)

        return df

    def latest_signal(self, data: pd.DataFrame) -> Signal:
        latest = data.dropna(subset=["signal"]).iloc[-1]
        action = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(int(latest["signal"]), "HOLD")
        rationale = self._explain(latest)
        return Signal(
            timestamp=latest.name,
            action=action,
            confidence=float(latest.get("signal_confidence", 0.0)),
            rationale=rationale,
        )

    @staticmethod
    def _explain(row: pd.Series) -> str:
        if row.get("signal", 0) == 1:
            return "Uptrend confirmed by SMA/EMA with strong ADX and healthy volume."
        if row.get("signal", 0) == -1:
            return "Downtrend or momentum weakening; defensive signal."
        return "No strong trend signal; hold position."
