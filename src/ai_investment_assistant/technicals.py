"""Technical indicator calculations."""

from __future__ import annotations

import pandas as pd

from .config import IndicatorConfig, DEFAULT_INDICATORS


class TechnicalIndicators:
    """Compute common technical indicators."""

    def __init__(self, config: IndicatorConfig | None = None) -> None:
        self.config = config or DEFAULT_INDICATORS

    def add_all(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        close = df["close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.astype(float)

        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        df["sma_fast"] = close.rolling(self.config.sma_fast).mean()
        df["sma_slow"] = close.rolling(self.config.sma_slow).mean()

        df["ema_fast"] = close.ewm(span=self.config.macd_fast, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.config.macd_slow, adjust=False).mean()
        df["macd"] = df["ema_fast"] - df["ema_slow"]
        df["macd_signal"] = df["macd"].ewm(span=self.config.macd_signal, adjust=False).mean()

        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(self.config.rsi_period).mean()
        roll_down = down.rolling(self.config.rsi_period).mean()
        rs = roll_up / roll_down.replace(0, pd.NA)
        df["rsi"] = 100 - (100 / (1 + rs))

        rolling = close.rolling(self.config.bollinger_window)
        df["bb_mid"] = rolling.mean()
        df["bb_upper"] = df["bb_mid"] + self.config.bollinger_std * rolling.std()
        df["bb_lower"] = df["bb_mid"] - self.config.bollinger_std * rolling.std()

        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.config.atr_period).mean()

        # ADX (trend strength)
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr_smooth = tr.rolling(self.config.atr_period).sum()
        plus_di = 100 * (plus_dm.rolling(self.config.atr_period).sum() / tr_smooth)
        minus_di = 100 * (minus_dm.rolling(self.config.atr_period).sum() / tr_smooth)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA) * 100
        df["adx"] = dx.rolling(self.config.atr_period).mean()

        # Volume trend
        df["volume_sma"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_sma"]

        # EMA crossover helper
        df["ema_trend"] = (df["ema_fast"] > df["ema_slow"]).astype(int)

        return df
