"""Cross-symbol comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_fetcher import MarketDataFetcher
from .technicals import TechnicalIndicators
from .signals import SignalGenerator


@dataclass
class ComparisonResult:
    table: pd.DataFrame


class Comparator:
    """Compare multiple symbols by recent performance and signals."""

    def __init__(self) -> None:
        self.fetcher = MarketDataFetcher()
        self.indicators = TechnicalIndicators()
        self.signals = SignalGenerator()

    def compare(self, symbols: list[str]) -> ComparisonResult:
        rows = []
        for symbol in symbols:
            data = self.fetcher.fetch(symbol)
            data = self.indicators.add_all(data)
            data = self.signals.generate(data)
            latest = self.signals.latest_signal(data)
            recent_return = data["close"].pct_change().tail(20).mean() * 252
            rows.append({
                "symbol": symbol,
                "latest_signal": latest.action,
                "signal_confidence": latest.confidence,
                "annualized_recent_return": float(recent_return),
            })
        table = pd.DataFrame(rows).sort_values("annualized_recent_return", ascending=False)
        return ComparisonResult(table=table)
