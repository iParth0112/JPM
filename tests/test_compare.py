import pandas as pd

from ai_investment_assistant.compare import Comparator
from ai_investment_assistant.data_fetcher import MarketDataFetcher


def test_compare_symbols(monkeypatch):
    def fake_fetch(self, symbol, start=None, end=None) -> pd.DataFrame:
        return MarketDataFetcher.synthetic(periods=80)

    monkeypatch.setattr("ai_investment_assistant.compare.MarketDataFetcher.fetch", fake_fetch)
    comparator = Comparator()
    result = comparator.compare(["AAA", "BBB"])
    assert len(result.table) == 2
