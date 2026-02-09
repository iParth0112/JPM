import pandas as pd

from ai_investment_assistant.assistant import InvestmentAssistant
from ai_investment_assistant.data_fetcher import MarketDataFetcher


class StubFetcher(MarketDataFetcher):
    def fetch(self, symbol: str, start=None, end=None) -> pd.DataFrame:
        return MarketDataFetcher.synthetic(periods=120)


def test_assistant_run():
    assistant = InvestmentAssistant(fetcher=StubFetcher())
    result = assistant.run("TEST")
    assert result.signal["action"] in {"BUY", "SELL", "HOLD"}
    assert "passed" in result.validation
    assert "total_return" in result.backtest
