from ai_investment_assistant.data_fetcher import MarketDataFetcher


def test_synthetic_data_shape():
    data = MarketDataFetcher.synthetic(periods=50)
    assert len(data) == 50
    assert {"open", "high", "low", "close", "adj_close", "volume"}.issubset(data.columns)
