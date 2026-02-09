from ai_investment_assistant.technicals import TechnicalIndicators


def test_indicators_added(synthetic_data):
    indicators = TechnicalIndicators()
    out = indicators.add_all(synthetic_data)
    assert "sma_fast" in out.columns
    assert "rsi" in out.columns
    assert "atr" in out.columns
