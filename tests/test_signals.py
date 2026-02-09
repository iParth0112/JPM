from ai_investment_assistant.signals import SignalGenerator
from ai_investment_assistant.technicals import TechnicalIndicators


def test_signal_generation(synthetic_data):
    indicators = TechnicalIndicators()
    data = indicators.add_all(synthetic_data)
    gen = SignalGenerator()
    data = gen.generate(data)
    assert "signal" in data.columns
    latest = gen.latest_signal(data)
    assert latest.action in {"BUY", "SELL", "HOLD"}
