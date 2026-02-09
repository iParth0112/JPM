from ai_investment_assistant.risk import RiskManager


def test_position_sizing():
    risk = RiskManager()
    size = risk.size_position(price=100, atr=2)
    assert size > 0
    assert size <= risk.config.max_position
