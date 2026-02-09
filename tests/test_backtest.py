from ai_investment_assistant.backtest import BacktestEngine
from ai_investment_assistant.signals import SignalGenerator
from ai_investment_assistant.technicals import TechnicalIndicators


def test_backtest_metrics(synthetic_data):
    data = TechnicalIndicators().add_all(synthetic_data)
    data = SignalGenerator().generate(data)
    result = BacktestEngine().run(data)
    assert "total_return" in result.metrics
    assert "sharpe" in result.metrics
