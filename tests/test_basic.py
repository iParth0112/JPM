from ai_investment_assistant.technicals import TechnicalIndicators
from ai_investment_assistant.signals import SignalGenerator
from ai_investment_assistant.backtest import BacktestEngine
from ai_investment_assistant.validation import Validator


def test_pipeline(synthetic_data):
    df = TechnicalIndicators().add_all(synthetic_data)
    df = SignalGenerator().generate(df)
    result = BacktestEngine().run(df)
    report = Validator().run_all(df)
    assert "total_return" in result.metrics
    assert report.checks["quality"]["volume_ok"] is True
