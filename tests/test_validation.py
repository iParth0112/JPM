from ai_investment_assistant.validation import Validator


def test_validation_passes(synthetic_data):
    validator = Validator()
    report = validator.run_all(synthetic_data)
    assert "quality" in report.checks
    assert report.checks["quality"]["missing_ok"]
