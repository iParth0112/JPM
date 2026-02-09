import pandas as pd

from ai_investment_assistant.audit import AuditEvent, AuditLogger


def test_audit_logger(tmp_path):
    logger = AuditLogger(audit_dir=tmp_path)
    path = logger.log_event(AuditEvent(event_type="test", payload={"a": 1}))
    assert path.exists()
    df = pd.DataFrame({"x": [1, 2]})
    path = logger.log_dataframe(df, "sample")
    assert path.exists()
