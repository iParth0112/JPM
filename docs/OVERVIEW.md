# Governance Overview

This project is structured to demonstrate model governance and validation practices.

Key artifacts:
- Audit logs: `artifacts/audit/audit_log.csv`
- Backtest datasets: `artifacts/audit/backtest_<SYMBOL>.parquet`
- Validation outputs: returned by the assistant and suitable for reporting

Recommended governance workflow:
1. Run the assistant on target symbols.
2. Review validation reports for data quality and drift.
3. Review audit logs and backtest outputs for traceability.
4. Schedule weekly validation via GitHub Actions.
