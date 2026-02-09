"""Validation and governance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ai_investment_assistant.validation import Validator

AUDIT_PATH = Path("artifacts/audit/audit_log_app.csv")
AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class GovernanceReport:
    model_version: str
    last_run: str
    signal_distribution: dict
    errors: int


def run_validation(df: pd.DataFrame) -> dict:
    validator = Validator()
    report = validator.run_all(df)
    return report.checks | {"passed": report.passed}


def governance_report(df: pd.DataFrame, model_version: str = "0.1.0") -> GovernanceReport:
    signals = df["signal"].value_counts().to_dict() if "signal" in df else {}
    return GovernanceReport(
        model_version=model_version,
        last_run=datetime.utcnow().isoformat(),
        signal_distribution=signals,
        errors=0,
    )


def audit_log(event: dict) -> None:
    df = pd.DataFrame([event])
    header = not AUDIT_PATH.exists()
    df.to_csv(AUDIT_PATH, mode="a", header=header, index=False)
