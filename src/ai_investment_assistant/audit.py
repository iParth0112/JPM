"""Audit logging utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import AUDIT_DIR

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    event_type: str
    payload: Dict[str, Any]


class AuditLogger:
    """Writes audit logs to CSV/Parquet."""

    def __init__(self, audit_dir: Path | None = None) -> None:
        self.audit_dir = audit_dir or AUDIT_DIR
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: AuditEvent) -> Path:
        path = self.audit_dir / "audit_log.csv"
        row = {"event_type": event.event_type, **event.payload}
        df = pd.DataFrame([row])
        header = not path.exists()
        df.to_csv(path, mode="a", header=header, index=False)
        logger.info("Audit event logged: %s", event.event_type)
        return path

    def log_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        path = self.audit_dir / f"{name}.parquet"
        df.to_parquet(path, index=True)
        logger.info("Audit dataframe stored: %s", path)
        return path
