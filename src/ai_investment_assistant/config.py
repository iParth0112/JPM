"""Configuration constants for the AI Investment Assistant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_NAME = "ai_investment_assistant"

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "BP.L"]
DEFAULT_START = "2018-01-01"
DEFAULT_END = None
DEFAULT_INTERVAL = "1d"

DATA_PROVIDER = "yfinance"

BASE_DIR = Path.cwd()
ARTIFACTS_DIR = BASE_DIR / "artifacts"
AUDIT_DIR = ARTIFACTS_DIR / "audit"
LOG_DIR = ARTIFACTS_DIR / "logs"
REPORT_DIR = ARTIFACTS_DIR / "reports"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    max_position: float = 0.2
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    transaction_cost_pct: float = 0.001


@dataclass(frozen=True)
class IndicatorConfig:
    sma_fast: int = 20
    sma_slow: int = 50
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14


@dataclass(frozen=True)
class ValidationConfig:
    max_missing_pct: float = 0.02
    max_zero_volume_pct: float = 0.05
    stability_window: int = 60
    drift_pvalue_threshold: float = 0.05


DEFAULT_RISK = RiskConfig()
DEFAULT_INDICATORS = IndicatorConfig()
DEFAULT_VALIDATION = ValidationConfig()
