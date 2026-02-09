"""Model validation and data quality tests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats

from .config import ValidationConfig, DEFAULT_VALIDATION


@dataclass
class ValidationReport:
    passed: bool
    checks: dict


class Validator:
    """Runs stability, drift, and quality tests."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or DEFAULT_VALIDATION

    def data_quality(self, data: pd.DataFrame) -> dict:
        missing_pct = data.isna().mean().max()
        zero_volume_pct = (data["volume"] == 0).mean() if "volume" in data else 0
        return {
            "missing_pct": float(missing_pct),
            "zero_volume_pct": float(zero_volume_pct),
            "missing_ok": missing_pct <= self.config.max_missing_pct,
            "volume_ok": zero_volume_pct <= self.config.max_zero_volume_pct,
        }

    def stability(self, data: pd.DataFrame) -> dict:
        window = self.config.stability_window
        rolling_mean = data["close"].rolling(window).mean()
        rolling_std = data["close"].rolling(window).std()
        latest_mean = rolling_mean.iloc[-1]
        latest_std = rolling_std.iloc[-1]
        return {
            "rolling_mean": float(latest_mean) if pd.notna(latest_mean) else None,
            "rolling_std": float(latest_std) if pd.notna(latest_std) else None,
        }

    def drift(self, data: pd.DataFrame) -> dict:
        midpoint = len(data) // 2
        first = data["close"].iloc[:midpoint].dropna()
        second = data["close"].iloc[midpoint:].dropna()
        if len(first) < 10 or len(second) < 10:
            return {"pvalue": None, "drift": False}
        stat, pvalue = stats.ks_2samp(first, second)
        drift = pvalue < self.config.drift_pvalue_threshold
        return {"pvalue": float(pvalue), "drift": bool(drift)}

    def run_all(self, data: pd.DataFrame) -> ValidationReport:
        quality = self.data_quality(data)
        stability = self.stability(data)
        drift = self.drift(data)
        passed = quality["missing_ok"] and quality["volume_ok"] and not drift["drift"]
        return ValidationReport(passed=passed, checks={
            "quality": quality,
            "stability": stability,
            "drift": drift,
        })
