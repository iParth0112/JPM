"""Signal engine with explanations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_investment_assistant.signals import SignalGenerator


@dataclass
class SignalExplanation:
    action: str
    confidence: float
    rationale: str
    details: dict


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    gen = SignalGenerator()
    return gen.generate(df)


def explain_latest(df: pd.DataFrame) -> SignalExplanation:
    gen = SignalGenerator()
    signal = gen.latest_signal(df)
    latest = df.iloc[-1]
    details = {
        "rsi": float(latest.get("rsi", 0.0)) if pd.notna(latest.get("rsi", None)) else None,
        "macd": float(latest.get("macd", 0.0)) if pd.notna(latest.get("macd", None)) else None,
        "sma_fast": float(latest.get("sma_fast", 0.0)) if pd.notna(latest.get("sma_fast", None)) else None,
        "sma_slow": float(latest.get("sma_slow", 0.0)) if pd.notna(latest.get("sma_slow", None)) else None,
    }
    return SignalExplanation(
        action=signal.action,
        confidence=signal.confidence,
        rationale=signal.rationale,
        details=details,
    )
