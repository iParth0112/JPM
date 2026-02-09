"""Signal engine with explanations and scoring."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SignalExplanation:
    action: str
    score: int
    confidence: float
    rationale: str
    details: dict


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal_score"] = 0

    # MA crossover
    out.loc[out["sma_fast"] > out["sma_slow"], "signal_score"] += 1
    out.loc[out["sma_fast"] < out["sma_slow"], "signal_score"] -= 1

    # RSI
    out.loc[out["rsi"] < 30, "signal_score"] += 1
    out.loc[out["rsi"] > 70, "signal_score"] -= 1

    # MACD
    out.loc[out["macd"] > out["macd_signal"], "signal_score"] += 1
    out.loc[out["macd"] < out["macd_signal"], "signal_score"] -= 1

    # Bollinger Bands
    out.loc[out["close"] < out["bb_lower"], "signal_score"] += 1
    out.loc[out["close"] > out["bb_upper"], "signal_score"] -= 1

    out["signal"] = "HOLD"
    out.loc[out["signal_score"] >= 2, "signal"] = "BUY"
    out.loc[out["signal_score"] <= -2, "signal"] = "SELL"

    return out


def explain_latest(df: pd.DataFrame) -> SignalExplanation:
    latest = df.iloc[-1]
    action = latest.get("signal", "HOLD")
    score = int(latest.get("signal_score", 0))
    confidence = min(max(abs(score) / 4, 0), 1)

    reasons = []
    if latest.get("rsi", 50) < 30:
        reasons.append("RSI oversold")
    if latest.get("rsi", 50) > 70:
        reasons.append("RSI overbought")
    if latest.get("sma_fast", 0) > latest.get("sma_slow", 0):
        reasons.append("SMA bullish crossover")
    if latest.get("sma_fast", 0) < latest.get("sma_slow", 0):
        reasons.append("SMA bearish crossover")
    if latest.get("macd", 0) > latest.get("macd_signal", 0):
        reasons.append("MACD bullish")
    if latest.get("macd", 0) < latest.get("macd_signal", 0):
        reasons.append("MACD bearish")
    if latest.get("close", 0) < latest.get("bb_lower", 0):
        reasons.append("Price below BB lower")
    if latest.get("close", 0) > latest.get("bb_upper", 0):
        reasons.append("Price above BB upper")

    rationale = "; ".join(reasons) if reasons else "No strong confluence"

    details = {
        "rsi": float(latest.get("rsi", 0.0)) if pd.notna(latest.get("rsi", None)) else None,
        "macd": float(latest.get("macd", 0.0)) if pd.notna(latest.get("macd", None)) else None,
        "sma_fast": float(latest.get("sma_fast", 0.0)) if pd.notna(latest.get("sma_fast", None)) else None,
        "sma_slow": float(latest.get("sma_slow", 0.0)) if pd.notna(latest.get("sma_slow", None)) else None,
    }

    return SignalExplanation(
        action=action,
        score=score,
        confidence=confidence,
        rationale=rationale,
        details=details,
    )
