"""AI Insights stub (LLM optional)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIResponse:
    text: str
    model: str = "stub"


def answer_question(question: str) -> AIResponse:
    # Stub: Replace with LLM call if API key is configured
    return AIResponse(text=f"Stub response: {question}")
