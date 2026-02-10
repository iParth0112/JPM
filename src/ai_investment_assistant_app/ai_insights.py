"""AI Insights stub (no external API)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIResponse:
    text: str
    model: str = "stub"


def answer_question(question: str, context: str | None = None) -> AIResponse:
    \"\"\"Rule-based local response (no external API).\"\"\"\n+    if context:\n+        text = f\"Question: {question}\\n\\nSummary: {context}\"\n+    else:\n+        text = f\"Question: {question}\\n\\nSummary: This assistant provides data-driven signals and risk context.\"\n+    return AIResponse(text=text, model=\"local\")\n*** End Patch} 
