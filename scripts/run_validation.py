"""Run validation checks for default symbols."""

from __future__ import annotations

import logging

from ai_investment_assistant import InvestmentAssistant
from ai_investment_assistant.config import DEFAULT_SYMBOLS


logging.basicConfig(level=logging.INFO)


def main() -> None:
    assistant = InvestmentAssistant()
    for symbol in DEFAULT_SYMBOLS:
        result = assistant.run(symbol)
        print(symbol, result.validation)


if __name__ == "__main__":
    main()
