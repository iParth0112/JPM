"""Run a quick demo of the investment assistant."""

from __future__ import annotations

import argparse
import logging

from ai_investment_assistant import InvestmentAssistant


logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    args = parser.parse_args()

    assistant = InvestmentAssistant()
    result = assistant.run(args.symbol)

    print("Signal:")
    print(result.signal)
    print("Validation:")
    print(result.validation)
    print("Backtest metrics:")
    print(result.backtest)


if __name__ == "__main__":
    main()
