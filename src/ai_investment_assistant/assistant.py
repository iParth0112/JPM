"""Main assistant orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .audit import AuditEvent, AuditLogger
from .backtest import BacktestEngine
from .config import DEFAULT_END, DEFAULT_START
from .data_fetcher import MarketDataFetcher
from .explain import Explainer
from .macro import MacroDataFetcher
from .ml_model import MLSignalModel
from .news_sentiment import NewsSentimentFetcher
from .preprocess import DataPreprocessor
from .risk import RiskManager
from .signals import SignalGenerator
from .technicals import TechnicalIndicators
from .validation import Validator

logger = logging.getLogger(__name__)


@dataclass
class AssistantResult:
    data: pd.DataFrame
    signal: Dict[str, Any]
    backtest: Dict[str, Any]
    validation: Dict[str, Any]


class InvestmentAssistant:
    """End-to-end assistant combining data, signals, risk, and validation."""

    def __init__(
        self,
        fetcher: Optional[MarketDataFetcher] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        indicators: Optional[TechnicalIndicators] = None,
        signals: Optional[SignalGenerator] = None,
        risk: Optional[RiskManager] = None,
        backtest: Optional[BacktestEngine] = None,
        validator: Optional[Validator] = None,
        ml_model: Optional[MLSignalModel] = None,
        sentiment: Optional[NewsSentimentFetcher] = None,
        macro: Optional[MacroDataFetcher] = None,
        explainer: Optional[Explainer] = None,
        audit: Optional[AuditLogger] = None,
    ) -> None:
        self.fetcher = fetcher or MarketDataFetcher()
        self.preprocessor = preprocessor or DataPreprocessor()
        self.indicators = indicators or TechnicalIndicators()
        self.signals = signals or SignalGenerator()
        self.risk = risk or RiskManager()
        self.backtest = backtest or BacktestEngine()
        self.validator = validator or Validator()
        self.ml_model = ml_model or MLSignalModel()
        self.sentiment = sentiment or NewsSentimentFetcher()
        self.macro = macro or MacroDataFetcher()
        self.explainer = explainer or Explainer()
        self.audit = audit or AuditLogger()

    def run(self, symbol: str, start: str = DEFAULT_START, end: Optional[str] = DEFAULT_END) -> AssistantResult:
        data = self.fetcher.fetch(symbol, start=start, end=end)
        data = self.preprocessor.clean(data)
        data = self.indicators.add_all(data)
        data = self.signals.generate(data)
        data = self.risk.apply(data)

        latest_signal = self.signals.latest_signal(data)
        self.ml_model.fit(data)
        ml_result = self.ml_model.predict_latest(data)
        ml_report = self.ml_model.evaluate_walk_forward(data)
        sentiment = self.sentiment.sentiment_score(symbol)
        macro_snapshot = self.macro.snapshot()
        explanation = self.explainer.explain(
            data.iloc[-1],
            latest_signal.action,
            ml_result,
            sentiment.score,
        )
        validation = self.validator.run_all(data)
        backtest = self.backtest.run(data)

        self.audit.log_event(AuditEvent(event_type="run", payload={
            "symbol": symbol,
            "signal": latest_signal.action,
            "confidence": latest_signal.confidence,
            "validation_passed": validation.passed,
        }))
        self.audit.log_dataframe(backtest.data, f"backtest_{symbol}")

        result = AssistantResult(
            data=data,
            signal={
                "timestamp": latest_signal.timestamp,
                "action": latest_signal.action,
                "confidence": latest_signal.confidence,
                "rationale": latest_signal.rationale,
                "risk": self.risk.decision(float(data["close"].iloc[-1]), float(data["atr"].iloc[-1])).__dict__,
                "ml": None if ml_result is None else ml_result.__dict__,
                "ml_report": ml_report,
                "sentiment": sentiment.__dict__,
                "macro": macro_snapshot.__dict__,
                "explainability": explanation.__dict__,
            },
            backtest=backtest.metrics,
            validation={"passed": validation.passed, **validation.checks},
        )

        logger.info("Run completed for %s", symbol)
        return result
