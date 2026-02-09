"""AI Investment Assistant package."""

from .assistant import InvestmentAssistant
from .preprocess import DataPreprocessor
from .ml_model import MLSignalModel
from .explain import Explainer
from .news_sentiment import NewsSentimentFetcher
from .macro import MacroDataFetcher
from .retrain import ReTrainer

__all__ = [
    "InvestmentAssistant",
    "DataPreprocessor",
    "MLSignalModel",
    "Explainer",
    "NewsSentimentFetcher",
    "MacroDataFetcher",
    "ReTrainer",
]
__version__ = "0.1.0"
