"""News sentiment integration (optional API)."""

from __future__ import annotations

from dataclasses import dataclass
import os
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    score: float
    summary: str


class NewsSentimentFetcher:
    """Fetch sentiment from NewsAPI if API key is set."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")

    def sentiment_score(self, symbol: str) -> SentimentResult:
        if not self.api_key:
            return SentimentResult(score=0.0, summary=f"No NEWSAPI_KEY set; neutral sentiment for {symbol}.")

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": symbol,
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": self.api_key,
                "language": "en",
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])

            pos_words = ["beat", "growth", "upgrade", "profit", "surge", "strong", "record"]
            neg_words = ["miss", "downgrade", "loss", "drop", "weak", "lawsuit", "recall"]

            score = 0
            for a in articles:
                text = (a.get("title", "") + " " + a.get("description", "")).lower()
                score += sum(w in text for w in pos_words)
                score -= sum(w in text for w in neg_words)

            normalized = max(min(score / 10, 1), -1)
            return SentimentResult(score=normalized, summary=f"News sentiment score {normalized:.2f} based on {len(articles)} articles.")

        except Exception as exc:
            logger.warning("News sentiment fetch failed: %s", exc)
            return SentimentResult(score=0.0, summary=f"News sentiment unavailable for {symbol}; neutral.")
