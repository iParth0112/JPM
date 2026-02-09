"""Macro data fetcher with optional FRED integration."""

from __future__ import annotations

from dataclasses import dataclass
import os
import logging
from typing import Dict

import requests

logger = logging.getLogger(__name__)


@dataclass
class MacroSnapshot:
    description: str
    values: Dict[str, float | str | None]


class MacroDataFetcher:
    """Fetch macro indicators via FRED if API key is provided."""

    FRED_SERIES = {
        "CPIAUCSL": "cpi",
        "FEDFUNDS": "fed_funds_rate",
        "UNRATE": "unemployment_rate",
    }

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def snapshot(self) -> MacroSnapshot:
        if not self.api_key:
            return MacroSnapshot(description="Macro snapshot placeholder (no FRED_API_KEY set)", values={})

        values: Dict[str, float | str | None] = {}
        for series_id, label in self.FRED_SERIES.items():
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                obs = data.get("observations", [])
                value = obs[0]["value"] if obs else None
                values[label] = float(value) if value not in (None, ".") else None
            except Exception as exc:
                logger.warning("Macro fetch failed for %s: %s", series_id, exc)
                values[label] = None

        return MacroSnapshot(description="FRED macro snapshot", values=values)
