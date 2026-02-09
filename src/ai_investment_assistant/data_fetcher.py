"""Market data fetching and safety checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import DEFAULT_INTERVAL, DEFAULT_START

logger = logging.getLogger(__name__)


@dataclass
class FetchConfig:
    interval: str = DEFAULT_INTERVAL
    start: str = DEFAULT_START
    end: Optional[str] = None


class MarketDataFetcher:
    """Fetches market data with 1D safety and basic cleaning."""

    def __init__(self, config: Optional[FetchConfig] = None) -> None:
        self.config = config or FetchConfig()
        self._cache: dict[tuple, pd.DataFrame] = {}

    def _normalize(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.get_level_values(-1):
                data = data.xs(symbol, level=-1, axis=1)
            else:
                data = data.droplevel(-1, axis=1)

        data = data.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })

        data.index = pd.to_datetime(data.index)
        data = data.sort_index().loc[~data.index.duplicated(keep="last")]
        data = data[["open", "high", "low", "close", "adj_close", "volume"]]
        data = data.dropna(how="all")
        return data

    def fetch(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        interval = self.config.interval
        if interval != "1d":
            raise ValueError("Only daily (1d) interval is supported for governance safety.")

        start = start or self.config.start
        end = end or self.config.end

        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required. Install with `pip install yfinance`." ) from exc

        cache_key = (symbol, start, end, interval)
        if cache_key in self._cache:
            logger.info("Using cached data for %s", symbol)
            return self._cache[cache_key].copy()

        logger.info("Fetching data for %s from %s to %s", symbol, start, end)
        data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False)
        if data.empty:
            raise ValueError(f"No data returned for symbol {symbol}.")

        data = self._normalize(data, symbol)
        self._cache[cache_key] = data.copy()
        return data

    def fetch_many(self, symbols: list[str], start: Optional[str] = None, end: Optional[str] = None) -> dict[str, pd.DataFrame]:
        interval = self.config.interval
        if interval != "1d":
            raise ValueError("Only daily (1d) interval is supported for governance safety.")

        start = start or self.config.start
        end = end or self.config.end

        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required. Install with `pip install yfinance`." ) from exc

        symbols = [s for s in symbols if s]
        if not symbols:
            return {}

        # Use cached entries when available
        remaining = []
        results: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            cache_key = (symbol, start, end, interval)
            if cache_key in self._cache:
                results[symbol] = self._cache[cache_key].copy()
            else:
                remaining.append(symbol)

        if remaining:
            logger.info("Batch fetching data for %d symbols", len(remaining))
            data = yf.download(
                tickers=" ".join(remaining),
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
            )
            if data.empty:
                return results

            for symbol in remaining:
                try:
                    symbol_df = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
                    normalized = self._normalize(symbol_df, symbol)
                    results[symbol] = normalized
                    cache_key = (symbol, start, end, interval)
                    self._cache[cache_key] = normalized.copy()
                except Exception:
                    continue

        return results

    @staticmethod
    def synthetic(symbol: str = "SYN", periods: int = 252) -> pd.DataFrame:
        """Create deterministic synthetic data for tests."""
        idx = pd.date_range("2020-01-01", periods=periods, freq="B")
        prices = pd.Series(range(periods), index=idx).astype(float) + 100.0
        data = pd.DataFrame({
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "adj_close": prices,
            "volume": 1_000_000,
        }, index=idx)
        data.index.name = "date"
        return data
