"""Data loading with caching and symbol validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class DataRequest:
    symbol: str
    period: str = "1y"
    interval: str = "1d"


ASSET_CLASS_PRESETS = {
    "Stocks": {"example": "AAPL"},
    "ETFs": {"example": "SPY"},
    "Crypto": {"example": "BTC-USD"},
    "FX": {"example": "EURUSD=X"},
}


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


@st.cache_data(show_spinner=False)
def fetch_market_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    logger.info("Fetching %s period=%s interval=%s", symbol, period, interval)
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
    if df.empty:
        # Fallback: try Yahoo search to resolve company name to ticker
        matches = search_symbols(symbol)
        if matches:
            resolved = matches[0].get("symbol", "")
            if resolved and resolved.upper() != symbol.upper():
                logger.info("Resolved symbol %s -> %s via search", symbol, resolved)
                df = yf.download(resolved, period=period, interval=interval, auto_adjust=False)
        if df.empty:
            return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"adj close": "adj_close"})
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def fetch_batch(symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = fetch_market_data(symbol, period, interval)
        if not df.empty:
            data[symbol] = df
    return data


def validate_symbol(df: pd.DataFrame, symbol: str) -> Optional[str]:
    if df is None or df.empty:
        return f"No data returned for {symbol}."
    return None


@st.cache_data(show_spinner=False)
def search_symbols(query: str) -> list[dict]:
    """Yahoo Finance search API."""
    query = query.strip()
    if not query:
        return []
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 10, "newsCount": 0}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("quotes", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Symbol search failed: %s", exc)
        return []
