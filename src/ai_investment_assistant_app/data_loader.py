"""Data loading with caching and symbol validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
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
def fetch_fundamentals(symbol: str) -> dict:
    """Fetch fundamentals from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        return {
            "info": ticker.info,
            "financials": ticker.financials,
            "balance_sheet": ticker.balance_sheet,
            "cashflow": ticker.cashflow,
        }
    except Exception:
        return {"info": {}, "financials": None, "balance_sheet": None, "cashflow": None}
