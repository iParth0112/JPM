"""FX utilities for currency normalization."""

from __future__ import annotations

from typing import Dict

import requests
import streamlit as st


@st.cache_data(show_spinner=False)
def fx_rates_usd() -> Dict[str, float]:
    """Fetch USD FX rates (cached)."""
    resp = requests.get("https://api.exchangerate.host/latest?base=USD", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("rates", {})


def convert_to_target(amount: float, from_ccy: str, target_ccy: str, rates: Dict[str, float]) -> float:
    if from_ccy == target_ccy:
        return float(amount)
    if from_ccy == "USD":
        rate = rates.get(target_ccy)
        return float(amount) * float(rate) if rate else float(amount)
    rate_from = rates.get(from_ccy)
    if not rate_from:
        return float(amount)
    usd = float(amount) / float(rate_from)
    if target_ccy == "USD":
        return usd
    rate_to = rates.get(target_ccy)
    return usd * float(rate_to) if rate_to else usd


def fmt_ccy(amount: float, ccy: str) -> str:
    symbol = {"USD": "$", "GBP": "£", "EUR": "€", "INR": "₹"}.get(ccy, "")
    return f"{symbol}{amount:,.2f}" if symbol else f"{amount:,.2f} {ccy}"
