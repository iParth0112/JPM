"""Simple portfolio simulator stored in session state."""

from __future__ import annotations

import streamlit as st
import pandas as pd


def init_portfolio() -> None:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []


def add_position(symbol: str, quantity: float, price: float, currency: str) -> None:
    init_portfolio()
    st.session_state.portfolio.append({
        "symbol": symbol,
        "quantity": float(quantity),
        "price": float(price),
        "currency": currency,
    })


def portfolio_df() -> pd.DataFrame:
    init_portfolio()
    if not st.session_state.portfolio:
        return pd.DataFrame(columns=["symbol", "quantity", "price", "currency"])
    return pd.DataFrame(st.session_state.portfolio)


def portfolio_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"positions": 0, "total_value": 0.0}
    total_value = (df["quantity"] * df["price"]).sum()
    return {"positions": len(df), "total_value": float(total_value)}
