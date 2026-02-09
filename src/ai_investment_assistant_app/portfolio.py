"""Portfolio simulator with PnL, exposure, allocation."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from .fx import fx_rates_usd, convert_to_target
from .data_loader import fetch_market_data


def init_portfolio() -> None:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []


def add_position(symbol: str, quantity: float, price: float, currency: str) -> None:
    init_portfolio()
    st.session_state.portfolio.append({
        "symbol": symbol,
        "quantity": float(quantity),
        "entry_price": float(price),
        "currency": currency,
    })


def portfolio_df() -> pd.DataFrame:
    init_portfolio()
    if not st.session_state.portfolio:
        return pd.DataFrame(columns=["symbol", "quantity", "entry_price", "currency"])
    return pd.DataFrame(st.session_state.portfolio)


def latest_price(symbol: str) -> float:
    df = fetch_market_data(symbol, "5d", "1d")
    if df.empty:
        return 0.0
    return float(df["close"].iloc[-1])


def portfolio_snapshot(target_ccy: str) -> pd.DataFrame:
    df = portfolio_df()
    if df.empty:
        return df
    rates = fx_rates_usd()
    rows = []
    for _, row in df.iterrows():
        price_now = latest_price(row["symbol"])
        mv_local = price_now * row["quantity"]
        mv = convert_to_target(mv_local, row["currency"], target_ccy, rates)
        cost_local = row["entry_price"] * row["quantity"]
        cost = convert_to_target(cost_local, row["currency"], target_ccy, rates)
        pnl = mv - cost
        rows.append({
            "symbol": row["symbol"],
            "quantity": row["quantity"],
            "entry_price": row["entry_price"],
            "price_now": price_now,
            "market_value": mv,
            "cost": cost,
            "pnl": pnl,
        })
    return pd.DataFrame(rows)


def portfolio_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"positions": 0, "total_value": 0.0, "total_pnl": 0.0}
    total_value = df["market_value"].sum()
    total_pnl = df["pnl"].sum()
    return {"positions": len(df), "total_value": float(total_value), "total_pnl": float(total_pnl)}


def allocation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    alloc = df[["symbol", "market_value"]].copy()
    alloc["weight"] = alloc["market_value"] / alloc["market_value"].sum()
    return alloc
