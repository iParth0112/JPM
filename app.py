"""Streamlit app entrypoint."""

import streamlit as st

from ai_investment_assistant_app.data_loader import ASSET_CLASS_PRESETS, normalize_symbol
from ai_investment_assistant_app.ui_pages import (
    market_dashboard,
    technical_analysis,
    fundamental_analysis,
    signal_engine,
    portfolio_simulator,
    backtesting,
    model_validation,
    ai_insights,
)

st.set_page_config(page_title="AI Investment Assistant", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Market Dashboard",
        "Technical Analysis",
        "Fundamental Analysis",
        "Signal Engine",
        "Portfolio Simulator",
        "Backtesting",
        "Model Validation & Governance",
        "AI Insights",
        "Settings",
    ],
)

st.sidebar.title("Market Data")
asset_class = st.sidebar.selectbox("Asset Class", list(ASSET_CLASS_PRESETS.keys()))
symbol_input = st.sidebar.text_input("Symbol", ASSET_CLASS_PRESETS[asset_class]["example"])
symbol = normalize_symbol(symbol_input)
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

base_ccy = st.sidebar.selectbox("Local Currency", ["USD", "GBP", "EUR"], index=0)
target_ccy = st.sidebar.selectbox("Normalize To", ["USD", "GBP", "EUR"], index=0)

indicators_enabled = {
    "rsi": st.sidebar.checkbox("RSI", value=True),
}

if page == "Market Dashboard":
    market_dashboard(symbol, period, interval, base_ccy, target_ccy)
elif page == "Technical Analysis":
    st.session_state["df"] = technical_analysis(symbol, period, interval, indicators_enabled)
elif page == "Fundamental Analysis":
    fundamental_analysis(symbol)
elif page == "Signal Engine":
    df = st.session_state.get("df")
    st.session_state["signals"] = signal_engine(df)
elif page == "Portfolio Simulator":
    portfolio_simulator(target_ccy)
elif page == "Backtesting":
    df = st.session_state.get("signals") or st.session_state.get("df")
    backtesting(df)
elif page == "Model Validation & Governance":
    df = st.session_state.get("signals") or st.session_state.get("df")
    model_validation(df)
elif page == "AI Insights":
    df = st.session_state.get("signals") or st.session_state.get("df")
    ai_insights(df)
elif page == "Settings":
    st.subheader("Settings")
    st.write("Add API keys and preferences here.")
