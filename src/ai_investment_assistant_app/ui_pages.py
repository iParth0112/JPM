"""Streamlit pages for navigation."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from .data_loader import (
    fetch_market_data,
    validate_symbol,
    fetch_fundamentals,
)
from .indicators import add_indicators
from .signals import generate_signals, explain_latest
from .backtest import run_backtest
from .portfolio import add_position, portfolio_snapshot, portfolio_metrics, allocation
from .validate import run_validation, governance_report, audit_log
from .ai_insights import answer_question
from .fx import fx_rates_usd, convert_to_target, fmt_ccy


def plot_candles(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig


def market_dashboard(symbol: str, period: str, interval: str, base_ccy: str, target_ccy: str) -> pd.DataFrame | None:
    st.subheader("Market Dashboard")
    df = fetch_market_data(symbol, period, interval)
    err = validate_symbol(df, symbol)
    if err:
        st.error(err)
        return None
    rates = fx_rates_usd()
    display = df.copy()
    for col in ["open", "high", "low", "close"]:
        display[col] = display[col].map(lambda x: convert_to_target(float(x), base_ccy, target_ccy, rates))
    st.plotly_chart(plot_candles(display, f"{symbol} Price ({target_ccy})"), use_container_width=True)

    latest = display.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Last Price", fmt_ccy(float(latest["close"]), target_ccy))
    col2.metric("Change %", f"{display['close'].pct_change().iloc[-1]*100:.2f}%")
    col3.metric("52W Range", f"{display['low'].min():.2f} - {display['high'].max():.2f}")

    st.dataframe(display.tail(10))
    return df


def technical_analysis(symbol: str, period: str, interval: str, indicators_enabled: dict) -> pd.DataFrame | None:
    st.subheader("Technical Analysis")
    df = fetch_market_data(symbol, period, interval)
    err = validate_symbol(df, symbol)
    if err:
        st.error(err)
        return None
    df = add_indicators(df)
    st.plotly_chart(plot_candles(df, f"{symbol} Candles"), use_container_width=True)
    if indicators_enabled.get("rsi"):
        st.line_chart(df["rsi"])
    if indicators_enabled.get("macd"):
        st.line_chart(df[["macd", "macd_signal"]])
    st.dataframe(df.tail(10))
    return df


def fundamental_analysis(symbol: str) -> None:
    st.subheader("Fundamental Analysis")
    fundamentals = fetch_fundamentals(symbol)
    info = fundamentals.get("info", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PE", info.get("trailingPE", "N/A"))
    col2.metric("ROE", info.get("returnOnEquity", "N/A"))
    col3.metric("Debt/Equity", info.get("debtToEquity", "N/A"))
    col4.metric("Dividend Yield", info.get("dividendYield", "N/A"))
    with st.expander("Financial Statements"):
        st.write(fundamentals.get("financials"))
        st.write(fundamentals.get("balance_sheet"))
        st.write(fundamentals.get("cashflow"))


def signal_engine(df: pd.DataFrame) -> pd.DataFrame | None:
    st.subheader("Signal Engine")
    if df is None or df.empty:
        st.info("Run Technical Analysis first.")
        return None
    df = generate_signals(df)
    explanation = explain_latest(df)
    st.metric("Signal", explanation.action)
    st.metric("Score", explanation.score)
    st.metric("Confidence", f"{explanation.confidence:.2f}")
    st.write(explanation.rationale)
    st.json(explanation.details)
    st.dataframe(df.tail(20))

    audit_log({
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "signal": explanation.action,
        "score": explanation.score,
        "confidence": explanation.confidence,
    })
    return df


def portfolio_simulator(target_ccy: str) -> None:
    st.subheader("Portfolio Simulator")
    with st.expander("Add Position"):
        symbol = st.text_input("Symbol", key="p_symbol")
        quantity = st.number_input("Quantity", min_value=0.0, step=1.0, key="p_qty")
        price = st.number_input("Entry Price", min_value=0.0, step=0.01, key="p_price")
        currency = st.text_input("Currency", value=target_ccy, key="p_ccy")
        if st.button("Add Position"):
            add_position(symbol, quantity, price, currency)
    snap = portfolio_snapshot(target_ccy)
    metrics = portfolio_metrics(snap)
    col1, col2, col3 = st.columns(3)
    col1.metric("Positions", metrics["positions"])
    col2.metric("Total Value", fmt_ccy(metrics["total_value"], target_ccy))
    col3.metric("Total PnL", fmt_ccy(metrics["total_pnl"], target_ccy))
    st.dataframe(snap)
    alloc = allocation(snap)
    if not alloc.empty:
        fig = go.Figure(data=[go.Pie(labels=alloc["symbol"], values=alloc["market_value"])])
        st.plotly_chart(fig, use_container_width=True)


def backtesting(df: pd.DataFrame) -> None:
    st.subheader("Backtesting")
    if df is None or df.empty:
        st.info("Run Signal Engine first.")
        return
    result = run_backtest(df)
    st.json(result["metrics"])
    st.line_chart(result["data"]["equity_curve"])
    st.dataframe(result["data"].tail(20))


def model_validation(df: pd.DataFrame) -> None:
    st.subheader("Model Validation & Governance")
    if df is None or df.empty:
        st.info("Run Signal Engine first.")
        return
    checks = run_validation(df)
    gov = governance_report(df)
    st.json(checks)
    st.json(gov.__dict__)


def ai_insights(df: pd.DataFrame) -> None:
    st.subheader("AI Insights")
    question = st.text_input("Ask a question")
    if question:
        response = answer_question(question)
        st.info(response.text)
    if df is None or df.empty:
        return
    latest = df.iloc[-1]
    st.write(
        f"Signal is based on RSI={latest.get('rsi', None):.2f} and SMA trend."
    )
