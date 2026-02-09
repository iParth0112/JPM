import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

from ai_investment_assistant import InvestmentAssistant

st.set_page_config(page_title="AI Investment Assistant", layout="wide")
st.title("AI Investment Assistant")

st_autorefresh(interval=60_000, key="autorefresh")

MARKETS = {
    "US": {
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Google (GOOGL)": "GOOGL",
        "Amazon (AMZN)": "AMZN",
        "Nvidia (NVDA)": "NVDA",
        "Tesla (TSLA)": "TSLA",
        "Meta (META)": "META",
        "Netflix (NFLX)": "NFLX",
        "JPMorgan (JPM)": "JPM",
        "Visa (V)": "V",
    },
    "UK": {
        "BP (BP.L)": "BP.L",
        "HSBC (HSBA.L)": "HSBA.L",
        "Unilever (ULVR.L)": "ULVR.L",
        "Vodafone (VOD.L)": "VOD.L",
        "Shell (SHEL.L)": "SHEL.L",
        "AstraZeneca (AZN.L)": "AZN.L",
        "Barclays (BARC.L)": "BARC.L",
        "Tesco (TSCO.L)": "TSCO.L",
    },
    "India": {
        "Reliance (RELIANCE.NS)": "RELIANCE.NS",
        "TCS (TCS.NS)": "TCS.NS",
        "Infosys (INFY.NS)": "INFY.NS",
        "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
        "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
        "HUL (HINDUNILVR.NS)": "HINDUNILVR.NS",
        "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
        "ITC (ITC.NS)": "ITC.NS",
    },
}

WATCHLIST_FILE = "watchlists.json"
SCAN_EXPORT_DIR = "watchlist_exports"
os.makedirs(SCAN_EXPORT_DIR, exist_ok=True)


def load_watchlists():
    if os.path.exists(WATCHLIST_FILE):
        return json.load(open(WATCHLIST_FILE))
    return {"default": ["AAPL", "MSFT", "BP.L", "RELIANCE.NS"]}


def save_watchlists(data):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f, indent=2)


def plot_signals_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price"
    ))
    buys = df[df["signal"].astype(str).str.contains("BUY", na=False)]
    sells = df[df["signal"].astype(str).str.contains("SELL", na=False)]
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["close"],
        mode="markers", name="BUY",
        marker=dict(color="green", size=10, symbol="triangle-up")
    ))
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["close"],
        mode="markers", name="SELL",
        marker=dict(color="red", size=10, symbol="triangle-down")
    ))
    fig.update_layout(title=f"{symbol} Signals", xaxis_title="Date", yaxis_title="Price")
    return fig


def compute_matrix_row(symbol):
    assistant = InvestmentAssistant()
    result = assistant.run(symbol)
    df = result.data
    latest = df.iloc[-1]

    returns = df["close"].pct_change().dropna()
    annual_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    volatility = returns.std() * (252 ** 0.5) * 100
    drawdown = (df["close"] / df["close"].cummax() - 1).min() * 100
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]

    trend = "Up" if latest["sma_fast"] > latest["sma_slow"] else "Down"

    if volatility < 20:
        risk_label = "Low"
    elif volatility < 35:
        risk_label = "Medium"
    else:
        risk_label = "High"

    return {
        "Symbol": symbol,
        "Price": round(float(latest["close"]), 2),
        "1Y Return %": round(annual_return, 2),
        "Volatility %": round(volatility, 2),
        "Max Drawdown %": round(drawdown, 2),
        "Avg Volume": int(avg_volume) if pd.notna(avg_volume) else None,
        "RSI": round(float(latest["rsi"]), 2),
        "Trend": trend,
        "Signal": result.signal["action"],
        "Risk Label": risk_label,
    }


st.sidebar.header("Market & Symbol")
market = st.sidebar.selectbox("Market", list(MARKETS.keys()))
options = MARKETS[market]

search = st.sidebar.text_input("Search company")
filtered = [k for k in options.keys() if search.lower() in k.lower()] or list(options.keys())
selected = st.sidebar.selectbox("Select company", filtered)
symbol = options[selected]

manual = st.sidebar.text_input("Or enter symbol manually", "")
if manual.strip():
    symbol = manual.strip().upper()

run_backtest = st.sidebar.checkbox("Run Backtest", value=False)
show_chart = st.sidebar.checkbox("Show Chart", value=True)

st.subheader(f"Selected Symbol: {symbol}")

if st.button("Analyze"):
    assistant = InvestmentAssistant()
    results = assistant.run(symbol)

    st.subheader("Latest Signal")
    st.json(results.signal)

    st.subheader("Validation")
    st.json(results.validation)

    if show_chart:
        st.subheader("Signal Chart")
        st.plotly_chart(plot_signals_chart(results.data, symbol), use_container_width=True)

    if run_backtest:
        st.subheader("Backtest Metrics")
        st.json(results.backtest)

st.divider()
st.header("Beginner Facts + Calculation Matrix (US/UK/India)")

matrix_symbols = st.text_input("Symbols (comma-separated)", "AAPL, MSFT, BP.L, HSBA.L, RELIANCE.NS, TCS.NS")

if st.button("Generate Matrix"):
    symbols = [s.strip().upper() for s in matrix_symbols.split(",") if s.strip()]
    rows = [compute_matrix_row(sym) for sym in symbols]
    matrix_df = pd.DataFrame(rows)
    st.dataframe(matrix_df)

st.divider()
st.header("Watchlists & Alerts")

watchlists = load_watchlists()
wl_name = st.text_input("Watchlist name", "default")
wl_symbols = st.text_area("Symbols (comma-separated)", ", ".join(watchlists.get(wl_name, [])))

col1, col2 = st.columns(2)

if col1.button("Save Watchlist"):
    watchlists[wl_name] = [s.strip().upper() for s in wl_symbols.split(",") if s.strip()]
    save_watchlists(watchlists)
    st.success("Watchlist saved")

if col2.button("Load Watchlist"):
    symbols = watchlists.get(wl_name, [])
    st.write("Loaded:", symbols)

st.subheader("Run Watchlist Scan")

if st.button("Scan Watchlist"):
    assistant = InvestmentAssistant()
    symbols = [s.strip().upper() for s in wl_symbols.split(",") if s.strip()]
    rows = []

    for sym in symbols:
        try:
            res = assistant.run(sym)
            rows.append({"Symbol": sym, "Signal": res.signal["action"], "Price": res.data["close"].iloc[-1]})
        except Exception:
            rows.append({"Symbol": sym, "Signal": "ERROR", "Price": None})

    scan_df = pd.DataFrame(rows)
    st.dataframe(scan_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{SCAN_EXPORT_DIR}/watchlist_scan_{ts}.csv"
    scan_df.to_csv(csv_path, index=False)
    st.success(f"Saved CSV: {csv_path}")
