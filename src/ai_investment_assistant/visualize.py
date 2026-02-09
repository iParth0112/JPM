"""Visualization helpers."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """Safe Plotly charting with graceful fallback."""

    def price_chart(self, data: pd.DataFrame, title: str = "Price") -> Optional[object]:
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not installed; skipping chart generation.")
            return None

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
        ))
        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
        return fig

    def equity_curve(self, data: pd.DataFrame, title: str = "Equity Curve") -> Optional[object]:
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not installed; skipping chart generation.")
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["equity_curve"], name="Equity"))
        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Equity")
        return fig
