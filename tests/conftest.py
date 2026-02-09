import pandas as pd
import pytest

from ai_investment_assistant.data_fetcher import MarketDataFetcher


@pytest.fixture()
def synthetic_data() -> pd.DataFrame:
    return MarketDataFetcher.synthetic(periods=200)
