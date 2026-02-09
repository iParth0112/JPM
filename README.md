# AI Investment Assistant 🤖📈

A comprehensive personal AI investment assistant for analyzing stocks, generating trading signals, backtesting strategies, and managing risk. Built for Google Colab with modular architecture and broker API integration capabilities.

## ⚠️ DISCLAIMER

**This system is for ANALYSIS and EDUCATIONAL PURPOSES ONLY.**

- Does NOT execute trades automatically
- Not financial advice
- Past performance does not guarantee future results
- Always do your own research
- Consult with a financial advisor before making investment decisions
- Trade at your own risk

## 🌟 Features

### Core Functionality
- ✅ **Real-time Market Data** - Fetch live stock data via yfinance
- ✅ **Technical Analysis** - Calculate 10+ indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- ✅ **Multi-Factor Signals** - Generate BUY/SELL/HOLD signals using multiple confirmations
- ✅ **Risk Management** - Position sizing, stop loss, take profit calculations
- ✅ **Backtesting** - Test strategies on historical data
- ✅ **Visualization** - Interactive charts with Plotly
- ✅ **Alert System** - Configurable alerts (extensible to Telegram/Email)
- ✅ **Multi-Market Support** - US stocks, Indian stocks (NSE/BSE), and global markets

### Advanced Features
- 📊 Portfolio comparison across multiple stocks
- 🔄 Intraday and long-term analysis
- 📈 Performance metrics (win rate, profit factor, max drawdown)
- 💰 Position sizing based on risk tolerance
- 📱 Extensible for broker API integration (Zerodha, Upstox, etc.)
- 🎯 Stock screening and opportunity detection
- 📁 Export results to CSV

## 🚀 Quick Start

### Installation (Google Colab)

```python
# Install required packages
!pip install yfinance ta pandas numpy matplotlib seaborn plotly -q

# Clone or copy the AI_Investment_Assistant.py file to your Colab

# Import and run
%run AI_Investment_Assistant.py

# Initialize the assistant
assistant = InvestmentAssistant(portfolio_value=100000)

# Analyze a stock
results = assistant.analyze_symbol('AAPL')
```

### Basic Usage

```python
# Analyze a US stock
results = assistant.analyze_symbol(
    symbol='AAPL',
    period='1y',
    interval='1d',
    run_backtest=True,
    send_alerts=True
)

# Analyze an Indian stock (NSE)
results = assistant.analyze_symbol('RELIANCE.NS', period='6mo')

# Compare multiple stocks
comparison = assistant.compare_symbols(['AAPL', 'GOOGL', 'MSFT'])
print(comparison)
```

## 📊 System Architecture

```
Investment Assistant
│
├── MarketDataFetcher     → Retrieves market data via yfinance
├── TechnicalIndicators   → Calculates technical indicators
├── SignalGenerator       → Generates trading signals
├── RiskManager           → Manages position sizing & risk
├── BacktestEngine        → Backtests strategies
├── Visualizer            → Creates charts & visualizations
└── AlertSystem           → Sends alerts & notifications
```

## 🎯 Signal Generation System

The system uses a **multi-factor scoring approach**:

### Signal Components (Each worth +1 or -1 point)

1. **Moving Average Crossover**
   - Golden Cross (SMA short > SMA long): BUY
   - Death Cross (SMA short < SMA long): SELL

2. **RSI (Relative Strength Index)**
   - RSI < 30: Oversold → BUY
   - RSI > 70: Overbought → SELL

3. **MACD (Moving Average Convergence Divergence)**
   - Bullish crossover: BUY
   - Bearish crossover: SELL

4. **Bollinger Bands**
   - Price below lower band: BUY
   - Price above upper band: SELL

5. **Momentum**
   - Price above both EMAs: BUY
   - Price below both EMAs: SELL

### Final Signal Determination

| Score | Signal |
|-------|--------|
| ≥ 3   | STRONG BUY |
| 2     | BUY |
| -2    | SELL |
| ≤ -3  | STRONG SELL |
| Other | HOLD |

## 💰 Risk Management

### Position Sizing Formula

```
Maximum Risk Amount = Portfolio Value × Risk Per Trade %
Risk Per Share = |Current Price - Stop Loss Price|
Shares = Maximum Risk Amount / Risk Per Share
```

### Stop Loss Calculation

```
Stop Loss = Current Price ± (ATR × Multiplier)

Default multiplier = 2.0
```

### Take Profit Calculation

```
Risk = |Current Price - Stop Loss|
Reward = Risk × Risk-Reward Ratio
Take Profit = Current Price + Reward

Default Risk-Reward Ratio = 2:1
```

## 📈 Backtesting Methodology

1. **Signal Generation**: Apply strategy rules to historical data
2. **Trade Execution**: Simulate trades when signals trigger
3. **Cost Modeling**: Include commission (0.1%) and slippage (0.05%)
4. **Performance Tracking**: Calculate returns, win rate, drawdown
5. **Visualization**: Plot equity curve and trade distribution

### Backtest Metrics

- Total Return (%)
- Number of Trades
- Win Rate (%)
- Average Win/Loss (%)
- Max Drawdown (%)
- Profit Factor
- Risk-Adjusted Returns

## 🔧 Configuration

Customize behavior in `InvestmentConfig`:

```python
# Technical Indicators
config.SMA_SHORT = 20
config.SMA_LONG = 50
config.RSI_PERIOD = 14
config.RSI_OVERBOUGHT = 70
config.RSI_OVERSOLD = 30

# Signal Thresholds
config.STRONG_BUY_SCORE = 3
config.BUY_SCORE = 2

# Risk Management
config.MAX_POSITION_SIZE_PCT = 5.0
config.DEFAULT_RISK_PER_TRADE = 2.0
config.DEFAULT_STOP_LOSS_ATR = 2.0

# Backtesting
config.INITIAL_CAPITAL = 100000
config.COMMISSION_PCT = 0.1
config.SLIPPAGE_PCT = 0.05
```

## 📱 Alert System

### Current Implementation
- Console output (print statements)
- Alert history tracking

### Integration Templates Provided

#### Telegram Bot
```python
alert_system.setup_telegram_bot(
    bot_token='YOUR_BOT_TOKEN',
    chat_id='YOUR_CHAT_ID'
)
```

#### Email Alerts
```python
alert_system.setup_email_alerts(
    smtp_server='smtp.gmail.com',
    smtp_port=587,
    email='your_email@gmail.com',
    password='your_password',
    recipient='recipient@gmail.com'
)
```

## 🔌 Broker API Integration

### Architecture for Extension

The system is designed to easily integrate with broker APIs:

```python
class BrokerIntegration:
    """Template for broker API integration"""
    
    def __init__(self, api_key, access_token):
        # Initialize broker connection
        pass
    
    def get_live_price(self, symbol):
        """Fetch real-time price from broker"""
        pass
    
    def get_positions(self):
        """Get current open positions"""
        pass
    
    def place_order_alert(self, symbol, signal, quantity, price):
        """Send order recommendation (NO AUTO-TRADING)"""
        pass
```

### Supported Brokers (Integration Templates)

- **Zerodha Kite Connect** (India)
- **Upstox API** (India)
- **Interactive Brokers** (Global)
- **Alpaca** (US)
- **TD Ameritrade** (US)

**Note**: The system sends alerts only. You must manually review and place orders.

## 📊 Usage Examples

### 1. Basic Stock Analysis
```python
assistant = InvestmentAssistant(portfolio_value=100000)
results = assistant.analyze_symbol('AAPL')
```

### 2. Intraday Trading Analysis
```python
results = assistant.analyze_symbol(
    symbol='TSLA',
    period='5d',
    interval='15m',
    run_backtest=False
)
```

### 3. Stock Screening
```python
watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA']
comparison = assistant.compare_symbols(watchlist)

# Filter for BUY signals
buys = comparison[comparison['Signal'].str.contains('BUY')]
print(buys)
```

### 4. Indian Stocks (NSE)
```python
results = assistant.analyze_symbol('RELIANCE.NS', period='1y')
```

### 5. Export Analysis
```python
results = assistant.analyze_symbol('AAPL')

# Export signals
signals_df = results['data'][['close', 'signal', 'rsi']].tail(30)
signals_df.to_csv('AAPL_signals.csv')

# Export backtest trades
results['backtest']['trades'].to_csv('AAPL_trades.csv')
```

## 📁 Project Structure

```
AI_Investment_Assistant/
│
├── AI_Investment_Assistant.py    # Main system (Cells 1-10)
├── Usage_Examples.py              # 11 practical examples
├── README.md                      # This file
│
└── Output Files/
    ├── {SYMBOL}_signals.csv       # Generated signals
    ├── {SYMBOL}_trades.csv        # Backtest trades
    └── {SYMBOL}_summary.csv       # Analysis summary
```

## 🎨 Visualizations

The system generates interactive Plotly charts:

1. **Price & Technical Indicators**
   - Candlestick chart
   - Moving averages (SMA/EMA)
   - Bollinger Bands
   - Volume

2. **MACD Chart**
   - MACD line
   - Signal line
   - Histogram

3. **RSI Chart**
   - RSI line
   - Overbought/oversold levels

4. **Buy/Sell Signals**
   - Price chart with marked signals
   - Buy signals (green triangles)
   - Sell signals (red triangles)

5. **Backtest Results**
   - Equity curve
   - Trade distribution
   - Win/loss pie chart
   - Monthly returns

## 🔐 Security & Best Practices

### Data Security
- Never hardcode API keys in notebooks
- Use environment variables for credentials
- Don't share notebooks with sensitive data

### Trading Safety
1. ✅ Always review signals before trading
2. ✅ Start with paper trading
3. ✅ Use proper position sizing
4. ✅ Set stop losses
5. ✅ Never risk more than 1-2% per trade
6. ✅ Diversify across multiple positions
7. ❌ Never use auto-trading without thorough testing

### Code Safety
```python
# Good: Use environment variables
import os
api_key = os.environ.get('BROKER_API_KEY')

# Bad: Hardcoded credentials
api_key = 'your_actual_key'  # Don't do this!
```

## 🛠️ Customization Examples

### Create Custom Strategy
```python
class CustomStrategy(SignalGenerator):
    @staticmethod
    def generate_custom_signals(df):
        # Your custom logic here
        df['custom_score'] = ...
        df['custom_signal'] = ...
        return df
```

### Modify Risk Parameters
```python
# More conservative settings
config.DEFAULT_RISK_PER_TRADE = 1.0  # Risk only 1%
config.MAX_POSITION_SIZE_PCT = 3.0   # Max 3% position size
config.DEFAULT_STOP_LOSS_ATR = 3.0   # Wider stop loss
```

### Add New Indicators
```python
# In TechnicalIndicators class
from ta.momentum import StochasticOscillator

stoch = StochasticOscillator(
    high=df['high'],
    low=df['low'],
    close=df['close']
)
df['stoch_k'] = stoch.stoch()
df['stoch_d'] = stoch.stoch_signal()
```

## 📚 Technical Indicators Reference

| Indicator | Purpose | Interpretation |
|-----------|---------|----------------|
| **SMA** | Trend identification | Price above SMA = Uptrend |
| **EMA** | Faster trend signals | More responsive to recent prices |
| **RSI** | Momentum & overbought/oversold | <30 oversold, >70 overbought |
| **MACD** | Trend & momentum | Crossovers signal trend changes |
| **Bollinger Bands** | Volatility & extremes | Price at bands = potential reversal |
| **ATR** | Volatility measurement | Higher ATR = higher volatility |

## 🐛 Troubleshooting

### Common Issues

**1. "No data found for symbol"**
```python
# Solution: Check symbol format
# US stocks: 'AAPL', 'GOOGL'
# Indian stocks: 'RELIANCE.NS', 'TCS.NS'
# Add exchange suffix for international stocks
```

**2. "Insufficient data for indicators"**
```python
# Solution: Use longer period
results = assistant.analyze_symbol('AAPL', period='1y')
```

**3. "No trades in backtest"**
```python
# Solution: Adjust signal thresholds
config.BUY_SCORE = 1  # More sensitive
config.SELL_SCORE = -1
```

**4. Charts not displaying**
```python
# Solution: In Colab, use
from google.colab import output
output.enable_custom_widget_manager()
```

## 📈 Performance Optimization

### For Faster Analysis
```python
# Use shorter periods for quick checks
assistant.analyze_symbol('AAPL', period='3mo', run_backtest=False)

# Enable caching (automatic)
# Second call to same symbol is instant
```

### For Multiple Symbols
```python
# Use batch comparison instead of individual analysis
comparison = assistant.compare_symbols(['AAPL', 'GOOGL', 'MSFT'])
```

## 🔄 Updates & Maintenance

### Keeping Data Fresh
```python
# Force refresh cache
df = data_fetcher.fetch_data('AAPL', force_refresh=True)
```

### Updating Parameters
```python
# Test different parameters
for rsi_period in [10, 14, 20]:
    config.RSI_PERIOD = rsi_period
    results = assistant.analyze_symbol('AAPL')
    print(f"RSI Period {rsi_period}: {results['latest_signal']['signal']}")
```

## 📖 Learning Resources

### Recommended Reading
- **Technical Analysis**: "Technical Analysis of the Financial Markets" by John Murphy
- **Risk Management**: "Trade Your Way to Financial Freedom" by Van Tharp
- **Algorithmic Trading**: "Algorithmic Trading" by Ernest Chan

### Useful Links
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [ta-lib Documentation](https://technical-analysis-library-in-python.readthedocs.io/)
- [Plotly Python](https://plotly.com/python/)

## 🤝 Contributing

To extend this system:

1. **Add New Indicators**: Modify `TechnicalIndicators` class
2. **Create Custom Strategies**: Extend `SignalGenerator` class
3. **Add Broker APIs**: Implement `BrokerIntegration` template
4. **Improve Backtesting**: Enhance `BacktestEngine` class

## 📄 License

MIT License - Free to use for personal and educational purposes.

## ⚠️ Final Disclaimer

This software is provided "as is" without warranty of any kind. The creators and contributors are not responsible for any financial losses incurred from using this system. Always conduct your own research and consult with licensed financial advisors before making investment decisions.

**Remember**: 
- Past performance ≠ Future results
- Backtests are not guarantees
- Markets are unpredictable
- Risk management is crucial
- Never invest more than you can afford to lose

## 🚀 Getting Started Checklist

- [ ] Install required packages
- [ ] Run AI_Investment_Assistant.py in Colab
- [ ] Analyze your first stock with `assistant.analyze_symbol('AAPL')`
- [ ] Review the generated signals and charts
- [ ] Run a backtest to understand historical performance
- [ ] Customize config parameters for your strategy
- [ ] Set up alerts (optional)
- [ ] Paper trade for at least 1 month before live trading
- [ ] Integrate broker API (if desired)
- [ ] Always review signals manually before trading

---

**Happy Analyzing! 📊🚀**

For questions and support, refer to the Usage_Examples.py file for comprehensive examples.
