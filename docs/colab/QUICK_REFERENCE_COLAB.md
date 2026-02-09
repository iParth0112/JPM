# AI INVESTMENT ASSISTANT - QUICK REFERENCE GUIDE 🚀

## 📋 Installation (One Line)

```python
!pip install yfinance ta pandas numpy matplotlib seaborn plotly -q
```

## ⚡ Quick Start (3 Lines)

```python
%run AI_Investment_Assistant.py
assistant = InvestmentAssistant(portfolio_value=100000)
results = assistant.analyze_symbol('AAPL')
```

## 🎯 Most Common Commands

### Basic Analysis
```python
# US Stock
assistant.analyze_symbol('AAPL')

# Indian Stock (NSE)
assistant.analyze_symbol('RELIANCE.NS')

# With custom parameters
assistant.analyze_symbol('TSLA', period='6mo', interval='1d')
```

### Compare Multiple Stocks
```python
comparison = assistant.compare_symbols(['AAPL', 'GOOGL', 'MSFT'])
print(comparison)
```

### Intraday Analysis
```python
assistant.analyze_symbol('TSLA', period='5d', interval='15m', run_backtest=False)
```

## 📊 Data Periods & Intervals

### Periods (Historical Data)
- `'1d'` - 1 day
- `'5d'` - 5 days
- `'1mo'` - 1 month
- `'3mo'` - 3 months
- `'6mo'` - 6 months
- `'1y'` - 1 year (default)
- `'2y'` - 2 years
- `'5y'` - 5 years
- `'max'` - Maximum available

### Intervals (Data Points)
- `'1m'` - 1 minute (for day traders)
- `'5m'` - 5 minutes
- `'15m'` - 15 minutes
- `'1h'` - 1 hour
- `'1d'` - 1 day (default)
- `'1wk'` - 1 week
- `'1mo'` - 1 month

## 🌍 Stock Symbol Formats

| Market | Format | Example |
|--------|--------|---------|
| **US** | SYMBOL | `AAPL`, `TSLA`, `GOOGL` |
| **India (NSE)** | SYMBOL.NS | `RELIANCE.NS`, `TCS.NS` |
| **India (BSE)** | SYMBOL.BO | `RELIANCE.BO` |
| **UK** | SYMBOL.L | `BARC.L`, `BP.L` |
| **Germany** | SYMBOL.DE | `BMW.DE` |
| **Japan** | SYMBOL.T | `7203.T` (Toyota) |

## ⚙️ Configuration Quick Changes

```python
# More Conservative
config.DEFAULT_RISK_PER_TRADE = 1.0  # Risk 1% per trade
config.MAX_POSITION_SIZE_PCT = 3.0   # Max 3% position
config.STRONG_BUY_SCORE = 4          # Need 4 signals

# More Aggressive
config.DEFAULT_RISK_PER_TRADE = 3.0  # Risk 3% per trade
config.MAX_POSITION_SIZE_PCT = 10.0  # Max 10% position
config.BUY_SCORE = 1                 # Need only 1 signal

# Adjust Indicators
config.RSI_OVERSOLD = 25    # More extreme oversold
config.RSI_OVERBOUGHT = 75  # More extreme overbought
config.SMA_SHORT = 10       # Faster moving average
config.SMA_LONG = 30        # Shorter long-term MA
```

## 🎨 Understanding Signals

| Signal | Score | Meaning |
|--------|-------|---------|
| **STRONG BUY** | ≥ 3 | 3+ indicators bullish |
| **BUY** | 2 | 2 indicators bullish |
| **HOLD** | -1 to 1 | Mixed signals |
| **SELL** | -2 | 2 indicators bearish |
| **STRONG SELL** | ≤ -3 | 3+ indicators bearish |

## 📈 Access Results Data

```python
results = assistant.analyze_symbol('AAPL')

# Current signal
print(results['latest_signal']['signal'])
print(results['latest_signal']['close'])

# Risk metrics
print(results['risk_metrics']['stop_loss'])
print(results['risk_metrics']['take_profit'])
print(results['risk_metrics']['shares'])

# Full data with indicators
df = results['data']
print(df[['close', 'rsi', 'macd', 'signal']].tail())

# Backtest performance
if results['backtest']:
    metrics = results['backtest']['metrics']
    print(f"Return: {metrics['total_return_pct']:.2f}%")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
```

## 💾 Export Data

```python
# Export signals
results['data'][['close', 'signal', 'rsi']].to_csv('signals.csv')

# Export backtest trades
results['backtest']['trades'].to_csv('trades.csv')

# Create summary
summary = {
    'Symbol': 'AAPL',
    'Signal': results['latest_signal']['signal'],
    'Price': results['latest_signal']['close'],
    'Stop Loss': results['risk_metrics']['stop_loss']
}
pd.DataFrame([summary]).to_csv('summary.csv')
```

## 🔍 Stock Screening Pattern

```python
watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA']
opportunities = []

for symbol in watchlist:
    try:
        df = assistant.data_fetcher.fetch_data(symbol, period='3mo')
        df = assistant.technical_indicators.calculate_all(df)
        df = assistant.signal_generator.generate_signals(df)
        latest = assistant.signal_generator.get_latest_signal(df)
        
        if latest['signal'] == 'STRONG BUY':
            opportunities.append({
                'Symbol': symbol,
                'Price': latest['close'],
                'RSI': latest['rsi']
            })
    except:
        continue

if opportunities:
    print(pd.DataFrame(opportunities))
```

## 🛠️ Troubleshooting Quick Fixes

### No data fetched
```python
# Check symbol format
assistant.analyze_symbol('AAPL')  # ✅ Correct
assistant.analyze_symbol('RELIANCE.NS')  # ✅ For NSE stocks

# Force refresh
df = assistant.data_fetcher.fetch_data('AAPL', force_refresh=True)
```

### No signals generated
```python
# Lower threshold
config.BUY_SCORE = 1
config.SELL_SCORE = -1

# Or use longer period
assistant.analyze_symbol('AAPL', period='2y')
```

### Charts not showing
```python
# In Google Colab, run this first
from google.colab import output
output.enable_custom_widget_manager()
```

## 📊 Technical Indicator Quick Reference

| Indicator | Bullish Signal | Bearish Signal |
|-----------|---------------|----------------|
| **RSI** | < 30 (oversold) | > 70 (overbought) |
| **MACD** | Crosses above signal | Crosses below signal |
| **SMA** | Price > SMA | Price < SMA |
| **BB** | Price touches lower | Price touches upper |
| **Golden Cross** | SMA(20) > SMA(50) | SMA(20) < SMA(50) |

## 💰 Position Sizing Formula

```
Risk Amount = Portfolio × Risk%
Shares = Risk Amount / (Entry Price - Stop Loss)
Max Position = Portfolio × Max Position%
```

**Example:**
- Portfolio: $100,000
- Risk per trade: 2% = $2,000
- Entry: $150
- Stop Loss: $145
- Risk per share: $5
- Shares: $2,000 / $5 = 400 shares
- Position value: 400 × $150 = $60,000

## 🎯 Strategy Templates

### Conservative Long-Term
```python
config.DEFAULT_PERIOD = "2y"
config.DEFAULT_INTERVAL = "1d"
config.DEFAULT_RISK_PER_TRADE = 1.0
config.STRONG_BUY_SCORE = 4
config.RSI_OVERSOLD = 25
```

### Aggressive Day Trading
```python
config.DEFAULT_PERIOD = "5d"
config.DEFAULT_INTERVAL = "5m"
config.DEFAULT_RISK_PER_TRADE = 3.0
config.BUY_SCORE = 1
config.RSI_OVERSOLD = 35
```

### Balanced Swing Trading
```python
config.DEFAULT_PERIOD = "6mo"
config.DEFAULT_INTERVAL = "1d"
config.DEFAULT_RISK_PER_TRADE = 2.0
config.STRONG_BUY_SCORE = 3
config.RSI_OVERSOLD = 30
```

## 🚨 Risk Management Checklist

- [ ] Never risk more than 1-2% per trade
- [ ] Always set stop losses
- [ ] Use position sizing formula
- [ ] Diversify across 5-10 positions
- [ ] Review signals manually
- [ ] Start with paper trading
- [ ] Track performance in a journal

## 🔄 Real-Time Monitoring Loop

```python
import time
from datetime import datetime

watchlist = ['AAPL', 'GOOGL', 'MSFT']

while True:
    print(f"\n{'='*70}")
    print(f"Scan at: {datetime.now()}")
    
    for symbol in watchlist:
        results = assistant.analyze_symbol(
            symbol, 
            period='5d', 
            run_backtest=False,
            send_alerts=False
        )
        print(f"{symbol}: {results['latest_signal']['signal']}")
    
    time.sleep(3600)  # Check every hour
```

## 📱 Broker Integration Template

```python
class BrokerIntegration:
    def place_order_alert(self, symbol, signal, shares, price):
        alert = f"""
        🔔 ORDER RECOMMENDATION
        Symbol: {symbol}
        Action: {signal}
        Shares: {shares}
        Price: ${price:.2f}
        
        ⚠️ Review and place manually
        """
        print(alert)
        return alert

# Use it
broker = BrokerIntegration()
results = assistant.analyze_symbol('AAPL')

if 'BUY' in results['latest_signal']['signal']:
    broker.place_order_alert(
        'AAPL',
        results['latest_signal']['signal'],
        results['risk_metrics']['shares'],
        results['latest_signal']['close']
    )
```

## 📊 Performance Metrics Explained

| Metric | Good Value | What It Means |
|--------|-----------|---------------|
| **Total Return** | > 10% | Overall profit/loss |
| **Win Rate** | > 50% | % of winning trades |
| **Profit Factor** | > 1.5 | Wins/Losses ratio |
| **Max Drawdown** | < 20% | Largest peak-to-trough decline |
| **Avg Win** | > 5% | Average winning trade % |

## 🎓 Learning Path

1. **Week 1**: Basic analysis on 5-10 stocks
2. **Week 2**: Run backtests, understand metrics
3. **Week 3**: Customize config, test strategies
4. **Week 4**: Paper trade with alerts
5. **Month 2**: Refine strategy, track performance
6. **Month 3+**: Consider live trading (small positions)

## 💡 Pro Tips

1. **Always backtest** before live trading
2. **Keep a trading journal** - track every decision
3. **Review signals at same time daily** - consistency matters
4. **Don't overtrade** - wait for quality setups
5. **Size down in uncertainty** - reduce risk when unsure
6. **Learn continuously** - markets evolve

## 🔗 Quick Links

- GitHub: [Your Repo]
- Documentation: See README.md
- Examples: See Usage_Examples.py
- Support: [Your Contact]

---

## 🚀 Copy-Paste Starter Template

```python
# COMPLETE STARTER SCRIPT
!pip install yfinance ta pandas numpy matplotlib seaborn plotly -q

%run AI_Investment_Assistant.py

# Initialize
assistant = InvestmentAssistant(portfolio_value=100000)

# Analyze
symbol = 'AAPL'  # Change this
results = assistant.analyze_symbol(symbol, period='1y')

# Print key info
print(f"Signal: {results['latest_signal']['signal']}")
print(f"Price: ${results['latest_signal']['close']:.2f}")
print(f"Stop Loss: ${results['risk_metrics']['stop_loss']:.2f}")
print(f"Take Profit: ${results['risk_metrics']['take_profit']:.2f}")
print(f"Position: {results['risk_metrics']['shares']} shares")

# Backtest results
if results['backtest']:
    m = results['backtest']['metrics']
    print(f"\nBacktest Return: {m['total_return_pct']:.2f}%")
    print(f"Win Rate: {m['win_rate_pct']:.2f}%")
```

---

**Remember: This is a tool, not a crystal ball. Always trade responsibly! 🎯📈**
