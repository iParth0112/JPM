"""
===============================================================================
AI INVESTMENT ASSISTANT - USAGE EXAMPLES & QUICK START GUIDE
===============================================================================

This notebook contains practical examples of how to use the AI Investment
Assistant for stock analysis, signal generation, and backtesting.

===============================================================================
"""

# ============================================================================
# EXAMPLE 1: BASIC STOCK ANALYSIS
# ============================================================================

print("="*70)
print("EXAMPLE 1: Basic Stock Analysis")
print("="*70)

# Initialize the assistant
assistant = InvestmentAssistant(portfolio_value=100000)

# Analyze a single stock
results = assistant.analyze_symbol(
    symbol='AAPL',           # Apple Inc.
    period='1y',             # 1 year of data
    interval='1d',           # Daily intervals
    run_backtest=True,       # Run backtest
    send_alerts=True         # Send alerts for strong signals
)

# Access results
print("\n📊 Analysis Results:")
print(f"Signal: {results['latest_signal']['signal']}")
print(f"Current Price: ${results['latest_signal']['close']:.2f}")
print(f"Stop Loss: ${results['risk_metrics']['stop_loss']:.2f}")
print(f"Take Profit: ${results['risk_metrics']['take_profit']:.2f}")


# ============================================================================
# EXAMPLE 2: ANALYZE INDIAN STOCKS (NSE)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Analyzing Indian Stocks")
print("="*70)

# Analyze Indian stocks (add .NS suffix for NSE)
indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']

for symbol in indian_stocks:
    print(f"\n{'='*70}")
    print(f"Analyzing: {symbol}")
    print(f"{'='*70}")
    
    results = assistant.analyze_symbol(
        symbol=symbol,
        period='6mo',
        run_backtest=True,
        send_alerts=False  # Don't send alerts for batch analysis
    )
    
    if 'error' not in results:
        print(f"✅ Signal: {results['latest_signal']['signal']}")
        print(f"📊 Score: {results['latest_signal']['score']}")


# ============================================================================
# EXAMPLE 3: COMPARE MULTIPLE STOCKS
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Compare Multiple Stocks")
print("="*70)

# Compare tech stocks
tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
comparison = assistant.compare_symbols(tech_stocks)

print("\n📊 Stock Comparison:")
print(comparison.to_string(index=False))

# Filter for BUY signals
buy_signals = comparison[comparison['Signal'].str.contains('BUY', na=False)]
if not buy_signals.empty:
    print("\n🟢 Stocks with BUY signals:")
    print(buy_signals.to_string(index=False))


# ============================================================================
# EXAMPLE 4: INTRADAY ANALYSIS (SHORT-TERM TRADING)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Intraday Analysis")
print("="*70)

# Analyze with shorter timeframes for day trading
intraday_results = assistant.analyze_symbol(
    symbol='TSLA',
    period='5d',      # Last 5 days
    interval='15m',   # 15-minute intervals
    run_backtest=False,  # Skip backtest for intraday
    send_alerts=True
)

print(f"\n📊 Intraday Signal: {intraday_results['latest_signal']['signal']}")


# ============================================================================
# EXAMPLE 5: CUSTOM RISK MANAGEMENT
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Custom Risk Management")
print("="*70)

# Create assistant with custom portfolio value
conservative_assistant = InvestmentAssistant(portfolio_value=50000)

# Modify risk parameters
config.DEFAULT_RISK_PER_TRADE = 1.0  # More conservative: 1% risk per trade
config.MAX_POSITION_SIZE_PCT = 3.0   # Smaller positions: 3% max

results = conservative_assistant.analyze_symbol('NVDA', period='6mo')

print(f"\n💰 Conservative Risk Management:")
print(f"Position Size: {results['risk_metrics']['shares']} shares")
print(f"Risk Amount: ${results['risk_metrics']['risk_amount']:.2f}")
print(f"Risk %: {results['risk_metrics']['risk_pct']:.2f}%")


# ============================================================================
# EXAMPLE 6: BACKTEST ONLY MODE
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: Backtest Historical Performance")
print("="*70)

# Fetch historical data
symbol = 'SPY'  # S&P 500 ETF
df = assistant.data_fetcher.fetch_data(symbol, period='2y', interval='1d')

# Calculate indicators
df = assistant.technical_indicators.calculate_all(df)

# Generate signals
df = assistant.signal_generator.generate_signals(df)

# Run backtest
trades_df, metrics = assistant.backtest_engine.run_backtest(df)

print(f"\n📈 Backtest Results for {symbol}:")
print(f"Period: 2 years")
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

# Visualize backtest
assistant.visualizer.plot_backtest_results(trades_df, metrics)


# ============================================================================
# EXAMPLE 7: SCREENING FOR OPPORTUNITIES
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 7: Stock Screening")
print("="*70)

# Screen a watchlist for trading opportunities
watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX']

opportunities = []

for symbol in watchlist:
    try:
        df = assistant.data_fetcher.fetch_data(symbol, period='3mo')
        df = assistant.technical_indicators.calculate_all(df)
        df = assistant.signal_generator.generate_signals(df)
        latest = assistant.signal_generator.get_latest_signal(df)
        
        # Filter for STRONG BUY signals with good RSI
        if latest['signal'] == 'STRONG BUY' and latest['rsi'] < 60:
            opportunities.append({
                'Symbol': symbol,
                'Price': latest['close'],
                'Signal': latest['signal'],
                'Score': latest['score'],
                'RSI': latest['rsi']
            })
    except:
        continue

if opportunities:
    print("\n🎯 Trading Opportunities Found:")
    opp_df = pd.DataFrame(opportunities)
    print(opp_df.to_string(index=False))
else:
    print("\n⚠️ No strong opportunities found in current watchlist")


# ============================================================================
# EXAMPLE 8: REAL-TIME MONITORING (LOOP)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 8: Real-Time Monitoring Setup")
print("="*70)

# This example shows how to set up continuous monitoring
# (Don't run this in a regular notebook - use a separate script)

print("""
For real-time monitoring, create a separate Python script:

import time

assistant = InvestmentAssistant(portfolio_value=100000)
watchlist = ['AAPL', 'GOOGL', 'MSFT']

while True:
    print(f"\\n{'='*70}")
    print(f"Monitoring at: {datetime.now()}")
    print(f"{'='*70}")
    
    for symbol in watchlist:
        results = assistant.analyze_symbol(
            symbol=symbol,
            period='5d',
            interval='1h',
            run_backtest=False,
            send_alerts=True
        )
        
        # Check for signals
        if 'STRONG' in results['latest_signal']['signal']:
            print(f"⚠️ ALERT: {symbol} - {results['latest_signal']['signal']}")
    
    # Wait for 1 hour before next check
    time.sleep(3600)
""")


# ============================================================================
# EXAMPLE 9: EXPORT RESULTS
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 9: Export Analysis Results")
print("="*70)

# Analyze and export results
symbol = 'AAPL'
results = assistant.analyze_symbol(symbol, period='1y')

# Export signals to CSV
signals_df = results['data'][['close', 'signal', 'signal_score', 'rsi', 'macd']].tail(30)
signals_df.to_csv(f'{symbol}_signals.csv')
print(f"✅ Signals exported to {symbol}_signals.csv")

# Export backtest trades
if results['backtest'] and len(results['backtest']['trades']) > 0:
    results['backtest']['trades'].to_csv(f'{symbol}_backtest_trades.csv', index=False)
    print(f"✅ Backtest trades exported to {symbol}_backtest_trades.csv")

# Create summary report
summary = {
    'Symbol': symbol,
    'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'Current Price': results['latest_signal']['close'],
    'Signal': results['latest_signal']['signal'],
    'Signal Score': results['latest_signal']['score'],
    'RSI': results['latest_signal']['rsi'],
    'Stop Loss': results['risk_metrics']['stop_loss'],
    'Take Profit': results['risk_metrics']['take_profit'],
    'Position Size': results['risk_metrics']['shares'],
    'Risk Amount': results['risk_metrics']['risk_amount']
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f'{symbol}_analysis_summary.csv', index=False)
print(f"✅ Summary exported to {symbol}_analysis_summary.csv")


# ============================================================================
# EXAMPLE 10: BROKER API INTEGRATION (TEMPLATE)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 10: Broker API Integration Template")
print("="*70)

print("""
To integrate with broker APIs (Zerodha, Upstox, etc.), use this template:

# ============================================================================
# ZERODHA KITE CONNECT INTEGRATION EXAMPLE
# ============================================================================

from kiteconnect import KiteConnect

class BrokerIntegration:
    def __init__(self, api_key, access_token):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
    
    def get_live_price(self, symbol):
        '''Get real-time price from broker'''
        quote = self.kite.quote(f"NSE:{symbol}")
        return quote[f"NSE:{symbol}"]['last_price']
    
    def get_positions(self):
        '''Get current positions'''
        return self.kite.positions()
    
    def place_order_alert(self, symbol, signal, quantity, price):
        '''
        Instead of placing order automatically, send alert
        User can review and place order manually
        '''
        alert_message = f'''
        📱 ORDER RECOMMENDATION
        
        Symbol: {symbol}
        Action: {signal}
        Quantity: {quantity}
        Price: {price}
        
        ⚠️ Review and place order manually in your broker app
        '''
        print(alert_message)
        # Send via Telegram/Email
        return alert_message

# Initialize broker connection
# broker = BrokerIntegration(api_key='your_key', access_token='your_token')

# Use with investment assistant
# live_price = broker.get_live_price('RELIANCE')
# results = assistant.analyze_symbol('RELIANCE.NS')
# 
# if 'BUY' in results['latest_signal']['signal']:
#     broker.place_order_alert(
#         symbol='RELIANCE',
#         signal=results['latest_signal']['signal'],
#         quantity=results['risk_metrics']['shares'],
#         price=results['latest_signal']['close']
#     )

# ============================================================================
""")

print("\n" + "="*70)
print("✅ All examples completed!")
print("="*70)


# ============================================================================
# EXAMPLE 11: CUSTOM INDICATOR STRATEGY
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 11: Create Custom Strategy")
print("="*70)

print("""
You can extend the system with custom strategies:

class CustomStrategy(SignalGenerator):
    '''Custom strategy using your own logic'''
    
    @staticmethod
    def generate_custom_signals(df):
        df = df.copy()
        
        # Example: Mean reversion strategy
        df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        df['custom_signal'] = 'HOLD'
        df.loc[df['z_score'] < -2, 'custom_signal'] = 'BUY'  # Oversold
        df.loc[df['z_score'] > 2, 'custom_signal'] = 'SELL'  # Overbought
        
        return df

# Use custom strategy
custom_strategy = CustomStrategy()
df = assistant.data_fetcher.fetch_data('AAPL', period='1y')
df = assistant.technical_indicators.calculate_all(df)
df = custom_strategy.generate_custom_signals(df)

print(df[['close', 'z_score', 'custom_signal']].tail(10))
""")


# ============================================================================
# CONFIGURATION GUIDE
# ============================================================================

print("\n" + "="*70)
print("CONFIGURATION CUSTOMIZATION GUIDE")
print("="*70)

print("""
Modify these parameters in the InvestmentConfig class:

# Technical Indicators
config.SMA_SHORT = 20         # Short-term moving average period
config.SMA_LONG = 50          # Long-term moving average period
config.RSI_PERIOD = 14        # RSI calculation period
config.RSI_OVERBOUGHT = 70    # RSI overbought threshold
config.RSI_OVERSOLD = 30      # RSI oversold threshold

# Signal Thresholds
config.STRONG_BUY_SCORE = 3   # Minimum signals for STRONG BUY
config.BUY_SCORE = 2          # Minimum signals for BUY
config.SELL_SCORE = -2        # Maximum signals for SELL

# Risk Management
config.MAX_POSITION_SIZE_PCT = 5.0    # Max % of portfolio per position
config.DEFAULT_RISK_PER_TRADE = 2.0   # Max % risk per trade
config.DEFAULT_STOP_LOSS_ATR = 2.0    # Stop loss in ATR multiples

# Backtesting
config.INITIAL_CAPITAL = 100000   # Starting capital
config.COMMISSION_PCT = 0.1       # Commission per trade (%)
config.SLIPPAGE_PCT = 0.05        # Slippage per trade (%)

Example:
# Make strategy more conservative
config.STRONG_BUY_SCORE = 4  # Require more confirmations
config.DEFAULT_RISK_PER_TRADE = 1.0  # Risk only 1% per trade
config.RSI_OVERSOLD = 25  # More extreme oversold condition
""")

print("\n✅ Usage guide complete!")
print("\n🚀 You're ready to start analyzing stocks!")
print("\nQuick start: assistant.analyze_symbol('AAPL')")
