"""
===============================================================================
AI INVESTMENT ASSISTANT - GOOGLE COLAB PROJECT
===============================================================================

A comprehensive personal investment assistant that:
- Fetches live market data
- Calculates technical indicators
- Generates trading signals
- Performs risk management
- Backtests strategies
- Sends alerts

Author: Senior Python Quant Developer
Version: 1.0
License: MIT

NOTE: This system is for ANALYSIS ONLY. It does NOT execute trades automatically.
===============================================================================
"""

# ============================================================================
# CELL 1: INSTALLATION AND IMPORTS
# ============================================================================

# Install required packages
!pip install yfinance ta pandas numpy matplotlib seaborn plotly -q

# Standard library imports
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Data manipulation
import pandas as pd
import numpy as np

# Market data
import yfinance as yf

# Technical analysis
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ All packages installed and imported successfully!")
print(f"📅 System initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# CELL 2: CONFIGURATION CLASS
# ============================================================================

class InvestmentConfig:
    """
    Centralized configuration for the investment assistant.
    Modify these parameters to customize behavior.
    """
    
    # Data Parameters
    DEFAULT_PERIOD = "1y"  # Historical data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
    DEFAULT_INTERVAL = "1d"  # Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    # Technical Indicator Parameters
    SMA_SHORT = 20
    SMA_LONG = 50
    EMA_SHORT = 12
    EMA_LONG = 26
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    ATR_PERIOD = 14
    
    # Signal Thresholds
    STRONG_BUY_SCORE = 3  # Minimum signals for STRONG BUY
    BUY_SCORE = 2         # Minimum signals for BUY
    SELL_SCORE = -2       # Maximum signals for SELL
    STRONG_SELL_SCORE = -3 # Maximum signals for STRONG SELL
    
    # Risk Management Parameters
    MAX_POSITION_SIZE_PCT = 5.0  # Maximum % of portfolio per position
    DEFAULT_RISK_PER_TRADE = 2.0  # Maximum % risk per trade
    DEFAULT_STOP_LOSS_ATR = 2.0   # Stop loss in ATR multiples
    DEFAULT_TAKE_PROFIT_RR = 2.0  # Risk-reward ratio for take profit
    
    # Backtesting Parameters
    INITIAL_CAPITAL = 100000  # Starting capital for backtesting
    COMMISSION_PCT = 0.1      # Commission per trade (%)
    SLIPPAGE_PCT = 0.05       # Slippage per trade (%)
    
    # Alert Configuration
    ENABLE_ALERTS = True
    ALERT_SIGNALS = ['STRONG BUY', 'STRONG SELL']  # Which signals trigger alerts

config = InvestmentConfig()
print("✅ Configuration loaded successfully!")


# ============================================================================
# CELL 3: DATA FETCHER CLASS
# ============================================================================

class MarketDataFetcher:
    """
    Handles all market data retrieval operations using yfinance.
    """
    
    def __init__(self):
        self.cache = {}  # Simple cache to avoid redundant API calls
    
    def fetch_data(
        self, 
        symbol: str, 
        period: str = config.DEFAULT_PERIOD,
        interval: str = config.DEFAULT_INTERVAL,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical market data for a given symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'AAPL', 'TSLA', 'RELIANCE.NS')
        period : str
            Data period (default from config)
        interval : str
            Data interval (default from config)
        force_refresh : bool
            Force refresh cache
            
        Returns:
        --------
        pd.DataFrame
            OHLCV data with datetime index
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        if not force_refresh and cache_key in self.cache:
            print(f"📦 Loading {symbol} from cache...")
            return self.cache[cache_key].copy()
        
        try:
            print(f"📡 Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Clean column names
            df.columns = [col.lower() for col in df.columns]
            
            # Store in cache
            self.cache[cache_key] = df.copy()
            
            print(f"✅ Fetched {len(df)} data points for {symbol}")
            print(f"📊 Date range: {df.index[0].date()} to {df.index[-1].date()}")
            
            return df
        
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get the most recent closing price."""
        df = self.fetch_data(symbol, period="5d", interval="1d")
        if not df.empty:
            return df['close'].iloc[-1]
        return 0.0
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
            }
        except:
            return {'name': symbol}

# Initialize data fetcher
data_fetcher = MarketDataFetcher()
print("✅ Market Data Fetcher initialized!")


# ============================================================================
# CELL 4: TECHNICAL INDICATORS CALCULATOR
# ============================================================================

class TechnicalIndicators:
    """
    Calculates various technical indicators for market analysis.
    """
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them to the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added indicator columns
        """
        df = df.copy()
        
        print("🔧 Calculating technical indicators...")
        
        # Moving Averages
        df['sma_short'] = SMAIndicator(
            close=df['close'], 
            window=config.SMA_SHORT
        ).sma_indicator()
        
        df['sma_long'] = SMAIndicator(
            close=df['close'], 
            window=config.SMA_LONG
        ).sma_indicator()
        
        df['ema_short'] = EMAIndicator(
            close=df['close'], 
            window=config.EMA_SHORT
        ).ema_indicator()
        
        df['ema_long'] = EMAIndicator(
            close=df['close'], 
            window=config.EMA_LONG
        ).ema_indicator()
        
        # RSI
        rsi_indicator = RSIIndicator(close=df['close'], window=config.RSI_PERIOD)
        df['rsi'] = rsi_indicator.rsi()
        
        # MACD
        macd_indicator = MACD(
            close=df['close'],
            window_fast=config.MACD_FAST,
            window_slow=config.MACD_SLOW,
            window_sign=config.MACD_SIGNAL
        )
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_diff'] = macd_indicator.macd_diff()
        
        # Bollinger Bands
        bb_indicator = BollingerBands(
            close=df['close'],
            window=config.BB_PERIOD,
            window_dev=config.BB_STD
        )
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = bb_indicator.bollinger_wband()
        
        # ATR (for volatility and stop loss calculation)
        atr_indicator = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=config.ATR_PERIOD
        )
        df['atr'] = atr_indicator.average_true_range()
        
        # Additional metrics
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        print("✅ Technical indicators calculated!")
        return df


# ============================================================================
# CELL 5: SIGNAL GENERATOR CLASS
# ============================================================================

class SignalGenerator:
    """
    Generates BUY/SELL/HOLD signals based on technical indicators.
    Uses a multi-factor scoring system.
    """
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on multiple technical indicators.
        
        Returns a dataframe with individual signal columns and aggregated score.
        """
        df = df.copy()
        
        print("🎯 Generating trading signals...")
        
        # Initialize signal columns
        df['signal_ma_cross'] = 0
        df['signal_rsi'] = 0
        df['signal_macd'] = 0
        df['signal_bb'] = 0
        df['signal_momentum'] = 0
        
        # 1. Moving Average Crossover Signal
        df.loc[
            (df['sma_short'] > df['sma_long']) & 
            (df['sma_short'].shift(1) <= df['sma_long'].shift(1)), 
            'signal_ma_cross'
        ] = 1  # Golden cross - BUY
        
        df.loc[
            (df['sma_short'] < df['sma_long']) & 
            (df['sma_short'].shift(1) >= df['sma_long'].shift(1)), 
            'signal_ma_cross'
        ] = -1  # Death cross - SELL
        
        # 2. RSI Signal
        df.loc[df['rsi'] < config.RSI_OVERSOLD, 'signal_rsi'] = 1  # Oversold - BUY
        df.loc[df['rsi'] > config.RSI_OVERBOUGHT, 'signal_rsi'] = -1  # Overbought - SELL
        
        # 3. MACD Signal
        df.loc[
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 
            'signal_macd'
        ] = 1  # Bullish crossover - BUY
        
        df.loc[
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 
            'signal_macd'
        ] = -1  # Bearish crossover - SELL
        
        # 4. Bollinger Bands Signal
        df.loc[df['close'] < df['bb_lower'], 'signal_bb'] = 1  # Below lower band - BUY
        df.loc[df['close'] > df['bb_upper'], 'signal_bb'] = -1  # Above upper band - SELL
        
        # 5. Momentum Signal (Price vs EMA)
        df.loc[
            (df['close'] > df['ema_long']) & 
            (df['close'] > df['ema_short']), 
            'signal_momentum'
        ] = 1  # Strong uptrend - BUY
        
        df.loc[
            (df['close'] < df['ema_long']) & 
            (df['close'] < df['ema_short']), 
            'signal_momentum'
        ] = -1  # Strong downtrend - SELL
        
        # Aggregate Signal Score
        df['signal_score'] = (
            df['signal_ma_cross'] + 
            df['signal_rsi'] + 
            df['signal_macd'] + 
            df['signal_bb'] + 
            df['signal_momentum']
        )
        
        # Generate Final Signal
        df['signal'] = 'HOLD'
        df.loc[df['signal_score'] >= config.STRONG_BUY_SCORE, 'signal'] = 'STRONG BUY'
        df.loc[
            (df['signal_score'] >= config.BUY_SCORE) & 
            (df['signal_score'] < config.STRONG_BUY_SCORE), 
            'signal'
        ] = 'BUY'
        df.loc[
            (df['signal_score'] <= config.SELL_SCORE) & 
            (df['signal_score'] > config.STRONG_SELL_SCORE), 
            'signal'
        ] = 'SELL'
        df.loc[df['signal_score'] <= config.STRONG_SELL_SCORE, 'signal'] = 'STRONG SELL'
        
        print("✅ Signals generated!")
        return df
    
    @staticmethod
    def get_latest_signal(df: pd.DataFrame) -> Dict:
        """Get the most recent signal with details."""
        latest = df.iloc[-1]
        
        return {
            'date': latest.name,
            'close': latest['close'],
            'signal': latest['signal'],
            'score': latest['signal_score'],
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'sma_short': latest['sma_short'],
            'sma_long': latest['sma_long'],
        }


# ============================================================================
# CELL 6: RISK MANAGEMENT CLASS
# ============================================================================

class RiskManager:
    """
    Handles position sizing, stop loss, and take profit calculations.
    """
    
    def __init__(self, portfolio_value: float = config.INITIAL_CAPITAL):
        self.portfolio_value = portfolio_value
    
    def calculate_position_size(
        self, 
        current_price: float, 
        stop_loss_price: float,
        risk_per_trade_pct: float = config.DEFAULT_RISK_PER_TRADE
    ) -> Dict:
        """
        Calculate optimal position size based on risk management rules.
        
        Parameters:
        -----------
        current_price : float
            Current stock price
        stop_loss_price : float
            Stop loss price level
        risk_per_trade_pct : float
            Maximum % of portfolio to risk on this trade
            
        Returns:
        --------
        Dict with position sizing details
        """
        # Maximum amount to risk
        max_risk_amount = self.portfolio_value * (risk_per_trade_pct / 100)
        
        # Risk per share
        risk_per_share = abs(current_price - stop_loss_price)
        
        if risk_per_share == 0:
            return {'shares': 0, 'error': 'Invalid stop loss price'}
        
        # Calculate number of shares
        shares = int(max_risk_amount / risk_per_share)
        
        # Position value
        position_value = shares * current_price
        
        # Check maximum position size constraint
        max_position_value = self.portfolio_value * (config.MAX_POSITION_SIZE_PCT / 100)
        
        if position_value > max_position_value:
            shares = int(max_position_value / current_price)
            position_value = shares * current_price
        
        return {
            'shares': shares,
            'position_value': position_value,
            'position_pct': (position_value / self.portfolio_value) * 100,
            'risk_amount': shares * risk_per_share,
            'risk_pct': (shares * risk_per_share / self.portfolio_value) * 100
        }
    
    def calculate_stop_loss(
        self, 
        current_price: float, 
        atr: float,
        signal: str
    ) -> float:
        """
        Calculate stop loss price based on ATR.
        
        Parameters:
        -----------
        current_price : float
            Current stock price
        atr : float
            Average True Range value
        signal : str
            Trading signal (BUY or SELL)
            
        Returns:
        --------
        float
            Stop loss price
        """
        atr_multiplier = config.DEFAULT_STOP_LOSS_ATR
        
        if 'BUY' in signal:
            # For long positions, stop loss below current price
            stop_loss = current_price - (atr * atr_multiplier)
        else:
            # For short positions, stop loss above current price
            stop_loss = current_price + (atr * atr_multiplier)
        
        return stop_loss
    
    def calculate_take_profit(
        self, 
        current_price: float, 
        stop_loss_price: float,
        signal: str
    ) -> float:
        """
        Calculate take profit price based on risk-reward ratio.
        
        Parameters:
        -----------
        current_price : float
            Current stock price
        stop_loss_price : float
            Stop loss price
        signal : str
            Trading signal
            
        Returns:
        --------
        float
            Take profit price
        """
        risk = abs(current_price - stop_loss_price)
        reward = risk * config.DEFAULT_TAKE_PROFIT_RR
        
        if 'BUY' in signal:
            take_profit = current_price + reward
        else:
            take_profit = current_price - reward
        
        return take_profit
    
    def get_risk_metrics(
        self, 
        df: pd.DataFrame, 
        signal: str
    ) -> Dict:
        """
        Get comprehensive risk management metrics for current position.
        """
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        stop_loss = self.calculate_stop_loss(current_price, atr, signal)
        take_profit = self.calculate_take_profit(current_price, stop_loss, signal)
        
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return {
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'risk_reward_ratio': config.DEFAULT_TAKE_PROFIT_RR,
            **position_size
        }


# ============================================================================
# CELL 7: BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """
    Backtests the trading strategy on historical data.
    """
    
    def __init__(
        self, 
        initial_capital: float = config.INITIAL_CAPITAL,
        commission_pct: float = config.COMMISSION_PCT,
        slippage_pct: float = config.SLIPPAGE_PCT
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
    
    def run_backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Run backtest simulation on historical data.
        
        Returns:
        --------
        Tuple of (trades_df, performance_metrics)
        """
        print("🔄 Running backtest...")
        
        df = df.copy()
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Number of shares held
        entry_price = 0
        trades = []
        
        # Iterate through data
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check for signal changes
            if position == 0:  # No position
                if 'BUY' in current_row['signal']:
                    # Enter long position
                    shares = int(capital * 0.95 / current_row['close'])  # Use 95% of capital
                    entry_price = current_row['close'] * (1 + self.slippage_pct / 100)
                    cost = shares * entry_price
                    commission = cost * (self.commission_pct / 100)
                    
                    if cost + commission <= capital:
                        position = shares
                        capital -= (cost + commission)
                        
                        trades.append({
                            'date': current_row.name,
                            'type': 'BUY',
                            'price': entry_price,
                            'shares': shares,
                            'value': cost,
                            'commission': commission,
                            'capital': capital
                        })
            
            elif position > 0:  # In long position
                if 'SELL' in current_row['signal'] or current_row['signal'] == 'HOLD':
                    # Exit position on SELL signal
                    if 'SELL' in current_row['signal']:
                        exit_price = current_row['close'] * (1 - self.slippage_pct / 100)
                        proceeds = position * exit_price
                        commission = proceeds * (self.commission_pct / 100)
                        capital += (proceeds - commission)
                        
                        pnl = proceeds - (position * entry_price)
                        pnl_pct = (pnl / (position * entry_price)) * 100
                        
                        trades.append({
                            'date': current_row.name,
                            'type': 'SELL',
                            'price': exit_price,
                            'shares': position,
                            'value': proceeds,
                            'commission': commission,
                            'capital': capital,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct
                        })
                        
                        position = 0
                        entry_price = 0
        
        # Close any open position at the end
        if position > 0:
            final_price = df['close'].iloc[-1]
            proceeds = position * final_price
            commission = proceeds * (self.commission_pct / 100)
            capital += (proceeds - commission)
            
            pnl = proceeds - (position * entry_price)
            pnl_pct = (pnl / (position * entry_price)) * 100
            
            trades.append({
                'date': df.index[-1],
                'type': 'SELL',
                'price': final_price,
                'shares': position,
                'value': proceeds,
                'commission': commission,
                'capital': capital,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            # Filter completed trades (buy-sell pairs)
            sell_trades = trades_df[trades_df['type'] == 'SELL'].copy()
            
            total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
            num_trades = len(sell_trades)
            
            if num_trades > 0:
                winning_trades = sell_trades[sell_trades['pnl'] > 0]
                losing_trades = sell_trades[sell_trades['pnl'] <= 0]
                
                win_rate = (len(winning_trades) / num_trades) * 100
                avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
                
                # Calculate max drawdown
                equity_curve = [self.initial_capital]
                for _, trade in trades_df.iterrows():
                    equity_curve.append(trade['capital'])
                
                equity_series = pd.Series(equity_curve)
                rolling_max = equity_series.expanding().max()
                drawdown = (equity_series - rolling_max) / rolling_max * 100
                max_drawdown = drawdown.min()
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                max_drawdown = 0
            
            metrics = {
                'initial_capital': self.initial_capital,
                'final_capital': capital,
                'total_return_pct': total_return,
                'total_return_amount': capital - self.initial_capital,
                'num_trades': num_trades,
                'win_rate_pct': win_rate,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'max_drawdown_pct': max_drawdown,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
            }
        else:
            metrics = {
                'initial_capital': self.initial_capital,
                'final_capital': capital,
                'total_return_pct': 0,
                'total_return_amount': 0,
                'num_trades': 0,
                'win_rate_pct': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'max_drawdown_pct': 0,
                'profit_factor': 0
            }
        
        print("✅ Backtest completed!")
        return trades_df, metrics


# ============================================================================
# CELL 8: VISUALIZATION CLASS
# ============================================================================

class Visualizer:
    """
    Creates various charts and visualizations for analysis.
    """
    
    @staticmethod
    def plot_price_with_indicators(df: pd.DataFrame, symbol: str):
        """
        Plot price chart with technical indicators.
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{symbol} - Price & Moving Averages',
                'MACD',
                'RSI',
                'Volume'
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price and Moving Averages
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['sma_short'],
                name=f'SMA {config.SMA_SHORT}',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['sma_long'],
                name=f'SMA {config.SMA_LONG}',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['macd'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['macd_signal'],
                name='Signal',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['macd_diff'],
                name='Histogram',
                marker_color='gray'
            ),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['rsi'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_hline(
            y=config.RSI_OVERBOUGHT, line_dash="dash",
            line_color="red", row=3, col=1
        )
        fig.add_hline(
            y=config.RSI_OVERSOLD, line_dash="dash",
            line_color="green", row=3, col=1
        )
        
        # Volume
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                  for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color=colors
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)
        
        fig.show()
    
    @staticmethod
    def plot_signals(df: pd.DataFrame, symbol: str):
        """
        Plot price with buy/sell signals marked.
        """
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        # Buy signals
        buy_signals = df[df['signal'].str.contains('BUY', na=False)]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['close'],
            mode='markers',
            name='BUY Signal',
            marker=dict(
                color='green',
                size=15,
                symbol='triangle-up',
                line=dict(color='darkgreen', width=2)
            )
        ))
        
        # Sell signals
        sell_signals = df[df['signal'].str.contains('SELL', na=False)]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['close'],
            mode='markers',
            name='SELL Signal',
            marker=dict(
                color='red',
                size=15,
                symbol='triangle-down',
                line=dict(color='darkred', width=2)
            )
        ))
        
        fig.update_layout(
            title=f'{symbol} - Trading Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600,
            hovermode='x unified'
        )
        
        fig.show()
    
    @staticmethod
    def plot_backtest_results(trades_df: pd.DataFrame, metrics: Dict):
        """
        Plot backtest performance.
        """
        if len(trades_df) == 0:
            print("⚠️ No trades to plot")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Equity Curve',
                'Trade Returns Distribution',
                'Win/Loss Analysis',
                'Monthly Returns'
            )
        )
        
        # Equity curve
        equity_curve = trades_df['capital'].values
        equity_dates = trades_df['date'].values
        
        fig.add_trace(
            go.Scatter(
                x=equity_dates,
                y=equity_curve,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Trade returns distribution
        sell_trades = trades_df[trades_df['type'] == 'SELL']
        if len(sell_trades) > 0:
            fig.add_trace(
                go.Histogram(
                    x=sell_trades['pnl_pct'],
                    name='Returns',
                    nbinsx=20,
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            
            # Win/Loss pie chart
            wins = len(sell_trades[sell_trades['pnl'] > 0])
            losses = len(sell_trades[sell_trades['pnl'] <= 0])
            
            fig.add_trace(
                go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[wins, losses],
                    marker_colors=['green', 'red']
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=800, showlegend=False)
        fig.show()


print("✅ All classes initialized successfully!")


# ============================================================================
# CELL 9: ALERT SYSTEM
# ============================================================================

class AlertSystem:
    """
    Handles alert notifications for trading signals.
    """
    
    def __init__(self):
        self.alerts_history = []
    
    def send_alert(
        self, 
        symbol: str, 
        signal: str, 
        price: float, 
        metrics: Dict
    ):
        """
        Send alert for a trading signal.
        
        In production, this would integrate with:
        - Telegram Bot API
        - Email (SMTP)
        - SMS services
        - Slack webhooks
        """
        alert_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        alert_message = f"""
        🚨 TRADING ALERT 🚨
        
        Symbol: {symbol}
        Signal: {signal}
        Price: ${price:.2f}
        Time: {alert_time}
        
        📊 Technical Indicators:
        RSI: {metrics.get('rsi', 0):.2f}
        MACD: {metrics.get('macd', 0):.2f}
        Signal Score: {metrics.get('score', 0)}
        
        💰 Risk Management:
        Stop Loss: ${metrics.get('stop_loss', 0):.2f}
        Take Profit: ${metrics.get('take_profit', 0):.2f}
        Position Size: {metrics.get('shares', 0)} shares
        Risk Amount: ${metrics.get('risk_amount', 0):.2f}
        
        ⚠️ This is for informational purposes only. Not financial advice.
        """
        
        # Print alert (in production, send via Telegram/Email)
        print("="*70)
        print(alert_message)
        print("="*70)
        
        # Store in history
        self.alerts_history.append({
            'time': alert_time,
            'symbol': symbol,
            'signal': signal,
            'price': price,
            'metrics': metrics
        })
    
    def setup_telegram_bot(self, bot_token: str, chat_id: str):
        """
        Setup Telegram bot for alerts.
        
        To use:
        1. Create a bot via @BotFather on Telegram
        2. Get your bot token
        3. Get your chat_id
        4. Call this method with those credentials
        
        Example integration (uncomment to use):
        
        import requests
        
        def send_telegram_message(bot_token, chat_id, message):
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {"chat_id": chat_id, "text": message}
            requests.post(url, data=data)
        """
        self.telegram_bot_token = bot_token
        self.telegram_chat_id = chat_id
        print("✅ Telegram bot configured!")
    
    def setup_email_alerts(self, smtp_server: str, smtp_port: int, 
                          email: str, password: str, recipient: str):
        """
        Setup email alerts.
        
        Example integration (uncomment to use):
        
        import smtplib
        from email.mime.text import MIMEText
        
        def send_email(subject, body):
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = email
            msg['To'] = recipient
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(email, password)
                server.send_message(msg)
        """
        self.smtp_config = {
            'server': smtp_server,
            'port': smtp_port,
            'email': email,
            'password': password,
            'recipient': recipient
        }
        print("✅ Email alerts configured!")

alert_system = AlertSystem()
print("✅ Alert System initialized!")


# ============================================================================
# CELL 10: MAIN INVESTMENT ASSISTANT CLASS
# ============================================================================

class InvestmentAssistant:
    """
    Main orchestrator class that brings everything together.
    """
    
    def __init__(self, portfolio_value: float = config.INITIAL_CAPITAL):
        self.data_fetcher = MarketDataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(portfolio_value)
        self.backtest_engine = BacktestEngine()
        self.visualizer = Visualizer()
        self.alert_system = AlertSystem()
        
        print("🤖 Investment Assistant initialized!")
    
    def analyze_symbol(
        self, 
        symbol: str,
        period: str = config.DEFAULT_PERIOD,
        interval: str = config.DEFAULT_INTERVAL,
        run_backtest: bool = True,
        send_alerts: bool = True
    ) -> Dict:
        """
        Complete analysis pipeline for a given symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to analyze
        period : str
            Historical data period
        interval : str
            Data interval
        run_backtest : bool
            Whether to run backtest
        send_alerts : bool
            Whether to send alerts
            
        Returns:
        --------
        Dict with complete analysis results
        """
        print(f"\n{'='*70}")
        print(f"🔍 ANALYZING: {symbol}")
        print(f"{'='*70}\n")
        
        # Step 1: Fetch data
        df = self.data_fetcher.fetch_data(symbol, period, interval)
        if df.empty:
            return {'error': 'Failed to fetch data'}
        
        # Step 2: Get company info
        company_info = self.data_fetcher.get_company_info(symbol)
        print(f"📈 Company: {company_info.get('name', symbol)}")
        print(f"🏢 Sector: {company_info.get('sector', 'N/A')}")
        print(f"💼 Industry: {company_info.get('industry', 'N/A')}\n")
        
        # Step 3: Calculate technical indicators
        df = self.technical_indicators.calculate_all(df)
        
        # Step 4: Generate signals
        df = self.signal_generator.generate_signals(df)
        
        # Step 5: Get latest signal
        latest_signal = self.signal_generator.get_latest_signal(df)
        
        print(f"\n📊 LATEST SIGNAL: {latest_signal['signal']}")
        print(f"📅 Date: {latest_signal['date'].strftime('%Y-%m-%d')}")
        print(f"💵 Price: ${latest_signal['close']:.2f}")
        print(f"🎯 Signal Score: {latest_signal['score']}")
        print(f"📈 RSI: {latest_signal['rsi']:.2f}")
        print(f"📊 MACD: {latest_signal['macd']:.2f}")
        
        # Step 6: Calculate risk management
        risk_metrics = self.risk_manager.get_risk_metrics(
            df, 
            latest_signal['signal']
        )
        
        print(f"\n💰 RISK MANAGEMENT:")
        print(f"🛑 Stop Loss: ${risk_metrics['stop_loss']:.2f}")
        print(f"🎯 Take Profit: ${risk_metrics['take_profit']:.2f}")
        print(f"📊 Position Size: {risk_metrics['shares']} shares")
        print(f"💵 Position Value: ${risk_metrics['position_value']:.2f}")
        print(f"⚠️ Risk Amount: ${risk_metrics['risk_amount']:.2f} ({risk_metrics['risk_pct']:.2f}%)")
        
        # Step 7: Run backtest
        backtest_results = None
        if run_backtest:
            trades_df, metrics = self.backtest_engine.run_backtest(df)
            backtest_results = {'trades': trades_df, 'metrics': metrics}
            
            print(f"\n📈 BACKTEST RESULTS:")
            print(f"💰 Initial Capital: ${metrics['initial_capital']:,.2f}")
            print(f"💰 Final Capital: ${metrics['final_capital']:,.2f}")
            print(f"📊 Total Return: {metrics['total_return_pct']:.2f}%")
            print(f"🔢 Number of Trades: {metrics['num_trades']}")
            print(f"✅ Win Rate: {metrics['win_rate_pct']:.2f}%")
            print(f"📈 Avg Win: {metrics['avg_win_pct']:.2f}%")
            print(f"📉 Avg Loss: {metrics['avg_loss_pct']:.2f}%")
            print(f"📉 Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        # Step 8: Send alerts
        if send_alerts and config.ENABLE_ALERTS:
            if latest_signal['signal'] in config.ALERT_SIGNALS:
                alert_metrics = {**latest_signal, **risk_metrics}
                self.alert_system.send_alert(
                    symbol,
                    latest_signal['signal'],
                    latest_signal['close'],
                    alert_metrics
                )
        
        # Step 9: Visualizations
        print(f"\n📊 Generating visualizations...")
        self.visualizer.plot_price_with_indicators(df, symbol)
        self.visualizer.plot_signals(df, symbol)
        
        if backtest_results and len(backtest_results['trades']) > 0:
            self.visualizer.plot_backtest_results(
                backtest_results['trades'],
                backtest_results['metrics']
            )
        
        return {
            'symbol': symbol,
            'data': df,
            'company_info': company_info,
            'latest_signal': latest_signal,
            'risk_metrics': risk_metrics,
            'backtest': backtest_results
        }
    
    def compare_symbols(self, symbols: List[str]) -> pd.DataFrame:
        """
        Compare multiple symbols side by side.
        """
        results = []
        
        for symbol in symbols:
            df = self.data_fetcher.fetch_data(symbol, period="3mo")
            if not df.empty:
                df = self.technical_indicators.calculate_all(df)
                df = self.signal_generator.generate_signals(df)
                latest = self.signal_generator.get_latest_signal(df)
                
                results.append({
                    'Symbol': symbol,
                    'Price': latest['close'],
                    'Signal': latest['signal'],
                    'Score': latest['score'],
                    'RSI': latest['rsi'],
                    'MACD': latest['macd']
                })
        
        comparison_df = pd.DataFrame(results)
        return comparison_df


print("✅ Investment Assistant ready to use!")
print("""
To analyze a stock, use:

assistant = InvestmentAssistant(portfolio_value=100000)
results = assistant.analyze_symbol('AAPL')

Or compare multiple stocks:

comparison = assistant.compare_symbols(['AAPL', 'GOOGL', 'MSFT'])
print(comparison)
""")
