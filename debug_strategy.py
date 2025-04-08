import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StrategyDebugger")

def get_market_data(ticker, period="1y", interval="1d"):
    """Get market data for a ticker."""
    try:
        # Get data from Yahoo Finance
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False
        )
        
        # Check if data is empty
        if data.empty:
            return None
        
        return data
    
    except Exception as e:
        logger.error(f"Error getting market data for {ticker}: {e}")
        return None

def calculate_indicators(data):
    """Calculate technical indicators."""
    
    # Create a copy of the dataframe to avoid alignment issues
    df = data.copy()
    
    # Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD_fast'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['MACD_slow'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['MACD_fast'] - df['MACD_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Volume indicators
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    
    # Fill NaN values with 0 to avoid issues
    df = df.fillna(0)
    
    return df

def generate_signals(df):
    """Generate trading signals."""
    
    # Initialize signal column
    df['signal'] = 0
    
    # Use numpy arrays for comparisons to avoid alignment issues
    sma20 = df['SMA20'].values
    sma50 = df['SMA50'].values
    sma200 = df['SMA200'].values
    close = df['Close'].values
    rsi = df['RSI'].values
    macd = df['MACD'].values
    macd_signal = df['MACD_signal'].values
    
    # Generate buy signals
    for i in range(len(df)):
        # Buy conditions - use individual comparisons to avoid Series truth value ambiguity
        buy_condition1 = sma20[i] > sma50[i]
        buy_condition2 = rsi[i] > 30
        buy_condition3 = rsi[i] < 70
        buy_condition4 = macd[i] > macd_signal[i]
        buy_condition5 = close[i] > sma200[i]
        
        if (buy_condition1 and buy_condition2 and buy_condition3 and 
            buy_condition4 and buy_condition5):
            df.iloc[i, df.columns.get_loc('signal')] = 1
        
        # Sell conditions - use individual comparisons to avoid Series truth value ambiguity
        else:
            sell_condition1 = sma20[i] < sma50[i]
            sell_condition2 = (rsi[i] > 70 or rsi[i] < 30)
            sell_condition3 = macd[i] < macd_signal[i]
            sell_condition4 = close[i] < sma200[i]
            
            if (sell_condition1 and sell_condition2 and 
                sell_condition3 and sell_condition4):
                df.iloc[i, df.columns.get_loc('signal')] = -1
    
    # Generate tomorrow's prediction
    last_row = df.iloc[-1]
    
    # Extract scalar values safely from the last row
    sma20_last = float(last_row['SMA20'].item())
    sma50_last = float(last_row['SMA50'].item())
    rsi_last = float(last_row['RSI'].item())
    atr_last = float(last_row['ATR'].item())
    atr_mean = float(df['ATR'].mean())
    bb_lower_last = float(last_row['BB_lower'].item())
    bb_upper_last = float(last_row['BB_upper'].item())
    signal_last = int(last_row['signal'].item())
    macd_last = float(last_row['MACD'].item())
    macd_signal_last = float(last_row['MACD_signal'].item())
    
    prediction = {
        'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'trend': 'bullish' if sma20_last > sma50_last else 'bearish',
        'momentum': 'strong' if rsi_last > 50 else 'weak',
        'volatility': 'high' if atr_last > atr_mean else 'low',
        'support': round(bb_lower_last, 2),
        'resistance': round(bb_upper_last, 2),
        'recommendation': 'buy' if signal_last == 1 else ('sell' if signal_last == -1 else 'hold')
    }
    
    # If no signal was generated for the last day, use a simplified approach
    if prediction['recommendation'] == 'hold':
        if sma20_last > sma50_last and rsi_last < 70:
            prediction['recommendation'] = 'buy'
        elif sma20_last < sma50_last and rsi_last > 30:
            prediction['recommendation'] = 'sell'
    
    return df, prediction

def generate_trades(df):
    """Generate trades based on signals."""
    
    trades = []
    position = 0
    entry_price = 0
    entry_date = None
    
    for i in range(1, len(df)):
        # Use scalar comparison to avoid Series truth value ambiguity
        signal_value = int(df['signal'].iloc[i])
        
        if signal_value == 1 and position == 0:  # Buy signal and no position
            position = 1
            entry_price = float(df['Close'].iloc[i].item())
            entry_date = df.index[i]
            trades.append({
                'date': entry_date.strftime('%Y-%m-%d'),
                'type': 'BUY',
                'price': f"${entry_price:.2f}",
                'shares': 100,
                'pnl': ''
            })
        elif (signal_value == -1 or i == len(df) - 1) and position == 1:  # Sell signal or last day and have position
            exit_price = float(df['Close'].iloc[i].item())
            exit_date = df.index[i]
            pnl = (exit_price - entry_price) / entry_price * 100
            trades.append({
                'date': exit_date.strftime('%Y-%m-%d'),
                'type': 'SELL',
                'price': f"${exit_price:.2f}",
                'shares': 100,
                'pnl': f"{pnl:.2f}%"
            })
            position = 0
    
    return trades

def calculate_performance(df, trades):
    """Calculate performance metrics."""
    
    # Calculate returns
    returns = df['Close'].pct_change().dropna()
    
    # Calculate total return
    first_close = float(df['Close'].iloc[0].item())
    last_close = float(df['Close'].iloc[-1].item())
    total_return = (last_close - first_close) / first_close * 100
    
    # Calculate Sharpe ratio (simplified)
    returns_mean = float(returns.mean())
    returns_std = float(returns.std())
    sharpe_ratio = (returns_mean / returns_std * np.sqrt(252)) if returns_std > 0 else 0.0
    
    # Calculate max drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = float(drawdown.min())
    
    # Calculate win rate
    if len(trades) > 1:
        win_count = sum(1 for trade in trades if trade['type'] == 'SELL' and float(trade['pnl'].strip('%')) > 0)
        total_trades = sum(1 for trade in trades if trade['type'] == 'SELL')
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
    else:
        win_rate = 0.0
    
    # Calculate volatility
    volatility = float(returns.std() * np.sqrt(252) * 100)
    
    performance = {
        'total_return': f"{total_return:.2f}%",
        'sharpe_ratio': f"{sharpe_ratio:.2f}",
        'max_drawdown': f"{max_drawdown:.2f}%",
        'win_rate': f"{win_rate:.2f}%",
        'volatility': f"{volatility:.2f}%"
    }
    
    return performance

def test_strategy_generation(ticker="AAPL"):
    """Test the strategy generation process."""
    try:
        print(f"Generating strategy for {ticker}...")
        
        # Get market data
        data = get_market_data(ticker, period="1y", interval="1d")
        
        if data is None:
            print("Failed to get market data")
            return
        
        print(f"Got market data with {len(data)} rows")
        
        # Calculate indicators
        df = calculate_indicators(data)
        print("Calculated indicators")
        
        # Generate signals
        df, prediction = generate_signals(df)
        print("Generated signals")
        print(f"Prediction: {prediction}")
        
        # Generate trades
        trades = generate_trades(df)
        print(f"Generated {len(trades)} trades")
        
        # Calculate performance
        performance = calculate_performance(df, trades)
        print("Calculated performance")
        print(f"Performance: {performance}")
        
        print("Strategy generation successful!")
        
        # Return the results for API testing
        return {
            'success': True,
            'ticker': ticker,
            'prediction': prediction,
            'performance': performance,
            'trades': trades
        }
        
    except Exception as e:
        print(f"Error generating strategy: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    test_strategy_generation("AAPL")
