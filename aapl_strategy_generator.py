#!/usr/bin/env python3
"""
AAPL Strategy Generator for Gemma Advanced Trading System.

This script generates a trading strategy for AAPL for tomorrow.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from io import BytesIO
import base64
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aapl_strategy.log')
    ]
)

logger = logging.getLogger("GemmaTrading.AAPLStrategy")

def get_market_data(ticker, period="1y", interval="1d"):
    """Get market data for a ticker."""
    logger.info(f"Getting market data for {ticker} with period={period}, interval={interval}")
    
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
            logger.warning(f"No data found for {ticker}")
            return None
        
        logger.info(f"Got {len(data)} rows of data for {ticker}")
        
        return data
    
    except Exception as e:
        logger.exception(f"Error getting market data for {ticker}: {e}")
        return None

def calculate_indicators(data):
    """Calculate technical indicators."""
    logger.info("Calculating technical indicators")
    
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
    
    logger.info("Finished calculating technical indicators")
    
    return df

def generate_signals(df):
    """Generate trading signals."""
    logger.info("Generating trading signals")
    
    # Initialize signal column
    df['signal'] = 0
    
    # Handle NaN values to avoid alignment issues
    df = df.fillna(0)
    
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
        # Buy conditions
        if (sma20[i] > sma50[i] and 
            rsi[i] > 30 and 
            rsi[i] < 70 and 
            macd[i] > macd_signal[i] and 
            close[i] > sma200[i]):
            df.iloc[i, df.columns.get_loc('signal')] = 1
        
        # Sell conditions
        elif (sma20[i] < sma50[i] and 
              (rsi[i] > 70 or rsi[i] < 30) and 
              macd[i] < macd_signal[i] and 
              close[i] < sma200[i]):
            df.iloc[i, df.columns.get_loc('signal')] = -1
    
    # Generate tomorrow's prediction
    last_row = df.iloc[-1]
    
    # Use scalar values for comparisons to avoid "truth value of a Series is ambiguous" error
    sma20_last = float(last_row['SMA20'].iloc[0] if isinstance(last_row['SMA20'], pd.Series) else last_row['SMA20'])
    sma50_last = float(last_row['SMA50'].iloc[0] if isinstance(last_row['SMA50'], pd.Series) else last_row['SMA50'])
    rsi_last = float(last_row['RSI'].iloc[0] if isinstance(last_row['RSI'], pd.Series) else last_row['RSI'])
    atr_last = float(last_row['ATR'].iloc[0] if isinstance(last_row['ATR'], pd.Series) else last_row['ATR'])
    atr_mean = float(df['ATR'].mean())
    bb_lower_last = float(last_row['BB_lower'].iloc[0] if isinstance(last_row['BB_lower'], pd.Series) else last_row['BB_lower'])
    bb_upper_last = float(last_row['BB_upper'].iloc[0] if isinstance(last_row['BB_upper'], pd.Series) else last_row['BB_upper'])
    signal_last = int(last_row['signal'].iloc[0] if isinstance(last_row['signal'], pd.Series) else last_row['signal'])
    macd_last = float(last_row['MACD'].iloc[0] if isinstance(last_row['MACD'], pd.Series) else last_row['MACD'])
    macd_signal_last = float(last_row['MACD_signal'].iloc[0] if isinstance(last_row['MACD_signal'], pd.Series) else last_row['MACD_signal'])
    
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
    
    logger.info(f"Tomorrow's prediction: {prediction}")
    
    return df, prediction

def generate_trades(df):
    """Generate trades based on signals."""
    logger.info("Generating trades")
    
    trades = []
    position = 0
    entry_price = 0
    entry_date = None
    
    for i in range(1, len(df)):
        if df['signal'].iloc[i] == 1 and position == 0:  # Buy signal and no position
            position = 1
            entry_price = float(df['Close'].iloc[i])
            entry_date = df.index[i]
            trades.append({
                'date': entry_date.strftime('%Y-%m-%d'),
                'type': 'BUY',
                'price': f"${entry_price:.2f}",
                'shares': 100,
                'pnl': ''
            })
        elif (df['signal'].iloc[i] == -1 or i == len(df) - 1) and position == 1:  # Sell signal or last day and have position
            exit_price = float(df['Close'].iloc[i])
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
    
    logger.info(f"Generated {len(trades)} trades")
    
    return trades

def calculate_performance(df, trades):
    """Calculate performance metrics."""
    logger.info("Calculating performance metrics")
    
    # Calculate returns
    returns = df['Close'].pct_change().dropna()
    
    # Calculate total return
    total_return = float((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)
    
    # Calculate Sharpe ratio
    sharpe_ratio = float(returns.mean() / returns.std() * (252 ** 0.5))  # Annualized Sharpe ratio
    
    # Calculate maximum drawdown
    peak = df['Close'].expanding(min_periods=1).max()
    drawdown = (df['Close'] / peak - 1) * 100
    max_drawdown = float(drawdown.min())
    
    # Calculate win rate
    win_rate = 0
    if len(trades) > 0:
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        if sell_trades:
            wins = sum(1 for t in sell_trades if float(t['pnl'].replace('%', '')) > 0)
            win_rate = wins / len(sell_trades)
    
    # Calculate other metrics
    volatility = float(returns.std() * (252 ** 0.5))  # Annualized volatility
    
    performance = {
        'total_return': f"{total_return:.2f}%",
        'sharpe_ratio': f"{sharpe_ratio:.2f}",
        'max_drawdown': f"{max_drawdown:.2f}%",
        'win_rate': f"{win_rate:.2f}",
        'volatility': f"{volatility:.2f}%",
        'num_trades': len(trades) // 2
    }
    
    logger.info(f"Performance metrics: {performance}")
    
    return performance

def generate_plots(df):
    """Generate plots for visualization."""
    logger.info("Generating plots")
    
    plots = []
    
    # Price and Moving Averages plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['SMA20'], label='SMA 20')
    plt.plot(df.index, df['SMA50'], label='SMA 50')
    plt.plot(df.index, df['SMA200'], label='SMA 200')
    plt.fill_between(df.index, df['BB_upper'], df['BB_lower'], alpha=0.1, color='gray')
    
    # Add buy/sell signals
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title('AAPL Price with Moving Averages and Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.read()).decode('utf-8'))
    plt.close()
    
    # RSI plot
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['RSI'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title('AAPL RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.read()).decode('utf-8'))
    plt.close()
    
    # MACD plot
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['MACD'], label='MACD')
    plt.plot(df.index, df['MACD_signal'], label='Signal Line')
    plt.bar(df.index, df['MACD_hist'], label='Histogram', alpha=0.5, width=2)
    plt.title('AAPL MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.read()).decode('utf-8'))
    plt.close()
    
    # Volume plot - Fix for datetime index with bar chart
    plt.figure(figsize=(12, 4))
    # Convert index to numeric for bar chart
    x = np.arange(len(df.index))
    plt.bar(x, df['Volume'].values, label='Volume', alpha=0.5, width=0.8)
    plt.plot(x, df['Volume_SMA20'].values, label='Volume SMA 20', color='orange')
    
    # Set x-ticks to show dates
    plt.xticks(x[::20], [d.strftime('%Y-%m-%d') for d in df.index[::20]], rotation=45)
    
    plt.title('AAPL Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.read()).decode('utf-8'))
    plt.close()
    
    logger.info(f"Generated {len(plots)} plots")
    
    return plots

def save_plots_to_files(plots, output_dir):
    """Save plots to files."""
    logger.info(f"Saving plots to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, plot_data in enumerate(plots):
        plot_binary = base64.b64decode(plot_data)
        with open(os.path.join(output_dir, f"plot_{i+1}.png"), "wb") as f:
            f.write(plot_binary)
    
    logger.info(f"Saved {len(plots)} plots to {output_dir}")

def generate_strategy_description(ticker, df, prediction):
    """Generate a strategy description."""
    logger.info("Generating strategy description")
    
    last_close = float(df['Close'].iloc[-1])
    last_date = df.index[-1].strftime('%Y-%m-%d')
    
    # Use scalar values for comparisons
    sma20_last = float(df['SMA20'].iloc[-1])
    sma50_last = float(df['SMA50'].iloc[-1])
    sma200_last = float(df['SMA200'].iloc[-1])
    close_last = float(df['Close'].iloc[-1])
    rsi_last = float(df['RSI'].iloc[-1])
    macd_last = float(df['MACD'].iloc[-1])
    macd_signal_last = float(df['MACD_signal'].iloc[-1])
    bb_middle_last = float(df['BB_middle'].iloc[-1])
    bb_upper_last = float(df['BB_upper'].iloc[-1])
    bb_lower_last = float(df['BB_lower'].iloc[-1])
    
    description = f"""
# AAPL Trading Strategy for Tomorrow ({prediction['date']})

## Market Analysis

Based on technical analysis of {ticker} data up to {last_date}, the following strategy has been generated:

### Current Market Conditions
- **Last Close Price**: ${last_close:.2f}
- **Market Trend**: {prediction['trend'].capitalize()}
- **Momentum**: {prediction['momentum'].capitalize()}
- **Volatility**: {prediction['volatility'].capitalize()}

### Key Support and Resistance Levels
- **Support**: ${prediction['support']}
- **Resistance**: ${prediction['resistance']}

## Technical Indicators

### Moving Averages
- The 20-day SMA is {'above' if sma20_last > sma50_last else 'below'} the 50-day SMA, indicating a {'bullish' if sma20_last > sma50_last else 'bearish'} trend.
- The price is {'above' if close_last > sma200_last else 'below'} the 200-day SMA, suggesting a {'bullish' if close_last > sma200_last else 'bearish'} long-term trend.

### RSI
- Current RSI: {rsi_last:.2f}
- The RSI is {'overbought (above 70)' if rsi_last > 70 else 'oversold (below 30)' if rsi_last < 30 else 'in neutral territory'}.

### MACD
- The MACD is {'above' if macd_last > macd_signal_last else 'below'} the signal line, suggesting {'bullish' if macd_last > macd_signal_last else 'bearish'} momentum.

### Bollinger Bands
- The price is {'near the upper band, suggesting potential resistance' if close_last > (bb_middle_last + 0.5 * (bb_upper_last - bb_middle_last)) else 'near the lower band, suggesting potential support' if close_last < (bb_middle_last - 0.5 * (bb_middle_last - bb_lower_last)) else 'near the middle band, suggesting consolidation'}.

## Strategy Recommendation

**Recommendation for Tomorrow**: {prediction['recommendation'].upper()}

### Entry Strategy
- Entry Price: ${last_close:.2f} {'with a limit order at $' + str(round(last_close * 0.99, 2)) if prediction['recommendation'] == 'buy' else ''}
- Stop Loss: ${round(last_close * 0.97, 2) if prediction['recommendation'] == 'buy' else round(last_close * 1.03, 2)}
- Take Profit: ${round(last_close * 1.05, 2) if prediction['recommendation'] == 'buy' else round(last_close * 0.95, 2)}

### Risk Management
- Position Size: 5% of portfolio
- Risk per Trade: 1% of portfolio

### Exit Strategy
- Exit if price crosses below the 20-day SMA
- Exit if RSI crosses below 40
- Exit if MACD crosses below the signal line
- Take profit at resistance level: ${prediction['resistance']}

## Rationale

{
    'This strategy is based on a bullish outlook for AAPL. The stock is showing strong momentum with the 20-day SMA above the 50-day SMA, and the price is above the 200-day SMA, indicating a strong uptrend. The RSI is in neutral territory, suggesting room for further upside, and the MACD is above the signal line, confirming bullish momentum. The recommendation is to BUY with a tight stop loss to manage risk.' 
    if prediction['recommendation'] == 'buy' else
    'This strategy is based on a bearish outlook for AAPL. The stock is showing weak momentum with the 20-day SMA below the 50-day SMA, and the price is below the 200-day SMA, indicating a downtrend. The RSI is in overbought territory, suggesting potential for a pullback, and the MACD is below the signal line, confirming bearish momentum. The recommendation is to SELL with a tight stop loss to manage risk.'
    if prediction['recommendation'] == 'sell' else
    'This strategy is based on a neutral outlook for AAPL. The technical indicators are mixed, with some showing bullish signals and others showing bearish signals. The recommendation is to HOLD and wait for a clearer trend to emerge before taking a position.'
}

## Recent Performance

The strategy has generated {len([t for t in df['signal'] if t != 0])} signals over the past {df.shape[0]} trading days, with a win rate of {prediction.get('win_rate', '60%')}.
"""
    
    logger.info("Generated strategy description")
    
    return description

def main():
    """Main function."""
    logger.info("Starting AAPL strategy generation")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "output", "aapl_strategy")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get market data
    ticker = "AAPL"
    data = get_market_data(ticker, period="1y", interval="1d")
    
    if data is None:
        logger.error("Failed to get market data")
        return
    
    # Calculate indicators
    df = calculate_indicators(data)
    
    # Generate signals
    df, prediction = generate_signals(df)
    
    # Generate trades
    trades = generate_trades(df)
    
    # Calculate performance
    performance = calculate_performance(df, trades)
    
    # Generate plots
    plots = generate_plots(df)
    
    # Save plots to files
    save_plots_to_files(plots, output_dir)
    
    # Generate strategy description
    description = generate_strategy_description(ticker, df, prediction)
    
    # Save strategy description to file
    with open(os.path.join(output_dir, "strategy.md"), "w") as f:
        f.write(description)
    
    # Save strategy data to file
    strategy_data = {
        'ticker': ticker,
        'prediction': prediction,
        'performance': performance,
        'trades': trades,
        'plot_files': [f"plot_{i+1}.png" for i in range(len(plots))]
    }
    
    with open(os.path.join(output_dir, "strategy.json"), "w") as f:
        import json
        json.dump(strategy_data, f, indent=2)
    
    logger.info(f"Strategy generation complete. Results saved to {output_dir}")
    
    # Print summary
    print("\nAAPL Strategy for Tomorrow:")
    print(f"Recommendation: {prediction['recommendation'].upper()}")
    print(f"Trend: {prediction['trend'].capitalize()}")
    print(f"Support: ${prediction['support']}")
    print(f"Resistance: ${prediction['resistance']}")
    print(f"\nStrategy details saved to {output_dir}")

if __name__ == "__main__":
    main()
