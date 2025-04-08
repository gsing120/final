from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from io import BytesIO
import base64
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create templates directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

# Create static directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_strategy', methods=['POST'])
def generate_strategy():
    ticker = request.form.get('ticker', 'AAPL')
    strategy_type = request.form.get('strategy_type', 'swing')
    
    # Get market data
    data = get_market_data(ticker, period="1y", interval="1d")
    
    if data is None:
        return jsonify({'error': 'Failed to get market data'})
    
    # Calculate indicators
    df = calculate_indicators(data)
    
    # Generate signals
    df, prediction = generate_signals(df)
    
    # Generate trades
    trades = generate_trades(df)
    
    # Calculate performance
    performance = calculate_performance(df, trades)
    
    # Generate plots
    plot_urls = generate_plots(df)
    
    # Generate strategy description
    description = generate_strategy_description(ticker, df, prediction)
    
    # Return results
    return jsonify({
        'ticker': ticker,
        'prediction': prediction,
        'performance': performance,
        'trades': trades,
        'plot_urls': plot_urls,
        'description': description
    })

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
        print(f"Error getting market data for {ticker}: {e}")
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
    
    return df

def generate_signals(df):
    """Generate trading signals."""
    
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
    
    return df, prediction

def generate_trades(df):
    """Generate trades based on signals."""
    
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
    
    return trades

def calculate_performance(df, trades):
    """Calculate performance metrics."""
    
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
    
    return performance

def generate_plots(df):
    """Generate plots for visualization."""
    
    plot_urls = []
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
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
    
    plt.title('Price with Moving Averages and Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    price_plot_path = os.path.join('static', 'price_plot.png')
    plt.savefig(os.path.join(os.path.dirname(__file__), price_plot_path))
    plt.close()
    plot_urls.append(price_plot_path)
    
    # RSI plot
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['RSI'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title('RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    rsi_plot_path = os.path.join('static', 'rsi_plot.png')
    plt.savefig(os.path.join(os.path.dirname(__file__), rsi_plot_path))
    plt.close()
    plot_urls.append(rsi_plot_path)
    
    # MACD plot
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['MACD'], label='MACD')
    plt.plot(df.index, df['MACD_signal'], label='Signal Line')
    
    # Use numeric x-axis for bar chart to avoid datetime issues
    x = np.arange(len(df.index))
    plt.bar(x, df['MACD_hist'].values, label='Histogram', alpha=0.5, width=0.8)
    
    # Set x-ticks to show dates
    plt.xticks(x[::20], [d.strftime('%Y-%m-%d') for d in df.index[::20]], rotation=45)
    
    plt.title('MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    macd_plot_path = os.path.join('static', 'macd_plot.png')
    plt.savefig(os.path.join(os.path.dirname(__file__), macd_plot_path))
    plt.close()
    plot_urls.append(macd_plot_path)
    
    # Volume plot - Fix for datetime index with bar chart
    plt.figure(figsize=(12, 4))
    # Convert index to numeric for bar chart
    x = np.arange(len(df.index))
    plt.bar(x, df['Volume'].values, label='Volume', alpha=0.5, width=0.8)
    plt.plot(x, df['Volume_SMA20'].values, label='Volume SMA 20', color='orange')
    
    # Set x-ticks to show dates
    plt.xticks(x[::20], [d.strftime('%Y-%m-%d') for d in df.index[::20]], rotation=45)
    
    plt.title('Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    volume_plot_path = os.path.join('static', 'volume_plot.png')
    plt.savefig(os.path.join(os.path.dirname(__file__), volume_plot_path))
    plt.close()
    plot_urls.append(volume_plot_path)
    
    return plot_urls

def generate_strategy_description(ticker, df, prediction):
    """Generate a strategy description."""
    
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
# {ticker} Trading Strategy for Tomorrow ({prediction['date']})

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
    'This strategy is based on a bullish outlook for ' + ticker + '. The stock is showing strong momentum with the 20-day SMA above the 50-day SMA, and the price is above the 200-day SMA, indicating a strong uptrend. The RSI is in neutral territory, suggesting room for further upside, and the MACD is above the signal line, confirming bullish momentum. The recommendation is to BUY with a tight stop loss to manage risk.' 
    if prediction['recommendation'] == 'buy' else
    'This strategy is based on a bearish outlook for ' + ticker + '. The stock is showing weak momentum with the 20-day SMA below the 50-day SMA, and the price is below the 200-day SMA, indicating a downtrend. The RSI is in overbought territory, suggesting potential for a pullback, and the MACD is below the signal line, confirming bearish momentum. The recommendation is to SELL with a tight stop loss to manage risk.'
    if prediction['recommendation'] == 'sell' else
    'This strategy is based on a neutral outlook for ' + ticker + '. The technical indicators are mixed, with some showing bullish signals and others showing bearish signals. The recommendation is to HOLD and wait for a clearer trend to emerge before taking a position.'
}
"""
    
    return description

if __name__ == '__main__':
    # Create index.html template
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gemma Advanced Trading System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, select {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: none;
            }
            .loading {
                text-align: center;
                display: none;
            }
            .plot-container {
                margin-top: 20px;
                text-align: center;
            }
            .plot-container img {
                max-width: 100%;
                height: auto;
                margin-bottom: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .metrics {
                display: flex;
                flex-wrap: wrap;
                margin-top: 20px;
            }
            .metric {
                flex: 1 0 30%;
                margin: 5px;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 4px;
                text-align: center;
            }
            .metric h3 {
                margin-top: 0;
                color: #333;
            }
            .metric p {
                font-size: 24px;
                font-weight: bold;
                margin: 5px 0;
            }
            .recommendation {
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                border-radius: 4px;
                font-size: 24px;
                font-weight: bold;
            }
            .buy {
                background-color: #d4edda;
                color: #155724;
            }
            .sell {
                background-color: #f8d7da;
                color: #721c24;
            }
            .hold {
                background-color: #fff3cd;
                color: #856404;
            }
            .description {
                white-space: pre-line;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Gemma Advanced Trading System</h1>
            <div class="form-group">
                <label for="ticker">Ticker Symbol:</label>
                <input type="text" id="ticker" name="ticker" value="AAPL" required>
            </div>
            <div class="form-group">
                <label for="strategy_type">Strategy Type:</label>
                <select id="strategy_type" name="strategy_type">
                    <option value="swing">Swing Trading</option>
                    <option value="day">Day Trading</option>
                    <option value="position">Position Trading</option>
                </select>
            </div>
            <button id="generate-btn">Generate Strategy</button>
            
            <div class="loading" id="loading">
                <p>Generating strategy... Please wait.</p>
                <img src="https://i.gifer.com/origin/b4/b4d657e7ef262b88eb5f7ac021edda87.gif" alt="Loading" width="50">
            </div>
            
            <div class="result" id="result">
                <div class="recommendation" id="recommendation"></div>
                
                <div class="metrics">
                    <div class="metric">
                        <h3>Trend</h3>
                        <p id="trend"></p>
                    </div>
                    <div class="metric">
                        <h3>Momentum</h3>
                        <p id="momentum"></p>
                    </div>
                    <div class="metric">
                        <h3>Volatility</h3>
                        <p id="volatility"></p>
                    </div>
                    <div class="metric">
                        <h3>Support</h3>
                        <p id="support"></p>
                    </div>
                    <div class="metric">
                        <h3>Resistance</h3>
                        <p id="resistance"></p>
                    </div>
                    <div class="metric">
                        <h3>Total Return</h3>
                        <p id="total-return"></p>
                    </div>
                    <div class="metric">
                        <h3>Sharpe Ratio</h3>
                        <p id="sharpe-ratio"></p>
                    </div>
                    <div class="metric">
                        <h3>Max Drawdown</h3>
                        <p id="max-drawdown"></p>
                    </div>
                    <div class="metric">
                        <h3>Win Rate</h3>
                        <p id="win-rate"></p>
                    </div>
                </div>
                
                <div class="plot-container" id="plots"></div>
                
                <h2>Trade History</h2>
                <table id="trades">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Type</th>
                            <th>Price</th>
                            <th>Shares</th>
                            <th>P&L</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
                
                <h2>Strategy Description</h2>
                <div class="description" id="description"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('generate-btn').addEventListener('click', function() {
                const ticker = document.getElementById('ticker').value;
                const strategyType = document.getElementById('strategy_type').value;
                
                if (!ticker) {
                    alert('Please enter a ticker symbol');
                    return;
                }
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                const formData = new FormData();
                formData.append('ticker', ticker);
                formData.append('strategy_type', strategyType);
                
                fetch('/generate_strategy', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('result').style.display = 'block';
                    
                    // Update recommendation
                    const recommendationEl = document.getElementById('recommendation');
                    recommendationEl.textContent = data.prediction.recommendation.toUpperCase();
                    recommendationEl.className = 'recommendation ' + data.prediction.recommendation.toLowerCase();
                    
                    // Update metrics
                    document.getElementById('trend').textContent = data.prediction.trend.charAt(0).toUpperCase() + data.prediction.trend.slice(1);
                    document.getElementById('momentum').textContent = data.prediction.momentum.charAt(0).toUpperCase() + data.prediction.momentum.slice(1);
                    document.getElementById('volatility').textContent = data.prediction.volatility.charAt(0).toUpperCase() + data.prediction.volatility.slice(1);
                    document.getElementById('support').textContent = '$' + data.prediction.support;
                    document.getElementById('resistance').textContent = '$' + data.prediction.resistance;
                    document.getElementById('total-return').textContent = data.performance.total_return;
                    document.getElementById('sharpe-ratio').textContent = data.performance.sharpe_ratio;
                    document.getElementById('max-drawdown').textContent = data.performance.max_drawdown;
                    document.getElementById('win-rate').textContent = data.performance.win_rate;
                    
                    // Update plots
                    const plotsContainer = document.getElementById('plots');
                    plotsContainer.innerHTML = '';
                    data.plot_urls.forEach(url => {
                        const img = document.createElement('img');
                        img.src = url;
                        img.alt = 'Strategy Plot';
                        plotsContainer.appendChild(img);
                    });
                    
                    // Update trades table
                    const tradesBody = document.getElementById('trades').getElementsByTagName('tbody')[0];
                    tradesBody.innerHTML = '';
                    data.trades.forEach(trade => {
                        const row = tradesBody.insertRow();
                        row.insertCell(0).textContent = trade.date;
                        row.insertCell(1).textContent = trade.type;
                        row.insertCell(2).textContent = trade.price;
                        row.insertCell(3).textContent = trade.shares;
                        row.insertCell(4).textContent = trade.pnl;
                    });
                    
                    // Update description
                    document.getElementById('description').textContent = data.description;
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    alert('Error generating strategy: ' + error);
                });
            });
        </script>
    </body>
    </html>
    """
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Write index.html template
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
