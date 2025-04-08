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

# Import the continuous research engine
from backend.continuous_research import continuous_research_engine

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create templates directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

# Create static directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)

# Create research_data directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'research_data'), exist_ok=True)

@app.route('/')
def index():
    # Get continuous research status
    research_status = continuous_research_engine.get_status()
    return render_template('index.html', research_status=research_status)

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
    
    # Get research data if continuous research is active
    research_data = None
    if continuous_research_engine.active:
        research_data = continuous_research_engine.get_latest_research(ticker)
    
    # Return results
    return jsonify({
        'ticker': ticker,
        'prediction': prediction,
        'performance': performance,
        'trades': trades,
        'plot_urls': plot_urls,
        'description': description,
        'research_data': research_data
    })

@app.route('/toggle_research', methods=['POST'])
def toggle_research():
    """Toggle continuous research on/off."""
    action = request.form.get('action', 'toggle')
    tickers = request.form.get('tickers', '').split(',')
    tickers = [t.strip() for t in tickers if t.strip()]
    
    if action == 'start' or (action == 'toggle' and not continuous_research_engine.active):
        # Start continuous research
        interval = int(request.form.get('interval', 3600))  # Default: 1 hour
        success = continuous_research_engine.start(tickers=tickers, interval=interval)
        message = "Continuous research started" if success else "Failed to start continuous research"
    elif action == 'stop' or (action == 'toggle' and continuous_research_engine.active):
        # Stop continuous research
        success = continuous_research_engine.stop()
        message = "Continuous research stopped" if success else "Failed to stop continuous research"
    else:
        success = False
        message = "Invalid action"
    
    # Get current status
    status = continuous_research_engine.get_status()
    
    return jsonify({
        'success': success,
        'message': message,
        'status': status
    })

@app.route('/research_status', methods=['GET'])
def research_status():
    """Get continuous research status."""
    status = continuous_research_engine.get_status()
    return jsonify(status)

@app.route('/latest_research', methods=['GET'])
def latest_research():
    """Get latest research data."""
    ticker = request.args.get('ticker')
    research = continuous_research_engine.get_latest_research(ticker)
    return jsonify(research)

@app.route('/add_tickers', methods=['POST'])
def add_tickers():
    """Add tickers to watch list."""
    tickers = request.form.get('tickers', '').split(',')
    tickers = [t.strip() for t in tickers if t.strip()]
    
    if not tickers:
        return jsonify({'success': False, 'message': 'No tickers provided'})
    
    success = continuous_research_engine.add_tickers(tickers)
    message = f"Added tickers: {', '.join(tickers)}" if success else "Failed to add tickers"
    
    return jsonify({
        'success': success,
        'message': message,
        'status': continuous_research_engine.get_status()
    })

@app.route('/remove_tickers', methods=['POST'])
def remove_tickers():
    """Remove tickers from watch list."""
    tickers = request.form.get('tickers', '').split(',')
    tickers = [t.strip() for t in tickers if t.strip()]
    
    if not tickers:
        return jsonify({'success': False, 'message': 'No tickers provided'})
    
    success = continuous_research_engine.remove_tickers(tickers)
    message = f"Removed tickers: {', '.join(tickers)}" if success else "Failed to remove tickers"
    
    return jsonify({
        'success': success,
        'message': message,
        'status': continuous_research_engine.get_status()
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
    
    # Get research data if available
    research_data = None
    if continuous_research_engine.active:
        research_data = continuous_research_engine.get_latest_research(ticker)
    
    # Generate description
    description = f"## {ticker} Trading Strategy\n\n"
    description += f"As of {last_date}, {ticker} closed at ${last_close:.2f}. "
    
    # Technical analysis
    if sma20_last > sma50_last:
        description += f"The stock is in an **uptrend** with the 20-day SMA (${sma20_last:.2f}) above the 50-day SMA (${sma50_last:.2f}). "
    else:
        description += f"The stock is in a **downtrend** with the 20-day SMA (${sma20_last:.2f}) below the 50-day SMA (${sma50_last:.2f}). "
    
    if close_last > sma200_last:
        description += f"It is trading above its 200-day SMA (${sma200_last:.2f}), indicating a **bullish** long-term trend. "
    else:
        description += f"It is trading below its 200-day SMA (${sma200_last:.2f}), indicating a **bearish** long-term trend. "
    
    # RSI analysis
    if rsi_last > 70:
        description += f"The RSI is at {rsi_last:.2f}, suggesting the stock is **overbought**. "
    elif rsi_last < 30:
        description += f"The RSI is at {rsi_last:.2f}, suggesting the stock is **oversold**. "
    else:
        description += f"The RSI is at {rsi_last:.2f}, in the **neutral** zone. "
    
    # MACD analysis
    if macd_last > macd_signal_last:
        description += f"The MACD ({macd_last:.2f}) is above the signal line ({macd_signal_last:.2f}), giving a **bullish** signal. "
    else:
        description += f"The MACD ({macd_last:.2f}) is below the signal line ({macd_signal_last:.2f}), giving a **bearish** signal. "
    
    # Add research insights if available
    if research_data and 'sentiment' in research_data:
        sentiment = research_data['sentiment']
        sentiment_desc = "neutral"
        if sentiment > 0.2:
            sentiment_desc = "positive"
        elif sentiment < -0.2:
            sentiment_desc = "negative"
        
        description += f"\n\n### Market Research Insights\n\n"
        description += f"Recent news sentiment for {ticker} is **{sentiment_desc}** with a score of {sentiment:.2f}. "
        
        if 'news' in research_data and research_data['news']:
            description += f"Here are some recent headlines:\n\n"
            for i, news in enumerate(research_data['news'][:3]):
                description += f"- {news['headline']}\n"
    
    # Strategy recommendation
    description += f"\n\n### Strategy Recommendation\n\n"
    description += f"Based on technical analysis, the recommendation for {ticker} is to **{prediction['recommendation'].upper()}**. "
    description += f"The stock is in a {prediction['trend']} trend with {prediction['momentum']} momentum and {prediction['volatility']} volatility. "
    description += f"Key support level is at ${prediction['support']:.2f} and resistance at ${prediction['resistance']:.2f}."
    
    return description

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
