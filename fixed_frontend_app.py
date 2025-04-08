from flask import Flask, render_template, request, jsonify, redirect, url_for
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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('frontend.log')
    ]
)

logger = logging.getLogger("GemmaTrading.Frontend")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SECRET_KEY'] = 'gemma_advanced_trading_system'

# Create directories if they don't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'js'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'img'), exist_ok=True)

# Sample portfolio data
portfolio = {
    'cash': 100000.00,
    'positions': [
        {'ticker': 'AAPL', 'shares': 100, 'entry_price': 175.25, 'current_price': 182.50},
        {'ticker': 'MSFT', 'shares': 50, 'entry_price': 310.75, 'current_price': 325.30},
        {'ticker': 'GOOGL', 'shares': 25, 'entry_price': 135.50, 'current_price': 142.75},
    ],
    'watchlist': ['AMZN', 'NVDA', 'TSLA', 'META', 'AMD']
}

# Sample market scanner results
scanner_results = [
    {'ticker': 'AAPL', 'signal': 'BUY', 'strength': 'Strong', 'pattern': 'Cup and Handle', 'volume': '+25%'},
    {'ticker': 'MSFT', 'signal': 'HOLD', 'strength': 'Neutral', 'pattern': 'Consolidation', 'volume': '+5%'},
    {'ticker': 'GOOGL', 'signal': 'BUY', 'strength': 'Moderate', 'pattern': 'Breakout', 'volume': '+15%'},
    {'ticker': 'AMZN', 'signal': 'SELL', 'strength': 'Weak', 'pattern': 'Double Top', 'volume': '-10%'},
    {'ticker': 'NVDA', 'signal': 'BUY', 'strength': 'Very Strong', 'pattern': 'Bullish Flag', 'volume': '+40%'},
    {'ticker': 'TSLA', 'signal': 'HOLD', 'strength': 'Neutral', 'pattern': 'Triangle', 'volume': '+2%'},
    {'ticker': 'META', 'signal': 'BUY', 'strength': 'Strong', 'pattern': 'Inverse Head & Shoulders', 'volume': '+20%'},
    {'ticker': 'AMD', 'signal': 'SELL', 'strength': 'Moderate', 'pattern': 'Descending Triangle', 'volume': '-5%'},
]

# Sample trade journal entries
trade_journal = [
    {'date': '2025-04-01', 'ticker': 'AAPL', 'action': 'BUY', 'price': 170.25, 'shares': 100, 'reason': 'Bullish breakout pattern with increasing volume', 'result': 'Open'},
    {'date': '2025-03-25', 'ticker': 'MSFT', 'action': 'BUY', 'price': 310.75, 'shares': 50, 'reason': 'Pullback to support with RSI oversold', 'result': 'Open'},
    {'date': '2025-03-20', 'ticker': 'GOOGL', 'action': 'BUY', 'price': 135.50, 'shares': 25, 'reason': 'Earnings beat with positive guidance', 'result': 'Open'},
    {'date': '2025-03-15', 'ticker': 'AMZN', 'action': 'BUY', 'price': 178.50, 'shares': 30, 'reason': 'Breakout from consolidation', 'result': 'Closed (+5.2%)'},
    {'date': '2025-03-10', 'ticker': 'NVDA', 'action': 'BUY', 'price': 850.25, 'shares': 10, 'reason': 'AI announcement catalyst', 'result': 'Closed (+12.8%)'},
    {'date': '2025-03-05', 'ticker': 'TSLA', 'action': 'SELL', 'price': 195.75, 'shares': 20, 'reason': 'Bearish divergence on RSI', 'result': 'Closed (+3.5%)'},
]

# Memory module settings
memory_settings = {
    'trade_history_depth': 50,
    'market_pattern_recognition': 75,
    'sentiment_analysis_weight': 60,
    'technical_indicator_memory': 80,
    'adaptive_learning_rate': 65
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Calculate portfolio value
    portfolio_value = portfolio['cash']
    for position in portfolio['positions']:
        portfolio_value += position['shares'] * position['current_price']
    
    # Calculate portfolio performance
    portfolio_cost = portfolio['cash']
    for position in portfolio['positions']:
        portfolio_cost += position['shares'] * position['entry_price']
    
    portfolio_performance = ((portfolio_value - portfolio_cost) / portfolio_cost) * 100
    
    # Calculate position values and profits
    for position in portfolio['positions']:
        position['value'] = position['shares'] * position['current_price']
        position['profit'] = position['shares'] * (position['current_price'] - position['entry_price'])
        position['profit_percent'] = ((position['current_price'] - position['entry_price']) / position['entry_price']) * 100
    
    return render_template('dashboard.html', 
                          portfolio=portfolio,
                          portfolio_value=portfolio_value,
                          portfolio_performance=portfolio_performance,
                          scanner_results=scanner_results,
                          trade_journal=trade_journal)

@app.route('/strategy')
def strategy():
    return render_template('strategy.html')

@app.route('/scanner')
def scanner():
    return render_template('scanner.html', scanner_results=scanner_results)

@app.route('/journal')
def journal():
    return render_template('journal.html', trade_journal=trade_journal)

@app.route('/backtesting')
def backtesting():
    return render_template('backtesting.html')

@app.route('/settings')
def settings():
    return render_template('settings.html', memory_settings=memory_settings)

@app.route('/update_memory_settings', methods=['POST'])
def update_memory_settings():
    try:
        memory_settings['trade_history_depth'] = int(request.form.get('trade_history_depth', 50))
        memory_settings['market_pattern_recognition'] = int(request.form.get('market_pattern_recognition', 75))
        memory_settings['sentiment_analysis_weight'] = int(request.form.get('sentiment_analysis_weight', 60))
        memory_settings['technical_indicator_memory'] = int(request.form.get('technical_indicator_memory', 80))
        memory_settings['adaptive_learning_rate'] = int(request.form.get('adaptive_learning_rate', 65))
        
        return jsonify({'success': True, 'message': 'Memory settings updated successfully'})
    except Exception as e:
        logger.error(f"Error updating memory settings: {e}")
        return jsonify({'success': False, 'message': f'Error updating memory settings: {str(e)}'})

@app.route('/generate_strategy', methods=['POST'])
def generate_strategy():
    try:
        ticker = request.form.get('ticker', 'AAPL')
        strategy_type = request.form.get('strategy_type', 'swing')
        
        logger.info(f"Generating strategy for {ticker} with type {strategy_type}")
        
        # Get market data
        data = get_market_data(ticker, period="1y", interval="1d")
        
        if data is None:
            return jsonify({'success': False, 'error': 'Failed to get market data'})
        
        # Calculate indicators
        df = calculate_indicators(data)
        
        # Generate signals
        df, prediction = generate_signals(df)
        
        # Generate trades
        trades = generate_trades(df)
        
        # Calculate performance
        performance = calculate_performance(df, trades)
        
        # Generate plots
        plot_urls = generate_plots(df, ticker)
        
        # Generate strategy description
        description = generate_strategy_description(ticker, df, prediction)
        
        # Return results
        result = {
            'success': True,
            'ticker': ticker,
            'prediction': prediction,
            'performance': performance,
            'trades': trades,
            'plot_urls': plot_urls,
            'description': description
        }
        
        logger.info(f"Strategy generation successful for {ticker}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error generating strategy: {e}")
        return jsonify({'success': False, 'error': f'Error generating strategy: {str(e)}'})

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    try:
        ticker = request.form.get('ticker', 'AAPL')
        start_date = request.form.get('start_date', '2024-01-01')
        end_date = request.form.get('end_date', '2025-04-01')
        strategy_type = request.form.get('strategy_type', 'swing')
        initial_capital = float(request.form.get('initial_capital', 10000))
        
        logger.info(f"Running backtest for {ticker} from {start_date} to {end_date}")
        
        # Get market data
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            return jsonify({'success': False, 'error': 'Failed to get market data'})
        
        # Calculate indicators
        df = calculate_indicators(data)
        
        # Generate signals
        df, _ = generate_signals(df)
        
        # Run backtest
        backtest_results = run_backtest_simulation(df, initial_capital)
        
        # Generate plots
        plot_urls = generate_backtest_plots(df, backtest_results, ticker)
        
        # Return results
        result = {
            'success': True,
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': backtest_results['final_capital'],
            'total_return': backtest_results['total_return'],
            'annual_return': backtest_results['annual_return'],
            'sharpe_ratio': backtest_results['sharpe_ratio'],
            'max_drawdown': backtest_results['max_drawdown'],
            'win_rate': backtest_results['win_rate'],
            'trades': backtest_results['trades'],
            'plot_urls': plot_urls
        }
        
        logger.info(f"Backtest successful for {ticker}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({'success': False, 'error': f'Error running backtest: {str(e)}'})

@app.route('/scan_market', methods=['POST'])
def scan_market():
    try:
        scan_type = request.form.get('scan_type', 'technical')
        market = request.form.get('market', 'US')
        min_price = float(request.form.get('min_price', 10))
        max_price = float(request.form.get('max_price', 1000))
        min_volume = float(request.form.get('min_volume', 1000000))
        
        logger.info(f"Scanning market with type {scan_type}")
        
        # For demo purposes, return the sample scanner results
        # In a real implementation, this would scan the market based on the criteria
        
        result = {
            'success': True,
            'scan_type': scan_type,
            'market': market,
            'results': scanner_results
        }
        
        logger.info(f"Market scan successful")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error scanning market: {e}")
        return jsonify({'success': False, 'error': f'Error scanning market: {str(e)}'})

@app.route('/add_journal_entry', methods=['POST'])
def add_journal_entry():
    try:
        date = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        ticker = request.form.get('ticker', '')
        action = request.form.get('action', '')
        price = float(request.form.get('price', 0))
        shares = int(request.form.get('shares', 0))
        reason = request.form.get('reason', '')
        result = request.form.get('result', 'Open')
        
        if not ticker or not action:
            return jsonify({'success': False, 'error': 'Ticker and action are required'})
        
        # Add the entry to the trade journal
        # In a real implementation, this would be stored in a database
        new_entry = {
            'date': date,
            'ticker': ticker,
            'action': action,
            'price': price,
            'shares': shares,
            'reason': reason,
            'result': result
        }
        
        trade_journal.insert(0, new_entry)
        
        logger.info(f"Added journal entry for {ticker}")
        return jsonify({'success': True, 'entry': new_entry})
    
    except Exception as e:
        logger.error(f"Error adding journal entry: {e}")
        return jsonify({'success': False, 'error': f'Error adding journal entry: {str(e)}'})

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
    sma20_last = float(last_row['SMA20'])
    sma50_last = float(last_row['SMA50'])
    rsi_last = float(last_row['RSI'])
    atr_last = float(last_row['ATR'])
    atr_mean = float(df['ATR'].mean())
    bb_lower_last = float(last_row['BB_lower'])
    bb_upper_last = float(last_row['BB_upper'])
    signal_last = int(last_row['signal'])
    macd_last = float(last_row['MACD'])
    macd_signal_last = float(last_row['MACD_signal'])
    
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

def generate_plots(df, ticker):
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
    
    plt.title(f'{ticker} Price with Moving Averages and Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    price_plot_path = os.path.join('static', f'{ticker}_price_plot.png')
    plt.savefig(os.path.join(os.path.dirname(__file__), price_plot_path))
    plt.close()
    plot_urls.append(price_plot_path)
    
    # RSI plot
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['RSI'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title(f'{ticker} RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    rsi_plot_path = os.path.join('static', f'{ticker}_rsi_plot.png')
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
    
    plt.title(f'{ticker} MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    macd_plot_path = os.path.join('static', f'{ticker}_macd_plot.png')
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
    
    plt.title(f'{ticker} Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    volume_plot_path = os.path.join('static', f'{ticker}_volume_plot.png')
    plt.savefig(os.path.join(os.path.dirname(__file__), volume_plot_path))
    plt.close()
    plot_urls.append(volume_plot_path)
    
    return plot_urls

def run_backtest_simulation(df, initial_capital):
    """Run a backtest simulation."""
    
    # Initialize variables
    cash = initial_capital
    shares = 0
    trades = []
    equity_curve = []
    
    # Run simulation
    for i in range(1, len(df)):
        # Record equity
        equity = cash + shares * df['Close'].iloc[i]
        equity_curve.append(equity)
        
        # Check for buy signal
        if df['signal'].iloc[i] == 1 and shares == 0:
            # Calculate number of shares to buy
            price = df['Close'].iloc[i]
            shares = int(cash / price)
            cash -= shares * price
            
            # Record trade
            trades.append({
                'date': df.index[i].strftime('%Y-%m-%d'),
                'type': 'BUY',
                'price': f"${price:.2f}",
                'shares': shares,
                'value': f"${shares * price:.2f}"
            })
        
        # Check for sell signal
        elif (df['signal'].iloc[i] == -1 or i == len(df) - 1) and shares > 0:
            # Sell all shares
            price = df['Close'].iloc[i]
            value = shares * price
            cash += value
            
            # Calculate profit/loss
            buy_price = float(trades[-1]['price'].replace('$', ''))
            pnl = (price - buy_price) / buy_price * 100
            
            # Record trade
            trades.append({
                'date': df.index[i].strftime('%Y-%m-%d'),
                'type': 'SELL',
                'price': f"${price:.2f}",
                'shares': shares,
                'value': f"${value:.2f}",
                'pnl': f"{pnl:.2f}%"
            })
            
            shares = 0
    
    # Calculate final equity
    final_equity = cash + shares * df['Close'].iloc[-1]
    
    # Calculate returns
    returns = pd.Series(equity_curve).pct_change().dropna()
    
    # Calculate performance metrics
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    annual_return = total_return / (len(df) / 252)  # Assuming 252 trading days per year
    sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)  # Annualized Sharpe ratio
    
    # Calculate maximum drawdown
    peak = pd.Series(equity_curve).expanding(min_periods=1).max()
    drawdown = (pd.Series(equity_curve) / peak - 1) * 100
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = 0
    if len(trades) > 0:
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        if sell_trades:
            wins = sum(1 for t in sell_trades if 'pnl' in t and float(t['pnl'].replace('%', '')) > 0)
            win_rate = wins / len(sell_trades)
    
    # Return results
    return {
        'initial_capital': initial_capital,
        'final_capital': final_equity,
        'total_return': f"{total_return:.2f}%",
        'annual_return': f"{annual_return:.2f}%",
        'sharpe_ratio': f"{sharpe_ratio:.2f}",
        'max_drawdown': f"{max_drawdown:.2f}%",
        'win_rate': f"{win_rate:.2f}",
        'trades': trades,
        'equity_curve': equity_curve
    }

def generate_backtest_plots(df, backtest_results, ticker):
    """Generate plots for backtest visualization."""
    
    plot_urls = []
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Equity curve plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[1:], backtest_results['equity_curve'], label='Equity Curve')
    
    # Add buy/sell markers
    for trade in backtest_results['trades']:
        date = datetime.strptime(trade['date'], '%Y-%m-%d')
        if date in df.index:
            idx = df.index.get_loc(date)
            if trade['type'] == 'BUY':
                plt.scatter(df.index[idx], backtest_results['equity_curve'][idx-1], marker='^', color='green', s=100)
            else:  # SELL
                plt.scatter(df.index[idx], backtest_results['equity_curve'][idx-1], marker='v', color='red', s=100)
    
    plt.title(f'{ticker} Backtest Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    equity_plot_path = os.path.join('static', f'{ticker}_backtest_equity.png')
    plt.savefig(os.path.join(os.path.dirname(__file__), equity_plot_path))
    plt.close()
    plot_urls.append(equity_plot_path)
    
    # Drawdown plot
    plt.figure(figsize=(12, 4))
    peak = pd.Series(backtest_results['equity_curve']).expanding(min_periods=1).max()
    drawdown = (pd.Series(backtest_results['equity_curve']) / peak - 1) * 100
    plt.fill_between(df.index[1:], drawdown, 0, color='red', alpha=0.3)
    plt.title(f'{ticker} Backtest Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    drawdown_plot_path = os.path.join('static', f'{ticker}_backtest_drawdown.png')
    plt.savefig(os.path.join(os.path.dirname(__file__), drawdown_plot_path))
    plt.close()
    plot_urls.append(drawdown_plot_path)
    
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

# Function to create HTML templates
def create_templates():
    """Create HTML templates for the frontend."""
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create static directories if they don't exist
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    css_dir = os.path.join(static_dir, 'css')
    js_dir = os.path.join(static_dir, 'js')
    img_dir = os.path.join(static_dir, 'img')
    
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(css_dir, exist_ok=True)
    os.makedirs(js_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # Create CSS file
    with open(os.path.join(css_dir, 'style.css'), 'w') as f:
        f.write("""
body {
    font-family: 'Roboto', Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.navbar {
    background-color: #2c3e50;
    color: white;
    padding: 10px 0;
    margin-bottom: 20px;
}

.navbar .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.navbar-brand {
    font-size: 24px;
    font-weight: bold;
    color: white;
    text-decoration: none;
}

.navbar-menu {
    display: flex;
    list-style: none;
    margin: 0;
    padding: 0;
}

.navbar-menu li {
    margin-left: 20px;
}

.navbar-menu a {
    color: white;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.3s;
}

.navbar-menu a:hover {
    color: #3498db;
}

.card {
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    overflow: hidden;
}

.card-header {
    background-color: #3498db;
    color: white;
    padding: 15px 20px;
    font-weight: bold;
}

.card-body {
    padding: 20px;
}

.form-group {
    margin-bottom: 15px;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
}

input, select, textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
}

button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #2980b9;
}

.btn-danger {
    background-color: #e74c3c;
}

.btn-danger:hover {
    background-color: #c0392b;
}

.btn-success {
    background-color: #2ecc71;
}

.btn-success:hover {
    background-color: #27ae60;
}

.btn-warning {
    background-color: #f39c12;
}

.btn-warning:hover {
    background-color: #d35400;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
}

th, td {
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

th {
    background-color: #f2f2f2;
    font-weight: bold;
}

tr:hover {
    background-color: #f5f5f5;
}

.dashboard-stats {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.stat-card {
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    padding: 20px;
    text-align: center;
}

.stat-card h3 {
    margin-top: 0;
    color: #7f8c8d;
    font-size: 16px;
}

.stat-card p {
    margin-bottom: 0;
    font-size: 24px;
    font-weight: bold;
    color: #2c3e50;
}

.chart-container {
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    padding: 20px;
    margin-bottom: 20px;
}

.plot-container {
    margin-top: 20px;
    text-align: center;
}

.plot-container img {
    max-width: 100%;
    height: auto;
    margin-bottom: 20px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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

.metrics {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 20px;
}

.metric {
    background-color: #f9f9f9;
    border-radius: 4px;
    padding: 15px;
    text-align: center;
}

.metric h3 {
    margin-top: 0;
    color: #7f8c8d;
    font-size: 14px;
}

.metric p {
    font-size: 20px;
    font-weight: bold;
    margin: 5px 0;
    color: #2c3e50;
}

.description {
    white-space: pre-line;
    line-height: 1.6;
}

.loading {
    text-align: center;
    padding: 20px;
    display: none;
}

.alert {
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 4px;
}

.alert-success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.alert-danger {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.alert-warning {
    background-color: #fff3cd;
    color: #856404;
    border: 1px solid #ffeeba;
}

.alert-info {
    background-color: #d1ecf1;
    color: #0c5460;
    border: 1px solid #bee5eb;
}

.tabs {
    display: flex;
    border-bottom: 1px solid #ddd;
    margin-bottom: 20px;
}

.tab {
    padding: 10px 20px;
    cursor: pointer;
    border: 1px solid transparent;
    border-bottom: none;
    margin-bottom: -1px;
    background-color: #f8f9fa;
}

.tab.active {
    background-color: white;
    border-color: #ddd;
    border-bottom-color: white;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.memory-slider {
    width: 100%;
    margin: 10px 0;
}

.memory-slider-container {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
}

.memory-slider-container label {
    flex: 1;
    margin-bottom: 0;
}

.memory-slider-container .slider-value {
    width: 50px;
    text-align: center;
    font-weight: bold;
}

.memory-slider-container input[type="range"] {
    flex: 2;
}

@media (max-width: 768px) {
    .navbar .container {
        flex-direction: column;
    }
    
    .navbar-menu {
        margin-top: 10px;
    }
    
    .navbar-menu li {
        margin-left: 10px;
        margin-right: 10px;
    }
    
    .dashboard-stats {
        grid-template-columns: 1fr;
    }
    
    .metrics {
        grid-template-columns: 1fr 1fr;
    }
}
        """)
    
    # Create JavaScript file
    with open(os.path.join(js_dir, 'main.js'), 'w') as f:
        f.write("""
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs if they exist
    const tabs = document.querySelectorAll('.tab');
    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs and tab contents
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                this.classList.add('active');
                const tabId = this.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            });
        });
    }
    
    // Initialize memory sliders if they exist
    const memorySliders = document.querySelectorAll('.memory-slider');
    if (memorySliders.length > 0) {
        memorySliders.forEach(slider => {
            const valueDisplay = slider.nextElementSibling;
            valueDisplay.textContent = slider.value;
            
            slider.addEventListener('input', function() {
                valueDisplay.textContent = this.value;
            });
        });
    }
    
    // Initialize strategy generation form if it exists
    const strategyForm = document.getElementById('strategy-form');
    if (strategyForm) {
        strategyForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('strategy-result').style.display = 'none';
            
            const formData = new FormData(this);
            
            fetch('/generate_strategy', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.success === false) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                document.getElementById('strategy-result').style.display = 'block';
                
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
                    row.insertCell(4).textContent = trade.pnl || '';
                });
                
                // Update description
                document.getElementById('description').textContent = data.description;
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Error generating strategy: ' + error);
            });
        });
    }
    
    // Initialize backtest form if it exists
    const backtestForm = document.getElementById('backtest-form');
    if (backtestForm) {
        backtestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            document.getElementById('backtest-loading').style.display = 'block';
            document.getElementById('backtest-result').style.display = 'none';
            
            const formData = new FormData(this);
            
            fetch('/run_backtest', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('backtest-loading').style.display = 'none';
                
                if (data.success === false) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                document.getElementById('backtest-result').style.display = 'block';
                
                // Update metrics
                document.getElementById('initial-capital').textContent = '$' + data.initial_capital.toFixed(2);
                document.getElementById('final-capital').textContent = '$' + data.final_capital.toFixed(2);
                document.getElementById('backtest-return').textContent = data.total_return;
                document.getElementById('annual-return').textContent = data.annual_return;
                document.getElementById('backtest-sharpe').textContent = data.sharpe_ratio;
                document.getElementById('backtest-drawdown').textContent = data.max_drawdown;
                document.getElementById('backtest-winrate').textContent = data.win_rate;
                
                // Update plots
                const plotsContainer = document.getElementById('backtest-plots');
                plotsContainer.innerHTML = '';
                data.plot_urls.forEach(url => {
                    const img = document.createElement('img');
                    img.src = url;
                    img.alt = 'Backtest Plot';
                    plotsContainer.appendChild(img);
                });
                
                // Update trades table
                const tradesBody = document.getElementById('backtest-trades').getElementsByTagName('tbody')[0];
                tradesBody.innerHTML = '';
                data.trades.forEach(trade => {
                    const row = tradesBody.insertRow();
                    row.insertCell(0).textContent = trade.date;
                    row.insertCell(1).textContent = trade.type;
                    row.insertCell(2).textContent = trade.price;
                    row.insertCell(3).textContent = trade.shares;
                    row.insertCell(4).textContent = trade.value;
                    if (trade.pnl) {
                        row.insertCell(5).textContent = trade.pnl;
                    } else {
                        row.insertCell(5).textContent = '';
                    }
                });
            })
            .catch(error => {
                document.getElementById('backtest-loading').style.display = 'none';
                alert('Error running backtest: ' + error);
            });
        });
    }
    
    // Initialize scanner form if it exists
    const scannerForm = document.getElementById('scanner-form');
    if (scannerForm) {
        scannerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            document.getElementById('scanner-loading').style.display = 'block';
            document.getElementById('scanner-result').style.display = 'none';
            
            const formData = new FormData(this);
            
            fetch('/scan_market', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('scanner-loading').style.display = 'none';
                
                if (data.success === false) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                document.getElementById('scanner-result').style.display = 'block';
                
                // Update results table
                const resultsBody = document.getElementById('scanner-results').getElementsByTagName('tbody')[0];
                resultsBody.innerHTML = '';
                data.results.forEach(result => {
                    const row = resultsBody.insertRow();
                    row.insertCell(0).textContent = result.ticker;
                    
                    const signalCell = row.insertCell(1);
                    signalCell.textContent = result.signal;
                    if (result.signal === 'BUY') {
                        signalCell.style.color = '#155724';
                        signalCell.style.backgroundColor = '#d4edda';
                    } else if (result.signal === 'SELL') {
                        signalCell.style.color = '#721c24';
                        signalCell.style.backgroundColor = '#f8d7da';
                    }
                    
                    row.insertCell(2).textContent = result.strength;
                    row.insertCell(3).textContent = result.pattern;
                    row.insertCell(4).textContent = result.volume;
                });
            })
            .catch(error => {
                document.getElementById('scanner-loading').style.display = 'none';
                alert('Error scanning market: ' + error);
            });
        });
    }
    
    // Initialize journal form if it exists
    const journalForm = document.getElementById('journal-form');
    if (journalForm) {
        journalForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/add_journal_entry', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success === false) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Add new entry to table
                const journalBody = document.getElementById('journal-entries').getElementsByTagName('tbody')[0];
                const row = journalBody.insertRow(0);
                row.insertCell(0).textContent = data.entry.date;
                row.insertCell(1).textContent = data.entry.ticker;
                row.insertCell(2).textContent = data.entry.action;
                row.insertCell(3).textContent = '$' + data.entry.price.toFixed(2);
                row.insertCell(4).textContent = data.entry.shares;
                row.insertCell(5).textContent = data.entry.reason;
                row.insertCell(6).textContent = data.entry.result;
                
                // Reset form
                journalForm.reset();
                
                // Show success message
                alert('Journal entry added successfully');
            })
            .catch(error => {
                alert('Error adding journal entry: ' + error);
            });
        });
    }
    
    // Initialize memory settings form if it exists
    const memoryForm = document.getElementById('memory-form');
    if (memoryForm) {
        memoryForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/update_memory_settings', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success === false) {
                    alert('Error: ' + data.message);
                    return;
                }
                
                // Show success message
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-success';
                alertDiv.textContent = data.message;
                
                const formContainer = document.querySelector('.card-body');
                formContainer.insertBefore(alertDiv, formContainer.firstChild);
                
                // Remove alert after 3 seconds
                setTimeout(() => {
                    alertDiv.remove();
                }, 3000);
            })
            .catch(error => {
                alert('Error updating memory settings: ' + error);
            });
        });
    }
});
        """)
    
    # Create index.html template
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemma Advanced Trading System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <a href="/" class="navbar-brand">Gemma Advanced Trading</a>
            <ul class="navbar-menu">
                <li><a href="/dashboard">Dashboard</a></li>
                <li><a href="/strategy">Strategy</a></li>
                <li><a href="/scanner">Scanner</a></li>
                <li><a href="/backtesting">Backtesting</a></li>
                <li><a href="/journal">Journal</a></li>
                <li><a href="/settings">Settings</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <div class="card">
            <div class="card-header">Welcome to Gemma Advanced Trading System</div>
            <div class="card-body">
                <h2>Advanced Trading with Gemma 3 Integration</h2>
                <p>The Gemma Advanced Trading System is a sophisticated trading platform that leverages Gemma 3's powerful capabilities to provide intelligent trading strategies, market analysis, and portfolio management.</p>
                
                <h3>Key Features:</h3>
                <ul>
                    <li><strong>Strategy Generation:</strong> Create custom trading strategies based on technical analysis, market conditions, and Gemma 3's advanced reasoning.</li>
                    <li><strong>Market Scanner:</strong> Scan the market for trading opportunities based on technical patterns, momentum, and other criteria.</li>
                    <li><strong>Backtesting:</strong> Test your strategies against historical data to evaluate performance before risking real capital.</li>
                    <li><strong>Portfolio Management:</strong> Track your positions, monitor performance, and get insights to optimize your portfolio.</li>
                    <li><strong>Trade Journal:</strong> Record your trades, reasons, and outcomes to improve your trading discipline and learn from experience.</li>
                    <li><strong>Memory Module Control:</strong> Fine-tune how the system learns from past trades and market patterns.</li>
                </ul>
                
                <div style="display: flex; justify-content: space-around; margin-top: 30px;">
                    <a href="/dashboard"><button>Go to Dashboard</button></a>
                    <a href="/strategy"><button>Create Strategy</button></a>
                    <a href="/scanner"><button>Scan Market</button></a>
                </div>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
        """)
    
    # Create dashboard.html template
    with open(os.path.join(templates_dir, 'dashboard.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Gemma Advanced Trading</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <a href="/" class="navbar-brand">Gemma Advanced Trading</a>
            <ul class="navbar-menu">
                <li><a href="/dashboard">Dashboard</a></li>
                <li><a href="/strategy">Strategy</a></li>
                <li><a href="/scanner">Scanner</a></li>
                <li><a href="/backtesting">Backtesting</a></li>
                <li><a href="/journal">Journal</a></li>
                <li><a href="/settings">Settings</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <h1>Trading Dashboard</h1>
        
        <div class="dashboard-stats">
            <div class="stat-card">
                <h3>Portfolio Value</h3>
                <p>${{ "%.2f"|format(portfolio_value) }}</p>
            </div>
            <div class="stat-card">
                <h3>Performance</h3>
                <p>{{ "%.2f"|format(portfolio_performance) }}%</p>
            </div>
            <div class="stat-card">
                <h3>Cash Available</h3>
                <p>${{ "%.2f"|format(portfolio.cash) }}</p>
            </div>
            <div class="stat-card">
                <h3>Open Positions</h3>
                <p>{{ portfolio.positions|length }}</p>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">Portfolio Positions</div>
            <div class="card-body">
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Shares</th>
                            <th>Entry Price</th>
                            <th>Current Price</th>
                            <th>Value</th>
                            <th>Profit/Loss</th>
                            <th>P/L %</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for position in portfolio.positions %}
                        <tr>
                            <td>{{ position.ticker }}</td>
                            <td>{{ position.shares }}</td>
                            <td>${{ "%.2f"|format(position.entry_price) }}</td>
                            <td>${{ "%.2f"|format(position.current_price) }}</td>
                            <td>${{ "%.2f"|format(position.value) }}</td>
                            <td>${{ "%.2f"|format(position.profit) }}</td>
                            <td>{{ "%.2f"|format(position.profit_percent) }}%</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">Watchlist</div>
            <div class="card-body">
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for ticker in portfolio.watchlist %}
                        <tr>
                            <td>{{ ticker }}</td>
                            <td>
                                <a href="/strategy?ticker={{ ticker }}"><button>Create Strategy</button></a>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">Market Scanner Results</div>
            <div class="card-body">
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Signal</th>
                            <th>Strength</th>
                            <th>Pattern</th>
                            <th>Volume</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for result in scanner_results %}
                        <tr>
                            <td>{{ result.ticker }}</td>
                            <td style="{% if result.signal == 'BUY' %}color: #155724; background-color: #d4edda;{% elif result.signal == 'SELL' %}color: #721c24; background-color: #f8d7da;{% endif %}">
                                {{ result.signal }}
                            </td>
                            <td>{{ result.strength }}</td>
                            <td>{{ result.pattern }}</td>
                            <td>{{ result.volume }}</td>
                            <td>
                                <a href="/strategy?ticker={{ result.ticker }}"><button>Create Strategy</button></a>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">Recent Trades</div>
            <div class="card-body">
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Ticker</th>
                            <th>Action</th>
                            <th>Price</th>
                            <th>Shares</th>
                            <th>Reason</th>
                            <th>Result</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for entry in trade_journal[:5] %}
                        <tr>
                            <td>{{ entry.date }}</td>
                            <td>{{ entry.ticker }}</td>
                            <td>{{ entry.action }}</td>
                            <td>${{ "%.2f"|format(entry.price) }}</td>
                            <td>{{ entry.shares }}</td>
                            <td>{{ entry.reason }}</td>
                            <td>{{ entry.result }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                <a href="/journal"><button style="margin-top: 10px;">View All Trades</button></a>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
        """)
    
    # Create strategy.html template
    with open(os.path.join(templates_dir, 'strategy.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategy Generator - Gemma Advanced Trading</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <a href="/" class="navbar-brand">Gemma Advanced Trading</a>
            <ul class="navbar-menu">
                <li><a href="/dashboard">Dashboard</a></li>
                <li><a href="/strategy">Strategy</a></li>
                <li><a href="/scanner">Scanner</a></li>
                <li><a href="/backtesting">Backtesting</a></li>
                <li><a href="/journal">Journal</a></li>
                <li><a href="/settings">Settings</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <h1>Strategy Generator</h1>
        
        <div class="card">
            <div class="card-header">Generate Trading Strategy</div>
            <div class="card-body">
                <form id="strategy-form">
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
                    <button type="submit">Generate Strategy</button>
                </form>
                
                <div class="loading" id="loading">
                    <p>Generating strategy... Please wait.</p>
                    <img src="https://i.gifer.com/origin/b4/b4d657e7ef262b88eb5f7ac021edda87.gif" alt="Loading" width="50">
                </div>
                
                <div id="strategy-result" style="display: none;">
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
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
        """)
    
    # Create scanner.html template
    with open(os.path.join(templates_dir, 'scanner.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Scanner - Gemma Advanced Trading</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <a href="/" class="navbar-brand">Gemma Advanced Trading</a>
            <ul class="navbar-menu">
                <li><a href="/dashboard">Dashboard</a></li>
                <li><a href="/strategy">Strategy</a></li>
                <li><a href="/scanner">Scanner</a></li>
                <li><a href="/backtesting">Backtesting</a></li>
                <li><a href="/journal">Journal</a></li>
                <li><a href="/settings">Settings</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <h1>Market Scanner</h1>
        
        <div class="card">
            <div class="card-header">Scan Parameters</div>
            <div class="card-body">
                <form id="scanner-form">
                    <div class="form-group">
                        <label for="scan_type">Scan Type:</label>
                        <select id="scan_type" name="scan_type">
                            <option value="technical">Technical Patterns</option>
                            <option value="momentum">Momentum</option>
                            <option value="breakout">Breakout</option>
                            <option value="reversal">Reversal</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="market">Market:</label>
                        <select id="market" name="market">
                            <option value="US">US Stocks</option>
                            <option value="crypto">Cryptocurrencies</option>
                            <option value="forex">Forex</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="min_price">Minimum Price:</label>
                        <input type="number" id="min_price" name="min_price" value="10" min="0" step="0.01">
                    </div>
                    <div class="form-group">
                        <label for="max_price">Maximum Price:</label>
                        <input type="number" id="max_price" name="max_price" value="1000" min="0" step="0.01">
                    </div>
                    <div class="form-group">
                        <label for="min_volume">Minimum Volume:</label>
                        <input type="number" id="min_volume" name="min_volume" value="1000000" min="0">
                    </div>
                    <button type="submit">Scan Market</button>
                </form>
                
                <div class="loading" id="scanner-loading" style="display: none;">
                    <p>Scanning market... Please wait.</p>
                    <img src="https://i.gifer.com/origin/b4/b4d657e7ef262b88eb5f7ac021edda87.gif" alt="Loading" width="50">
                </div>
                
                <div id="scanner-result" style="display: block;">
                    <h2>Scan Results</h2>
                    <table id="scanner-results">
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Signal</th>
                                <th>Strength</th>
                                <th>Pattern</th>
                                <th>Volume</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for result in scanner_results %}
                            <tr>
                                <td>{{ result.ticker }}</td>
                                <td style="{% if result.signal == 'BUY' %}color: #155724; background-color: #d4edda;{% elif result.signal == 'SELL' %}color: #721c24; background-color: #f8d7da;{% endif %}">
                                    {{ result.signal }}
                                </td>
                                <td>{{ result.strength }}</td>
                                <td>{{ result.pattern }}</td>
                                <td>{{ result.volume }}</td>
                                <td>
                                    <a href="/strategy?ticker={{ result.ticker }}"><button>Create Strategy</button></a>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
        """)
    
    # Create backtesting.html template
    with open(os.path.join(templates_dir, 'backtesting.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtesting - Gemma Advanced Trading</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <a href="/" class="navbar-brand">Gemma Advanced Trading</a>
            <ul class="navbar-menu">
                <li><a href="/dashboard">Dashboard</a></li>
                <li><a href="/strategy">Strategy</a></li>
                <li><a href="/scanner">Scanner</a></li>
                <li><a href="/backtesting">Backtesting</a></li>
                <li><a href="/journal">Journal</a></li>
                <li><a href="/settings">Settings</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <h1>Strategy Backtesting</h1>
        
        <div class="card">
            <div class="card-header">Backtest Parameters</div>
            <div class="card-body">
                <form id="backtest-form">
                    <div class="form-group">
                        <label for="ticker">Ticker Symbol:</label>
                        <input type="text" id="ticker" name="ticker" value="AAPL" required>
                    </div>
                    <div class="form-group">
                        <label for="start_date">Start Date:</label>
                        <input type="date" id="start_date" name="start_date" value="2024-01-01" required>
                    </div>
                    <div class="form-group">
                        <label for="end_date">End Date:</label>
                        <input type="date" id="end_date" name="end_date" value="2025-04-01" required>
                    </div>
                    <div class="form-group">
                        <label for="strategy_type">Strategy Type:</label>
                        <select id="strategy_type" name="strategy_type">
                            <option value="swing">Swing Trading</option>
                            <option value="day">Day Trading</option>
                            <option value="position">Position Trading</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="initial_capital">Initial Capital:</label>
                        <input type="number" id="initial_capital" name="initial_capital" value="10000" min="1000" step="1000" required>
                    </div>
                    <button type="submit">Run Backtest</button>
                </form>
                
                <div class="loading" id="backtest-loading" style="display: none;">
                    <p>Running backtest... Please wait.</p>
                    <img src="https://i.gifer.com/origin/b4/b4d657e7ef262b88eb5f7ac021edda87.gif" alt="Loading" width="50">
                </div>
                
                <div id="backtest-result" style="display: none;">
                    <h2>Backtest Results</h2>
                    
                    <div class="metrics">
                        <div class="metric">
                            <h3>Initial Capital</h3>
                            <p id="initial-capital"></p>
                        </div>
                        <div class="metric">
                            <h3>Final Capital</h3>
                            <p id="final-capital"></p>
                        </div>
                        <div class="metric">
                            <h3>Total Return</h3>
                            <p id="backtest-return"></p>
                        </div>
                        <div class="metric">
                            <h3>Annual Return</h3>
                            <p id="annual-return"></p>
                        </div>
                        <div class="metric">
                            <h3>Sharpe Ratio</h3>
                            <p id="backtest-sharpe"></p>
                        </div>
                        <div class="metric">
                            <h3>Max Drawdown</h3>
                            <p id="backtest-drawdown"></p>
                        </div>
                        <div class="metric">
                            <h3>Win Rate</h3>
                            <p id="backtest-winrate"></p>
                        </div>
                    </div>
                    
                    <div class="plot-container" id="backtest-plots"></div>
                    
                    <h2>Trade History</h2>
                    <table id="backtest-trades">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Type</th>
                                <th>Price</th>
                                <th>Shares</th>
                                <th>Value</th>
                                <th>P&L</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
        """)
    
    # Create journal.html template
    with open(os.path.join(templates_dir, 'journal.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trade Journal - Gemma Advanced Trading</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <a href="/" class="navbar-brand">Gemma Advanced Trading</a>
            <ul class="navbar-menu">
                <li><a href="/dashboard">Dashboard</a></li>
                <li><a href="/strategy">Strategy</a></li>
                <li><a href="/scanner">Scanner</a></li>
                <li><a href="/backtesting">Backtesting</a></li>
                <li><a href="/journal">Journal</a></li>
                <li><a href="/settings">Settings</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <h1>Trade Journal</h1>
        
        <div class="card">
            <div class="card-header">Add Journal Entry</div>
            <div class="card-body">
                <form id="journal-form">
                    <div class="form-group">
                        <label for="date">Date:</label>
                        <input type="date" id="date" name="date" value="{{ now.strftime('%Y-%m-%d') }}" required>
                    </div>
                    <div class="form-group">
                        <label for="ticker">Ticker:</label>
                        <input type="text" id="ticker" name="ticker" required>
                    </div>
                    <div class="form-group">
                        <label for="action">Action:</label>
                        <select id="action" name="action" required>
                            <option value="BUY">BUY</option>
                            <option value="SELL">SELL</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="price">Price:</label>
                        <input type="number" id="price" name="price" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label for="shares">Shares:</label>
                        <input type="number" id="shares" name="shares" required>
                    </div>
                    <div class="form-group">
                        <label for="reason">Reason for Trade:</label>
                        <textarea id="reason" name="reason" rows="3" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="result">Result:</label>
                        <select id="result" name="result">
                            <option value="Open">Open</option>
                            <option value="Closed (+%)">Closed (Profit)</option>
                            <option value="Closed (-%)">Closed (Loss)</option>
                        </select>
                    </div>
                    <button type="submit">Add Entry</button>
                </form>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">Journal Entries</div>
            <div class="card-body">
                <table id="journal-entries">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Ticker</th>
                            <th>Action</th>
                            <th>Price</th>
                            <th>Shares</th>
                            <th>Reason</th>
                            <th>Result</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for entry in trade_journal %}
                        <tr>
                            <td>{{ entry.date }}</td>
                            <td>{{ entry.ticker }}</td>
                            <td>{{ entry.action }}</td>
                            <td>${{ "%.2f"|format(entry.price) }}</td>
                            <td>{{ entry.shares }}</td>
                            <td>{{ entry.reason }}</td>
                            <td>{{ entry.result }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
        """)
    
    # Create settings.html template
    with open(os.path.join(templates_dir, 'settings.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Settings - Gemma Advanced Trading</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <a href="/" class="navbar-brand">Gemma Advanced Trading</a>
            <ul class="navbar-menu">
                <li><a href="/dashboard">Dashboard</a></li>
                <li><a href="/strategy">Strategy</a></li>
                <li><a href="/scanner">Scanner</a></li>
                <li><a href="/backtesting">Backtesting</a></li>
                <li><a href="/journal">Journal</a></li>
                <li><a href="/settings">Settings</a></li>
            </ul>
        </div>
    </nav>
    
    <div class="container">
        <h1>System Settings</h1>
        
        <div class="tabs">
            <div class="tab active" data-tab="memory-settings">Memory Module</div>
            <div class="tab" data-tab="general-settings">General Settings</div>
            <div class="tab" data-tab="api-settings">API Settings</div>
        </div>
        
        <div class="card">
            <div class="card-header">Settings</div>
            <div class="card-body">
                <div id="memory-settings" class="tab-content active">
                    <h2>Memory Module Control</h2>
                    <p>Adjust how the system learns from past trades and market patterns. Higher values increase the weight given to historical data.</p>
                    
                    <form id="memory-form">
                        <div class="memory-slider-container">
                            <label for="trade_history_depth">Trade History Depth:</label>
                            <input type="range" id="trade_history_depth" name="trade_history_depth" min="0" max="100" value="{{ memory_settings.trade_history_depth }}" class="memory-slider">
                            <span class="slider-value">{{ memory_settings.trade_history_depth }}</span>
                        </div>
                        
                        <div class="memory-slider-container">
                            <label for="market_pattern_recognition">Market Pattern Recognition:</label>
                            <input type="range" id="market_pattern_recognition" name="market_pattern_recognition" min="0" max="100" value="{{ memory_settings.market_pattern_recognition }}" class="memory-slider">
                            <span class="slider-value">{{ memory_settings.market_pattern_recognition }}</span>
                        </div>
                        
                        <div class="memory-slider-container">
                            <label for="sentiment_analysis_weight">Sentiment Analysis Weight:</label>
                            <input type="range" id="sentiment_analysis_weight" name="sentiment_analysis_weight" min="0" max="100" value="{{ memory_settings.sentiment_analysis_weight }}" class="memory-slider">
                            <span class="slider-value">{{ memory_settings.sentiment_analysis_weight }}</span>
                        </div>
                        
                        <div class="memory-slider-container">
                            <label for="technical_indicator_memory">Technical Indicator Memory:</label>
                            <input type="range" id="technical_indicator_memory" name="technical_indicator_memory" min="0" max="100" value="{{ memory_settings.technical_indicator_memory }}" class="memory-slider">
                            <span class="slider-value">{{ memory_settings.technical_indicator_memory }}</span>
                        </div>
                        
                        <div class="memory-slider-container">
                            <label for="adaptive_learning_rate">Adaptive Learning Rate:</label>
                            <input type="range" id="adaptive_learning_rate" name="adaptive_learning_rate" min="0" max="100" value="{{ memory_settings.adaptive_learning_rate }}" class="memory-slider">
                            <span class="slider-value">{{ memory_settings.adaptive_learning_rate }}</span>
                        </div>
                        
                        <button type="submit">Save Memory Settings</button>
                    </form>
                </div>
                
                <div id="general-settings" class="tab-content">
                    <h2>General Settings</h2>
                    <p>Configure general system settings.</p>
                    
                    <form>
                        <div class="form-group">
                            <label for="default_ticker">Default Ticker:</label>
                            <input type="text" id="default_ticker" name="default_ticker" value="AAPL">
                        </div>
                        
                        <div class="form-group">
                            <label for="default_strategy">Default Strategy Type:</label>
                            <select id="default_strategy" name="default_strategy">
                                <option value="swing">Swing Trading</option>
                                <option value="day">Day Trading</option>
                                <option value="position">Position Trading</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="risk_per_trade">Default Risk Per Trade (%):</label>
                            <input type="number" id="risk_per_trade" name="risk_per_trade" value="1" min="0.1" max="10" step="0.1">
                        </div>
                        
                        <div class="form-group">
                            <label for="position_size">Default Position Size (%):</label>
                            <input type="number" id="position_size" name="position_size" value="5" min="1" max="100" step="1">
                        </div>
                        
                        <button type="submit">Save General Settings</button>
                    </form>
                </div>
                
                <div id="api-settings" class="tab-content">
                    <h2>API Settings</h2>
                    <p>Configure API connections for data sources.</p>
                    
                    <form>
                        <div class="form-group">
                            <label for="api_key">API Key:</label>
                            <input type="text" id="api_key" name="api_key" placeholder="Enter API key">
                        </div>
                        
                        <div class="form-group">
                            <label for="api_secret">API Secret:</label>
                            <input type="password" id="api_secret" name="api_secret" placeholder="Enter API secret">
                        </div>
                        
                        <div class="form-group">
                            <label for="data_provider">Data Provider:</label>
                            <select id="data_provider" name="data_provider">
                                <option value="yahoo">Yahoo Finance</option>
                                <option value="alpha_vantage">Alpha Vantage</option>
                                <option value="iex">IEX Cloud</option>
                                <option value="polygon">Polygon.io</option>
                            </select>
                        </div>
                        
                        <button type="submit">Save API Settings</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
        """)

if __name__ == '__main__':
    # Create HTML templates
    create_templates()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
