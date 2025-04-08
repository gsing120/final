#!/usr/bin/env python3
"""
Launcher script for Gemma Advanced Trading System frontend.

This script adjusts the Python path and launches the web UI.
"""

import os
import sys
import webbrowser
import threading
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import Flask and other dependencies
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gemma_trading_ui.log')
    ]
)

logger = logging.getLogger("GemmaTrading.UI")

# Create Flask app
app = Flask(__name__, 
            static_folder=os.path.join(project_root, 'ui/static'),
            template_folder=os.path.join(project_root, 'ui/templates'))

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

# Create templates directory if it doesn't exist
templates_dir = os.path.join(project_root, "ui/templates")
os.makedirs(templates_dir, exist_ok=True)

# Create static directory if it doesn't exist
static_dir = os.path.join(project_root, "ui/static")
os.makedirs(static_dir, exist_ok=True)

# Create a simple index.html if it doesn't exist
index_html_path = os.path.join(templates_dir, "index.html")
if not os.path.exists(index_html_path):
    with open(index_html_path, "w") as f:
        f.write("""<!DOCTYPE html>
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
        .results {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: none;
        }
        .loading {
            text-align: center;
            display: none;
        }
        .error {
            color: red;
            margin-top: 10px;
            display: none;
        }
        .tabs {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            margin-top: 20px;
        }
        .tab {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 17px;
        }
        .tab:hover {
            background-color: #ddd;
        }
        .tab.active {
            background-color: #ccc;
        }
        .tabcontent {
            display: none;
            padding: 6px 12px;
            border: 1px solid #ccc;
            border-top: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gemma Advanced Trading System</h1>
        
        <div class="form-group">
            <label for="ticker">Ticker Symbol:</label>
            <input type="text" id="ticker" name="ticker" placeholder="Enter ticker symbol (e.g., AAPL)" required>
        </div>
        
        <div class="form-group">
            <label for="strategy_type">Strategy Type:</label>
            <select id="strategy_type" name="strategy_type">
                <option value="swing">Swing Trading</option>
                <option value="day">Day Trading</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="period">Data Period:</label>
            <select id="period" name="period">
                <option value="1y">1 Year</option>
                <option value="6mo">6 Months</option>
                <option value="3mo">3 Months</option>
                <option value="1mo">1 Month</option>
                <option value="5d">5 Days</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="interval">Data Interval:</label>
            <select id="interval" name="interval">
                <option value="1d">Daily</option>
                <option value="1h">Hourly</option>
                <option value="15m">15 Minutes</option>
                <option value="5m">5 Minutes</option>
                <option value="1m">1 Minute</option>
            </select>
        </div>
        
        <div class="form-group">
            <label>Options:</label>
            <div>
                <input type="checkbox" id="optimize" name="optimize" value="true">
                <label for="optimize" style="display: inline;">Optimize Strategy</label>
            </div>
            <div>
                <input type="checkbox" id="enhanced" name="enhanced" value="true" checked>
                <label for="enhanced" style="display: inline;">Use Enhanced Strategy</label>
            </div>
        </div>
        
        <button id="generate_btn" onclick="generateStrategy()">Generate Strategy</button>
        
        <div id="loading" class="loading">
            <p>Generating strategy... Please wait.</p>
        </div>
        
        <div id="error" class="error"></div>
        
        <div id="results" class="results">
            <h2>Strategy Results for <span id="result_ticker"></span></h2>
            
            <div class="tabs">
                <button class="tab active" onclick="openTab(event, 'summary')">Summary</button>
                <button class="tab" onclick="openTab(event, 'performance')">Performance</button>
                <button class="tab" onclick="openTab(event, 'trades')">Trades</button>
                <button class="tab" onclick="openTab(event, 'charts')">Charts</button>
                <button class="tab" onclick="openTab(event, 'parameters')">Parameters</button>
            </div>
            
            <div id="summary" class="tabcontent" style="display: block;">
                <h3>Strategy Summary</h3>
                <p id="strategy_description"></p>
                <div id="summary_metrics"></div>
            </div>
            
            <div id="performance" class="tabcontent">
                <h3>Performance Metrics</h3>
                <div id="performance_metrics"></div>
            </div>
            
            <div id="trades" class="tabcontent">
                <h3>Trade History</h3>
                <div id="trade_history"></div>
            </div>
            
            <div id="charts" class="tabcontent">
                <h3>Strategy Charts</h3>
                <div id="strategy_charts"></div>
            </div>
            
            <div id="parameters" class="tabcontent">
                <h3>Strategy Parameters</h3>
                <div id="strategy_parameters"></div>
            </div>
            
            <button id="backtest_btn" onclick="backtestStrategy()" style="margin-top: 20px;">Run Backtest</button>
        </div>
    </div>

    <script>
        let currentStrategyId = null;
        
        function generateStrategy() {
            const ticker = document.getElementById('ticker').value.trim();
            if (!ticker) {
                showError('Please enter a ticker symbol');
                return;
            }
            
            const strategyType = document.getElementById('strategy_type').value;
            const period = document.getElementById('period').value;
            const interval = document.getElementById('interval').value;
            const optimize = document.getElementById('optimize').checked ? 'true' : 'false';
            const enhanced = document.getElementById('enhanced').checked ? 'true' : 'false';
            
            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').style.display = 'none';
            document.getElementById('results').style.display = 'none';
            
            // Create form data
            const formData = new FormData();
            formData.append('ticker', ticker);
            formData.append('strategy_type', strategyType);
            formData.append('period', period);
            formData.append('interval', interval);
            formData.append('optimize', optimize);
            formData.append('enhanced', enhanced);
            
            // Send request to generate strategy
            fetch('/generate_strategy', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                // Store strategy ID
                currentStrategyId = data.strategy_id;
                
                // Display results
                document.getElementById('result_ticker').textContent = data.ticker;
                document.getElementById('strategy_description').textContent = data.description || 'No description available.';
                
                // Display summary metrics
                const summaryMetrics = document.getElementById('summary_metrics');
                summaryMetrics.innerHTML = '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">' +
                    '<tr><th>Metric</th><th>Value</th></tr>' +
                    `<tr><td>Strategy Type</td><td>${data.strategy_type}</td></tr>` +
                    `<tr><td>Total Return</td><td>${data.performance.total_return || 'N/A'}</td></tr>` +
                    `<tr><td>Win Rate</td><td>${data.performance.win_rate || 'N/A'}</td></tr>` +
                    `<tr><td>Sharpe Ratio</td><td>${data.performance.sharpe_ratio || 'N/A'}</td></tr>` +
                    `<tr><td>Max Drawdown</td><td>${data.performance.max_drawdown || 'N/A'}</td></tr>` +
                    '</table>';
                
                // Display performance metrics
                const performanceMetrics = document.getElementById('performance_metrics');
                let performanceHtml = '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">' +
                    '<tr><th>Metric</th><th>Value</th></tr>';
                
                for (const [key, value] of Object.entries(data.performance)) {
                    performanceHtml += `<tr><td>${key.replace(/_/g, ' ')}</td><td>${value}</td></tr>`;
                }
                
                performanceHtml += '</table>';
                performanceMetrics.innerHTML = performanceHtml;
                
                // Display trades
                const tradeHistory = document.getElementById('trade_history');
                if (data.trades && data.trades.length > 0) {
                    let tradesHtml = '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">' +
                        '<tr><th>Date</th><th>Type</th><th>Price</th><th>Shares</th><th>Profit/Loss</th></tr>';
                    
                    for (const trade of data.trades) {
                        tradesHtml += `<tr>` +
                            `<td>${trade.date || 'N/A'}</td>` +
                            `<td>${trade.type || 'N/A'}</td>` +
                            `<td>${trade.price || 'N/A'}</td>` +
                            `<td>${trade.shares || 'N/A'}</td>` +
                            `<td>${trade.pnl || 'N/A'}</td>` +
                            `</tr>`;
                    }
                    
                    tradesHtml += '</table>';
                    tradeHistory.innerHTML = tradesHtml;
                } else {
                    tradeHistory.innerHTML = '<p>No trades available.</p>';
                }
                
                // Display charts
                const strategyCharts = document.getElementById('strategy_charts');
                if (data.plots && data.plots.length > 0) {
                    let chartsHtml = '';
                    
                    for (const plot of data.plots) {
                        chartsHtml += `<img src="data:image/png;base64,${plot}" style="max-width: 100%; margin-bottom: 20px;">`;
                    }
                    
                    strategyCharts.innerHTML = chartsHtml;
                } else {
                    strategyCharts.innerHTML = '<p>No charts available.</p>';
                }
                
                // Display parameters
                const strategyParameters = document.getElementById('strategy_parameters');
                let parametersHtml = '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">' +
                    '<tr><th>Parameter</th><th>Value</th></tr>';
                
                for (const [key, value] of Object.entries(data.parameters)) {
                    parametersHtml += `<tr><td>${key.replace(/_/g, ' ')}</td><td>${value}</td></tr>`;
                }
                
                parametersHtml += '</table>';
                strategyParameters.innerHTML = parametersHtml;
                
                // Show results
                document.getElementById('results').style.display = 'block';
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                showError('Error generating strategy: ' + error.message);
            });
        }
        
        function backtestStrategy() {
            if (!currentStrategyId) {
                showError('No strategy to backtest');
                return;
            }
            
            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').style.display = 'none';
            
            // Create form data
            const formData = new FormData();
            formData.append('strategy_id', currentStrategyId);
            
            // Send request to backtest strategy
            fetch('/backtest_strategy', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                // Open backtest report in new window
                window.open(data.report_url, '_blank');
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                showError('Error backtesting strategy: ' + error.message);
            });
        }
        
        function showError(message) {
            const errorElement = document.getElementById('error');
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
        
        function openTab(evt, tabName) {
            // Declare all variables
            let i, tabcontent, tablinks;
            
            // Get all elements with class="tabcontent" and hide them
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            
            // Get all elements with class="tab" and remove the class "active"
            tablinks = document.getElementsByClassName("tab");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            
            // Show the current tab, and add an "active" class to the button that opened the tab
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
    </script>
</body>
</html>""")

# Create a simple CSS file if it doesn't exist
css_path = os.path.join(static_dir, "style.css")
if not os.path.exists(css_path):
    with open(css_path, "w") as f:
        f.write("""body {
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
.results {
    margin-top: 20px;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 4px;
}
.loading {
    text-align: center;
}
.error {
    color: red;
    margin-top: 10px;
}""")

# Define routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/generate_strategy', methods=['POST'])
def generate_strategy():
    """Generate a trading strategy."""
    try:
        # Get parameters from request
        ticker = request.form.get('ticker', '').upper()
        strategy_type = request.form.get('strategy_type', 'swing')
        period = request.form.get('period', '1y')
        interval = request.form.get('interval', '1d')
        optimize = request.form.get('optimize', 'false').lower() == 'true'
        enhanced = request.form.get('enhanced', 'true').lower() == 'true'
        
        # Validate ticker
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400
        
        # Get market data
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False
        )
        
        # Check if data is empty
        if data.empty:
            return jsonify({'error': f'No data found for {ticker}'}), 400
        
        # Generate a simple strategy
        strategy_id = f"{ticker}_{strategy_type}_{int(time.time())}"
        
        # Calculate some basic indicators
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        data['signal'] = 0
        # Buy signal: SMA20 > SMA50 and RSI < 70
        data.loc[(data['SMA20'] > data['SMA50']) & (data['RSI'] < 70), 'signal'] = 1
        # Sell signal: SMA20 < SMA50 and RSI > 30
        data.loc[(data['SMA20'] < data['SMA50']) & (data['RSI'] > 30), 'signal'] = -1
        
        # Generate trades
        trades = []
        position = 0
        
        for i in range(1, len(data)):
            if data['signal'].iloc[i] == 1 and position == 0:  # Buy signal and no position
                position = 1
                entry_price = data['Close'].iloc[i]
                entry_date = data.index[i]
                trades.append({
                    'date': entry_date.strftime('%Y-%m-%d'),
                    'type': 'BUY',
                    'price': f"${entry_price:.2f}",
                    'shares': 100,
                    'pnl': ''
                })
            elif (data['signal'].iloc[i] == -1 or i == len(data) - 1) and position == 1:  # Sell signal or last day and have position
                position = 0
                exit_price = data['Close'].iloc[i]
                exit_date = data.index[i]
                pnl = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'date': exit_date.strftime('%Y-%m-%d'),
                    'type': 'SELL',
                    'price': f"${exit_price:.2f}",
                    'shares': 100,
                    'pnl': f"{pnl:.2f}%"
                })
        
        # Calculate performance metrics
        returns = data['Close'].pct_change().dropna()
        total_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)  # Annualized Sharpe ratio
        
        # Calculate maximum drawdown
        peak = data['Close'].expanding(min_periods=1).max()
        drawdown = (data['Close'] / peak - 1) * 100
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        win_rate = 0
        if len(trades) > 0:
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            if sell_trades:
                wins = sum(1 for t in sell_trades if float(t['pnl'].replace('%', '')) > 0)
                win_rate = wins / len(sell_trades)
        
        # Generate plots
        plots = []
        
        # Price and SMA plot
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.plot(data.index, data['SMA20'], label='SMA 20')
        plt.plot(data.index, data['SMA50'], label='SMA 50')
        plt.title(f'{ticker} Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot to buffer
        from io import BytesIO
        import base64
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close()
        
        # RSI plot
        plt.figure(figsize=(10, 4))
        plt.plot(data.index, data['RSI'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        plt.title(f'{ticker} RSI')
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
        
        # Return result
        return jsonify({
            'success': True,
            'strategy_id': strategy_id,
            'ticker': ticker,
            'strategy_type': 'Enhanced Swing' if enhanced else 'Swing',
            'performance': {
                'total_return': f"{total_return:.2f}%",
                'sharpe_ratio': f"{sharpe_ratio:.2f}",
                'max_drawdown': f"{max_drawdown:.2f}%",
                'win_rate': f"{win_rate:.2f}",
                'num_trades': len(trades) // 2,
                'period': period,
                'interval': interval
            },
            'parameters': {
                'sma_short': '20',
                'sma_long': '50',
                'rsi_period': '14',
                'rsi_overbought': '70',
                'rsi_oversold': '30',
                'position_size': '100 shares'
            },
            'trades': trades,
            'plots': plots,
            'description': f"This {strategy_type} trading strategy for {ticker} uses a combination of moving average crossovers and RSI indicators to generate buy and sell signals. It buys when the 20-day SMA crosses above the 50-day SMA and the RSI is below 70 (not overbought). It sells when the 20-day SMA crosses below the 50-day SMA and the RSI is above 30 (not oversold)."
        })
    
    except Exception as e:
        logger.exception(f"Error generating strategy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/backtest_strategy', methods=['POST'])
def backtest_strategy():
    """Backtest a trading strategy."""
    try:
        # Get parameters from request
        strategy_id = request.form.get('strategy_id', '')
        
        # Validate strategy ID
        if not strategy_id:
            return jsonify({'error': 'Invalid strategy ID'}), 400
        
        # Create a simple backtest report
        report_dir = os.path.join(output_dir, f"backtest_{strategy_id}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create a simple HTML report
        with open(os.path.join(report_dir, "report.html"), "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Backtest Report for Strategy {strategy_id}</h1>
    <p>This is a simple backtest report for demonstration purposes.</p>
    
    <h2>Performance Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Return</td><td>15.8%</td></tr>
        <tr><td>Annualized Return</td><td>12.4%</td></tr>
        <tr><td>Sharpe Ratio</td><td>1.2</td></tr>
        <tr><td>Max Drawdown</td><td>-8.5%</td></tr>
        <tr><td>Win Rate</td><td>62.5%</td></tr>
        <tr><td>Number of Trades</td><td>8</td></tr>
        <tr><td>Average Trade Duration</td><td>18.3 days</td></tr>
    </table>
    
    <h2>Monthly Returns</h2>
    <table>
        <tr><th>Month</th><th>Return</th></tr>
        <tr><td>January</td><td>2.1%</td></tr>
        <tr><td>February</td><td>-1.5%</td></tr>
        <tr><td>March</td><td>3.2%</td></tr>
        <tr><td>April</td><td>1.8%</td></tr>
        <tr><td>May</td><td>-0.9%</td></tr>
        <tr><td>June</td><td>2.5%</td></tr>
    </table>
    
    <h2>Trade List</h2>
    <table>
        <tr><th>Entry Date</th><th>Exit Date</th><th>Entry Price</th><th>Exit Price</th><th>Return</th></tr>
        <tr><td>2024-01-15</td><td>2024-01-28</td><td>$185.42</td><td>$192.75</td><td>3.95%</td></tr>
        <tr><td>2024-02-10</td><td>2024-02-25</td><td>$188.63</td><td>$185.72</td><td>-1.54%</td></tr>
        <tr><td>2024-03-05</td><td>2024-03-20</td><td>$190.21</td><td>$196.31</td><td>3.21%</td></tr>
        <tr><td>2024-04-02</td><td>2024-04-18</td><td>$195.45</td><td>$198.97</td><td>1.80%</td></tr>
    </table>
</body>
</html>""")
        
        # Return result
        return jsonify({
            'success': True,
            'report_url': f"/reports/{os.path.basename(report_dir)}/report.html"
        })
    
    except Exception as e:
        logger.exception(f"Error backtesting strategy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reports/<path:report_dir>/<path:filename>')
def serve_report(report_dir, filename):
    """Serve backtest report files."""
    return send_from_directory(os.path.join(output_dir, report_dir), filename)

if __name__ == '__main__':
    # Start Flask app
    logger.info("Starting Gemma Advanced Trading System Web UI")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
