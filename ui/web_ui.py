"""
Simple Web User Interface for Gemma Advanced Trading System.

This module provides a web-based user interface for the trading system.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import threading
import webbrowser
from flask import Flask, render_template, request, jsonify, send_from_directory

# Import system components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app import GemmaAdvancedTradingSystem

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
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Create trading system
trading_system = GemmaAdvancedTradingSystem()

# Create output directory if it doesn't exist
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# Store generated strategies
generated_strategies = {}

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
        
        # Generate strategy
        strategy_result = trading_system.generate_strategy(
            ticker=ticker,
            strategy_type=strategy_type,
            period=period,
            interval=interval,
            optimize=optimize,
            enhanced=enhanced,
            output_dir=output_dir
        )
        
        # Store strategy
        strategy_id = f"{ticker}_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        generated_strategies[strategy_id] = strategy_result
        
        # Generate plots
        plots = generate_plots(strategy_result)
        
        # Format performance metrics
        performance = {}
        for key, value in strategy_result['performance'].items():
            if isinstance(value, (int, float)):
                if key in ['win_rate']:
                    performance[key] = f"{value * 100:.2f}%"
                elif key in ['total_return', 'max_drawdown']:
                    performance[key] = f"{value:.2f}%"
                else:
                    performance[key] = f"{value:.4f}"
            else:
                performance[key] = str(value)
        
        # Format parameters
        parameters = {}
        for key, value in strategy_result['parameters'].items():
            parameters[key] = str(value)
        
        # Format trades
        trades = []
        if 'trades' in strategy_result and strategy_result['trades']:
            for trade in strategy_result['trades']:
                if isinstance(trade.get('date'), datetime):
                    trade['date'] = trade['date'].strftime('%Y-%m-%d %H:%M:%S')
                trades.append(trade)
        
        # Return result
        return jsonify({
            'success': True,
            'strategy_id': strategy_id,
            'ticker': ticker,
            'strategy_type': strategy_result.get('strategy_type', strategy_type),
            'performance': performance,
            'parameters': parameters,
            'trades': trades,
            'plots': plots,
            'description': strategy_result.get('description', '')
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
        if not strategy_id or strategy_id not in generated_strategies:
            return jsonify({'error': 'Invalid strategy ID'}), 400
        
        # Get strategy
        strategy_result = generated_strategies[strategy_id]
        
        # Backtest strategy
        backtest_result = trading_system.backtest_strategy(
            strategy_result=strategy_result,
            output_dir=output_dir
        )
        
        # Return result
        return jsonify({
            'success': True,
            'report_url': f"/reports/{os.path.basename(backtest_result['report_dir'])}/report.html"
        })
    
    except Exception as e:
        logger.exception(f"Error backtesting strategy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/monte_carlo', methods=['POST'])
def monte_carlo():
    """Run Monte Carlo simulation."""
    try:
        # Get parameters from request
        strategy_id = request.form.get('strategy_id', '')
        num_simulations = int(request.form.get('num_simulations', '1000'))
        confidence_level = float(request.form.get('confidence_level', '0.95'))
        
        # Validate strategy ID
        if not strategy_id or strategy_id not in generated_strategies:
            return jsonify({'error': 'Invalid strategy ID'}), 400
        
        # Get strategy
        strategy_result = generated_strategies[strategy_id]
        
        # Run Monte Carlo simulation
        mc_results = trading_system.run_monte_carlo_simulation(
            strategy_result=strategy_result,
            num_simulations=num_simulations,
            confidence_level=confidence_level,
            output_dir=output_dir
        )
        
        # Generate plots
        plots = generate_monte_carlo_plots(mc_results, strategy_result['ticker'])
        
        # Format results
        results = {
            'num_simulations': mc_results['num_simulations'],
            'confidence_level': f"{mc_results['confidence_level'] * 100:.0f}%",
            'mean_return': f"{mc_results['mean_return']:.2f}%",
            'median_return': f"{mc_results['median_return']:.2f}%",
            'mean_drawdown': f"{mc_results['mean_drawdown']:.2f}%",
            'median_drawdown': f"{mc_results['median_drawdown']:.2f}%",
            'return_ci': f"({mc_results['return_ci'][0]:.2f}%, {mc_results['return_ci'][1]:.2f}%)",
            'drawdown_ci': f"({mc_results['drawdown_ci'][0]:.2f}%, {mc_results['drawdown_ci'][1]:.2f}%)",
            'worst_case_return': f"{mc_results['worst_case_return']:.2f}%",
            'best_case_return': f"{mc_results['best_case_return']:.2f}%",
            'worst_case_drawdown': f"{mc_results['worst_case_drawdown']:.2f}%"
        }
        
        # Return result
        return jsonify({
            'success': True,
            'results': results,
            'plots': plots
        })
    
    except Exception as e:
        logger.exception(f"Error running Monte Carlo simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reports/<path:path>')
def reports(path):
    """Serve report files."""
    return send_from_directory(os.path.join(output_dir), path)

def generate_plots(strategy_result):
    """
    Generate plots for a strategy.
    
    Parameters:
    -----------
    strategy_result : dict
        Strategy result
        
    Returns:
    --------
    dict
        Dictionary of plot URLs
    """
    plots = {}
    
    # Check if signals are available
    if 'signals' not in strategy_result or strategy_result['signals'] is None:
        return plots
    
    signals = strategy_result['signals']
    ticker = strategy_result['ticker']
    
    # Plot price and signals
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot price
    ax.plot(signals.index, signals['Close'], label='Close Price')
    
    # Plot buy signals
    buy_signals = signals[signals['Signal'] == 1]
    if not buy_signals.empty:
        ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_signals = signals[signals['Signal'] == -1]
    if not sell_signals.empty:
        ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
    
    # Add labels and title
    ax.set_title(f'Price and Signals for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plots['price_signals'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    plt.close(fig)
    
    # Plot indicators if available
    indicator_columns = [col for col in signals.columns if any(
        ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD', 'Stochastic', 'Bollinger', 'ATR']
    )]
    
    if indicator_columns:
        # Group indicators by type
        trend_indicators = [col for col in indicator_columns if any(ind in col for ind in ['SMA', 'EMA', 'MACD'])]
        momentum_indicators = [col for col in indicator_columns if any(ind in col for ind in ['RSI', 'Stochastic'])]
        volatility_indicators = [col for col in indicator_columns if any(ind in col for ind in ['ATR', 'Bollinger'])]
        
        # Create subplots
        num_plots = sum([bool(trend_indicators), bool(momentum_indicators), bool(volatility_indicators)])
        if num_plots > 0:
            fig, axes = plt.subplots(num_plots + 1, 1, figsize=(12, 4 * (num_plots + 1)), sharex=True)
            
            # Handle single subplot case
            if num_plots == 0:
                axes = [axes]
            
            # Plot price in first subplot
            axes[0].plot(signals.index, signals['Close'], label='Close Price')
            axes[0].set_title(f'Price for {ticker}')
            axes[0].set_ylabel('Price')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            plot_idx = 1
            
            # Plot trend indicators
            if trend_indicators:
                ax = axes[plot_idx] if num_plots > 0 else axes
                for col in trend_indicators:
                    ax.plot(signals.index, signals[col], label=col)
                ax.set_title('Trend Indicators')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plot_idx += 1
            
            # Plot momentum indicators
            if momentum_indicators:
                ax = axes[plot_idx] if num_plots > 0 else axes
                for col in momentum_indicators:
                    ax.plot(signals.index, signals[col], label=col)
                ax.set_title('Momentum Indicators')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plot_idx += 1
            
            # Plot volatility indicators
            if volatility_indicators:
                ax = axes[plot_idx] if num_plots > 0 else axes
                for col in volatility_indicators:
                    ax.plot(signals.index, signals[col], label=col)
                ax.set_title('Volatility Indicators')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Set x-label for bottom subplot
            if num_plots > 0:
                axes[-1].set_xlabel('Date')
            else:
                axes.set_xlabel('Date')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plots['indicators'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            plt.close(fig)
    
    # Plot equity curve if trades are available
    if 'trades' in strategy_result and strategy_result['trades']:
        trades = strategy_result['trades']
        
        # Extract completed trades
        completed_trades = []
        for i in range(len(trades)):
            if trades[i]['type'] == 'exit':
                entry_index = trades[i].get('entry_trade_index')
                if entry_index is not None and entry_index < len(trades):
                    entry = trades[entry_index]
                    exit = trades[i]
                    completed_trades.append({
                        'entry_date': entry['date'],
                        'exit_date': exit['date'],
                        'profit_loss': exit['profit_loss']
                    })
        
        if completed_trades:
            # Sort trades by exit date
            completed_trades.sort(key=lambda x: x['exit_date'])
            
            # Create equity curve
            initial_capital = 10000  # Default initial capital
            equity = [initial_capital]
            dates = [completed_trades[0]['entry_date']]
            
            for trade in completed_trades:
                equity.append(equity[-1] + trade['profit_loss'])
                dates.append(trade['exit_date'])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, equity, label='Equity Curve')
            
            # Add drawdown
            peak = equity[0]
            drawdown = [0]
            for value in equity[1:]:
                if value > peak:
                    peak = value
                drawdown.append((peak - value) / peak * 100)
            
            ax2 = ax.twinx()
            ax2.fill_between(dates, drawdown, color='red', alpha=0.3, label='Drawdown %')
            
            # Add labels and title
            ax.set_title(f'Equity Curve for {ticker}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity ($)')
            ax2.set_ylabel('Drawdown (%)')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add performance metrics
            metrics_text = (
                f"Total Return: {strategy_result['performance'].get('total_return', 0):.2f}%\n"
                f"Sharpe Ratio: {strategy_result['performance'].get('sharpe_ratio', 0):.2f}\n"
                f"Max Drawdown: {strategy_result['performance'].get('max_drawdown', 0):.2f}%\n"
                f"Win Rate: {strategy_result['performance'].get('win_rate', 0)*100:.2f}%\n"
                f"Profit Factor: {strategy_result['performance'].get('profit_factor', 0):.2f}"
            )
            
            plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
            
            # Save plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plots['equity_curve'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            plt.close(fig)
    
    return plots

def generate_monte_carlo_plots(mc_results, ticker):
    """
    Generate plots for Monte Carlo simulation.
    
    Parameters:
    -----------
    mc_results : dict
        Monte Carlo simulation results
    ticker : str
        Ticker symbol
        
    Returns:
    --------
    dict
        Dictionary of plot URLs
    """
    plots = {}
    
    # Plot return distribution
    returns = [result['total_return'] for result in mc_results['simulation_results']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=50, alpha=0.7)
    
    # Add confidence interval
    ax.axvline(mc_results['return_ci'][0], color='red', linestyle='--', 
              label=f"{(1-mc_results['confidence_level'])/2*100:.1f}% Percentile")
    ax.axvline(mc_results['return_ci'][1], color='red', linestyle='--', 
              label=f"{(1-(1-mc_results['confidence_level'])/2)*100:.1f}% Percentile")
    
    # Add mean and median
    ax.axvline(mc_results['mean_return'], color='green', linestyle='-', label='Mean')
    ax.axvline(mc_results['median_return'], color='blue', linestyle='-', label='Median')
    
    # Add labels and title
    ax.set_title(f'Return Distribution for {ticker} ({mc_results["num_simulations"]} Simulations)')
    ax.set_xlabel('Total Return (%)')
    ax.set_ylabel('Frequency')
    
    # Add legend
    ax.legend()
    
    # Save plot to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plots['return_distribution'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    plt.close(fig)
    
    # Plot drawdown distribution
    drawdowns = [result['max_drawdown'] for result in mc_results['simulation_results']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(drawdowns, bins=50, alpha=0.7)
    
    # Add confidence interval
    ax.axvline(mc_results['drawdown_ci'][0], color='red', linestyle='--', 
              label=f"{(1-mc_results['confidence_level'])/2*100:.1f}% Percentile")
    ax.axvline(mc_results['drawdown_ci'][1], color='red', linestyle='--', 
              label=f"{(1-(1-mc_results['confidence_level'])/2)*100:.1f}% Percentile")
    
    # Add mean and median
    ax.axvline(mc_results['mean_drawdown'], color='green', linestyle='-', label='Mean')
    ax.axvline(mc_results['median_drawdown'], color='blue', linestyle='-', label='Median')
    
    # Add labels and title
    ax.set_title(f'Max Drawdown Distribution for {ticker} ({mc_results["num_simulations"]} Simulations)')
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_ylabel('Frequency')
    
    # Add legend
    ax.legend()
    
    # Save plot to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plots['drawdown_distribution'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    plt.close(fig)
    
    return plots

def open_browser():
    """Open browser after a delay."""
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create static directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Create index.html
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gemma Advanced Trading System</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                padding-top: 20px;
                padding-bottom: 20px;
            }
            .header {
                padding-bottom: 20px;
                margin-bottom: 30px;
                border-bottom: 1px solid #e5e5e5;
            }
            .footer {
                padding-top: 20px;
                margin-top: 30px;
                border-top: 1px solid #e5e5e5;
                text-align: center;
            }
            .plot-container {
                margin-bottom: 20px;
            }
            .plot-container img {
                max-width: 100%;
                height: auto;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .loading img {
                width: 50px;
                height: 50px;
            }
            .tab-content {
                padding: 20px;
                border: 1px solid #ddd;
                border-top: none;
            }
            .nav-tabs {
                margin-bottom: 0;
            }
            .result-section {
                display: none;
            }
            .parameter-group {
                margin-bottom: 15px;
                padding: 10px;
                border: 1px solid #eee;
                border-radius: 5px;
            }
            .parameter-group h5 {
                margin-top: 0;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }
            .trade-table {
                font-size: 0.9rem;
            }
            .positive {
                color: green;
            }
            .negative {
                color: red;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Gemma Advanced Trading System</h1>
                <p class="lead">Generate and analyze trading strategies using Gemma 3 AI</p>
            </div>
            
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h4>Generate Strategy</h4>
                        </div>
                        <div class="card-body">
                            <form id="strategy-form">
                                <div class="form-group">
                                    <label for="ticker">Ticker Symbol</label>
                                    <input type="text" class="form-control" id="ticker" name="ticker" placeholder="e.g., AAPL" required>
                                </div>
                                <div class="form-group">
                                    <label for="strategy-type">Strategy Type</label>
                                    <select class="form-control" id="strategy-type" name="strategy_type">
                                        <option value="swing">Swing Trading</option>
                                        <option value="day">Day Trading</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="period">Data Period</label>
                                    <select class="form-control" id="period" name="period">
                                        <option value="1mo">1 Month</option>
                                        <option value="3mo">3 Months</option>
                                        <option value="6mo">6 Months</option>
                                        <option value="1y" selected>1 Year</option>
                                        <option value="2y">2 Years</option>
                                        <option value="5y">5 Years</option>
                                        <option value="10y">10 Years</option>
                                        <option value="max">Max</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="interval">Data Interval</label>
                                    <select class="form-control" id="interval" name="interval">
                                        <option value="1m">1 Minute</option>
                                        <option value="5m">5 Minutes</option>
                                        <option value="15m">15 Minutes</option>
                                        <option value="30m">30 Minutes</option>
                                        <option value="60m">60 Minutes</option>
                                        <option value="1d" selected>1 Day</option>
                                        <option value="1wk">1 Week</option>
                                        <option value="1mo">1 Month</option>
                                    </select>
                                </div>
                                <div class="form-check mb-3">
                                    <input type="checkbox" class="form-check-input" id="optimize" name="optimize" value="true">
                                    <label class="form-check-label" for="optimize">Optimize Parameters</label>
                                </div>
                                <div class="form-check mb-3">
                                    <input type="checkbox" class="form-check-input" id="enhanced" name="enhanced" value="true" checked>
                                    <label class="form-check-label" for="enhanced">Use Enhanced Strategy</label>
                                </div>
                                <button type="submit" class="btn btn-primary btn-block">Generate Strategy</button>
                            </form>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-8">
                    <div class="loading">
                        <div class="spinner-border text-primary" role="status">
                            <span class="sr-only">Loading...</span>
                        </div>
                        <p>Processing... This may take a few minutes.</p>
                    </div>
                    
                    <div id="result-section" class="result-section">
                        <ul class="nav nav-tabs" id="result-tabs" role="tablist">
                            <li class="nav-item">
                                <a class="nav-link active" id="overview-tab" data-toggle="tab" href="#overview" role="tab">Overview</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" id="charts-tab" data-toggle="tab" href="#charts" role="tab">Charts</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" id="trades-tab" data-toggle="tab" href="#trades" role="tab">Trades</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" id="parameters-tab" data-toggle="tab" href="#parameters" role="tab">Parameters</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" id="analysis-tab" data-toggle="tab" href="#analysis" role="tab">Analysis</a>
                            </li>
                        </ul>
                        
                        <div class="tab-content" id="result-tabs-content">
                            <div class="tab-pane fade show active" id="overview" role="tabpanel">
                                <h3 id="strategy-title">Strategy Overview</h3>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="card mb-3">
                                            <div class="card-header">
                                                <h5>Performance Metrics</h5>
                                            </div>
                                            <div class="card-body">
                                                <table class="table table-sm">
                                                    <tbody id="performance-metrics">
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="card mb-3">
                                            <div class="card-header">
                                                <h5>Trade Statistics</h5>
                                            </div>
                                            <div class="card-body">
                                                <table class="table table-sm">
                                                    <tbody id="trade-statistics">
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="card mb-3">
                                    <div class="card-header">
                                        <h5>Actions</h5>
                                    </div>
                                    <div class="card-body">
                                        <div class="row">
                                            <div class="col-md-6">
                                                <button id="backtest-btn" class="btn btn-success btn-block mb-2">Run Detailed Backtest</button>
                                            </div>
                                            <div class="col-md-6">
                                                <button id="monte-carlo-btn" class="btn btn-info btn-block mb-2">Run Monte Carlo Simulation</button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div id="backtest-result" class="card mb-3" style="display: none;">
                                    <div class="card-header">
                                        <h5>Backtest Results</h5>
                                    </div>
                                    <div class="card-body">
                                        <p>Backtest completed successfully!</p>
                                        <a id="backtest-report-link" href="#" target="_blank" class="btn btn-primary">View Detailed Report</a>
                                    </div>
                                </div>
                                
                                <div id="monte-carlo-result" class="card mb-3" style="display: none;">
                                    <div class="card-header">
                                        <h5>Monte Carlo Simulation Results</h5>
                                    </div>
                                    <div class="card-body">
                                        <div class="row">
                                            <div class="col-md-6">
                                                <table class="table table-sm">
                                                    <tbody id="monte-carlo-metrics">
                                                    </tbody>
                                                </table>
                                            </div>
                                            <div class="col-md-6">
                                                <div id="monte-carlo-plots">
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="tab-pane fade" id="charts" role="tabpanel">
                                <h3>Strategy Charts</h3>
                                <div id="strategy-plots">
                                </div>
                            </div>
                            
                            <div class="tab-pane fade" id="trades" role="tabpanel">
                                <h3>Trade List</h3>
                                <div class="table-responsive">
                                    <table class="table table-striped table-sm trade-table">
                                        <thead>
                                            <tr>
                                                <th>#</th>
                                                <th>Type</th>
                                                <th>Date</th>
                                                <th>Price</th>
                                                <th>Size</th>
                                                <th>Value</th>
                                                <th>P&L</th>
                                            </tr>
                                        </thead>
                                        <tbody id="trade-list">
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            
                            <div class="tab-pane fade" id="parameters" role="tabpanel">
                                <h3>Strategy Parameters</h3>
                                <div id="strategy-parameters">
                                </div>
                            </div>
                            
                            <div class="tab-pane fade" id="analysis" role="tabpanel">
                                <h3>Strategy Analysis</h3>
                                <div id="strategy-description" class="card">
                                    <div class="card-body">
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>&copy; 2025 Gemma Advanced Trading System</p>
            </div>
        </div>
        
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <script>
            $(document).ready(function() {
                let currentStrategyId = null;
                
                // Handle strategy form submission
                $('#strategy-form').submit(function(e) {
                    e.preventDefault();
                    
                    // Show loading indicator
                    $('.loading').show();
                    $('#result-section').hide();
                    
                    // Get form data
                    const formData = new FormData(this);
                    
                    // Send request
                    $.ajax({
                        url: '/generate_strategy',
                        type: 'POST',
                        data: formData,
                        processData: false,
                        contentType: false,
                        success: function(response) {
                            // Hide loading indicator
                            $('.loading').hide();
                            
                            // Store strategy ID
                            currentStrategyId = response.strategy_id;
                            
                            // Update strategy title
                            $('#strategy-title').text(`${response.strategy_type.toUpperCase()} Strategy for ${response.ticker}`);
                            
                            // Update performance metrics
                            const performanceMetrics = $('#performance-metrics');
                            performanceMetrics.empty();
                            
                            const keyMetrics = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate', 'profit_factor'];
                            for (const key of keyMetrics) {
                                if (response.performance[key]) {
                                    const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                    performanceMetrics.append(`
                                        <tr>
                                            <td>${formattedKey}</td>
                                            <td>${response.performance[key]}</td>
                                        </tr>
                                    `);
                                }
                            }
                            
                            // Update trade statistics
                            const tradeStatistics = $('#trade-statistics');
                            tradeStatistics.empty();
                            
                            const tradeMetrics = ['total_trades', 'winning_trades', 'losing_trades', 'avg_profit', 'avg_loss', 'avg_holding_period'];
                            for (const key of tradeMetrics) {
                                if (response.performance[key]) {
                                    const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                    tradeStatistics.append(`
                                        <tr>
                                            <td>${formattedKey}</td>
                                            <td>${response.performance[key]}</td>
                                        </tr>
                                    `);
                                }
                            }
                            
                            // Update strategy plots
                            const strategyPlots = $('#strategy-plots');
                            strategyPlots.empty();
                            
                            if (response.plots) {
                                if (response.plots.price_signals) {
                                    strategyPlots.append(`
                                        <div class="plot-container">
                                            <h4>Price and Signals</h4>
                                            <img src="${response.plots.price_signals}" alt="Price and Signals">
                                        </div>
                                    `);
                                }
                                
                                if (response.plots.indicators) {
                                    strategyPlots.append(`
                                        <div class="plot-container">
                                            <h4>Technical Indicators</h4>
                                            <img src="${response.plots.indicators}" alt="Technical Indicators">
                                        </div>
                                    `);
                                }
                                
                                if (response.plots.equity_curve) {
                                    strategyPlots.append(`
                                        <div class="plot-container">
                                            <h4>Equity Curve</h4>
                                            <img src="${response.plots.equity_curve}" alt="Equity Curve">
                                        </div>
                                    `);
                                }
                            }
                            
                            // Update trade list
                            const tradeList = $('#trade-list');
                            tradeList.empty();
                            
                            if (response.trades && response.trades.length > 0) {
                                response.trades.forEach((trade, index) => {
                                    let pnlClass = '';
                                    let pnlValue = '';
                                    
                                    if (trade.type === 'exit' && trade.profit_loss) {
                                        pnlClass = trade.profit_loss > 0 ? 'positive' : 'negative';
                                        pnlValue = `$${trade.profit_loss.toFixed(2)}`;
                                    }
                                    
                                    tradeList.append(`
                                        <tr>
                                            <td>${index + 1}</td>
                                            <td>${trade.type.charAt(0).toUpperCase() + trade.type.slice(1)}</td>
                                            <td>${trade.date}</td>
                                            <td>$${trade.price.toFixed(2)}</td>
                                            <td>${trade.position_size.toFixed(2)}</td>
                                            <td>$${trade.position_value.toFixed(2)}</td>
                                            <td class="${pnlClass}">${pnlValue}</td>
                                        </tr>
                                    `);
                                });
                            } else {
                                tradeList.append(`
                                    <tr>
                                        <td colspan="7" class="text-center">No trades generated</td>
                                    </tr>
                                `);
                            }
                            
                            // Update strategy parameters
                            const strategyParameters = $('#strategy-parameters');
                            strategyParameters.empty();
                            
                            if (response.parameters) {
                                // Group parameters by type
                                const paramGroups = {
                                    'Trend Indicators': {},
                                    'Momentum Indicators': {},
                                    'Volatility Indicators': {},
                                    'Volume Indicators': {},
                                    'Risk Management': {},
                                    'Strategy Settings': {}
                                };
                                
                                // Categorize parameters
                                for (const [key, value] of Object.entries(response.parameters)) {
                                    if (key.includes('sma') || key.includes('ema') || key.includes('macd')) {
                                        paramGroups['Trend Indicators'][key] = value;
                                    } else if (key.includes('rsi') || key.includes('stochastic') || key.includes('cci')) {
                                        paramGroups['Momentum Indicators'][key] = value;
                                    } else if (key.includes('atr') || key.includes('bollinger') || key.includes('std')) {
                                        paramGroups['Volatility Indicators'][key] = value;
                                    } else if (key.includes('obv') || key.includes('vwap') || key.includes('volume')) {
                                        paramGroups['Volume Indicators'][key] = value;
                                    } else if (key.includes('risk') || key.includes('stop') || key.includes('profit')) {
                                        paramGroups['Risk Management'][key] = value;
                                    } else {
                                        paramGroups['Strategy Settings'][key] = value;
                                    }
                                }
                                
                                // Create parameter groups
                                for (const [groupName, params] of Object.entries(paramGroups)) {
                                    if (Object.keys(params).length > 0) {
                                        const paramGroup = $(`
                                            <div class="parameter-group">
                                                <h5>${groupName}</h5>
                                                <table class="table table-sm">
                                                    <tbody></tbody>
                                                </table>
                                            </div>
                                        `);
                                        
                                        const tbody = paramGroup.find('tbody');
                                        
                                        for (const [key, value] of Object.entries(params)) {
                                            const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                            tbody.append(`
                                                <tr>
                                                    <td>${formattedKey}</td>
                                                    <td>${value}</td>
                                                </tr>
                                            `);
                                        }
                                        
                                        strategyParameters.append(paramGroup);
                                    }
                                }
                            }
                            
                            // Update strategy description
                            const strategyDescription = $('#strategy-description .card-body');
                            strategyDescription.empty();
                            
                            if (response.description) {
                                strategyDescription.html(`<pre>${response.description}</pre>`);
                            } else {
                                strategyDescription.html('<p>No detailed analysis available.</p>');
                            }
                            
                            // Show result section
                            $('#result-section').show();
                            
                            // Activate first tab
                            $('#result-tabs a:first').tab('show');
                        },
                        error: function(xhr) {
                            // Hide loading indicator
                            $('.loading').hide();
                            
                            // Show error
                            alert('Error: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error'));
                        }
                    });
                });
                
                // Handle backtest button click
                $('#backtest-btn').click(function() {
                    if (!currentStrategyId) {
                        alert('Please generate a strategy first.');
                        return;
                    }
                    
                    // Show loading indicator
                    $('.loading').show();
                    $('#backtest-result').hide();
                    
                    // Send request
                    $.ajax({
                        url: '/backtest_strategy',
                        type: 'POST',
                        data: {
                            strategy_id: currentStrategyId
                        },
                        success: function(response) {
                            // Hide loading indicator
                            $('.loading').hide();
                            
                            // Update backtest result
                            $('#backtest-report-link').attr('href', response.report_url);
                            $('#backtest-result').show();
                        },
                        error: function(xhr) {
                            // Hide loading indicator
                            $('.loading').hide();
                            
                            // Show error
                            alert('Error: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error'));
                        }
                    });
                });
                
                // Handle Monte Carlo button click
                $('#monte-carlo-btn').click(function() {
                    if (!currentStrategyId) {
                        alert('Please generate a strategy first.');
                        return;
                    }
                    
                    // Show loading indicator
                    $('.loading').show();
                    $('#monte-carlo-result').hide();
                    
                    // Send request
                    $.ajax({
                        url: '/monte_carlo',
                        type: 'POST',
                        data: {
                            strategy_id: currentStrategyId,
                            num_simulations: 1000,
                            confidence_level: 0.95
                        },
                        success: function(response) {
                            // Hide loading indicator
                            $('.loading').hide();
                            
                            // Update Monte Carlo metrics
                            const monteCarloMetrics = $('#monte-carlo-metrics');
                            monteCarloMetrics.empty();
                            
                            for (const [key, value] of Object.entries(response.results)) {
                                const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                monteCarloMetrics.append(`
                                    <tr>
                                        <td>${formattedKey}</td>
                                        <td>${value}</td>
                                    </tr>
                                `);
                            }
                            
                            // Update Monte Carlo plots
                            const monteCarloPlots = $('#monte-carlo-plots');
                            monteCarloPlots.empty();
                            
                            if (response.plots) {
                                if (response.plots.return_distribution) {
                                    monteCarloPlots.append(`
                                        <div class="plot-container">
                                            <img src="${response.plots.return_distribution}" alt="Return Distribution">
                                        </div>
                                    `);
                                }
                                
                                if (response.plots.drawdown_distribution) {
                                    monteCarloPlots.append(`
                                        <div class="plot-container">
                                            <img src="${response.plots.drawdown_distribution}" alt="Drawdown Distribution">
                                        </div>
                                    `);
                                }
                            }
                            
                            // Show Monte Carlo result
                            $('#monte-carlo-result').show();
                        },
                        error: function(xhr) {
                            // Hide loading indicator
                            $('.loading').hide();
                            
                            // Show error
                            alert('Error: ' + (xhr.responseJSON ? xhr.responseJSON.error : 'Unknown error'));
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    """
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    # Start browser after a delay
    threading.Timer(1.0, open_browser).start()
    
    # Start Flask app
    app.run(debug=True)
