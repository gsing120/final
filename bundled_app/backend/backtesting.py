"""
Backtesting and Optimization Module for Gemma Advanced Trading System.

This module provides advanced backtesting and optimization capabilities for trading strategies.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os

# Import system components
import data_access
from strategy_builder import Strategy

# Configure logging
logger = logging.getLogger("GemmaTrading.Backtesting")

class BacktestResult:
    """Class to store and analyze backtest results."""
    
    def __init__(self, strategy_name, ticker, parameters, signals, trades, performance):
        """
        Initialize backtest result.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        ticker : str
            Ticker symbol
        parameters : dict
            Strategy parameters
        signals : pandas.DataFrame
            Data with signals
        trades : list
            List of trades
        performance : dict
            Performance metrics
        """
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.parameters = parameters
        self.signals = signals
        self.trades = trades
        self.performance = performance
        self.logger = logging.getLogger(f"GemmaTrading.BacktestResult.{strategy_name}")
    
    def plot_equity_curve(self, save_path=None):
        """
        Plot equity curve.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.trades:
            self.logger.warning("No trades to plot equity curve")
            return None
        
        # Extract completed trades
        completed_trades = []
        for i in range(len(self.trades)):
            if self.trades[i]['type'] == 'exit':
                entry_index = self.trades[i].get('entry_trade_index')
                if entry_index is not None and entry_index < len(self.trades):
                    entry = self.trades[entry_index]
                    exit = self.trades[i]
                    completed_trades.append({
                        'entry_date': entry['date'],
                        'exit_date': exit['date'],
                        'profit_loss': exit['profit_loss']
                    })
        
        if not completed_trades:
            self.logger.warning("No completed trades to plot equity curve")
            return None
        
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
        ax.set_title(f'Equity Curve for {self.strategy_name} on {self.ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax2.set_ylabel('Drawdown (%)')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Add performance metrics
        metrics_text = (
            f"Total Return: {self.performance.get('total_return', 0):.2f}%\n"
            f"Sharpe Ratio: {self.performance.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {self.performance.get('max_drawdown', 0):.2f}%\n"
            f"Win Rate: {self.performance.get('win_rate', 0)*100:.2f}%\n"
            f"Profit Factor: {self.performance.get('profit_factor', 0):.2f}"
        )
        
        plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trades(self, save_path=None):
        """
        Plot trades on price chart.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.trades or self.signals is None:
            self.logger.warning("No trades or signals to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot price
        ax.plot(self.signals.index, self.signals['Close'], label='Close Price')
        
        # Plot buy signals
        buy_signals = self.signals[self.signals['Signal'] == 1]
        ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = self.signals[self.signals['Signal'] == -1]
        ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
        
        # Plot stop loss and take profit levels
        for i in range(len(self.trades)):
            if self.trades[i]['type'] == 'entry':
                entry_date = self.trades[i]['date']
                entry_price = self.trades[i]['price']
                
                # Plot entry point
                ax.scatter(entry_date, entry_price, marker='o', color='blue', s=80)
                
                # Plot stop loss if available
                if 'stop_loss' in self.trades[i]:
                    stop_loss = self.trades[i]['stop_loss']
                    ax.hlines(stop_loss, entry_date, entry_date + timedelta(days=30), 
                             colors='red', linestyles='dashed', label='_nolegend_')
                
                # Plot take profit if available
                if 'take_profit' in self.trades[i]:
                    take_profit = self.trades[i]['take_profit']
                    ax.hlines(take_profit, entry_date, entry_date + timedelta(days=30), 
                             colors='green', linestyles='dashed', label='_nolegend_')
        
        # Add labels and title
        ax.set_title(f'Trades for {self.strategy_name} on {self.ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_performance_metrics(self, save_path=None):
        """
        Plot performance metrics.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not self.performance:
            self.logger.warning("No performance metrics to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract metrics
        metrics = {
            'Win Rate': self.performance.get('win_rate', 0) * 100,
            'Profit Factor': self.performance.get('profit_factor', 0),
            'Sharpe Ratio': self.performance.get('sharpe_ratio', 0),
            'Sortino Ratio': self.performance.get('sortino_ratio', 0),
            'Calmar Ratio': min(self.performance.get('calmar_ratio', 0), 10)  # Cap at 10 for visualization
        }
        
        # Create bar chart
        bars = ax.bar(metrics.keys(), metrics.values())
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Add labels and title
        ax.set_title(f'Performance Metrics for {self.strategy_name} on {self.ticker}')
        ax.set_ylabel('Value')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_report(self, output_dir):
        """
        Save comprehensive backtest report.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the report
            
        Returns:
        --------
        str
            Path to the report directory
        """
        # Create output directory if it doesn't exist
        report_dir = os.path.join(output_dir, f"{self.ticker}_{self.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Save performance metrics
        with open(os.path.join(report_dir, 'performance.json'), 'w') as f:
            json.dump(self.performance, f, indent=4, default=str)
        
        # Save parameters
        with open(os.path.join(report_dir, 'parameters.json'), 'w') as f:
            json.dump(self.parameters, f, indent=4, default=str)
        
        # Save trades
        trades_df = pd.DataFrame([
            {k: v for k, v in trade.items() if k != 'entry_trade_index'} 
            for trade in self.trades
        ])
        trades_df.to_csv(os.path.join(report_dir, 'trades.csv'), index=False)
        
        # Save signals
        if self.signals is not None:
            self.signals.to_csv(os.path.join(report_dir, 'signals.csv'))
        
        # Generate and save plots
        self.plot_equity_curve(save_path=os.path.join(report_dir, 'equity_curve.png'))
        self.plot_trades(save_path=os.path.join(report_dir, 'trades.png'))
        self.plot_performance_metrics(save_path=os.path.join(report_dir, 'performance_metrics.png'))
        
        # Generate HTML report
        html_report = self._generate_html_report()
        with open(os.path.join(report_dir, 'report.html'), 'w') as f:
            f.write(html_report)
        
        self.logger.info(f"Backtest report saved to {report_dir}")
        return report_dir
    
    def _generate_html_report(self):
        """
        Generate HTML report.
        
        Returns:
        --------
        str
            HTML report
        """
        # Format performance metrics
        performance_html = ""
        for key, value in self.performance.items():
            if isinstance(value, (int, float)):
                if key in ['win_rate']:
                    formatted_value = f"{value * 100:.2f}%"
                elif key in ['total_return', 'max_drawdown']:
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            performance_html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
        
        # Format parameters
        parameters_html = ""
        for key, value in self.parameters.items():
            parameters_html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        # Format trades
        trades_html = ""
        for i, trade in enumerate(self.trades):
            if trade['type'] == 'entry':
                trade_type = "Entry"
                direction = trade.get('direction', 'long').capitalize()
                price = f"${trade['price']:.2f}"
                size = f"{trade['position_size']:.2f}"
                value = f"${trade['position_value']:.2f}"
                date = trade['date'].strftime('%Y-%m-%d %H:%M')
                
                trades_html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{trade_type}</td>
                    <td>{direction}</td>
                    <td>{date}</td>
                    <td>{price}</td>
                    <td>{size}</td>
                    <td>{value}</td>
                    <td>-</td>
                </tr>
                """
            elif trade['type'] == 'exit':
                trade_type = "Exit"
                price = f"${trade['price']:.2f}"
                size = f"{trade['position_size']:.2f}"
                value = f"${trade['position_value']:.2f}"
                date = trade['date'].strftime('%Y-%m-%d %H:%M')
                pnl = trade['profit_loss']
                pnl_class = "positive" if pnl > 0 else "negative"
                pnl_str = f"${pnl:.2f}"
                
                trades_html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{trade_type}</td>
                    <td>{trade.get('exit_reason', '-').capitalize()}</td>
                    <td>{date}</td>
                    <td>{price}</td>
                    <td>{size}</td>
                    <td>{value}</td>
                    <td class="{pnl_class}">{pnl_str}</td>
                </tr>
                """
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report: {self.strategy_name} on {self.ticker}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .summary {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }}
                .summary-box {{
                    border: 1px solid #ddd;
                    padding: 15px;
                    border-radius: 5px;
                    width: 30%;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .chart-container {{
                    margin: 20px 0;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <h1>Backtest Report: {self.strategy_name} on {self.ticker}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <div class="summary-box">
                    <h3>Key Metrics</h3>
                    <p>Total Return: <span class="{('positive' if self.performance.get('total_return', 0) > 0 else 'negative')}">{self.performance.get('total_return', 0):.2f}%</span></p>
                    <p>Sharpe Ratio: {self.performance.get('sharpe_ratio', 0):.2f}</p>
                    <p>Win Rate: {self.performance.get('win_rate', 0) * 100:.2f}%</p>
                </div>
                <div class="summary-box">
                    <h3>Trade Summary</h3>
                    <p>Total Trades: {self.performance.get('total_trades', 0)}</p>
                    <p>Winning Trades: {self.performance.get('winning_trades', 0)}</p>
                    <p>Losing Trades: {self.performance.get('losing_trades', 0)}</p>
                </div>
                <div class="summary-box">
                    <h3>Risk Metrics</h3>
                    <p>Max Drawdown: {self.performance.get('max_drawdown', 0):.2f}%</p>
                    <p>Profit Factor: {self.performance.get('profit_factor', 0):.2f}</p>
                    <p>Calmar Ratio: {self.performance.get('calmar_ratio', 0):.2f}</p>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Equity Curve</h2>
                <img src="equity_curve.png" alt="Equity Curve">
            </div>
            
            <div class="chart-container">
                <h2>Trades</h2>
                <img src="trades.png" alt="Trades">
            </div>
            
            <div class="chart-container">
                <h2>Performance Metrics</h2>
                <img src="performance_metrics.png" alt="Performance Metrics">
            </div>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                {performance_html}
            </table>
            
            <h2>Strategy Parameters</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                {parameters_html}
            </table>
            
            <h2>Trades</h2>
            <table>
                <tr>
                    <th>#</th>
                    <th>Type</th>
                    <th>Direction/Reason</th>
                    <th>Date</th>
                    <th>Price</th>
                    <th>Size</th>
                    <th>Value</th>
                    <th>P&L</th>
                </tr>
                {trades_html}
            </table>
        </body>
        </html>
        """
        
        return html


class Backtester:
    """Advanced backtesting engine for trading strategies."""
    
    def __init__(self):
        """Initialize the backtester."""
        self.logger = logging.getLogger("GemmaTrading.Backtester")
    
    def backtest_strategy(self, strategy, data, initial_capital=10000.0):
        """
        Backtest a strategy on historical data.
        
        Parameters:
        -----------
        strategy : Strategy
            Strategy to backtest
        data : pandas.DataFrame
            Market data
        initial_capital : float, optional
            Initial capital for backtesting
            
        Returns:
        --------
        BacktestResult
            Backtest result
        """
        self.logger.info(f"Backtesting strategy {strategy.name}")
        
        # Run backtest
        backtest_result = strategy.backtest(data, initial_capital)
        
        # Create and return BacktestResult object
        result = BacktestResult(
            strategy_name=strategy.name,
            ticker=data.name if hasattr(data, 'name') else "Unknown",
            parameters=strategy.parameters,
            signals=backtest_result['signals'],
            trades=backtest_result['trades'],
            performance=backtest_result['performance']
        )
        
        return result
    
    def walk_forward_analysis(self, strategy_class, data, train_size=0.7, window_size=None, step_size=None, 
                             initial_capital=10000.0, parameter_grid=None, metric='sharpe_ratio'):
        """
        Perform walk-forward analysis.
        
        Parameters:
        -----------
        strategy_class : class
            Strategy class to use
        data : pandas.DataFrame
            Market data
        train_size : float, optional
            Proportion of data to use for training
        window_size : int, optional
            Size of the rolling window in days
        step_size : int, optional
            Step size for the rolling window in days
        initial_capital : float, optional
            Initial capital for backtesting
        parameter_grid : dict, optional
            Dictionary of parameter names and lists of values to try
        metric : str, optional
            Performance metric to optimize for
            
        Returns:
        --------
        dict
            Walk-forward analysis results
        """
        self.logger.info("Performing walk-forward analysis")
        
        # Determine window and step size if not provided
        if window_size is None:
            window_size = len(data) // 3
        
        if step_size is None:
            step_size = window_size // 2
        
        # Initialize results
        results = {
            'windows': [],
            'parameters': [],
            'train_performance': [],
            'test_performance': [],
            'combined_performance': {}
        }
        
        # Initialize combined equity curve
        combined_equity = initial_capital
        combined_trades = []
        
        # Iterate through windows
        for i in range(0, len(data) - window_size, step_size):
            # Extract window
            window_data = data.iloc[i:i+window_size]
            
            # Split into train and test
            train_size_idx = int(len(window_data) * train_size)
            train_data = window_data.iloc[:train_size_idx]
            test_data = window_data.iloc[train_size_idx:]
            
            # Create and optimize strategy on training data
            strategy = strategy_class(name=f"WFA_Window_{i}")
            
            if parameter_grid:
                optimization_result = strategy.optimize(train_data, parameter_grid, initial_capital, metric)
                best_params = optimization_result['best_parameters']
                train_performance = optimization_result['best_performance']
            else:
                # Use default parameters
                best_params = strategy.parameters
                backtest_result = strategy.backtest(train_data, initial_capital)
                train_performance = backtest_result['performance']
            
            # Test strategy on test data
            test_backtest_result = strategy.backtest(test_data, initial_capital)
            test_performance = test_backtest_result['performance']
            
            # Store results
            results['windows'].append((window_data.index[0], window_data.index[-1]))
            results['parameters'].append(best_params)
            results['train_performance'].append(train_performance)
            results['test_performance'].append(test_performance)
            
            # Update combined equity and trades
            for trade in test_backtest_result['trades']:
                if trade['type'] == 'exit':
                    combined_equity += trade['profit_loss']
                    
                    # Adjust trade for combined list
                    adjusted_trade = trade.copy()
                    adjusted_trade['window_index'] = i
                    combined_trades.append(adjusted_trade)
        
        # Calculate combined performance
        if combined_trades:
            # Calculate return
            total_return = (combined_equity - initial_capital) / initial_capital * 100
            
            # Calculate win rate
            winning_trades = sum(1 for trade in combined_trades if trade['profit_loss'] > 0)
            total_trades = len(combined_trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            total_profit = sum(trade['profit_loss'] for trade in combined_trades if trade['profit_loss'] > 0)
            total_loss = sum(trade['profit_loss'] for trade in combined_trades if trade['profit_loss'] <= 0)
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            
            results['combined_performance'] = {
                'total_return': total_return,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'final_capital': combined_equity
            }
        
        return results
    
    def monte_carlo_simulation(self, strategy, data, initial_capital=10000.0, num_simulations=1000, 
                              confidence_level=0.95):
        """
        Perform Monte Carlo simulation.
        
        Parameters:
        -----------
        strategy : Strategy
            Strategy to simulate
        data : pandas.DataFrame
            Market data
        initial_capital : float, optional
            Initial capital for backtesting
        num_simulations : int, optional
            Number of simulations to run
        confidence_level : float, optional
            Confidence level for results
            
        Returns:
        --------
        dict
            Monte Carlo simulation results
        """
        self.logger.info(f"Performing Monte Carlo simulation for strategy {strategy.name}")
        
        # Run backtest to get trades
        backtest_result = strategy.backtest(data, initial_capital)
        trades = backtest_result['trades']
        
        # Extract profit/loss from completed trades
        pnl_values = []
        for i in range(len(trades)):
            if trades[i]['type'] == 'exit':
                pnl_values.append(trades[i]['profit_loss'])
        
        if not pnl_values:
            self.logger.warning("No completed trades for Monte Carlo simulation")
            return None
        
        # Run simulations
        simulation_results = []
        
        for _ in range(num_simulations):
            # Shuffle profit/loss values
            np.random.shuffle(pnl_values)
            
            # Calculate equity curve
            equity = [initial_capital]
            for pnl in pnl_values:
                equity.append(equity[-1] + pnl)
            
            # Calculate metrics
            final_equity = equity[-1]
            total_return = (final_equity - initial_capital) / initial_capital * 100
            
            # Calculate drawdown
            peak = equity[0]
            max_drawdown = 0
            for value in equity:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            simulation_results.append({
                'equity_curve': equity,
                'final_equity': final_equity,
                'total_return': total_return,
                'max_drawdown': max_drawdown
            })
        
        # Calculate statistics
        returns = [result['total_return'] for result in simulation_results]
        drawdowns = [result['max_drawdown'] for result in simulation_results]
        final_equities = [result['final_equity'] for result in simulation_results]
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_lower_idx = int(alpha/2 * num_simulations)
        ci_upper_idx = int((1-alpha/2) * num_simulations)
        
        returns_sorted = sorted(returns)
        drawdowns_sorted = sorted(drawdowns)
        final_equities_sorted = sorted(final_equities)
        
        return {
            'num_simulations': num_simulations,
            'confidence_level': confidence_level,
            'original_performance': backtest_result['performance'],
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'mean_drawdown': np.mean(drawdowns),
            'median_drawdown': np.median(drawdowns),
            'mean_final_equity': np.mean(final_equities),
            'median_final_equity': np.median(final_equities),
            'return_ci': (returns_sorted[ci_lower_idx], returns_sorted[ci_upper_idx]),
            'drawdown_ci': (drawdowns_sorted[ci_lower_idx], drawdowns_sorted[ci_upper_idx]),
            'final_equity_ci': (final_equities_sorted[ci_lower_idx], final_equities_sorted[ci_upper_idx]),
            'worst_case_return': min(returns),
            'best_case_return': max(returns),
            'worst_case_drawdown': max(drawdowns),
            'simulation_results': simulation_results
        }
    
    def parameter_sensitivity_analysis(self, strategy_class, data, base_parameters, 
                                      parameter_ranges, initial_capital=10000.0):
        """
        Perform parameter sensitivity analysis.
        
        Parameters:
        -----------
        strategy_class : class
            Strategy class to use
        data : pandas.DataFrame
            Market data
        base_parameters : dict
            Base parameters to use
        parameter_ranges : dict
            Dictionary of parameter names and ranges to test
        initial_capital : float, optional
            Initial capital for backtesting
            
        Returns:
        --------
        dict
            Sensitivity analysis results
        """
        self.logger.info("Performing parameter sensitivity analysis")
        
        results = {}
        
        # Iterate through parameters
        for param_name, param_range in parameter_ranges.items():
            param_results = []
            
            for param_value in param_range:
                # Create parameters with current value
                current_params = base_parameters.copy()
                current_params[param_name] = param_value
                
                # Create and backtest strategy
                strategy = strategy_class(name=f"Sensitivity_{param_name}_{param_value}")
                strategy.set_parameters(current_params)
                backtest_result = strategy.backtest(data, initial_capital)
                
                # Store results
                param_results.append({
                    'param_value': param_value,
                    'performance': backtest_result['performance']
                })
            
            results[param_name] = param_results
        
        return results
    
    def plot_sensitivity_analysis(self, sensitivity_results, metrics=None, save_path=None):
        """
        Plot parameter sensitivity analysis results.
        
        Parameters:
        -----------
        sensitivity_results : dict
            Results from parameter_sensitivity_analysis
        metrics : list, optional
            List of metrics to plot
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not sensitivity_results:
            self.logger.warning("No sensitivity results to plot")
            return None
        
        # Default metrics if not provided
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        # Create figure
        num_params = len(sensitivity_results)
        num_metrics = len(metrics)
        
        fig, axes = plt.subplots(num_params, num_metrics, figsize=(num_metrics*4, num_params*3))
        
        # Handle single parameter case
        if num_params == 1:
            axes = axes.reshape(1, -1)
        
        # Iterate through parameters
        for i, (param_name, param_results) in enumerate(sensitivity_results.items()):
            # Extract values
            param_values = [result['param_value'] for result in param_results]
            
            # Iterate through metrics
            for j, metric in enumerate(metrics):
                metric_values = [result['performance'].get(metric, 0) for result in param_results]
                
                # Plot
                ax = axes[i, j]
                ax.plot(param_values, metric_values, marker='o')
                
                # Add labels
                ax.set_xlabel(param_name)
                ax.set_ylabel(metric)
                
                # Add title for top row
                if i == 0:
                    ax.set_title(metric.replace('_', ' ').title())
                
                # Add grid
                ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class Optimizer:
    """Advanced optimization engine for trading strategies."""
    
    def __init__(self):
        """Initialize the optimizer."""
        self.logger = logging.getLogger("GemmaTrading.Optimizer")
    
    def grid_search(self, strategy_class, data, parameter_grid, initial_capital=10000.0, 
                   metric='sharpe_ratio', n_jobs=None):
        """
        Perform grid search optimization.
        
        Parameters:
        -----------
        strategy_class : class
            Strategy class to use
        data : pandas.DataFrame
            Market data
        parameter_grid : dict
            Dictionary of parameter names and lists of values to try
        initial_capital : float, optional
            Initial capital for backtesting
        metric : str, optional
            Performance metric to optimize for
        n_jobs : int, optional
            Number of parallel jobs to run
            
        Returns:
        --------
        dict
            Optimization results
        """
        self.logger.info("Performing grid search optimization")
        
        # Create base strategy
        base_strategy = strategy_class(name="GridSearch_Base")
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Determine number of jobs
        if n_jobs is None:
            n_jobs = min(multiprocessing.cpu_count(), len(param_combinations))
        
        # Initialize results
        results = []
        
        # Define worker function
        def evaluate_parameters(params_dict):
            strategy = strategy_class(name=f"GridSearch_{hash(str(params_dict))}")
            strategy.set_parameters(params_dict)
            backtest_result = strategy.backtest(data, initial_capital)
            
            return {
                'parameters': params_dict,
                'performance': backtest_result['performance']
            }
        
        # Run grid search
        if n_jobs > 1:
            # Parallel execution
            with multiprocessing.Pool(n_jobs) as pool:
                param_dicts = [dict(zip(param_names, combo)) for combo in param_combinations]
                results = list(tqdm(pool.imap(evaluate_parameters, param_dicts), total=len(param_dicts)))
        else:
            # Sequential execution
            for combo in tqdm(param_combinations):
                params_dict = dict(zip(param_names, combo))
                results.append(evaluate_parameters(params_dict))
        
        # Find best parameters
        best_result = max(results, key=lambda x: x['performance'].get(metric, float('-inf')))
        
        return {
            'best_parameters': best_result['parameters'],
            'best_performance': best_result['performance'],
            'all_results': results
        }
    
    def random_search(self, strategy_class, data, parameter_space, num_iterations=100, 
                     initial_capital=10000.0, metric='sharpe_ratio', n_jobs=None):
        """
        Perform random search optimization.
        
        Parameters:
        -----------
        strategy_class : class
            Strategy class to use
        data : pandas.DataFrame
            Market data
        parameter_space : dict
            Dictionary of parameter names and (min, max) ranges or lists of values
        num_iterations : int, optional
            Number of random combinations to try
        initial_capital : float, optional
            Initial capital for backtesting
        metric : str, optional
            Performance metric to optimize for
        n_jobs : int, optional
            Number of parallel jobs to run
            
        Returns:
        --------
        dict
            Optimization results
        """
        self.logger.info("Performing random search optimization")
        
        # Create base strategy
        base_strategy = strategy_class(name="RandomSearch_Base")
        
        # Determine number of jobs
        if n_jobs is None:
            n_jobs = min(multiprocessing.cpu_count(), num_iterations)
        
        # Initialize results
        results = []
        
        # Generate random parameter combinations
        param_combinations = []
        for _ in range(num_iterations):
            params_dict = {}
            for param_name, param_range in parameter_space.items():
                if isinstance(param_range, (list, tuple)) and len(param_range) == 2 and all(isinstance(x, (int, float)) for x in param_range):
                    # Continuous range
                    min_val, max_val = param_range
                    if all(isinstance(x, int) for x in param_range):
                        # Integer range
                        params_dict[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        # Float range
                        params_dict[param_name] = np.random.uniform(min_val, max_val)
                elif isinstance(param_range, (list, tuple)):
                    # Discrete values
                    params_dict[param_name] = np.random.choice(param_range)
                else:
                    raise ValueError(f"Invalid parameter range for {param_name}: {param_range}")
            
            param_combinations.append(params_dict)
        
        # Define worker function
        def evaluate_parameters(params_dict):
            strategy = strategy_class(name=f"RandomSearch_{hash(str(params_dict))}")
            strategy.set_parameters(params_dict)
            backtest_result = strategy.backtest(data, initial_capital)
            
            return {
                'parameters': params_dict,
                'performance': backtest_result['performance']
            }
        
        # Run random search
        if n_jobs > 1:
            # Parallel execution
            with multiprocessing.Pool(n_jobs) as pool:
                results = list(tqdm(pool.imap(evaluate_parameters, param_combinations), total=len(param_combinations)))
        else:
            # Sequential execution
            for params_dict in tqdm(param_combinations):
                results.append(evaluate_parameters(params_dict))
        
        # Find best parameters
        best_result = max(results, key=lambda x: x['performance'].get(metric, float('-inf')))
        
        return {
            'best_parameters': best_result['parameters'],
            'best_performance': best_result['performance'],
            'all_results': results
        }
    
    def plot_optimization_results(self, optimization_results, top_n=10, metrics=None, save_path=None):
        """
        Plot optimization results.
        
        Parameters:
        -----------
        optimization_results : dict
            Results from grid_search or random_search
        top_n : int, optional
            Number of top results to show
        metrics : list, optional
            List of metrics to plot
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if not optimization_results or 'all_results' not in optimization_results:
            self.logger.warning("No optimization results to plot")
            return None
        
        # Default metrics if not provided
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        # Extract results
        all_results = optimization_results['all_results']
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
        
        # Handle single metric case
        if len(metrics) == 1:
            axes = [axes]
        
        # Iterate through metrics
        for i, metric in enumerate(metrics):
            # Sort results by metric
            sorted_results = sorted(all_results, key=lambda x: x['performance'].get(metric, 0), reverse=True)
            
            # Extract top N results
            top_results = sorted_results[:top_n]
            
            # Extract values
            metric_values = [result['performance'].get(metric, 0) for result in top_results]
            
            # Create labels from parameters
            labels = []
            for result in top_results:
                params = result['parameters']
                label = ", ".join([f"{k}={v}" for k, v in params.items()])
                if len(label) > 30:
                    label = label[:27] + "..."
                labels.append(label)
            
            # Plot
            ax = axes[i]
            bars = ax.barh(range(len(metric_values)), metric_values)
            
            # Add values
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01 * max(metric_values), bar.get_y() + bar.get_height()/2, 
                       f'{width:.2f}', ha='left', va='center')
            
            # Add labels
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'Top {top_n} Results by {metric.replace("_", " ").title()}')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='x')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Create singleton instances for easy access
backtester = Backtester()
optimizer = Optimizer()

def backtest_strategy(strategy, data, initial_capital=10000.0):
    """
    Backtest a strategy on historical data using the default backtester.
    
    Parameters:
    -----------
    strategy : Strategy
        Strategy to backtest
    data : pandas.DataFrame
        Market data
    initial_capital : float, optional
        Initial capital for backtesting
        
    Returns:
    --------
    BacktestResult
        Backtest result
    """
    return backtester.backtest_strategy(strategy, data, initial_capital)

def walk_forward_analysis(strategy_class, data, train_size=0.7, window_size=None, step_size=None, 
                         initial_capital=10000.0, parameter_grid=None, metric='sharpe_ratio'):
    """
    Perform walk-forward analysis using the default backtester.
    
    Parameters:
    -----------
    strategy_class : class
        Strategy class to use
    data : pandas.DataFrame
        Market data
    train_size : float, optional
        Proportion of data to use for training
    window_size : int, optional
        Size of the rolling window in days
    step_size : int, optional
        Step size for the rolling window in days
    initial_capital : float, optional
        Initial capital for backtesting
    parameter_grid : dict, optional
        Dictionary of parameter names and lists of values to try
    metric : str, optional
        Performance metric to optimize for
        
    Returns:
    --------
    dict
        Walk-forward analysis results
    """
    return backtester.walk_forward_analysis(strategy_class, data, train_size, window_size, step_size, 
                                          initial_capital, parameter_grid, metric)

def monte_carlo_simulation(strategy, data, initial_capital=10000.0, num_simulations=1000, 
                          confidence_level=0.95):
    """
    Perform Monte Carlo simulation using the default backtester.
    
    Parameters:
    -----------
    strategy : Strategy
        Strategy to simulate
    data : pandas.DataFrame
        Market data
    initial_capital : float, optional
        Initial capital for backtesting
    num_simulations : int, optional
        Number of simulations to run
    confidence_level : float, optional
        Confidence level for results
        
    Returns:
    --------
    dict
        Monte Carlo simulation results
    """
    return backtester.monte_carlo_simulation(strategy, data, initial_capital, num_simulations, 
                                           confidence_level)

def grid_search(strategy_class, data, parameter_grid, initial_capital=10000.0, 
               metric='sharpe_ratio', n_jobs=None):
    """
    Perform grid search optimization using the default optimizer.
    
    Parameters:
    -----------
    strategy_class : class
        Strategy class to use
    data : pandas.DataFrame
        Market data
    parameter_grid : dict
        Dictionary of parameter names and lists of values to try
    initial_capital : float, optional
        Initial capital for backtesting
    metric : str, optional
        Performance metric to optimize for
    n_jobs : int, optional
        Number of parallel jobs to run
        
    Returns:
    --------
    dict
        Optimization results
    """
    return optimizer.grid_search(strategy_class, data, parameter_grid, initial_capital, metric, n_jobs)

def random_search(strategy_class, data, parameter_space, num_iterations=100, 
                 initial_capital=10000.0, metric='sharpe_ratio', n_jobs=None):
    """
    Perform random search optimization using the default optimizer.
    
    Parameters:
    -----------
    strategy_class : class
        Strategy class to use
    data : pandas.DataFrame
        Market data
    parameter_space : dict
        Dictionary of parameter names and (min, max) ranges or lists of values
    num_iterations : int, optional
        Number of random combinations to try
    initial_capital : float, optional
        Initial capital for backtesting
    metric : str, optional
        Performance metric to optimize for
    n_jobs : int, optional
        Number of parallel jobs to run
        
    Returns:
    --------
    dict
        Optimization results
    """
    return optimizer.random_search(strategy_class, data, parameter_space, num_iterations, 
                                 initial_capital, metric, n_jobs)
