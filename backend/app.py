"""
Main Application for Gemma Advanced Trading System.

This module integrates all components into a cohesive application.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

# Import system components
import data_access
from strategy_builder import (
    Strategy, SwingTradingStrategy, DayTradingStrategy,
    create_strategy, generate_strategy_for_ticker, optimize_strategy_for_ticker
)
from swing_trading import (
    EnhancedSwingTradingStrategy, generate_swing_strategy
)
from backtesting import (
    backtest_strategy, walk_forward_analysis, monte_carlo_simulation,
    grid_search, random_search
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gemma_trading.log')
    ]
)

logger = logging.getLogger("GemmaTrading.Main")

class GemmaAdvancedTradingSystem:
    """
    Main application class for Gemma Advanced Trading System.
    """
    
    def __init__(self):
        """Initialize the trading system."""
        self.logger = logging.getLogger("GemmaTrading.Main")
        self.logger.info("Initializing Gemma Advanced Trading System")
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_strategy(self, ticker, strategy_type="swing", period="1y", interval="1d", 
                         parameters=None, optimize=False, enhanced=True, output_dir=None):
        """
        Generate a trading strategy for the specified ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to generate a strategy for
        strategy_type : str, optional
            Type of strategy to generate ("swing", "day", "trend", "breakout", etc.)
        period : str, optional
            Period to fetch data for (e.g., "1d", "1mo", "1y")
        interval : str, optional
            Data interval (e.g., "1m", "5m", "1h", "1d")
        parameters : dict, optional
            Strategy parameters
        optimize : bool, optional
            Whether to optimize the strategy parameters
        enhanced : bool, optional
            Whether to use enhanced strategy implementations
        output_dir : str, optional
            Directory to save output files
            
        Returns:
        --------
        dict
            Strategy definition and analysis
        """
        self.logger.info(f"Generating {strategy_type} strategy for {ticker}")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir
        
        # Create strategy based on type
        if strategy_type.lower() == "swing":
            if enhanced:
                # Use enhanced swing trading strategy
                strategy_result = generate_swing_strategy(
                    ticker=ticker,
                    period=period,
                    interval=interval,
                    parameters=parameters,
                    optimize=optimize
                )
            else:
                # Use basic swing trading strategy
                strategy_result = generate_strategy_for_ticker(
                    ticker=ticker,
                    strategy_type=strategy_type,
                    period=period,
                    interval=interval,
                    parameters=parameters
                )
                
                # Optimize if requested
                if optimize:
                    strategy_result = optimize_strategy_for_ticker(
                        ticker=ticker,
                        strategy_type=strategy_type,
                        period=period,
                        interval=interval
                    )
        elif strategy_type.lower() == "day":
            # Use day trading strategy
            strategy_result = generate_strategy_for_ticker(
                ticker=ticker,
                strategy_type=strategy_type,
                period=period,
                interval=interval,
                parameters=parameters
            )
            
            # Optimize if requested
            if optimize:
                strategy_result = optimize_strategy_for_ticker(
                    ticker=ticker,
                    strategy_type=strategy_type,
                    period=period,
                    interval=interval
                )
        else:
            raise ValueError(f"Strategy type '{strategy_type}' not supported")
        
        # Save strategy result
        self._save_strategy_result(strategy_result, ticker, strategy_type, output_dir)
        
        return strategy_result
    
    def backtest_strategy(self, strategy_result, output_dir=None):
        """
        Backtest a strategy and generate detailed reports.
        
        Parameters:
        -----------
        strategy_result : dict
            Strategy result from generate_strategy
        output_dir : str, optional
            Directory to save output files
            
        Returns:
        --------
        dict
            Backtest results
        """
        self.logger.info(f"Backtesting strategy for {strategy_result['ticker']}")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir
        
        # Create backtest result
        from backtesting import BacktestResult
        
        backtest_result = BacktestResult(
            strategy_name=strategy_result.get('strategy_type', 'Unknown'),
            ticker=strategy_result['ticker'],
            parameters=strategy_result['parameters'],
            signals=strategy_result['signals'],
            trades=strategy_result['trades'],
            performance=strategy_result['performance']
        )
        
        # Save backtest report
        report_dir = backtest_result.save_report(output_dir)
        
        return {
            'backtest_result': backtest_result,
            'report_dir': report_dir
        }
    
    def run_monte_carlo_simulation(self, strategy_result, num_simulations=1000, 
                                  confidence_level=0.95, output_dir=None):
        """
        Run Monte Carlo simulation for a strategy.
        
        Parameters:
        -----------
        strategy_result : dict
            Strategy result from generate_strategy
        num_simulations : int, optional
            Number of simulations to run
        confidence_level : float, optional
            Confidence level for results
        output_dir : str, optional
            Directory to save output files
            
        Returns:
        --------
        dict
            Monte Carlo simulation results
        """
        self.logger.info(f"Running Monte Carlo simulation for {strategy_result['ticker']}")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir
        
        # Create strategy object
        if strategy_result.get('strategy_type', '').lower() == 'enhanced_swing':
            strategy = EnhancedSwingTradingStrategy(
                name=f"{strategy_result['ticker']}_enhanced_swing"
            )
        elif strategy_result.get('strategy_type', '').lower() == 'swing':
            strategy = SwingTradingStrategy(
                name=f"{strategy_result['ticker']}_swing"
            )
        elif strategy_result.get('strategy_type', '').lower() == 'day':
            strategy = DayTradingStrategy(
                name=f"{strategy_result['ticker']}_day"
            )
        else:
            raise ValueError(f"Strategy type '{strategy_result.get('strategy_type', 'Unknown')}' not supported")
        
        # Set parameters
        strategy.set_parameters(strategy_result['parameters'])
        
        # Get data
        data = strategy_result['signals']
        
        # Run Monte Carlo simulation
        mc_results = monte_carlo_simulation(
            strategy=strategy,
            data=data,
            num_simulations=num_simulations,
            confidence_level=confidence_level
        )
        
        # Save results
        self._save_monte_carlo_results(mc_results, strategy_result['ticker'], output_dir)
        
        return mc_results
    
    def _save_strategy_result(self, strategy_result, ticker, strategy_type, output_dir):
        """
        Save strategy result to files.
        
        Parameters:
        -----------
        strategy_result : dict
            Strategy result
        ticker : str
            Ticker symbol
        strategy_type : str
            Strategy type
        output_dir : str
            Output directory
        """
        # Create directory
        strategy_dir = os.path.join(output_dir, f"{ticker}_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Save parameters
        with open(os.path.join(strategy_dir, 'parameters.json'), 'w') as f:
            json.dump(strategy_result['parameters'], f, indent=4, default=str)
        
        # Save performance
        with open(os.path.join(strategy_dir, 'performance.json'), 'w') as f:
            json.dump(strategy_result['performance'], f, indent=4, default=str)
        
        # Save signals
        if 'signals' in strategy_result and strategy_result['signals'] is not None:
            strategy_result['signals'].to_csv(os.path.join(strategy_dir, 'signals.csv'))
        
        # Save trades
        if 'trades' in strategy_result and strategy_result['trades']:
            trades_df = pd.DataFrame([
                {k: v for k, v in trade.items() if k != 'entry_trade_index'} 
                for trade in strategy_result['trades']
            ])
            trades_df.to_csv(os.path.join(strategy_dir, 'trades.csv'), index=False)
        
        # Save description
        if 'description' in strategy_result and strategy_result['description']:
            with open(os.path.join(strategy_dir, 'description.txt'), 'w') as f:
                f.write(strategy_result['description'])
        
        # Generate and save plots
        self._generate_strategy_plots(strategy_result, strategy_dir)
        
        self.logger.info(f"Strategy results saved to {strategy_dir}")
    
    def _generate_strategy_plots(self, strategy_result, output_dir):
        """
        Generate and save strategy plots.
        
        Parameters:
        -----------
        strategy_result : dict
            Strategy result
        output_dir : str
            Output directory
        """
        # Check if signals are available
        if 'signals' not in strategy_result or strategy_result['signals'] is None:
            return
        
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
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'price_signals.png'), dpi=300, bbox_inches='tight')
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
                
                # Plot price in first subplot
                axes[0].plot(signals.index, signals['Close'], label='Close Price')
                axes[0].set_title(f'Price for {ticker}')
                axes[0].set_ylabel('Price')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                
                plot_idx = 1
                
                # Plot trend indicators
                if trend_indicators:
                    ax = axes[plot_idx]
                    for col in trend_indicators:
                        ax.plot(signals.index, signals[col], label=col)
                    ax.set_title('Trend Indicators')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plot_idx += 1
                
                # Plot momentum indicators
                if momentum_indicators:
                    ax = axes[plot_idx]
                    for col in momentum_indicators:
                        ax.plot(signals.index, signals[col], label=col)
                    ax.set_title('Momentum Indicators')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plot_idx += 1
                
                # Plot volatility indicators
                if volatility_indicators:
                    ax = axes[plot_idx]
                    for col in volatility_indicators:
                        ax.plot(signals.index, signals[col], label=col)
                    ax.set_title('Volatility Indicators')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                
                # Set x-label for bottom subplot
                axes[-1].set_xlabel('Date')
                
                # Adjust layout
                plt.tight_layout()
                
                # Save plot
                plt.savefig(os.path.join(output_dir, 'indicators.png'), dpi=300, bbox_inches='tight')
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
                
                # Save plot
                plt.savefig(os.path.join(output_dir, 'equity_curve.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    def _save_monte_carlo_results(self, mc_results, ticker, output_dir):
        """
        Save Monte Carlo simulation results to files.
        
        Parameters:
        -----------
        mc_results : dict
            Monte Carlo simulation results
        ticker : str
            Ticker symbol
        output_dir : str
            Output directory
        """
        # Create directory
        mc_dir = os.path.join(output_dir, f"{ticker}_monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(mc_dir, exist_ok=True)
        
        # Save results
        with open(os.path.join(mc_dir, 'monte_carlo_results.json'), 'w') as f:
            # Extract key metrics
            results = {
                'num_simulations': mc_results['num_simulations'],
                'confidence_level': mc_results['confidence_level'],
                'mean_return': mc_results['mean_return'],
                'median_return': mc_results['median_return'],
                'mean_drawdown': mc_results['mean_drawdown'],
                'median_drawdown': mc_results['median_drawdown'],
                'mean_final_equity': mc_results['mean_final_equity'],
                'median_final_equity': mc_results['median_final_equity'],
                'return_ci': mc_results['return_ci'],
                'drawdown_ci': mc_results['drawdown_ci'],
                'final_equity_ci': mc_results['final_equity_ci'],
                'worst_case_return': mc_results['worst_case_return'],
                'best_case_return': mc_results['best_case_return'],
                'worst_case_drawdown': mc_results['worst_case_drawdown']
            }
            json.dump(results, f, indent=4, default=str)
        
        # Generate and save plots
        self._generate_monte_carlo_plots(mc_results, ticker, mc_dir)
        
        self.logger.info(f"Monte Carlo simulation results saved to {mc_dir}")
    
    def _generate_monte_carlo_plots(self, mc_results, ticker, output_dir):
        """
        Generate and save Monte Carlo simulation plots.
        
        Parameters:
        -----------
        mc_results : dict
            Monte Carlo simulation results
        ticker : str
            Ticker symbol
        output_dir : str
            Output directory
        """
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
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'return_distribution.png'), dpi=300, bbox_inches='tight')
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
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'drawdown_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Plot equity curves for a sample of simulations
        num_samples = min(100, mc_results['num_simulations'])
        sample_indices = np.random.choice(mc_results['num_simulations'], num_samples, replace=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot sample equity curves
        for idx in sample_indices:
            equity_curve = mc_results['simulation_results'][idx]['equity_curve']
            ax.plot(equity_curve, alpha=0.1, color='blue')
        
        # Plot original equity curve if available
        if 'original_performance' in mc_results and 'equity_curve' in mc_results['original_performance']:
            ax.plot(mc_results['original_performance']['equity_curve'], color='red', linewidth=2, label='Original')
        
        # Add labels and title
        ax.set_title(f'Equity Curves for {ticker} ({num_samples} Sample Simulations)')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Equity ($)')
        
        # Add legend if original equity curve is plotted
        if 'original_performance' in mc_results and 'equity_curve' in mc_results['original_performance']:
            ax.legend()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'equity_curves.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Gemma Advanced Trading System')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate a trading strategy')
    generate_parser.add_argument('ticker', help='Ticker symbol')
    generate_parser.add_argument('--strategy-type', choices=['swing', 'day'], default='swing',
                               help='Type of strategy to generate')
    generate_parser.add_argument('--period', default='1y',
                               help='Period to fetch data for (e.g., "1d", "1mo", "1y")')
    generate_parser.add_argument('--interval', default='1d',
                               help='Data interval (e.g., "1m", "5m", "1h", "1d")')
    generate_parser.add_argument('--optimize', action='store_true',
                               help='Optimize strategy parameters')
    generate_parser.add_argument('--enhanced', action='store_true',
                               help='Use enhanced strategy implementations')
    generate_parser.add_argument('--output-dir', help='Directory to save output files')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest a trading strategy')
    backtest_parser.add_argument('ticker', help='Ticker symbol')
    backtest_parser.add_argument('--strategy-type', choices=['swing', 'day'], default='swing',
                               help='Type of strategy to backtest')
    backtest_parser.add_argument('--period', default='1y',
                               help='Period to fetch data for (e.g., "1d", "1mo", "1y")')
    backtest_parser.add_argument('--interval', default='1d',
                               help='Data interval (e.g., "1m", "5m", "1h", "1d")')
    backtest_parser.add_argument('--optimize', action='store_true',
                               help='Optimize strategy parameters')
    backtest_parser.add_argument('--enhanced', action='store_true',
                               help='Use enhanced strategy implementations')
    backtest_parser.add_argument('--output-dir', help='Directory to save output files')
    
    # Monte Carlo command
    mc_parser = subparsers.add_parser('monte-carlo', help='Run Monte Carlo simulation')
    mc_parser.add_argument('ticker', help='Ticker symbol')
    mc_parser.add_argument('--strategy-type', choices=['swing', 'day'], default='swing',
                         help='Type of strategy to simulate')
    mc_parser.add_argument('--period', default='1y',
                         help='Period to fetch data for (e.g., "1d", "1mo", "1y")')
    mc_parser.add_argument('--interval', default='1d',
                         help='Data interval (e.g., "1m", "5m", "1h", "1d")')
    mc_parser.add_argument('--optimize', action='store_true',
                         help='Optimize strategy parameters')
    mc_parser.add_argument('--enhanced', action='store_true',
                         help='Use enhanced strategy implementations')
    mc_parser.add_argument('--num-simulations', type=int, default=1000,
                         help='Number of simulations to run')
    mc_parser.add_argument('--confidence-level', type=float, default=0.95,
                         help='Confidence level for results')
    mc_parser.add_argument('--output-dir', help='Directory to save output files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create trading system
    trading_system = GemmaAdvancedTradingSystem()
    
    # Execute command
    if args.command == 'generate':
        strategy_result = trading_system.generate_strategy(
            ticker=args.ticker,
            strategy_type=args.strategy_type,
            period=args.period,
            interval=args.interval,
            optimize=args.optimize,
            enhanced=args.enhanced,
            output_dir=args.output_dir
        )
        
        # Print summary
        print(f"\nStrategy generated for {args.ticker}:")
        print(f"Strategy type: {strategy_result.get('strategy_type', args.strategy_type)}")
        print(f"Performance:")
        for key, value in strategy_result['performance'].items():
            if isinstance(value, (int, float)):
                if key in ['win_rate']:
                    print(f"  {key.replace('_', ' ').title()}: {value * 100:.2f}%")
                elif key in ['total_return', 'max_drawdown']:
                    print(f"  {key.replace('_', ' ').title()}: {value:.2f}%")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    elif args.command == 'backtest':
        # Generate strategy
        strategy_result = trading_system.generate_strategy(
            ticker=args.ticker,
            strategy_type=args.strategy_type,
            period=args.period,
            interval=args.interval,
            optimize=args.optimize,
            enhanced=args.enhanced,
            output_dir=args.output_dir
        )
        
        # Backtest strategy
        backtest_result = trading_system.backtest_strategy(
            strategy_result=strategy_result,
            output_dir=args.output_dir
        )
        
        # Print summary
        print(f"\nBacktest completed for {args.ticker}:")
        print(f"Strategy type: {strategy_result.get('strategy_type', args.strategy_type)}")
        print(f"Report saved to: {backtest_result['report_dir']}")
    
    elif args.command == 'monte-carlo':
        # Generate strategy
        strategy_result = trading_system.generate_strategy(
            ticker=args.ticker,
            strategy_type=args.strategy_type,
            period=args.period,
            interval=args.interval,
            optimize=args.optimize,
            enhanced=args.enhanced,
            output_dir=args.output_dir
        )
        
        # Run Monte Carlo simulation
        mc_results = trading_system.run_monte_carlo_simulation(
            strategy_result=strategy_result,
            num_simulations=args.num_simulations,
            confidence_level=args.confidence_level,
            output_dir=args.output_dir
        )
        
        # Print summary
        print(f"\nMonte Carlo simulation completed for {args.ticker}:")
        print(f"Strategy type: {strategy_result.get('strategy_type', args.strategy_type)}")
        print(f"Number of simulations: {mc_results['num_simulations']}")
        print(f"Confidence level: {mc_results['confidence_level']}")
        print(f"Mean return: {mc_results['mean_return']:.2f}%")
        print(f"Return {mc_results['confidence_level']*100:.0f}% confidence interval: "
              f"({mc_results['return_ci'][0]:.2f}%, {mc_results['return_ci'][1]:.2f}%)")
        print(f"Mean max drawdown: {mc_results['mean_drawdown']:.2f}%")
        print(f"Max drawdown {mc_results['confidence_level']*100:.0f}% confidence interval: "
              f"({mc_results['drawdown_ci'][0]:.2f}%, {mc_results['drawdown_ci'][1]:.2f}%)")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
