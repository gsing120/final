"""
Simplified strategy optimization for Gemma Advanced Trading System.

This script implements a simpler approach to strategy optimization that focuses on
core functionality while ensuring proper handling of pandas Series objects.
"""

import os
import sys
import json
import logging
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Tuple
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/ubuntu/gemma_advanced/simplified_optimization.log')
    ]
)

logger = logging.getLogger("GemmaTrading.SimplifiedOptimization")

class PerformanceThresholds:
    """
    Defines performance thresholds for validating trading strategies.
    """
    
    def __init__(self, min_total_return: float = 0.0,
               min_sharpe_ratio: float = 0.3,
               max_drawdown: float = -25.0,
               min_win_rate: float = 50.0):
        """
        Initialize performance thresholds.
        
        Parameters:
        -----------
        min_total_return : float, optional
            Minimum acceptable total return. Default is 0.0 (positive return).
        min_sharpe_ratio : float, optional
            Minimum acceptable Sharpe ratio. Default is 0.3.
        max_drawdown : float, optional
            Maximum acceptable drawdown. Default is -25.0%.
        min_win_rate : float, optional
            Minimum acceptable win rate. Default is 50.0%.
        """
        self.min_total_return = min_total_return
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate
        
        logger.info(f"Initialized PerformanceThresholds with min_total_return={min_total_return}, "
                  f"min_sharpe_ratio={min_sharpe_ratio}, max_drawdown={max_drawdown}, "
                  f"min_win_rate={min_win_rate}")
    
    def is_strategy_valid(self, performance: Dict[str, Any]) -> tuple:
        """
        Check if a strategy meets performance thresholds.
        
        Parameters:
        -----------
        performance : Dict[str, Any]
            Performance metrics of the strategy.
            
        Returns:
        --------
        tuple
            (is_valid, validation_results)
            is_valid: True if strategy meets all thresholds, False otherwise.
            validation_results: Dict with validation result for each metric.
        """
        validation_results = {}
        
        # Extract performance metrics
        total_return = self._extract_numeric_value(performance.get('total_return', 0))
        sharpe_ratio = self._extract_numeric_value(performance.get('sharpe_ratio', 0))
        max_drawdown = self._extract_numeric_value(performance.get('max_drawdown', 0))
        win_rate = self._extract_numeric_value(performance.get('win_rate', 0))
        
        # Validate each metric
        validation_results['total_return'] = total_return >= self.min_total_return
        validation_results['sharpe_ratio'] = sharpe_ratio >= self.min_sharpe_ratio
        validation_results['max_drawdown'] = max_drawdown >= self.max_drawdown
        validation_results['win_rate'] = win_rate >= self.min_win_rate
        
        # Strategy is valid if all metrics meet thresholds
        is_valid = all(validation_results.values())
        
        return is_valid, validation_results
    
    def _extract_numeric_value(self, value: Any) -> float:
        """
        Extract numeric value from various formats.
        
        Parameters:
        -----------
        value : Any
            Value to extract numeric value from.
            
        Returns:
        --------
        float
            Extracted numeric value.
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Remove percentage sign and convert to float
            return float(value.replace('%', ''))
        else:
            return 0.0


class StrategyBacktester:
    """
    Backtests trading strategies to evaluate their performance.
    """
    
    def __init__(self):
        """
        Initialize the StrategyBacktester.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyBacktester")
        self.logger.info("Initialized StrategyBacktester")
    
    def backtest_strategy(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Backtest a strategy on historical data.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to backtest.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Backtest result with performance metrics.
        """
        self.logger.info(f"Backtesting strategy for {ticker}")
        
        try:
            # Get historical data
            data = self._get_historical_data(ticker)
            
            if data is None or len(data) == 0:
                self.logger.error(f"Failed to get historical data for {ticker}")
                return {
                    "success": False,
                    "error": f"Failed to get historical data for {ticker}"
                }
            
            # Apply strategy to historical data
            signals = self._generate_signals(data, strategy)
            
            # Calculate performance metrics
            performance = self._calculate_performance(data, signals)
            
            self.logger.info(f"Backtest completed for {ticker}")
            
            return {
                "success": True,
                "performance": performance,
                "signals": signals.to_dict() if isinstance(signals, pd.DataFrame) else signals
            }
        
        except Exception as e:
            self.logger.error(f"Error backtesting strategy for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_historical_data(self, ticker: str) -> pd.DataFrame:
        """
        Get historical data for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        pd.DataFrame
            Historical data.
        """
        self.logger.info(f"Getting historical data for {ticker}")
        
        try:
            # Get data from Yahoo Finance
            data = yf.download(ticker, period="1y")
            
            if len(data) == 0:
                self.logger.error(f"No data found for {ticker}")
                return None
            
            self.logger.info(f"Got {len(data)} data points for {ticker}")
            
            return data
        
        except Exception as e:
            self.logger.error(f"Error getting historical data for {ticker}: {e}")
            return None
    
    def _generate_signals(self, data: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate trading signals based on strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical data.
        strategy : Dict[str, Any]
            Strategy to apply.
            
        Returns:
        --------
        pd.DataFrame
            Trading signals.
        """
        self.logger.info("Generating trading signals")
        
        # Create a copy of the data
        signals = data.copy()
        
        # Initialize signal column
        signals['signal'] = 0
        
        # Get strategy parameters
        short_ma = strategy.get('short_ma', 20)
        long_ma = strategy.get('long_ma', 50)
        
        # Calculate moving averages
        signals['short_ma'] = signals['Close'].rolling(window=short_ma).mean()
        signals['long_ma'] = signals['Close'].rolling(window=long_ma).mean()
        
        # Generate signals based on moving average crossover
        for i in range(1, len(signals)):
            if signals['short_ma'].iloc[i] > signals['long_ma'].iloc[i]:
                signals.loc[signals.index[i], 'signal'] = 1
            elif signals['short_ma'].iloc[i] < signals['long_ma'].iloc[i]:
                signals.loc[signals.index[i], 'signal'] = -1
        
        # Generate positions (signal changes)
        signals['position'] = signals['signal'].diff()
        
        self.logger.info("Generated trading signals")
        
        return signals
    
    def _calculate_performance(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical data.
        signals : pd.DataFrame
            Trading signals.
            
        Returns:
        --------
        Dict[str, Any]
            Performance metrics.
        """
        self.logger.info("Calculating performance metrics")
        
        # Calculate returns
        signals['returns'] = data['Close'].pct_change()
        signals['strategy_returns'] = signals['signal'].shift(1) * signals['returns']
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['returns']).cumprod()
        signals['cumulative_strategy_returns'] = (1 + signals['strategy_returns']).cumprod()
        
        # Calculate total return
        total_return = signals['cumulative_strategy_returns'].iloc[-1] - 1
        
        # Calculate Sharpe ratio
        sharpe_ratio = signals['strategy_returns'].mean() / signals['strategy_returns'].std() * np.sqrt(252)
        
        # Calculate maximum drawdown
        cumulative_max = signals['cumulative_strategy_returns'].cummax()
        drawdown = (signals['cumulative_strategy_returns'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100
        
        # Calculate win rate
        winning_days = signals[signals['strategy_returns'] > 0]
        total_days_with_returns = len(signals[signals['strategy_returns'] != 0])
        win_rate = (len(winning_days) / total_days_with_returns * 100) if total_days_with_returns > 0 else 0
        
        # Calculate volatility
        volatility = signals['strategy_returns'].std() * np.sqrt(252) * 100
        
        # Prepare performance metrics
        performance = {
            "total_return": total_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "volatility": volatility,
            "num_trades": len(signals[signals['position'] != 0]),
            "start_date": data.index[0].strftime("%Y-%m-%d"),
            "end_date": data.index[-1].strftime("%Y-%m-%d")
        }
        
        self.logger.info(f"Calculated performance metrics: {performance}")
        
        return performance


class StrategyOptimizer:
    """
    Optimizes trading strategies to improve performance.
    """
    
    def __init__(self, backtester: StrategyBacktester = None,
               performance_thresholds: PerformanceThresholds = None,
               max_iterations: int = 10):
        """
        Initialize the StrategyOptimizer.
        
        Parameters:
        -----------
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        max_iterations : int, optional
            Maximum number of optimization iterations. Default is 10.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyOptimizer")
        
        # Create or use provided components
        self.backtester = backtester or StrategyBacktester()
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        
        # Configuration
        self.max_iterations = max_iterations
        
        self.logger.info(f"Initialized StrategyOptimizer with max_iterations={max_iterations}")
    
    def optimize_strategy(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Optimize a strategy to improve performance.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to optimize.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Optimized strategy with improved performance.
        """
        self.logger.info(f"Optimizing strategy for {ticker}")
        
        # Initialize optimization variables
        current_strategy = strategy.copy()
        best_strategy = strategy.copy()
        best_performance = None
        optimization_history = []
        
        # Backtest the initial strategy
        initial_backtest = self.backtester.backtest_strategy(current_strategy, ticker)
        
        if not initial_backtest["success"]:
            self.logger.error(f"Failed to backtest initial strategy: {initial_backtest.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": "Failed to backtest initial strategy",
                "original_strategy": strategy
            }
        
        # Get initial performance
        initial_performance = initial_backtest["performance"]
        best_performance = initial_performance
        
        # Check if initial strategy meets performance thresholds
        is_valid, validation_results = self.performance_thresholds.is_strategy_valid(initial_performance)
        
        if is_valid:
            self.logger.info("Initial strategy already meets performance thresholds")
            return {
                "success": True,
                "strategy": current_strategy,
                "performance": initial_performance,
                "validation_results": validation_results,
                "optimization_history": [],
                "message": "Initial strategy already meets performance thresholds"
            }
        
        # Record initial performance in optimization history
        optimization_history.append({
            "iteration": 0,
            "strategy": current_strategy,
            "performance": initial_performance,
            "validation_results": validation_results
        })
        
        # Optimize the strategy iteratively
        for iteration in range(self.max_iterations):
            self.logger.info(f"Optimization iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations()
            
            # Test each parameter combination
            for params in param_combinations:
                # Update strategy with parameters
                test_strategy = current_strategy.copy()
                test_strategy.update(params)
                
                # Backtest the test strategy
                backtest_result = self.backtester.backtest_strategy(test_strategy, ticker)
                
                if not backtest_result["success"]:
                    self.logger.warning(f"Failed to backtest test strategy with params {params}")
                    continue
                
                # Get test performance
                test_performance = backtest_result["performance"]
                
                # Check if test strategy is better than current best
                if self._is_better_performance(test_performance, best_performance):
                    best_strategy = test_strategy
                    best_performance = test_performance
                    self.logger.info(f"Found better strategy with total return: {best_performance['total_return']}")
            
            # Update current strategy to best found in this iteration
            current_strategy = best_strategy
            
            # Check if best strategy meets performance thresholds
            is_valid, validation_results = self.performance_thresholds.is_strategy_valid(best_performance)
            
            # Record optimization in history
            optimization_history.append({
                "iteration": iteration + 1,
                "strategy": best_strategy,
                "performance": best_performance,
                "validation_results": validation_results
            })
            
            # If strategy meets performance thresholds, we're done
            if is_valid:
                self.logger.info(f"Strategy meets performance thresholds after {iteration + 1} iterations")
                break
        
        # Final validation
        is_valid, validation_results = self.performance_thresholds.is_strategy_valid(best_performance)
        
        # Prepare result
        result = {
            "success": is_valid,
            "strategy": best_strategy,
            "performance": best_performance,
            "validation_results": validation_results,
            "optimization_history": optimization_history,
            "original_strategy": strategy,
            "message": f"Strategy {'meets' if is_valid else 'does not meet'} performance thresholds after {len(optimization_history) - 1} optimization iterations"
        }
        
        self.logger.info(f"Completed strategy optimization for {ticker} with success: {is_valid}")
        
        return result
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for optimization.
        
        Returns:
        --------
        List[Dict[str, Any]]
            Parameter combinations.
        """
        # Define parameter ranges
        short_ma_range = [5, 10, 15, 20, 25, 30]
        long_ma_range = [30, 40, 50, 60, 70, 80, 90, 100]
        
        # Generate combinations
        combinations = []
        
        for short_ma in short_ma_range:
            for long_ma in long_ma_range:
                if short_ma < long_ma:  # Ensure short MA is less than long MA
                    combinations.append({
                        'short_ma': short_ma,
                        'long_ma': long_ma
                    })
        
        return combinations
    
    def _is_better_performance(self, performance1: Dict[str, Any], performance2: Dict[str, Any]) -> bool:
        """
        Check if performance1 is better than performance2.
        
        Parameters:
        -----------
        performance1 : Dict[str, Any]
            First performance metrics.
        performance2 : Dict[str, Any]
            Second performance metrics.
            
        Returns:
        --------
        bool
            True if performance1 is better than performance2, False otherwise.
        """
        if performance2 is None:
            return True
        
        # Extract total return
        return1 = performance1.get("total_return", 0)
        return2 = performance2.get("total_return", 0)
        
        # Compare total return
        if return1 > return2:
            return True
        
        # If returns are equal, compare Sharpe ratio
        if return1 == return2:
            sharpe1 = performance1.get("sharpe_ratio", 0)
            sharpe2 = performance2.get("sharpe_ratio", 0)
            
            return sharpe1 > sharpe2
        
        return False


def generate_optimized_strategy(ticker: str = 'AAPL') -> Dict[str, Any]:
    """
    Generate an optimized trading strategy for a ticker.
    
    Parameters:
    -----------
    ticker : str, optional
        Ticker symbol. Default is 'AAPL'.
        
    Returns:
    --------
    Dict[str, Any]
        Optimized strategy.
    """
    logger.info(f"Generating optimized strategy for {ticker}")
    
    # Create components
    backtester = StrategyBacktester()
    performance_thresholds = PerformanceThresholds()
    optimizer = StrategyOptimizer(
        backtester=backtester,
        performance_thresholds=performance_thresholds,
        max_iterations=10
    )
    
    # Initial strategy
    initial_strategy = {
        "name": f"Optimized {ticker} Strategy",
        "type": "trend_following",
        "short_ma": 20,
        "long_ma": 50
    }
    
    # Optimize strategy
    optimization_result = optimizer.optimize_strategy(initial_strategy, ticker)
    
    if optimization_result["success"]:
        logger.info(f"Successfully generated optimized strategy for {ticker}")
        
        # Save the optimized strategy
        with open(f'/home/ubuntu/gemma_advanced/{ticker.lower()}_optimized_strategy.json', 'w') as f:
            json.dump(optimization_result, f, indent=2)
        
        logger.info(f"Saved optimized strategy to {ticker.lower()}_optimized_strategy.json")
        
        return optimization_result
    else:
        logger.warning(f"Failed to generate optimized strategy for {ticker}")
        
        # Save the best strategy found
        with open(f'/home/ubuntu/gemma_advanced/{ticker.lower()}_best_strategy.json', 'w') as f:
            json.dump(optimization_result, f, indent=2)
        
        logger.info(f"Saved best strategy to {ticker.lower()}_best_strategy.json")
        
        return optimization_result


if __name__ == "__main__":
    logger.info("Starting simplified strategy optimization")
    
    result = generate_optimized_strategy('AAPL')
    
    if result["success"]:
        logger.info("Successfully generated optimized strategy")
        print(f"SUCCESS: Generated optimized strategy with {result['performance']['total_return']}% return and {result['performance']['sharpe_ratio']} Sharpe ratio")
    else:
        logger.warning("Failed to generate fully optimized strategy, but found best possible strategy")
        print(f"PARTIAL SUCCESS: Generated best possible strategy with {result['performance']['total_return']}% return and {result['performance']['sharpe_ratio']} Sharpe ratio")
        print(f"Validation results: {result['validation_results']}")
