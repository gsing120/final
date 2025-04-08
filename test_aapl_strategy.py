"""
Test script for the improved Gemma Advanced Trading System with AAPL data.

This script tests the strategy optimization improvements to ensure only strategies
with positive historical performance are presented to users.
"""

import os
import sys
import json
import logging
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/ubuntu/gemma_advanced/test_aapl_strategy.log')
    ]
)

logger = logging.getLogger("GemmaTrading.TestAAPLStrategy")

class PerformanceThresholds:
    """
    Defines performance thresholds for validating trading strategies.
    """
    
    def __init__(self, min_total_return: float = 0.0,
               min_sharpe_ratio: float = 0.5,
               max_drawdown: float = -20.0,
               min_win_rate: float = 50.0):
        """
        Initialize performance thresholds.
        
        Parameters:
        -----------
        min_total_return : float, optional
            Minimum acceptable total return. Default is 0.0 (positive return).
        min_sharpe_ratio : float, optional
            Minimum acceptable Sharpe ratio. Default is 0.5.
        max_drawdown : float, optional
            Maximum acceptable drawdown. Default is -20.0%.
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
        
        # Apply strategy logic
        # For this implementation, we'll use a simple moving average crossover strategy
        # In a real implementation, this would parse the strategy and apply its logic
        
        # Calculate moving averages
        signals['short_ma'] = signals['Close'].rolling(window=20).mean()
        signals['long_ma'] = signals['Close'].rolling(window=50).mean()
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['short_ma'] > signals['long_ma'], 'signal'] = 1
        signals.loc[signals['short_ma'] < signals['long_ma'], 'signal'] = -1
        
        # Generate positions
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
        win_rate = len(winning_days) / len(signals[signals['strategy_returns'] != 0]) * 100
        
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


class StrategyRefinementEngine:
    """
    Automatically refines trading strategies until they meet performance thresholds.
    """
    
    def __init__(self, backtester: StrategyBacktester = None,
               performance_thresholds: PerformanceThresholds = None,
               max_refinement_iterations: int = 5):
        """
        Initialize the StrategyRefinementEngine.
        
        Parameters:
        -----------
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        max_refinement_iterations : int, optional
            Maximum number of refinement iterations. Default is 5.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyRefinementEngine")
        
        # Create or use provided components
        self.backtester = backtester or StrategyBacktester()
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        
        # Configuration
        self.max_refinement_iterations = max_refinement_iterations
        
        self.logger.info("Initialized StrategyRefinementEngine")
    
    def refine_strategy(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Refine a strategy until it meets performance thresholds.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to refine.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Refined strategy with improved performance.
        """
        self.logger.info(f"Refining strategy for {ticker}")
        
        # Initialize refinement variables
        current_strategy = strategy.copy()
        best_strategy = strategy.copy()
        best_performance = None
        refinement_history = []
        
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
                "refinement_history": [],
                "message": "Initial strategy already meets performance thresholds"
            }
        
        # Record initial performance in refinement history
        refinement_history.append({
            "iteration": 0,
            "performance": initial_performance,
            "validation_results": validation_results,
            "changes": None
        })
        
        # Refine the strategy iteratively
        for iteration in range(self.max_refinement_iterations):
            self.logger.info(f"Refinement iteration {iteration + 1}/{self.max_refinement_iterations}")
            
            # Generate refinement plan
            refinement_plan = self._generate_refinement_plan(current_strategy, best_performance, validation_results)
            
            # Apply refinements
            refined_strategy = self._apply_refinements(current_strategy, refinement_plan)
            
            # Backtest the refined strategy
            backtest_result = self.backtester.backtest_strategy(refined_strategy, ticker)
            
            if not backtest_result["success"]:
                self.logger.warning(f"Failed to backtest refined strategy in iteration {iteration + 1}")
                continue
            
            # Get refined performance
            refined_performance = backtest_result["performance"]
            
            # Check if refined strategy meets performance thresholds
            is_valid, validation_results = self.performance_thresholds.is_strategy_valid(refined_performance)
            
            # Record refinement in history
            refinement_history.append({
                "iteration": iteration + 1,
                "performance": refined_performance,
                "validation_results": validation_results,
                "changes": refinement_plan
            })
            
            # Update current strategy
            current_strategy = refined_strategy
            
            # Update best strategy if better than previous best
            if self._is_better_performance(refined_performance, best_performance):
                best_strategy = refined_strategy
                best_performance = refined_performance
                self.logger.info(f"Found better strategy with total return: {best_performance['total_return']}")
            
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
            "refinement_history": refinement_history,
            "original_strategy": strategy,
            "message": f"Strategy {'meets' if is_valid else 'does not meet'} performance thresholds after {len(refinement_history) - 1} refinement iterations"
        }
        
        self.logger.info(f"Completed strategy refinement for {ticker} with success: {is_valid}")
        
        return result
    
    def _generate_refinement_plan(self, strategy: Dict[str, Any], 
                                performance: Dict[str, Any],
                                validation_results: Dict[str, bool]) -> Dict[str, Any]:
        """
        Generate a refinement plan based on performance issues.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to refine.
        performance : Dict[str, Any]
            Performance metrics of the strategy.
        validation_results : Dict[str, bool]
            Validation results for each performance metric.
            
        Returns:
        --------
        Dict[str, Any]
            Refinement plan.
        """
        self.logger.info("Generating refinement plan")
        
        # Identify failing metrics
        failing_metrics = [metric for metric, is_valid in validation_results.items() if not is_valid]
        
        # Create refinement plan
        refinement_plan = {
            "parameter_adjustments": {},
            "indicator_changes": [],
            "signal_generation_changes": [],
            "risk_management_changes": []
        }
        
        # Generate refinement plan based on failing metrics
        for metric in failing_metrics:
            if metric == "total_return":
                refinement_plan["parameter_adjustments"]["short_ma"] = 15  # Adjust short MA period
                refinement_plan["parameter_adjustments"]["long_ma"] = 40   # Adjust long MA period
                refinement_plan["indicator_changes"].append("Add trend strength filter")
            elif metric == "sharpe_ratio":
                refinement_plan["parameter_adjustments"]["volatility_filter"] = "add"
                refinement_plan["indicator_changes"].append("Add volatility filter")
                refinement_plan["risk_management_changes"].append("Implement dynamic position sizing")
            elif metric == "max_drawdown":
                refinement_plan["parameter_adjustments"]["stop_loss"] = 5.0  # 5% stop loss
                refinement_plan["risk_management_changes"].append("Add trailing stops")
                refinement_plan["risk_management_changes"].append("Add drawdown protection rules")
            elif metric == "win_rate":
                refinement_plan["parameter_adjustments"]["confirmation_threshold"] = "stricter"
                refinement_plan["indicator_changes"].append("Add confirmation indicator")
                refinement_plan["signal_generation_changes"].append("Filter weak signals")
        
        self.logger.info(f"Generated refinement plan with {len(refinement_plan['parameter_adjustments'])} parameter adjustments, {len(refinement_plan['indicator_changes'])} indicator changes")
        
        return refinement_plan
    
    def _apply_refinements(self, strategy: Dict[str, Any], 
                         refinement_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply refinements to a strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to refine.
        refinement_plan : Dict[str, Any]
            Refinement plan.
            
        Returns:
        --------
        Dict[str, Any]
            Refined strategy.
        """
        self.logger.info("Applying refinements to strategy")
        
        # Create a copy of the strategy
        refined_strategy = strategy.copy()
        
        # Apply parameter adjustments
        parameter_adjustments = refinement_plan.get("parameter_adjustments", {})
        
        for param, value in parameter_adjustments.items():
            refined_strategy[param] = value
        
        # Apply indicator changes
        indicator_changes = refinement_plan.get("indicator_changes", [])
        
        if "indicators" not in refined_strategy:
            refined_strategy["indicators"] = []
        
        for change in indicator_changes:
            if change == "Add trend strength filter":
                refined_strategy["indicators"].append({
                    "name": "ADX",
                    "parameters": {"period": 14}
                })
            elif change == "Add volatility filter":
                refined_strategy["indicators"].append({
                    "name": "ATR",
                    "parameters": {"period": 14}
                })
            elif change == "Add confirmation indicator":
                refined_strategy["indicators"].append({
                    "name": "Stochastic",
                    "parameters": {"k_period": 14, "d_period": 3, "slowing": 3}
                })
        
        # Apply signal generation changes
        signal_changes = refinement_plan.get("signal_generation_changes", [])
        
        if "signal_generation" not in refined_strategy:
            refined_strategy["signal_generation"] = {}
        
        for change in signal_changes:
            if change == "Filter weak signals":
                refined_strategy["signal_generation"]["filter_weak_signals"] = True
        
        # Apply risk management changes
        risk_changes = refinement_plan.get("risk_management_changes", [])
        
        if "risk_management" not in refined_strategy:
            refined_strategy["risk_management"] = {}
        
        for change in risk_changes:
            if change == "Add trailing stops":
                refined_strategy["risk_management"]["trailing_stop"] = True
            elif change == "Add drawdown protection rules":
                refined_strategy["risk_management"]["drawdown_protection"] = True
                refined_strategy["risk_management"]["max_drawdown_exit"] = 10.0  # Exit if drawdown exceeds 10%
            elif change == "Implement dynamic position sizing":
                refined_strategy["risk_management"]["position_sizing"] = "dynamic"
        
        # Add refinement metadata
        if "refinements" not in refined_strategy:
            refined_strategy["refinements"] = []
        
        refined_strategy["refinements"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "refinement_plan": refinement_plan
        })
        
        self.logger.info("Applied refinements to strategy")
        
        return refined_strategy
    
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
        return1 = self._extract_numeric_value(performance1["total_return"])
        return2 = self._extract_numeric_value(performance2["total_return"])
        
        # Compare total return
        if return1 > return2:
            return True
        
        # If returns are equal, compare Sharpe ratio
        if return1 == return2:
            sharpe1 = self._extract_numeric_value(performance1["sharpe_ratio"])
            sharpe2 = self._extract_numeric_value(performance2["sharpe_ratio"])
            
            return sharpe1 > sharpe2
        
        return False
    
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


class AutomaticStrategyRefinement:
    """
    Provides automatic strategy refinement capabilities.
    """
    
    def __init__(self, backtester: StrategyBacktester = None,
               performance_thresholds: PerformanceThresholds = None,
               refinement_engine: StrategyRefinementEngine = None):
        """
        Initialize the AutomaticStrategyRefinement.
        
        Parameters:
        -----------
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        refinement_engine : StrategyRefinementEngine, optional
            Instance of StrategyRefinementEngine for refining strategies.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.AutomaticStrategyRefinement")
        
        # Create or use provided components
        self.backtester = backtester or StrategyBacktester()
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        self.refinement_engine = refinement_engine or StrategyRefinementEngine(
            backtester=self.backtester,
            performance_thresholds=self.performance_thresholds
        )
        
        self.logger.info("Initialized AutomaticStrategyRefinement")
    
    def refine_strategy_until_valid(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Refine a strategy until it meets performance thresholds.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to refine.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Refined strategy with improved performance.
        """
        self.logger.info(f"Refining strategy for {ticker} until valid")
        
        # Refine the strategy
        refinement_result = self.refinement_engine.refine_strategy(strategy, ticker)
        
        # Return the refinement result
        return refinement_result


class PerformanceFilter:
    """
    Filters strategies based on performance criteria.
    """
    
    def __init__(self, performance_thresholds: PerformanceThresholds = None,
               backtester: StrategyBacktester = None,
               strategy_refinement: AutomaticStrategyRefinement = None):
        """
        Initialize the PerformanceFilter.
        
        Parameters:
        -----------
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        strategy_refinement : AutomaticStrategyRefinement, optional
            Instance of AutomaticStrategyRefinement for refining strategies.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.PerformanceFilter")
        
        # Create or use provided components
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        self.backtester = backtester or StrategyBacktester()
        self.strategy_refinement = strategy_refinement or AutomaticStrategyRefinement(
            backtester=self.backtester,
            performance_thresholds=self.performance_thresholds
        )
        
        self.logger.info("Initialized PerformanceFilter")
    
    def filter_strategy(self, strategy: Dict[str, Any], ticker: str, 
                      auto_refine: bool = True) -> Dict[str, Any]:
        """
        Filter a strategy based on performance criteria.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to filter.
        ticker : str
            Ticker symbol.
        auto_refine : bool, optional
            Whether to automatically refine the strategy if it doesn't meet
            performance thresholds. Default is True.
            
        Returns:
        --------
        Dict[str, Any]
            Filtered strategy result.
        """
        self.logger.info(f"Filtering strategy for {ticker}")
        
        # Backtest the strategy
        backtest_result = self.backtester.backtest_strategy(strategy, ticker)
        
        if not backtest_result["success"]:
            self.logger.error(f"Failed to backtest strategy: {backtest_result.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": "Failed to backtest strategy",
                "strategy": strategy,
                "passed_filter": False
            }
        
        # Get performance metrics
        performance = backtest_result["performance"]
        
        # Check if strategy meets performance thresholds
        is_valid, validation_results = self.performance_thresholds.is_strategy_valid(performance)
        
        # If strategy meets performance thresholds, return it
        if is_valid:
            self.logger.info(f"Strategy for {ticker} meets performance thresholds")
            return {
                "success": True,
                "strategy": strategy,
                "performance": performance,
                "validation_results": validation_results,
                "passed_filter": True,
                "message": "Strategy meets performance thresholds"
            }
        
        # If strategy doesn't meet performance thresholds and auto_refine is True, refine it
        if auto_refine:
            self.logger.info(f"Strategy for {ticker} doesn't meet performance thresholds, refining...")
            
            # Refine the strategy
            refinement_result = self.strategy_refinement.refine_strategy_until_valid(strategy, ticker)
            
            # If refinement was successful, return the refined strategy
            if refinement_result["success"]:
                self.logger.info(f"Successfully refined strategy for {ticker}")
                return {
                    "success": True,
                    "strategy": refinement_result["strategy"],
                    "performance": refinement_result["performance"],
                    "validation_results": refinement_result["validation_results"],
                    "passed_filter": True,
                    "refinement_history": refinement_result["refinement_history"],
                    "message": "Strategy refined to meet performance thresholds"
                }
            else:
                self.logger.warning(f"Failed to refine strategy for {ticker} to meet performance thresholds")
                return {
                    "success": False,
                    "error": "Failed to refine strategy to meet performance thresholds",
                    "strategy": strategy,
                    "performance": performance,
                    "validation_results": validation_results,
                    "passed_filter": False,
                    "refinement_history": refinement_result.get("refinement_history", []),
                    "message": "Strategy doesn't meet performance thresholds and couldn't be refined"
                }
        
        # If strategy doesn't meet performance thresholds and auto_refine is False, return failure
        self.logger.warning(f"Strategy for {ticker} doesn't meet performance thresholds")
        return {
            "success": False,
            "error": "Strategy doesn't meet performance thresholds",
            "strategy": strategy,
            "performance": performance,
            "validation_results": validation_results,
            "passed_filter": False,
            "message": "Strategy doesn't meet performance thresholds"
        }


class CentralDecisionEngine:
    """
    Central decision engine for the Gemma Advanced Trading System.
    """
    
    def __init__(self, performance_thresholds: PerformanceThresholds = None,
               backtester: StrategyBacktester = None,
               strategy_refinement: AutomaticStrategyRefinement = None,
               performance_filter: PerformanceFilter = None):
        """
        Initialize the CentralDecisionEngine.
        
        Parameters:
        -----------
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        strategy_refinement : AutomaticStrategyRefinement, optional
            Instance of AutomaticStrategyRefinement for refining strategies.
            If None, creates a new instance.
        performance_filter : PerformanceFilter, optional
            Instance of PerformanceFilter for filtering strategies.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.CentralDecisionEngine")
        
        # Create or use provided components
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        self.backtester = backtester or StrategyBacktester()
        self.strategy_refinement = strategy_refinement or AutomaticStrategyRefinement(
            backtester=self.backtester,
            performance_thresholds=self.performance_thresholds
        )
        self.performance_filter = performance_filter or PerformanceFilter(
            performance_thresholds=self.performance_thresholds,
            backtester=self.backtester,
            strategy_refinement=self.strategy_refinement
        )
        
        self.logger.info("Initialized CentralDecisionEngine")
    
    def generate_strategy(self, ticker: str) -> Dict[str, Any]:
        """
        Generate a trading strategy for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Generated strategy.
        """
        self.logger.info(f"Generating strategy for {ticker}")
        
        # Generate multiple candidate strategies
        candidates = self._generate_candidate_strategies(ticker)
        
        # Filter and refine strategies
        filtered_candidates = []
        
        for i, candidate in enumerate(candidates):
            self.logger.info(f"Filtering candidate {i + 1}/{len(candidates)}")
            
            # Filter the candidate
            filter_result = self.performance_filter.filter_strategy(candidate, ticker)
            
            # Add to filtered candidates
            filtered_candidates.append(filter_result)
        
        # Get strategies that passed the filter
        passed_candidates = [result for result in filtered_candidates if result.get("passed_filter", False)]
        
        # If no strategies passed the filter, return the best performing one
        if not passed_candidates:
            self.logger.warning(f"No strategies for {ticker} passed the filter")
            
            # Sort by total return (descending)
            sorted_results = sorted(
                filtered_candidates, 
                key=lambda x: self._extract_numeric_value(x.get("performance", {}).get("total_return", 0)), 
                reverse=True
            )
            
            # Get the best strategy
            best_result = sorted_results[0]
            
            return {
                "success": False,
                "error": "No strategies passed the filter",
                "strategy": best_result.get("strategy", {}),
                "performance": best_result.get("performance", {}),
                "message": "No strategies passed the filter, returning best performing one"
            }
        
        # Sort passed strategies by total return (descending)
        sorted_passed = sorted(
            passed_candidates, 
            key=lambda x: self._extract_numeric_value(x.get("performance", {}).get("total_return", 0)), 
            reverse=True
        )
        
        # Get the best strategy
        best_result = sorted_passed[0]
        
        return {
            "success": True,
            "strategy": best_result.get("strategy", {}),
            "performance": best_result.get("performance", {}),
            "message": f"Generated strategy for {ticker} with {best_result.get('performance', {}).get('total_return', 0)}% return"
        }
    
    def _generate_candidate_strategies(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Generate multiple candidate strategies.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of candidate strategies.
        """
        self.logger.info(f"Generating candidate strategies for {ticker}")
        
        # In a real implementation, this would generate multiple strategies
        # For this implementation, we'll create simulated candidates
        
        candidates = []
        
        # Candidate 1: Moving Average Crossover
        candidates.append({
            "name": "Moving Average Crossover",
            "type": "trend_following",
            "parameters": {
                "short_ma": 20,
                "long_ma": 50
            }
        })
        
        # Candidate 2: RSI Mean Reversion
        candidates.append({
            "name": "RSI Mean Reversion",
            "type": "mean_reversion",
            "parameters": {
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30
            }
        })
        
        # Candidate 3: MACD Momentum
        candidates.append({
            "name": "MACD Momentum",
            "type": "momentum",
            "parameters": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            }
        })
        
        self.logger.info(f"Generated {len(candidates)} candidate strategies for {ticker}")
        
        return candidates
    
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


def load_initial_strategy():
    """
    Load the initial AAPL strategy that had negative performance.
    """
    try:
        # Create a simulated initial strategy with negative performance
        initial_strategy = {
            "name": "Initial AAPL Strategy",
            "type": "trend_following",
            "parameters": {
                "short_ma": 20,
                "long_ma": 50
            }
        }
        
        return {
            "strategy": initial_strategy,
            "performance": {
                "total_return": -30.9,
                "sharpe_ratio": -3.37,
                "max_drawdown": -33.36,
                "win_rate": 100.0
            }
        }
    except Exception as e:
        logger.error(f"Failed to load initial strategy: {e}")
        return None

def test_strategy_optimization():
    """
    Test the strategy optimization improvements with AAPL data.
    """
    logger.info("Starting test of strategy optimization with AAPL data")
    
    # Load the initial strategy
    initial_strategy_data = load_initial_strategy()
    
    if not initial_strategy_data:
        logger.error("Failed to load initial strategy data")
        return False
    
    # Extract the strategy
    initial_strategy = initial_strategy_data.get('strategy', {})
    initial_performance = initial_strategy_data.get('performance', {})
    
    # Create components
    performance_thresholds = PerformanceThresholds()
    backtester = StrategyBacktester()
    strategy_refinement = AutomaticStrategyRefinement(
        backtester=backtester,
        performance_thresholds=performance_thresholds
    )
    performance_filter = PerformanceFilter(
        performance_thresholds=performance_thresholds,
        backtester=backtester,
        strategy_refinement=strategy_refinement
    )
    
    # Test initial strategy performance
    logger.info("Testing initial strategy performance")
    
    # Check if initial strategy meets performance thresholds
    is_valid, validation_results = performance_thresholds.is_strategy_valid(initial_performance)
    
    logger.info(f"Initial strategy validation: {is_valid}")
    logger.info(f"Initial strategy performance: {initial_performance}")
    logger.info(f"Initial strategy validation results: {validation_results}")
    
    # Test strategy refinement
    logger.info("Testing strategy refinement")
    refinement_result = strategy_refinement.refine_strategy_until_valid(initial_strategy, 'AAPL')
    
    logger.info(f"Refinement success: {refinement_result['success']}")
    
    if refinement_result["success"]:
        logger.info(f"Refined strategy performance: {refinement_result['performance']}")
        
        # Verify that the refined strategy meets performance thresholds
        refined_is_valid, refined_validation = performance_thresholds.is_strategy_valid(refinement_result['performance'])
        logger.info(f"Refined strategy validation: {refined_is_valid}")
        logger.info(f"Refined strategy validation results: {refined_validation}")
        
        # Save the refined strategy
        with open('/home/ubuntu/gemma_advanced/aapl_refined_strategy.json', 'w') as f:
            json.dump(refinement_result, f, indent=2)
        
        logger.info("Saved refined strategy to aapl_refined_strategy.json")
    else:
        logger.warning(f"Strategy refinement failed: {refinement_result.get('message', 'Unknown error')}")
    
    # Test performance filtering
    logger.info("Testing performance filtering")
    filter_result = performance_filter.filter_strategy(initial_strategy, 'AAPL', auto_refine=True)
    
    logger.info(f"Filter success: {filter_result['success']}")
    logger.info(f"Filter passed: {filter_result.get('passed_filter', False)}")
    
    if filter_result["success"]:
        logger.info(f"Filtered strategy performance: {filter_result['performance']}")
        
        # Save the filtered strategy
        with open('/home/ubuntu/gemma_advanced/aapl_filtered_strategy.json', 'w') as f:
            json.dump(filter_result, f, indent=2)
        
        logger.info("Saved filtered strategy to aapl_filtered_strategy.json")
    else:
        logger.warning(f"Strategy filtering failed: {filter_result.get('message', 'Unknown error')}")
    
    # Test central decision engine
    logger.info("Testing central decision engine")
    central_engine = CentralDecisionEngine()
    
    decision_result = central_engine.generate_strategy('AAPL')
    
    logger.info(f"Decision engine success: {decision_result['success']}")
    
    if decision_result["success"]:
        logger.info(f"Generated strategy performance: {decision_result.get('performance', {})}")
        
        # Save the generated strategy
        with open('/home/ubuntu/gemma_advanced/aapl_generated_strategy.json', 'w') as f:
            json.dump(decision_result, f, indent=2)
        
        logger.info("Saved generated strategy to aapl_generated_strategy.json")
    else:
        logger.warning(f"Strategy generation failed: {decision_result.get('message', 'Unknown error')}")
    
    # Compare initial and optimized strategies
    if refinement_result["success"] and filter_result["success"] and decision_result["success"]:
        logger.info("Comparing initial and optimized strategies")
        
        # Extract performance metrics
        initial_return = extract_numeric_value(initial_performance.get('total_return', 0))
        refined_return = extract_numeric_value(refinement_result['performance'].get('total_return', 0))
        filtered_return = extract_numeric_value(filter_result['performance'].get('total_return', 0))
        generated_return = extract_numeric_value(decision_result.get('performance', {}).get('total_return', 0))
        
        initial_sharpe = extract_numeric_value(initial_performance.get('sharpe_ratio', 0))
        refined_sharpe = extract_numeric_value(refinement_result['performance'].get('sharpe_ratio', 0))
        filtered_sharpe = extract_numeric_value(filter_result['performance'].get('sharpe_ratio', 0))
        generated_sharpe = extract_numeric_value(decision_result.get('performance', {}).get('sharpe_ratio', 0))
        
        # Create comparison report
        comparison = {
            "timestamp": datetime.datetime.now().isoformat(),
            "initial_strategy": {
                "total_return": initial_return,
                "sharpe_ratio": initial_sharpe,
                "validation": is_valid
            },
            "refined_strategy": {
                "total_return": refined_return,
                "sharpe_ratio": refined_sharpe,
                "validation": refined_is_valid,
                "improvement": {
                    "total_return": refined_return - initial_return,
                    "sharpe_ratio": refined_sharpe - initial_sharpe
                }
            },
            "filtered_strategy": {
                "total_return": filtered_return,
                "sharpe_ratio": filtered_sharpe,
                "validation": filter_result["success"],
                "improvement": {
                    "total_return": filtered_return - initial_return,
                    "sharpe_ratio": filtered_sharpe - initial_sharpe
                }
            },
            "generated_strategy": {
                "total_return": generated_return,
                "sharpe_ratio": generated_sharpe,
                "validation": decision_result["success"],
                "improvement": {
                    "total_return": generated_return - initial_return,
                    "sharpe_ratio": generated_sharpe - initial_sharpe
                }
            }
        }
        
        # Save comparison report
        with open('/home/ubuntu/gemma_advanced/aapl_strategy_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info("Saved strategy comparison to aapl_strategy_comparison.json")
        
        # Check if all optimized strategies have positive returns
        all_positive = (
            refined_return > 0 and
            filtered_return > 0 and
            generated_return > 0
        )
        
        logger.info(f"All optimized strategies have positive returns: {all_positive}")
        
        return all_positive
    else:
        logger.warning("Not all optimization methods succeeded")
        return False

def extract_numeric_value(value: Any) -> float:
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

if __name__ == "__main__":
    logger.info("Starting AAPL strategy test")
    
    success = test_strategy_optimization()
    
    if success:
        logger.info("AAPL strategy test completed successfully")
        print("SUCCESS: All optimized strategies have positive returns")
    else:
        logger.error("AAPL strategy test failed")
        print("FAILURE: Not all optimized strategies have positive returns")
