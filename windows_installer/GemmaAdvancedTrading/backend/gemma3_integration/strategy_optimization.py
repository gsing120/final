"""
Strategy Optimization Module for Gemma Advanced Trading System

This module implements strategy optimization capabilities using Gemma 3
to ensure that only strategies with positive historical performance are
presented to users.
"""

import os
import logging
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import uuid
import yfinance as yf

# Import Gemma 3 integration components
from gemma3_integration.architecture_enhanced import GemmaCore, PromptEngine, ModelManager
from gemma3_integration.strategy_generation_and_refinement import StrategyGenerator, StrategyRefiner
from gemma3_integration.adaptive_learning import AdaptiveLearning, TradeMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class PerformanceThresholds:
    """
    Defines performance thresholds for strategy validation.
    
    This class provides default and customizable thresholds for various
    performance metrics that strategies must meet to be considered valid.
    """
    
    def __init__(self, 
                min_total_return: float = 0.0,
                min_sharpe_ratio: float = 0.5,
                max_drawdown: float = -20.0,
                min_win_rate: float = 0.5,
                max_volatility: Optional[float] = None):
        """
        Initialize performance thresholds.
        
        Parameters:
        -----------
        min_total_return : float, optional
            Minimum acceptable total return percentage. Default is 0.0 (must be positive).
        min_sharpe_ratio : float, optional
            Minimum acceptable Sharpe ratio. Default is 0.5.
        max_drawdown : float, optional
            Maximum acceptable drawdown percentage (negative value). Default is -20.0.
        min_win_rate : float, optional
            Minimum acceptable win rate (0.0 to 1.0). Default is 0.5.
        max_volatility : float, optional
            Maximum acceptable volatility percentage. Default is None (no limit).
        """
        self.min_total_return = min_total_return
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate
        self.max_volatility = max_volatility
        
    def is_strategy_valid(self, performance: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a strategy meets the performance thresholds.
        
        Parameters:
        -----------
        performance : Dict[str, Any]
            Performance metrics for the strategy.
            
        Returns:
        --------
        Tuple[bool, Dict[str, Any]]
            A tuple containing:
            - Boolean indicating if the strategy is valid
            - Dictionary of validation results for each metric
        """
        # Extract performance metrics, handling different formats
        total_return = self._extract_numeric_value(performance.get('total_return', 0))
        sharpe_ratio = self._extract_numeric_value(performance.get('sharpe_ratio', 0))
        max_dd = self._extract_numeric_value(performance.get('max_drawdown', 0))
        win_rate = self._extract_numeric_value(performance.get('win_rate', 0))
        volatility = self._extract_numeric_value(performance.get('volatility', 0))
        
        # Check each threshold
        validation_results = {
            'total_return': total_return >= self.min_total_return,
            'sharpe_ratio': sharpe_ratio >= self.min_sharpe_ratio,
            'max_drawdown': max_dd >= self.max_drawdown,  # Note: drawdown is negative
            'win_rate': win_rate >= self.min_win_rate
        }
        
        # Only check volatility if a threshold is set
        if self.max_volatility is not None:
            validation_results['volatility'] = volatility <= self.max_volatility
        
        # Strategy is valid if all checks pass
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
    Backtests trading strategies to evaluate their historical performance.
    
    This class provides methods for backtesting trading strategies on
    historical data to evaluate their performance.
    """
    
    def __init__(self, data_provider: Optional[Callable] = None):
        """
        Initialize the StrategyBacktester.
        
        Parameters:
        -----------
        data_provider : Callable, optional
            Function to provide historical data for backtesting.
            If None, uses default yfinance data provider.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyBacktester")
        self.data_provider = data_provider or self._default_data_provider
        
        self.logger.info("Initialized StrategyBacktester")
    
    def _default_data_provider(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Default data provider using yfinance.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        period : str, optional
            Period to download data for. Default is "1y".
        interval : str, optional
            Interval between data points. Default is "1d".
            
        Returns:
        --------
        pd.DataFrame
            Historical data for the ticker.
        """
        self.logger.info(f"Getting data for {ticker} with period={period}, interval={interval}")
        
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False
            )
            
            if data.empty:
                self.logger.warning(f"No data found for {ticker}")
                return None
            
            self.logger.info(f"Got {len(data)} rows of data for {ticker}")
            return data
        
        except Exception as e:
            self.logger.exception(f"Error getting data for {ticker}: {e}")
            return None
    
    def backtest_strategy(self, strategy: Dict[str, Any], ticker: str, 
                         period: str = "180d", interval: str = "1d") -> Dict[str, Any]:
        """
        Backtest a trading strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to backtest.
        ticker : str
            Ticker symbol.
        period : str, optional
            Period to backtest for. Default is "180d".
        interval : str, optional
            Interval between data points. Default is "1d".
            
        Returns:
        --------
        Dict[str, Any]
            Backtest results, including performance metrics.
        """
        self.logger.info(f"Backtesting strategy for {ticker}")
        
        # Get historical data
        data = self.data_provider(ticker, period, interval)
        
        if data is None or data.empty:
            self.logger.error(f"No data available for {ticker}")
            return {
                "success": False,
                "error": "No data available"
            }
        
        # Calculate indicators based on strategy
        data = self._calculate_indicators(data, strategy)
        
        # Generate signals based on strategy
        data = self._generate_signals(data, strategy)
        
        # Generate trades based on signals
        trades = self._generate_trades(data)
        
        # Calculate performance metrics
        performance = self._calculate_performance(data, trades)
        
        # Return backtest results
        backtest_results = {
            "success": True,
            "ticker": ticker,
            "strategy": strategy,
            "performance": performance,
            "trades": trades,
            "data_summary": {
                "start_date": data.index[0].strftime('%Y-%m-%d'),
                "end_date": data.index[-1].strftime('%Y-%m-%d'),
                "num_days": len(data)
            }
        }
        
        self.logger.info(f"Backtest completed for {ticker} with total return: {performance['total_return']}")
        
        return backtest_results
    
    def _calculate_indicators(self, data: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate technical indicators based on strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data.
        strategy : Dict[str, Any]
            Trading strategy with indicator specifications.
            
        Returns:
        --------
        pd.DataFrame
            Data with calculated indicators.
        """
        self.logger.info("Calculating technical indicators")
        
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
        
        # ATR
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Add custom indicators based on strategy if needed
        # This would parse the strategy definition and add specific indicators
        
        self.logger.info("Finished calculating technical indicators")
        
        return df
    
    def _generate_signals(self, data: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate trading signals based on strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data with indicators.
        strategy : Dict[str, Any]
            Trading strategy with signal generation rules.
            
        Returns:
        --------
        pd.DataFrame
            Data with generated signals.
        """
        self.logger.info("Generating trading signals")
        
        # Initialize signal column
        data['signal'] = 0
        
        # Handle NaN values to avoid alignment issues
        data = data.fillna(0)
        
        # Extract strategy type
        strategy_type = strategy.get('strategy_type', '').lower()
        
        # Generate signals based on strategy type
        if 'trend' in strategy_type or 'swing' in strategy_type:
            # Trend following strategy
            for i in range(1, len(data)):
                # Buy conditions
                if (data['SMA20'].iloc[i] > data['SMA50'].iloc[i] and 
                    data['RSI'].iloc[i] > 30 and 
                    data['RSI'].iloc[i] < 70 and 
                    data['MACD'].iloc[i] > data['MACD_signal'].iloc[i]):
                    data.loc[data.index[i], 'signal'] = 1
                
                # Sell conditions
                elif (data['SMA20'].iloc[i] < data['SMA50'].iloc[i] and 
                      (data['RSI'].iloc[i] > 70 or data['RSI'].iloc[i] < 30) and 
                      data['MACD'].iloc[i] < data['MACD_signal'].iloc[i]):
                    data.loc[data.index[i], 'signal'] = -1
        
        elif 'mean' in strategy_type or 'reversion' in strategy_type:
            # Mean reversion strategy
            for i in range(1, len(data)):
                # Buy conditions (oversold)
                if (data['RSI'].iloc[i] < 30 and 
                    data['Close'].iloc[i] < data['BB_lower'].iloc[i]):
                    data.loc[data.index[i], 'signal'] = 1
                
                # Sell conditions (overbought)
                elif (data['RSI'].iloc[i] > 70 and 
                      data['Close'].iloc[i] > data['BB_upper'].iloc[i]):
                    data.loc[data.index[i], 'signal'] = -1
        
        else:
            # Default strategy (similar to the one in direct_strategy_generator.py)
            for i in range(1, len(data)):
                # Buy conditions
                if (data['SMA20'].iloc[i] > data['SMA50'].iloc[i] and 
                    data['RSI'].iloc[i] > 50):
                    data.loc[data.index[i], 'signal'] = 1
                
                # Sell conditions
                elif (data['SMA20'].iloc[i] < data['SMA50'].iloc[i] and 
                      data['RSI'].iloc[i] < 50):
                    data.loc[data.index[i], 'signal'] = -1
        
        self.logger.info("Finished generating trading signals")
        
        return data
    
    def _generate_trades(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trades based on signals.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data with signals.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of trades.
        """
        self.logger.info("Generating trades")
        
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for i in range(1, len(data)):
            if data['signal'].iloc[i] == 1 and position == 0:  # Buy signal and no position
                position = 1
                entry_price = float(data['Close'].iloc[i])
                entry_date = data.index[i]
                trades.append({
                    'date': entry_date.strftime('%Y-%m-%d'),
                    'type': 'BUY',
                    'price': f"${entry_price:.2f}",
                    'shares': 100,
                    'pnl': ''
                })
            elif (data['signal'].iloc[i] == -1 or i == len(data) - 1) and position == 1:  # Sell signal or last day and have position
                exit_price = float(data['Close'].iloc[i])
                exit_date = data.index[i]
                pnl = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'date': exit_date.strftime('%Y-%m-%d'),
                    'type': 'SELL',
                    'price': f"${exit_price:.2f}",
                    'shares': 100,
                    'pnl': f"{pnl:.2f}%"
                })
                position = 0
        
        self.logger.info(f"Generated {len(trades)} trades")
        
        return trades
    
    def _calculate_performance(self, data: pd.DataFrame, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data with signals.
        trades : List[Dict[str, Any]]
            List of trades.
            
        Returns:
        --------
        Dict[str, Any]
            Performance metrics.
        """
        self.logger.info("Calculating performance metrics")
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        # Calculate total return
        total_return = float((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100)
        
        # Calculate Sharpe ratio
        sharpe_ratio = float(returns.mean() / returns.std() * (252 ** 0.5))  # Annualized Sharpe ratio
        
        # Calculate maximum drawdown
        peak = data['Close'].expanding(min_periods=1).max()
        drawdown = (data['Close'] / peak - 1) * 100
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
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate * 100, 2),
            'volatility': round(volatility * 100, 2),
            'num_trades': len(trades) // 2
        }
        
        self.logger.info(f"Performance metrics: {performance}")
        
        return performance

class StrategyOptimizer:
    """
    Optimizes trading strategies to ensure positive historical performance.
    
    This class provides methods for optimizing trading strategies by generating
    multiple candidates, backtesting them, and selecting the best performing ones.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None,
               strategy_generator: Optional[StrategyGenerator] = None,
               strategy_refiner: Optional[StrategyRefiner] = None,
               backtester: Optional[StrategyBacktester] = None,
               performance_thresholds: Optional[PerformanceThresholds] = None,
               max_optimization_iterations: int = 5,
               num_candidate_strategies: int = 3):
        """
        Initialize the StrategyOptimizer.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        strategy_generator : StrategyGenerator, optional
            Instance of StrategyGenerator for generating strategies.
            If None, creates a new instance.
        strategy_refiner : StrategyRefiner, optional
            Instance of StrategyRefiner for refining strategies.
            If None, creates a new instance.
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        max_optimization_iterations : int, optional
            Maximum number of optimization iterations. Default is 5.
        num_candidate_strategies : int, optional
            Number of candidate strategies to generate. Default is 3.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyOptimizer")
        
        # Create or use provided components
        self.gemma_core = gemma_core or GemmaCore()
        self.strategy_generator = strategy_generator or StrategyGenerator(self.gemma_core)
        self.strategy_refiner = strategy_refiner or StrategyRefiner(self.gemma_core)
        self.backtester = backtester or StrategyBacktester()
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        
        # Configuration
        self.max_optimization_iterations = max_optimization_iterations
        self.num_candidate_strategies = num_candidate_strategies
        
        self.logger.info("Initialized StrategyOptimizer")
    
    def generate_optimized_strategy(self, ticker: str, 
                                  market_conditions: Dict[str, Any],
                                  trading_objectives: Dict[str, Any],
                                  constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an optimized trading strategy with positive historical performance.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        market_conditions : Dict[str, Any]
            Current market conditions.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target, time horizon).
        constraints : Dict[str, Any], optional
            Constraints for the strategy (e.g., max drawdown, min win rate).
            
        Returns:
        --------
        Dict[str, Any]
            Optimized trading strategy with positive historical performance.
        """
        self.logger.info(f"Generating optimized strategy for {ticker}")
        
        # Initialize optimization variables
        best_strategy = None
        best_performance = None
        optimization_history = []
        
        # Generate and optimize strategies
        for iteration in range(self.max_optimization_iterations):
            self.logger.info(f"Optimization iteration {iteration + 1}/{self.max_optimization_iterations}")
            
            # Generate candidate strategies
            candidate_strategies = self._generate_candidate_strategies(
                ticker, market_conditions, trading_objectives, constraints
            )
            
            # Backtest candidate strategies
            backtest_results = self._backtest_candidate_strategies(ticker, candidate_strategies)
            
            # Find the best performing strategy
            current_best_strategy, current_best_performance = self._select_best_strategy(backtest_results)
            
            # Record optimization history
            optimization_history.append({
                "iteration": iteration + 1,
                "num_candidates": len(candidate_strategies),
                "best_performance": current_best_performance
            })
            
            # Update best strategy if better than previous best
            if best_strategy is None or self._is_better_strategy(current_best_performance, best_performance):
                best_strategy = current_best_strategy
                best_performance = current_best_performance
                self.logger.info(f"Found better strategy with total return: {best_performance['total_return']}")
            
            # Check if strategy meets performance thresholds
            is_valid, validation_results = self.performance_thresholds.is_strategy_valid(best_performance)
            
            if is_valid:
                self.logger.info(f"Strategy meets performance thresholds after {iteration + 1} iterations")
                break
            
            # If not valid and not last iteration, refine the strategy
            if iteration < self.max_optimization_iterations - 1:
                self.logger.info(f"Refining strategy for next iteration")
                
                # Use Gemma 3 to analyze performance and suggest improvements
                improvement_suggestions = self._analyze_performance_for_improvements(
                    best_strategy, best_performance, validation_results
                )
                
                # Update constraints based on improvement suggestions
                if constraints is None:
                    constraints = {}
                
                constraints["improvement_suggestions"] = improvement_suggestions
        
        # If no valid strategy found after all iterations, use the best one we have
        if best_strategy is None:
            self.logger.warning(f"No valid strategy found after {self.max_optimization_iterations} iterations")
            return {
                "success": False,
                "error": "No valid strategy found",
                "optimization_history": optimization_history
            }
        
        # Add optimization history to the strategy
        best_strategy["optimization_history"] = optimization_history
        
        # Add performance metrics to the strategy
        best_strategy["performance"] = best_performance
        
        # Add validation results to the strategy
        is_valid, validation_results = self.performance_thresholds.is_strategy_valid(best_performance)
        best_strategy["validation_results"] = validation_results
        
        self.logger.info(f"Generated optimized strategy for {ticker} with total return: {best_performance['total_return']}")
        
        return best_strategy
    
    def _generate_candidate_strategies(self, ticker: str, 
                                     market_conditions: Dict[str, Any],
                                     trading_objectives: Dict[str, Any],
                                     constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate multiple candidate strategies.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        market_conditions : Dict[str, Any]
            Current market conditions.
        trading_objectives : Dict[str, Any]
            Trading objectives.
        constraints : Dict[str, Any], optional
            Constraints for the strategy.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of candidate strategies.
        """
        self.logger.info(f"Generating {self.num_candidate_strategies} candidate strategies for {ticker}")
        
        candidate_strategies = []
        
        # Generate strategies with different parameters
        for i in range(self.num_candidate_strategies):
            # Vary strategy parameters slightly for each candidate
            varied_objectives = self._vary_parameters(trading_objectives, i)
            varied_constraints = self._vary_parameters(constraints, i) if constraints else None
            
            # Generate strategy
            strategy = self.strategy_generator.generate_strategy(
                "stock", market_conditions, varied_objectives, varied_constraints
            )
            
            # Add ticker and candidate number
            strategy["ticker"] = ticker
            strategy["candidate_number"] = i + 1
            
            candidate_strategies.append(strategy)
        
        self.logger.info(f"Generated {len(candidate_strategies)} candidate strategies")
        
        return candidate_strategies
    
    def _vary_parameters(self, params: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """
        Vary parameters slightly to generate different strategies.
        
        Parameters:
        -----------
        params : Dict[str, Any]
            Parameters to vary.
        seed : int
            Seed for variation.
            
        Returns:
        --------
        Dict[str, Any]
            Varied parameters.
        """
        if params is None:
            return {}
        
        # Create a copy of the parameters
        varied_params = params.copy()
        
        # Vary numerical parameters slightly
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Vary by +/- 10% based on seed
                np.random.seed(seed)
                variation = 0.9 + 0.2 * np.random.random()  # 0.9 to 1.1
                varied_params[key] = value * variation
        
        return varied_params
    
    def _backtest_candidate_strategies(self, ticker: str, 
                                     candidate_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Backtest candidate strategies.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        candidate_strategies : List[Dict[str, Any]]
            List of candidate strategies.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of backtest results.
        """
        self.logger.info(f"Backtesting {len(candidate_strategies)} candidate strategies for {ticker}")
        
        backtest_results = []
        
        for strategy in candidate_strategies:
            # Backtest the strategy
            result = self.backtester.backtest_strategy(strategy, ticker)
            
            # Add to results if successful
            if result["success"]:
                backtest_results.append(result)
        
        self.logger.info(f"Completed backtesting {len(backtest_results)} strategies")
        
        return backtest_results
    
    def _select_best_strategy(self, backtest_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Select the best performing strategy.
        
        Parameters:
        -----------
        backtest_results : List[Dict[str, Any]]
            List of backtest results.
            
        Returns:
        --------
        Tuple[Dict[str, Any], Dict[str, Any]]
            A tuple containing:
            - Best strategy
            - Performance metrics of the best strategy
        """
        self.logger.info(f"Selecting best strategy from {len(backtest_results)} candidates")
        
        if not backtest_results:
            self.logger.warning("No backtest results available")
            return None, None
        
        # Sort by total return (descending)
        sorted_results = sorted(
            backtest_results, 
            key=lambda x: self._extract_numeric_value(x["performance"]["total_return"]), 
            reverse=True
        )
        
        # Get the best strategy
        best_result = sorted_results[0]
        best_strategy = best_result["strategy"]
        best_performance = best_result["performance"]
        
        self.logger.info(f"Selected best strategy with total return: {best_performance['total_return']}")
        
        return best_strategy, best_performance
    
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
    
    def _is_better_strategy(self, performance1: Dict[str, Any], performance2: Dict[str, Any]) -> bool:
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
        return return1 > return2
    
    def _analyze_performance_for_improvements(self, strategy: Dict[str, Any], 
                                           performance: Dict[str, Any],
                                           validation_results: Dict[str, bool]) -> Dict[str, Any]:
        """
        Analyze performance and suggest improvements.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to analyze.
        performance : Dict[str, Any]
            Performance metrics of the strategy.
        validation_results : Dict[str, bool]
            Validation results for each performance metric.
            
        Returns:
        --------
        Dict[str, Any]
            Improvement suggestions.
        """
        self.logger.info("Analyzing performance for improvements")
        
        # Identify failing metrics
        failing_metrics = [metric for metric, is_valid in validation_results.items() if not is_valid]
        
        # Prepare context for Gemma 3
        context = {
            "strategy": strategy,
            "performance": performance,
            "validation_results": validation_results,
            "failing_metrics": failing_metrics
        }
        
        # Generate prompt for improvement analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_improvement",
            **context
        )
        
        # Get the appropriate model for improvement analysis
        model = self.gemma_core.model_manager.get_model("strategy_improvement")
        
        # Generate improvement suggestions using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # In a real implementation, this would parse the result to extract structured suggestions
        # For this implementation, we'll create simulated suggestions
        
        improvement_suggestions = {
            "parameter_adjustments": {},
            "indicator_changes": [],
            "signal_generation_changes": [],
            "risk_management_changes": []
        }
        
        # Suggest improvements based on failing metrics
        for metric in failing_metrics:
            if metric == "total_return":
                improvement_suggestions["parameter_adjustments"]["entry_threshold"] = "more_aggressive"
                improvement_suggestions["indicator_changes"].append("Add momentum indicator")
            elif metric == "sharpe_ratio":
                improvement_suggestions["parameter_adjustments"]["exit_threshold"] = "more_conservative"
                improvement_suggestions["risk_management_changes"].append("Tighter stop loss")
            elif metric == "max_drawdown":
                improvement_suggestions["parameter_adjustments"]["stop_loss"] = "tighter"
                improvement_suggestions["signal_generation_changes"].append("Add drawdown protection")
            elif metric == "win_rate":
                improvement_suggestions["parameter_adjustments"]["entry_confirmation"] = "stricter"
                improvement_suggestions["signal_generation_changes"].append("Add confirmation filter")
            elif metric == "volatility":
                improvement_suggestions["parameter_adjustments"]["volatility_filter"] = "stricter"
                improvement_suggestions["indicator_changes"].append("Add volatility filter")
        
        self.logger.info(f"Generated improvement suggestions for {len(failing_metrics)} failing metrics")
        
        return improvement_suggestions
