"""
Comprehensive backtesting validation for optimized trading strategies.

This script performs extensive backtesting validation on optimized trading strategies
to ensure they perform well under various market conditions.
"""

import os
import sys
import json
import logging
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/ubuntu/gemma_advanced/strategy_validation.log')
    ]
)

logger = logging.getLogger("GemmaTrading.StrategyValidation")

class StrategyValidator:
    """
    Validates trading strategies through comprehensive backtesting.
    """
    
    def __init__(self):
        """
        Initialize the StrategyValidator.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyValidator")
        self.logger.info("Initialized StrategyValidator")
    
    def validate_strategy(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Validate a strategy through comprehensive backtesting.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to validate.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Validation results.
        """
        self.logger.info(f"Validating strategy for {ticker}")
        
        validation_results = {}
        
        # Perform various validation tests
        validation_results["full_history"] = self._validate_full_history(strategy, ticker)
        validation_results["different_periods"] = self._validate_different_periods(strategy, ticker)
        validation_results["market_regimes"] = self._validate_market_regimes(strategy, ticker)
        validation_results["robustness"] = self._validate_robustness(strategy, ticker)
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(validation_results)
        
        # Generate validation report
        validation_report = self._generate_validation_report(validation_results, validation_score)
        
        # Generate performance charts
        chart_paths = self._generate_performance_charts(validation_results, ticker)
        
        # Prepare final validation result
        result = {
            "strategy": strategy,
            "ticker": ticker,
            "validation_results": validation_results,
            "validation_score": validation_score,
            "validation_report": validation_report,
            "chart_paths": chart_paths
        }
        
        self.logger.info(f"Completed strategy validation for {ticker} with score: {validation_score}")
        
        return result
    
    def _validate_full_history(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Validate strategy on full available history.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to validate.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Full history validation results.
        """
        self.logger.info(f"Validating strategy on full history for {ticker}")
        
        try:
            # Get full history data
            data = self._get_historical_data(ticker, period="max")
            
            if data is None or len(data) == 0:
                self.logger.error(f"Failed to get full history data for {ticker}")
                return {
                    "success": False,
                    "error": f"Failed to get full history data for {ticker}"
                }
            
            # Apply strategy to historical data
            signals = self._generate_signals(data, strategy)
            
            # Calculate performance metrics
            performance = self._calculate_performance(data, signals)
            
            # Calculate annual returns
            annual_returns = self._calculate_annual_returns(data, signals)
            
            # Calculate monthly returns
            monthly_returns = self._calculate_monthly_returns(data, signals)
            
            self.logger.info(f"Completed full history validation for {ticker}")
            
            return {
                "success": True,
                "performance": performance,
                "annual_returns": annual_returns,
                "monthly_returns": monthly_returns,
                "data_length": len(data),
                "start_date": data.index[0].strftime("%Y-%m-%d"),
                "end_date": data.index[-1].strftime("%Y-%m-%d")
            }
        
        except Exception as e:
            self.logger.error(f"Error validating strategy on full history for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_different_periods(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Validate strategy on different time periods.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to validate.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Different periods validation results.
        """
        self.logger.info(f"Validating strategy on different periods for {ticker}")
        
        periods = {
            "1y": "Last 1 Year",
            "2y": "Last 2 Years",
            "5y": "Last 5 Years",
            "10y": "Last 10 Years"
        }
        
        period_results = {}
        
        for period_key, period_name in periods.items():
            self.logger.info(f"Validating strategy for period: {period_name}")
            
            try:
                # Get period data
                data = self._get_historical_data(ticker, period=period_key)
                
                if data is None or len(data) == 0:
                    self.logger.warning(f"Failed to get data for period {period_name}")
                    period_results[period_key] = {
                        "success": False,
                        "error": f"Failed to get data for period {period_name}"
                    }
                    continue
                
                # Apply strategy to historical data
                signals = self._generate_signals(data, strategy)
                
                # Calculate performance metrics
                performance = self._calculate_performance(data, signals)
                
                period_results[period_key] = {
                    "success": True,
                    "performance": performance,
                    "data_length": len(data),
                    "start_date": data.index[0].strftime("%Y-%m-%d"),
                    "end_date": data.index[-1].strftime("%Y-%m-%d")
                }
                
                self.logger.info(f"Completed validation for period: {period_name}")
            
            except Exception as e:
                self.logger.error(f"Error validating strategy for period {period_name}: {e}")
                period_results[period_key] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.logger.info(f"Completed different periods validation for {ticker}")
        
        return {
            "success": True,
            "period_results": period_results
        }
    
    def _validate_market_regimes(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Validate strategy on different market regimes.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to validate.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Market regimes validation results.
        """
        self.logger.info(f"Validating strategy on different market regimes for {ticker}")
        
        # Define market regimes
        market_regimes = {
            "bull_market": {
                "name": "Bull Market",
                "start_date": "2020-04-01",
                "end_date": "2021-12-31"
            },
            "bear_market": {
                "name": "Bear Market",
                "start_date": "2022-01-01",
                "end_date": "2022-12-31"
            },
            "sideways_market": {
                "name": "Sideways Market",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            },
            "volatile_market": {
                "name": "Volatile Market",
                "start_date": "2020-02-01",
                "end_date": "2020-05-31"
            }
        }
        
        regime_results = {}
        
        for regime_key, regime_info in market_regimes.items():
            self.logger.info(f"Validating strategy for market regime: {regime_info['name']}")
            
            try:
                # Get regime data
                data = self._get_historical_data_by_date_range(
                    ticker, 
                    start_date=regime_info["start_date"], 
                    end_date=regime_info["end_date"]
                )
                
                if data is None or len(data) == 0:
                    self.logger.warning(f"Failed to get data for market regime {regime_info['name']}")
                    regime_results[regime_key] = {
                        "success": False,
                        "error": f"Failed to get data for market regime {regime_info['name']}"
                    }
                    continue
                
                # Apply strategy to historical data
                signals = self._generate_signals(data, strategy)
                
                # Calculate performance metrics
                performance = self._calculate_performance(data, signals)
                
                regime_results[regime_key] = {
                    "success": True,
                    "performance": performance,
                    "data_length": len(data),
                    "start_date": data.index[0].strftime("%Y-%m-%d"),
                    "end_date": data.index[-1].strftime("%Y-%m-%d")
                }
                
                self.logger.info(f"Completed validation for market regime: {regime_info['name']}")
            
            except Exception as e:
                self.logger.error(f"Error validating strategy for market regime {regime_info['name']}: {e}")
                regime_results[regime_key] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.logger.info(f"Completed market regimes validation for {ticker}")
        
        return {
            "success": True,
            "regime_results": regime_results
        }
    
    def _validate_robustness(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Validate strategy robustness through Monte Carlo simulations.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to validate.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Robustness validation results.
        """
        self.logger.info(f"Validating strategy robustness for {ticker}")
        
        try:
            # Get historical data
            data = self._get_historical_data(ticker, period="5y")
            
            if data is None or len(data) == 0:
                self.logger.error(f"Failed to get historical data for {ticker}")
                return {
                    "success": False,
                    "error": f"Failed to get historical data for {ticker}"
                }
            
            # Apply strategy to historical data
            signals = self._generate_signals(data, strategy)
            
            # Get strategy returns
            strategy_returns = signals['strategy_returns'].dropna()
            
            # Run Monte Carlo simulations
            num_simulations = 100
            simulation_results = []
            
            for i in range(num_simulations):
                # Resample returns with replacement
                simulated_returns = strategy_returns.sample(n=len(strategy_returns), replace=True)
                
                # Calculate cumulative returns
                cumulative_returns = (1 + simulated_returns).cumprod()
                
                # Calculate total return
                total_return = cumulative_returns.iloc[-1] - 1
                
                # Calculate maximum drawdown
                cumulative_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns - cumulative_max) / cumulative_max
                max_drawdown = drawdown.min() * 100
                
                # Calculate Sharpe ratio
                sharpe_ratio = simulated_returns.mean() / simulated_returns.std() * np.sqrt(252)
                
                # Add to simulation results
                simulation_results.append({
                    "total_return": total_return * 100,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown
                })
            
            # Calculate statistics
            total_returns = [result["total_return"] for result in simulation_results]
            sharpe_ratios = [result["sharpe_ratio"] for result in simulation_results]
            max_drawdowns = [result["max_drawdown"] for result in simulation_results]
            
            # Calculate percentiles
            total_return_5th = np.percentile(total_returns, 5)
            total_return_95th = np.percentile(total_returns, 95)
            sharpe_ratio_5th = np.percentile(sharpe_ratios, 5)
            max_drawdown_5th = np.percentile(max_drawdowns, 5)
            
            self.logger.info(f"Completed robustness validation for {ticker}")
            
            return {
                "success": True,
                "num_simulations": num_simulations,
                "total_return_mean": np.mean(total_returns),
                "total_return_std": np.std(total_returns),
                "total_return_5th": total_return_5th,
                "total_return_95th": total_return_95th,
                "sharpe_ratio_mean": np.mean(sharpe_ratios),
                "sharpe_ratio_std": np.std(sharpe_ratios),
                "sharpe_ratio_5th": sharpe_ratio_5th,
                "max_drawdown_mean": np.mean(max_drawdowns),
                "max_drawdown_std": np.std(max_drawdowns),
                "max_drawdown_5th": max_drawdown_5th
            }
        
        except Exception as e:
            self.logger.error(f"Error validating strategy robustness for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical data for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        period : str, optional
            Period to get data for. Default is "1y".
            
        Returns:
        --------
        pd.DataFrame
            Historical data.
        """
        self.logger.info(f"Getting historical data for {ticker} with period {period}")
        
        try:
            # Get data from Yahoo Finance
            data = yf.download(ticker, period=period)
            
            if len(data) == 0:
                self.logger.error(f"No data found for {ticker}")
                return None
            
            self.logger.info(f"Got {len(data)} data points for {ticker}")
            
            return data
        
        except Exception as e:
            self.logger.error(f"Error getting historical data for {ticker}: {e}")
            return None
    
    def _get_historical_data_by_date_range(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical data for a ticker by date range.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        start_date : str
            Start date in format "YYYY-MM-DD".
        end_date : str
            End date in format "YYYY-MM-DD".
            
        Returns:
        --------
        pd.DataFrame
            Historical data.
        """
        self.logger.info(f"Getting historical data for {ticker} from {start_date} to {end_date}")
        
        try:
            # Get data from Yahoo Finance
            data = yf.download(ticker, start=start_date, end=end_date)
            
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
        
        # Calculate returns
        signals['returns'] = data['Close'].pct_change()
        signals['strategy_returns'] = signals['signal'].shift(1) * signals['returns']
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['returns']).cumprod()
        signals['cumulative_strategy_returns'] = (1 + signals['strategy_returns']).cumprod()
        
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
        
        # Calculate Sortino ratio
        downside_returns = signals['strategy_returns'][signals['strategy_returns'] < 0]
        sortino_ratio = signals['strategy_returns'].mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else np.nan
        
        # Calculate Calmar ratio
        calmar_ratio = (total_return * 100) / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # Calculate maximum consecutive wins and losses
        wins = (signals['strategy_returns'] > 0).astype(int)
        losses = (signals['strategy_returns'] < 0).astype(int)
        
        win_streaks = wins * (wins.groupby((wins != wins.shift()).cumsum()).cumcount() + 1)
        loss_streaks = losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1)
        
        max_consecutive_wins = win_streaks.max()
        max_consecutive_losses = loss_streaks.max()
        
        # Calculate profit factor
        gross_profit = signals['strategy_returns'][signals['strategy_returns'] > 0].sum()
        gross_loss = abs(signals['strategy_returns'][signals['strategy_returns'] < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
        
        # Calculate average trade metrics
        winning_trades = signals['strategy_returns'][signals['strategy_returns'] > 0]
        losing_trades = signals['strategy_returns'][signals['strategy_returns'] < 0]
        
        avg_winning_trade = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_losing_trade = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Calculate market comparison
        market_return = signals['cumulative_returns'].iloc[-1] - 1
        outperformance = total_return - market_return
        
        # Calculate beta
        covariance = np.cov(signals['strategy_returns'].dropna(), signals['returns'].dropna())[0, 1]
        market_variance = np.var(signals['returns'].dropna())
        beta = covariance / market_variance if market_variance != 0 else np.nan
        
        # Calculate alpha
        risk_free_rate = 0.03 / 252  # Assuming 3% annual risk-free rate
        alpha = signals['strategy_returns'].mean() - risk_free_rate - beta * (signals['returns'].mean() - risk_free_rate)
        alpha = alpha * 252  # Annualize alpha
        
        # Prepare performance metrics
        performance = {
            "total_return": total_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "volatility": volatility,
            "profit_factor": profit_factor,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "avg_winning_trade": avg_winning_trade * 100,
            "avg_losing_trade": avg_losing_trade * 100,
            "num_trades": len(signals[signals['position'] != 0]),
            "market_return": market_return * 100,
            "outperformance": outperformance * 100,
            "beta": beta,
            "alpha": alpha * 100,
            "start_date": data.index[0].strftime("%Y-%m-%d"),
            "end_date": data.index[-1].strftime("%Y-%m-%d")
        }
        
        self.logger.info(f"Calculated performance metrics: {performance}")
        
        return performance
    
    def _calculate_annual_returns(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate annual returns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical data.
        signals : pd.DataFrame
            Trading signals.
            
        Returns:
        --------
        Dict[str, float]
            Annual returns.
        """
        self.logger.info("Calculating annual returns")
        
        # Resample to annual returns
        annual_returns = signals['strategy_returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        # Convert to dictionary
        annual_returns_dict = {str(year): return_value * 100 for year, return_value in annual_returns.items()}
        
        return annual_returns_dict
    
    def _calculate_monthly_returns(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate monthly returns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical data.
        signals : pd.DataFrame
            Trading signals.
            
        Returns:
        --------
        Dict[str, float]
            Monthly returns.
        """
        self.logger.info("Calculating monthly returns")
        
        # Resample to monthly returns
        monthly_returns = signals['strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Convert to dictionary
        monthly_returns_dict = {str(month)[:7]: return_value * 100 for month, return_value in monthly_returns.items()}
        
        return monthly_returns_dict
    
    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """
        Calculate overall validation score.
        
        Parameters:
        -----------
        validation_results : Dict[str, Any]
            Validation results.
            
        Returns:
        --------
        float
            Validation score.
        """
        self.logger.info("Calculating validation score")
        
        # Initialize score components
        full_history_score = 0
        different_periods_score = 0
        market_regimes_score = 0
        robustness_score = 0
        
        # Calculate full history score
        if validation_results["full_history"]["success"]:
            performance = validation_results["full_history"]["performance"]
            
            # Score based on total return
            total_return = performance.get("total_return", 0)
            if total_return > 50:
                full_history_score += 25
            elif total_return > 30:
                full_history_score += 20
            elif total_return > 10:
                full_history_score += 15
            elif total_return > 0:
                full_history_score += 10
            
            # Score based on Sharpe ratio
            sharpe_ratio = performance.get("sharpe_ratio", 0)
            if sharpe_ratio > 1.5:
                full_history_score += 25
            elif sharpe_ratio > 1.0:
                full_history_score += 20
            elif sharpe_ratio > 0.5:
                full_history_score += 15
            elif sharpe_ratio > 0:
                full_history_score += 10
        
        # Calculate different periods score
        if validation_results["different_periods"]["success"]:
            period_results = validation_results["different_periods"]["period_results"]
            
            # Count successful periods
            successful_periods = sum(1 for period_result in period_results.values() if period_result["success"])
            
            # Calculate average performance across periods
            total_returns = []
            sharpe_ratios = []
            
            for period_key, period_result in period_results.items():
                if period_result["success"]:
                    performance = period_result["performance"]
                    total_returns.append(performance.get("total_return", 0))
                    sharpe_ratios.append(performance.get("sharpe_ratio", 0))
            
            # Score based on consistency across periods
            if successful_periods == len(period_results):
                different_periods_score += 10
            elif successful_periods >= len(period_results) * 0.75:
                different_periods_score += 7
            elif successful_periods >= len(period_results) * 0.5:
                different_periods_score += 5
            
            # Score based on average total return
            avg_total_return = np.mean(total_returns) if total_returns else 0
            if avg_total_return > 30:
                different_periods_score += 10
            elif avg_total_return > 20:
                different_periods_score += 8
            elif avg_total_return > 10:
                different_periods_score += 6
            elif avg_total_return > 0:
                different_periods_score += 4
            
            # Score based on average Sharpe ratio
            avg_sharpe_ratio = np.mean(sharpe_ratios) if sharpe_ratios else 0
            if avg_sharpe_ratio > 1.5:
                different_periods_score += 10
            elif avg_sharpe_ratio > 1.0:
                different_periods_score += 8
            elif avg_sharpe_ratio > 0.5:
                different_periods_score += 6
            elif avg_sharpe_ratio > 0:
                different_periods_score += 4
        
        # Calculate market regimes score
        if validation_results["market_regimes"]["success"]:
            regime_results = validation_results["market_regimes"]["regime_results"]
            
            # Count successful regimes
            successful_regimes = sum(1 for regime_result in regime_results.values() if regime_result["success"])
            
            # Calculate performance across regimes
            regime_performance = {}
            
            for regime_key, regime_result in regime_results.items():
                if regime_result["success"]:
                    performance = regime_result["performance"]
                    regime_performance[regime_key] = {
                        "total_return": performance.get("total_return", 0),
                        "sharpe_ratio": performance.get("sharpe_ratio", 0)
                    }
            
            # Score based on consistency across regimes
            if successful_regimes == len(regime_results):
                market_regimes_score += 10
            elif successful_regimes >= len(regime_results) * 0.75:
                market_regimes_score += 7
            elif successful_regimes >= len(regime_results) * 0.5:
                market_regimes_score += 5
            
            # Score based on performance in bull market
            if "bull_market" in regime_performance:
                bull_return = regime_performance["bull_market"]["total_return"]
                if bull_return > 30:
                    market_regimes_score += 5
                elif bull_return > 20:
                    market_regimes_score += 4
                elif bull_return > 10:
                    market_regimes_score += 3
                elif bull_return > 0:
                    market_regimes_score += 2
            
            # Score based on performance in bear market
            if "bear_market" in regime_performance:
                bear_return = regime_performance["bear_market"]["total_return"]
                if bear_return > 0:
                    market_regimes_score += 10  # Positive returns in bear market is excellent
                elif bear_return > -10:
                    market_regimes_score += 5
                elif bear_return > -20:
                    market_regimes_score += 3
            
            # Score based on performance in sideways market
            if "sideways_market" in regime_performance:
                sideways_return = regime_performance["sideways_market"]["total_return"]
                if sideways_return > 10:
                    market_regimes_score += 5
                elif sideways_return > 5:
                    market_regimes_score += 4
                elif sideways_return > 0:
                    market_regimes_score += 3
        
        # Calculate robustness score
        if validation_results["robustness"]["success"]:
            # Score based on mean total return
            total_return_mean = validation_results["robustness"].get("total_return_mean", 0)
            if total_return_mean > 30:
                robustness_score += 5
            elif total_return_mean > 20:
                robustness_score += 4
            elif total_return_mean > 10:
                robustness_score += 3
            elif total_return_mean > 0:
                robustness_score += 2
            
            # Score based on 5th percentile total return
            total_return_5th = validation_results["robustness"].get("total_return_5th", 0)
            if total_return_5th > 0:
                robustness_score += 10  # Positive returns even in worst 5% of simulations is excellent
            elif total_return_5th > -10:
                robustness_score += 5
            elif total_return_5th > -20:
                robustness_score += 3
            
            # Score based on Sharpe ratio stability
            sharpe_ratio_mean = validation_results["robustness"].get("sharpe_ratio_mean", 0)
            sharpe_ratio_std = validation_results["robustness"].get("sharpe_ratio_std", 0)
            
            if sharpe_ratio_mean > 0 and sharpe_ratio_std < 0.5:
                robustness_score += 5  # Stable positive Sharpe ratio
            elif sharpe_ratio_mean > 0:
                robustness_score += 3
            
            # Score based on drawdown stability
            max_drawdown_mean = validation_results["robustness"].get("max_drawdown_mean", 0)
            
            if max_drawdown_mean > -20:
                robustness_score += 5
            elif max_drawdown_mean > -30:
                robustness_score += 3
            elif max_drawdown_mean > -40:
                robustness_score += 1
        
        # Calculate total score (out of 100)
        total_score = full_history_score + different_periods_score + market_regimes_score + robustness_score
        
        # Normalize to 0-100 scale
        normalized_score = min(100, total_score)
        
        self.logger.info(f"Calculated validation score: {normalized_score}")
        
        return normalized_score
    
    def _generate_validation_report(self, validation_results: Dict[str, Any], validation_score: float) -> str:
        """
        Generate validation report.
        
        Parameters:
        -----------
        validation_results : Dict[str, Any]
            Validation results.
        validation_score : float
            Validation score.
            
        Returns:
        --------
        str
            Validation report.
        """
        self.logger.info("Generating validation report")
        
        report = []
        
        # Add report header
        report.append("# Strategy Validation Report")
        report.append("")
        report.append(f"## Overall Validation Score: {validation_score:.2f}/100")
        report.append("")
        
        # Add score interpretation
        if validation_score >= 80:
            report.append("**Excellent Strategy**: This strategy demonstrates exceptional performance across various market conditions and time periods.")
        elif validation_score >= 60:
            report.append("**Good Strategy**: This strategy performs well in most market conditions and time periods.")
        elif validation_score >= 40:
            report.append("**Average Strategy**: This strategy shows acceptable performance but may struggle in certain market conditions.")
        else:
            report.append("**Below Average Strategy**: This strategy needs improvement as it doesn't perform consistently across different market conditions.")
        
        report.append("")
        
        # Add full history section
        report.append("## Full History Performance")
        
        if validation_results["full_history"]["success"]:
            performance = validation_results["full_history"]["performance"]
            start_date = validation_results["full_history"]["start_date"]
            end_date = validation_results["full_history"]["end_date"]
            
            report.append(f"- **Period**: {start_date} to {end_date}")
            report.append(f"- **Total Return**: {performance['total_return']:.2f}%")
            report.append(f"- **Sharpe Ratio**: {performance['sharpe_ratio']:.2f}")
            report.append(f"- **Maximum Drawdown**: {performance['max_drawdown']:.2f}%")
            report.append(f"- **Win Rate**: {performance['win_rate']:.2f}%")
            report.append(f"- **Market Return**: {performance['market_return']:.2f}%")
            report.append(f"- **Outperformance**: {performance['outperformance']:.2f}%")
            
            # Add annual returns
            annual_returns = validation_results["full_history"]["annual_returns"]
            
            if annual_returns:
                report.append("")
                report.append("### Annual Returns")
                
                for year, return_value in sorted(annual_returns.items()):
                    report.append(f"- **{year}**: {return_value:.2f}%")
        else:
            report.append(f"Failed to validate on full history: {validation_results['full_history'].get('error', 'Unknown error')}")
        
        report.append("")
        
        # Add different periods section
        report.append("## Performance Across Different Time Periods")
        
        if validation_results["different_periods"]["success"]:
            period_results = validation_results["different_periods"]["period_results"]
            
            for period_key, period_name in {
                "1y": "Last 1 Year",
                "2y": "Last 2 Years",
                "5y": "Last 5 Years",
                "10y": "Last 10 Years"
            }.items():
                if period_key in period_results:
                    period_result = period_results[period_key]
                    
                    report.append(f"### {period_name}")
                    
                    if period_result["success"]:
                        performance = period_result["performance"]
                        start_date = period_result["start_date"]
                        end_date = period_result["end_date"]
                        
                        report.append(f"- **Period**: {start_date} to {end_date}")
                        report.append(f"- **Total Return**: {performance['total_return']:.2f}%")
                        report.append(f"- **Sharpe Ratio**: {performance['sharpe_ratio']:.2f}")
                        report.append(f"- **Maximum Drawdown**: {performance['max_drawdown']:.2f}%")
                        report.append(f"- **Market Return**: {performance['market_return']:.2f}%")
                    else:
                        report.append(f"Failed to validate: {period_result.get('error', 'Unknown error')}")
                    
                    report.append("")
        else:
            report.append(f"Failed to validate across different time periods: {validation_results['different_periods'].get('error', 'Unknown error')}")
            report.append("")
        
        # Add market regimes section
        report.append("## Performance Across Market Regimes")
        
        if validation_results["market_regimes"]["success"]:
            regime_results = validation_results["market_regimes"]["regime_results"]
            
            for regime_key, regime_name in {
                "bull_market": "Bull Market",
                "bear_market": "Bear Market",
                "sideways_market": "Sideways Market",
                "volatile_market": "Volatile Market"
            }.items():
                if regime_key in regime_results:
                    regime_result = regime_results[regime_key]
                    
                    report.append(f"### {regime_name}")
                    
                    if regime_result["success"]:
                        performance = regime_result["performance"]
                        start_date = regime_result["start_date"]
                        end_date = regime_result["end_date"]
                        
                        report.append(f"- **Period**: {start_date} to {end_date}")
                        report.append(f"- **Total Return**: {performance['total_return']:.2f}%")
                        report.append(f"- **Sharpe Ratio**: {performance['sharpe_ratio']:.2f}")
                        report.append(f"- **Maximum Drawdown**: {performance['max_drawdown']:.2f}%")
                        report.append(f"- **Market Return**: {performance['market_return']:.2f}%")
                    else:
                        report.append(f"Failed to validate: {regime_result.get('error', 'Unknown error')}")
                    
                    report.append("")
        else:
            report.append(f"Failed to validate across market regimes: {validation_results['market_regimes'].get('error', 'Unknown error')}")
            report.append("")
        
        # Add robustness section
        report.append("## Strategy Robustness")
        
        if validation_results["robustness"]["success"]:
            robustness = validation_results["robustness"]
            
            report.append(f"- **Number of Simulations**: {robustness['num_simulations']}")
            report.append(f"- **Mean Total Return**: {robustness['total_return_mean']:.2f}%")
            report.append(f"- **Standard Deviation of Total Return**: {robustness['total_return_std']:.2f}%")
            report.append(f"- **5th Percentile Total Return**: {robustness['total_return_5th']:.2f}%")
            report.append(f"- **95th Percentile Total Return**: {robustness['total_return_95th']:.2f}%")
            report.append(f"- **Mean Sharpe Ratio**: {robustness['sharpe_ratio_mean']:.2f}")
            report.append(f"- **5th Percentile Sharpe Ratio**: {robustness['sharpe_ratio_5th']:.2f}")
            report.append(f"- **Mean Maximum Drawdown**: {robustness['max_drawdown_mean']:.2f}%")
            report.append(f"- **5th Percentile Maximum Drawdown**: {robustness['max_drawdown_5th']:.2f}%")
        else:
            report.append(f"Failed to validate robustness: {validation_results['robustness'].get('error', 'Unknown error')}")
        
        report.append("")
        
        # Add conclusion
        report.append("## Conclusion")
        
        if validation_score >= 80:
            report.append("This strategy demonstrates excellent performance across various market conditions and time periods. It has shown consistent positive returns, good risk-adjusted performance, and robustness to different market regimes. The strategy is recommended for implementation.")
        elif validation_score >= 60:
            report.append("This strategy demonstrates good performance across most market conditions and time periods. While it performs well overall, there are some areas where it could be improved. The strategy is suitable for implementation with regular monitoring.")
        elif validation_score >= 40:
            report.append("This strategy demonstrates acceptable performance but may struggle in certain market conditions. It should be used with caution and may require further optimization before implementation.")
        else:
            report.append("This strategy needs significant improvement as it doesn't perform consistently across different market conditions. It is not recommended for implementation in its current form.")
        
        # Join report lines
        report_text = "\n".join(report)
        
        self.logger.info("Generated validation report")
        
        return report_text
    
    def _generate_performance_charts(self, validation_results: Dict[str, Any], ticker: str) -> List[str]:
        """
        Generate performance charts.
        
        Parameters:
        -----------
        validation_results : Dict[str, Any]
            Validation results.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        List[str]
            Paths to generated charts.
        """
        self.logger.info("Generating performance charts")
        
        chart_paths = []
        
        # Create charts directory if it doesn't exist
        charts_dir = "/home/ubuntu/gemma_advanced/charts"
        os.makedirs(charts_dir, exist_ok=True)
        
        try:
            # Generate full history performance chart
            if validation_results["full_history"]["success"]:
                # Get historical data
                data = self._get_historical_data(ticker, period="max")
                
                if data is not None and len(data) > 0:
                    # Apply strategy to historical data
                    signals = self._generate_signals(data, {"short_ma": 20, "long_ma": 50})
                    
                    # Create figure
                    plt.figure(figsize=(12, 8))
                    
                    # Plot cumulative returns
                    plt.plot(signals.index, signals['cumulative_returns'], label='Buy and Hold')
                    plt.plot(signals.index, signals['cumulative_strategy_returns'], label='Strategy')
                    
                    # Add labels and title
                    plt.xlabel('Date')
                    plt.ylabel('Cumulative Returns')
                    plt.title(f'{ticker} Strategy Performance')
                    plt.legend()
                    plt.grid(True)
                    
                    # Save figure
                    chart_path = os.path.join(charts_dir, f"{ticker}_performance.png")
                    plt.savefig(chart_path)
                    plt.close()
                    
                    chart_paths.append(chart_path)
                    
                    self.logger.info(f"Generated performance chart: {chart_path}")
                    
                    # Generate drawdown chart
                    plt.figure(figsize=(12, 8))
                    
                    # Calculate drawdowns
                    cumulative_max = signals['cumulative_strategy_returns'].cummax()
                    drawdown = (signals['cumulative_strategy_returns'] - cumulative_max) / cumulative_max * 100
                    
                    # Plot drawdowns
                    plt.plot(signals.index, drawdown)
                    
                    # Add labels and title
                    plt.xlabel('Date')
                    plt.ylabel('Drawdown (%)')
                    plt.title(f'{ticker} Strategy Drawdown')
                    plt.grid(True)
                    
                    # Save figure
                    chart_path = os.path.join(charts_dir, f"{ticker}_drawdown.png")
                    plt.savefig(chart_path)
                    plt.close()
                    
                    chart_paths.append(chart_path)
                    
                    self.logger.info(f"Generated drawdown chart: {chart_path}")
                    
                    # Generate annual returns chart
                    annual_returns = validation_results["full_history"]["annual_returns"]
                    
                    if annual_returns:
                        plt.figure(figsize=(12, 8))
                        
                        # Sort years
                        years = sorted(annual_returns.keys())
                        returns = [annual_returns[year] for year in years]
                        
                        # Plot annual returns
                        plt.bar(years, returns)
                        
                        # Add labels and title
                        plt.xlabel('Year')
                        plt.ylabel('Return (%)')
                        plt.title(f'{ticker} Annual Returns')
                        plt.grid(True)
                        
                        # Rotate x-axis labels
                        plt.xticks(rotation=45)
                        
                        # Save figure
                        chart_path = os.path.join(charts_dir, f"{ticker}_annual_returns.png")
                        plt.savefig(chart_path)
                        plt.close()
                        
                        chart_paths.append(chart_path)
                        
                        self.logger.info(f"Generated annual returns chart: {chart_path}")
            
            # Generate robustness chart
            if validation_results["robustness"]["success"]:
                # Get robustness data
                total_return_mean = validation_results["robustness"]["total_return_mean"]
                total_return_std = validation_results["robustness"]["total_return_std"]
                total_return_5th = validation_results["robustness"]["total_return_5th"]
                total_return_95th = validation_results["robustness"]["total_return_95th"]
                
                # Create figure
                plt.figure(figsize=(12, 8))
                
                # Plot robustness data
                plt.errorbar(['Total Return'], [total_return_mean], yerr=[total_return_std], fmt='o', capsize=10)
                plt.plot(['Total Return'], [total_return_5th], 'v', label='5th Percentile')
                plt.plot(['Total Return'], [total_return_95th], '^', label='95th Percentile')
                
                # Add labels and title
                plt.ylabel('Return (%)')
                plt.title(f'{ticker} Strategy Robustness')
                plt.legend()
                plt.grid(True)
                
                # Save figure
                chart_path = os.path.join(charts_dir, f"{ticker}_robustness.png")
                plt.savefig(chart_path)
                plt.close()
                
                chart_paths.append(chart_path)
                
                self.logger.info(f"Generated robustness chart: {chart_path}")
        
        except Exception as e:
            self.logger.error(f"Error generating performance charts: {e}")
        
        self.logger.info(f"Generated {len(chart_paths)} performance charts")
        
        return chart_paths


def validate_strategy(strategy_file: str) -> Dict[str, Any]:
    """
    Validate a strategy from a JSON file.
    
    Parameters:
    -----------
    strategy_file : str
        Path to strategy JSON file.
        
    Returns:
    --------
    Dict[str, Any]
        Validation results.
    """
    logger.info(f"Validating strategy from file: {strategy_file}")
    
    try:
        # Load strategy from file
        with open(strategy_file, 'r') as f:
            strategy_data = json.load(f)
        
        # Extract strategy and ticker
        strategy = strategy_data.get("strategy", {})
        ticker = strategy_data.get("ticker", "AAPL")
        
        # Create validator
        validator = StrategyValidator()
        
        # Validate strategy
        validation_result = validator.validate_strategy(strategy, ticker)
        
        # Save validation result
        output_file = strategy_file.replace(".json", "_validation.json")
        with open(output_file, 'w') as f:
            json.dump(validation_result, f, indent=2)
        
        # Save validation report
        report_file = strategy_file.replace(".json", "_validation_report.md")
        with open(report_file, 'w') as f:
            f.write(validation_result["validation_report"])
        
        logger.info(f"Saved validation result to {output_file}")
        logger.info(f"Saved validation report to {report_file}")
        
        return validation_result
    
    except Exception as e:
        logger.error(f"Error validating strategy: {e}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    logger.info("Starting strategy validation")
    
    # Validate optimized strategy
    strategy_file = "/home/ubuntu/gemma_advanced/aapl_optimized_strategy.json"
    
    if os.path.exists(strategy_file):
        result = validate_strategy(strategy_file)
        
        if "validation_score" in result:
            logger.info(f"Validation score: {result['validation_score']}")
            print(f"SUCCESS: Validated strategy with score {result['validation_score']:.2f}/100")
            print(f"Validation report saved to {strategy_file.replace('.json', '_validation_report.md')}")
        else:
            logger.error("Failed to validate strategy")
            print(f"FAILURE: Could not validate strategy: {result.get('error', 'Unknown error')}")
    else:
        logger.error(f"Strategy file not found: {strategy_file}")
        print(f"FAILURE: Strategy file not found: {strategy_file}")
