"""
Strategy Builder Module for Gemma Advanced Trading System.

This module provides functionality for building, testing, and optimizing trading strategies.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import system components
from indicators.trend_indicators import (
    simple_moving_average, exponential_moving_average, 
    moving_average_convergence_divergence, bollinger_bands
)
from indicators.momentum_indicators import (
    relative_strength_index, stochastic_oscillator,
    commodity_channel_index, williams_percent_r
)
from indicators.volatility_indicators import (
    average_true_range, standard_deviation
)
from indicators.volume_indicators import (
    on_balance_volume, volume_weighted_average_price
)
from risk_management.core import RiskManager
from risk_management.utils import calculate_position_size
import data_access

# Configure logging
logger = logging.getLogger("GemmaTrading.StrategyBuilder")

class Strategy:
    """Base class for trading strategies."""
    
    def __init__(self, name, description=None):
        """
        Initialize a trading strategy.
        
        Parameters:
        -----------
        name : str
            Strategy name
        description : str, optional
            Strategy description
        """
        self.name = name
        self.description = description or f"Strategy: {name}"
        self.parameters = {}
        self.signals = None
        self.trades = []
        self.performance = {}
        self.logger = logging.getLogger(f"GemmaTrading.Strategy.{name}")
    
    def set_parameters(self, parameters):
        """
        Set strategy parameters.
        
        Parameters:
        -----------
        parameters : dict
            Strategy parameters
        """
        self.parameters = parameters
        self.logger.info(f"Set parameters for strategy {self.name}: {parameters}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
            
        Returns:
        --------
        pandas.DataFrame
            Data with signals
        """
        self.logger.info(f"Generating signals for strategy {self.name}")
        # This is a base method that should be overridden by subclasses
        return data
    
    def backtest(self, data, initial_capital=10000.0):
        """
        Backtest the strategy on historical data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
        initial_capital : float, optional
            Initial capital for backtesting
            
        Returns:
        --------
        dict
            Backtest results
        """
        self.logger.info(f"Backtesting strategy {self.name}")
        
        # Generate signals if not already done
        if self.signals is None or len(self.signals) != len(data):
            self.signals = self.generate_signals(data)
        
        # Initialize backtest variables
        position = 0  # 0: no position, 1: long, -1: short
        capital = initial_capital
        self.trades = []
        
        # Iterate through the data
        for i in range(1, len(self.signals)):
            current_date = self.signals.index[i]
            current_price = self.signals['Close'].iloc[i]
            signal = self.signals['Signal'].iloc[i] if 'Signal' in self.signals.columns else 0
            
            # Check for entry signals
            if position == 0 and signal == 1:  # Enter long position
                # Calculate position size
                if 'Stop_Loss' in self.signals.columns and not pd.isna(self.signals['Stop_Loss'].iloc[i]):
                    stop_loss = self.signals['Stop_Loss'].iloc[i]
                    risk_amount = initial_capital * self.parameters.get('risk_per_trade', 0.02)
                    risk_per_share = current_price - stop_loss
                    position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
                else:
                    # Default position sizing if stop loss is not defined
                    position_size = (capital * self.parameters.get('position_size', 0.1)) / current_price
                
                position_value = position_size * current_price
                
                # Record trade entry
                entry_trade = {
                    'type': 'entry',
                    'direction': 'long',
                    'date': current_date,
                    'price': current_price,
                    'position_size': position_size,
                    'position_value': position_value
                }
                
                # Add stop loss and take profit if available
                if 'Stop_Loss' in self.signals.columns and not pd.isna(self.signals['Stop_Loss'].iloc[i]):
                    entry_trade['stop_loss'] = self.signals['Stop_Loss'].iloc[i]
                
                if 'Take_Profit' in self.signals.columns and not pd.isna(self.signals['Take_Profit'].iloc[i]):
                    entry_trade['take_profit'] = self.signals['Take_Profit'].iloc[i]
                
                self.trades.append(entry_trade)
                position = 1
                
            elif position == 0 and signal == -1:  # Enter short position (if supported)
                if self.parameters.get('allow_short', False):
                    # Calculate position size
                    if 'Stop_Loss' in self.signals.columns and not pd.isna(self.signals['Stop_Loss'].iloc[i]):
                        stop_loss = self.signals['Stop_Loss'].iloc[i]
                        risk_amount = initial_capital * self.parameters.get('risk_per_trade', 0.02)
                        risk_per_share = stop_loss - current_price
                        position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
                    else:
                        # Default position sizing if stop loss is not defined
                        position_size = (capital * self.parameters.get('position_size', 0.1)) / current_price
                    
                    position_value = position_size * current_price
                    
                    # Record trade entry
                    entry_trade = {
                        'type': 'entry',
                        'direction': 'short',
                        'date': current_date,
                        'price': current_price,
                        'position_size': position_size,
                        'position_value': position_value
                    }
                    
                    # Add stop loss and take profit if available
                    if 'Stop_Loss' in self.signals.columns and not pd.isna(self.signals['Stop_Loss'].iloc[i]):
                        entry_trade['stop_loss'] = self.signals['Stop_Loss'].iloc[i]
                    
                    if 'Take_Profit' in self.signals.columns and not pd.isna(self.signals['Take_Profit'].iloc[i]):
                        entry_trade['take_profit'] = self.signals['Take_Profit'].iloc[i]
                    
                    self.trades.append(entry_trade)
                    position = -1
            
            # Check for exit conditions
            elif position != 0:
                entry_trade = next((t for t in reversed(self.trades) if t['type'] == 'entry'), None)
                if entry_trade:
                    exit_triggered = False
                    exit_reason = None
                    exit_price = current_price
                    
                    # Check if stop loss was hit
                    if 'stop_loss' in entry_trade:
                        if (position == 1 and current_price <= entry_trade['stop_loss']) or \
                           (position == -1 and current_price >= entry_trade['stop_loss']):
                            exit_triggered = True
                            exit_reason = 'stop_loss'
                            exit_price = entry_trade['stop_loss']
                    
                    # Check if take profit was hit
                    if not exit_triggered and 'take_profit' in entry_trade:
                        if (position == 1 and current_price >= entry_trade['take_profit']) or \
                           (position == -1 and current_price <= entry_trade['take_profit']):
                            exit_triggered = True
                            exit_reason = 'take_profit'
                            exit_price = entry_trade['take_profit']
                    
                    # Check for exit signal
                    if not exit_triggered and ((position == 1 and signal == -1) or (position == -1 and signal == 1)):
                        exit_triggered = True
                        exit_reason = 'signal'
                    
                    # Execute exit if triggered
                    if exit_triggered:
                        position_size = entry_trade['position_size']
                        
                        # Calculate profit/loss
                        if position == 1:  # Long position
                            profit_loss = position_size * (exit_price - entry_trade['price'])
                        else:  # Short position
                            profit_loss = position_size * (entry_trade['price'] - exit_price)
                        
                        # Record trade exit
                        exit_trade = {
                            'type': 'exit',
                            'date': current_date,
                            'price': exit_price,
                            'position_size': position_size,
                            'position_value': position_size * exit_price,
                            'exit_reason': exit_reason,
                            'profit_loss': profit_loss,
                            'entry_trade_index': len(self.trades) - 1
                        }
                        
                        self.trades.append(exit_trade)
                        capital += profit_loss
                        position = 0
        
        # Close any open positions at the end of the backtest
        if position != 0:
            entry_trade = next((t for t in reversed(self.trades) if t['type'] == 'entry'), None)
            if entry_trade:
                position_size = entry_trade['position_size']
                last_price = self.signals['Close'].iloc[-1]
                
                # Calculate profit/loss
                if position == 1:  # Long position
                    profit_loss = position_size * (last_price - entry_trade['price'])
                else:  # Short position
                    profit_loss = position_size * (entry_trade['price'] - last_price)
                
                # Record trade exit
                exit_trade = {
                    'type': 'exit',
                    'date': self.signals.index[-1],
                    'price': last_price,
                    'position_size': position_size,
                    'position_value': position_size * last_price,
                    'exit_reason': 'end_of_data',
                    'profit_loss': profit_loss,
                    'entry_trade_index': len(self.trades) - 1
                }
                
                self.trades.append(exit_trade)
                capital += profit_loss
        
        # Calculate performance metrics
        self._calculate_performance_metrics(initial_capital, capital)
        
        return {
            'signals': self.signals,
            'trades': self.trades,
            'performance': self.performance
        }
    
    def optimize(self, data, parameter_grid, initial_capital=10000.0, metric='sharpe_ratio'):
        """
        Optimize strategy parameters using grid search.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
        parameter_grid : dict
            Dictionary of parameter names and lists of values to try
        initial_capital : float, optional
            Initial capital for backtesting
        metric : str, optional
            Performance metric to optimize for
            
        Returns:
        --------
        dict
            Optimization results
        """
        self.logger.info(f"Optimizing strategy {self.name}")
        
        # Generate all parameter combinations
        import itertools
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Track best parameters and performance
        best_params = None
        best_performance = None
        best_metric_value = float('-inf')
        
        # Test each parameter combination
        results = []
        for combo in param_combinations:
            # Create parameter dictionary
            params = dict(zip(param_names, combo))
            
            # Set parameters and backtest
            self.set_parameters(params)
            backtest_result = self.backtest(data, initial_capital)
            
            # Extract metric value
            metric_value = backtest_result['performance'].get(metric, float('-inf'))
            
            # Track result
            result = {
                'parameters': params,
                'performance': backtest_result['performance']
            }
            results.append(result)
            
            # Update best parameters if this combination is better
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_params = params
                best_performance = backtest_result['performance']
        
        # Set the best parameters
        if best_params:
            self.set_parameters(best_params)
            self.performance = best_performance
        
        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'all_results': results
        }
    
    def _calculate_performance_metrics(self, initial_capital, final_capital):
        """
        Calculate performance metrics based on trades.
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital
        final_capital : float
            Final capital
        """
        # Filter for completed trades (entry and exit pairs)
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
                        'entry_price': entry['price'],
                        'exit_price': exit['price'],
                        'position_size': entry['position_size'],
                        'direction': entry.get('direction', 'long'),
                        'profit_loss': exit['profit_loss'],
                        'exit_reason': exit['exit_reason'],
                        'duration': (exit['date'] - entry['date']).days
                    })
        
        # Calculate performance metrics
        total_trades = len(completed_trades)
        
        if total_trades > 0:
            winning_trades = sum(1 for trade in completed_trades if trade['profit_loss'] > 0)
            losing_trades = sum(1 for trade in completed_trades if trade['profit_loss'] <= 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = sum(trade['profit_loss'] for trade in completed_trades if trade['profit_loss'] > 0)
            total_loss = sum(trade['profit_loss'] for trade in completed_trades if trade['profit_loss'] <= 0)
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            
            average_win = total_profit / winning_trades if winning_trades > 0 else 0
            average_loss = total_loss / losing_trades if losing_trades > 0 else 0
            
            total_return = (final_capital - initial_capital) / initial_capital * 100
            
            # Calculate drawdown
            equity_curve = [initial_capital]
            for trade in completed_trades:
                equity_curve.append(equity_curve[-1] + trade['profit_loss'])
            
            peak = equity_curve[0]
            max_drawdown = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0%)
            if len(completed_trades) > 1:
                returns = [(trade['profit_loss'] / initial_capital) for trade in completed_trades]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return * np.sqrt(252 / len(completed_trades)) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate Sortino ratio (using only negative returns for denominator)
            if len(completed_trades) > 1:
                negative_returns = [(trade['profit_loss'] / initial_capital) for trade in completed_trades if trade['profit_loss'] < 0]
                if negative_returns:
                    downside_deviation = np.std(negative_returns)
                    sortino_ratio = avg_return / downside_deviation * np.sqrt(252 / len(completed_trades)) if downside_deviation > 0 else 0
                else:
                    sortino_ratio = float('inf')  # No losing trades
            else:
                sortino_ratio = 0
            
            # Calculate Calmar ratio
            calmar_ratio = (total_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else float('inf')
            
            self.performance = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_win': average_win,
                'average_loss': average_loss,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'final_capital': final_capital
            }
        else:
            self.performance = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_win': 0,
                'average_loss': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'final_capital': initial_capital
            }


class SwingTradingStrategy(Strategy):
    """Swing Trading Strategy implementation."""
    
    def __init__(self, name="Swing Trading Strategy", description=None):
        """
        Initialize a swing trading strategy.
        
        Parameters:
        -----------
        name : str, optional
            Strategy name
        description : str, optional
            Strategy description
        """
        super().__init__(name, description)
        
        # Set default parameters
        self.set_parameters({
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'sma_short_period': 20,
            'sma_long_period': 50,
            'atr_period': 14,
            'risk_per_trade': 0.02,  # 2% risk per trade
            'risk_reward_ratio': 2.0,  # 1:2 risk-reward ratio
            'stop_loss_atr_multiple': 2.0,  # Stop loss at 2x ATR
            'allow_short': False  # Whether to allow short positions
        })
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the swing trading strategy.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
            
        Returns:
        --------
        pandas.DataFrame
            Data with signals
        """
        self.logger.info(f"Generating signals for {self.name}")
        
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Calculate indicators
        df['SMA_Short'] = simple_moving_average(df['Close'], period=self.parameters['sma_short_period'])
        df['SMA_Long'] = simple_moving_average(df['Close'], period=self.parameters['sma_long_period'])
        df['RSI'] = relative_strength_index(df['Close'], period=self.parameters['rsi_period'])
        df['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], period=self.parameters['atr_period'])
        
        # Generate strategy signals
        df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Buy signal: RSI crosses above oversold level AND short SMA crosses above long SMA
        buy_condition = (
            (df['RSI'] > self.parameters['rsi_oversold']) & 
            (df['RSI'].shift(1) <= self.parameters['rsi_oversold']) &
            (df['SMA_Short'] > df['SMA_Long']) & 
            (df['SMA_Short'].shift(1) <= df['SMA_Long'].shift(1))
        )
        df.loc[buy_condition, 'Signal'] = 1
        
        # Sell signal: RSI crosses below overbought level OR short SMA crosses below long SMA
        sell_condition = (
            ((df['RSI'] < self.parameters['rsi_overbought']) & 
             (df['RSI'].shift(1) >= self.parameters['rsi_overbought'])) |
            ((df['SMA_Short'] < df['SMA_Long']) & 
             (df['SMA_Short'].shift(1) >= df['SMA_Long'].shift(1)))
        )
        df.loc[sell_condition, 'Signal'] = -1
        
        # Calculate stop loss and take profit levels for each buy signal
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan
        
        for i in range(1, len(df)):
            if df['Signal'].iloc[i] == 1:  # Buy signal
                entry_price = df['Close'].iloc[i]
                atr_value = df['ATR'].iloc[i]
                
                # Set stop loss at entry_price - (ATR * multiple)
                stop_loss = entry_price - (atr_value * self.parameters['stop_loss_atr_multiple'])
                
                # Set take profit based on risk-reward ratio
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * self.parameters['risk_reward_ratio'])
                
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Take_Profit'] = take_profit
        
        self.signals = df
        return df


class DayTradingStrategy(Strategy):
    """Day Trading Strategy implementation."""
    
    def __init__(self, name="Day Trading Strategy", description=None):
        """
        Initialize a day trading strategy.
        
        Parameters:
        -----------
        name : str, optional
            Strategy name
        description : str, optional
            Strategy description
        """
        super().__init__(name, description)
        
        # Set default parameters
        self.set_parameters({
            'ema_short_period': 9,
            'ema_medium_period': 21,
            'ema_long_period': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_period': 14,
            'risk_per_trade': 0.01,  # 1% risk per trade
            'risk_reward_ratio': 1.5,  # 1:1.5 risk-reward ratio
            'stop_loss_atr_multiple': 1.5,  # Stop loss at 1.5x ATR
            'allow_short': True  # Allow short positions
        })
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the day trading strategy.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
            
        Returns:
        --------
        pandas.DataFrame
            Data with signals
        """
        self.logger.info(f"Generating signals for {self.name}")
        
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Calculate indicators
        df['EMA_Short'] = exponential_moving_average(df['Close'], period=self.parameters['ema_short_period'])
        df['EMA_Medium'] = exponential_moving_average(df['Close'], period=self.parameters['ema_medium_period'])
        df['EMA_Long'] = exponential_moving_average(df['Close'], period=self.parameters['ema_long_period'])
        df['RSI'] = relative_strength_index(df['Close'], period=self.parameters['rsi_period'])
        df['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], period=self.parameters['atr_period'])
        
        # Generate strategy signals
        df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Buy signal: EMA alignment (Short > Medium > Long) AND RSI < 70
        buy_condition = (
            (df['EMA_Short'] > df['EMA_Medium']) & 
            (df['EMA_Medium'] > df['EMA_Long']) & 
            (df['RSI'] < self.parameters['rsi_overbought'])
        )
        
        # Additional condition: EMA Short crosses above EMA Medium
        buy_cross = (
            (df['EMA_Short'] > df['EMA_Medium']) & 
            (df['EMA_Short'].shift(1) <= df['EMA_Medium'].shift(1))
        )
        
        df.loc[buy_condition & buy_cross, 'Signal'] = 1
        
        # Sell signal: EMA alignment (Short < Medium < Long) OR RSI > 70
        sell_condition = (
            (df['EMA_Short'] < df['EMA_Medium']) & 
            (df['EMA_Medium'] < df['EMA_Long'])
        ) | (df['RSI'] > self.parameters['rsi_overbought'])
        
        # Additional condition: EMA Short crosses below EMA Medium
        sell_cross = (
            (df['EMA_Short'] < df['EMA_Medium']) & 
            (df['EMA_Short'].shift(1) >= df['EMA_Medium'].shift(1))
        )
        
        df.loc[sell_condition & sell_cross, 'Signal'] = -1
        
        # Calculate stop loss and take profit levels for each signal
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan
        
        for i in range(1, len(df)):
            if df['Signal'].iloc[i] == 1:  # Buy signal
                entry_price = df['Close'].iloc[i]
                atr_value = df['ATR'].iloc[i]
                
                # Set stop loss at entry_price - (ATR * multiple)
                stop_loss = entry_price - (atr_value * self.parameters['stop_loss_atr_multiple'])
                
                # Set take profit based on risk-reward ratio
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * self.parameters['risk_reward_ratio'])
                
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Take_Profit'] = take_profit
                
            elif df['Signal'].iloc[i] == -1 and self.parameters['allow_short']:  # Sell signal (for short)
                entry_price = df['Close'].iloc[i]
                atr_value = df['ATR'].iloc[i]
                
                # Set stop loss at entry_price + (ATR * multiple)
                stop_loss = entry_price + (atr_value * self.parameters['stop_loss_atr_multiple'])
                
                # Set take profit based on risk-reward ratio
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * self.parameters['risk_reward_ratio'])
                
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Take_Profit'] = take_profit
        
        self.signals = df
        return df


class StrategyBuilder:
    """
    Strategy Builder for creating and managing trading strategies.
    """
    
    def __init__(self):
        """Initialize the strategy builder."""
        self.strategies = {}
        self.logger = logging.getLogger("GemmaTrading.StrategyBuilder")
    
    def create_strategy(self, strategy_type, name=None, description=None, parameters=None):
        """
        Create a new trading strategy.
        
        Parameters:
        -----------
        strategy_type : str
            Type of strategy to create ("swing", "day", "trend", "breakout", etc.)
        name : str, optional
            Strategy name
        description : str, optional
            Strategy description
        parameters : dict, optional
            Strategy parameters
            
        Returns:
        --------
        Strategy
            Created strategy
        """
        self.logger.info(f"Creating {strategy_type} strategy: {name}")
        
        # Create strategy based on type
        if strategy_type.lower() == "swing":
            strategy = SwingTradingStrategy(name or "Swing Trading Strategy", description)
        elif strategy_type.lower() == "day":
            strategy = DayTradingStrategy(name or "Day Trading Strategy", description)
        else:
            raise ValueError(f"Strategy type '{strategy_type}' not supported")
        
        # Set parameters if provided
        if parameters:
            strategy.set_parameters(parameters)
        
        # Store strategy
        self.strategies[strategy.name] = strategy
        
        return strategy
    
    def get_strategy(self, name):
        """
        Get a strategy by name.
        
        Parameters:
        -----------
        name : str
            Strategy name
            
        Returns:
        --------
        Strategy
            Retrieved strategy
        """
        if name in self.strategies:
            return self.strategies[name]
        else:
            raise ValueError(f"Strategy '{name}' not found")
    
    def list_strategies(self):
        """
        List all available strategies.
        
        Returns:
        --------
        list
            List of strategy names
        """
        return list(self.strategies.keys())
    
    def delete_strategy(self, name):
        """
        Delete a strategy.
        
        Parameters:
        -----------
        name : str
            Strategy name
            
        Returns:
        --------
        bool
            True if strategy was deleted, False otherwise
        """
        if name in self.strategies:
            del self.strategies[name]
            return True
        else:
            return False
    
    def generate_strategy_for_ticker(self, ticker, strategy_type="swing", period="1y", interval="1d", parameters=None):
        """
        Generate a strategy for the specified ticker.
        
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
            
        Returns:
        --------
        dict
            Strategy definition and analysis
        """
        self.logger.info(f"Generating {strategy_type} strategy for {ticker}")
        
        # Fetch market data
        data = data_access.fetch_historical_data(ticker, period=period, interval=interval)
        
        # Create strategy
        strategy_name = f"{ticker}_{strategy_type}_strategy"
        strategy = self.create_strategy(
            strategy_type=strategy_type,
            name=strategy_name,
            description=f"{strategy_type.capitalize()} Trading Strategy for {ticker}",
            parameters=parameters
        )
        
        # Generate signals and backtest
        backtest_result = strategy.backtest(data)
        
        # Generate strategy summary
        strategy_summary = {
            'ticker': ticker,
            'strategy_type': strategy_type,
            'parameters': strategy.parameters,
            'signals': strategy.signals,
            'trades': strategy.trades,
            'performance': strategy.performance,
            'description': self._generate_strategy_description(strategy, ticker)
        }
        
        return strategy_summary
    
    def optimize_strategy_for_ticker(self, ticker, strategy_type="swing", period="1y", interval="1d", 
                                    parameter_grid=None, metric='sharpe_ratio'):
        """
        Optimize a strategy for the specified ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to optimize a strategy for
        strategy_type : str, optional
            Type of strategy to optimize ("swing", "day", "trend", "breakout", etc.)
        period : str, optional
            Period to fetch data for (e.g., "1d", "1mo", "1y")
        interval : str, optional
            Data interval (e.g., "1m", "5m", "1h", "1d")
        parameter_grid : dict, optional
            Dictionary of parameter names and lists of values to try
        metric : str, optional
            Performance metric to optimize for
            
        Returns:
        --------
        dict
            Optimization results
        """
        self.logger.info(f"Optimizing {strategy_type} strategy for {ticker}")
        
        # Fetch market data
        data = data_access.fetch_historical_data(ticker, period=period, interval=interval)
        
        # Create strategy
        strategy_name = f"{ticker}_{strategy_type}_strategy_optimized"
        strategy = self.create_strategy(
            strategy_type=strategy_type,
            name=strategy_name,
            description=f"Optimized {strategy_type.capitalize()} Trading Strategy for {ticker}"
        )
        
        # Set default parameter grid if not provided
        if parameter_grid is None:
            if strategy_type.lower() == "swing":
                parameter_grid = {
                    'rsi_period': [7, 14, 21],
                    'rsi_oversold': [20, 30, 40],
                    'rsi_overbought': [60, 70, 80],
                    'sma_short_period': [10, 20, 50],
                    'sma_long_period': [50, 100, 200],
                    'risk_per_trade': [0.01, 0.02, 0.03],
                    'risk_reward_ratio': [1.5, 2.0, 3.0],
                    'stop_loss_atr_multiple': [1.5, 2.0, 2.5]
                }
            elif strategy_type.lower() == "day":
                parameter_grid = {
                    'ema_short_period': [5, 9, 13],
                    'ema_medium_period': [13, 21, 34],
                    'ema_long_period': [34, 50, 89],
                    'rsi_period': [7, 14, 21],
                    'risk_per_trade': [0.005, 0.01, 0.015],
                    'risk_reward_ratio': [1.0, 1.5, 2.0],
                    'stop_loss_atr_multiple': [1.0, 1.5, 2.0]
                }
        
        # Optimize strategy
        optimization_result = strategy.optimize(data, parameter_grid, metric=metric)
        
        # Generate signals and backtest with optimized parameters
        backtest_result = strategy.backtest(data)
        
        # Generate strategy summary
        strategy_summary = {
            'ticker': ticker,
            'strategy_type': strategy_type,
            'parameters': strategy.parameters,
            'signals': strategy.signals,
            'trades': strategy.trades,
            'performance': strategy.performance,
            'description': self._generate_strategy_description(strategy, ticker, optimized=True),
            'optimization_results': optimization_result
        }
        
        return strategy_summary
    
    def _generate_strategy_description(self, strategy, ticker, optimized=False):
        """
        Generate a human-readable description of the strategy.
        
        Parameters:
        -----------
        strategy : Strategy
            Strategy to describe
        ticker : str
            Ticker symbol
        optimized : bool, optional
            Whether the strategy has been optimized
            
        Returns:
        --------
        str
            Strategy description
        """
        if isinstance(strategy, SwingTradingStrategy):
            description = f"""
                {'Optimized ' if optimized else ''}Swing Trading Strategy for {ticker}
                
                Entry Conditions:
                - RSI crosses above {strategy.parameters['rsi_oversold']} (oversold level)
                - {strategy.parameters['sma_short_period']}-day SMA crosses above {strategy.parameters['sma_long_period']}-day SMA
                
                Exit Conditions:
                - RSI crosses below {strategy.parameters['rsi_overbought']} (overbought level)
                - {strategy.parameters['sma_short_period']}-day SMA crosses below {strategy.parameters['sma_long_period']}-day SMA
                - Stop Loss: Entry Price - ({strategy.parameters['stop_loss_atr_multiple']} x ATR)
                - Take Profit: Entry Price + (Risk x {strategy.parameters['risk_reward_ratio']})
                
                Risk Management:
                - Risk per trade: {strategy.parameters['risk_per_trade'] * 100}% of capital
                - Risk-Reward Ratio: 1:{strategy.parameters['risk_reward_ratio']}
            """
        elif isinstance(strategy, DayTradingStrategy):
            description = f"""
                {'Optimized ' if optimized else ''}Day Trading Strategy for {ticker}
                
                Entry Conditions:
                - EMA alignment: {strategy.parameters['ema_short_period']}-period EMA > {strategy.parameters['ema_medium_period']}-period EMA > {strategy.parameters['ema_long_period']}-period EMA
                - {strategy.parameters['ema_short_period']}-period EMA crosses above {strategy.parameters['ema_medium_period']}-period EMA
                - RSI < {strategy.parameters['rsi_overbought']}
                
                Exit Conditions:
                - EMA alignment: {strategy.parameters['ema_short_period']}-period EMA < {strategy.parameters['ema_medium_period']}-period EMA < {strategy.parameters['ema_long_period']}-period EMA
                - {strategy.parameters['ema_short_period']}-period EMA crosses below {strategy.parameters['ema_medium_period']}-period EMA
                - RSI > {strategy.parameters['rsi_overbought']}
                - Stop Loss: Entry Price - ({strategy.parameters['stop_loss_atr_multiple']} x ATR)
                - Take Profit: Entry Price + (Risk x {strategy.parameters['risk_reward_ratio']})
                
                Risk Management:
                - Risk per trade: {strategy.parameters['risk_per_trade'] * 100}% of capital
                - Risk-Reward Ratio: 1:{strategy.parameters['risk_reward_ratio']}
                - Short positions: {'Allowed' if strategy.parameters.get('allow_short', False) else 'Not allowed'}
            """
        else:
            description = f"""
                {'Optimized ' if optimized else ''}Trading Strategy for {ticker}
                
                Parameters:
                {strategy.parameters}
            """
        
        return description


# Create a singleton instance for easy access
strategy_builder = StrategyBuilder()

def create_strategy(strategy_type, name=None, description=None, parameters=None):
    """
    Create a new trading strategy using the default strategy builder.
    
    Parameters:
    -----------
    strategy_type : str
        Type of strategy to create ("swing", "day", "trend", "breakout", etc.)
    name : str, optional
        Strategy name
    description : str, optional
        Strategy description
    parameters : dict, optional
        Strategy parameters
        
    Returns:
    --------
    Strategy
        Created strategy
    """
    return strategy_builder.create_strategy(strategy_type, name, description, parameters)

def generate_strategy_for_ticker(ticker, strategy_type="swing", period="1y", interval="1d", parameters=None):
    """
    Generate a strategy for the specified ticker using the default strategy builder.
    
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
        
    Returns:
    --------
    dict
        Strategy definition and analysis
    """
    return strategy_builder.generate_strategy_for_ticker(ticker, strategy_type, period, interval, parameters)

def optimize_strategy_for_ticker(ticker, strategy_type="swing", period="1y", interval="1d", 
                               parameter_grid=None, metric='sharpe_ratio'):
    """
    Optimize a strategy for the specified ticker using the default strategy builder.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to optimize a strategy for
    strategy_type : str, optional
        Type of strategy to optimize ("swing", "day", "trend", "breakout", etc.)
    period : str, optional
        Period to fetch data for (e.g., "1d", "1mo", "1y")
    interval : str, optional
        Data interval (e.g., "1m", "5m", "1h", "1d")
    parameter_grid : dict, optional
        Dictionary of parameter names and lists of values to try
    metric : str, optional
        Performance metric to optimize for
        
    Returns:
    --------
    dict
        Optimization results
    """
    return strategy_builder.optimize_strategy_for_ticker(ticker, strategy_type, period, interval, parameter_grid, metric)
