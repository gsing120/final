"""
Swing Trading Strategy Generator for Gemma Advanced Trading System.

This module provides specialized functionality for generating swing trading strategies.
"""

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
import data_access
from strategy_builder import Strategy, create_strategy, generate_strategy_for_ticker, optimize_strategy_for_ticker

# Configure logging
logger = logging.getLogger("GemmaTrading.SwingTrading")

class EnhancedSwingTradingStrategy(Strategy):
    """Enhanced Swing Trading Strategy implementation with additional features."""
    
    def __init__(self, name="Enhanced Swing Trading Strategy", description=None):
        """
        Initialize an enhanced swing trading strategy.
        
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
            # Trend indicators
            'sma_short_period': 20,
            'sma_long_period': 50,
            'ema_short_period': 9,
            'ema_long_period': 21,
            
            # Momentum indicators
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'stochastic_k_period': 14,
            'stochastic_d_period': 3,
            'stochastic_overbought': 80,
            'stochastic_oversold': 20,
            
            # Volatility indicators
            'atr_period': 14,
            'bollinger_period': 20,
            'bollinger_std_dev': 2,
            
            # Volume indicators
            'obv_use': True,
            'vwap_use': True,
            
            # Risk management
            'risk_per_trade': 0.02,  # 2% risk per trade
            'risk_reward_ratio': 2.0,  # 1:2 risk-reward ratio
            'stop_loss_atr_multiple': 2.0,  # Stop loss at 2x ATR
            'trailing_stop_use': False,
            'trailing_stop_activation': 1.0,  # ATR multiple for activation
            'trailing_stop_distance': 2.0,  # ATR multiple for distance
            
            # Strategy specific
            'allow_short': False,  # Whether to allow short positions
            'confirmation_needed': 2,  # Number of indicators needed for confirmation
            'exit_on_opposite_signal': True,  # Exit when opposite signal is generated
            'max_holding_period': 20,  # Maximum holding period in days
            'min_holding_period': 2,  # Minimum holding period in days
            
            # Market regime filters
            'use_market_regime_filter': False,
            'market_regime_sma_period': 200,
            'market_regime_index': '^GSPC'  # S&P 500 as default market index
        })
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the enhanced swing trading strategy.
        
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
        
        # Calculate trend indicators
        df['SMA_Short'] = simple_moving_average(df['Close'], period=self.parameters.get('sma_short_period', 10))
        df['SMA_Long'] = simple_moving_average(df['Close'], period=self.parameters.get('sma_long_period', 50))
        
        # Use get() with defaults for EMA parameters that might be removed during optimization
        ema_short_period = self.parameters.get('ema_short_period', 9)
        ema_long_period = self.parameters.get('ema_long_period', 21)
        df['EMA_Short'] = exponential_moving_average(df['Close'], period=ema_short_period)
        df['EMA_Long'] = exponential_moving_average(df['Close'], period=ema_long_period)
        
        # Calculate momentum indicators
        df['RSI'] = relative_strength_index(df['Close'], period=self.parameters.get('rsi_period', 14))
        stoch_k, stoch_d = stochastic_oscillator(
            df['High'], df['Low'], df['Close'], 
            k_period=self.parameters.get('stochastic_k_period', 14), 
            d_period=self.parameters.get('stochastic_d_period', 3)
        )
        df['Stochastic_K'] = stoch_k
        df['Stochastic_D'] = stoch_d
        
        # Calculate volatility indicators
        df['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], period=self.parameters.get('atr_period', 14))
        upper_band, middle_band, lower_band = bollinger_bands(
            df['Close'], 
            period=self.parameters.get('bollinger_period', 20), 
            std_dev=self.parameters.get('bollinger_std_dev', 2)
        )
        df['Bollinger_Upper'] = upper_band
        df['Bollinger_Middle'] = middle_band
        df['Bollinger_Lower'] = lower_band
        
        # Calculate volume indicators
        if self.parameters.get('obv_use', True):
            df['OBV'] = on_balance_volume(df['Close'], df['Volume'])
        
        if self.parameters.get('vwap_use', True):
            df['VWAP'] = volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Apply market regime filter if enabled
        if self.parameters.get('use_market_regime_filter', False):
            try:
                # Fetch market index data
                market_data = data_access.fetch_historical_data(
                    self.parameters.get('market_regime_index', '^GSPC'),
                    start_date=df.index[0],
                    end_date=df.index[-1]
                )
                
                # Calculate market regime
                market_data['SMA_200'] = simple_moving_average(
                    market_data['Close'], 
                    period=self.parameters.get('market_regime_sma_period', 200)
                )
                market_data['Market_Regime'] = np.where(
                    market_data['Close'] > market_data['SMA_200'], 
                    1,  # Bullish
                    -1  # Bearish
                )
                
                # Merge market regime with price data
                df = pd.merge_asof(
                    df, 
                    market_data[['Market_Regime']], 
                    left_index=True, 
                    right_index=True, 
                    direction='backward'
                )
            except Exception as e:
                self.logger.warning(f"Failed to apply market regime filter: {e}")
                df['Market_Regime'] = 1  # Default to bullish
        else:
            df['Market_Regime'] = 1  # Default to bullish
        
        # Initialize signal columns
        df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
        df['Signal_Strength'] = 0  # Number of confirming indicators
        
        # Calculate individual indicator signals
        df['SMA_Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, -1)
        df['EMA_Signal'] = np.where(df['EMA_Short'] > df['EMA_Long'], 1, -1)
        df['RSI_Signal'] = np.where(df['RSI'] < self.parameters.get('rsi_oversold', 30), 1, 
                                   np.where(df['RSI'] > self.parameters.get('rsi_overbought', 70), -1, 0))
        df['Stochastic_Signal'] = np.where(df['Stochastic_K'] < self.parameters.get('stochastic_oversold', 20), 1, 
                                          np.where(df['Stochastic_K'] > self.parameters.get('stochastic_overbought', 80), -1, 0))
        df['Bollinger_Signal'] = np.where(df['Close'] < df['Bollinger_Lower'], 1, 
                                         np.where(df['Close'] > df['Bollinger_Upper'], -1, 0))
        
        # Calculate signal strength (number of confirming indicators)
        for i in range(1, len(df)):
            # Count bullish signals
            bullish_signals = sum([
                df['SMA_Signal'].iloc[i] == 1,
                df['EMA_Signal'].iloc[i] == 1,
                df['RSI_Signal'].iloc[i] == 1,
                df['Stochastic_Signal'].iloc[i] == 1,
                df['Bollinger_Signal'].iloc[i] == 1
            ])
            
            # Count bearish signals
            bearish_signals = sum([
                df['SMA_Signal'].iloc[i] == -1,
                df['EMA_Signal'].iloc[i] == -1,
                df['RSI_Signal'].iloc[i] == -1,
                df['Stochastic_Signal'].iloc[i] == -1,
                df['Bollinger_Signal'].iloc[i] == -1
            ])
            
            # Determine signal strength
            if bullish_signals >= self.parameters.get('confirmation_needed', 2) and df['Market_Regime'].iloc[i] == 1:
                df.loc[df.index[i], 'Signal_Strength'] = bullish_signals
                df.loc[df.index[i], 'Signal'] = 1
            elif bearish_signals >= self.parameters.get('confirmation_needed', 2) and (
                self.parameters.get('allow_short', False) or df['Signal'].iloc[i-1] == 1
            ):
                df.loc[df.index[i], 'Signal_Strength'] = -bearish_signals
                df.loc[df.index[i], 'Signal'] = -1
        
        # Apply swing trading specific logic
        self._apply_swing_trading_logic(df)
        
        # Calculate stop loss and take profit levels for each buy signal
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan
        df['Trailing_Stop'] = np.nan
        
        for i in range(1, len(df)):
            if df['Signal'].iloc[i] == 1:  # Buy signal
                entry_price = df['Close'].iloc[i]
                atr_value = df['ATR'].iloc[i]
                
                # Set stop loss at entry_price - (ATR * multiple)
                stop_loss = entry_price - (atr_value * self.parameters.get('stop_loss_atr_multiple', 2.0))
                
                # Set take profit based on risk-reward ratio
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * self.parameters.get('risk_reward_ratio', 2.0))
                
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Take_Profit'] = take_profit
                
                # Initialize trailing stop if enabled
                if self.parameters.get('trailing_stop_use', False):
                    df.loc[df.index[i], 'Trailing_Stop'] = entry_price - (atr_value * self.parameters.get('trailing_stop_distance', 2.0))
            
            elif df['Signal'].iloc[i] == -1 and self.parameters.get('allow_short', False):  # Sell signal (for short)
                entry_price = df['Close'].iloc[i]
                atr_value = df['ATR'].iloc[i]
                
                # Set stop loss at entry_price + (ATR * multiple)
                stop_loss = entry_price + (atr_value * self.parameters.get('stop_loss_atr_multiple', 2.0))
                
                # Set take profit based on risk-reward ratio
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * self.parameters.get('risk_reward_ratio', 2.0))
                
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Take_Profit'] = take_profit
                
                # Initialize trailing stop if enabled
                if self.parameters.get('trailing_stop_use', False):
                    df.loc[df.index[i], 'Trailing_Stop'] = entry_price + (atr_value * self.parameters.get('trailing_stop_distance', 2.0))
        
        self.signals = df
        return df
    
    def _apply_swing_trading_logic(self, df):
        """
        Apply swing trading specific logic to the signals.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data with preliminary signals
        """
        # Track active trades
        in_position = False
        entry_date = None
        entry_price = None
        position_type = None  # 'long' or 'short'
        
        # Process signals
        for i in range(1, len(df)):
            current_date = df.index[i]
            
            if not in_position:
                # Check for entry signal
                if df['Signal'].iloc[i] == 1:  # Buy signal
                    in_position = True
                    entry_date = current_date
                    entry_price = df['Close'].iloc[i]
                    position_type = 'long'
                elif df['Signal'].iloc[i] == -1 and self.parameters.get('allow_short', False):  # Sell signal (for short)
                    in_position = True
                    entry_date = current_date
                    entry_price = df['Close'].iloc[i]
                    position_type = 'short'
            else:
                # Check for exit conditions
                holding_period = (current_date - entry_date).days
                
                # Minimum holding period check
                if holding_period < self.parameters.get('min_holding_period', 2):
                    df.loc[df.index[i], 'Signal'] = 0  # Cancel any exit signals during minimum holding period
                    continue
                
                # Maximum holding period check
                if holding_period >= self.parameters.get('max_holding_period', 20):
                    df.loc[df.index[i], 'Signal'] = -1 if position_type == 'long' else 1
                    in_position = False
                    entry_date = None
                    entry_price = None
                    position_type = None
                    continue
                
                # Check for exit signal
                if position_type == 'long' and df['Signal'].iloc[i] == -1:
                    # Keep the exit signal
                    in_position = False
                    entry_date = None
                    entry_price = None
                    position_type = None
                elif position_type == 'short' and df['Signal'].iloc[i] == 1:
                    # Keep the exit signal
                    in_position = False
                    entry_date = None
                    entry_price = None
                    position_type = None
                else:
                    # No exit signal, cancel any entry signals while in position
                    df.loc[df.index[i], 'Signal'] = 0


class SwingTradingGenerator:
    """
    Specialized generator for swing trading strategies.
    """
    
    def __init__(self):
        """Initialize the swing trading generator."""
        self.logger = logging.getLogger("GemmaTrading.SwingTradingGenerator")
    
    def generate_strategy(self, ticker, period="1y", interval="1d", parameters=None, optimize=False):
        """
        Generate a swing trading strategy for the specified ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to generate a strategy for
        period : str, optional
            Period to fetch data for (e.g., "1d", "1mo", "1y")
        interval : str, optional
            Data interval (e.g., "1m", "5m", "1h", "1d")
        parameters : dict, optional
            Strategy parameters
        optimize : bool, optional
            Whether to optimize the strategy parameters
            
        Returns:
        --------
        dict
            Strategy definition and analysis
        """
        self.logger.info(f"Generating swing trading strategy for {ticker}")
        
        # Fetch market data
        data = data_access.fetch_historical_data(ticker, period=period, interval=interval)
        
        # Create strategy
        strategy_name = f"{ticker}_swing_trading_strategy"
        strategy = EnhancedSwingTradingStrategy(
            name=strategy_name,
            description=f"Enhanced Swing Trading Strategy for {ticker}"
        )
        
        # Set custom parameters if provided
        if parameters:
            strategy.set_parameters(parameters)
        
        # Optimize if requested
        if optimize:
            parameter_grid = {
                'sma_short_period': [10, 20, 50],
                'sma_long_period': [50, 100, 200],
                'rsi_period': [7, 14, 21],
                'rsi_oversold': [20, 30, 40],
                'rsi_overbought': [60, 70, 80],
                'stochastic_k_period': [9, 14, 21],
                'stochastic_oversold': [10, 20, 30],
                'stochastic_overbought': [70, 80, 90],
                'risk_per_trade': [0.01, 0.02, 0.03],
                'risk_reward_ratio': [1.5, 2.0, 3.0],
                'stop_loss_atr_multiple': [1.5, 2.0, 2.5],
                'confirmation_needed': [1, 2, 3]
            }
            
            optimization_result = strategy.optimize(data, parameter_grid, metric='sharpe_ratio')
            self.logger.info(f"Optimization complete. Best parameters: {strategy.parameters}")
        
        # Generate signals and backtest
        backtest_result = strategy.backtest(data)
        
        # Generate strategy summary
        strategy_summary = {
            'ticker': ticker,
            'strategy_type': 'enhanced_swing',
            'parameters': strategy.parameters,
            'signals': strategy.signals,
            'trades': strategy.trades,
            'performance': strategy.performance,
            'description': self._generate_strategy_description(strategy, ticker, optimized=optimize),
            'optimization_results': optimization_result if optimize else None
        }
        
        return strategy_summary
    
    def _generate_strategy_description(self, strategy, ticker, optimized=False):
        """
        Generate a human-readable description of the strategy.
        
        Parameters:
        -----------
        strategy : EnhancedSwingTradingStrategy
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
        description = f"""
            {'Optimized ' if optimized else ''}Enhanced Swing Trading Strategy for {ticker}
            
            Entry Conditions (need {strategy.parameters['confirmation_needed']} or more):
            - SMA: {strategy.parameters['sma_short_period']}-day SMA crosses above {strategy.parameters['sma_long_period']}-day SMA
            - EMA: {strategy.parameters['ema_short_period']}-day EMA crosses above {strategy.parameters['ema_long_period']}-day EMA
            - RSI: RSI({strategy.parameters['rsi_period']}) crosses above {strategy.parameters['rsi_oversold']} (oversold level)
            - Stochastic: Stochastic %K({strategy.parameters['stochastic_k_period']}) crosses above {strategy.parameters['stochastic_oversold']} (oversold level)
            - Bollinger Bands: Price touches or crosses below lower Bollinger Band
            
            Exit Conditions:
            - SMA: {strategy.parameters['sma_short_period']}-day SMA crosses below {strategy.parameters['sma_long_period']}-day SMA
            - EMA: {strategy.parameters['ema_short_period']}-day EMA crosses below {strategy.parameters['ema_long_period']}-day EMA
            - RSI: RSI({strategy.parameters['rsi_period']}) crosses above {strategy.parameters['rsi_overbought']} (overbought level)
            - Stochastic: Stochastic %K({strategy.parameters['stochastic_k_period']}) crosses above {strategy.parameters['stochastic_overbought']} (overbought level)
            - Bollinger Bands: Price touches or crosses above upper Bollinger Band
            - Stop Loss: Entry Price - ({strategy.parameters['stop_loss_atr_multiple']} x ATR)
            - Take Profit: Entry Price + (Risk x {strategy.parameters['risk_reward_ratio']})
            - Maximum Holding Period: {strategy.parameters['max_holding_period']} days
            
            Risk Management:
            - Risk per trade: {strategy.parameters['risk_per_trade'] * 100}% of capital
            - Risk-Reward Ratio: 1:{strategy.parameters['risk_reward_ratio']}
            - Minimum Holding Period: {strategy.parameters['min_holding_period']} days
            - Trailing Stop: {'Enabled' if strategy.parameters.get('trailing_stop_use', False) else 'Disabled'}
            
            Market Regime Filter:
            - {'Enabled' if strategy.parameters.get('use_market_regime_filter', False) else 'Disabled'}
            - Market Index: {strategy.parameters.get('market_regime_index', '^GSPC')}
            - SMA Period: {strategy.parameters.get('market_regime_sma_period', 200)} days
        """
        
        return description


# Create a singleton instance for easy access
swing_trading_generator = SwingTradingGenerator()

def generate_swing_strategy(ticker, period="1y", interval="1d", parameters=None, optimize=False):
    """
    Generate a swing trading strategy for the specified ticker using the default generator.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to generate a strategy for
    period : str, optional
        Period to fetch data for (e.g., "1d", "1mo", "1y")
    interval : str, optional
        Data interval (e.g., "1m", "5m", "1h", "1d")
    parameters : dict, optional
        Strategy parameters
    optimize : bool, optional
        Whether to optimize the strategy parameters
        
    Returns:
    --------
    dict
        Strategy definition and analysis
    """
    return swing_trading_generator.generate_strategy(ticker, period, interval, parameters, optimize)
