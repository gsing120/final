"""
Volatility Indicators for Gemma Advanced Trading System.

This module provides various volatility indicators used for market analysis and trading strategies.
"""

import numpy as np
import pandas as pd


class VolatilityIndicators:
    """Class containing volatility-based technical indicators."""
    
    def atr(self, high, low, close, period=14):
        """
        Calculate Average True Range (ATR).
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        period : int
            ATR calculation period
            
        Returns:
        --------
        pandas.Series
            ATR values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # Get the maximum of the three true ranges
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def standard_deviation(self, data, period=20):
        """
        Calculate Standard Deviation.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Calculation period
            
        Returns:
        --------
        pandas.Series
            Standard Deviation values
        """
        # Calculate Standard Deviation
        std = data.rolling(window=period).std()
        
        return std
    
    def keltner_channel(self, high, low, close, ema_period=20, atr_period=10, multiplier=2):
        """
        Calculate Keltner Channel.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        ema_period : int
            EMA calculation period
        atr_period : int
            ATR calculation period
        multiplier : float
            ATR multiplier
            
        Returns:
        --------
        tuple of pandas.Series
            (Upper band, Middle band, Lower band)
        """
        # Calculate EMA
        middle_band = close.ewm(span=ema_period, adjust=False).mean()
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        # Calculate Keltner Channel
        upper_band = middle_band + (multiplier * atr)
        lower_band = middle_band - (multiplier * atr)
        
        return upper_band, middle_band, lower_band
    
    def historical_volatility(self, data, period=20, trading_periods=252):
        """
        Calculate Historical Volatility.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Calculation period
        trading_periods : int
            Number of trading periods in a year
            
        Returns:
        --------
        pandas.Series
            Historical Volatility values (annualized)
        """
        # Calculate daily returns
        returns = data.pct_change()
        
        # Calculate standard deviation of returns
        std = returns.rolling(window=period).std()
        
        # Annualize the standard deviation
        historical_volatility = std * np.sqrt(trading_periods)
        
        return historical_volatility
    
    def ulcer_index(self, data, period=14):
        """
        Calculate Ulcer Index.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Calculation period
            
        Returns:
        --------
        pandas.Series
            Ulcer Index values
        """
        # Calculate percentage drawdown
        roll_max = data.rolling(window=period).max()
        drawdown = ((data - roll_max) / roll_max) * 100
        
        # Calculate squared drawdown
        squared_drawdown = drawdown ** 2
        
        # Calculate Ulcer Index
        ulcer_index = np.sqrt(squared_drawdown.rolling(window=period).mean())
        
        return ulcer_index
    
    def normalized_atr(self, high, low, close, period=14):
        """
        Calculate Normalized Average True Range (NATR).
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        period : int
            ATR calculation period
            
        Returns:
        --------
        pandas.Series
            NATR values
        """
        # Calculate ATR
        atr_values = self.atr(high, low, close, period)
        
        # Calculate NATR (ATR as a percentage of close price)
        natr = (atr_values / close) * 100
        
        return natr
    
    def chaikin_volatility(self, high, low, ema_period=10, roc_period=10):
        """
        Calculate Chaikin Volatility.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        ema_period : int
            EMA calculation period
        roc_period : int
            Rate of Change period
            
        Returns:
        --------
        pandas.Series
            Chaikin Volatility values
        """
        # Calculate high-low range
        hl_range = high - low
        
        # Calculate EMA of high-low range
        ema_hl_range = hl_range.ewm(span=ema_period, adjust=False).mean()
        
        # Calculate Rate of Change of EMA of high-low range
        chaikin_volatility = ((ema_hl_range / ema_hl_range.shift(roc_period)) - 1) * 100
        
        return chaikin_volatility


# Create a singleton instance of VolatilityIndicators
_volatility_indicators = VolatilityIndicators()

# Function-based interface for compatibility with main application
def average_true_range(high, low, close, period=14):
    """
    Calculate Average True Range (ATR).
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    period : int
        ATR calculation period
        
    Returns:
    --------
    pandas.Series
        ATR values
    """
    return _volatility_indicators.atr(high, low, close, period)

def standard_deviation(data, period=20):
    """
    Calculate Standard Deviation.
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    period : int
        Calculation period
        
    Returns:
    --------
    pandas.Series
        Standard Deviation values
    """
    return _volatility_indicators.standard_deviation(data, period)

def keltner_channel(high, low, close, ema_period=20, atr_period=10, multiplier=2):
    """
    Calculate Keltner Channel.
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    ema_period : int
        EMA calculation period
    atr_period : int
        ATR calculation period
    multiplier : float
        ATR multiplier
        
    Returns:
    --------
    tuple of pandas.Series
        (Upper band, Middle band, Lower band)
    """
    return _volatility_indicators.keltner_channel(high, low, close, ema_period, atr_period, multiplier)

def historical_volatility(data, period=20, trading_periods=252):
    """
    Calculate Historical Volatility.
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    period : int
        Calculation period
    trading_periods : int
        Number of trading periods in a year
        
    Returns:
    --------
    pandas.Series
        Historical Volatility values (annualized)
    """
    return _volatility_indicators.historical_volatility(data, period, trading_periods)

def ulcer_index(data, period=14):
    """
    Calculate Ulcer Index.
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    period : int
        Calculation period
        
    Returns:
    --------
    pandas.Series
        Ulcer Index values
    """
    return _volatility_indicators.ulcer_index(data, period)

def normalized_atr(high, low, close, period=14):
    """
    Calculate Normalized Average True Range (NATR).
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    period : int
        ATR calculation period
        
    Returns:
    --------
    pandas.Series
        NATR values
    """
    return _volatility_indicators.normalized_atr(high, low, close, period)

def chaikin_volatility(high, low, ema_period=10, roc_period=10):
    """
    Calculate Chaikin Volatility.
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    ema_period : int
        EMA calculation period
    roc_period : int
        Rate of Change period
        
    Returns:
    --------
    pandas.Series
        Chaikin Volatility values
    """
    return _volatility_indicators.chaikin_volatility(high, low, ema_period, roc_period)
