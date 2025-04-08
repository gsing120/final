"""
Momentum Indicators for Gemma Advanced Trading System.

This module provides various momentum indicators used for market analysis and trading strategies.
"""

import numpy as np
import pandas as pd


class MomentumIndicators:
    """Class containing momentum-based technical indicators."""
    
    def rsi(self, data, period=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            RSI calculation period
            
        Returns:
        --------
        pandas.Series
            RSI values
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stochastic(self, high, low, close, k_period=14, d_period=3, slowing=3):
        """
        Calculate Stochastic Oscillator.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        k_period : int
            %K period
        d_period : int
            %D period
        slowing : int
            Slowing period
            
        Returns:
        --------
        tuple of pandas.Series
            (%K, %D)
        """
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # Fast %K
        fast_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Slow %K (with slowing)
        k = fast_k.rolling(window=slowing).mean()
        
        # %D (SMA of %K)
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def cci(self, high, low, close, period=20):
        """
        Calculate Commodity Channel Index (CCI).
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        period : int
            CCI calculation period
            
        Returns:
        --------
        pandas.Series
            CCI values
        """
        # Calculate typical price
        tp = (high + low + close) / 3
        
        # Calculate SMA of typical price
        sma_tp = tp.rolling(window=period).mean()
        
        # Calculate Mean Deviation
        mean_deviation = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        # Calculate CCI
        cci = (tp - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def williams_r(self, high, low, close, period=14):
        """
        Calculate Williams %R.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        period : int
            Calculation period
            
        Returns:
        --------
        pandas.Series
            Williams %R values
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        # Calculate Williams %R
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def rate_of_change(self, data, period=10):
        """
        Calculate Rate of Change (ROC).
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            ROC calculation period
            
        Returns:
        --------
        pandas.Series
            ROC values
        """
        # Calculate ROC
        roc = 100 * ((data / data.shift(period)) - 1)
        
        return roc
    
    def awesome_oscillator(self, high, low, fast_period=5, slow_period=34):
        """
        Calculate Awesome Oscillator.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        fast_period : int
            Fast period
        slow_period : int
            Slow period
            
        Returns:
        --------
        pandas.Series
            Awesome Oscillator values
        """
        # Calculate median price
        median_price = (high + low) / 2
        
        # Calculate SMA of median price for fast and slow periods
        fast_sma = median_price.rolling(window=fast_period).mean()
        slow_sma = median_price.rolling(window=slow_period).mean()
        
        # Calculate Awesome Oscillator
        ao = fast_sma - slow_sma
        
        return ao
    
    def money_flow_index(self, high, low, close, volume, period=14):
        """
        Calculate Money Flow Index (MFI).
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        volume : pandas.Series
            Volume series
        period : int
            MFI calculation period
            
        Returns:
        --------
        pandas.Series
            MFI values
        """
        # Calculate typical price
        tp = (high + low + close) / 3
        
        # Calculate raw money flow
        raw_money_flow = tp * volume
        
        # Calculate money flow direction
        direction = np.where(tp > tp.shift(1), 1, -1)
        
        # Calculate positive and negative money flow
        positive_flow = pd.Series(np.where(direction > 0, raw_money_flow, 0), index=close.index)
        negative_flow = pd.Series(np.where(direction < 0, raw_money_flow, 0), index=close.index)
        
        # Calculate sum of positive and negative money flow over period
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        # Calculate money flow ratio
        money_flow_ratio = positive_sum / negative_sum
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    def true_strength_index(self, data, long_period=25, short_period=13):
        """
        Calculate True Strength Index (TSI).
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        long_period : int
            Long EMA period
        short_period : int
            Short EMA period
            
        Returns:
        --------
        pandas.Series
            TSI values
        """
        # Calculate price change
        momentum = data.diff()
        
        # Calculate long EMA of momentum
        long_ema = momentum.ewm(span=long_period, adjust=False).mean()
        
        # Calculate short EMA of long EMA
        double_ema = long_ema.ewm(span=short_period, adjust=False).mean()
        
        # Calculate absolute momentum
        abs_momentum = momentum.abs()
        
        # Calculate long EMA of absolute momentum
        abs_long_ema = abs_momentum.ewm(span=long_period, adjust=False).mean()
        
        # Calculate short EMA of long EMA of absolute momentum
        abs_double_ema = abs_long_ema.ewm(span=short_period, adjust=False).mean()
        
        # Calculate TSI
        tsi = 100 * (double_ema / abs_double_ema)
        
        return tsi


# Create a singleton instance of MomentumIndicators
_momentum_indicators = MomentumIndicators()

# Function-based interface for compatibility with main application
def relative_strength_index(data, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    period : int
        RSI calculation period
        
    Returns:
    --------
    pandas.Series
        RSI values
    """
    return _momentum_indicators.rsi(data, period)

def stochastic_oscillator(high, low, close, k_period=14, d_period=3, slowing=3):
    """
    Calculate Stochastic Oscillator.
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    k_period : int
        %K period
    d_period : int
        %D period
    slowing : int
        Slowing period
        
    Returns:
    --------
    tuple of pandas.Series
        (%K, %D)
    """
    return _momentum_indicators.stochastic(high, low, close, k_period, d_period, slowing)

def commodity_channel_index(high, low, close, period=20):
    """
    Calculate Commodity Channel Index (CCI).
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    period : int
        CCI calculation period
        
    Returns:
    --------
    pandas.Series
        CCI values
    """
    return _momentum_indicators.cci(high, low, close, period)

def williams_percent_r(high, low, close, period=14):
    """
    Calculate Williams %R.
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    period : int
        Calculation period
        
    Returns:
    --------
    pandas.Series
        Williams %R values
    """
    return _momentum_indicators.williams_r(high, low, close, period)

def rate_of_change(data, period=10):
    """
    Calculate Rate of Change (ROC).
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    period : int
        ROC calculation period
        
    Returns:
    --------
    pandas.Series
        ROC values
    """
    return _momentum_indicators.rate_of_change(data, period)

def awesome_oscillator(high, low, fast_period=5, slow_period=34):
    """
    Calculate Awesome Oscillator.
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    fast_period : int
        Fast period
    slow_period : int
        Slow period
        
    Returns:
    --------
    pandas.Series
        Awesome Oscillator values
    """
    return _momentum_indicators.awesome_oscillator(high, low, fast_period, slow_period)

def money_flow_index(high, low, close, volume, period=14):
    """
    Calculate Money Flow Index (MFI).
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    volume : pandas.Series
        Volume series
    period : int
        MFI calculation period
        
    Returns:
    --------
    pandas.Series
        MFI values
    """
    return _momentum_indicators.money_flow_index(high, low, close, volume, period)

def true_strength_index(data, long_period=25, short_period=13):
    """
    Calculate True Strength Index (TSI).
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    long_period : int
        Long EMA period
    short_period : int
        Short EMA period
        
    Returns:
    --------
    pandas.Series
        TSI values
    """
    return _momentum_indicators.true_strength_index(data, long_period, short_period)
