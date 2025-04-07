"""
Volume Indicators for Gemma Advanced Trading System.

This module provides various volume-based indicators used for market analysis and trading strategies.
"""

import numpy as np
import pandas as pd


class VolumeIndicators:
    """Class containing volume-based technical indicators."""
    
    def on_balance_volume(self, close, volume):
        """
        Calculate On Balance Volume (OBV).
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        volume : pandas.Series
            Volume series
            
        Returns:
        --------
        pandas.Series
            OBV values
        """
        # Calculate price direction
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        
        # Calculate OBV
        obv = (direction * volume).cumsum()
        
        return pd.Series(obv, index=close.index)
    
    def vwap(self, high, low, close, volume):
        """
        Calculate Volume Weighted Average Price (VWAP).
        
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
            
        Returns:
        --------
        pandas.Series
            VWAP values
        """
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate VWAP
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    def accumulation_distribution_line(self, high, low, close, volume):
        """
        Calculate Accumulation/Distribution Line.
        
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
            
        Returns:
        --------
        pandas.Series
            Accumulation/Distribution Line values
        """
        # Calculate Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], 0)  # Handle division by zero
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate Accumulation/Distribution Line
        adl = mfv.cumsum()
        
        return adl
    
    def chaikin_money_flow(self, high, low, close, volume, period=20):
        """
        Calculate Chaikin Money Flow.
        
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
            Calculation period
            
        Returns:
        --------
        pandas.Series
            Chaikin Money Flow values
        """
        # Calculate Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], 0)  # Handle division by zero
        
        # Calculate Money Flow Volume
        mfv = mfm * volume
        
        # Calculate Chaikin Money Flow
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    
    def force_index(self, close, volume, period=13):
        """
        Calculate Force Index.
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        volume : pandas.Series
            Volume series
        period : int
            EMA period
            
        Returns:
        --------
        pandas.Series
            Force Index values
        """
        # Calculate Force Index
        fi = close.diff() * volume
        
        # Apply EMA smoothing
        fi_ema = fi.ewm(span=period, adjust=False).mean()
        
        return fi_ema
    
    def ease_of_movement(self, high, low, volume, period=14):
        """
        Calculate Ease of Movement.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        volume : pandas.Series
            Volume series
        period : int
            Calculation period
            
        Returns:
        --------
        pandas.Series
            Ease of Movement values
        """
        # Calculate distance moved
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        
        # Calculate box ratio
        box_ratio = (volume / 1000000) / (high - low)
        
        # Calculate Ease of Movement
        eom = distance / box_ratio
        
        # Apply smoothing
        eom_ma = eom.rolling(window=period).mean()
        
        return eom_ma
    
    def volume_price_trend(self, close, volume):
        """
        Calculate Volume Price Trend (VPT).
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        volume : pandas.Series
            Volume series
            
        Returns:
        --------
        pandas.Series
            Volume Price Trend values
        """
        # Calculate percentage price change
        pct_change = close.pct_change()
        
        # Calculate VPT
        vpt = (volume * pct_change).cumsum()
        
        return vpt
    
    def negative_volume_index(self, close, volume):
        """
        Calculate Negative Volume Index (NVI).
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        volume : pandas.Series
            Volume series
            
        Returns:
        --------
        pandas.Series
            Negative Volume Index values
        """
        # Initialize NVI with 1000
        nvi = pd.Series(1000.0, index=close.index)
        
        # Calculate NVI
        for i in range(1, len(close)):
            if volume[i] < volume[i-1]:
                nvi[i] = nvi[i-1] * (1 + ((close[i] - close[i-1]) / close[i-1]))
            else:
                nvi[i] = nvi[i-1]
        
        return nvi
    
    def positive_volume_index(self, close, volume):
        """
        Calculate Positive Volume Index (PVI).
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        volume : pandas.Series
            Volume series
            
        Returns:
        --------
        pandas.Series
            Positive Volume Index values
        """
        # Initialize PVI with 1000
        pvi = pd.Series(1000.0, index=close.index)
        
        # Calculate PVI
        for i in range(1, len(close)):
            if volume[i] > volume[i-1]:
                pvi[i] = pvi[i-1] * (1 + ((close[i] - close[i-1]) / close[i-1]))
            else:
                pvi[i] = pvi[i-1]
        
        return pvi
    
    def volume_oscillator(self, volume, fast_period=5, slow_period=10):
        """
        Calculate Volume Oscillator.
        
        Parameters:
        -----------
        volume : pandas.Series
            Volume series
        fast_period : int
            Fast MA period
        slow_period : int
            Slow MA period
            
        Returns:
        --------
        pandas.Series
            Volume Oscillator values
        """
        # Calculate fast and slow moving averages
        fast_ma = volume.rolling(window=fast_period).mean()
        slow_ma = volume.rolling(window=slow_period).mean()
        
        # Calculate Volume Oscillator
        vo = ((fast_ma - slow_ma) / slow_ma) * 100
        
        return vo


# Create a singleton instance of VolumeIndicators
_volume_indicators = VolumeIndicators()

# Function-based interface for compatibility with main application
def on_balance_volume(close, volume):
    """
    Calculate On Balance Volume (OBV).
    
    Parameters:
    -----------
    close : pandas.Series
        Close price series
    volume : pandas.Series
        Volume series
        
    Returns:
    --------
    pandas.Series
        OBV values
    """
    return _volume_indicators.on_balance_volume(close, volume)

def volume_weighted_average_price(high, low, close, volume):
    """
    Calculate Volume Weighted Average Price (VWAP).
    
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
        
    Returns:
    --------
    pandas.Series
        VWAP values
    """
    return _volume_indicators.vwap(high, low, close, volume)

def accumulation_distribution_line(high, low, close, volume):
    """
    Calculate Accumulation/Distribution Line.
    
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
        
    Returns:
    --------
    pandas.Series
        Accumulation/Distribution Line values
    """
    return _volume_indicators.accumulation_distribution_line(high, low, close, volume)

def chaikin_money_flow(high, low, close, volume, period=20):
    """
    Calculate Chaikin Money Flow.
    
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
        Calculation period
        
    Returns:
    --------
    pandas.Series
        Chaikin Money Flow values
    """
    return _volume_indicators.chaikin_money_flow(high, low, close, volume, period)

def force_index(close, volume, period=13):
    """
    Calculate Force Index.
    
    Parameters:
    -----------
    close : pandas.Series
        Close price series
    volume : pandas.Series
        Volume series
    period : int
        EMA period
        
    Returns:
    --------
    pandas.Series
        Force Index values
    """
    return _volume_indicators.force_index(close, volume, period)

def ease_of_movement(high, low, volume, period=14):
    """
    Calculate Ease of Movement.
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    volume : pandas.Series
        Volume series
    period : int
        Calculation period
        
    Returns:
    --------
    pandas.Series
        Ease of Movement values
    """
    return _volume_indicators.ease_of_movement(high, low, volume, period)

def volume_price_trend(close, volume):
    """
    Calculate Volume Price Trend (VPT).
    
    Parameters:
    -----------
    close : pandas.Series
        Close price series
    volume : pandas.Series
        Volume series
        
    Returns:
    --------
    pandas.Series
        Volume Price Trend values
    """
    return _volume_indicators.volume_price_trend(close, volume)

def negative_volume_index(close, volume):
    """
    Calculate Negative Volume Index (NVI).
    
    Parameters:
    -----------
    close : pandas.Series
        Close price series
    volume : pandas.Series
        Volume series
        
    Returns:
    --------
    pandas.Series
        Negative Volume Index values
    """
    return _volume_indicators.negative_volume_index(close, volume)

def positive_volume_index(close, volume):
    """
    Calculate Positive Volume Index (PVI).
    
    Parameters:
    -----------
    close : pandas.Series
        Close price series
    volume : pandas.Series
        Volume series
        
    Returns:
    --------
    pandas.Series
        Positive Volume Index values
    """
    return _volume_indicators.positive_volume_index(close, volume)

def volume_oscillator(volume, fast_period=5, slow_period=10):
    """
    Calculate Volume Oscillator.
    
    Parameters:
    -----------
    volume : pandas.Series
        Volume series
    fast_period : int
        Fast MA period
    slow_period : int
        Slow MA period
        
    Returns:
    --------
    pandas.Series
        Volume Oscillator values
    """
    return _volume_indicators.volume_oscillator(volume, fast_period, slow_period)
