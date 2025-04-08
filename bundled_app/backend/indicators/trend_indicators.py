"""
Trend Indicators for Gemma Advanced Trading System.

This module provides various trend indicators used for market analysis and trading strategies.
"""

import numpy as np
import pandas as pd


class TrendIndicators:
    """Class containing trend-based technical indicators."""
    
    def sma(self, data, period=20):
        """
        Calculate Simple Moving Average (SMA).
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Number of periods for moving average calculation
            
        Returns:
        --------
        pandas.Series
            Simple Moving Average values
        """
        return data.rolling(window=period).mean()
    
    def ema(self, data, period=20, smoothing=2):
        """
        Calculate Exponential Moving Average (EMA).
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Number of periods for moving average calculation
        smoothing : int
            Smoothing factor
            
        Returns:
        --------
        pandas.Series
            Exponential Moving Average values
        """
        # Initialize with NaN values for the first period-1 elements
        ema = pd.Series(np.nan, index=data.index)
        
        # Use SMA as the initial value for EMA calculation
        ema.iloc[period-1] = data.iloc[:period].mean()
        
        # Calculate EMA for the rest of the series
        for i in range(period, len(data)):
            ema.iloc[i] = (data.iloc[i] * (2 / (period + 1)) + 
                          ema.iloc[i-1] * (1 - (2 / (period + 1))))
        
        return ema
    
    def macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        fast_period : int
            Fast EMA period
        slow_period : int
            Slow EMA period
        signal_period : int
            Signal line period
            
        Returns:
        --------
        tuple of pandas.Series
            (MACD line, Signal line, Histogram)
        """
        fast_ema = self.ema(data, period=fast_period)
        slow_ema = self.ema(data, period=slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, period=signal_period)
        
        # Calculate histogram and handle NaN values consistently
        histogram = pd.Series(np.nan, index=data.index)
        for i in range(len(data)):
            if not np.isnan(macd_line.iloc[i]) and not np.isnan(signal_line.iloc[i]):
                histogram.iloc[i] = macd_line.iloc[i] - signal_line.iloc[i]
        
        return macd_line, signal_line, histogram
    
    def bollinger_bands(self, data, period=20, std_dev=2):
        """
        Calculate Bollinger Bands.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Number of periods for moving average calculation
        std_dev : int
            Number of standard deviations for bands
            
        Returns:
        --------
        tuple of pandas.Series
            (Upper band, Middle band, Lower band)
        """
        middle_band = self.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def parabolic_sar(self, high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
        """
        Calculate Parabolic SAR (Stop and Reverse).
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        af_start : float
            Starting acceleration factor
        af_increment : float
            Acceleration factor increment
        af_max : float
            Maximum acceleration factor
            
        Returns:
        --------
        pandas.Series
            Parabolic SAR values
        """
        length = len(high)
        psar = pd.Series(np.nan, index=high.index)
        
        # Initialize trend, extreme point, and acceleration factor
        trend = 1  # 1 for uptrend, -1 for downtrend
        ep = low[0]  # Extreme point
        af = af_start  # Acceleration factor
        
        # Set first SAR value
        psar[1] = high[0]
        
        for i in range(2, length):
            # Previous SAR value
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            
            # Ensure SAR is below/above the previous two candles' lows/highs in up/downtrends
            if trend == 1:
                psar[i] = min(psar[i], low[i-1], low[i-2])
                # Check if SAR is penetrated, if so, switch trend
                if psar[i] > low[i]:
                    trend = -1
                    psar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    # If not penetrated, check for new EP
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)
            else:
                psar[i] = max(psar[i], high[i-1], high[i-2])
                # Check if SAR is penetrated, if so, switch trend
                if psar[i] < high[i]:
                    trend = 1
                    psar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    # If not penetrated, check for new EP
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)
        
        return psar
    
    def ichimoku_cloud(self, high, low, close, tenkan_period=9, kijun_period=26, senkou_b_period=52, chikou_period=26):
        """
        Calculate Ichimoku Cloud components.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        tenkan_period : int
            Tenkan-sen (Conversion Line) period
        kijun_period : int
            Kijun-sen (Base Line) period
        senkou_b_period : int
            Senkou Span B period
        chikou_period : int
            Chikou Span period
            
        Returns:
        --------
        tuple of pandas.Series
            (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
        """
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan_period).max() + 
                      low.rolling(window=tenkan_period).min()) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun_period).max() + 
                     low.rolling(window=kijun_period).min()) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=senkou_b_period).max() + 
                          low.rolling(window=senkou_b_period).min()) / 2).shift(kijun_period)
        
        # Calculate Chikou Span (Lagging Span)
        chikou_span = close.shift(-chikou_period)
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    def adx(self, high, low, close, period=14):
        """
        Calculate Average Directional Index (ADX).
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        period : int
            ADX period
            
        Returns:
        --------
        pandas.Series
            ADX values
        """
        # Calculate True Range
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
        plus_dm = high - high.shift(1)
        minus_dm = low.shift(1) - low
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate Directional Index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def supertrend(self, high, low, close, period=10, multiplier=3.0):
        """
        Calculate Supertrend indicator.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        period : int
            ATR period
        multiplier : float
            ATR multiplier
            
        Returns:
        --------
        tuple of pandas.Series
            (Supertrend, Trend direction)
        """
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate Upper and Lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize Supertrend and trend direction
        supertrend = pd.Series(np.nan, index=close.index)
        trend = pd.Series(np.nan, index=close.index)
        
        # Set initial values
        supertrend.iloc[period] = upper_band.iloc[period]
        trend.iloc[period] = 1  # 1 for uptrend, -1 for downtrend
        
        # Calculate Supertrend
        for i in range(period + 1, len(close)):
            if close.iloc[i-1] <= supertrend.iloc[i-1]:
                # Previous trend was down
                if close.iloc[i] > upper_band.iloc[i]:
                    # Price crossed above upper band, switch to uptrend
                    supertrend.iloc[i] = lower_band.iloc[i]
                    trend.iloc[i] = -1
                else:
                    # Continue downtrend
                    supertrend.iloc[i] = upper_band.iloc[i]
                    trend.iloc[i] = 1
            else:
                # Previous trend was up
                if close.iloc[i] < lower_band.iloc[i]:
                    # Price crossed below lower band, switch to downtrend
                    supertrend.iloc[i] = upper_band.iloc[i]
                    trend.iloc[i] = 1
                else:
                    # Continue uptrend
                    supertrend.iloc[i] = lower_band.iloc[i]
                    trend.iloc[i] = -1
        
        return supertrend, trend


# Create a singleton instance of TrendIndicators
_trend_indicators = TrendIndicators()

# Function-based interface for compatibility with main application
def simple_moving_average(data, period=20):
    """
    Calculate Simple Moving Average (SMA).
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    period : int
        Number of periods for moving average calculation
        
    Returns:
    --------
    pandas.Series
        Simple Moving Average values
    """
    return _trend_indicators.sma(data, period)

def exponential_moving_average(data, period=20, smoothing=2):
    """
    Calculate Exponential Moving Average (EMA).
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    period : int
        Number of periods for moving average calculation
    smoothing : int
        Smoothing factor
        
    Returns:
    --------
    pandas.Series
        Exponential Moving Average values
    """
    return _trend_indicators.ema(data, period, smoothing)

def moving_average_convergence_divergence(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    fast_period : int
        Fast EMA period
    slow_period : int
        Slow EMA period
    signal_period : int
        Signal line period
        
    Returns:
    --------
    tuple of pandas.Series
        (MACD line, Signal line, Histogram)
    """
    return _trend_indicators.macd(data, fast_period, slow_period, signal_period)

def bollinger_bands(data, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    
    Parameters:
    -----------
    data : pandas.Series
        Price series data
    period : int
        Number of periods for moving average calculation
    std_dev : int
        Number of standard deviations for bands
        
    Returns:
    --------
    tuple of pandas.Series
        (Upper band, Middle band, Lower band)
    """
    return _trend_indicators.bollinger_bands(data, period, std_dev)

def parabolic_sar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
    """
    Calculate Parabolic SAR (Stop and Reverse).
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    af_start : float
        Starting acceleration factor
    af_increment : float
        Acceleration factor increment
    af_max : float
        Maximum acceleration factor
        
    Returns:
    --------
    pandas.Series
        Parabolic SAR values
    """
    return _trend_indicators.parabolic_sar(high, low, af_start, af_increment, af_max)

def ichimoku_cloud(high, low, close, tenkan_period=9, kijun_period=26, senkou_b_period=52, chikou_period=26):
    """
    Calculate Ichimoku Cloud components.
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    tenkan_period : int
        Tenkan-sen (Conversion Line) period
    kijun_period : int
        Kijun-sen (Base Line) period
    senkou_b_period : int
        Senkou Span B period
    chikou_period : int
        Chikou Span period
        
    Returns:
    --------
    tuple of pandas.Series
        (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
    """
    return _trend_indicators.ichimoku_cloud(high, low, close, tenkan_period, kijun_period, senkou_b_period, chikou_period)

def average_directional_index(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    period : int
        ADX period
        
    Returns:
    --------
    pandas.Series
        ADX values
    """
    return _trend_indicators.adx(high, low, close, period)

def supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    
    Parameters:
    -----------
    high : pandas.Series
        High price series
    low : pandas.Series
        Low price series
    close : pandas.Series
        Close price series
    period : int
        ATR period
    multiplier : float
        ATR multiplier
        
    Returns:
    --------
    tuple of pandas.Series
        (Supertrend, Trend direction)
    """
    return _trend_indicators.supertrend(high, low, close, period, multiplier)
