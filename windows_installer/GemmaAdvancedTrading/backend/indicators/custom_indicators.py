"""
Custom Indicators for Gemma Advanced Trading System.

This module provides custom technical indicators developed specifically for the Gemma Advanced Trading System.
"""

import numpy as np
import pandas as pd


class CustomIndicators:
    """Class containing custom technical indicators."""
    
    def volume_weighted_macd(self, close, volume, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Volume-Weighted MACD.
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        volume : pandas.Series
            Volume series
        fast_period : int
            Fast EMA period
        slow_period : int
            Slow EMA period
        signal_period : int
            Signal line period
            
        Returns:
        --------
        tuple of pandas.Series
            (Volume-Weighted MACD line, Signal line, Histogram)
        """
        # Calculate volume-weighted price
        vw_price = close * volume / volume.rolling(window=fast_period).mean()
        
        # Calculate EMAs
        fast_ema = vw_price.ewm(span=fast_period, adjust=False).mean()
        slow_ema = vw_price.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def adaptive_rsi(self, close, min_period=5, max_period=30, volatility_lookback=20):
        """
        Calculate Adaptive RSI with period adjusted based on volatility.
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        min_period : int
            Minimum RSI period
        max_period : int
            Maximum RSI period
        volatility_lookback : int
            Lookback period for volatility calculation
            
        Returns:
        --------
        pandas.Series
            Adaptive RSI values
        """
        # Calculate returns
        returns = close.pct_change()
        
        # Calculate volatility (standard deviation of returns)
        volatility = returns.rolling(window=volatility_lookback).std()
        
        # Normalize volatility to range [0, 1]
        vol_min = volatility.rolling(window=100).min()
        vol_max = volatility.rolling(window=100).max()
        norm_vol = (volatility - vol_min) / (vol_max - vol_min)
        norm_vol = norm_vol.fillna(0.5)  # Default to middle value
        
        # Calculate adaptive period
        adaptive_period = min_period + (max_period - min_period) * (1 - norm_vol)
        adaptive_period = adaptive_period.round().astype(int)
        adaptive_period = adaptive_period.clip(min_period, max_period)
        
        # Initialize result series
        adaptive_rsi = pd.Series(np.nan, index=close.index)
        
        # Calculate RSI for each period
        for i in range(max_period, len(close)):
            period = adaptive_period[i]
            price_slice = close[i-period:i+1]
            
            delta = price_slice.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain[1:].mean()
            avg_loss = loss[1:].mean()
            
            if avg_loss == 0:
                adaptive_rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                adaptive_rsi[i] = 100 - (100 / (1 + rs))
        
        return adaptive_rsi
    
    def volume_breakout_indicator(self, close, volume, price_period=20, volume_period=20, price_threshold=2.0, volume_threshold=2.0):
        """
        Calculate Volume Breakout Indicator.
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        volume : pandas.Series
            Volume series
        price_period : int
            Lookback period for price standard deviation
        volume_period : int
            Lookback period for volume standard deviation
        price_threshold : float
            Number of standard deviations for price breakout
        volume_threshold : float
            Number of standard deviations for volume breakout
            
        Returns:
        --------
        pandas.Series
            Volume Breakout Indicator values (1 for bullish breakout, -1 for bearish breakout, 0 for no breakout)
        """
        # Initialize result series
        breakout = pd.Series(np.zeros(len(close)), index=close.index)
        
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate mean and standard deviation of price change
        price_mean = price_change.rolling(window=price_period).mean()
        price_std = price_change.rolling(window=price_period).std()
        
        # Calculate volume mean and standard deviation
        volume_mean = volume.rolling(window=volume_period).mean()
        volume_std = volume.rolling(window=volume_period).std()
        
        # Identify breakouts
        for i in range(max(price_period, volume_period), len(close)):
            # Check if current price change exceeds threshold
            if price_change[i] > price_mean[i] + price_threshold * price_std[i]:
                # Check if current volume exceeds threshold
                if volume[i] > volume_mean[i] + volume_threshold * volume_std[i]:
                    breakout[i] = 1  # Bullish breakout
            
            elif price_change[i] < price_mean[i] - price_threshold * price_std[i]:
                # Check if current volume exceeds threshold
                if volume[i] > volume_mean[i] + volume_threshold * volume_std[i]:
                    breakout[i] = -1  # Bearish breakout
        
        return breakout
    
    def relative_strength_index_divergence(self, close, rsi_period=14, lookback=5):
        """
        Calculate RSI Divergence.
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        rsi_period : int
            RSI calculation period
        lookback : int
            Lookback period for divergence detection
            
        Returns:
        --------
        pandas.Series
            RSI Divergence values (1 for bullish divergence, -1 for bearish divergence, 0 for no divergence)
        """
        # Calculate RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Initialize result series
        divergence = pd.Series(np.zeros(len(close)), index=close.index)
        
        # Detect divergence
        for i in range(rsi_period + lookback, len(close)):
            # Get local window
            window_close = close[i-lookback:i+1]
            window_rsi = rsi[i-lookback:i+1]
            
            # Check for bullish divergence (price makes lower low, RSI makes higher low)
            if window_close.iloc[-1] < window_close.min() and window_rsi.iloc[-1] > window_rsi.min():
                divergence[i] = 1
            
            # Check for bearish divergence (price makes higher high, RSI makes lower high)
            elif window_close.iloc[-1] > window_close.max() and window_rsi.iloc[-1] < window_rsi.max():
                divergence[i] = -1
        
        return divergence
    
    def multi_timeframe_momentum(self, close, short_period=5, medium_period=10, long_period=20):
        """
        Calculate Multi-Timeframe Momentum Indicator.
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        short_period : int
            Short-term momentum period
        medium_period : int
            Medium-term momentum period
        long_period : int
            Long-term momentum period
            
        Returns:
        --------
        pandas.Series
            Multi-Timeframe Momentum values (range from -3 to 3)
        """
        # Calculate momentum for different timeframes
        short_momentum = close.pct_change(short_period)
        medium_momentum = close.pct_change(medium_period)
        long_momentum = close.pct_change(long_period)
        
        # Initialize result series
        mtf_momentum = pd.Series(np.zeros(len(close)), index=close.index)
        
        # Combine momentum signals
        for i in range(long_period, len(close)):
            momentum_score = 0
            
            # Add short-term momentum
            if short_momentum[i] > 0:
                momentum_score += 1
            elif short_momentum[i] < 0:
                momentum_score -= 1
            
            # Add medium-term momentum
            if medium_momentum[i] > 0:
                momentum_score += 1
            elif medium_momentum[i] < 0:
                momentum_score -= 1
            
            # Add long-term momentum
            if long_momentum[i] > 0:
                momentum_score += 1
            elif long_momentum[i] < 0:
                momentum_score -= 1
            
            mtf_momentum[i] = momentum_score
        
        return mtf_momentum
    
    def volatility_adjusted_returns(self, close, period=20):
        """
        Calculate Volatility-Adjusted Returns.
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        period : int
            Calculation period
            
        Returns:
        --------
        pandas.Series
            Volatility-Adjusted Returns values
        """
        # Calculate returns
        returns = close.pct_change()
        
        # Calculate volatility (standard deviation of returns)
        volatility = returns.rolling(window=period).std()
        
        # Calculate volatility-adjusted returns
        var = returns / volatility
        
        return var
    
    def volume_price_confirmation_indicator(self, close, volume, period=20):
        """
        Calculate Volume-Price Confirmation Indicator.
        
        Parameters:
        -----------
        close : pandas.Series
            Close price series
        volume : pandas.Series
            Volume series
        period : int
            Calculation period
            
        Returns:
        --------
        pandas.Series
            Volume-Price Confirmation Indicator values
        """
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate volume change
        volume_change = volume.pct_change()
        
        # Calculate correlation between price and volume changes
        vpci = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(period, len(close)):
            # Calculate correlation for the window
            correlation = np.corrcoef(
                price_change[i-period+1:i+1].fillna(0),
                volume_change[i-period+1:i+1].fillna(0)
            )[0, 1]
            
            vpci[i] = correlation
        
        return vpci
    
    def support_resistance_zones(self, high, low, close, period=50, zone_threshold=0.02):
        """
        Identify Support and Resistance Zones.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        period : int
            Lookback period for zone identification
        zone_threshold : float
            Threshold for zone width (percentage of price)
            
        Returns:
        --------
        tuple of pandas.Series
            (Support zones, Resistance zones)
        """
        # Initialize result series
        support = pd.Series(np.zeros(len(close)), index=close.index)
        resistance = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(period, len(close)):
            # Get the window
            window_high = high[i-period:i]
            window_low = low[i-period:i]
            
            # Find potential support levels (recent lows)
            potential_supports = []
            for j in range(1, len(window_low)-1):
                if window_low.iloc[j] < window_low.iloc[j-1] and window_low.iloc[j] < window_low.iloc[j+1]:
                    potential_supports.append(window_low.iloc[j])
            
            # Find potential resistance levels (recent highs)
            potential_resistances = []
            for j in range(1, len(window_high)-1):
                if window_high.iloc[j] > window_high.iloc[j-1] and window_high.iloc[j] > window_high.iloc[j+1]:
                    potential_resistances.append(window_high.iloc[j])
            
            # Check if current price is near support or resistance
            current_price = close[i]
            
            # Check support
            for level in potential_supports:
                if abs(current_price - level) / current_price < zone_threshold:
                    support[i] = level
                    break
            
            # Check resistance
            for level in potential_resistances:
                if abs(current_price - level) / current_price < zone_threshold:
                    resistance[i] = level
                    break
        
        return support, resistance
