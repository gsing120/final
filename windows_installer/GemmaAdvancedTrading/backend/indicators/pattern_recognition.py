"""
Pattern Recognition for Gemma Advanced Trading System.

This module provides various pattern recognition algorithms used for market analysis and trading strategies.
"""

import numpy as np
import pandas as pd


class PatternRecognition:
    """Class containing pattern recognition algorithms for technical analysis."""
    
    def __init__(self):
        """Initialize the PatternRecognition class."""
        # Tolerance for pattern recognition (percentage)
        self.tolerance = 0.05
    
    def _is_equal_with_tolerance(self, a, b):
        """
        Check if two values are equal within the tolerance range.
        
        Parameters:
        -----------
        a : float
            First value
        b : float
            Second value
            
        Returns:
        --------
        bool
            True if values are equal within tolerance, False otherwise
        """
        return abs(a - b) <= (a * self.tolerance)
    
    def _is_higher_with_tolerance(self, a, b):
        """
        Check if a is higher than b with tolerance.
        
        Parameters:
        -----------
        a : float
            First value
        b : float
            Second value
            
        Returns:
        --------
        bool
            True if a is higher than b with tolerance, False otherwise
        """
        return a > b * (1 + self.tolerance)
    
    def _is_lower_with_tolerance(self, a, b):
        """
        Check if a is lower than b with tolerance.
        
        Parameters:
        -----------
        a : float
            First value
        b : float
            Second value
            
        Returns:
        --------
        bool
            True if a is lower than b with tolerance, False otherwise
        """
        return a < b * (1 - self.tolerance)
    
    def double_top(self, high, close, lookback=20, tolerance=0.02):
        """
        Identify Double Top pattern.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        close : pandas.Series
            Close price series
        lookback : int
            Lookback period for pattern identification
        tolerance : float
            Tolerance for price comparison (percentage)
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        self.tolerance = tolerance
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(lookback * 2, len(close)):
            # Get the window
            window_high = high[i-lookback*2:i]
            
            # Find the two highest points
            highest_idx = window_high.idxmax()
            highest = window_high[highest_idx]
            
            # Remove the highest point and find the second highest
            window_high_without_highest = window_high.drop(highest_idx)
            second_highest_idx = window_high_without_highest.idxmax()
            second_highest = window_high_without_highest[second_highest_idx]
            
            # Check if the two highest points are approximately equal
            if self._is_equal_with_tolerance(highest, second_highest):
                # Check if there's a significant valley between the two peaks
                between_idx = min(highest_idx, second_highest_idx) + 1
                end_idx = max(highest_idx, second_highest_idx)
                
                if between_idx < end_idx:
                    valley = close[between_idx:end_idx].min()
                    
                    # Check if the valley is significantly lower than the peaks
                    if valley < min(highest, second_highest) * 0.95:
                        # Check if the current close is below the valley (confirmation)
                        if close[i] < valley:
                            result[i] = 1
        
        return result
    
    def double_bottom(self, low, close, lookback=20, tolerance=0.02):
        """
        Identify Double Bottom pattern.
        
        Parameters:
        -----------
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        lookback : int
            Lookback period for pattern identification
        tolerance : float
            Tolerance for price comparison (percentage)
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        self.tolerance = tolerance
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(lookback * 2, len(close)):
            # Get the window
            window_low = low[i-lookback*2:i]
            
            # Find the two lowest points
            lowest_idx = window_low.idxmin()
            lowest = window_low[lowest_idx]
            
            # Remove the lowest point and find the second lowest
            window_low_without_lowest = window_low.drop(lowest_idx)
            second_lowest_idx = window_low_without_lowest.idxmin()
            second_lowest = window_low_without_lowest[second_lowest_idx]
            
            # Check if the two lowest points are approximately equal
            if self._is_equal_with_tolerance(lowest, second_lowest):
                # Check if there's a significant peak between the two bottoms
                between_idx = min(lowest_idx, second_lowest_idx) + 1
                end_idx = max(lowest_idx, second_lowest_idx)
                
                if between_idx < end_idx:
                    peak = close[between_idx:end_idx].max()
                    
                    # Check if the peak is significantly higher than the bottoms
                    if peak > max(lowest, second_lowest) * 1.05:
                        # Check if the current close is above the peak (confirmation)
                        if close[i] > peak:
                            result[i] = 1
        
        return result
    
    def head_and_shoulders(self, high, low, close, lookback=30, tolerance=0.03):
        """
        Identify Head and Shoulders pattern.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        lookback : int
            Lookback period for pattern identification
        tolerance : float
            Tolerance for price comparison (percentage)
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        self.tolerance = tolerance
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(lookback * 2, len(close)):
            # Get the window
            window_high = high[i-lookback*2:i]
            window_low = low[i-lookback*2:i]
            
            # Find the highest point (head)
            head_idx = window_high.idxmax()
            head = window_high[head_idx]
            
            # Split the window into left and right parts
            left_window = window_high[:head_idx]
            right_window = window_high[head_idx:]
            
            if len(left_window) > 5 and len(right_window) > 5:
                # Find the left shoulder
                left_shoulder_idx = left_window.idxmax()
                left_shoulder = left_window[left_shoulder_idx]
                
                # Find the right shoulder
                right_shoulder_idx = right_window.idxmax()
                right_shoulder = right_window[right_shoulder_idx]
                
                # Check if the head is higher than both shoulders
                if (head > left_shoulder * 1.05 and head > right_shoulder * 1.05 and
                    self._is_equal_with_tolerance(left_shoulder, right_shoulder)):
                    
                    # Find the neckline (connecting the lows between shoulders and head)
                    left_low_idx = window_low[left_shoulder_idx:head_idx].idxmin()
                    right_low_idx = window_low[head_idx:right_shoulder_idx].idxmin()
                    
                    if left_low_idx < right_low_idx:
                        left_low = window_low[left_low_idx]
                        right_low = window_low[right_low_idx]
                        
                        # Check if the neckline is relatively flat
                        if self._is_equal_with_tolerance(left_low, right_low):
                            # Check if the current close is below the neckline (confirmation)
                            neckline = (left_low + right_low) / 2
                            if close[i] < neckline:
                                result[i] = 1
        
        return result
    
    def inverse_head_and_shoulders(self, high, low, close, lookback=30, tolerance=0.03):
        """
        Identify Inverse Head and Shoulders pattern.
        
        Parameters:
        -----------
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
        lookback : int
            Lookback period for pattern identification
        tolerance : float
            Tolerance for price comparison (percentage)
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        self.tolerance = tolerance
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(lookback * 2, len(close)):
            # Get the window
            window_low = low[i-lookback*2:i]
            window_high = high[i-lookback*2:i]
            
            # Find the lowest point (head)
            head_idx = window_low.idxmin()
            head = window_low[head_idx]
            
            # Split the window into left and right parts
            left_window = window_low[:head_idx]
            right_window = window_low[head_idx:]
            
            if len(left_window) > 5 and len(right_window) > 5:
                # Find the left shoulder
                left_shoulder_idx = left_window.idxmin()
                left_shoulder = left_window[left_shoulder_idx]
                
                # Find the right shoulder
                right_shoulder_idx = right_window.idxmin()
                right_shoulder = right_window[right_shoulder_idx]
                
                # Check if the head is lower than both shoulders
                if (head < left_shoulder * 0.95 and head < right_shoulder * 0.95 and
                    self._is_equal_with_tolerance(left_shoulder, right_shoulder)):
                    
                    # Find the neckline (connecting the highs between shoulders and head)
                    left_high_idx = window_high[left_shoulder_idx:head_idx].idxmax()
                    right_high_idx = window_high[head_idx:right_shoulder_idx].idxmax()
                    
                    if left_high_idx < right_high_idx:
                        left_high = window_high[left_high_idx]
                        right_high = window_high[right_high_idx]
                        
                        # Check if the neckline is relatively flat
                        if self._is_equal_with_tolerance(left_high, right_high):
                            # Check if the current close is above the neckline (confirmation)
                            neckline = (left_high + right_high) / 2
                            if close[i] > neckline:
                                result[i] = 1
        
        return result
    
    def bullish_engulfing(self, open, close):
        """
        Identify Bullish Engulfing pattern.
        
        Parameters:
        -----------
        open : pandas.Series
            Open price series
        close : pandas.Series
            Close price series
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(1, len(close)):
            # Check for bullish engulfing pattern
            if (close[i-1] < open[i-1] and  # Previous day is bearish
                close[i] > open[i] and      # Current day is bullish
                open[i] < close[i-1] and    # Current open is lower than previous close
                close[i] > open[i-1]):      # Current close is higher than previous open
                
                result[i] = 1
        
        return result
    
    def bearish_engulfing(self, open, close):
        """
        Identify Bearish Engulfing pattern.
        
        Parameters:
        -----------
        open : pandas.Series
            Open price series
        close : pandas.Series
            Close price series
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(1, len(close)):
            # Check for bearish engulfing pattern
            if (close[i-1] > open[i-1] and  # Previous day is bullish
                close[i] < open[i] and      # Current day is bearish
                open[i] > close[i-1] and    # Current open is higher than previous close
                close[i] < open[i-1]):      # Current close is lower than previous open
                
                result[i] = 1
        
        return result
    
    def morning_star(self, open, high, low, close):
        """
        Identify Morning Star pattern.
        
        Parameters:
        -----------
        open : pandas.Series
            Open price series
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(2, len(close)):
            # First day: bearish candle
            if close[i-2] < open[i-2]:
                # Second day: small body (doji-like)
                body_size_1 = abs(open[i-1] - close[i-1])
                range_1 = high[i-1] - low[i-1]
                
                if body_size_1 < 0.3 * range_1:
                    # Third day: bullish candle
                    if close[i] > open[i]:
                        # Gap down between first and second day
                        if max(open[i-1], close[i-1]) < close[i-2]:
                            # Third day closes above the midpoint of the first day
                            midpoint_1 = (open[i-2] + close[i-2]) / 2
                            
                            if close[i] > midpoint_1:
                                result[i] = 1
        
        return result
    
    def evening_star(self, open, high, low, close):
        """
        Identify Evening Star pattern.
        
        Parameters:
        -----------
        open : pandas.Series
            Open price series
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(2, len(close)):
            # First day: bullish candle
            if close[i-2] > open[i-2]:
                # Second day: small body (doji-like)
                body_size_1 = abs(open[i-1] - close[i-1])
                range_1 = high[i-1] - low[i-1]
                
                if body_size_1 < 0.3 * range_1:
                    # Third day: bearish candle
                    if close[i] < open[i]:
                        # Gap up between first and second day
                        if min(open[i-1], close[i-1]) > close[i-2]:
                            # Third day closes below the midpoint of the first day
                            midpoint_1 = (open[i-2] + close[i-2]) / 2
                            
                            if close[i] < midpoint_1:
                                result[i] = 1
        
        return result
    
    def hammer(self, open, high, low, close):
        """
        Identify Hammer pattern.
        
        Parameters:
        -----------
        open : pandas.Series
            Open price series
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(1, len(close)):
            # Calculate body and shadow sizes
            body_size = abs(open[i] - close[i])
            upper_shadow = high[i] - max(open[i], close[i])
            lower_shadow = min(open[i], close[i]) - low[i]
            
            # Check for hammer pattern
            if (lower_shadow > 2 * body_size and  # Long lower shadow
                upper_shadow < 0.1 * body_size):  # Very small upper shadow
                
                # Check if in downtrend (using simple 5-day moving average)
                if i >= 5:
                    ma5 = close[i-5:i].mean()
                    if close[i] < ma5:
                        result[i] = 1
        
        return result
    
    def shooting_star(self, open, high, low, close):
        """
        Identify Shooting Star pattern.
        
        Parameters:
        -----------
        open : pandas.Series
            Open price series
        high : pandas.Series
            High price series
        low : pandas.Series
            Low price series
        close : pandas.Series
            Close price series
            
        Returns:
        --------
        pandas.Series
            Binary series where 1 indicates pattern detection
        """
        result = pd.Series(np.zeros(len(close)), index=close.index)
        
        for i in range(1, len(close)):
            # Calculate body and shadow sizes
            body_size = abs(open[i] - close[i])
            upper_shadow = high[i] - max(open[i], close[i])
            lower_shadow = min(open[i], close[i]) - low[i]
            
            # Check for shooting star pattern
            if (upper_shadow > 2 * body_size and  # Long upper shadow
                lower_shadow < 0.1 * body_size):  # Very small lower shadow
                
                # Check if in uptrend (using simple 5-day moving average)
                if i >= 5:
                    ma5 = close[i-5:i].mean()
                    if close[i] > ma5:
                        result[i] = 1
        
        return result
