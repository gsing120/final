"""
Cycle Indicators for Gemma Advanced Trading System.

This module provides various cycle-based indicators used for market analysis and trading strategies.
"""

import numpy as np
import pandas as pd
from scipy import signal


class CycleIndicators:
    """Class containing cycle-based technical indicators."""
    
    def hilbert_transform(self, data, period=7):
        """
        Calculate Hilbert Transform Indicator.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Calculation period
            
        Returns:
        --------
        tuple of pandas.Series
            (In-phase, Quadrature)
        """
        # Smooth the data
        smooth = data.rolling(window=period).mean()
        
        # Calculate the Hilbert Transform
        in_phase = pd.Series(np.zeros(len(data)), index=data.index)
        quadrature = pd.Series(np.zeros(len(data)), index=data.index)
        
        # Simple implementation of Hilbert Transform
        for i in range(period, len(data)):
            in_phase[i] = smooth[i]
            quadrature[i] = smooth[i-period//2]
        
        return in_phase, quadrature
    
    def sine_wave_indicator(self, data, period=10):
        """
        Calculate Sine Wave Indicator.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Cycle period
            
        Returns:
        --------
        tuple of pandas.Series
            (Sine, Lead Sine)
        """
        # Calculate the Hilbert Transform
        in_phase, quadrature = self.hilbert_transform(data, period)
        
        # Calculate the Sine Wave Indicator
        sine = pd.Series(np.zeros(len(data)), index=data.index)
        lead_sine = pd.Series(np.zeros(len(data)), index=data.index)
        
        for i in range(period, len(data)):
            if in_phase[i] != 0 and quadrature[i] != 0:
                # Calculate the phase angle
                phase = np.arctan2(in_phase[i], quadrature[i])
                
                # Calculate the sine and lead sine
                sine[i] = np.sin(phase)
                lead_sine[i] = np.sin(phase + np.pi/4)  # 45 degrees lead
        
        return sine, lead_sine
    
    def dominant_cycle_period(self, data, min_period=10, max_period=100):
        """
        Calculate Dominant Cycle Period using FFT.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        min_period : int
            Minimum cycle period to consider
        max_period : int
            Maximum cycle period to consider
            
        Returns:
        --------
        pandas.Series
            Dominant Cycle Period values
        """
        # Initialize the result series
        dcp = pd.Series(np.zeros(len(data)), index=data.index)
        
        # Use a rolling window approach
        window_size = max_period * 2
        
        for i in range(window_size, len(data)):
            # Get the data window
            window = data[i-window_size:i].values
            
            # Detrend the data
            detrended = signal.detrend(window)
            
            # Apply FFT
            fft_result = np.fft.fft(detrended)
            fft_freq = np.fft.fftfreq(len(detrended))
            
            # Get the positive frequencies only
            positive_freq_idx = np.where(fft_freq > 0)[0]
            positive_freq = fft_freq[positive_freq_idx]
            power = np.abs(fft_result[positive_freq_idx])
            
            # Convert frequencies to periods
            periods = 1.0 / positive_freq
            
            # Filter periods within the specified range
            valid_idx = np.where((periods >= min_period) & (periods <= max_period))[0]
            
            if len(valid_idx) > 0:
                # Find the period with the maximum power
                max_power_idx = np.argmax(power[valid_idx])
                dominant_period = periods[valid_idx[max_power_idx]]
                dcp[i] = dominant_period
            else:
                # Default to the average of min and max if no valid period is found
                dcp[i] = (min_period + max_period) / 2
        
        return dcp
    
    def stochastic_rsi_oscillator(self, data, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        """
        Calculate Stochastic RSI Oscillator.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        rsi_period : int
            RSI calculation period
        stoch_period : int
            Stochastic calculation period
        k_period : int
            %K smoothing period
        d_period : int
            %D smoothing period
            
        Returns:
        --------
        tuple of pandas.Series
            (Stochastic RSI %K, Stochastic RSI %D)
        """
        # Calculate RSI
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic RSI
        stoch_rsi = pd.Series(np.zeros(len(data)), index=data.index)
        
        for i in range(stoch_period, len(rsi)):
            rsi_min = rsi[i-stoch_period:i].min()
            rsi_max = rsi[i-stoch_period:i].max()
            
            if rsi_max - rsi_min != 0:
                stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min)
            else:
                stoch_rsi[i] = 0.5  # Default to middle value if range is zero
        
        # Calculate %K and %D
        k = stoch_rsi.rolling(window=k_period).mean() * 100
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def fisher_transform(self, data, period=10):
        """
        Calculate Fisher Transform.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Calculation period
            
        Returns:
        --------
        pandas.Series
            Fisher Transform values
        """
        # Calculate the median price
        median_price = data
        
        # Initialize the result series
        fisher = pd.Series(np.zeros(len(data)), index=data.index)
        
        for i in range(period, len(data)):
            # Get the highest and lowest values in the period
            highest = median_price[i-period:i].max()
            lowest = median_price[i-period:i].min()
            
            # Normalize the price between -1 and 1
            if highest != lowest:
                value = 2 * ((median_price[i] - lowest) / (highest - lowest) - 0.5)
            else:
                value = 0
            
            # Bound the value between -0.999 and 0.999 to avoid infinity
            value = max(min(value, 0.999), -0.999)
            
            # Apply the Fisher Transform
            fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        return fisher
    
    def ehlers_mesa_indicator(self, data, period=14):
        """
        Calculate Ehlers MESA Indicator.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Calculation period
            
        Returns:
        --------
        tuple of pandas.Series
            (MESA Indicator, MESA Phase)
        """
        # Smooth the data
        smooth = data.rolling(window=period).mean()
        
        # Initialize the result series
        mesa = pd.Series(np.zeros(len(data)), index=data.index)
        phase = pd.Series(np.zeros(len(data)), index=data.index)
        
        # Calculate the MESA Indicator
        for i in range(period*2, len(data)):
            # Simple implementation of MESA
            # In a real implementation, this would use more complex calculations
            # involving quadrature filters and phase accumulation
            
            # Calculate a simple oscillator
            osc = smooth[i] - smooth[i-period]
            
            # Calculate the phase
            phase[i] = np.arctan2(osc, smooth[i] - smooth[i-period//2])
            
            # Calculate the MESA indicator
            mesa[i] = np.sin(phase[i])
        
        return mesa, phase
    
    def center_of_gravity(self, data, period=10):
        """
        Calculate Center of Gravity Oscillator.
        
        Parameters:
        -----------
        data : pandas.Series
            Price series data
        period : int
            Calculation period
            
        Returns:
        --------
        pandas.Series
            Center of Gravity Oscillator values
        """
        # Initialize the result series
        cog = pd.Series(np.zeros(len(data)), index=data.index)
        
        for i in range(period, len(data)):
            # Get the window
            window = data[i-period+1:i+1].values
            
            # Calculate numerator and denominator
            numerator = sum(window * np.arange(1, period+1))
            denominator = sum(window)
            
            # Calculate Center of Gravity
            if denominator != 0:
                cog[i] = -numerator / denominator + (period + 1) / 2
            else:
                cog[i] = 0
        
        return cog
