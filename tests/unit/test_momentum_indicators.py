import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.indicators.momentum_indicators import (
    relative_strength_index,
    stochastic_oscillator,
    commodity_channel_index,
    rate_of_change
)

class TestMomentumIndicators:
    """
    Unit tests for momentum indicators in the indicator library.
    """
    
    def test_relative_strength_index(self, sample_price_data):
        """Test the RSI calculation."""
        # Calculate RSI with period 14
        result = relative_strength_index(sample_price_data['close'], period=14)
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert valid_values.min() >= 0
        assert valid_values.max() <= 100
        
        # First 14 values should be NaN
        for i in range(14):
            assert np.isnan(result.iloc[i])
        
        # Value at index 14 should not be NaN
        assert not np.isnan(result.iloc[14])
        
        # Test RSI behavior with uptrend and downtrend
        # Create data with clear uptrend
        uptrend_data = pd.Series(np.linspace(100, 200, 50))
        uptrend_rsi = relative_strength_index(uptrend_data, period=14)
        
        # Create data with clear downtrend
        downtrend_data = pd.Series(np.linspace(200, 100, 50))
        downtrend_rsi = relative_strength_index(downtrend_data, period=14)
        
        # RSI should be higher in uptrend and lower in downtrend
        assert uptrend_rsi.iloc[-1] > 50
        assert downtrend_rsi.iloc[-1] < 50
    
    def test_stochastic_oscillator(self, sample_price_data):
        """Test the Stochastic Oscillator calculation."""
        # Calculate Stochastic Oscillator with default parameters
        k_line, d_line = stochastic_oscillator(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            k_period=14,
            d_period=3,
            slowing=3
        )
        
        # Verify the results
        assert isinstance(k_line, pd.Series)
        assert isinstance(d_line, pd.Series)
        
        assert len(k_line) == len(sample_price_data)
        assert len(d_line) == len(sample_price_data)
        
        # Stochastic values should be between 0 and 100
        valid_k_values = k_line.dropna()
        valid_d_values = d_line.dropna()
        
        assert valid_k_values.min() >= 0
        assert valid_k_values.max() <= 100
        assert valid_d_values.min() >= 0
        assert valid_d_values.max() <= 100
        
        # First 14 values of K should be NaN
        for i in range(14):
            assert np.isnan(k_line.iloc[i])
        
        # First 16 values of D should be NaN (14 + 3 - 1)
        for i in range(16):
            assert np.isnan(d_line.iloc[i])
        
        # D line should be smoother than K line
        # Calculate standard deviation of both lines (excluding NaN values)
        k_std = k_line.iloc[16:].std()
        d_std = d_line.iloc[16:].std()
        
        assert d_std < k_std
    
    def test_commodity_channel_index(self, sample_price_data):
        """Test the CCI calculation."""
        # Calculate CCI with period 20
        result = commodity_channel_index(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            period=20
        )
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # First 19 values should be NaN
        for i in range(19):
            assert np.isnan(result.iloc[i])
        
        # Value at index 19 should not be NaN
        assert not np.isnan(result.iloc[19])
        
        # CCI typically ranges between -100 and +100 in normal market conditions
        # but can go beyond these values in extreme conditions
        # Let's check if most values are within this range
        valid_values = result.dropna()
        within_range = ((valid_values >= -100) & (valid_values <= 100)).mean()
        
        # At least 70% of values should be within the typical range
        assert within_range > 0.7
        
        # Test CCI behavior with uptrend and downtrend
        # Create data with clear uptrend
        uptrend_high = pd.Series(np.linspace(110, 220, 50))
        uptrend_low = pd.Series(np.linspace(100, 200, 50))
        uptrend_close = pd.Series(np.linspace(105, 210, 50))
        
        uptrend_cci = commodity_channel_index(uptrend_high, uptrend_low, uptrend_close, period=20)
        
        # Create data with clear downtrend
        downtrend_high = pd.Series(np.linspace(220, 110, 50))
        downtrend_low = pd.Series(np.linspace(200, 100, 50))
        downtrend_close = pd.Series(np.linspace(210, 105, 50))
        
        downtrend_cci = commodity_channel_index(downtrend_high, downtrend_low, downtrend_close, period=20)
        
        # CCI should be positive in uptrend and negative in downtrend
        assert uptrend_cci.iloc[-1] > 0
        assert downtrend_cci.iloc[-1] < 0
    
    def test_rate_of_change(self, sample_price_data):
        """Test the Rate of Change calculation."""
        # Calculate ROC with period 10
        result = rate_of_change(sample_price_data['close'], period=10)
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # First 10 values should be NaN
        for i in range(10):
            assert np.isnan(result.iloc[i])
        
        # Value at index 10 should not be NaN
        assert not np.isnan(result.iloc[10])
        
        # Manually calculate ROC for a specific point and compare
        # ROC = ((current_price - price_n_periods_ago) / price_n_periods_ago) * 100
        i = 20
        expected_roc = ((sample_price_data['close'].iloc[i] - sample_price_data['close'].iloc[i-10]) / 
                        sample_price_data['close'].iloc[i-10]) * 100
        
        assert result.iloc[i] == pytest.approx(expected_roc)
        
        # Test ROC behavior with uptrend and downtrend
        # Create data with clear uptrend
        uptrend_data = pd.Series(np.linspace(100, 200, 50))
        uptrend_roc = rate_of_change(uptrend_data, period=10)
        
        # Create data with clear downtrend
        downtrend_data = pd.Series(np.linspace(200, 100, 50))
        downtrend_roc = rate_of_change(downtrend_data, period=10)
        
        # ROC should be positive in uptrend and negative in downtrend
        assert uptrend_roc.iloc[-1] > 0
        assert downtrend_roc.iloc[-1] < 0
