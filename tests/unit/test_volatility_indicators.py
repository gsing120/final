import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.indicators.volatility_indicators import (
    average_true_range,
    bollinger_bandwidth,
    standard_deviation,
    keltner_channels
)

class TestVolatilityIndicators:
    """
    Unit tests for volatility indicators in the indicator library.
    """
    
    def test_average_true_range(self, sample_price_data):
        """Test the Average True Range calculation."""
        # Calculate ATR with period 14
        result = average_true_range(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            period=14
        )
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # ATR should be positive
        valid_values = result.dropna()
        assert (valid_values > 0).all()
        
        # First 13 values should be NaN
        for i in range(13):
            assert np.isnan(result.iloc[i])
        
        # Value at index 13 should not be NaN
        assert not np.isnan(result.iloc[13])
        
        # Test ATR behavior with increased volatility
        # Create a copy with a volatile period
        modified_data = sample_price_data.copy()
        for i in range(60, 70):
            if i % 2 == 0:
                modified_data.loc[modified_data.index[i], 'high'] += 5.0
            else:
                modified_data.loc[modified_data.index[i], 'low'] -= 5.0
        
        # Calculate ATR for both datasets
        normal_atr = average_true_range(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            period=14
        )
        
        volatile_atr = average_true_range(
            modified_data['high'],
            modified_data['low'],
            modified_data['close'],
            period=14
        )
        
        # ATR should be higher in the volatile period
        assert volatile_atr.iloc[70] > normal_atr.iloc[70]
    
    def test_bollinger_bandwidth(self, sample_price_data):
        """Test the Bollinger Bandwidth calculation."""
        # Calculate Bollinger Bandwidth with default parameters
        result = bollinger_bandwidth(
            sample_price_data['close'],
            period=20,
            std_dev=2.0
        )
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # Bandwidth should be positive
        valid_values = result.dropna()
        assert (valid_values > 0).all()
        
        # First 19 values should be NaN
        for i in range(19):
            assert np.isnan(result.iloc[i])
        
        # Value at index 19 should not be NaN
        assert not np.isnan(result.iloc[19])
        
        # Test bandwidth behavior with increased volatility
        # Create a copy with a volatile period
        modified_data = sample_price_data.copy()
        for i in range(60, 70):
            if i % 2 == 0:
                modified_data.loc[modified_data.index[i], 'close'] += 5.0
            else:
                modified_data.loc[modified_data.index[i], 'close'] -= 5.0
        
        # Calculate bandwidth for both datasets
        normal_bandwidth = bollinger_bandwidth(
            sample_price_data['close'],
            period=20,
            std_dev=2.0
        )
        
        volatile_bandwidth = bollinger_bandwidth(
            modified_data['close'],
            period=20,
            std_dev=2.0
        )
        
        # Bandwidth should be higher in the volatile period
        assert volatile_bandwidth.iloc[70] > normal_bandwidth.iloc[70]
    
    def test_standard_deviation(self, sample_price_data):
        """Test the Standard Deviation calculation."""
        # Calculate Standard Deviation with period 20
        result = standard_deviation(sample_price_data['close'], period=20)
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # Standard Deviation should be positive
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        
        # First 19 values should be NaN
        for i in range(19):
            assert np.isnan(result.iloc[i])
        
        # Value at index 19 should not be NaN
        assert not np.isnan(result.iloc[19])
        
        # Manually calculate standard deviation for a specific point and compare
        i = 30
        expected_std = sample_price_data['close'].iloc[i-19:i+1].std()
        assert result.iloc[i] == pytest.approx(expected_std)
        
        # Test standard deviation behavior with increased volatility
        # Create a copy with a volatile period
        modified_data = sample_price_data.copy()
        for i in range(60, 70):
            if i % 2 == 0:
                modified_data.loc[modified_data.index[i], 'close'] += 5.0
            else:
                modified_data.loc[modified_data.index[i], 'close'] -= 5.0
        
        # Calculate standard deviation for both datasets
        normal_std = standard_deviation(sample_price_data['close'], period=20)
        volatile_std = standard_deviation(modified_data['close'], period=20)
        
        # Standard deviation should be higher in the volatile period
        assert volatile_std.iloc[70] > normal_std.iloc[70]
    
    def test_keltner_channels(self, sample_price_data):
        """Test the Keltner Channels calculation."""
        # Calculate Keltner Channels with default parameters
        upper_band, middle_band, lower_band = keltner_channels(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            ema_period=20,
            atr_period=10,
            atr_multiplier=2.0
        )
        
        # Verify the results
        assert isinstance(upper_band, pd.Series)
        assert isinstance(middle_band, pd.Series)
        assert isinstance(lower_band, pd.Series)
        
        assert len(upper_band) == len(sample_price_data)
        assert len(middle_band) == len(sample_price_data)
        assert len(lower_band) == len(sample_price_data)
        
        # First 19 values should be NaN (due to EMA period)
        for i in range(19):
            assert np.isnan(upper_band.iloc[i])
            assert np.isnan(middle_band.iloc[i])
            assert np.isnan(lower_band.iloc[i])
        
        # Value at index 19 should not be NaN
        assert not np.isnan(upper_band.iloc[19])
        assert not np.isnan(middle_band.iloc[19])
        assert not np.isnan(lower_band.iloc[19])
        
        # Upper band should be greater than middle band, which should be greater than lower band
        for i in range(20, len(sample_price_data)):
            assert upper_band.iloc[i] > middle_band.iloc[i]
            assert middle_band.iloc[i] > lower_band.iloc[i]
        
        # Test channel width behavior with increased volatility
        # Create a copy with a volatile period
        modified_data = sample_price_data.copy()
        for i in range(60, 70):
            if i % 2 == 0:
                modified_data.loc[modified_data.index[i], 'high'] += 5.0
            else:
                modified_data.loc[modified_data.index[i], 'low'] -= 5.0
        
        # Calculate channels for both datasets
        normal_upper, _, normal_lower = keltner_channels(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            ema_period=20,
            atr_period=10,
            atr_multiplier=2.0
        )
        
        volatile_upper, _, volatile_lower = keltner_channels(
            modified_data['high'],
            modified_data['low'],
            modified_data['close'],
            ema_period=20,
            atr_period=10,
            atr_multiplier=2.0
        )
        
        # Channel width should be greater in the volatile period
        normal_width = normal_upper.iloc[70] - normal_lower.iloc[70]
        volatile_width = volatile_upper.iloc[70] - volatile_lower.iloc[70]
        
        assert volatile_width > normal_width
