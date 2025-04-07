import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.indicators.trend_indicators import (
    simple_moving_average,
    exponential_moving_average,
    moving_average_convergence_divergence,
    bollinger_bands
)

class TestTrendIndicators:
    """
    Unit tests for trend indicators in the indicator library.
    """
    
    def test_simple_moving_average(self, sample_price_data):
        """Test the simple moving average calculation."""
        # Calculate SMA with period 20
        result = simple_moving_average(sample_price_data['close'], period=20)
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert np.isnan(result.iloc[0])  # First value should be NaN
        assert np.isnan(result.iloc[18])  # 19th value should be NaN
        assert not np.isnan(result.iloc[19])  # 20th value should not be NaN
        
        # Verify calculation for a specific point
        # For SMA, we can manually calculate and compare
        manual_sma = sample_price_data['close'].iloc[0:20].mean()
        assert result.iloc[19] == pytest.approx(manual_sma)
    
    def test_exponential_moving_average(self, sample_price_data):
        """Test the exponential moving average calculation."""
        # Calculate EMA with period 20
        result = exponential_moving_average(sample_price_data['close'], period=20)
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        assert np.isnan(result.iloc[0])  # First value should be NaN
        assert not np.isnan(result.iloc[19])  # 20th value should not be NaN
        
        # EMA should be more responsive to recent price changes than SMA
        # Calculate both and compare after a price spike
        sma_result = simple_moving_average(sample_price_data['close'], period=20)
        
        # Create a copy of the data and add a price spike
        modified_data = sample_price_data.copy()
        modified_data.loc[modified_data.index[50], 'close'] += 10.0  # Add spike at position 50
        
        # Calculate indicators with the modified data
        sma_after_spike = simple_moving_average(modified_data['close'], period=20)
        ema_after_spike = exponential_moving_average(modified_data['close'], period=20)
        
        # EMA should react more strongly to the spike
        sma_change = sma_after_spike.iloc[50] - sma_result.iloc[50]
        ema_change = ema_after_spike.iloc[50] - result.iloc[50]
        
        assert ema_change > sma_change
    
    def test_moving_average_convergence_divergence(self, sample_price_data):
        """Test the MACD calculation."""
        # Calculate MACD with default parameters
        macd_line, signal_line, histogram = moving_average_convergence_divergence(
            sample_price_data['close'], 
            fast_period=12, 
            slow_period=26, 
            signal_period=9
        )
        
        # Verify the results
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        
        assert len(macd_line) == len(sample_price_data)
        assert len(signal_line) == len(sample_price_data)
        assert len(histogram) == len(sample_price_data)
        
        # MACD line should be the difference between fast and slow EMAs
        fast_ema = exponential_moving_average(sample_price_data['close'], period=12)
        slow_ema = exponential_moving_average(sample_price_data['close'], period=26)
        expected_macd = fast_ema - slow_ema
        
        # Check a few points (skipping initial NaN values)
        for i in range(30, 40):
            if not np.isnan(macd_line.iloc[i]) and not np.isnan(expected_macd.iloc[i]):
                assert macd_line.iloc[i] == pytest.approx(expected_macd.iloc[i])
        
        # Histogram should be MACD line minus signal line
        expected_histogram = macd_line - signal_line
        for i in range(40, 50):
            if not np.isnan(histogram.iloc[i]) and not np.isnan(expected_histogram.iloc[i]):
                assert histogram.iloc[i] == pytest.approx(expected_histogram.iloc[i])
    
    def test_bollinger_bands(self, sample_price_data):
        """Test the Bollinger Bands calculation."""
        # Calculate Bollinger Bands with default parameters
        upper_band, middle_band, lower_band = bollinger_bands(
            sample_price_data['close'], 
            period=20, 
            std_dev=2.0
        )
        
        # Verify the results
        assert isinstance(upper_band, pd.Series)
        assert isinstance(middle_band, pd.Series)
        assert isinstance(lower_band, pd.Series)
        
        assert len(upper_band) == len(sample_price_data)
        assert len(middle_band) == len(sample_price_data)
        assert len(lower_band) == len(sample_price_data)
        
        # Middle band should be the SMA
        sma_result = simple_moving_average(sample_price_data['close'], period=20)
        for i in range(20, 30):
            assert middle_band.iloc[i] == pytest.approx(sma_result.iloc[i])
        
        # Upper and lower bands should be SMA ± (std_dev * standard deviation)
        for i in range(20, 30):
            std = sample_price_data['close'].rolling(window=20).std().iloc[i]
            assert upper_band.iloc[i] == pytest.approx(middle_band.iloc[i] + 2.0 * std)
            assert lower_band.iloc[i] == pytest.approx(middle_band.iloc[i] - 2.0 * std)
        
        # Verify bands widen during volatile periods and narrow during stable periods
        # Create a copy with a volatile period
        modified_data = sample_price_data.copy()
        for i in range(60, 70):
            if i % 2 == 0:
                modified_data.loc[modified_data.index[i], 'close'] += 5.0
            else:
                modified_data.loc[modified_data.index[i], 'close'] -= 5.0
        
        # Calculate bands for both datasets
        _, _, _ = bollinger_bands(sample_price_data['close'], period=20, std_dev=2.0)
        upper_volatile, _, lower_volatile = bollinger_bands(modified_data['close'], period=20, std_dev=2.0)
        
        # Calculate band width for both
        original_width = upper_band.iloc[69] - lower_band.iloc[69]
        volatile_width = upper_volatile.iloc[69] - lower_volatile.iloc[69]
        
        # Band width should be greater in the volatile period
        assert volatile_width > original_width
