import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.indicators.volume_indicators import (
    on_balance_volume,
    volume_weighted_average_price,
    accumulation_distribution_line,
    money_flow_index
)

class TestVolumeIndicators:
    """
    Unit tests for volume indicators in the indicator library.
    """
    
    def test_on_balance_volume(self, sample_price_data):
        """Test the On Balance Volume calculation."""
        # Calculate OBV
        result = on_balance_volume(sample_price_data['close'], sample_price_data['volume'])
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # First value should not be NaN
        assert not np.isnan(result.iloc[0])
        
        # Manually calculate OBV for a few points and compare
        # OBV starts at the first volume value
        expected_obv = [sample_price_data['volume'].iloc[0]]
        
        for i in range(1, 10):
            if sample_price_data['close'].iloc[i] > sample_price_data['close'].iloc[i-1]:
                # Price up, add volume
                expected_obv.append(expected_obv[-1] + sample_price_data['volume'].iloc[i])
            elif sample_price_data['close'].iloc[i] < sample_price_data['close'].iloc[i-1]:
                # Price down, subtract volume
                expected_obv.append(expected_obv[-1] - sample_price_data['volume'].iloc[i])
            else:
                # Price unchanged, OBV unchanged
                expected_obv.append(expected_obv[-1])
        
        for i in range(10):
            assert result.iloc[i] == pytest.approx(expected_obv[i])
        
        # Test OBV behavior with uptrend and downtrend
        # Create data with clear uptrend
        uptrend_close = pd.Series(np.linspace(100, 200, 50))
        uptrend_volume = pd.Series(np.random.randint(1000, 10000, 50))
        uptrend_obv = on_balance_volume(uptrend_close, uptrend_volume)
        
        # Create data with clear downtrend
        downtrend_close = pd.Series(np.linspace(200, 100, 50))
        downtrend_volume = pd.Series(np.random.randint(1000, 10000, 50))
        downtrend_obv = on_balance_volume(downtrend_close, downtrend_volume)
        
        # OBV should increase in uptrend and decrease in downtrend
        assert uptrend_obv.iloc[-1] > uptrend_obv.iloc[0]
        assert downtrend_obv.iloc[-1] < downtrend_obv.iloc[0]
    
    def test_volume_weighted_average_price(self, sample_price_data):
        """Test the Volume Weighted Average Price calculation."""
        # Calculate VWAP with period 20
        result = volume_weighted_average_price(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume'],
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
        
        # Manually calculate VWAP for a specific point and compare
        i = 30
        typical_prices = (sample_price_data['high'].iloc[i-19:i+1] + 
                          sample_price_data['low'].iloc[i-19:i+1] + 
                          sample_price_data['close'].iloc[i-19:i+1]) / 3
        
        volume = sample_price_data['volume'].iloc[i-19:i+1]
        expected_vwap = (typical_prices * volume).sum() / volume.sum()
        
        assert result.iloc[i] == pytest.approx(expected_vwap)
        
        # VWAP should be between the highest and lowest price in the period
        for i in range(20, len(sample_price_data)):
            period_high = sample_price_data['high'].iloc[i-19:i+1].max()
            period_low = sample_price_data['low'].iloc[i-19:i+1].min()
            
            assert result.iloc[i] <= period_high
            assert result.iloc[i] >= period_low
    
    def test_accumulation_distribution_line(self, sample_price_data):
        """Test the Accumulation/Distribution Line calculation."""
        # Calculate A/D Line
        result = accumulation_distribution_line(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # First value should not be NaN
        assert not np.isnan(result.iloc[0])
        
        # Manually calculate A/D Line for a few points and compare
        # A/D Line starts at 0 by default
        expected_ad = [0]
        
        for i in range(1, 10):
            # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
            high = sample_price_data['high'].iloc[i]
            low = sample_price_data['low'].iloc[i]
            close = sample_price_data['close'].iloc[i]
            volume = sample_price_data['volume'].iloc[i]
            
            if high == low:  # Avoid division by zero
                mfm = 0
            else:
                mfm = ((close - low) - (high - close)) / (high - low)
            
            # Money Flow Volume = Money Flow Multiplier * Volume
            mfv = mfm * volume
            
            # A/D Line = Previous A/D Line + Money Flow Volume
            expected_ad.append(expected_ad[-1] + mfv)
        
        for i in range(10):
            assert result.iloc[i] == pytest.approx(expected_ad[i])
        
        # Test A/D Line behavior with uptrend and downtrend
        # Create data with clear uptrend and closing near highs
        uptrend_high = pd.Series(np.linspace(110, 220, 50))
        uptrend_low = pd.Series(np.linspace(100, 200, 50))
        uptrend_close = pd.Series(np.linspace(108, 218, 50))  # Close near high
        uptrend_volume = pd.Series(np.random.randint(1000, 10000, 50))
        
        uptrend_ad = accumulation_distribution_line(uptrend_high, uptrend_low, uptrend_close, uptrend_volume)
        
        # Create data with clear downtrend and closing near lows
        downtrend_high = pd.Series(np.linspace(220, 110, 50))
        downtrend_low = pd.Series(np.linspace(200, 100, 50))
        downtrend_close = pd.Series(np.linspace(202, 102, 50))  # Close near low
        downtrend_volume = pd.Series(np.random.randint(1000, 10000, 50))
        
        downtrend_ad = accumulation_distribution_line(downtrend_high, downtrend_low, downtrend_close, downtrend_volume)
        
        # A/D Line should increase in uptrend with closes near highs
        # and decrease in downtrend with closes near lows
        assert uptrend_ad.iloc[-1] > uptrend_ad.iloc[0]
        assert downtrend_ad.iloc[-1] < downtrend_ad.iloc[0]
    
    def test_money_flow_index(self, sample_price_data):
        """Test the Money Flow Index calculation."""
        # Calculate MFI with period 14
        result = money_flow_index(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume'],
            period=14
        )
        
        # Verify the result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_price_data)
        
        # MFI should be between 0 and 100
        valid_values = result.dropna()
        assert valid_values.min() >= 0
        assert valid_values.max() <= 100
        
        # First 14 values should be NaN
        for i in range(14):
            assert np.isnan(result.iloc[i])
        
        # Value at index 14 should not be NaN
        assert not np.isnan(result.iloc[14])
        
        # Test MFI behavior with uptrend and downtrend
        # Create data with clear uptrend
        uptrend_high = pd.Series(np.linspace(110, 220, 50))
        uptrend_low = pd.Series(np.linspace(100, 200, 50))
        uptrend_close = pd.Series(np.linspace(105, 210, 50))
        uptrend_volume = pd.Series(np.random.randint(1000, 10000, 50))
        
        uptrend_mfi = money_flow_index(uptrend_high, uptrend_low, uptrend_close, uptrend_volume, period=14)
        
        # Create data with clear downtrend
        downtrend_high = pd.Series(np.linspace(220, 110, 50))
        downtrend_low = pd.Series(np.linspace(200, 100, 50))
        downtrend_close = pd.Series(np.linspace(210, 105, 50))
        downtrend_volume = pd.Series(np.random.randint(1000, 10000, 50))
        
        downtrend_mfi = money_flow_index(downtrend_high, downtrend_low, downtrend_close, downtrend_volume, period=14)
        
        # MFI should be higher in uptrend and lower in downtrend
        assert uptrend_mfi.iloc[-1] > 50
        assert downtrend_mfi.iloc[-1] < 50
