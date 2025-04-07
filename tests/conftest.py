import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.linspace(100, 120, 100) + np.random.normal(0, 1, 100),
        'high': np.linspace(105, 125, 100) + np.random.normal(0, 1, 100),
        'low': np.linspace(95, 115, 100) + np.random.normal(0, 1, 100),
        'close': np.linspace(100, 120, 100) + np.random.normal(0, 1, 100),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data
