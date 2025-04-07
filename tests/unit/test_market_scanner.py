import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.market_scanner import (
    MarketScanner,
    ScanCriteria,
    ScanTemplate,
    ScanResult
)

class TestMarketScanner:
    """
    Unit tests for the Market Scanner module.
    """
    
    @pytest.fixture
    def mock_market_data_client(self):
        """Create a mock market data client for testing."""
        mock_client = MagicMock()
        
        # Configure the mock to return sample data
        # Sample universe of stocks
        universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "PG"]
        
        # Sample price data for each stock
        price_data = {}
        for symbol in universe:
            # Generate random price data
            np.random.seed(hash(symbol) % 2**32)  # Different seed for each symbol
            
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            close_prices = 100 + np.random.normal(0, 1, 100).cumsum()
            
            # Add some trend based on the symbol to create variety
            if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
                # Uptrend
                close_prices += np.linspace(0, 20, 100)
            elif symbol in ["META", "TSLA"]:
                # Downtrend
                close_prices += np.linspace(20, 0, 100)
            # Others remain random walk
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': close_prices - np.random.uniform(0, 2, 100),
                'high': close_prices + np.random.uniform(0, 2, 100),
                'low': close_prices - np.random.uniform(0, 2, 100),
                'close': close_prices,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
            
            price_data[symbol] = df
        
        # Configure mock methods
        mock_client.get_universe.return_value = universe
        mock_client.get_historical_data_bulk.return_value = price_data
        
        return mock_client
    
    def test_scan_criteria_creation(self):
        """Test ScanCriteria class creation and validation."""
        # Create a basic scan criteria
        criteria = ScanCriteria(
            name="Moving Average Crossover",
            description="Stocks where the 20-day SMA crosses above the 50-day SMA",
            indicator_conditions=[
                {"indicator": "sma", "params": {"period": 20}, "operator": ">", "value": "sma", "value_params": {"period": 50}},
                {"indicator": "volume", "operator": ">", "value": "volume_sma", "value_params": {"period": 20}}
            ],
            price_conditions=[
                {"price_field": "close", "operator": ">", "value": 50.0}
            ],
            fundamental_conditions=[
                {"field": "market_cap", "operator": ">", "value": 1e9}  # > $1B market cap
            ],
            time_conditions=[
                {"lookback_days": 1}  # Condition must be true within the last day
            ]
        )
        
        # Verify basic properties
        assert criteria.name == "Moving Average Crossover"
        assert criteria.description == "Stocks where the 20-day SMA crosses above the 50-day SMA"
        assert len(criteria.indicator_conditions) == 2
        assert len(criteria.price_conditions) == 1
        assert len(criteria.fundamental_conditions) == 1
        assert len(criteria.time_conditions) == 1
        
        # Test validation
        assert criteria.validate() == True
        
        # Test invalid criteria
        invalid_criteria = ScanCriteria(
            name="Invalid Criteria",
            description="This criteria has invalid conditions",
            indicator_conditions=[
                {"indicator": "unknown_indicator", "params": {"period": 20}, "operator": ">", "value": 0}
            ],
            price_conditions=[],
            fundamental_conditions=[],
            time_conditions=[]
        )
        
        assert invalid_criteria.validate() == False
    
    def test_scan_template_creation(self):
        """Test ScanTemplate class creation and management."""
        # Create scan criteria
        criteria1 = ScanCriteria(
            name="Moving Average Crossover",
            description="Stocks where the 20-day SMA crosses above the 50-day SMA",
            indicator_conditions=[
                {"indicator": "sma", "params": {"period": 20}, "operator": ">", "value": "sma", "value_params": {"period": 50}}
            ],
            price_conditions=[],
            fundamental_conditions=[],
            time_conditions=[]
        )
        
        criteria2 = ScanCriteria(
            name="Volume Spike",
            description="Stocks with volume spike above 200% of average",
            indicator_conditions=[
                {"indicator": "volume", "operator": ">", "value": "volume_sma", "value_params": {"period": 20}, "multiplier": 2.0}
            ],
            price_conditions=[],
            fundamental_conditions=[],
            time_conditions=[]
        )
        
        # Create a scan template
        template = ScanTemplate(
            name="Momentum Breakout",
            description="Identifies stocks breaking out with strong volume",
            criteria=[criteria1, criteria2],
            universe_filter={"market_cap_min": 1e9, "sector": ["Technology", "Healthcare"]},
            sort_by={"field": "volume", "direction": "desc"},
            max_results=20
        )
        
        # Verify basic properties
        assert template.name == "Momentum Breakout"
        assert template.description == "Identifies stocks breaking out with strong volume"
        assert len(template.criteria) == 2
        assert template.universe_filter["market_cap_min"] == 1e9
        assert template.sort_by["field"] == "volume"
        assert template.max_results == 20
        
        # Test adding criteria
        criteria3 = ScanCriteria(
            name="RSI Oversold",
            description="Stocks with RSI below 30",
            indicator_conditions=[
                {"indicator": "rsi", "params": {"period": 14}, "operator": "<", "value": 30}
            ],
            price_conditions=[],
            fundamental_conditions=[],
            time_conditions=[]
        )
        
        template.add_criteria(criteria3)
        assert len(template.criteria) == 3
        
        # Test removing criteria
        template.remove_criteria("Volume Spike")
        assert len(template.criteria) == 2
        assert template.criteria[0].name == "Moving Average Crossover"
        assert template.criteria[1].name == "RSI Oversold"
    
    def test_market_scanner_initialization(self, mock_market_data_client):
        """Test MarketScanner class initialization."""
        # Initialize the scanner
        scanner = MarketScanner(market_data_client=mock_market_data_client)
        
        # Verify initialization
        assert scanner.market_data_client == mock_market_data_client
        assert len(scanner.templates) == 0
        
        # Test adding templates
        criteria = ScanCriteria(
            name="Moving Average Crossover",
            description="Stocks where the 20-day SMA crosses above the 50-day SMA",
            indicator_conditions=[
                {"indicator": "sma", "params": {"period": 20}, "operator": ">", "value": "sma", "value_params": {"period": 50}}
            ],
            price_conditions=[],
            fundamental_conditions=[],
            time_conditions=[]
        )
        
        template = ScanTemplate(
            name="Momentum Breakout",
            description="Identifies stocks breaking out with strong volume",
            criteria=[criteria],
            universe_filter={},
            sort_by={"field": "volume", "direction": "desc"},
            max_results=20
        )
        
        scanner.add_template(template)
        assert len(scanner.templates) == 1
        assert scanner.templates[0].name == "Momentum Breakout"
    
    def test_market_scan_execution(self, mock_market_data_client):
        """Test market scan execution functionality."""
        # Initialize the scanner
        scanner = MarketScanner(market_data_client=mock_market_data_client)
        
        # Create scan criteria
        criteria = ScanCriteria(
            name="Moving Average Crossover",
            description="Stocks where the 20-day SMA crosses above the 50-day SMA",
            indicator_conditions=[
                {"indicator": "sma", "params": {"period": 20}, "operator": ">", "value": "sma", "value_params": {"period": 50}}
            ],
            price_conditions=[
                {"price_field": "close", "operator": ">", "value": 0}  # All stocks should pass this
            ],
            fundamental_conditions=[],
            time_conditions=[]
        )
        
        # Execute scan
        with patch('backend.indicators.trend_indicators.simple_moving_average') as mock_sma:
            # Configure mock SMA to return values that will trigger the condition for some stocks
            def mock_sma_implementation(prices, period):
                if period == 20:
                    return pd.Series(prices.values + 5)  # 20-day SMA is higher than price
                else:
                    return pd.Series(prices.values)  # 50-day SMA equals price
            
            mock_sma.side_effect = mock_sma_implementation
            
            scan_result = scanner.execute_scan(criteria)
            
            # Verify the result
            assert isinstance(scan_result, ScanResult)
            assert scan_result.criteria_name == "Moving Average Crossover"
            assert len(scan_result.matching_symbols) > 0  # Some symbols should match
            assert scan_result.execution_time > 0
            assert scan_result.total_symbols_scanned == 10  # Our mock universe has 10 symbols
    
    def test_template_scan_execution(self, mock_market_data_client):
        """Test template-based scan execution."""
        # Initialize the scanner
        scanner = MarketScanner(market_data_client=mock_market_data_client)
        
        # Create scan criteria
        criteria1 = ScanCriteria(
            name="Price Above SMA",
            description="Stocks trading above their 50-day SMA",
            indicator_conditions=[
                {"indicator": "close", "operator": ">", "value": "sma", "value_params": {"period": 50}}
            ],
            price_conditions=[],
            fundamental_conditions=[],
            time_conditions=[]
        )
        
        criteria2 = ScanCriteria(
            name="Volume Spike",
            description="Stocks with volume spike above 150% of average",
            indicator_conditions=[
                {"indicator": "volume", "operator": ">", "value": "volume_sma", "value_params": {"period": 20}, "multiplier": 1.5}
            ],
            price_conditions=[],
            fundamental_conditions=[],
            time_conditions=[]
        )
        
        # Create template
        template = ScanTemplate(
            name="Momentum Breakout",
            description="Identifies stocks breaking out with strong volume",
            criteria=[criteria1, criteria2],
            universe_filter={},
            sort_by={"field": "volume", "direction": "desc"},
            max_results=5
        )
        
        scanner.add_template(template)
        
        # Execute template scan
        with patch('backend.indicators.trend_indicators.simple_moving_average') as mock_sma:
            # Configure mocks to return values that will trigger the conditions for some stocks
            def mock_sma_implementation(prices, period):
                if isinstance(prices, pd.Series):
                    return pd.Series(prices.values * 0.9)  # SMA is 90% of price, so price > SMA
                else:
                    return pd.Series([0] * len(prices))
            
            mock_sma.side_effect = mock_sma_implementation
            
            with patch('backend.indicators.volume_indicators.volume_sma') as mock_volume_sma:
                def mock_volume_sma_implementation(volume, period):
                    return pd.Series(volume.values * 0.6)  # Volume SMA is 60% of volume, so volume > 150% of SMA
                
                mock_volume_sma.side_effect = mock_volume_sma_implementation
                
                scan_result = scanner.execute_template_scan("Momentum Breakout")
                
                # Verify the result
                assert isinstance(scan_result, list)
                assert len(scan_result) == 2  # Two criteria in the template
                assert scan_result[0].criteria_name == "Price Above SMA"
                assert scan_result[1].criteria_name == "Volume Spike"
                assert len(scan_result[0].matching_symbols) > 0
                assert len(scan_result[1].matching_symbols) > 0
                
                # Test combined results
                combined_result = scanner.get_combined_template_results("Momentum Breakout", scan_result)
                assert isinstance(combined_result, dict)
                assert "matching_symbols" in combined_result
                assert "all_criteria_matched" in combined_result
                assert "any_criteria_matched" in combined_result
                
                # Verify the combined results are correct
                all_matched = set(scan_result[0].matching_symbols).intersection(set(scan_result[1].matching_symbols))
                any_matched = set(scan_result[0].matching_symbols).union(set(scan_result[1].matching_symbols))
                
                assert set(combined_result["all_criteria_matched"]) == all_matched
                assert set(combined_result["any_criteria_matched"]) == any_matched
