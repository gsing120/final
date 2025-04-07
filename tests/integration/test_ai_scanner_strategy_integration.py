import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.ai.gemma_quantitative_analysis import GemmaQuantitativeAnalysis
from backend.market_scanner import MarketScanner, ScanCriteria, ScanTemplate
from backend.strategy_engine import StrategyEngine, Strategy

class TestAIScannerStrategyIntegration:
    """
    Integration tests for the interaction between AI analysis, market scanner, and strategy engine.
    """
    
    @pytest.fixture
    def mock_gemma_model(self):
        """Create a mock Gemma model for testing."""
        mock_model = MagicMock()
        
        # Configure the mock to return sample responses
        mock_model.analyze_market_regime.return_value = {
            "market_regime": "bullish",
            "confidence": 0.85,
            "key_factors": ["strong momentum", "positive breadth", "supportive macro"]
        }
        
        mock_model.generate_trading_strategy.return_value = {
            "strategy_name": "Momentum Breakout",
            "description": "Identifies stocks breaking out of consolidation patterns with strong volume",
            "market_regime": "bullish",
            "indicators": [
                {"name": "SMA", "period": 20},
                {"name": "RSI", "period": 14},
                {"name": "Volume SMA", "period": 50}
            ],
            "entry_conditions": [
                "price > SMA(20)",
                "RSI(14) > 50",
                "volume > Volume SMA(50) * 1.5"
            ],
            "exit_conditions": [
                "price < SMA(20)",
                "RSI(14) < 30"
            ],
            "risk_management": {
                "position_size_percent": 2.0,
                "stop_loss_percent": 5.0,
                "take_profit_percent": 15.0,
                "max_positions": 5
            }
        }
        
        return mock_model
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        # Create a dictionary of DataFrames for different assets
        data = {}
        
        # Generate sample data for multiple stocks
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        for symbol in symbols:
            # Use symbol hash for reproducible randomness
            np.random.seed(hash(symbol) % 2**32)
            
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            
            # Base price varies by symbol
            base_price = 100 + hash(symbol) % 400
            
            # Generate price data with some randomness
            close_prices = base_price + np.random.normal(0, base_price * 0.01, 100).cumsum()
            
            # Add some trend based on the symbol to create variety
            if symbol in ["AAPL", "MSFT", "GOOGL"]:
                # Uptrend
                close_prices += np.linspace(0, base_price * 0.2, 100)
            elif symbol in ["AMZN"]:
                # Downtrend
                close_prices += np.linspace(base_price * 0.1, -base_price * 0.1, 100)
            # META remains random walk
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': close_prices - np.random.uniform(0, base_price * 0.01, 100),
                'high': close_prices + np.random.uniform(0, base_price * 0.01, 100),
                'low': close_prices - np.random.uniform(0, base_price * 0.01, 100),
                'close': close_prices,
                'volume': np.random.randint(base_price * 100, base_price * 1000, 100)
            }, index=dates)
            
            data[symbol] = df
        
        return data
    
    @pytest.fixture
    def gemma_quant_analysis(self, mock_gemma_model):
        """Create a GemmaQuantitativeAnalysis instance for testing."""
        return GemmaQuantitativeAnalysis(model=mock_gemma_model)
    
    @pytest.fixture
    def market_scanner(self):
        """Create a MarketScanner instance for testing."""
        mock_market_data_client = MagicMock()
        
        # Configure the mock to return sample universe
        mock_market_data_client.get_universe.return_value = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        return MarketScanner(market_data_client=mock_market_data_client)
    
    @pytest.fixture
    def strategy_engine(self):
        """Create a StrategyEngine instance for testing."""
        return StrategyEngine()
    
    def test_ai_generated_strategy_execution(self, gemma_quant_analysis, strategy_engine, sample_market_data):
        """Test generating a strategy with AI and executing it."""
        # Generate a strategy using AI
        with patch.object(gemma_quant_analysis.model, 'generate_trading_strategy') as mock_generate:
            # Configure mock to return a specific strategy
            mock_generate.return_value = {
                "strategy_name": "AI Momentum Strategy",
                "description": "AI-generated momentum strategy",
                "indicators": [
                    {"name": "sma", "params": {"period": 20}},
                    {"name": "rsi", "params": {"period": 14}}
                ],
                "entry_conditions": [
                    {"type": "greater_than", "left": "close", "right": "sma"},
                    {"type": "greater_than", "left": "rsi", "right": 50}
                ],
                "exit_conditions": [
                    {"type": "less_than", "left": "close", "right": "sma"},
                    {"type": "less_than", "left": "rsi", "right": 30}
                ],
                "risk_management": {
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.15,
                    "position_size_pct": 0.02
                }
            }
            
            # Generate the strategy
            ai_strategy_config = gemma_quant_analysis.generate_trading_strategy(
                sample_market_data,
                strategy_type="momentum",
                risk_profile="moderate"
            )
            
            # Convert AI strategy to Strategy object
            ai_strategy = Strategy(
                name=ai_strategy_config["strategy_name"],
                description=ai_strategy_config["description"],
                parameters={
                    "stop_loss_pct": ai_strategy_config["risk_management"]["stop_loss_pct"],
                    "take_profit_pct": ai_strategy_config["risk_management"]["take_profit_pct"]
                }
            )
            
            # Add indicators
            for indicator in ai_strategy_config["indicators"]:
                ai_strategy.add_indicator(
                    name=indicator["name"],
                    function=MagicMock(),  # Mock the indicator function
                    params=indicator["params"],
                    input_data="close"
                )
            
            # Add entry conditions
            for condition in ai_strategy_config["entry_conditions"]:
                ai_strategy.add_entry_condition(
                    condition_type=condition["type"],
                    left_operand=condition["left"],
                    right_operand=condition["right"]
                )
            
            # Add exit conditions
            for condition in ai_strategy_config["exit_conditions"]:
                ai_strategy.add_exit_condition(
                    condition_type=condition["type"],
                    left_operand=condition["left"],
                    right_operand=condition["right"]
                )
            
            # Register strategy with engine
            strategy_engine.register_strategy(ai_strategy)
            
            # Mock the backtest method to return sample results
            with patch.object(strategy_engine, 'backtest') as mock_backtest:
                mock_backtest.return_value = {
                    "trades": [
                        {"entry_date": "2023-01-15", "exit_date": "2023-02-01", "return": 0.08},
                        {"entry_date": "2023-02-15", "exit_date": "2023-03-01", "return": 0.05}
                    ],
                    "performance_metrics": {
                        "total_return": 0.13,
                        "sharpe_ratio": 1.5,
                        "max_drawdown": 0.05,
                        "win_rate": 0.75
                    },
                    "equity_curve": pd.Series([10000, 10200, 10400, 10600, 10800, 11000, 11200, 11300], 
                                             index=pd.date_range(start='2023-01-01', periods=8, freq='W'))
                }
                
                # Run backtest on AAPL data
                backtest_result = strategy_engine.backtest(
                    strategy_name=ai_strategy.name,
                    data=sample_market_data["AAPL"],
                    initial_capital=10000.0,
                    start_date=sample_market_data["AAPL"].index[20],  # Start after indicators are calculated
                    end_date=sample_market_data["AAPL"].index[-1]
                )
                
                # Verify backtest results
                assert isinstance(backtest_result, dict)
                assert "trades" in backtest_result
                assert "performance_metrics" in backtest_result
                assert "equity_curve" in backtest_result
                assert backtest_result["performance_metrics"]["total_return"] == 0.13
                assert backtest_result["performance_metrics"]["sharpe_ratio"] == 1.5
                
                # Verify the strategy was generated and executed correctly
                mock_generate.assert_called_once()
                mock_backtest.assert_called_once()
    
    def test_market_scanner_strategy_integration(self, market_scanner, strategy_engine, sample_market_data):
        """Test integrating market scanner with strategy execution."""
        # Create scan criteria
        criteria = ScanCriteria(
            name="Momentum Scan",
            description="Scan for stocks with momentum",
            indicator_conditions=[
                {"indicator": "sma", "params": {"period": 20}, "operator": "<", "value": "close"}
            ],
            price_conditions=[
                {"price_field": "close", "operator": ">", "value": 100.0}
            ],
            fundamental_conditions=[],
            time_conditions=[]
        )
        
        # Mock the execute_scan method to return sample results
        with patch.object(market_scanner, 'execute_scan') as mock_scan:
            mock_scan.return_value = MagicMock(
                criteria_name="Momentum Scan",
                matching_symbols=["AAPL", "MSFT", "GOOGL"],
                execution_time=0.5,
                total_symbols_scanned=5
            )
            
            # Execute scan
            scan_result = market_scanner.execute_scan(criteria)
            
            # Create a strategy
            momentum_strategy = Strategy(
                name="Momentum Strategy",
                description="Strategy for stocks with momentum",
                parameters={
                    "ma_period": 20,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.15
                }
            )
            
            # Add indicators to the strategy
            momentum_strategy.add_indicator(
                name="sma",
                function=MagicMock(),  # Mock the indicator function
                params={"period": 20},
                input_data="close"
            )
            
            # Add entry conditions
            momentum_strategy.add_entry_condition(
                condition_type="greater_than",
                left_operand="close",
                right_operand="sma"
            )
            
            # Add exit conditions
            momentum_strategy.add_exit_condition(
                condition_type="less_than",
                left_operand="close",
                right_operand="sma"
            )
            
            # Register strategy with engine
            strategy_engine.register_strategy(momentum_strategy)
            
            # Mock the backtest method to return sample results
            with patch.object(strategy_engine, 'backtest') as mock_backtest:
                mock_backtest.return_value = {
                    "trades": [
                        {"entry_date": "2023-01-15", "exit_date": "2023-02-01", "return": 0.08}
                    ],
                    "performance_metrics": {
                        "total_return": 0.08,
                        "sharpe_ratio": 1.2,
                        "max_drawdown": 0.03,
                        "win_rate": 1.0
                    },
                    "equity_curve": pd.Series([10000, 10200, 10400, 10600, 10800], 
                                             index=pd.date_range(start='2023-01-01', periods=5, freq='W'))
                }
                
                # Run strategy on each matching symbol from scan
                results = {}
                for symbol in scan_result.matching_symbols:
                    if symbol in sample_market_data:
                        results[symbol] = strategy_engine.backtest(
                            strategy_name="Momentum Strategy",
                            data=sample_market_data[symbol],
                            initial_capital=10000.0,
                            start_date=sample_market_data[symbol].index[20],
                            end_date=sample_market_data[symbol].index[-1]
                        )
                
                # Verify results
                assert len(results) == 3  # Three matching symbols
                for symbol in scan_result.matching_symbols:
                    assert symbol in results
                    assert "performance_metrics" in results[symbol]
                    assert "total_return" in results[symbol]["performance_metrics"]
                
                # Verify the scan and backtest methods were called correctly
                mock_scan.assert_called_once()
                assert mock_backtest.call_count == 3  # Called once for each matching symbol
    
    def test_ai_market_regime_strategy_adaptation(self, gemma_quant_analysis, strategy_engine, sample_market_data):
        """Test adapting strategy based on AI-detected market regime."""
        # Analyze market regime
        with patch.object(gemma_quant_analysis.model, 'analyze_market_data') as mock_analyze:
            # Configure mock to return different market regimes
            mock_analyze.side_effect = [
                {
                    "market_regime": "bullish",
                    "confidence": 0.85,
                    "key_factors": ["strong momentum", "positive breadth", "supportive macro"]
                },
                {
                    "market_regime": "bearish",
                    "confidence": 0.80,
                    "key_factors": ["negative momentum", "deteriorating breadth", "economic concerns"]
                }
            ]
            
            # Analyze current market regime (bullish)
            bullish_regime = gemma_quant_analysis.analyze_market_regime(sample_market_data)
            
            # Create a strategy that adapts to market regime
            adaptive_strategy = Strategy(
                name="Adaptive Strategy",
                description="Strategy that adapts to market regime",
                parameters={
                    "ma_period": 20,
                    "rsi_period": 14,
                    "bullish_rsi_threshold": 40,  # Lower entry threshold in bullish regime
                    "bearish_rsi_threshold": 60,  # Higher entry threshold in bearish regime
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.15
                }
            )
            
            # Add indicators to the strategy
            adaptive_strategy.add_indicator(
                name="sma",
                function=MagicMock(),  # Mock the indicator function
                params={"period": 20},
                input_data="close"
            )
            
            adaptive_strategy.add_indicator(
                name="rsi",
                function=MagicMock(),  # Mock the indicator function
                params={"period": 14},
                input_data="close"
            )
            
            # Add entry conditions based on market regime
            if bullish_regime["market_regime"] == "bullish":
                # In bullish regime, use lower RSI threshold for entry
                adaptive_strategy.add_entry_condition(
                    condition_type="greater_than",
                    left_operand="close",
                    right_operand="sma"
                )
                
                adaptive_strategy.add_entry_condition(
                    condition_type="greater_than",
                    left_operand="rsi",
                    right_operand=40  # Lower threshold in bullish regime
                )
            else:
                # In bearish regime, use higher RSI threshold for entry
                adaptive_strategy.add_entry_condition(
                    condition_type="greater_than",
                    left_operand="close",
                    right_operand="sma"
                )
                
                adaptive_strategy.add_entry_condition(
                    condition_type="greater_than",
                    left_operand="rsi",
                    right_operand=60  # Higher threshold in bearish regime
                )
            
            # Add exit conditions
            adaptive_strategy.add_exit_condition(
                condition_type="less_than",
                left_operand="close",
                right_operand="sma"
            )
            
            # Register strategy with engine
            strategy_engine.register_strategy(adaptive_strategy)
            
            # Mock the backtest method to return sample results
            with patch.object(strategy_engine, 'backtest') as mock_backtest:
                mock_backtest.return_value = {
                    "trades": [
                        {"entry_date": "2023-01-15", "exit_date": "2023-02-01", "return": 0.08}
                    ],
                    "performance_metrics": {
                        "total_return": 0.08,
                        "sharpe_ratio": 1.2,
                        "max_drawdown": 0.03,
                        "win_rate": 1.0
                    },
                    "equity_curve": pd.Series([10000, 10200, 10400, 10600, 10800], 
                                             index=pd.date_range(start='2023-01-01', periods=5, freq='W'))
                }
                
                # Run backtest with bullish strategy
                bullish_result = strategy_engine.backtest(
                    strategy_name="Adaptive Strategy",
                    data=sample_market_data["AAPL"],
                    initial_capital=10000.0,
                    start_date=sample_market_data["AAPL"].index[20],
                    end_date=sample_market_data["AAPL"].index[-1]
                )
                
                # Now analyze market regime again (bearish)
                bearish_regime = gemma_quant_analysis.analyze_market_regime(sample_market_data)
                
                # Create a new strategy for bearish regime
                bearish_adaptive_strategy = Strategy(
                    name="Bearish Adaptive Strategy",
                    description="Strategy adapted for bearish regime",
                    parameters={
                        "ma_period": 20,
                        "rsi_period": 14,
                        "bullish_rsi_threshold": 40,
                        "bearish_rsi_threshold": 60,
                        "stop_loss_pct": 0.05,
                        "take_profit_pct": 0.10  # Lower take profit in bearish regime
                    }
                )
                
                # Add indicators to the strategy
                bearish_adaptive_strategy.add_indicator(
                    name="sma",
                    function=MagicMock(),
                    params={"period": 20},
                    input_data="close"
                )
                
                bearish_adaptive_strategy.add_indicator(
                    name="rsi",
                    function=MagicMock(),
                    params={"period": 14},
                    input_data="close"
                )
                
                # Add entry conditions based on market regime
                if bearish_regime["market_regime"] == "bearish":
                    # In bearish regime, use higher RSI threshold for entry
                    bearish_adaptive_strategy.add_entry_condition(
                        condition_type="greater_than",
                        left_operand="close",
                        right_operand="sma"
                    )
                    
                    bearish_adaptive_strategy.add_entry_condition(
                        condition_type="greater_than",
                        left_operand="rsi",
                        right_operand=60  # Higher threshold in bearish regime
                    )
                
                # Add exit conditions
                bearish_adaptive_strategy.add_exit_condition(
                    condition_type="less_than",
                    left_operand="close",
                    right_operand="sma"
                )
                
                # Register strategy with engine
                strategy_engine.register_strategy(bearish_adaptive_strategy)
                
                # Configure mock to return different results for bearish strategy
                mock_backtest.return_value = {
                    "trades": [
                        {"entry_date": "2023-01-15", "exit_date": "2023-01-25", "return": 0.04}
                    ],
                    "performance_metrics": {
                        "total_return": 0.04,
                        "sharpe_ratio": 0.9,
                        "max_drawdown": 0.02,
                        "win_rate": 1.0
                    },
                    "equity_curve": pd.Series([10000, 10100, 10200, 10300, 10400], 
                                             index=pd.date_range(start='2023-01-01', periods=5, freq='W'))
                }
                
                # Run backtest with bearish strategy
                bearish_result = strategy_engine.backtest(
                    strategy_name="Bearish Adaptive Strategy",
                    data=sample_market_data["AAPL"],
                    initial_capital=10000.0,
                    start_date=sample_market_data["AAPL"].index[20],
                    end_date=sample_market_data["AAPL"].index[-1]
                )
                
                # Verify results
                assert bullish_result["performance_metrics"]["total_return"] == 0.08
                assert bearish_result["performance_metrics"]["total_return"] == 0.04
                
                # Verify the market regime analysis and backtest methods were called correctly
                assert mock_analyze.call_count == 2
                assert mock_backtest.call_count == 2
