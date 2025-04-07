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
    moving_average_convergence_divergence
)
from backend.indicators.momentum_indicators import (
    relative_strength_index
)
from backend.indicators.volume_indicators import (
    on_balance_volume
)
from backend.strategy_engine import (
    StrategyEngine,
    Strategy
)

class TestIndicatorStrategyIntegration:
    """
    Integration tests for the interaction between indicators and strategy engine.
    """
    
    @pytest.fixture
    def sample_price_data(self):
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
    
    @pytest.fixture
    def strategy_engine(self):
        """Create a strategy engine for testing."""
        return StrategyEngine()
    
    def test_moving_average_crossover_strategy(self, sample_price_data, strategy_engine):
        """Test a moving average crossover strategy using indicator library."""
        # Create strategy
        ma_crossover_strategy = Strategy(
            name="MA Crossover",
            description="Simple moving average crossover strategy",
            parameters={
                "fast_ma_period": 20,
                "slow_ma_period": 50,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.15
            }
        )
        
        # Add indicators to the strategy
        ma_crossover_strategy.add_indicator(
            name="fast_ma",
            function=simple_moving_average,
            params={"period": 20},
            input_data="close"
        )
        
        ma_crossover_strategy.add_indicator(
            name="slow_ma",
            function=simple_moving_average,
            params={"period": 50},
            input_data="close"
        )
        
        # Add entry conditions
        ma_crossover_strategy.add_entry_condition(
            condition_type="cross_above",
            left_operand="fast_ma",
            right_operand="slow_ma"
        )
        
        # Add exit conditions
        ma_crossover_strategy.add_exit_condition(
            condition_type="cross_below",
            left_operand="fast_ma",
            right_operand="slow_ma"
        )
        
        # Register strategy with engine
        strategy_engine.register_strategy(ma_crossover_strategy)
        
        # Run backtest
        backtest_result = strategy_engine.backtest(
            strategy_name="MA Crossover",
            data=sample_price_data,
            initial_capital=10000.0,
            start_date=sample_price_data.index[50],  # Start after slow MA is calculated
            end_date=sample_price_data.index[-1]
        )
        
        # Verify backtest results
        assert isinstance(backtest_result, dict)
        assert "trades" in backtest_result
        assert "performance_metrics" in backtest_result
        assert "equity_curve" in backtest_result
        
        # Verify indicators were calculated correctly
        indicators = strategy_engine.get_calculated_indicators("MA Crossover", sample_price_data)
        assert "fast_ma" in indicators
        assert "slow_ma" in indicators
        
        # Verify indicator values
        fast_ma = simple_moving_average(sample_price_data['close'], period=20)
        slow_ma = simple_moving_average(sample_price_data['close'], period=50)
        
        pd.testing.assert_series_equal(indicators["fast_ma"], fast_ma)
        pd.testing.assert_series_equal(indicators["slow_ma"], slow_ma)
    
    def test_multi_indicator_strategy(self, sample_price_data, strategy_engine):
        """Test a strategy using multiple indicators from different categories."""
        # Create strategy
        multi_indicator_strategy = Strategy(
            name="Multi-Indicator Strategy",
            description="Strategy using trend, momentum, and volume indicators",
            parameters={
                "ma_period": 20,
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "rsi_oversold": 30,
                "rsi_overbought": 70
            }
        )
        
        # Add indicators to the strategy
        multi_indicator_strategy.add_indicator(
            name="sma",
            function=simple_moving_average,
            params={"period": 20},
            input_data="close"
        )
        
        multi_indicator_strategy.add_indicator(
            name="rsi",
            function=relative_strength_index,
            params={"period": 14},
            input_data="close"
        )
        
        multi_indicator_strategy.add_indicator(
            name="macd_line",
            function=lambda prices: moving_average_convergence_divergence(prices, 12, 26, 9)[0],
            params={},
            input_data="close"
        )
        
        multi_indicator_strategy.add_indicator(
            name="macd_signal",
            function=lambda prices: moving_average_convergence_divergence(prices, 12, 26, 9)[1],
            params={},
            input_data="close"
        )
        
        multi_indicator_strategy.add_indicator(
            name="obv",
            function=on_balance_volume,
            params={},
            input_data=["close", "volume"]
        )
        
        # Add entry conditions
        multi_indicator_strategy.add_entry_condition(
            condition_type="cross_above",
            left_operand="macd_line",
            right_operand="macd_signal"
        )
        
        multi_indicator_strategy.add_entry_condition(
            condition_type="less_than",
            left_operand="rsi",
            right_operand=40  # Enter when RSI is below 40 (oversold)
        )
        
        # Add exit conditions
        multi_indicator_strategy.add_exit_condition(
            condition_type="cross_below",
            left_operand="macd_line",
            right_operand="macd_signal"
        )
        
        multi_indicator_strategy.add_exit_condition(
            condition_type="greater_than",
            left_operand="rsi",
            right_operand=70  # Exit when RSI is above 70 (overbought)
        )
        
        # Register strategy with engine
        strategy_engine.register_strategy(multi_indicator_strategy)
        
        # Run backtest
        backtest_result = strategy_engine.backtest(
            strategy_name="Multi-Indicator Strategy",
            data=sample_price_data,
            initial_capital=10000.0,
            start_date=sample_price_data.index[50],  # Start after indicators are calculated
            end_date=sample_price_data.index[-1]
        )
        
        # Verify backtest results
        assert isinstance(backtest_result, dict)
        assert "trades" in backtest_result
        assert "performance_metrics" in backtest_result
        assert "equity_curve" in backtest_result
        
        # Verify indicators were calculated correctly
        indicators = strategy_engine.get_calculated_indicators("Multi-Indicator Strategy", sample_price_data)
        assert "sma" in indicators
        assert "rsi" in indicators
        assert "macd_line" in indicators
        assert "macd_signal" in indicators
        assert "obv" in indicators
        
        # Verify indicator values
        sma = simple_moving_average(sample_price_data['close'], period=20)
        rsi = relative_strength_index(sample_price_data['close'], period=14)
        macd_line, macd_signal, _ = moving_average_convergence_divergence(
            sample_price_data['close'], 12, 26, 9
        )
        obv = on_balance_volume(sample_price_data['close'], sample_price_data['volume'])
        
        pd.testing.assert_series_equal(indicators["sma"], sma)
        pd.testing.assert_series_equal(indicators["rsi"], rsi)
        pd.testing.assert_series_equal(indicators["macd_line"], macd_line)
        pd.testing.assert_series_equal(indicators["macd_signal"], macd_signal)
        pd.testing.assert_series_equal(indicators["obv"], obv)
    
    def test_strategy_parameter_optimization(self, sample_price_data, strategy_engine):
        """Test strategy parameter optimization using indicator library."""
        # Create strategy
        ma_strategy = Strategy(
            name="MA Strategy",
            description="Moving average strategy with optimizable parameters",
            parameters={
                "ma_period": 20,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.15
            }
        )
        
        # Add indicators to the strategy with parameter reference
        ma_strategy.add_indicator(
            name="ma",
            function=simple_moving_average,
            params={"period": "ma_period"},  # Reference to strategy parameter
            input_data="close"
        )
        
        # Add entry conditions
        ma_strategy.add_entry_condition(
            condition_type="greater_than",
            left_operand="close",
            right_operand="ma"
        )
        
        # Add exit conditions
        ma_strategy.add_exit_condition(
            condition_type="less_than",
            left_operand="close",
            right_operand="ma"
        )
        
        # Register strategy with engine
        strategy_engine.register_strategy(ma_strategy)
        
        # Define parameter grid
        param_grid = {
            "ma_period": [10, 20, 50],
            "stop_loss_pct": [0.03, 0.05, 0.07]
        }
        
        # Run parameter optimization
        optimization_result = strategy_engine.optimize_parameters(
            strategy_name="MA Strategy",
            data=sample_price_data,
            param_grid=param_grid,
            initial_capital=10000.0,
            start_date=sample_price_data.index[50],
            end_date=sample_price_data.index[-1],
            metric="total_return"
        )
        
        # Verify optimization results
        assert isinstance(optimization_result, dict)
        assert "best_parameters" in optimization_result
        assert "performance" in optimization_result
        assert "all_results" in optimization_result
        
        # Verify best parameters
        assert "ma_period" in optimization_result["best_parameters"]
        assert "stop_loss_pct" in optimization_result["best_parameters"]
        
        # Verify all results
        assert len(optimization_result["all_results"]) == 9  # 3 x 3 parameter combinations
        
        # Verify each result has the expected structure
        for result in optimization_result["all_results"]:
            assert "parameters" in result
            assert "performance" in result
            assert "ma_period" in result["parameters"]
            assert "stop_loss_pct" in result["parameters"]
            assert "total_return" in result["performance"]
