import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.distributed.backtesting import DistributedBacktester
from backend.enhanced_backtesting import EnhancedBacktester
from backend.performance_tracking import PerformanceTracker

class TestBacktestingPerformanceIntegration:
    """
    Integration tests for the interaction between backtesting and performance tracking.
    """
    
    @pytest.fixture
    def sample_strategy(self):
        """Create a sample strategy for testing."""
        strategy = {
            "name": "Moving Average Crossover",
            "parameters": {
                "fast_ma_period": 20,
                "slow_ma_period": 50,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.15
            },
            "entry_conditions": [
                "fast_ma > slow_ma",
                "volume > volume_sma(20) * 1.5"
            ],
            "exit_conditions": [
                "fast_ma < slow_ma",
                "price < stop_loss",
                "price > take_profit"
            ]
        }
        return strategy
    
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
    def distributed_backtester(self):
        """Create a DistributedBacktester instance for testing."""
        return DistributedBacktester(num_workers=2)
    
    @pytest.fixture
    def enhanced_backtester(self):
        """Create an EnhancedBacktester instance for testing."""
        return EnhancedBacktester()
    
    @pytest.fixture
    def performance_tracker(self):
        """Create a PerformanceTracker instance for testing."""
        return PerformanceTracker()
    
    def test_distributed_backtest_performance_tracking(self, distributed_backtester, performance_tracker, sample_strategy, sample_price_data):
        """Test tracking performance of distributed backtests."""
        # Mock the run_backtest method
        with patch.object(distributed_backtester, 'run_backtest') as mock_backtest:
            # Configure mock to return sample backtest results
            mock_backtest.return_value = {
                "trades": [
                    {"entry_date": "2023-01-15", "exit_date": "2023-02-01", "entry_price": 105.0, "exit_price": 110.0, "quantity": 100, "return": 0.0476, "pnl": 500.0},
                    {"entry_date": "2023-02-15", "exit_date": "2023-03-01", "entry_price": 112.0, "exit_price": 118.0, "quantity": 100, "return": 0.0536, "pnl": 600.0}
                ],
                "performance_metrics": {
                    "total_return": 0.11,
                    "annualized_return": 0.65,
                    "sharpe_ratio": 1.5,
                    "sortino_ratio": 2.2,
                    "max_drawdown": 0.05,
                    "win_rate": 0.75,
                    "profit_factor": 2.5,
                    "avg_win": 0.06,
                    "avg_loss": -0.02
                },
                "equity_curve": pd.Series([10000, 10200, 10400, 10600, 10800, 11000, 11100], 
                                         index=pd.date_range(start='2023-01-01', periods=7, freq='W'))
            }
            
            # Run backtest
            backtest_result = distributed_backtester.run_backtest(
                strategy=sample_strategy,
                data=sample_price_data,
                parameters={"fast_ma_period": 15, "slow_ma_period": 45},
                start_date="2023-01-15",
                end_date="2023-03-15",
                initial_capital=10000.0
            )
            
            # Track performance
            strategy_id = "MA_Crossover_15_45"
            performance_tracker.add_backtest_result(strategy_id, backtest_result)
            
            # Verify performance tracking
            tracked_strategies = performance_tracker.get_tracked_strategies()
            assert strategy_id in tracked_strategies
            
            # Get performance metrics
            metrics = performance_tracker.get_performance_metrics(strategy_id)
            assert metrics["total_return"] == 0.11
            assert metrics["sharpe_ratio"] == 1.5
            assert metrics["max_drawdown"] == 0.05
            
            # Get equity curve
            equity_curve = performance_tracker.get_equity_curve(strategy_id)
            assert len(equity_curve) == 7
            assert equity_curve.iloc[-1] == 11100
            
            # Get trades
            trades = performance_tracker.get_trades(strategy_id)
            assert len(trades) == 2
            assert trades[0]["pnl"] == 500.0
            assert trades[1]["pnl"] == 600.0
    
    def test_parameter_optimization_performance_comparison(self, distributed_backtester, performance_tracker, sample_strategy, sample_price_data):
        """Test comparing performance of different parameter sets."""
        # Define parameter grid
        param_grid = {
            "fast_ma_period": [10, 15, 20],
            "slow_ma_period": [30, 45, 60]
        }
        
        # Mock the optimize_parameters method
        with patch.object(distributed_backtester, 'optimize_parameters') as mock_optimize:
            # Configure mock to return sample optimization results
            mock_optimize.return_value = {
                "best_parameters": {"fast_ma_period": 15, "slow_ma_period": 45},
                "performance": {"total_return": 0.15, "sharpe_ratio": 1.8},
                "all_results": [
                    {"parameters": {"fast_ma_period": 10, "slow_ma_period": 30}, 
                     "performance": {"total_return": 0.10, "sharpe_ratio": 1.2}},
                    {"parameters": {"fast_ma_period": 10, "slow_ma_period": 45}, 
                     "performance": {"total_return": 0.12, "sharpe_ratio": 1.4}},
                    {"parameters": {"fast_ma_period": 10, "slow_ma_period": 60}, 
                     "performance": {"total_return": 0.08, "sharpe_ratio": 1.0}},
                    {"parameters": {"fast_ma_period": 15, "slow_ma_period": 30}, 
                     "performance": {"total_return": 0.13, "sharpe_ratio": 1.5}},
                    {"parameters": {"fast_ma_period": 15, "slow_ma_period": 45}, 
                     "performance": {"total_return": 0.15, "sharpe_ratio": 1.8}},
                    {"parameters": {"fast_ma_period": 15, "slow_ma_period": 60}, 
                     "performance": {"total_return": 0.11, "sharpe_ratio": 1.3}},
                    {"parameters": {"fast_ma_period": 20, "slow_ma_period": 30}, 
                     "performance": {"total_return": 0.09, "sharpe_ratio": 1.1}},
                    {"parameters": {"fast_ma_period": 20, "slow_ma_period": 45}, 
                     "performance": {"total_return": 0.14, "sharpe_ratio": 1.6}},
                    {"parameters": {"fast_ma_period": 20, "slow_ma_period": 60}, 
                     "performance": {"total_return": 0.12, "sharpe_ratio": 1.4}}
                ]
            }
            
            # Run parameter optimization
            optimization_result = distributed_backtester.optimize_parameters(
                strategy=sample_strategy,
                data=sample_price_data,
                param_grid=param_grid,
                start_date="2023-01-15",
                end_date="2023-03-15",
                initial_capital=10000.0,
                metric="total_return"
            )
            
            # Track performance for each parameter set
            for result in optimization_result["all_results"]:
                params = result["parameters"]
                strategy_id = f"MA_Crossover_{params['fast_ma_period']}_{params['slow_ma_period']}"
                
                # Create a mock backtest result with the optimization performance
                mock_backtest_result = {
                    "trades": [],  # No trades for simplicity
                    "performance_metrics": result["performance"],
                    "equity_curve": pd.Series([10000 * (1 + result["performance"]["total_return"])], 
                                             index=[pd.Timestamp("2023-03-15")])
                }
                
                performance_tracker.add_backtest_result(strategy_id, mock_backtest_result)
            
            # Compare performance of different parameter sets
            comparison = performance_tracker.compare_strategies(metric="total_return")
            
            # Verify comparison results
            assert len(comparison) == 9  # 9 parameter combinations
            assert comparison[0][0] == "MA_Crossover_15_45"  # Best strategy
            assert comparison[0][1] == 0.15  # Best return
            
            # Verify ranking
            assert comparison[1][1] <= comparison[0][1]  # Second best <= best
            assert comparison[-1][1] <= comparison[-2][1]  # Worst <= second worst
            
            # Compare using different metric
            sharpe_comparison = performance_tracker.compare_strategies(metric="sharpe_ratio")
            assert sharpe_comparison[0][0] == "MA_Crossover_15_45"  # Best strategy
            assert sharpe_comparison[0][1] == 1.8  # Best Sharpe
    
    def test_enhanced_backtest_with_performance_analysis(self, enhanced_backtester, performance_tracker, sample_strategy, sample_price_data):
        """Test enhanced backtesting with detailed performance analysis."""
        # Mock the run_backtest method
        with patch.object(enhanced_backtester, 'run_backtest') as mock_backtest:
            # Configure mock to return sample backtest results with enhanced metrics
            mock_backtest.return_value = {
                "trades": [
                    {"entry_date": "2023-01-15", "exit_date": "2023-02-01", "entry_price": 105.0, "exit_price": 110.0, 
                     "quantity": 100, "return": 0.0476, "pnl": 500.0, "holding_period": 17, "exit_reason": "target_hit"},
                    {"entry_date": "2023-02-15", "exit_date": "2023-03-01", "entry_price": 112.0, "exit_price": 118.0, 
                     "quantity": 100, "return": 0.0536, "pnl": 600.0, "holding_period": 14, "exit_reason": "target_hit"},
                    {"entry_date": "2023-03-15", "exit_date": "2023-03-20", "entry_price": 115.0, "exit_price": 113.0, 
                     "quantity": 100, "return": -0.0174, "pnl": -200.0, "holding_period": 5, "exit_reason": "stop_loss"}
                ],
                "performance_metrics": {
                    "total_return": 0.09,
                    "annualized_return": 0.55,
                    "sharpe_ratio": 1.3,
                    "sortino_ratio": 1.9,
                    "max_drawdown": 0.06,
                    "win_rate": 0.67,
                    "profit_factor": 5.5,
                    "avg_win": 0.0506,
                    "avg_loss": -0.0174,
                    "avg_holding_period": 12,
                    "recovery_factor": 1.5,
                    "ulcer_index": 0.03,
                    "calmar_ratio": 9.17,
                    "omega_ratio": 1.8
                },
                "equity_curve": pd.Series([10000, 10200, 10400, 10600, 10800, 11000, 10900], 
                                         index=pd.date_range(start='2023-01-01', periods=7, freq='W')),
                "drawdowns": [
                    {"start_date": "2023-03-15", "end_date": "2023-03-20", "recovery_date": "2023-03-25", 
                     "depth": 0.06, "length": 5, "recovery_time": 5}
                ],
                "monthly_returns": {
                    "2023-01": 0.04,
                    "2023-02": 0.06,
                    "2023-03": -0.01
                },
                "trade_statistics": {
                    "by_exit_reason": {
                        "target_hit": {"count": 2, "win_rate": 1.0, "avg_return": 0.0506},
                        "stop_loss": {"count": 1, "win_rate": 0.0, "avg_return": -0.0174}
                    },
                    "by_holding_period": {
                        "0-7": {"count": 1, "win_rate": 0.0, "avg_return": -0.0174},
                        "8-14": {"count": 1, "win_rate": 1.0, "avg_return": 0.0536},
                        "15+": {"count": 1, "win_rate": 1.0, "avg_return": 0.0476}
                    }
                }
            }
            
            # Run enhanced backtest
            backtest_result = enhanced_backtester.run_backtest(
                strategy=sample_strategy,
                data=sample_price_data,
                parameters={"fast_ma_period": 15, "slow_ma_period": 45},
                start_date="2023-01-15",
                end_date="2023-03-15",
                initial_capital=10000.0,
                include_trade_stats=True,
                include_drawdowns=True,
                include_monthly_returns=True
            )
            
            # Track performance
            strategy_id = "Enhanced_MA_Crossover"
            performance_tracker.add_backtest_result(strategy_id, backtest_result)
            
            # Verify enhanced metrics are tracked
            metrics = performance_tracker.get_performance_metrics(strategy_id)
            assert "recovery_factor" in metrics
            assert "ulcer_index" in metrics
            assert "calmar_ratio" in metrics
            assert "omega_ratio" in metrics
            
            # Get drawdowns
            drawdowns = performance_tracker.get_drawdowns(strategy_id)
            assert len(drawdowns) == 1
            assert drawdowns[0]["depth"] == 0.06
            
            # Get monthly returns
            monthly_returns = performance_tracker.get_monthly_returns(strategy_id)
            assert len(monthly_returns) == 3
            assert monthly_returns["2023-01"] == 0.04
            assert monthly_returns["2023-03"] == -0.01
            
            # Get trade statistics
            trade_stats = performance_tracker.get_trade_statistics(strategy_id)
            assert "by_exit_reason" in trade_stats
            assert "by_holding_period" in trade_stats
            assert trade_stats["by_exit_reason"]["target_hit"]["win_rate"] == 1.0
            assert trade_stats["by_holding_period"]["0-7"]["win_rate"] == 0.0
    
    def test_walk_forward_optimization_performance_tracking(self, distributed_backtester, performance_tracker, sample_strategy, sample_price_data):
        """Test tracking performance of walk-forward optimization."""
        # Mock the walk_forward_optimization method
        with patch.object(distributed_backtester, 'walk_forward_optimization') as mock_wfo:
            # Configure mock to return sample WFO results
            mock_wfo.return_value = {
                "periods": [
                    {
                        "train_start": "2023-01-01",
                        "train_end": "2023-01-31",
                        "test_start": "2023-02-01",
                        "test_end": "2023-02-15",
                        "best_parameters": {"fast_ma_period": 10, "slow_ma_period": 30},
                        "train_performance": {"total_return": 0.08, "sharpe_ratio": 1.2},
                        "test_performance": {"total_return": 0.05, "sharpe_ratio": 1.0}
                    },
                    {
                        "train_start": "2023-01-16",
                        "train_end": "2023-02-15",
                        "test_start": "2023-02-16",
                        "test_end": "2023-03-01",
                        "best_parameters": {"fast_ma_period": 15, "slow_ma_period": 45},
                        "train_performance": {"total_return": 0.10, "sharpe_ratio": 1.5},
                        "test_performance": {"total_return": 0.06, "sharpe_ratio": 1.2}
                    },
                    {
                        "train_start": "2023-02-01",
                        "train_end": "2023-03-01",
                        "test_start": "2023-03-02",
                        "test_end": "2023-03-15",
                        "best_parameters": {"fast_ma_period": 20, "slow_ma_period": 60},
                        "train_performance": {"total_return": 0.09, "sharpe_ratio": 1.4},
                        "test_performance": {"total_return": 0.04, "sharpe_ratio": 0.9}
                    }
                ],
                "overall_performance": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.1,
                    "max_drawdown": 0.07,
                    "win_rate": 0.65
                }
            }
            
            # Define parameter grid
            param_grid = {
                "fast_ma_period": [10, 15, 20],
                "slow_ma_period": [30, 45, 60]
            }
            
            # Run walk-forward optimization
            wfo_result = distributed_backtester.walk_forward_optimization(
                strategy=sample_strategy,
                data=sample_price_data,
                param_grid=param_grid,
                start_date="2023-01-01",
                end_date="2023-03-15",
                initial_capital=10000.0,
                train_period=30,
                test_period=15,
                metric="total_return"
            )
            
            # Track WFO performance
            strategy_id = "WFO_MA_Crossover"
            performance_tracker.add_wfo_result(strategy_id, wfo_result)
            
            # Verify WFO tracking
            wfo_strategies = performance_tracker.get_wfo_strategies()
            assert strategy_id in wfo_strategies
            
            # Get WFO periods
            periods = performance_tracker.get_wfo_periods(strategy_id)
            assert len(periods) == 3
            
            # Get overall WFO performance
            overall_performance = performance_tracker.get_wfo_overall_performance(strategy_id)
            assert overall_performance["total_return"] == 0.15
            assert overall_performance["sharpe_ratio"] == 1.1
            
            # Calculate in-sample vs out-of-sample performance ratio
            is_vs_oos = performance_tracker.calculate_is_vs_oos_ratio(strategy_id, metric="total_return")
            
            # Expected calculation:
            # Period 1: 0.08 (train) vs 0.05 (test) = 1.6
            # Period 2: 0.10 (train) vs 0.06 (test) = 1.67
            # Period 3: 0.09 (train) vs 0.04 (test) = 2.25
            # Average: (1.6 + 1.67 + 2.25) / 3 = 1.84
            expected_ratio = (0.08/0.05 + 0.10/0.06 + 0.09/0.04) / 3
            assert is_vs_oos == pytest.approx(expected_ratio, 0.01)
            
            # Check for overfitting
            is_overfitting = performance_tracker.check_wfo_overfitting(strategy_id, threshold=1.5)
            assert is_overfitting == True  # Ratio > 1.5 indicates overfitting
