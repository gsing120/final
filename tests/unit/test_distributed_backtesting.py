import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, call

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.distributed.backtesting import (
    DistributedBacktester,
    BacktestTask,
    BacktestWorker,
    TaskManager,
    ParameterOptimizer
)

class TestDistributedBacktesting:
    """
    Unit tests for the Distributed Backtesting module.
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
    
    def test_backtest_task_creation(self, sample_strategy, sample_price_data):
        """Test BacktestTask class creation and properties."""
        # Create a basic backtest task
        task = BacktestTask(
            task_id="task_001",
            strategy=sample_strategy,
            data=sample_price_data,
            parameters={"fast_ma_period": 15, "slow_ma_period": 45},
            start_date="2023-01-15",
            end_date="2023-03-15",
            initial_capital=10000.0
        )
        
        # Verify basic properties
        assert task.task_id == "task_001"
        assert task.strategy == sample_strategy
        assert task.parameters["fast_ma_period"] == 15
        assert task.parameters["slow_ma_period"] == 45
        assert task.start_date == "2023-01-15"
        assert task.end_date == "2023-03-15"
        assert task.initial_capital == 10000.0
        assert task.status == "pending"
        assert task.result is None
        
        # Test task serialization
        serialized = task.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["task_id"] == "task_001"
        assert serialized["parameters"]["fast_ma_period"] == 15
        assert serialized["status"] == "pending"
        
        # Test task from dict
        new_task = BacktestTask.from_dict(serialized)
        assert new_task.task_id == task.task_id
        assert new_task.parameters["fast_ma_period"] == task.parameters["fast_ma_period"]
        assert new_task.status == task.status
    
    def test_backtest_worker(self, sample_strategy, sample_price_data):
        """Test BacktestWorker class functionality."""
        # Create a worker
        worker = BacktestWorker(worker_id="worker_001")
        
        # Create a task
        task = BacktestTask(
            task_id="task_001",
            strategy=sample_strategy,
            data=sample_price_data,
            parameters={"fast_ma_period": 15, "slow_ma_period": 45},
            start_date="2023-01-15",
            end_date="2023-03-15",
            initial_capital=10000.0
        )
        
        # Mock the execute_backtest method
        with patch.object(worker, 'execute_backtest') as mock_execute:
            # Configure mock to return a sample result
            mock_result = {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.65,
                "trades": [
                    {"entry_date": "2023-01-20", "exit_date": "2023-02-05", "return": 0.07},
                    {"entry_date": "2023-02-15", "exit_date": "2023-03-01", "return": 0.08}
                ]
            }
            mock_execute.return_value = mock_result
            
            # Process the task
            processed_task = worker.process_task(task)
            
            # Verify the task was processed
            assert processed_task.status == "completed"
            assert processed_task.result == mock_result
            
            # Verify execute_backtest was called with the correct parameters
            mock_execute.assert_called_once_with(
                strategy=task.strategy,
                data=task.data,
                parameters=task.parameters,
                start_date=task.start_date,
                end_date=task.end_date,
                initial_capital=task.initial_capital
            )
    
    def test_task_manager(self, sample_strategy, sample_price_data):
        """Test TaskManager class functionality."""
        # Create a task manager
        manager = TaskManager()
        
        # Verify initial state
        assert len(manager.tasks) == 0
        assert len(manager.pending_tasks) == 0
        assert len(manager.completed_tasks) == 0
        assert len(manager.failed_tasks) == 0
        
        # Create tasks
        task1 = BacktestTask(
            task_id="task_001",
            strategy=sample_strategy,
            data=sample_price_data,
            parameters={"fast_ma_period": 15, "slow_ma_period": 45},
            start_date="2023-01-15",
            end_date="2023-03-15",
            initial_capital=10000.0
        )
        
        task2 = BacktestTask(
            task_id="task_002",
            strategy=sample_strategy,
            data=sample_price_data,
            parameters={"fast_ma_period": 10, "slow_ma_period": 30},
            start_date="2023-01-15",
            end_date="2023-03-15",
            initial_capital=10000.0
        )
        
        # Add tasks to manager
        manager.add_task(task1)
        manager.add_task(task2)
        
        # Verify tasks were added
        assert len(manager.tasks) == 2
        assert len(manager.pending_tasks) == 2
        assert "task_001" in manager.tasks
        assert "task_002" in manager.tasks
        
        # Update task status
        task1.status = "completed"
        task1.result = {"total_return": 0.15}
        manager.update_task(task1)
        
        # Verify task was updated
        assert len(manager.pending_tasks) == 1
        assert len(manager.completed_tasks) == 1
        assert manager.tasks["task_001"].status == "completed"
        assert manager.tasks["task_001"].result["total_return"] == 0.15
        
        # Test get next task
        next_task = manager.get_next_task()
        assert next_task.task_id == "task_002"
        
        # Test task failure
        task2.status = "failed"
        task2.error = "Execution error"
        manager.update_task(task2)
        
        # Verify task failure was recorded
        assert len(manager.pending_tasks) == 0
        assert len(manager.completed_tasks) == 1
        assert len(manager.failed_tasks) == 1
        assert manager.tasks["task_002"].status == "failed"
        assert manager.tasks["task_002"].error == "Execution error"
        
        # Test get task by id
        retrieved_task = manager.get_task("task_001")
        assert retrieved_task.task_id == "task_001"
        assert retrieved_task.status == "completed"
    
    def test_parameter_optimizer(self, sample_strategy, sample_price_data):
        """Test ParameterOptimizer class functionality."""
        # Create a parameter optimizer
        optimizer = ParameterOptimizer()
        
        # Define parameter grid
        param_grid = {
            "fast_ma_period": [10, 15, 20],
            "slow_ma_period": [30, 40, 50],
            "stop_loss_pct": [0.03, 0.05]
        }
        
        # Generate tasks
        tasks = optimizer.generate_tasks(
            strategy=sample_strategy,
            data=sample_price_data,
            param_grid=param_grid,
            start_date="2023-01-15",
            end_date="2023-03-15",
            initial_capital=10000.0
        )
        
        # Verify tasks were generated correctly
        assert len(tasks) == 18  # 3 x 3 x 2 parameter combinations
        
        # Verify each task has a unique parameter combination
        param_combinations = set()
        for task in tasks:
            param_tuple = (
                task.parameters["fast_ma_period"],
                task.parameters["slow_ma_period"],
                task.parameters["stop_loss_pct"]
            )
            assert param_tuple not in param_combinations
            param_combinations.add(param_tuple)
        
        # Mock task results
        for i, task in enumerate(tasks):
            # Assign some results based on parameters to simulate optimization
            fast_period = task.parameters["fast_ma_period"]
            slow_period = task.parameters["slow_ma_period"]
            stop_loss = task.parameters["stop_loss_pct"]
            
            # Create a formula that favors certain parameter combinations
            # (just for testing purposes)
            return_value = 0.1 + (20 - fast_period) * 0.005 + (slow_period - 30) * 0.002 + (0.05 - stop_loss) * 0.1
            sharpe_value = 1.0 + return_value * 5
            
            task.status = "completed"
            task.result = {
                "total_return": return_value,
                "sharpe_ratio": sharpe_value,
                "max_drawdown": 0.1 - return_value * 0.2,
                "win_rate": 0.5 + return_value * 0.5
            }
        
        # Find best parameters
        best_params = optimizer.find_best_parameters(
            tasks=tasks,
            metric="total_return"
        )
        
        # Verify best parameters
        assert isinstance(best_params, dict)
        assert "parameters" in best_params
        assert "performance" in best_params
        
        # The best parameters should be the ones that maximize the return
        # Based on our formula, this should be:
        # fast_ma_period = 10 (lowest value)
        # slow_ma_period = 50 (highest value)
        # stop_loss_pct = 0.03 (lowest value)
        assert best_params["parameters"]["fast_ma_period"] == 10
        assert best_params["parameters"]["slow_ma_period"] == 50
        assert best_params["parameters"]["stop_loss_pct"] == 0.03
        
        # Test finding best parameters with a different metric
        best_params_sharpe = optimizer.find_best_parameters(
            tasks=tasks,
            metric="sharpe_ratio"
        )
        
        # Since our formula makes sharpe proportional to return,
        # the best parameters should be the same
        assert best_params_sharpe["parameters"]["fast_ma_period"] == 10
        assert best_params_sharpe["parameters"]["slow_ma_period"] == 50
        assert best_params_sharpe["parameters"]["stop_loss_pct"] == 0.03
    
    def test_distributed_backtester(self, sample_strategy, sample_price_data):
        """Test DistributedBacktester class functionality."""
        # Create a distributed backtester
        backtester = DistributedBacktester(num_workers=3)
        
        # Verify initialization
        assert backtester.num_workers == 3
        assert isinstance(backtester.task_manager, TaskManager)
        assert len(backtester.workers) == 3
        
        # Mock worker process_task method
        for worker in backtester.workers:
            worker.process_task = MagicMock()
            
            # Configure mock to return a completed task with results
            def side_effect(task):
                task.status = "completed"
                task.result = {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.08,
                    "win_rate": 0.65
                }
                return task
            
            worker.process_task.side_effect = side_effect
        
        # Define parameter grid for optimization
        param_grid = {
            "fast_ma_period": [10, 15, 20],
            "slow_ma_period": [30, 40, 50]
        }
        
        # Run parameter optimization
        with patch.object(backtester, '_distribute_tasks') as mock_distribute:
            # Configure mock to simulate task distribution and processing
            def distribute_side_effect(tasks):
                # Mark all tasks as completed with results
                for task in tasks:
                    task.status = "completed"
                    # Create results based on parameters
                    fast_period = task.parameters["fast_ma_period"]
                    slow_period = task.parameters["slow_ma_period"]
                    return_value = 0.1 + (20 - fast_period) * 0.005 + (slow_period - 30) * 0.002
                    
                    task.result = {
                        "total_return": return_value,
                        "sharpe_ratio": 1.0 + return_value * 5,
                        "max_drawdown": 0.1 - return_value * 0.2,
                        "win_rate": 0.5 + return_value * 0.5
                    }
                return tasks
            
            mock_distribute.side_effect = distribute_side_effect
            
            # Run optimization
            optimization_result = backtester.optimize_parameters(
                strategy=sample_strategy,
                data=sample_price_data,
                param_grid=param_grid,
                start_date="2023-01-15",
                end_date="2023-03-15",
                initial_capital=10000.0,
                metric="total_return"
            )
            
            # Verify optimization result
            assert isinstance(optimization_result, dict)
            assert "best_parameters" in optimization_result
            assert "performance" in optimization_result
            assert "all_results" in optimization_result
            
            # Verify best parameters
            assert optimization_result["best_parameters"]["fast_ma_period"] == 10
            assert optimization_result["best_parameters"]["slow_ma_period"] == 50
            
            # Verify all results were returned
            assert len(optimization_result["all_results"]) == 9  # 3 x 3 parameter combinations
            
            # Verify _distribute_tasks was called
            mock_distribute.assert_called_once()
    
    def test_walk_forward_optimization(self, sample_strategy, sample_price_data):
        """Test walk-forward optimization functionality."""
        # Create a distributed backtester
        backtester = DistributedBacktester(num_workers=2)
        
        # Mock the optimize_parameters method
        with patch.object(backtester, 'optimize_parameters') as mock_optimize:
            # Configure mock to return different optimal parameters for different periods
            def optimize_side_effect(strategy, data, param_grid, start_date, end_date, initial_capital, metric):
                # Return different results based on the date range
                if start_date == "2023-01-01":
                    return {
                        "best_parameters": {"fast_ma_period": 10, "slow_ma_period": 50},
                        "performance": {"total_return": 0.15}
                    }
                elif start_date == "2023-02-01":
                    return {
                        "best_parameters": {"fast_ma_period": 15, "slow_ma_period": 40},
                        "performance": {"total_return": 0.12}
                    }
                else:
                    return {
                        "best_parameters": {"fast_ma_period": 20, "slow_ma_period": 30},
                        "performance": {"total_return": 0.10}
                    }
            
            mock_optimize.side_effect = optimize_side_effect
            
            # Define parameter grid
            param_grid = {
                "fast_ma_period": [10, 15, 20],
                "slow_ma_period": [30, 40, 50]
            }
            
            # Run walk-forward optimization
            wfo_result = backtester.walk_forward_optimization(
                strategy=sample_strategy,
                data=sample_price_data,
                param_grid=param_grid,
                start_date="2023-01-01",
                end_date="2023-04-01",
                initial_capital=10000.0,
                train_period=30,
                test_period=15,
                metric="total_return"
            )
            
            # Verify the result
            assert isinstance(wfo_result, dict)
            assert "periods" in wfo_result
            assert "overall_performance" in wfo_result
            
            # Verify periods
            assert len(wfo_result["periods"]) > 0
            
            # Verify each period has the expected structure
            for period in wfo_result["periods"]:
                assert "train_start" in period
                assert "train_end" in period
                assert "test_start" in period
                assert "test_end" in period
                assert "best_parameters" in period
                assert "train_performance" in period
                assert "test_performance" in period
            
            # Verify optimize_parameters was called multiple times
            assert mock_optimize.call_count > 1
    
    def test_monte_carlo_simulation(self, sample_strategy, sample_price_data):
        """Test Monte Carlo simulation functionality."""
        # Create a distributed backtester
        backtester = DistributedBacktester(num_workers=2)
        
        # Mock the run_backtest method
        with patch.object(backtester, 'run_backtest') as mock_backtest:
            # Configure mock to return sample backtest results
            mock_backtest.return_value = {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.65,
                "trades": [
                    {"entry_date": "2023-01-20", "exit_date": "2023-02-05", "return": 0.07},
                    {"entry_date": "2023-02-15", "exit_date": "2023-03-01", "return": 0.08}
                ]
            }
            
            # Run Monte Carlo simulation
            mc_result = backtester.monte_carlo_simulation(
                strategy=sample_strategy,
                data=sample_price_data,
                parameters={"fast_ma_period": 15, "slow_ma_period": 45},
                start_date="2023-01-15",
                end_date="2023-03-15",
                initial_capital=10000.0,
                num_simulations=100,
                confidence_level=0.95
            )
            
            # Verify the result
            assert isinstance(mc_result, dict)
            assert "simulations" in mc_result
            assert "statistics" in mc_result
            assert "confidence_intervals" in mc_result
            
            # Verify simulations
            assert len(mc_result["simulations"]) == 100
            
            # Verify statistics
            assert "mean_return" in mc_result["statistics"]
            assert "std_return" in mc_result["statistics"]
            assert "median_return" in mc_result["statistics"]
            assert "min_return" in mc_result["statistics"]
            assert "max_return" in mc_result["statistics"]
            
            # Verify confidence intervals
            assert "lower_bound" in mc_result["confidence_intervals"]
            assert "upper_bound" in mc_result["confidence_intervals"]
            
            # Verify run_backtest was called multiple times
            assert mock_backtest.call_count == 100
