import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import psutil
import gc
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BacktestOptimizer")

class BacktestOptimizer:
    """
    Optimized backtesting engine for improved performance when testing strategies
    on large datasets or running multiple parameter combinations.
    """
    
    def __init__(self, data_dir="backtest_data", max_workers=None, use_processes=True):
        """
        Initialize the backtest optimizer.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store backtest data and results
        max_workers : int, optional
            Maximum number of workers for parallel processing
            If None, uses CPU count - 1 (leaves one core free)
        use_processes : bool
            Whether to use processes (True) or threads (False) for parallelization
        """
        self.data_dir = data_dir
        self.max_workers = max_workers if max_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        self.use_processes = use_processes
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"BacktestOptimizer initialized with {self.max_workers} workers using {'processes' if use_processes else 'threads'}")
    
    def optimize_dataframe(self, df):
        """
        Optimize a pandas DataFrame for memory usage.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to optimize
            
        Returns:
        --------
        pandas.DataFrame
            Optimized DataFrame
        """
        start_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Convert to smallest possible integer type
            if c_min >= 0:
                if c_max < 2**8:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 2**16:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < 2**32:
                    df[col] = df[col].astype(np.uint32)
            else:
                if c_min > -2**7 and c_max < 2**7:
                    df[col] = df[col].astype(np.int8)
                elif c_min > -2**15 and c_max < 2**15:
                    df[col] = df[col].astype(np.int16)
                elif c_min > -2**31 and c_max < 2**31:
                    df[col] = df[col].astype(np.int32)
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].astype(np.float32)
        
        # Optimize object columns (usually strings)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"DataFrame memory usage after optimization: {end_mem:.2f} MB")
        logger.info(f"Memory usage reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")
        
        return df
    
    def run_parallel_backtests(self, strategy_func, param_grid, data, **kwargs):
        """
        Run multiple backtests in parallel with different parameter combinations.
        
        Parameters:
        -----------
        strategy_func : function
            Function that implements the strategy and returns performance metrics
        param_grid : dict
            Dictionary of parameter names and lists of values to test
        data : pandas.DataFrame
            Market data for backtesting
        **kwargs : dict
            Additional keyword arguments to pass to strategy_func
            
        Returns:
        --------
        pandas.DataFrame
            Results of all parameter combinations with performance metrics
        """
        try:
            # Generate all parameter combinations
            import itertools
            param_keys = list(param_grid.keys())
            param_values = list(itertools.product(*[param_grid[k] for k in param_keys]))
            param_dicts = [dict(zip(param_keys, values)) for values in param_values]
            
            logger.info(f"Running {len(param_dicts)} parameter combinations in parallel")
            
            # Define worker function
            def worker(params, data=data, **kwargs):
                try:
                    # Create a copy of the data to avoid race conditions
                    data_copy = data.copy()
                    
                    # Run strategy with these parameters
                    start_time = time.time()
                    result = strategy_func(data_copy, **params, **kwargs)
                    end_time = time.time()
                    
                    # Add execution time and parameters to result
                    if isinstance(result, dict):
                        result['execution_time'] = end_time - start_time
                        for k, v in params.items():
                            result[k] = v
                    
                    return result
                except Exception as e:
                    logger.error(f"Error in worker with params {params}: {str(e)}")
                    logger.error(traceback.format_exc())
                    return None
            
            # Run in parallel
            results = []
            executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                futures = [executor.submit(worker, params, **kwargs) for params in param_dicts]
                
                # Collect results as they complete
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                        
                        # Log progress
                        if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                            logger.info(f"Completed {i + 1}/{len(futures)} parameter combinations")
                    
                    except Exception as e:
                        logger.error(f"Error getting result from future: {str(e)}")
                        logger.error(traceback.format_exc())
            
            # Convert results to DataFrame
            if results:
                results_df = pd.DataFrame(results)
                return results_df
            else:
                logger.warning("No valid results returned from parallel backtests")
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=param_keys + ['execution_time'])
        except Exception as e:
            logger.error(f"Error in run_parallel_backtests: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def optimize_strategy(self, strategy_func, param_grid, data, metric='sharpe_ratio', **kwargs):
        """
        Find the optimal parameters for a strategy based on a performance metric.
        
        Parameters:
        -----------
        strategy_func : function
            Function that implements the strategy and returns performance metrics
        param_grid : dict
            Dictionary of parameter names and lists of values to test
        data : pandas.DataFrame
            Market data for backtesting
        metric : str
            Performance metric to optimize (e.g., 'sharpe_ratio', 'total_return')
        **kwargs : dict
            Additional keyword arguments to pass to strategy_func
            
        Returns:
        --------
        dict
            Best parameters and performance metrics
        """
        try:
            # Run parallel backtests
            results_df = self.run_parallel_backtests(strategy_func, param_grid, data, **kwargs)
            
            if results_df.empty:
                logger.warning("No valid results to optimize")
                return None
            
            # Find best parameters
            if metric in results_df.columns:
                # Handle different optimization directions
                if metric in ['sharpe_ratio', 'total_return', 'win_rate']:
                    best_idx = results_df[metric].idxmax()
                elif metric in ['max_drawdown', 'volatility']:
                    best_idx = results_df[metric].idxmin()
                else:
                    # Default to maximizing
                    best_idx = results_df[metric].idxmax()
                
                best_result = results_df.loc[best_idx].to_dict()
                
                logger.info(f"Best parameters found: {best_result}")
                return best_result
            else:
                logger.error(f"Metric '{metric}' not found in results")
                return None
        except Exception as e:
            logger.error(f"Error in optimize_strategy: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def chunked_backtest(self, strategy_func, data, chunk_size=None, **kwargs):
        """
        Run a backtest on large datasets by processing in chunks to reduce memory usage.
        
        Parameters:
        -----------
        strategy_func : function
            Function that implements the strategy and returns performance metrics
        data : pandas.DataFrame
            Market data for backtesting
        chunk_size : int, optional
            Size of each chunk in days, if None calculates based on available memory
        **kwargs : dict
            Additional keyword arguments to pass to strategy_func
            
        Returns:
        --------
        dict
            Combined performance metrics
        """
        try:
            # Determine chunk size if not provided
            if chunk_size is None:
                # Get available memory in GB
                available_memory = psutil.virtual_memory().available / (1024**3)
                
                # Estimate memory usage per row (very rough estimate)
                estimated_row_size_mb = data.memory_usage(deep=True).sum() / (len(data) * 1024**2)
                
                # Calculate chunk size to use 25% of available memory
                chunk_size = int((available_memory * 0.25 * 1024) / estimated_row_size_mb)
                
                # Ensure chunk size is at least 100 days
                chunk_size = max(100, chunk_size)
                
                logger.info(f"Automatically determined chunk size: {chunk_size} days")
            
            # Split data into chunks
            total_days = len(data)
            num_chunks = (total_days + chunk_size - 1) // chunk_size  # Ceiling division
            
            logger.info(f"Processing {total_days} days in {num_chunks} chunks of {chunk_size} days")
            
            # Process each chunk
            all_trades = []
            all_signals = []
            all_metrics = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_days)
                
                logger.info(f"Processing chunk {i+1}/{num_chunks} (days {start_idx} to {end_idx})")
                
                # Get chunk of data
                chunk_data = data.iloc[start_idx:end_idx].copy()
                
                # Run strategy on chunk
                result = strategy_func(chunk_data, **kwargs)
                
                # Collect results
                if 'trades' in result:
                    all_trades.extend(result['trades'])
                
                if 'signals' in result:
                    all_signals.append(result['signals'])
                
                all_metrics.append({
                    k: v for k, v in result.items() 
                    if k not in ['trades', 'signals'] and not isinstance(v, (list, dict, pd.DataFrame))
                })
                
                # Force garbage collection
                gc.collect()
            
            # Combine results
            combined_result = {}
            
            # Combine trades
            if all_trades:
                combined_result['trades'] = all_trades
            
            # Combine signals
            if all_signals:
                combined_result['signals'] = pd.concat(all_signals)
            
            # Combine metrics (average or sum as appropriate)
            metrics_df = pd.DataFrame(all_metrics)
            for col in metrics_df.columns:
                # Metrics to sum
                if col in ['total_trades', 'winning_trades', 'losing_trades']:
                    combined_result[col] = metrics_df[col].sum()
                # Metrics to average weighted by chunk size
                else:
                    combined_result[col] = metrics_df[col].mean()
            
            # Calculate overall win rate
            if 'winning_trades' in combined_result and 'total_trades' in combined_result:
                combined_result['win_rate'] = combined_result['winning_trades'] / combined_result['total_trades'] if combined_result['total_trades'] > 0 else 0
            
            logger.info(f"Completed chunked backtest with {len(all_trades)} total trades")
            
            return combined_result
        except Exception as e:
            logger.error(f"Error in chunked_backtest: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def monte_carlo_simulation(self, trades, initial_capital=10000, num_simulations=1000):
        """
        Run Monte Carlo simulations to analyze the robustness of a strategy.
        
        Parameters:
        -----------
        trades : list
            List of trade results (percentage returns)
        initial_capital : float
            Initial capital for simulations
        num_simulations : int
            Number of Monte Carlo simulations to run
            
        Returns:
        --------
        dict
            Simulation results including confidence intervals
        """
        try:
            logger.info(f"Running {num_simulations} Monte Carlo simulations")
            
            # Convert trades to numpy array for performance
            trade_returns = np.array([float(t.replace('%', '')) / 100 if isinstance(t, str) else t for t in trades])
            
            # Run simulations in parallel
            def run_simulation(seed):
                np.random.seed(seed)
                # Shuffle trade order
                shuffled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=False)
                
                # Calculate equity curve
                equity = initial_capital
                equity_curve = [equity]
                
                for ret in shuffled_returns:
                    equity *= (1 + ret)
                    equity_curve.append(equity)
                
                # Calculate metrics
                final_equity = equity_curve[-1]
                total_return = (final_equity / initial_capital - 1) * 100
                
                # Calculate drawdown
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - peak) / peak * 100
                max_drawdown = np.min(drawdown)
                
                return {
                    'equity_curve': equity_curve,
                    'final_equity': final_equity,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown
                }
            
            # Run simulations in parallel
            executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                simulation_results = list(executor.map(run_simulation, range(num_simulations)))
            
            # Analyze results
            final_equities = [result['final_equity'] for result in simulation_results]
            total_returns = [result['total_return'] for result in simulation_results]
            max_drawdowns = [result['max_drawdown'] for result in simulation_results]
            
            # Calculate statistics
            mean_return = np.mean(total_returns)
            median_return = np.median(total_returns)
            std_return = np.std(total_returns)
            
            mean_drawdown = np.mean(max_drawdowns)
            median_drawdown = np.median(max_drawdowns)
            worst_drawdown = np.min(max_drawdowns)
            
            # Calculate confidence intervals
            ci_5 = np.percentile(total_returns, 5)
            ci_95 = np.percentile(total_returns, 95)
            
            # Calculate probability of profit
            prob_profit = np.mean(np.array(total_returns) > 0) * 100
            
            # Calculate probability of exceeding certain returns
            prob_exceed_5 = np.mean(np.array(total_returns) > 5) * 100
            prob_exceed_10 = np.mean(np.array(total_returns) > 10) * 100
            prob_exceed_20 = np.mean(np.array(total_returns) > 20) * 100
            
            # Calculate probability of drawdown exceeding certain thresholds
            prob_dd_exceed_10 = np.mean(np.array(max_drawdowns) < -10) * 100
            prob_dd_exceed_20 = np.mean(np.array(max_drawdowns) < -20) * 100
            
            # Prepare result
            result = {
                'mean_return': mean_return,
                'median_return': median_return,
                'std_return': std_return,
                'ci_5': ci_5,
                'ci_95': ci_95,
                'mean_drawdown': mean_drawdown,
                'median_drawdown': median_drawdown,
                'worst_drawdown': worst_drawdown,
                'prob_profit': prob_profit,
                'prob_exceed_5': prob_exceed_5,
                'prob_exceed_10': prob_exceed_10,
                'prob_exceed_20': prob_exceed_20,
                'prob_dd_exceed_10': prob_dd_exceed_10,
                'prob_dd_exceed_20': prob_dd_exceed_20,
                'returns': total_returns,
                'drawdowns': max_drawdowns
            }
            
            logger.info(f"Monte Carlo simulation completed: Mean return: {mean_return:.2f}%, 90% CI: [{ci_5:.2f}%, {ci_95:.2f}%]")
            
            return result
        except Exception as e:
            logger.error(f"Error in monte_carlo_simulation: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def plot_monte_carlo_results(self, simulation_results, save_path=None):
        """
        Plot Monte Carlo simulation results.
        
        Parameters:
        -----------
        simulation_results : dict
            Results from monte_carlo_simulation
        save_path : str, optional
            Path to save the plot, if None shows the plot
            
        Returns:
        --------
        str
            Path to saved plot if save_path is provided, None otherwise
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot return distribution
            ax1.hist(simulation_results['returns'], bins=50, alpha=0.7, color='blue')
            ax1.axvline(simulation_results['mean_return'], color='red', linestyle='--', label=f'Mean: {simulation_results["mean_return"]:.2f}%')
            ax1.axvline(simulation_results['ci_5'], color='green', linestyle='--', label=f'5% CI: {simulation_results["ci_5"]:.2f}%')
            ax1.axvline(simulation_results['ci_95'], color='green', linestyle='--', label=f'95% CI: {simulation_results["ci_95"]:.2f}%')
            ax1.set_title('Distribution of Returns')
            ax1.set_xlabel('Total Return (%)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            
            # Plot drawdown distribution
            ax2.hist(simulation_results['drawdowns'], bins=50, alpha=0.7, color='red')
            ax2.axvline(simulation_results['mean_drawdown'], color='blue', linestyle='--', label=f'Mean: {simulation_results["mean_drawdown"]:.2f}%')
            ax2.axvline(simulation_results['worst_drawdown'], color='black', linestyle='--', label=f'Worst: {simulation_results["worst_drawdown"]:.2f}%')
            ax2.set_title('Distribution of Maximum Drawdowns')
            ax2.set_xlabel('Maximum Drawdown (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return save_path
            else:
                plt.show()
                return None
        except Exception as e:
            logger.error(f"Error in plot_monte_carlo_results: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def walk_forward_analysis(self, strategy_func, data, train_size=0.7, test_size=0.3, step=0.1, **kwargs):
        """
        Perform walk-forward analysis to test strategy robustness.
        
        Parameters:
        -----------
        strategy_func : function
            Function that implements the strategy and returns performance metrics
        data : pandas.DataFrame
            Market data for backtesting
        train_size : float
            Proportion of data to use for training
        test_size : float
            Proportion of data to use for testing
        step : float
            Step size for moving the window
        **kwargs : dict
            Additional keyword arguments to pass to strategy_func
            
        Returns:
        --------
        dict
            Walk-forward analysis results
        """
        try:
            logger.info(f"Performing walk-forward analysis with train_size={train_size}, test_size={test_size}, step={step}")
            
            # Calculate window sizes
            total_size = len(data)
            train_window = int(total_size * train_size)
            test_window = int(total_size * test_size)
            step_size = int(total_size * step)
            
            # Calculate number of windows
            num_windows = max(1, int((total_size - train_window - test_window) / step_size) + 1)
            
            logger.info(f"Data size: {total_size}, train window: {train_window}, test window: {test_window}, step size: {step_size}")
            logger.info(f"Number of windows: {num_windows}")
            
            # Perform walk-forward analysis
            results = []
            
            for i in range(num_windows):
                # Calculate window indices
                train_start = i * step_size
                train_end = train_start + train_window
                test_start = train_end
                test_end = min(test_start + test_window, total_size)
                
                logger.info(f"Window {i+1}/{num_windows}: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")
                
                # Get train and test data
                train_data = data.iloc[train_start:train_end].copy()
                test_data = data.iloc[test_start:test_end].copy()
                
                # Run strategy on train data to optimize parameters
                if 'param_grid' in kwargs:
                    param_grid = kwargs.pop('param_grid')
                    best_params = self.optimize_strategy(strategy_func, param_grid, train_data, **kwargs)
                    
                    if best_params:
                        # Extract strategy parameters
                        strategy_params = {k: v for k, v in best_params.items() if k in param_grid}
                        
                        # Update kwargs with best parameters
                        kwargs.update(strategy_params)
                
                # Run strategy on test data with optimized parameters
                test_result = strategy_func(test_data, **kwargs)
                
                # Add window information
                test_result['window'] = i + 1
                test_result['train_start'] = train_start
                test_result['train_end'] = train_end
                test_result['test_start'] = test_start
                test_result['test_end'] = test_end
                
                results.append(test_result)
                
                # Force garbage collection
                gc.collect()
            
            # Combine results
            combined_results = {
                'window_results': results,
                'num_windows': num_windows
            }
            
            # Calculate overall metrics
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            for metric in metrics:
                if all(metric in result for result in results):
                    values = [result[metric] for result in results]
                    combined_results[f'mean_{metric}'] = np.mean(values)
                    combined_results[f'median_{metric}'] = np.median(values)
                    combined_results[f'std_{metric}'] = np.std(values)
                    combined_results[f'min_{metric}'] = np.min(values)
                    combined_results[f'max_{metric}'] = np.max(values)
            
            logger.info(f"Walk-forward analysis completed with {num_windows} windows")
            
            return combined_results
        except Exception as e:
            logger.error(f"Error in walk_forward_analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

# Create a global instance for use in the application
backtest_optimizer = BacktestOptimizer()
