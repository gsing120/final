"""
Distributed Backtesting Module for Gemma Advanced Trading System.

This module provides functionality for running backtests in a distributed manner
to improve performance and handle large datasets.
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
import os
import json
import concurrent.futures
from functools import partial
import multiprocessing
import pickle
import hashlib
import uuid
import queue
import threading
import socket
import struct
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("distributed_backtesting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DistributedBacktesting")


class BacktestTask:
    """Class representing a single backtest task."""
    
    def __init__(self, strategy, data, parameters, start_date=None, end_date=None, task_id=None):
        """
        Initialize a backtest task.
        
        Parameters:
        -----------
        strategy : object
            Strategy object to backtest
        data : dict
            Market data for the backtest
        parameters : dict
            Parameters for the backtest
        start_date : datetime, optional
            Start date for the backtest
        end_date : datetime, optional
            End date for the backtest
        task_id : str, optional
            Unique identifier for the task
        """
        self.strategy = strategy
        self.data = data
        self.parameters = parameters
        self.start_date = start_date
        self.end_date = end_date
        self.task_id = task_id or str(uuid.uuid4())
        self.status = "created"
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        
    def to_dict(self):
        """
        Convert the task to a dictionary.
        
        Returns:
        --------
        dict
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "parameters": self.parameters,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
        
    def serialize(self):
        """
        Serialize the task for transmission.
        
        Returns:
        --------
        bytes
            Serialized task
        """
        return pickle.dumps(self)
        
    @staticmethod
    def deserialize(data):
        """
        Deserialize a task.
        
        Parameters:
        -----------
        data : bytes
            Serialized task
            
        Returns:
        --------
        BacktestTask
            Deserialized task
        """
        return pickle.loads(data)


class BacktestWorker:
    """Class representing a worker for running backtest tasks."""
    
    def __init__(self, worker_id=None):
        """
        Initialize a backtest worker.
        
        Parameters:
        -----------
        worker_id : str, optional
            Unique identifier for the worker
        """
        self.worker_id = worker_id or f"worker-{socket.gethostname()}-{os.getpid()}"
        self.status = "idle"
        self.current_task = None
        self.task_history = []
        self.start_time = datetime.now()
        self.tasks_completed = 0
        
    def run_task(self, task):
        """
        Run a backtest task.
        
        Parameters:
        -----------
        task : BacktestTask
            Task to run
            
        Returns:
        --------
        BacktestTask
            Completed task with results
        """
        self.current_task = task
        self.status = "running"
        task.status = "running"
        task.started_at = datetime.now()
        
        try:
            logger.info(f"Worker {self.worker_id} starting task {task.task_id}")
            
            # Run the backtest
            result = self._execute_backtest(task)
            
            # Update task with result
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            
            # Update worker status
            self.status = "idle"
            self.current_task = None
            self.tasks_completed += 1
            self.task_history.append(task.task_id)
            
            logger.info(f"Worker {self.worker_id} completed task {task.task_id}")
            
            return task
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} error on task {task.task_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update task with error
            task.error = str(e)
            task.status = "failed"
            task.completed_at = datetime.now()
            
            # Update worker status
            self.status = "idle"
            self.current_task = None
            self.task_history.append(task.task_id)
            
            return task
            
    def _execute_backtest(self, task):
        """
        Execute a backtest.
        
        Parameters:
        -----------
        task : BacktestTask
            Task to execute
            
        Returns:
        --------
        dict
            Backtest results
        """
        strategy = task.strategy
        data = task.data
        parameters = task.parameters
        start_date = task.start_date
        end_date = task.end_date
        
        # Apply parameters to strategy
        for param_name, param_value in parameters.items():
            if hasattr(strategy, param_name):
                setattr(strategy, param_name, param_value)
                
        # Run the backtest
        result = strategy.backtest(
            data=data,
            start_date=start_date,
            end_date=end_date
        )
        
        return result
        
    def to_dict(self):
        """
        Convert the worker to a dictionary.
        
        Returns:
        --------
        dict
            Dictionary representation of the worker
        """
        return {
            "worker_id": self.worker_id,
            "status": self.status,
            "current_task": self.current_task.task_id if self.current_task else None,
            "tasks_completed": self.tasks_completed,
            "uptime": (datetime.now() - self.start_time).total_seconds()
        }


class BacktestCoordinator:
    """Class for coordinating distributed backtest tasks."""
    
    def __init__(self, config=None):
        """
        Initialize the backtest coordinator.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the coordinator
        """
        self.config = config or {}
        self.default_config = {
            "max_workers": multiprocessing.cpu_count(),
            "task_timeout": 3600,  # 1 hour
            "result_dir": "./results/backtests",
            "cache_dir": "./cache/backtests",
            "use_process_pool": True,
            "use_thread_pool": False,
            "distributed_mode": "local",  # local, network, or cluster
            "network_host": "localhost",
            "network_port": 5555,
            "cluster_config": None
        }
        
        # Merge default config with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        # Initialize task queue and results
        self.task_queue = queue.Queue()
        self.results = {}
        self.workers = {}
        self.running = False
        self.coordinator_thread = None
        
        # Create result and cache directories
        os.makedirs(self.config["result_dir"], exist_ok=True)
        os.makedirs(self.config["cache_dir"], exist_ok=True)
        
        logger.info("BacktestCoordinator initialized")
        
    def start(self):
        """
        Start the backtest coordinator.
        
        Returns:
        --------
        bool
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Coordinator already running")
            return False
            
        self.running = True
        
        # Start coordinator thread
        self.coordinator_thread = threading.Thread(target=self._coordinator_loop)
        self.coordinator_thread.daemon = True
        self.coordinator_thread.start()
        
        logger.info("BacktestCoordinator started")
        
        return True
        
    def stop(self):
        """
        Stop the backtest coordinator.
        
        Returns:
        --------
        bool
            True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Coordinator not running")
            return False
            
        self.running = False
        
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=5.0)
            
        logger.info("BacktestCoordinator stopped")
        
        return True
        
    def _coordinator_loop(self):
        """Main coordinator loop."""
        logger.info("Coordinator loop started")
        
        while self.running:
            try:
                # Process any completed tasks
                self._process_completed_tasks()
                
                # Check for timed-out tasks
                self._check_task_timeouts()
                
                # Dispatch tasks to available workers
                self._dispatch_tasks()
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in coordinator loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(1.0)  # Sleep longer after an error
                
        logger.info("Coordinator loop stopped")
        
    def _process_completed_tasks(self):
        """Process completed tasks."""
        # Implementation depends on the distributed mode
        if self.config["distributed_mode"] == "local":
            # For local mode, workers update the results directly
            pass
        else:
            # For network or cluster mode, we would need to check for results
            # from remote workers
            pass
            
    def _check_task_timeouts(self):
        """Check for timed-out tasks."""
        current_time = datetime.now()
        timeout_seconds = self.config["task_timeout"]
        
        for worker_id, worker in list(self.workers.items()):
            if worker.status == "running" and worker.current_task:
                task = worker.current_task
                if task.started_at and (current_time - task.started_at).total_seconds() > timeout_seconds:
                    logger.warning(f"Task {task.task_id} timed out on worker {worker_id}")
                    
                    # Mark task as failed
                    task.status = "failed"
                    task.error = "Task timed out"
                    task.completed_at = current_time
                    
                    # Update worker status
                    worker.status = "idle"
                    worker.current_task = None
                    
                    # Requeue the task
                    self.task_queue.put(task)
                    
    def _dispatch_tasks(self):
        """Dispatch tasks to available workers."""
        # Check for available workers
        available_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker.status == "idle"
        ]
        
        if not available_workers:
            # No available workers
            return
            
        # Dispatch tasks to available workers
        for worker_id in available_workers:
            if self.task_queue.empty():
                # No more tasks to dispatch
                break
                
            try:
                # Get next task
                task = self.task_queue.get_nowait()
                
                # Dispatch task to worker
                worker = self.workers[worker_id]
                
                if self.config["distributed_mode"] == "local":
                    # For local mode, run the task in a separate thread
                    threading.Thread(
                        target=self._run_task_on_worker,
                        args=(worker, task)
                    ).start()
                else:
                    # For network or cluster mode, we would need to send the task
                    # to the remote worker
                    pass
                    
            except queue.Empty:
                # No more tasks
                break
                
    def _run_task_on_worker(self, worker, task):
        """
        Run a task on a worker.
        
        Parameters:
        -----------
        worker : BacktestWorker
            Worker to run the task on
        task : BacktestTask
            Task to run
        """
        try:
            # Run the task
            completed_task = worker.run_task(task)
            
            # Store the result
            self.results[completed_task.task_id] = completed_task
            
            # Save the result to disk
            self._save_result(completed_task)
            
        except Exception as e:
            logger.error(f"Error running task {task.task_id} on worker {worker.worker_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Mark task as failed
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()
            
            # Update worker status
            worker.status = "idle"
            worker.current_task = None
            
            # Store the result
            self.results[task.task_id] = task
            
    def _save_result(self, task):
        """
        Save a task result to disk.
        
        Parameters:
        -----------
        task : BacktestTask
            Completed task
        """
        try:
            # Create result file path
            result_path = os.path.join(self.config["result_dir"], f"{task.task_id}.json")
            
            # Extract result data
            result_data = {
                "task_id": task.task_id,
                "parameters": task.parameters,
                "start_date": task.start_date.isoformat() if task.start_date else None,
                "end_date": task.end_date.isoformat() if task.end_date else None,
                "status": task.status,
                "error": task.error,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            
            # Add result if available
            if task.result:
                result_data["result"] = task.result
                
            # Save to file
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=4)
                
            logger.debug(f"Saved result for task {task.task_id}")
            
        except Exception as e:
            logger.error(f"Error saving result for task {task.task_id}: {str(e)}")
            
    def add_worker(self, worker=None):
        """
        Add a worker to the coordinator.
        
        Parameters:
        -----------
        worker : BacktestWorker, optional
            Worker to add (if None, a new worker is created)
            
        Returns:
        --------
        str
            Worker ID
        """
        if worker is None:
            worker = BacktestWorker()
            
        self.workers[worker.worker_id] = worker
        logger.info(f"Added worker {worker.worker_id}")
        
        return worker.worker_id
        
    def remove_worker(self, worker_id):
        """
        Remove a worker from the coordinator.
        
        Parameters:
        -----------
        worker_id : str
            ID of the worker to remove
            
        Returns:
        --------
        bool
            True if removed successfully, False otherwise
        """
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found")
            return False
            
        del self.workers[worker_id]
        logger.info(f"Removed worker {worker_id}")
        
        return True
        
    def add_task(self, task):
        """
        Add a task to the queue.
        
        Parameters:
        -----------
        task : BacktestTask
            Task to add
            
        Returns:
        --------
        str
            Task ID
        """
        self.task_queue.put(task)
        logger.info(f"Added task {task.task_id} to queue")
        
        return task.task_id
        
    def get_task_status(self, task_id):
        """
        Get the status of a task.
        
        Parameters:
        -----------
        task_id : str
            ID of the task
            
        Returns:
        --------
        dict
            Task status
        """
        # Check if task is in results
        if task_id in self.results:
            task = self.results[task_id]
            return {
                "task_id": task.task_id,
                "status": task.status,
                "error": task.error,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            
        # Check if task is in queue
        for i in range(self.task_queue.qsize()):
            try:
                task = self.task_queue.queue[i]
                if task.task_id == task_id:
                    return {
                        "task_id": task.task_id,
                        "status": "queued",
                        "created_at": task.created_at.isoformat()
                    }
            except:
                pass
                
        # Check if task is running on a worker
        for worker_id, worker in self.workers.items():
            if worker.current_task and worker.current_task.task_id == task_id:
                task = worker.current_task
                return {
                    "task_id": task.task_id,
                    "status": "running",
                    "worker_id": worker_id,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None
                }
                
        # Check if task result is on disk
        result_path = os.path.join(self.config["result_dir"], f"{task_id}.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                    return {
                        "task_id": result_data["task_id"],
                        "status": result_data["status"],
                        "error": result_data.get("error"),
                        "created_at": result_data["created_at"],
                        "started_at": result_data.get("started_at"),
                        "completed_at": result_data.get("completed_at")
                    }
            except:
                pass
                
        # Task not found
        return {
            "task_id": task_id,
            "status": "unknown"
        }
        
    def get_task_result(self, task_id):
        """
        Get the result of a completed task.
        
        Parameters:
        -----------
        task_id : str
            ID of the task
            
        Returns:
        --------
        dict
            Task result or None if not found
        """
        # Check if task is in results
        if task_id in self.results:
            task = self.results[task_id]
            if task.status == "completed":
                return task.result
                
        # Check if task result is on disk
        result_path = os.path.join(self.config["result_dir"], f"{task_id}.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                    if result_data["status"] == "completed":
                        return result_data.get("result")
            except:
                pass
                
        # Task result not found
        return None
        
    def get_all_tasks(self):
        """
        Get all tasks.
        
        Returns:
        --------
        list
            List of task statuses
        """
        tasks = []
        
        # Get tasks from results
        for task_id, task in self.results.items():
            tasks.append({
                "task_id": task.task_id,
                "status": task.status,
                "error": task.error,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            })
            
        # Get tasks from queue
        for i in range(self.task_queue.qsize()):
            try:
                task = self.task_queue.queue[i]
                tasks.append({
                    "task_id": task.task_id,
                    "status": "queued",
                    "created_at": task.created_at.isoformat()
                })
            except:
                pass
                
        # Get tasks from workers
        for worker_id, worker in self.workers.items():
            if worker.current_task:
                task = worker.current_task
                tasks.append({
                    "task_id": task.task_id,
                    "status": "running",
                    "worker_id": worker_id,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None
                })
                
        return tasks
        
    def get_worker_status(self, worker_id):
        """
        Get the status of a worker.
        
        Parameters:
        -----------
        worker_id : str
            ID of the worker
            
        Returns:
        --------
        dict
            Worker status or None if not found
        """
        if worker_id in self.workers:
            return self.workers[worker_id].to_dict()
        else:
            return None
            
    def get_all_workers(self):
        """
        Get all workers.
        
        Returns:
        --------
        list
            List of worker statuses
        """
        return [worker.to_dict() for worker in self.workers.values()]
        
    def clear_completed_tasks(self):
        """
        Clear completed tasks from memory.
        
        Returns:
        --------
        int
            Number of tasks cleared
        """
        cleared_count = 0
        
        # Clear completed tasks from results
        for task_id in list(self.results.keys()):
            task = self.results[task_id]
            if task.status in ["completed", "failed"]:
                del self.results[task_id]
                cleared_count += 1
                
        logger.info(f"Cleared {cleared_count} completed tasks")
        
        return cleared_count


class DistributedBacktesting:
    """Main class for distributed backtesting."""
    
    def __init__(self, config=None):
        """
        Initialize the distributed backtesting system.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the system
        """
        self.config = config or {}
        self.default_config = {
            "max_workers": multiprocessing.cpu_count(),
            "task_timeout": 3600,  # 1 hour
            "result_dir": "./results/backtests",
            "cache_dir": "./cache/backtests",
            "use_process_pool": True,
            "use_thread_pool": False,
            "distributed_mode": "local",  # local, network, or cluster
            "network_host": "localhost",
            "network_port": 5555,
            "cluster_config": None,
            "parameter_grid_max_size": 1000,
            "optimization_method": "grid",  # grid, random, bayesian
            "optimization_iterations": 100,
            "optimization_random_state": 42,
            "walk_forward_windows": 5,
            "walk_forward_train_size": 0.7
        }
        
        # Merge default config with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        # Initialize coordinator
        self.coordinator = BacktestCoordinator(config=self.config)
        
        # Create result and cache directories
        os.makedirs(self.config["result_dir"], exist_ok=True)
        os.makedirs(self.config["cache_dir"], exist_ok=True)
        
        logger.info("DistributedBacktesting initialized")
        
    def start(self):
        """
        Start the distributed backtesting system.
        
        Returns:
        --------
        bool
            True if started successfully, False otherwise
        """
        # Start coordinator
        if not self.coordinator.start():
            logger.error("Failed to start coordinator")
            return False
            
        # Add workers
        for _ in range(self.config["max_workers"]):
            self.coordinator.add_worker()
            
        logger.info(f"Started distributed backtesting with {self.config['max_workers']} workers")
        
        return True
        
    def stop(self):
        """
        Stop the distributed backtesting system.
        
        Returns:
        --------
        bool
            True if stopped successfully, False otherwise
        """
        return self.coordinator.stop()
        
    def run_backtest(self, strategy, data, parameters=None, start_date=None, end_date=None):
        """
        Run a single backtest.
        
        Parameters:
        -----------
        strategy : object
            Strategy object to backtest
        data : dict
            Market data for the backtest
        parameters : dict, optional
            Parameters for the backtest
        start_date : datetime, optional
            Start date for the backtest
        end_date : datetime, optional
            End date for the backtest
            
        Returns:
        --------
        str
            Task ID
        """
        # Create task
        task = BacktestTask(
            strategy=strategy,
            data=data,
            parameters=parameters or {},
            start_date=start_date,
            end_date=end_date
        )
        
        # Add task to coordinator
        return self.coordinator.add_task(task)
        
    def run_parameter_optimization(self, strategy, data, parameter_grid, start_date=None, end_date=None, method=None):
        """
        Run parameter optimization.
        
        Parameters:
        -----------
        strategy : object
            Strategy object to backtest
        data : dict
            Market data for the backtest
        parameter_grid : dict
            Grid of parameters to optimize
        start_date : datetime, optional
            Start date for the backtest
        end_date : datetime, optional
            End date for the backtest
        method : str, optional
            Optimization method (grid, random, bayesian)
            
        Returns:
        --------
        str
            Optimization ID
        """
        method = method or self.config["optimization_method"]
        
        if method == "grid":
            return self._run_grid_search(strategy, data, parameter_grid, start_date, end_date)
        elif method == "random":
            return self._run_random_search(strategy, data, parameter_grid, start_date, end_date)
        elif method == "bayesian":
            return self._run_bayesian_optimization(strategy, data, parameter_grid, start_date, end_date)
        else:
            logger.error(f"Unknown optimization method: {method}")
            return None
            
    def _run_grid_search(self, strategy, data, parameter_grid, start_date=None, end_date=None):
        """
        Run grid search optimization.
        
        Parameters:
        -----------
        strategy : object
            Strategy object to backtest
        data : dict
            Market data for the backtest
        parameter_grid : dict
            Grid of parameters to optimize
        start_date : datetime, optional
            Start date for the backtest
        end_date : datetime, optional
            End date for the backtest
            
        Returns:
        --------
        str
            Optimization ID
        """
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_grid)
        
        # Check if grid is too large
        if len(param_combinations) > self.config["parameter_grid_max_size"]:
            logger.warning(f"Parameter grid size ({len(param_combinations)}) exceeds maximum ({self.config['parameter_grid_max_size']})")
            logger.warning("Using random search instead")
            return self._run_random_search(strategy, data, parameter_grid, start_date, end_date)
            
        # Create optimization ID
        optimization_id = f"grid_search_{uuid.uuid4()}"
        
        # Create tasks for each parameter combination
        task_ids = []
        for params in param_combinations:
            task = BacktestTask(
                strategy=strategy,
                data=data,
                parameters=params,
                start_date=start_date,
                end_date=end_date
            )
            
            # Add metadata to task
            task.parameters["optimization_id"] = optimization_id
            
            # Add task to coordinator
            task_id = self.coordinator.add_task(task)
            task_ids.append(task_id)
            
        # Save optimization metadata
        self._save_optimization_metadata(
            optimization_id=optimization_id,
            method="grid",
            parameter_grid=parameter_grid,
            task_ids=task_ids,
            start_date=start_date,
            end_date=end_date
        )
        
        return optimization_id
        
    def _run_random_search(self, strategy, data, parameter_grid, start_date=None, end_date=None):
        """
        Run random search optimization.
        
        Parameters:
        -----------
        strategy : object
            Strategy object to backtest
        data : dict
            Market data for the backtest
        parameter_grid : dict
            Grid of parameters to optimize
        start_date : datetime, optional
            Start date for the backtest
        end_date : datetime, optional
            End date for the backtest
            
        Returns:
        --------
        str
            Optimization ID
        """
        # Set random seed
        np.random.seed(self.config["optimization_random_state"])
        
        # Determine number of iterations
        n_iterations = self.config["optimization_iterations"]
        
        # Create optimization ID
        optimization_id = f"random_search_{uuid.uuid4()}"
        
        # Create tasks for each iteration
        task_ids = []
        for _ in range(n_iterations):
            # Generate random parameters
            params = {}
            for param_name, param_values in parameter_grid.items():
                if isinstance(param_values, list):
                    # Discrete parameter
                    params[param_name] = np.random.choice(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # Continuous parameter
                    low, high = param_values
                    if isinstance(low, int) and isinstance(high, int):
                        # Integer parameter
                        params[param_name] = np.random.randint(low, high + 1)
                    else:
                        # Float parameter
                        params[param_name] = np.random.uniform(low, high)
                        
            # Create task
            task = BacktestTask(
                strategy=strategy,
                data=data,
                parameters=params,
                start_date=start_date,
                end_date=end_date
            )
            
            # Add metadata to task
            task.parameters["optimization_id"] = optimization_id
            
            # Add task to coordinator
            task_id = self.coordinator.add_task(task)
            task_ids.append(task_id)
            
        # Save optimization metadata
        self._save_optimization_metadata(
            optimization_id=optimization_id,
            method="random",
            parameter_grid=parameter_grid,
            task_ids=task_ids,
            start_date=start_date,
            end_date=end_date
        )
        
        return optimization_id
        
    def _run_bayesian_optimization(self, strategy, data, parameter_grid, start_date=None, end_date=None):
        """
        Run Bayesian optimization.
        
        Parameters:
        -----------
        strategy : object
            Strategy object to backtest
        data : dict
            Market data for the backtest
        parameter_grid : dict
            Grid of parameters to optimize
        start_date : datetime, optional
            Start date for the backtest
        end_date : datetime, optional
            End date for the backtest
            
        Returns:
        --------
        str
            Optimization ID
        """
        logger.warning("Bayesian optimization not fully implemented")
        logger.warning("Using random search instead")
        return self._run_random_search(strategy, data, parameter_grid, start_date, end_date)
        
    def _generate_parameter_combinations(self, parameter_grid):
        """
        Generate all combinations of parameters.
        
        Parameters:
        -----------
        parameter_grid : dict
            Grid of parameters
            
        Returns:
        --------
        list
            List of parameter combinations
        """
        import itertools
        
        # Convert continuous parameters to discrete
        discrete_grid = {}
        for param_name, param_values in parameter_grid.items():
            if isinstance(param_values, list):
                # Already discrete
                discrete_grid[param_name] = param_values
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Continuous parameter
                low, high = param_values
                if isinstance(low, int) and isinstance(high, int):
                    # Integer parameter
                    discrete_grid[param_name] = list(range(low, high + 1))
                else:
                    # Float parameter - create a reasonable number of points
                    n_points = 10
                    discrete_grid[param_name] = list(np.linspace(low, high, n_points))
            else:
                # Unknown parameter type
                logger.warning(f"Unknown parameter type for {param_name}: {param_values}")
                discrete_grid[param_name] = [param_values]
                
        # Generate all combinations
        param_names = list(discrete_grid.keys())
        param_values = list(discrete_grid.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = {name: value for name, value in zip(param_names, combo)}
            combinations.append(param_dict)
            
        return combinations
        
    def _save_optimization_metadata(self, optimization_id, method, parameter_grid, task_ids, start_date=None, end_date=None):
        """
        Save optimization metadata.
        
        Parameters:
        -----------
        optimization_id : str
            Optimization ID
        method : str
            Optimization method
        parameter_grid : dict
            Parameter grid
        task_ids : list
            List of task IDs
        start_date : datetime, optional
            Start date for the backtest
        end_date : datetime, optional
            End date for the backtest
        """
        metadata = {
            "optimization_id": optimization_id,
            "method": method,
            "parameter_grid": parameter_grid,
            "task_ids": task_ids,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to file
        metadata_path = os.path.join(self.config["result_dir"], f"{optimization_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Saved optimization metadata for {optimization_id}")
        
    def get_optimization_results(self, optimization_id):
        """
        Get optimization results.
        
        Parameters:
        -----------
        optimization_id : str
            Optimization ID
            
        Returns:
        --------
        dict
            Optimization results
        """
        # Load optimization metadata
        metadata_path = os.path.join(self.config["result_dir"], f"{optimization_id}_metadata.json")
        if not os.path.exists(metadata_path):
            logger.error(f"Optimization metadata not found: {optimization_id}")
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Get task results
        task_results = []
        for task_id in metadata["task_ids"]:
            result = self.coordinator.get_task_result(task_id)
            status = self.coordinator.get_task_status(task_id)
            
            if result is not None:
                # Add task parameters and status
                result_with_params = {
                    "task_id": task_id,
                    "parameters": status.get("parameters", {}),
                    "status": status.get("status"),
                    "result": result
                }
                task_results.append(result_with_params)
                
        # Sort results by performance metric
        if task_results:
            # Determine performance metric
            # This depends on the structure of the result
            # For now, we'll assume there's a 'performance' field with a 'sharpe_ratio' metric
            try:
                task_results.sort(
                    key=lambda x: x["result"]["performance"]["sharpe_ratio"],
                    reverse=True
                )
            except (KeyError, TypeError):
                # If the expected structure isn't found, don't sort
                pass
                
        # Create optimization results
        results = {
            "optimization_id": optimization_id,
            "method": metadata["method"],
            "parameter_grid": metadata["parameter_grid"],
            "start_date": metadata["start_date"],
            "end_date": metadata["end_date"],
            "created_at": metadata["created_at"],
            "completed_tasks": len(task_results),
            "total_tasks": len(metadata["task_ids"]),
            "best_parameters": task_results[0]["parameters"] if task_results else None,
            "best_result": task_results[0]["result"] if task_results else None,
            "all_results": task_results
        }
        
        return results
        
    def run_walk_forward_optimization(self, strategy, data, parameter_grid, start_date=None, end_date=None, windows=None, train_size=None):
        """
        Run walk-forward optimization.
        
        Parameters:
        -----------
        strategy : object
            Strategy object to backtest
        data : dict
            Market data for the backtest
        parameter_grid : dict
            Grid of parameters to optimize
        start_date : datetime, optional
            Start date for the backtest
        end_date : datetime, optional
            End date for the backtest
        windows : int, optional
            Number of windows
        train_size : float, optional
            Size of training set as a fraction of window
            
        Returns:
        --------
        str
            Walk-forward optimization ID
        """
        windows = windows or self.config["walk_forward_windows"]
        train_size = train_size or self.config["walk_forward_train_size"]
        
        # Create walk-forward optimization ID
        wfo_id = f"walk_forward_{uuid.uuid4()}"
        
        # Determine date range
        if start_date is None or end_date is None:
            # Use full data range
            all_dates = []
            for ticker, ticker_data in data.items():
                if isinstance(ticker_data, pd.DataFrame) and not ticker_data.empty:
                    if ticker_data.index.dtype == 'datetime64[ns]':
                        all_dates.extend(ticker_data.index.tolist())
                    elif 'date' in ticker_data.columns:
                        all_dates.extend(ticker_data['date'].tolist())
                        
            if all_dates:
                if start_date is None:
                    start_date = min(all_dates)
                if end_date is None:
                    end_date = max(all_dates)
            else:
                logger.error("Could not determine date range from data")
                return None
                
        # Calculate window size
        total_days = (end_date - start_date).days
        window_days = total_days // windows
        
        # Create windows
        window_starts = []
        window_ends = []
        
        for i in range(windows):
            window_start = start_date + timedelta(days=i * window_days)
            window_end = start_date + timedelta(days=(i + 1) * window_days)
            
            if i == windows - 1:
                # Make sure the last window includes the end date
                window_end = end_date
                
            window_starts.append(window_start)
            window_ends.append(window_end)
            
        # Run optimization for each window
        window_optimizations = []
        
        for i in range(windows):
            window_start = window_starts[i]
            window_end = window_ends[i]
            
            # Calculate train/test split
            train_days = int(window_days * train_size)
            train_end = window_start + timedelta(days=train_days)
            test_start = train_end
            
            # Run optimization on training set
            optimization_id = self._run_grid_search(
                strategy=strategy,
                data=data,
                parameter_grid=parameter_grid,
                start_date=window_start,
                end_date=train_end
            )
            
            # Create test task with best parameters
            # (this will be filled in later when optimization completes)
            test_task = BacktestTask(
                strategy=strategy,
                data=data,
                parameters={"optimization_id": optimization_id},
                start_date=test_start,
                end_date=window_end
            )
            
            # Add metadata to task
            test_task.parameters["wfo_id"] = wfo_id
            test_task.parameters["window"] = i
            test_task.parameters["window_type"] = "test"
            
            # Add task to coordinator
            test_task_id = self.coordinator.add_task(test_task)
            
            window_optimizations.append({
                "window": i,
                "train_start": window_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": window_end.isoformat(),
                "optimization_id": optimization_id,
                "test_task_id": test_task_id
            })
            
        # Save walk-forward optimization metadata
        metadata = {
            "wfo_id": wfo_id,
            "parameter_grid": parameter_grid,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "windows": windows,
            "train_size": train_size,
            "window_optimizations": window_optimizations,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to file
        metadata_path = os.path.join(self.config["result_dir"], f"{wfo_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Started walk-forward optimization {wfo_id} with {windows} windows")
        
        return wfo_id
        
    def get_walk_forward_results(self, wfo_id):
        """
        Get walk-forward optimization results.
        
        Parameters:
        -----------
        wfo_id : str
            Walk-forward optimization ID
            
        Returns:
        --------
        dict
            Walk-forward optimization results
        """
        # Load walk-forward optimization metadata
        metadata_path = os.path.join(self.config["result_dir"], f"{wfo_id}_metadata.json")
        if not os.path.exists(metadata_path):
            logger.error(f"Walk-forward optimization metadata not found: {wfo_id}")
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Get results for each window
        window_results = []
        
        for window_opt in metadata["window_optimizations"]:
            # Get optimization results
            optimization_id = window_opt["optimization_id"]
            optimization_results = self.get_optimization_results(optimization_id)
            
            # Get test task result
            test_task_id = window_opt["test_task_id"]
            test_result = self.coordinator.get_task_result(test_task_id)
            test_status = self.coordinator.get_task_status(test_task_id)
            
            window_result = {
                "window": window_opt["window"],
                "train_start": window_opt["train_start"],
                "train_end": window_opt["train_end"],
                "test_start": window_opt["test_start"],
                "test_end": window_opt["test_end"],
                "optimization_id": optimization_id,
                "test_task_id": test_task_id,
                "optimization_status": "completed" if optimization_results else "in_progress",
                "test_status": test_status.get("status"),
                "best_parameters": optimization_results.get("best_parameters") if optimization_results else None,
                "train_performance": optimization_results.get("best_result", {}).get("performance") if optimization_results else None,
                "test_performance": test_result.get("performance") if test_result else None
            }
            
            window_results.append(window_result)
            
        # Calculate overall performance
        overall_performance = {}
        
        if all(wr.get("test_performance") for wr in window_results):
            # Calculate average performance across all windows
            metrics = window_results[0].get("test_performance", {}).keys()
            
            for metric in metrics:
                values = [wr["test_performance"][metric] for wr in window_results if metric in wr.get("test_performance", {})]
                if values:
                    overall_performance[metric] = sum(values) / len(values)
                    
        # Create walk-forward results
        results = {
            "wfo_id": wfo_id,
            "parameter_grid": metadata["parameter_grid"],
            "start_date": metadata["start_date"],
            "end_date": metadata["end_date"],
            "windows": metadata["windows"],
            "train_size": metadata["train_size"],
            "created_at": metadata["created_at"],
            "window_results": window_results,
            "overall_performance": overall_performance,
            "status": "completed" if all(wr["test_status"] == "completed" for wr in window_results) else "in_progress"
        }
        
        return results
        
    def run_monte_carlo_simulation(self, backtest_result, iterations=1000, confidence_level=0.95):
        """
        Run Monte Carlo simulation on a backtest result.
        
        Parameters:
        -----------
        backtest_result : dict
            Backtest result
        iterations : int, optional
            Number of iterations
        confidence_level : float, optional
            Confidence level for intervals
            
        Returns:
        --------
        dict
            Monte Carlo simulation results
        """
        # Extract trades from backtest result
        if "trades" not in backtest_result:
            logger.error("Backtest result does not contain trades")
            return None
            
        trades = backtest_result["trades"]
        
        # Extract returns from trades
        returns = []
        for trade in trades:
            if "pnl_percent" in trade:
                returns.append(trade["pnl_percent"] / 100.0)  # Convert percentage to decimal
                
        if not returns:
            logger.error("No returns found in trades")
            return None
            
        # Run Monte Carlo simulation
        simulation_results = self._run_monte_carlo(
            returns=returns,
            initial_capital=100000,
            iterations=iterations,
            confidence_level=confidence_level
        )
        
        return simulation_results
        
    def _run_monte_carlo(self, returns, initial_capital, iterations, confidence_level):
        """
        Run Monte Carlo simulation.
        
        Parameters:
        -----------
        returns : list
            List of trade returns
        initial_capital : float
            Initial capital
        iterations : int
            Number of iterations
        confidence_level : float
            Confidence level for intervals
            
        Returns:
        --------
        dict
            Monte Carlo simulation results
        """
        # Create array to store final equity for each iteration
        final_equities = np.zeros(iterations)
        max_drawdowns = np.zeros(iterations)
        
        # Create array to store equity curves
        equity_curves = np.zeros((iterations, len(returns) + 1))
        equity_curves[:, 0] = initial_capital
        
        # Run simulations
        for i in range(iterations):
            # Shuffle returns
            np.random.shuffle(returns)
            
            # Calculate equity curve
            equity = initial_capital
            equity_curve = [equity]
            peak = equity
            max_drawdown = 0
            
            for r in returns:
                equity *= (1 + r)
                equity_curve.append(equity)
                
                # Update peak and drawdown
                if equity > peak:
                    peak = equity
                else:
                    drawdown = (peak - equity) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                    
            # Store results
            final_equities[i] = equity
            max_drawdowns[i] = max_drawdown
            equity_curves[i, :] = equity_curve
            
        # Calculate statistics
        mean_final_equity = np.mean(final_equities)
        median_final_equity = np.median(final_equities)
        std_final_equity = np.std(final_equities)
        
        # Calculate confidence intervals
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        ci_lower = np.percentile(final_equities, lower_percentile)
        ci_upper = np.percentile(final_equities, upper_percentile)
        
        # Calculate drawdown statistics
        mean_max_drawdown = np.mean(max_drawdowns)
        median_max_drawdown = np.median(max_drawdowns)
        worst_max_drawdown = np.max(max_drawdowns)
        
        # Calculate drawdown confidence interval
        dd_ci_upper = np.percentile(max_drawdowns, upper_percentile)
        
        # Calculate average equity curve and confidence bands
        mean_equity_curve = np.mean(equity_curves, axis=0)
        lower_band = np.percentile(equity_curves, lower_percentile, axis=0)
        upper_band = np.percentile(equity_curves, upper_percentile, axis=0)
        
        # Create results
        results = {
            "initial_capital": initial_capital,
            "iterations": iterations,
            "confidence_level": confidence_level,
            "final_equity": {
                "mean": float(mean_final_equity),
                "median": float(median_final_equity),
                "std": float(std_final_equity),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper)
            },
            "max_drawdown": {
                "mean": float(mean_max_drawdown),
                "median": float(median_max_drawdown),
                "worst": float(worst_max_drawdown),
                "ci_upper": float(dd_ci_upper)
            },
            "equity_curve": {
                "mean": mean_equity_curve.tolist(),
                "lower_band": lower_band.tolist(),
                "upper_band": upper_band.tolist()
            },
            "return": {
                "mean": float((mean_final_equity / initial_capital) - 1),
                "median": float((median_final_equity / initial_capital) - 1),
                "ci_lower": float((ci_lower / initial_capital) - 1),
                "ci_upper": float((ci_upper / initial_capital) - 1)
            }
        }
        
        return results
