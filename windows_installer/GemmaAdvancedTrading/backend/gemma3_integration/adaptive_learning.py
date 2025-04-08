"""
Adaptive Learning Module for Gemma Advanced Trading System

This module implements adaptive learning capabilities using Gemma 3
to learn from past trades and market conditions to continuously improve
strategies and recommendations.
"""

import os
import logging
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import pickle
from collections import defaultdict

# Import Gemma 3 integration architecture
from gemma3_integration.architecture_enhanced import GemmaCore, PromptEngine, ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class TradeMemory:
    """
    Stores and manages historical trade data for adaptive learning.
    
    This class provides methods for storing, retrieving, and analyzing
    historical trade data to support adaptive learning.
    """
    
    def __init__(self, storage_path: str = "data/trade_memory"):
        """
        Initialize the TradeMemory.
        
        Parameters:
        -----------
        storage_path : str, optional
            Path to store trade memory data. Default is "data/trade_memory".
        """
        self.logger = logging.getLogger("GemmaTrading.TradeMemory")
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize memory structures
        self.trades = []
        self.trade_outcomes = {}
        self.strategy_performance = defaultdict(list)
        self.market_condition_performance = defaultdict(list)
        
        # Load existing data if available
        self._load_memory()
        
        self.logger.info("Initialized TradeMemory")
    
    def _load_memory(self):
        """Load memory data from storage."""
        try:
            # Load trades
            trades_path = os.path.join(self.storage_path, "trades.pkl")
            if os.path.exists(trades_path):
                with open(trades_path, "rb") as f:
                    self.trades = pickle.load(f)
            
            # Load trade outcomes
            outcomes_path = os.path.join(self.storage_path, "trade_outcomes.pkl")
            if os.path.exists(outcomes_path):
                with open(outcomes_path, "rb") as f:
                    self.trade_outcomes = pickle.load(f)
            
            # Load strategy performance
            strategy_path = os.path.join(self.storage_path, "strategy_performance.pkl")
            if os.path.exists(strategy_path):
                with open(strategy_path, "rb") as f:
                    self.strategy_performance = pickle.load(f)
            
            # Load market condition performance
            market_path = os.path.join(self.storage_path, "market_condition_performance.pkl")
            if os.path.exists(market_path):
                with open(market_path, "rb") as f:
                    self.market_condition_performance = pickle.load(f)
            
            self.logger.info(f"Loaded {len(self.trades)} trades from memory")
        except Exception as e:
            self.logger.error(f"Error loading memory: {e}")
    
    def _save_memory(self):
        """Save memory data to storage."""
        try:
            # Save trades
            trades_path = os.path.join(self.storage_path, "trades.pkl")
            with open(trades_path, "wb") as f:
                pickle.dump(self.trades, f)
            
            # Save trade outcomes
            outcomes_path = os.path.join(self.storage_path, "trade_outcomes.pkl")
            with open(outcomes_path, "wb") as f:
                pickle.dump(self.trade_outcomes, f)
            
            # Save strategy performance
            strategy_path = os.path.join(self.storage_path, "strategy_performance.pkl")
            with open(strategy_path, "wb") as f:
                pickle.dump(self.strategy_performance, f)
            
            # Save market condition performance
            market_path = os.path.join(self.storage_path, "market_condition_performance.pkl")
            with open(market_path, "wb") as f:
                pickle.dump(self.market_condition_performance, f)
            
            self.logger.info("Saved memory to storage")
        except Exception as e:
            self.logger.error(f"Error saving memory: {e}")
    
    def add_trade(self, trade_id: str, trade_data: Dict[str, Any]) -> bool:
        """
        Add a new trade to memory.
        
        Parameters:
        -----------
        trade_id : str
            Unique identifier for the trade.
        trade_data : Dict[str, Any]
            Trade data to store.
            
        Returns:
        --------
        bool
            True if the trade was added successfully, False otherwise.
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in trade_data:
                trade_data["timestamp"] = datetime.datetime.now().isoformat()
            
            # Add trade to memory
            self.trades.append({
                "trade_id": trade_id,
                "data": trade_data
            })
            
            # Save memory
            self._save_memory()
            
            self.logger.info(f"Added trade {trade_id} to memory")
            return True
        except Exception as e:
            self.logger.error(f"Error adding trade {trade_id} to memory: {e}")
            return False
    
    def add_trade_outcome(self, trade_id: str, outcome_data: Dict[str, Any]) -> bool:
        """
        Add outcome data for a trade.
        
        Parameters:
        -----------
        trade_id : str
            Unique identifier for the trade.
        outcome_data : Dict[str, Any]
            Trade outcome data to store.
            
        Returns:
        --------
        bool
            True if the outcome was added successfully, False otherwise.
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in outcome_data:
                outcome_data["timestamp"] = datetime.datetime.now().isoformat()
            
            # Add outcome to memory
            self.trade_outcomes[trade_id] = outcome_data
            
            # Update strategy performance if strategy is present
            trade_data = self._get_trade_data(trade_id)
            if trade_data and "strategy" in trade_data:
                strategy = trade_data["strategy"]
                return_pct = outcome_data.get("return_pct", 0)
                
                self.strategy_performance[strategy].append({
                    "trade_id": trade_id,
                    "return_pct": return_pct,
                    "timestamp": outcome_data["timestamp"]
                })
            
            # Update market condition performance if market condition is present
            if trade_data and "market_condition" in trade_data:
                market_condition = trade_data["market_condition"]
                return_pct = outcome_data.get("return_pct", 0)
                
                self.market_condition_performance[market_condition].append({
                    "trade_id": trade_id,
                    "return_pct": return_pct,
                    "timestamp": outcome_data["timestamp"]
                })
            
            # Save memory
            self._save_memory()
            
            self.logger.info(f"Added outcome for trade {trade_id} to memory")
            return True
        except Exception as e:
            self.logger.error(f"Error adding outcome for trade {trade_id} to memory: {e}")
            return False
    
    def _get_trade_data(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get trade data for a specific trade ID."""
        for trade in self.trades:
            if trade["trade_id"] == trade_id:
                return trade["data"]
        return None
    
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trade data and outcome for a specific trade.
        
        Parameters:
        -----------
        trade_id : str
            Unique identifier for the trade.
            
        Returns:
        --------
        Optional[Dict[str, Any]]
            Trade data and outcome, or None if not found.
        """
        trade_data = self._get_trade_data(trade_id)
        if not trade_data:
            return None
        
        outcome_data = self.trade_outcomes.get(trade_id, {})
        
        return {
            "trade_id": trade_id,
            "trade_data": trade_data,
            "outcome_data": outcome_data
        }
    
    def get_trades_by_strategy(self, strategy: str) -> List[Dict[str, Any]]:
        """
        Get all trades for a specific strategy.
        
        Parameters:
        -----------
        strategy : str
            Strategy name.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of trades for the strategy.
        """
        strategy_trades = []
        
        for trade in self.trades:
            trade_data = trade["data"]
            if trade_data.get("strategy") == strategy:
                trade_id = trade["trade_id"]
                outcome_data = self.trade_outcomes.get(trade_id, {})
                
                strategy_trades.append({
                    "trade_id": trade_id,
                    "trade_data": trade_data,
                    "outcome_data": outcome_data
                })
        
        return strategy_trades
    
    def get_trades_by_market_condition(self, market_condition: str) -> List[Dict[str, Any]]:
        """
        Get all trades for a specific market condition.
        
        Parameters:
        -----------
        market_condition : str
            Market condition name.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of trades for the market condition.
        """
        condition_trades = []
        
        for trade in self.trades:
            trade_data = trade["data"]
            if trade_data.get("market_condition") == market_condition:
                trade_id = trade["trade_id"]
                outcome_data = self.trade_outcomes.get(trade_id, {})
                
                condition_trades.append({
                    "trade_id": trade_id,
                    "trade_data": trade_data,
                    "outcome_data": outcome_data
                })
        
        return condition_trades
    
    def get_strategy_performance(self, strategy: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific strategy.
        
        Parameters:
        -----------
        strategy : str
            Strategy name.
            
        Returns:
        --------
        Dict[str, Any]
            Performance metrics for the strategy.
        """
        performance_data = self.strategy_performance.get(strategy, [])
        
        if not performance_data:
            return {
                "strategy": strategy,
                "trade_count": 0,
                "avg_return": 0,
                "win_rate": 0,
                "max_return": 0,
                "min_return": 0
            }
        
        returns = [p["return_pct"] for p in performance_data]
        win_count = sum(1 for r in returns if r > 0)
        
        return {
            "strategy": strategy,
            "trade_count": len(performance_data),
            "avg_return": np.mean(returns),
            "win_rate": win_count / len(performance_data) if performance_data else 0,
            "max_return": max(returns) if returns else 0,
            "min_return": min(returns) if returns else 0
        }
    
    def get_market_condition_performance(self, market_condition: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific market condition.
        
        Parameters:
        -----------
        market_condition : str
            Market condition name.
            
        Returns:
        --------
        Dict[str, Any]
            Performance metrics for the market condition.
        """
        performance_data = self.market_condition_performance.get(market_condition, [])
        
        if not performance_data:
            return {
                "market_condition": market_condition,
                "trade_count": 0,
                "avg_return": 0,
                "win_rate": 0,
                "max_return": 0,
                "min_return": 0
            }
        
        returns = [p["return_pct"] for p in performance_data]
        win_count = sum(1 for r in returns if r > 0)
        
        return {
            "market_condition": market_condition,
            "trade_count": len(performance_data),
            "avg_return": np.mean(returns),
            "win_rate": win_count / len(performance_data) if performance_data else 0,
            "max_return": max(returns) if returns else 0,
            "min_return": min(returns) if returns else 0
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent trades.
        
        Parameters:
        -----------
        limit : int, optional
            Maximum number of trades to return. Default is 10.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of recent trades.
        """
        recent_trades = []
        
        # Sort trades by timestamp (newest first)
        sorted_trades = sorted(
            self.trades,
            key=lambda t: t["data"].get("timestamp", ""),
            reverse=True
        )
        
        # Get the most recent trades
        for trade in sorted_trades[:limit]:
            trade_id = trade["trade_id"]
            trade_data = trade["data"]
            outcome_data = self.trade_outcomes.get(trade_id, {})
            
            recent_trades.append({
                "trade_id": trade_id,
                "trade_data": trade_data,
                "outcome_data": outcome_data
            })
        
        return recent_trades
    
    def get_best_performing_strategies(self, min_trades: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best performing strategies.
        
        Parameters:
        -----------
        min_trades : int, optional
            Minimum number of trades required for a strategy to be considered.
            Default is 5.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of strategies sorted by performance.
        """
        strategy_metrics = []
        
        for strategy, performance_data in self.strategy_performance.items():
            if len(performance_data) >= min_trades:
                metrics = self.get_strategy_performance(strategy)
                strategy_metrics.append(metrics)
        
        # Sort by average return (highest first)
        return sorted(strategy_metrics, key=lambda m: m["avg_return"], reverse=True)
    
    def get_best_performing_market_conditions(self, min_trades: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best performing market conditions.
        
        Parameters:
        -----------
        min_trades : int, optional
            Minimum number of trades required for a market condition to be considered.
            Default is 5.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of market conditions sorted by performance.
        """
        condition_metrics = []
        
        for condition, performance_data in self.market_condition_performance.items():
            if len(performance_data) >= min_trades:
                metrics = self.get_market_condition_performance(condition)
                condition_metrics.append(metrics)
        
        # Sort by average return (highest first)
        return sorted(condition_metrics, key=lambda m: m["avg_return"], reverse=True)

class StrategyOptimizer:
    """
    Optimizes trading strategies based on historical performance.
    
    This class provides methods for optimizing trading strategies based on
    historical performance data and current market conditions.
    """
    
    def __init__(self, trade_memory: Optional[TradeMemory] = None, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the StrategyOptimizer.
        
        Parameters:
        -----------
        trade_memory : TradeMemory, optional
            Instance of TradeMemory for accessing historical trade data.
            If None, creates a new instance.
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyOptimizer")
        self.trade_memory = trade_memory or TradeMemory()
        self.gemma_core = gemma_core or GemmaCore()
        
        self.logger.info("Initialized StrategyOptimizer")
    
    def optimize_strategy(self, strategy: Dict[str, Any], 
                        market_conditions: Dict[str, Any],
                        optimization_goals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize a trading strategy based on historical performance and current market conditions.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to optimize.
        market_conditions : Dict[str, Any]
            Current market conditions.
        optimization_goals : Dict[str, Any], optional
            Goals for the optimization (e.g., maximize return, minimize risk).
            
        Returns:
        --------
        Dict[str, Any]
            Optimized trading strategy.
        """
        self.logger.info(f"Optimizing strategy {strategy.get('name', 'unknown')}")
        
        # Get strategy performance
        strategy_name = strategy.get("name", "unknown")
        strategy_performance = self.trade_memory.get_strategy_performance(strategy_name)
        
        # Get market condition performance
        market_condition = market_conditions.get("regime", "unknown")
        market_performance = self.trade_memory.get_market_condition_performance(market_condition)
        
        # Get recent trades for the strategy
        strategy_trades = self.trade_memory.get_trades_by_strategy(strategy_name)
        
        # Prepare context for Gemma 3
        context = {
            "strategy": strategy,
            "strategy_performance": strategy_performance,
            "market_conditions": market_conditions,
            "market_performance": market_performance,
            "recent_trades": strategy_trades[:10]  # Limit to 10 recent trades
        }
        
        if optimization_goals:
            context["optimization_goals"] = optimization_goals
        
        # Generate prompt for strategy optimization
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_optimization",
            **context
        )
        
        # Get the appropriate model for strategy optimization
        model = self.gemma_core.model_manager.get_model("strategy_optimization")
        
        # Generate optimization using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract optimized strategy from chain-of-thought result
        # In a real implementation, this would parse the result to extract a structured strategy
        # For this implementation, we'll create a simulated optimized strategy
        
        # Start with a copy of the original strategy
        optimized_strategy = strategy.copy()
        
        # Simulate parameter adjustments based on performance
        if "parameters" in optimized_strategy:
            parameters = optimized_strategy["parameters"]
            
            # Adjust parameters based on strategy performance
            if strategy_performance["avg_return"] < 0:
                # If strategy is losing money, make more conservative
                for param_name, param_value in parameters.items():
                    if isinstance(param_value, (int, float)):
                        if "threshold" in param_name.lower():
                            # Increase thresholds by 10%
                            parameters[param_name] = param_value * 1.1
                        elif "period" in param_name.lower():
                            # Increase periods by 20%
                            parameters[param_name] = int(param_value * 1.2)
            else:
                # If strategy is profitable, fine-tune
                for param_name, param_value in parameters.items():
                    if isinstance(param_value, (int, float)):
                        if "threshold" in param_name.lower():
                            # Adjust thresholds by 5%
                            parameters[param_name] = param_value * 1.05
        
        # Simulate risk management adjustments based on market conditions
        if "risk_management" in optimized_strategy:
            risk_management = optimized_strategy["risk_management"]
            
            # Adjust risk management based on market volatility
            market_volatility = market_conditions.get("volatility", "moderate")
            
            if market_volatility == "high":
                # If market volatility is high, tighten risk management
                if "stop_loss_pct" in risk_management:
                    risk_management["stop_loss_pct"] *= 0.8  # Tighter stop loss
                if "position_size_pct" in risk_management:
                    risk_management["position_size_pct"] *= 0.8  # Smaller position size
            elif market_volatility == "low":
                # If market volatility is low, relax risk management
                if "stop_loss_pct" in risk_management:
                    risk_management["stop_loss_pct"] *= 1.2  # Wider stop loss
                if "position_size_pct" in risk_management:
                    risk_management["position_size_pct"] *= 1.2  # Larger position size
        
        # Add optimization metadata
        optimized_strategy["optimization"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "original_strategy": strategy_name,
            "market_condition": market_condition,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", "")
        }
        
        self.logger.info(f"Optimized strategy {strategy_name}")
        return optimized_strategy
    
    def recommend_strategy_adjustments(self, strategy: Dict[str, Any], 
                                     performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend adjustments to a trading strategy based on performance data.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to adjust.
        performance_data : Dict[str, Any]
            Performance data for the strategy.
            
        Returns:
        --------
        Dict[str, Any]
            Recommended strategy adjustments.
        """
        self.logger.info(f"Recommending adjustments for strategy {strategy.get('name', 'unknown')}")
        
        # Prepare context for Gemma 3
        context = {
            "strategy": strategy,
            "performance_data": performance_data
        }
        
        # Generate prompt for strategy adjustment
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_adjustment",
            **context
        )
        
        # Get the appropriate model for strategy adjustment
        model = self.gemma_core.model_manager.get_model("strategy_optimization")
        
        # Generate adjustments using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract adjustments from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured adjustments
        # For this implementation, we'll create simulated adjustments
        
        # Simulate parameter adjustments
        parameter_adjustments = {}
        
        if "parameters" in strategy:
            for param_name, param_value in strategy["parameters"].items():
                if isinstance(param_value, (int, float)):
                    # Simulate a small adjustment
                    adjustment_pct = np.random.uniform(-0.1, 0.1)
                    new_value = param_value * (1 + adjustment_pct)
                    
                    # Round to appropriate precision
                    if isinstance(param_value, int):
                        new_value = int(new_value)
                    else:
                        new_value = round(new_value, 2)
                    
                    parameter_adjustments[param_name] = {
                        "current_value": param_value,
                        "recommended_value": new_value,
                        "adjustment_pct": adjustment_pct * 100
                    }
        
        # Simulate risk management adjustments
        risk_adjustments = {}
        
        if "risk_management" in strategy:
            for param_name, param_value in strategy["risk_management"].items():
                if isinstance(param_value, (int, float)):
                    # Simulate a small adjustment
                    adjustment_pct = np.random.uniform(-0.15, 0.15)
                    new_value = param_value * (1 + adjustment_pct)
                    
                    # Round to appropriate precision
                    if isinstance(param_value, int):
                        new_value = int(new_value)
                    else:
                        new_value = round(new_value, 2)
                    
                    risk_adjustments[param_name] = {
                        "current_value": param_value,
                        "recommended_value": new_value,
                        "adjustment_pct": adjustment_pct * 100
                    }
        
        # Simulate entry/exit condition adjustments
        condition_adjustments = []
        
        if "entry_conditions" in strategy:
            condition_adjustments.append({
                "type": "entry_condition",
                "current": strategy["entry_conditions"][0] if strategy["entry_conditions"] else "None",
                "recommended": "Modified " + (strategy["entry_conditions"][0] if strategy["entry_conditions"] else "condition"),
                "reason": "Improve entry timing based on performance data"
            })
        
        if "exit_conditions" in strategy:
            condition_adjustments.append({
                "type": "exit_condition",
                "current": strategy["exit_conditions"][0] if strategy["exit_conditions"] else "None",
                "recommended": "Modified " + (strategy["exit_conditions"][0] if strategy["exit_conditions"] else "condition"),
                "reason": "Improve exit timing based on performance data"
            })
        
        # Create adjustment recommendations
        adjustments = {
            "strategy_name": strategy.get("name", "unknown"),
            "parameter_adjustments": parameter_adjustments,
            "risk_adjustments": risk_adjustments,
            "condition_adjustments": condition_adjustments,
            "overall_recommendation": "Fine-tune parameters and risk management based on recent performance",
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated adjustment recommendations for strategy {strategy.get('name', 'unknown')}")
        return adjustments

class PerformanceAnalyzer:
    """
    Analyzes trading performance and identifies patterns and insights.
    
    This class provides methods for analyzing trading performance and identifying
    patterns and insights to improve future trading decisions.
    """
    
    def __init__(self, trade_memory: Optional[TradeMemory] = None, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the PerformanceAnalyzer.
        
        Parameters:
        -----------
        trade_memory : TradeMemory, optional
            Instance of TradeMemory for accessing historical trade data.
            If None, creates a new instance.
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.PerformanceAnalyzer")
        self.trade_memory = trade_memory or TradeMemory()
        self.gemma_core = gemma_core or GemmaCore()
        
        self.logger.info("Initialized PerformanceAnalyzer")
    
    def analyze_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """
        Analyze the performance of a specific strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Performance analysis for the strategy.
        """
        self.logger.info(f"Analyzing performance for strategy {strategy_name}")
        
        # Get strategy trades
        strategy_trades = self.trade_memory.get_trades_by_strategy(strategy_name)
        
        # Get strategy performance metrics
        performance_metrics = self.trade_memory.get_strategy_performance(strategy_name)
        
        # Prepare context for Gemma 3
        context = {
            "strategy_name": strategy_name,
            "trades": strategy_trades,
            "performance_metrics": performance_metrics
        }
        
        # Generate prompt for strategy performance analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_performance_analysis",
            **context
        )
        
        # Get the appropriate model for performance analysis
        model = self.gemma_core.model_manager.get_model("performance_analysis")
        
        # Generate analysis using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract analysis from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured analysis
        # For this implementation, we'll create a simulated analysis
        
        # Calculate performance metrics
        trade_count = len(strategy_trades)
        
        if trade_count == 0:
            return {
                "strategy_name": strategy_name,
                "error": "No trades found for this strategy"
            }
        
        # Extract returns from trades
        returns = []
        holding_periods = []
        win_trades = []
        loss_trades = []
        
        for trade in strategy_trades:
            outcome = trade.get("outcome_data", {})
            return_pct = outcome.get("return_pct", 0)
            holding_period = outcome.get("holding_period", 0)
            
            returns.append(return_pct)
            holding_periods.append(holding_period)
            
            if return_pct > 0:
                win_trades.append(trade)
            else:
                loss_trades.append(trade)
        
        # Calculate metrics
        avg_return = np.mean(returns) if returns else 0
        median_return = np.median(returns) if returns else 0
        win_rate = len(win_trades) / trade_count if trade_count > 0 else 0
        avg_win = np.mean([t.get("outcome_data", {}).get("return_pct", 0) for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t.get("outcome_data", {}).get("return_pct", 0) for t in loss_trades]) if loss_trades else 0
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Calculate profit factor
        total_wins = sum([t.get("outcome_data", {}).get("return_pct", 0) for t in win_trades]) if win_trades else 0
        total_losses = abs(sum([t.get("outcome_data", {}).get("return_pct", 0) for t in loss_trades])) if loss_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Simulate strengths and weaknesses
        strengths = [
            "Consistent performance across different market conditions",
            "Good risk-reward ratio",
            "Effective use of technical indicators"
        ]
        
        weaknesses = [
            "Suboptimal exit timing",
            "Inconsistent position sizing",
            "Vulnerability to sudden market shifts"
        ]
        
        # Simulate improvement suggestions
        improvement_suggestions = [
            "Refine exit criteria to capture more profit",
            "Implement more dynamic position sizing based on volatility",
            "Add additional filters to avoid false signals"
        ]
        
        # Create analysis result
        analysis = {
            "strategy_name": strategy_name,
            "trade_count": trade_count,
            "performance_metrics": {
                "avg_return": avg_return,
                "median_return": median_return,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "avg_holding_period": avg_holding_period
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvement_suggestions": improvement_suggestions,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed performance analysis for strategy {strategy_name}")
        return analysis
    
    def analyze_market_condition_performance(self, market_condition: str) -> Dict[str, Any]:
        """
        Analyze performance under a specific market condition.
        
        Parameters:
        -----------
        market_condition : str
            Market condition to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Performance analysis for the market condition.
        """
        self.logger.info(f"Analyzing performance for market condition {market_condition}")
        
        # Get trades for the market condition
        condition_trades = self.trade_memory.get_trades_by_market_condition(market_condition)
        
        # Get market condition performance metrics
        performance_metrics = self.trade_memory.get_market_condition_performance(market_condition)
        
        # Prepare context for Gemma 3
        context = {
            "market_condition": market_condition,
            "trades": condition_trades,
            "performance_metrics": performance_metrics
        }
        
        # Generate prompt for market condition performance analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "market_condition_analysis",
            **context
        )
        
        # Get the appropriate model for performance analysis
        model = self.gemma_core.model_manager.get_model("performance_analysis")
        
        # Generate analysis using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract analysis from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured analysis
        # For this implementation, we'll create a simulated analysis
        
        # Calculate performance metrics by strategy
        strategy_performance = {}
        
        for trade in condition_trades:
            trade_data = trade.get("trade_data", {})
            outcome_data = trade.get("outcome_data", {})
            
            strategy = trade_data.get("strategy", "unknown")
            return_pct = outcome_data.get("return_pct", 0)
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    "returns": [],
                    "trade_count": 0
                }
            
            strategy_performance[strategy]["returns"].append(return_pct)
            strategy_performance[strategy]["trade_count"] += 1
        
        # Calculate average return and win rate for each strategy
        for strategy, data in strategy_performance.items():
            returns = data["returns"]
            win_count = sum(1 for r in returns if r > 0)
            
            data["avg_return"] = np.mean(returns) if returns else 0
            data["win_rate"] = win_count / len(returns) if returns else 0
        
        # Sort strategies by average return
        best_strategies = sorted(
            strategy_performance.items(),
            key=lambda x: x[1]["avg_return"],
            reverse=True
        )
        
        # Simulate best practices
        best_practices = [
            f"Use {best_strategies[0][0]} strategy when market is in {market_condition} condition",
            "Adjust position sizing to be more conservative",
            "Focus on shorter-term trades with tighter stop losses"
        ] if best_strategies else [
            "Insufficient data to determine best practices"
        ]
        
        # Simulate challenges
        challenges = [
            f"High volatility during {market_condition} conditions",
            "Difficulty in determining precise entry points",
            "Increased likelihood of false breakouts"
        ]
        
        # Create analysis result
        analysis = {
            "market_condition": market_condition,
            "trade_count": len(condition_trades),
            "performance_metrics": performance_metrics,
            "strategy_performance": {s: d for s, d in strategy_performance.items()},
            "best_strategies": [s for s, _ in best_strategies[:3]] if best_strategies else [],
            "best_practices": best_practices,
            "challenges": challenges,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed performance analysis for market condition {market_condition}")
        return analysis
    
    def identify_performance_patterns(self, lookback_period: int = 90) -> Dict[str, Any]:
        """
        Identify patterns in trading performance.
        
        Parameters:
        -----------
        lookback_period : int, optional
            Number of days to look back for pattern identification. Default is 90.
            
        Returns:
        --------
        Dict[str, Any]
            Identified performance patterns.
        """
        self.logger.info(f"Identifying performance patterns with lookback period {lookback_period}")
        
        # Get recent trades
        recent_trades = self.trade_memory.get_recent_trades(100)  # Get more trades than needed for filtering
        
        # Filter trades by lookback period
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=lookback_period)
        cutoff_date_str = cutoff_date.isoformat()
        
        filtered_trades = []
        for trade in recent_trades:
            trade_date_str = trade.get("trade_data", {}).get("timestamp", "")
            if trade_date_str and trade_date_str >= cutoff_date_str:
                filtered_trades.append(trade)
        
        # Prepare context for Gemma 3
        context = {
            "trades": filtered_trades,
            "lookback_period": lookback_period
        }
        
        # Generate prompt for pattern identification
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "performance_pattern_identification",
            **context
        )
        
        # Get the appropriate model for pattern identification
        model = self.gemma_core.model_manager.get_model("pattern_recognition")
        
        # Generate patterns using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract patterns from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured patterns
        # For this implementation, we'll create simulated patterns
        
        # Simulate time-based patterns
        time_patterns = [
            {
                "pattern": "Day of Week Effect",
                "description": "Trades entered on Mondays have higher success rate",
                "confidence": 0.75,
                "supporting_evidence": "8 out of 10 Monday trades were profitable"
            },
            {
                "pattern": "Time of Day Effect",
                "description": "Trades entered in the first hour of trading have lower success rate",
                "confidence": 0.65,
                "supporting_evidence": "Only 3 out of 12 early morning trades were profitable"
            }
        ]
        
        # Simulate market condition patterns
        market_patterns = [
            {
                "pattern": "Volatility Correlation",
                "description": "Performance improves during periods of higher volatility",
                "confidence": 0.8,
                "supporting_evidence": "Average return of 2.5% during high volatility vs 1.2% during low volatility"
            },
            {
                "pattern": "Trend Strength Correlation",
                "description": "Strategies perform better in strong trend environments",
                "confidence": 0.7,
                "supporting_evidence": "Win rate of 68% in strong trends vs 42% in weak trends"
            }
        ]
        
        # Simulate strategy-specific patterns
        strategy_patterns = [
            {
                "pattern": "Strategy Rotation",
                "description": "Different strategies perform optimally in different market phases",
                "confidence": 0.85,
                "supporting_evidence": "Momentum strategies outperform in bullish phases, mean reversion in sideways markets"
            },
            {
                "pattern": "Parameter Sensitivity",
                "description": "Some strategies show high sensitivity to parameter changes",
                "confidence": 0.6,
                "supporting_evidence": "10% change in lookback period resulted in 30% performance difference"
            }
        ]
        
        # Create pattern identification result
        patterns = {
            "time_patterns": time_patterns,
            "market_patterns": market_patterns,
            "strategy_patterns": strategy_patterns,
            "key_insights": [
                "Market condition is the strongest predictor of strategy performance",
                "Proper strategy selection based on market regime is critical",
                "Risk management parameters should be adjusted based on volatility"
            ],
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info("Completed performance pattern identification")
        return patterns

class AdaptiveLearning:
    """
    Main class for adaptive learning using Gemma 3.
    
    This class provides a unified interface for all adaptive learning capabilities,
    including trade memory, strategy optimization, and performance analysis.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the AdaptiveLearning.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.AdaptiveLearning")
        
        # Create or use provided GemmaCore
        self.gemma_core = gemma_core or GemmaCore()
        
        # Initialize components
        self.trade_memory = TradeMemory()
        self.strategy_optimizer = StrategyOptimizer(self.trade_memory, self.gemma_core)
        self.performance_analyzer = PerformanceAnalyzer(self.trade_memory, self.gemma_core)
        
        self.logger.info("Initialized AdaptiveLearning")
    
    def record_trade(self, trade_id: str, trade_data: Dict[str, Any]) -> bool:
        """
        Record a new trade in memory.
        
        Parameters:
        -----------
        trade_id : str
            Unique identifier for the trade.
        trade_data : Dict[str, Any]
            Trade data to store.
            
        Returns:
        --------
        bool
            True if the trade was recorded successfully, False otherwise.
        """
        return self.trade_memory.add_trade(trade_id, trade_data)
    
    def record_trade_outcome(self, trade_id: str, outcome_data: Dict[str, Any]) -> bool:
        """
        Record the outcome of a trade.
        
        Parameters:
        -----------
        trade_id : str
            Unique identifier for the trade.
        outcome_data : Dict[str, Any]
            Trade outcome data to store.
            
        Returns:
        --------
        bool
            True if the outcome was recorded successfully, False otherwise.
        """
        return self.trade_memory.add_trade_outcome(trade_id, outcome_data)
    
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trade data and outcome for a specific trade.
        
        Parameters:
        -----------
        trade_id : str
            Unique identifier for the trade.
            
        Returns:
        --------
        Optional[Dict[str, Any]]
            Trade data and outcome, or None if not found.
        """
        return self.trade_memory.get_trade(trade_id)
    
    def optimize_strategy(self, strategy: Dict[str, Any], 
                        market_conditions: Dict[str, Any],
                        optimization_goals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize a trading strategy based on historical performance and current market conditions.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to optimize.
        market_conditions : Dict[str, Any]
            Current market conditions.
        optimization_goals : Dict[str, Any], optional
            Goals for the optimization (e.g., maximize return, minimize risk).
            
        Returns:
        --------
        Dict[str, Any]
            Optimized trading strategy.
        """
        return self.strategy_optimizer.optimize_strategy(
            strategy, market_conditions, optimization_goals
        )
    
    def recommend_strategy_adjustments(self, strategy: Dict[str, Any], 
                                     performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend adjustments to a trading strategy based on performance data.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to adjust.
        performance_data : Dict[str, Any]
            Performance data for the strategy.
            
        Returns:
        --------
        Dict[str, Any]
            Recommended strategy adjustments.
        """
        return self.strategy_optimizer.recommend_strategy_adjustments(
            strategy, performance_data
        )
    
    def analyze_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """
        Analyze the performance of a specific strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Performance analysis for the strategy.
        """
        return self.performance_analyzer.analyze_strategy_performance(strategy_name)
    
    def analyze_market_condition_performance(self, market_condition: str) -> Dict[str, Any]:
        """
        Analyze performance under a specific market condition.
        
        Parameters:
        -----------
        market_condition : str
            Market condition to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Performance analysis for the market condition.
        """
        return self.performance_analyzer.analyze_market_condition_performance(market_condition)
    
    def identify_performance_patterns(self, lookback_period: int = 90) -> Dict[str, Any]:
        """
        Identify patterns in trading performance.
        
        Parameters:
        -----------
        lookback_period : int, optional
            Number of days to look back for pattern identification. Default is 90.
            
        Returns:
        --------
        Dict[str, Any]
            Identified performance patterns.
        """
        return self.performance_analyzer.identify_performance_patterns(lookback_period)
    
    def get_best_performing_strategies(self, min_trades: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best performing strategies.
        
        Parameters:
        -----------
        min_trades : int, optional
            Minimum number of trades required for a strategy to be considered.
            Default is 5.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of strategies sorted by performance.
        """
        return self.trade_memory.get_best_performing_strategies(min_trades)
    
    def get_best_performing_market_conditions(self, min_trades: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best performing market conditions.
        
        Parameters:
        -----------
        min_trades : int, optional
            Minimum number of trades required for a market condition to be considered.
            Default is 5.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of market conditions sorted by performance.
        """
        return self.trade_memory.get_best_performing_market_conditions(min_trades)
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent trades.
        
        Parameters:
        -----------
        limit : int, optional
            Maximum number of trades to return. Default is 10.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of recent trades.
        """
        return self.trade_memory.get_recent_trades(limit)
    
    def learn_from_trade(self, trade_id: str) -> Dict[str, Any]:
        """
        Extract learning insights from a specific trade.
        
        Parameters:
        -----------
        trade_id : str
            Unique identifier for the trade.
            
        Returns:
        --------
        Dict[str, Any]
            Learning insights from the trade.
        """
        self.logger.info(f"Extracting learning insights from trade {trade_id}")
        
        # Get trade data
        trade = self.trade_memory.get_trade(trade_id)
        
        if not trade:
            return {
                "error": f"Trade {trade_id} not found"
            }
        
        # Prepare context for Gemma 3
        context = {
            "trade_id": trade_id,
            "trade": trade
        }
        
        # Generate prompt for trade learning
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "trade_learning",
            **context
        )
        
        # Get the appropriate model for learning
        model = self.gemma_core.model_manager.get_model("adaptive_learning")
        
        # Generate learning insights using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract learning insights from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured insights
        # For this implementation, we'll create simulated insights
        
        # Simulate key learnings
        key_learnings = [
            "Entry timing could be improved by waiting for confirmation",
            "Exit was premature, could have captured more profit",
            "Position sizing was appropriate for the risk level"
        ]
        
        # Simulate improvement suggestions
        improvement_suggestions = [
            "Add additional confirmation indicator before entry",
            "Implement trailing stop loss to capture more profit",
            "Consider market regime when setting take profit levels"
        ]
        
        # Create learning insights
        insights = {
            "trade_id": trade_id,
            "key_learnings": key_learnings,
            "improvement_suggestions": improvement_suggestions,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Extracted learning insights from trade {trade_id}")
        return insights
    
    def generate_learning_report(self, period: str = "1m") -> Dict[str, Any]:
        """
        Generate a comprehensive learning report for a specific period.
        
        Parameters:
        -----------
        period : str, optional
            Period for the report (e.g., "1w", "1m", "3m", "6m", "1y").
            Default is "1m" (1 month).
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive learning report.
        """
        self.logger.info(f"Generating learning report for period {period}")
        
        # Convert period to days
        days = 30  # Default to 1 month
        
        if period.endswith("d"):
            days = int(period[:-1])
        elif period.endswith("w"):
            days = int(period[:-1]) * 7
        elif period.endswith("m"):
            days = int(period[:-1]) * 30
        elif period.endswith("y"):
            days = int(period[:-1]) * 365
        
        # Get recent trades
        recent_trades = self.trade_memory.get_recent_trades(100)  # Get more trades than needed for filtering
        
        # Filter trades by period
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_date_str = cutoff_date.isoformat()
        
        filtered_trades = []
        for trade in recent_trades:
            trade_date_str = trade.get("trade_data", {}).get("timestamp", "")
            if trade_date_str and trade_date_str >= cutoff_date_str:
                filtered_trades.append(trade)
        
        # Get performance patterns
        performance_patterns = self.identify_performance_patterns(days)
        
        # Get best performing strategies
        best_strategies = self.trade_memory.get_best_performing_strategies()
        
        # Prepare context for Gemma 3
        context = {
            "period": period,
            "trades": filtered_trades,
            "performance_patterns": performance_patterns,
            "best_strategies": best_strategies
        }
        
        # Generate prompt for learning report
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "learning_report",
            **context
        )
        
        # Get the appropriate model for report generation
        model = self.gemma_core.model_manager.get_model("report_generation")
        
        # Generate report using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract report from chain-of-thought result
        # In a real implementation, this would parse the result to extract a structured report
        # For this implementation, we'll create a simulated report
        
        # Calculate performance metrics
        trade_count = len(filtered_trades)
        
        if trade_count == 0:
            return {
                "period": period,
                "error": "No trades found for this period"
            }
        
        # Extract returns from trades
        returns = []
        win_trades = []
        loss_trades = []
        
        for trade in filtered_trades:
            outcome = trade.get("outcome_data", {})
            return_pct = outcome.get("return_pct", 0)
            
            returns.append(return_pct)
            
            if return_pct > 0:
                win_trades.append(trade)
            else:
                loss_trades.append(trade)
        
        # Calculate metrics
        avg_return = np.mean(returns) if returns else 0
        win_rate = len(win_trades) / trade_count if trade_count > 0 else 0
        
        # Simulate key learnings
        key_learnings = [
            "Market regime detection is critical for strategy selection",
            "Proper position sizing has significant impact on overall performance",
            "Exit timing is more important than entry timing for profitability"
        ]
        
        # Simulate improvement areas
        improvement_areas = [
            "Refine exit criteria to capture more profit",
            "Implement more dynamic position sizing based on volatility",
            "Improve market regime detection accuracy"
        ]
        
        # Simulate action items
        action_items = [
            "Update exit criteria in all strategies",
            "Implement dynamic position sizing module",
            "Enhance market regime detection algorithm"
        ]
        
        # Create learning report
        report = {
            "period": period,
            "trade_count": trade_count,
            "performance_summary": {
                "avg_return": avg_return,
                "win_rate": win_rate,
                "best_strategy": best_strategies[0]["strategy"] if best_strategies else "unknown"
            },
            "key_learnings": key_learnings,
            "improvement_areas": improvement_areas,
            "action_items": action_items,
            "performance_patterns": performance_patterns.get("key_insights", []),
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated learning report for period {period}")
        return report
