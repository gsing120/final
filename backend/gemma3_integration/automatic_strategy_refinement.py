"""
Strategy Refinement Module for Gemma Advanced Trading System

This module implements automatic strategy refinement capabilities using Gemma 3
to iteratively improve strategies until they meet performance thresholds.
"""

import os
import logging
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import uuid

# Import Gemma 3 integration components
from gemma3_integration.architecture_enhanced import GemmaCore, PromptEngine, ModelManager
from gemma3_integration.strategy_optimization import PerformanceThresholds, StrategyBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class StrategyRefinementEngine:
    """
    Automatically refines trading strategies until they meet performance thresholds.
    
    This class provides methods for analyzing strategy performance issues and
    iteratively refining strategies to improve their performance.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None,
               backtester: Optional[StrategyBacktester] = None,
               performance_thresholds: Optional[PerformanceThresholds] = None,
               max_refinement_iterations: int = 5):
        """
        Initialize the StrategyRefinementEngine.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        max_refinement_iterations : int, optional
            Maximum number of refinement iterations. Default is 5.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyRefinementEngine")
        
        # Create or use provided components
        self.gemma_core = gemma_core or GemmaCore()
        self.backtester = backtester or StrategyBacktester()
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        
        # Configuration
        self.max_refinement_iterations = max_refinement_iterations
        
        self.logger.info("Initialized StrategyRefinementEngine")
    
    def refine_strategy(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Refine a strategy until it meets performance thresholds.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to refine.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Refined strategy with improved performance.
        """
        self.logger.info(f"Refining strategy for {ticker}")
        
        # Initialize refinement variables
        current_strategy = strategy.copy()
        best_strategy = strategy.copy()
        best_performance = None
        refinement_history = []
        
        # Backtest the initial strategy
        initial_backtest = self.backtester.backtest_strategy(current_strategy, ticker)
        
        if not initial_backtest["success"]:
            self.logger.error(f"Failed to backtest initial strategy: {initial_backtest.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": "Failed to backtest initial strategy",
                "original_strategy": strategy
            }
        
        # Get initial performance
        initial_performance = initial_backtest["performance"]
        best_performance = initial_performance
        
        # Check if initial strategy meets performance thresholds
        is_valid, validation_results = self.performance_thresholds.is_strategy_valid(initial_performance)
        
        if is_valid:
            self.logger.info("Initial strategy already meets performance thresholds")
            return {
                "success": True,
                "strategy": current_strategy,
                "performance": initial_performance,
                "validation_results": validation_results,
                "refinement_history": [],
                "message": "Initial strategy already meets performance thresholds"
            }
        
        # Record initial performance in refinement history
        refinement_history.append({
            "iteration": 0,
            "performance": initial_performance,
            "validation_results": validation_results,
            "changes": None
        })
        
        # Refine the strategy iteratively
        for iteration in range(self.max_refinement_iterations):
            self.logger.info(f"Refinement iteration {iteration + 1}/{self.max_refinement_iterations}")
            
            # Analyze performance issues
            performance_issues = self._analyze_performance_issues(current_strategy, best_performance, validation_results)
            
            # Generate refinement plan
            refinement_plan = self._generate_refinement_plan(current_strategy, performance_issues)
            
            # Apply refinements
            refined_strategy = self._apply_refinements(current_strategy, refinement_plan)
            
            # Backtest the refined strategy
            backtest_result = self.backtester.backtest_strategy(refined_strategy, ticker)
            
            if not backtest_result["success"]:
                self.logger.warning(f"Failed to backtest refined strategy in iteration {iteration + 1}")
                continue
            
            # Get refined performance
            refined_performance = backtest_result["performance"]
            
            # Check if refined strategy meets performance thresholds
            is_valid, validation_results = self.performance_thresholds.is_strategy_valid(refined_performance)
            
            # Record refinement in history
            refinement_history.append({
                "iteration": iteration + 1,
                "performance": refined_performance,
                "validation_results": validation_results,
                "changes": refinement_plan
            })
            
            # Update current strategy
            current_strategy = refined_strategy
            
            # Update best strategy if better than previous best
            if self._is_better_performance(refined_performance, best_performance):
                best_strategy = refined_strategy
                best_performance = refined_performance
                self.logger.info(f"Found better strategy with total return: {best_performance['total_return']}")
            
            # If strategy meets performance thresholds, we're done
            if is_valid:
                self.logger.info(f"Strategy meets performance thresholds after {iteration + 1} iterations")
                break
        
        # Final validation
        is_valid, validation_results = self.performance_thresholds.is_strategy_valid(best_performance)
        
        # Prepare result
        result = {
            "success": is_valid,
            "strategy": best_strategy,
            "performance": best_performance,
            "validation_results": validation_results,
            "refinement_history": refinement_history,
            "original_strategy": strategy,
            "message": f"Strategy {'meets' if is_valid else 'does not meet'} performance thresholds after {len(refinement_history) - 1} refinement iterations"
        }
        
        self.logger.info(f"Completed strategy refinement for {ticker} with success: {is_valid}")
        
        return result
    
    def _analyze_performance_issues(self, strategy: Dict[str, Any], 
                                  performance: Dict[str, Any],
                                  validation_results: Dict[str, bool]) -> Dict[str, Any]:
        """
        Analyze performance issues in a strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to analyze.
        performance : Dict[str, Any]
            Performance metrics of the strategy.
        validation_results : Dict[str, bool]
            Validation results for each performance metric.
            
        Returns:
        --------
        Dict[str, Any]
            Performance issues analysis.
        """
        self.logger.info("Analyzing performance issues")
        
        # Identify failing metrics
        failing_metrics = [metric for metric, is_valid in validation_results.items() if not is_valid]
        
        # Prepare context for Gemma 3
        context = {
            "strategy": strategy,
            "performance": performance,
            "validation_results": validation_results,
            "failing_metrics": failing_metrics
        }
        
        # Generate prompt for performance analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "performance_analysis",
            **context
        )
        
        # Get the appropriate model for performance analysis
        model = self.gemma_core.model_manager.get_model("performance_analysis")
        
        # Generate performance analysis using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # In a real implementation, this would parse the result to extract structured analysis
        # For this implementation, we'll create simulated analysis
        
        performance_issues = {
            "failing_metrics": failing_metrics,
            "issues": {}
        }
        
        # Analyze issues based on failing metrics
        for metric in failing_metrics:
            if metric == "total_return":
                performance_issues["issues"]["total_return"] = {
                    "severity": "high",
                    "root_causes": [
                        "Poor entry timing",
                        "Premature exits",
                        "Insufficient trend confirmation"
                    ],
                    "potential_solutions": [
                        "Adjust entry criteria to wait for stronger confirmation",
                        "Extend holding periods",
                        "Add trend strength filter"
                    ]
                }
            elif metric == "sharpe_ratio":
                performance_issues["issues"]["sharpe_ratio"] = {
                    "severity": "medium",
                    "root_causes": [
                        "High volatility in trades",
                        "Inconsistent returns",
                        "Poor risk-adjusted performance"
                    ],
                    "potential_solutions": [
                        "Add volatility filter",
                        "Tighten stop losses",
                        "Implement position sizing based on volatility"
                    ]
                }
            elif metric == "max_drawdown":
                performance_issues["issues"]["max_drawdown"] = {
                    "severity": "high",
                    "root_causes": [
                        "No downside protection",
                        "Holding through adverse moves",
                        "Poor exit strategy"
                    ],
                    "potential_solutions": [
                        "Add trailing stops",
                        "Implement drawdown protection rules",
                        "Add market regime filter"
                    ]
                }
            elif metric == "win_rate":
                performance_issues["issues"]["win_rate"] = {
                    "severity": "medium",
                    "root_causes": [
                        "Weak entry criteria",
                        "Insufficient signal filtering",
                        "Poor trade selection"
                    ],
                    "potential_solutions": [
                        "Add confirmation indicators",
                        "Implement signal strength threshold",
                        "Add volume confirmation"
                    ]
                }
            elif metric == "volatility":
                performance_issues["issues"]["volatility"] = {
                    "severity": "low",
                    "root_causes": [
                        "Trading in high volatility periods",
                        "No volatility adjustment",
                        "Inconsistent position sizing"
                    ],
                    "potential_solutions": [
                        "Add volatility-based position sizing",
                        "Implement volatility regime filter",
                        "Adjust parameters based on volatility"
                    ]
                }
        
        self.logger.info(f"Identified issues with {len(failing_metrics)} failing metrics")
        
        return performance_issues
    
    def _generate_refinement_plan(self, strategy: Dict[str, Any], 
                                performance_issues: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a refinement plan based on performance issues.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to refine.
        performance_issues : Dict[str, Any]
            Performance issues analysis.
            
        Returns:
        --------
        Dict[str, Any]
            Refinement plan.
        """
        self.logger.info("Generating refinement plan")
        
        # Prepare context for Gemma 3
        context = {
            "strategy": strategy,
            "performance_issues": performance_issues
        }
        
        # Generate prompt for refinement plan
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "refinement_plan",
            **context
        )
        
        # Get the appropriate model for refinement plan
        model = self.gemma_core.model_manager.get_model("refinement_plan")
        
        # Generate refinement plan using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # In a real implementation, this would parse the result to extract structured plan
        # For this implementation, we'll create simulated plan
        
        refinement_plan = {
            "parameter_adjustments": {},
            "indicator_changes": [],
            "signal_generation_changes": [],
            "risk_management_changes": []
        }
        
        # Generate refinement plan based on performance issues
        failing_metrics = performance_issues.get("failing_metrics", [])
        
        for metric in failing_metrics:
            issue = performance_issues.get("issues", {}).get(metric, {})
            solutions = issue.get("potential_solutions", [])
            
            if not solutions:
                continue
            
            # Apply first solution for each issue
            solution = solutions[0]
            
            if "entry" in solution.lower():
                refinement_plan["parameter_adjustments"]["entry_threshold"] = "stricter" if "wait" in solution.lower() else "looser"
            elif "exit" in solution.lower() or "stop" in solution.lower():
                refinement_plan["parameter_adjustments"]["exit_threshold"] = "stricter" if "tighten" in solution.lower() else "looser"
                refinement_plan["risk_management_changes"].append("Adjust stop loss levels")
            elif "volatility" in solution.lower():
                refinement_plan["parameter_adjustments"]["volatility_filter"] = "add"
                refinement_plan["indicator_changes"].append("Add volatility filter")
            elif "confirmation" in solution.lower():
                refinement_plan["parameter_adjustments"]["confirmation_threshold"] = "stricter"
                refinement_plan["indicator_changes"].append("Add confirmation indicator")
            elif "trend" in solution.lower():
                refinement_plan["parameter_adjustments"]["trend_filter"] = "add"
                refinement_plan["indicator_changes"].append("Add trend strength indicator")
            elif "position" in solution.lower():
                refinement_plan["parameter_adjustments"]["position_sizing"] = "dynamic"
                refinement_plan["risk_management_changes"].append("Implement dynamic position sizing")
            elif "drawdown" in solution.lower():
                refinement_plan["parameter_adjustments"]["drawdown_protection"] = "add"
                refinement_plan["risk_management_changes"].append("Add drawdown protection rules")
            elif "regime" in solution.lower():
                refinement_plan["parameter_adjustments"]["regime_filter"] = "add"
                refinement_plan["indicator_changes"].append("Add market regime filter")
        
        self.logger.info(f"Generated refinement plan with {len(refinement_plan['parameter_adjustments'])} parameter adjustments, {len(refinement_plan['indicator_changes'])} indicator changes")
        
        return refinement_plan
    
    def _apply_refinements(self, strategy: Dict[str, Any], 
                         refinement_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply refinements to a strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to refine.
        refinement_plan : Dict[str, Any]
            Refinement plan.
            
        Returns:
        --------
        Dict[str, Any]
            Refined strategy.
        """
        self.logger.info("Applying refinements to strategy")
        
        # Create a copy of the strategy
        refined_strategy = strategy.copy()
        
        # Apply parameter adjustments
        parameter_adjustments = refinement_plan.get("parameter_adjustments", {})
        
        for param, adjustment in parameter_adjustments.items():
            self._adjust_parameter(refined_strategy, param, adjustment)
        
        # Apply indicator changes
        indicator_changes = refinement_plan.get("indicator_changes", [])
        
        for change in indicator_changes:
            self._apply_indicator_change(refined_strategy, change)
        
        # Apply signal generation changes
        signal_changes = refinement_plan.get("signal_generation_changes", [])
        
        for change in signal_changes:
            self._apply_signal_change(refined_strategy, change)
        
        # Apply risk management changes
        risk_changes = refinement_plan.get("risk_management_changes", [])
        
        for change in risk_changes:
            self._apply_risk_change(refined_strategy, change)
        
        # Add refinement metadata
        if "refinements" not in refined_strategy:
            refined_strategy["refinements"] = []
        
        refined_strategy["refinements"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "refinement_plan": refinement_plan
        })
        
        self.logger.info("Applied refinements to strategy")
        
        return refined_strategy
    
    def _adjust_parameter(self, strategy: Dict[str, Any], param: str, adjustment: str) -> None:
        """
        Adjust a parameter in a strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to adjust.
        param : str
            Parameter to adjust.
        adjustment : str
            Adjustment to apply.
        """
        self.logger.info(f"Adjusting parameter {param} with {adjustment}")
        
        # In a real implementation, this would modify specific parameters
        # For this implementation, we'll add the adjustment to the strategy
        
        if "parameters" not in strategy:
            strategy["parameters"] = {}
        
        if "adjustments" not in strategy["parameters"]:
            strategy["parameters"]["adjustments"] = {}
        
        strategy["parameters"]["adjustments"][param] = adjustment
        
        # Apply specific adjustments based on parameter and adjustment
        if param == "entry_threshold":
            if adjustment == "stricter":
                # Make entry criteria more strict
                if "entry_conditions" in strategy:
                    strategy["entry_conditions"] = [f"Stricter: {cond}" for cond in strategy["entry_conditions"]]
            elif adjustment == "looser":
                # Make entry criteria less strict
                if "entry_conditions" in strategy:
                    strategy["entry_conditions"] = [f"Looser: {cond}" for cond in strategy["entry_conditions"]]
        
        elif param == "exit_threshold":
            if adjustment == "stricter":
                # Make exit criteria more strict
                if "exit_conditions" in strategy:
                    strategy["exit_conditions"] = [f"Stricter: {cond}" for cond in strategy["exit_conditions"]]
            elif adjustment == "looser":
                # Make exit criteria less strict
                if "exit_conditions" in strategy:
                    strategy["exit_conditions"] = [f"Looser: {cond}" for cond in strategy["exit_conditions"]]
        
        elif param == "volatility_filter" and adjustment == "add":
            # Add volatility filter
            if "indicators" not in strategy:
                strategy["indicators"] = []
            
            strategy["indicators"].append({
                "name": "ATR",
                "parameters": {"period": 14}
            })
        
        elif param == "trend_filter" and adjustment == "add":
            # Add trend filter
            if "indicators" not in strategy:
                strategy["indicators"] = []
            
            strategy["indicators"].append({
                "name": "ADX",
                "parameters": {"period": 14}
            })
        
        elif param == "regime_filter" and adjustment == "add":
            # Add regime filter
            if "indicators" not in strategy:
                strategy["indicators"] = []
            
            strategy["indicators"].append({
                "name": "Regime Filter",
                "parameters": {"lookback": 50}
            })
    
    def _apply_indicator_change(self, strategy: Dict[str, Any], change: str) -> None:
        """
        Apply an indicator change to a strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to modify.
        change : str
            Indicator change to apply.
        """
        self.logger.info(f"Applying indicator change: {change}")
        
        # In a real implementation, this would modify specific indicators
        # For this implementation, we'll add the change to the strategy
        
        if "indicator_changes" not in strategy:
            strategy["indicator_changes"] = []
        
        strategy["indicator_changes"].append(change)
        
        # Apply specific changes based on the change description
        if "Add volatility filter" in change:
            if "indicators" not in strategy:
                strategy["indicators"] = []
            
            strategy["indicators"].append({
                "name": "ATR",
                "parameters": {"period": 14}
            })
        
        elif "Add confirmation indicator" in change:
            if "indicators" not in strategy:
                strategy["indicators"] = []
            
            strategy["indicators"].append({
                "name": "Stochastic",
                "parameters": {"k_period": 14, "d_period": 3, "slowing": 3}
            })
        
        elif "Add trend strength indicator" in change:
            if "indicators" not in strategy:
                strategy["indicators"] = []
            
            strategy["indicators"].append({
                "name": "ADX",
                "parameters": {"period": 14}
            })
        
        elif "Add market regime filter" in change:
            if "indicators" not in strategy:
                strategy["indicators"] = []
            
            strategy["indicators"].append({
                "name": "Regime Filter",
                "parameters": {"lookback": 50}
            })
    
    def _apply_signal_change(self, strategy: Dict[str, Any], change: str) -> None:
        """
        Apply a signal generation change to a strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to modify.
        change : str
            Signal generation change to apply.
        """
        self.logger.info(f"Applying signal generation change: {change}")
        
        # In a real implementation, this would modify specific signal generation rules
        # For this implementation, we'll add the change to the strategy
        
        if "signal_changes" not in strategy:
            strategy["signal_changes"] = []
        
        strategy["signal_changes"].append(change)
    
    def _apply_risk_change(self, strategy: Dict[str, Any], change: str) -> None:
        """
        Apply a risk management change to a strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to modify.
        change : str
            Risk management change to apply.
        """
        self.logger.info(f"Applying risk management change: {change}")
        
        # In a real implementation, this would modify specific risk management rules
        # For this implementation, we'll add the change to the strategy
        
        if "risk_changes" not in strategy:
            strategy["risk_changes"] = []
        
        strategy["risk_changes"].append(change)
        
        # Apply specific changes based on the change description
        if "Adjust stop loss levels" in change:
            if "risk_management" not in strategy:
                strategy["risk_management"] = {}
            
            # Make stop loss tighter
            if "stop_loss" in strategy["risk_management"]:
                current_stop = strategy["risk_management"]["stop_loss"]
                if isinstance(current_stop, (int, float)):
                    # Tighten stop loss by 20%
                    strategy["risk_management"]["stop_loss"] = current_stop * 0.8
        
        elif "Implement dynamic position sizing" in change:
            if "risk_management" not in strategy:
                strategy["risk_management"] = {}
            
            strategy["risk_management"]["position_sizing"] = "dynamic"
        
        elif "Add drawdown protection rules" in change:
            if "risk_management" not in strategy:
                strategy["risk_management"] = {}
            
            strategy["risk_management"]["drawdown_protection"] = True
            strategy["risk_management"]["max_drawdown_exit"] = 10.0  # Exit if drawdown exceeds 10%
    
    def _is_better_performance(self, performance1: Dict[str, Any], performance2: Dict[str, Any]) -> bool:
        """
        Check if performance1 is better than performance2.
        
        Parameters:
        -----------
        performance1 : Dict[str, Any]
            First performance metrics.
        performance2 : Dict[str, Any]
            Second performance metrics.
            
        Returns:
        --------
        bool
            True if performance1 is better than performance2, False otherwise.
        """
        if performance2 is None:
            return True
        
        # Extract total return
        return1 = self._extract_numeric_value(performance1["total_return"])
        return2 = self._extract_numeric_value(performance2["total_return"])
        
        # Compare total return
        if return1 > return2:
            return True
        
        # If returns are equal, compare Sharpe ratio
        if return1 == return2:
            sharpe1 = self._extract_numeric_value(performance1["sharpe_ratio"])
            sharpe2 = self._extract_numeric_value(performance2["sharpe_ratio"])
            
            return sharpe1 > sharpe2
        
        return False
    
    def _extract_numeric_value(self, value: Any) -> float:
        """
        Extract numeric value from various formats.
        
        Parameters:
        -----------
        value : Any
            Value to extract numeric value from.
            
        Returns:
        --------
        float
            Extracted numeric value.
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Remove percentage sign and convert to float
            return float(value.replace('%', ''))
        else:
            return 0.0


class AutomaticStrategyRefinement:
    """
    Provides automatic strategy refinement capabilities.
    
    This class coordinates the strategy refinement process, integrating with
    the StrategyRefinementEngine and other components to automatically refine
    strategies until they meet performance thresholds.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None,
               refinement_engine: Optional[StrategyRefinementEngine] = None,
               backtester: Optional[StrategyBacktester] = None,
               performance_thresholds: Optional[PerformanceThresholds] = None):
        """
        Initialize the AutomaticStrategyRefinement.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        refinement_engine : StrategyRefinementEngine, optional
            Instance of StrategyRefinementEngine for refining strategies.
            If None, creates a new instance.
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        """
        self.logger = logging.getLogger("GemmaTrading.AutomaticStrategyRefinement")
        
        # Create or use provided components
        self.gemma_core = gemma_core or GemmaCore()
        self.backtester = backtester or StrategyBacktester()
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        self.refinement_engine = refinement_engine or StrategyRefinementEngine(
            gemma_core=self.gemma_core,
            backtester=self.backtester,
            performance_thresholds=self.performance_thresholds
        )
        
        self.logger.info("Initialized AutomaticStrategyRefinement")
    
    def refine_strategy_until_valid(self, strategy: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Refine a strategy until it meets performance thresholds.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to refine.
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict[str, Any]
            Refined strategy with improved performance.
        """
        self.logger.info(f"Refining strategy for {ticker} until valid")
        
        # Backtest the initial strategy
        initial_backtest = self.backtester.backtest_strategy(strategy, ticker)
        
        if not initial_backtest["success"]:
            self.logger.error(f"Failed to backtest initial strategy: {initial_backtest.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": "Failed to backtest initial strategy",
                "original_strategy": strategy
            }
        
        # Get initial performance
        initial_performance = initial_backtest["performance"]
        
        # Check if initial strategy meets performance thresholds
        is_valid, validation_results = self.performance_thresholds.is_strategy_valid(initial_performance)
        
        if is_valid:
            self.logger.info("Initial strategy already meets performance thresholds")
            return {
                "success": True,
                "strategy": strategy,
                "performance": initial_performance,
                "validation_results": validation_results,
                "refinement_history": [],
                "message": "Initial strategy already meets performance thresholds"
            }
        
        # Refine the strategy
        refinement_result = self.refinement_engine.refine_strategy(strategy, ticker)
        
        # Return the refinement result
        return refinement_result
    
    def generate_refinement_report(self, refinement_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a report on the refinement process.
        
        Parameters:
        -----------
        refinement_result : Dict[str, Any]
            Result of the refinement process.
            
        Returns:
        --------
        Dict[str, Any]
            Refinement report.
        """
        self.logger.info("Generating refinement report")
        
        # Extract information from refinement result
        original_strategy = refinement_result.get("original_strategy", {})
        refined_strategy = refinement_result.get("strategy", {})
        refinement_history = refinement_result.get("refinement_history", [])
        
        # Prepare context for Gemma 3
        context = {
            "original_strategy": original_strategy,
            "refined_strategy": refined_strategy,
            "refinement_history": refinement_history,
            "success": refinement_result.get("success", False)
        }
        
        # Generate prompt for refinement report
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "refinement_report",
            **context
        )
        
        # Get the appropriate model for refinement report
        model = self.gemma_core.model_manager.get_model("refinement_report")
        
        # Generate refinement report using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # In a real implementation, this would parse the result to extract structured report
        # For this implementation, we'll create simulated report
        
        # Create performance improvement summary
        performance_improvement = {}
        
        if refinement_history:
            initial_performance = refinement_history[0]["performance"]
            final_performance = refinement_result["performance"]
            
            for metric in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
                initial_value = self._extract_numeric_value(initial_performance.get(metric, 0))
                final_value = self._extract_numeric_value(final_performance.get(metric, 0))
                
                performance_improvement[metric] = {
                    "initial": initial_value,
                    "final": final_value,
                    "change": final_value - initial_value,
                    "percent_change": ((final_value - initial_value) / abs(initial_value) * 100) if initial_value != 0 else 0
                }
        
        # Create changes summary
        changes_summary = {}
        
        for i, iteration in enumerate(refinement_history):
            if i == 0:  # Skip initial state
                continue
            
            changes = iteration.get("changes", {})
            
            for change_type, change_items in changes.items():
                if change_type not in changes_summary:
                    changes_summary[change_type] = []
                
                if isinstance(change_items, dict):
                    for param, value in change_items.items():
                        changes_summary[change_type].append(f"Iteration {i}: {param} = {value}")
                elif isinstance(change_items, list):
                    for item in change_items:
                        changes_summary[change_type].append(f"Iteration {i}: {item}")
        
        # Create report
        report = {
            "success": refinement_result.get("success", False),
            "num_iterations": len(refinement_history) - 1 if refinement_history else 0,
            "performance_improvement": performance_improvement,
            "changes_summary": changes_summary,
            "final_validation": refinement_result.get("validation_results", {}),
            "message": refinement_result.get("message", "")
        }
        
        self.logger.info(f"Generated refinement report with {report['num_iterations']} iterations")
        
        return report
    
    def _extract_numeric_value(self, value: Any) -> float:
        """
        Extract numeric value from various formats.
        
        Parameters:
        -----------
        value : Any
            Value to extract numeric value from.
            
        Returns:
        --------
        float
            Extracted numeric value.
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Remove percentage sign and convert to float
            return float(value.replace('%', ''))
        else:
            return 0.0
