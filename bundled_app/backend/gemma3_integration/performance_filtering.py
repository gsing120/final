"""
Performance Filtering System for Gemma Advanced Trading System

This module implements a performance filtering system to ensure only strategies
with positive historical performance are presented to users.
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
from gemma3_integration.automatic_strategy_refinement import AutomaticStrategyRefinement

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class PerformanceFilter:
    """
    Filters strategies based on performance criteria.
    
    This class provides methods for filtering strategies based on performance
    criteria to ensure only strategies with positive historical performance
    are presented to users.
    """
    
    def __init__(self, performance_thresholds: Optional[PerformanceThresholds] = None,
               backtester: Optional[StrategyBacktester] = None,
               strategy_refinement: Optional[AutomaticStrategyRefinement] = None):
        """
        Initialize the PerformanceFilter.
        
        Parameters:
        -----------
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        strategy_refinement : AutomaticStrategyRefinement, optional
            Instance of AutomaticStrategyRefinement for refining strategies.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.PerformanceFilter")
        
        # Create or use provided components
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        self.backtester = backtester or StrategyBacktester()
        self.strategy_refinement = strategy_refinement or AutomaticStrategyRefinement(
            backtester=self.backtester,
            performance_thresholds=self.performance_thresholds
        )
        
        self.logger.info("Initialized PerformanceFilter")
    
    def filter_strategy(self, strategy: Dict[str, Any], ticker: str, 
                      auto_refine: bool = True) -> Dict[str, Any]:
        """
        Filter a strategy based on performance criteria.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to filter.
        ticker : str
            Ticker symbol.
        auto_refine : bool, optional
            Whether to automatically refine the strategy if it doesn't meet
            performance thresholds. Default is True.
            
        Returns:
        --------
        Dict[str, Any]
            Filtered strategy result.
        """
        self.logger.info(f"Filtering strategy for {ticker}")
        
        # Backtest the strategy
        backtest_result = self.backtester.backtest_strategy(strategy, ticker)
        
        if not backtest_result["success"]:
            self.logger.error(f"Failed to backtest strategy: {backtest_result.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": "Failed to backtest strategy",
                "strategy": strategy,
                "passed_filter": False
            }
        
        # Get performance metrics
        performance = backtest_result["performance"]
        
        # Check if strategy meets performance thresholds
        is_valid, validation_results = self.performance_thresholds.is_strategy_valid(performance)
        
        # If strategy meets performance thresholds, return it
        if is_valid:
            self.logger.info(f"Strategy for {ticker} meets performance thresholds")
            return {
                "success": True,
                "strategy": strategy,
                "performance": performance,
                "validation_results": validation_results,
                "passed_filter": True,
                "message": "Strategy meets performance thresholds"
            }
        
        # If strategy doesn't meet performance thresholds and auto_refine is True, refine it
        if auto_refine:
            self.logger.info(f"Strategy for {ticker} doesn't meet performance thresholds, refining...")
            
            # Refine the strategy
            refinement_result = self.strategy_refinement.refine_strategy_until_valid(strategy, ticker)
            
            # If refinement was successful, return the refined strategy
            if refinement_result["success"]:
                self.logger.info(f"Successfully refined strategy for {ticker}")
                return {
                    "success": True,
                    "strategy": refinement_result["strategy"],
                    "performance": refinement_result["performance"],
                    "validation_results": refinement_result["validation_results"],
                    "passed_filter": True,
                    "refinement_history": refinement_result["refinement_history"],
                    "message": "Strategy refined to meet performance thresholds"
                }
            else:
                self.logger.warning(f"Failed to refine strategy for {ticker} to meet performance thresholds")
                return {
                    "success": False,
                    "error": "Failed to refine strategy to meet performance thresholds",
                    "strategy": strategy,
                    "performance": performance,
                    "validation_results": validation_results,
                    "passed_filter": False,
                    "refinement_history": refinement_result.get("refinement_history", []),
                    "message": "Strategy doesn't meet performance thresholds and couldn't be refined"
                }
        
        # If strategy doesn't meet performance thresholds and auto_refine is False, return failure
        self.logger.warning(f"Strategy for {ticker} doesn't meet performance thresholds")
        return {
            "success": False,
            "error": "Strategy doesn't meet performance thresholds",
            "strategy": strategy,
            "performance": performance,
            "validation_results": validation_results,
            "passed_filter": False,
            "message": "Strategy doesn't meet performance thresholds"
        }
    
    def filter_multiple_strategies(self, strategies: List[Dict[str, Any]], ticker: str,
                                auto_refine: bool = True) -> Dict[str, Any]:
        """
        Filter multiple strategies based on performance criteria.
        
        Parameters:
        -----------
        strategies : List[Dict[str, Any]]
            Strategies to filter.
        ticker : str
            Ticker symbol.
        auto_refine : bool, optional
            Whether to automatically refine strategies that don't meet
            performance thresholds. Default is True.
            
        Returns:
        --------
        Dict[str, Any]
            Filtered strategies result.
        """
        self.logger.info(f"Filtering {len(strategies)} strategies for {ticker}")
        
        filtered_strategies = []
        
        # Filter each strategy
        for i, strategy in enumerate(strategies):
            self.logger.info(f"Filtering strategy {i + 1}/{len(strategies)} for {ticker}")
            
            # Filter the strategy
            filter_result = self.filter_strategy(strategy, ticker, auto_refine)
            
            # Add to filtered strategies
            filtered_strategies.append(filter_result)
        
        # Get strategies that passed the filter
        passed_strategies = [result for result in filtered_strategies if result["passed_filter"]]
        
        # If no strategies passed the filter, return the best performing one
        if not passed_strategies:
            self.logger.warning(f"No strategies for {ticker} passed the filter")
            
            # Sort by total return (descending)
            sorted_results = sorted(
                filtered_strategies, 
                key=lambda x: self._extract_numeric_value(x["performance"]["total_return"]), 
                reverse=True
            )
            
            # Get the best strategy
            best_result = sorted_results[0]
            
            return {
                "success": False,
                "error": "No strategies passed the filter",
                "best_strategy": best_result["strategy"],
                "best_performance": best_result["performance"],
                "filtered_strategies": filtered_strategies,
                "message": "No strategies passed the filter, returning best performing one"
            }
        
        # Sort passed strategies by total return (descending)
        sorted_passed = sorted(
            passed_strategies, 
            key=lambda x: self._extract_numeric_value(x["performance"]["total_return"]), 
            reverse=True
        )
        
        # Get the best strategy
        best_result = sorted_passed[0]
        
        return {
            "success": True,
            "best_strategy": best_result["strategy"],
            "best_performance": best_result["performance"],
            "passed_strategies": sorted_passed,
            "filtered_strategies": filtered_strategies,
            "message": f"{len(passed_strategies)}/{len(strategies)} strategies passed the filter"
        }
    
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


class StrategyFilteringSystem:
    """
    Comprehensive system for filtering and refining strategies.
    
    This class provides a complete system for filtering and refining strategies
    to ensure only strategies with positive historical performance are presented
    to users.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None,
               performance_filter: Optional[PerformanceFilter] = None,
               performance_thresholds: Optional[PerformanceThresholds] = None,
               backtester: Optional[StrategyBacktester] = None,
               strategy_refinement: Optional[AutomaticStrategyRefinement] = None):
        """
        Initialize the StrategyFilteringSystem.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        performance_filter : PerformanceFilter, optional
            Instance of PerformanceFilter for filtering strategies.
            If None, creates a new instance.
        performance_thresholds : PerformanceThresholds, optional
            Instance of PerformanceThresholds for validating strategies.
            If None, creates a new instance with default thresholds.
        backtester : StrategyBacktester, optional
            Instance of StrategyBacktester for backtesting strategies.
            If None, creates a new instance.
        strategy_refinement : AutomaticStrategyRefinement, optional
            Instance of AutomaticStrategyRefinement for refining strategies.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyFilteringSystem")
        
        # Create or use provided components
        self.gemma_core = gemma_core or GemmaCore()
        self.performance_thresholds = performance_thresholds or PerformanceThresholds()
        self.backtester = backtester or StrategyBacktester()
        self.strategy_refinement = strategy_refinement or AutomaticStrategyRefinement(
            gemma_core=self.gemma_core,
            backtester=self.backtester,
            performance_thresholds=self.performance_thresholds
        )
        self.performance_filter = performance_filter or PerformanceFilter(
            performance_thresholds=self.performance_thresholds,
            backtester=self.backtester,
            strategy_refinement=self.strategy_refinement
        )
        
        self.logger.info("Initialized StrategyFilteringSystem")
    
    def process_strategy(self, strategy: Dict[str, Any], ticker: str,
                       auto_refine: bool = True) -> Dict[str, Any]:
        """
        Process a strategy through the filtering system.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Strategy to process.
        ticker : str
            Ticker symbol.
        auto_refine : bool, optional
            Whether to automatically refine the strategy if it doesn't meet
            performance thresholds. Default is True.
            
        Returns:
        --------
        Dict[str, Any]
            Processed strategy result.
        """
        self.logger.info(f"Processing strategy for {ticker}")
        
        # Filter the strategy
        filter_result = self.performance_filter.filter_strategy(strategy, ticker, auto_refine)
        
        # If strategy passed the filter, return it
        if filter_result["passed_filter"]:
            self.logger.info(f"Strategy for {ticker} passed the filter")
            
            # Add processing metadata
            filter_result["processing"] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "auto_refine": auto_refine,
                "performance_thresholds": {
                    "min_total_return": self.performance_thresholds.min_total_return,
                    "min_sharpe_ratio": self.performance_thresholds.min_sharpe_ratio,
                    "max_drawdown": self.performance_thresholds.max_drawdown,
                    "min_win_rate": self.performance_thresholds.min_win_rate
                }
            }
            
            return filter_result
        
        # If strategy didn't pass the filter and auto_refine is True, it means
        # refinement was attempted but failed
        if auto_refine:
            self.logger.warning(f"Strategy for {ticker} couldn't be refined to meet performance thresholds")
            
            # Generate explanation for why refinement failed
            explanation = self._generate_refinement_failure_explanation(filter_result)
            
            # Add explanation to the result
            filter_result["explanation"] = explanation
            
            return filter_result
        
        # If strategy didn't pass the filter and auto_refine is False, suggest refinement
        self.logger.warning(f"Strategy for {ticker} didn't pass the filter, suggesting refinement")
        
        # Generate suggestion for refinement
        suggestion = self._generate_refinement_suggestion(filter_result)
        
        # Add suggestion to the result
        filter_result["suggestion"] = suggestion
        
        return filter_result
    
    def process_multiple_strategies(self, strategies: List[Dict[str, Any]], ticker: str,
                                  auto_refine: bool = True) -> Dict[str, Any]:
        """
        Process multiple strategies through the filtering system.
        
        Parameters:
        -----------
        strategies : List[Dict[str, Any]]
            Strategies to process.
        ticker : str
            Ticker symbol.
        auto_refine : bool, optional
            Whether to automatically refine strategies that don't meet
            performance thresholds. Default is True.
            
        Returns:
        --------
        Dict[str, Any]
            Processed strategies result.
        """
        self.logger.info(f"Processing {len(strategies)} strategies for {ticker}")
        
        # Filter multiple strategies
        filter_result = self.performance_filter.filter_multiple_strategies(strategies, ticker, auto_refine)
        
        # Add processing metadata
        filter_result["processing"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "auto_refine": auto_refine,
            "num_strategies": len(strategies),
            "performance_thresholds": {
                "min_total_return": self.performance_thresholds.min_total_return,
                "min_sharpe_ratio": self.performance_thresholds.min_sharpe_ratio,
                "max_drawdown": self.performance_thresholds.max_drawdown,
                "min_win_rate": self.performance_thresholds.min_win_rate
            }
        }
        
        # If at least one strategy passed the filter, return the result
        if filter_result["success"]:
            self.logger.info(f"{len(filter_result['passed_strategies'])}/{len(strategies)} strategies for {ticker} passed the filter")
            return filter_result
        
        # If no strategies passed the filter, generate explanation
        self.logger.warning(f"No strategies for {ticker} passed the filter")
        
        # Generate explanation for why all strategies failed
        explanation = self._generate_all_strategies_failed_explanation(filter_result)
        
        # Add explanation to the result
        filter_result["explanation"] = explanation
        
        return filter_result
    
    def _generate_refinement_failure_explanation(self, filter_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an explanation for why refinement failed.
        
        Parameters:
        -----------
        filter_result : Dict[str, Any]
            Filter result.
            
        Returns:
        --------
        Dict[str, Any]
            Explanation for why refinement failed.
        """
        self.logger.info("Generating refinement failure explanation")
        
        # Extract information from filter result
        strategy = filter_result["strategy"]
        performance = filter_result["performance"]
        validation_results = filter_result["validation_results"]
        refinement_history = filter_result.get("refinement_history", [])
        
        # Prepare context for Gemma 3
        context = {
            "strategy": strategy,
            "performance": performance,
            "validation_results": validation_results,
            "refinement_history": refinement_history
        }
        
        # Generate prompt for refinement failure explanation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "refinement_failure_explanation",
            **context
        )
        
        # Get the appropriate model for refinement failure explanation
        model = self.gemma_core.model_manager.get_model("refinement_failure_explanation")
        
        # Generate refinement failure explanation using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # In a real implementation, this would parse the result to extract structured explanation
        # For this implementation, we'll create simulated explanation
        
        # Identify failing metrics
        failing_metrics = [metric for metric, is_valid in validation_results.items() if not is_valid]
        
        # Create explanation
        explanation = {
            "failing_metrics": failing_metrics,
            "reasons": {},
            "suggestions": []
        }
        
        # Add reasons for each failing metric
        for metric in failing_metrics:
            if metric == "total_return":
                explanation["reasons"]["total_return"] = "The strategy consistently generates negative returns in the historical period. This could be due to poor market timing, incorrect trend identification, or trading against the prevailing market direction."
            elif metric == "sharpe_ratio":
                explanation["reasons"]["sharpe_ratio"] = "The strategy has poor risk-adjusted returns. This indicates that the returns generated do not adequately compensate for the risk taken."
            elif metric == "max_drawdown":
                explanation["reasons"]["max_drawdown"] = "The strategy experiences excessive drawdowns. This suggests inadequate risk management or poor exit timing during adverse market movements."
            elif metric == "win_rate":
                explanation["reasons"]["win_rate"] = "The strategy has a low win rate. This indicates that a majority of trades are unprofitable, suggesting poor entry criteria or premature exits."
        
        # Add suggestions
        explanation["suggestions"] = [
            "Consider a completely different strategy type that better aligns with the current market regime",
            "Extend the historical testing period to include different market conditions",
            "Incorporate additional data sources such as fundamental data or alternative data",
            "Use more sophisticated machine learning techniques to identify patterns",
            "Consult with domain experts for insights on the specific asset"
        ]
        
        # Add summary
        explanation["summary"] = f"After {len(refinement_history) - 1 if refinement_history else 0} refinement iterations, the strategy still fails to meet performance thresholds on {len(failing_metrics)} metrics. The refinement process has reached its maximum iterations without finding a viable solution."
        
        return explanation
    
    def _generate_refinement_suggestion(self, filter_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a suggestion for refinement.
        
        Parameters:
        -----------
        filter_result : Dict[str, Any]
            Filter result.
            
        Returns:
        --------
        Dict[str, Any]
            Suggestion for refinement.
        """
        self.logger.info("Generating refinement suggestion")
        
        # Extract information from filter result
        strategy = filter_result["strategy"]
        performance = filter_result["performance"]
        validation_results = filter_result["validation_results"]
        
        # Prepare context for Gemma 3
        context = {
            "strategy": strategy,
            "performance": performance,
            "validation_results": validation_results
        }
        
        # Generate prompt for refinement suggestion
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "refinement_suggestion",
            **context
        )
        
        # Get the appropriate model for refinement suggestion
        model = self.gemma_core.model_manager.get_model("refinement_suggestion")
        
        # Generate refinement suggestion using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # In a real implementation, this would parse the result to extract structured suggestion
        # For this implementation, we'll create simulated suggestion
        
        # Identify failing metrics
        failing_metrics = [metric for metric, is_valid in validation_results.items() if not is_valid]
        
        # Create suggestion
        suggestion = {
            "failing_metrics": failing_metrics,
            "suggested_refinements": {},
            "expected_improvements": {}
        }
        
        # Add suggested refinements for each failing metric
        for metric in failing_metrics:
            if metric == "total_return":
                suggestion["suggested_refinements"]["total_return"] = [
                    "Adjust entry criteria to wait for stronger confirmation",
                    "Extend holding periods",
                    "Add trend strength filter"
                ]
                suggestion["expected_improvements"]["total_return"] = "Potential improvement of 5-10% in total return by better aligning with market trends"
            elif metric == "sharpe_ratio":
                suggestion["suggested_refinements"]["sharpe_ratio"] = [
                    "Add volatility filter",
                    "Tighten stop losses",
                    "Implement position sizing based on volatility"
                ]
                suggestion["expected_improvements"]["sharpe_ratio"] = "Potential improvement of 0.3-0.5 in Sharpe ratio by reducing unnecessary risk"
            elif metric == "max_drawdown":
                suggestion["suggested_refinements"]["max_drawdown"] = [
                    "Add trailing stops",
                    "Implement drawdown protection rules",
                    "Add market regime filter"
                ]
                suggestion["expected_improvements"]["max_drawdown"] = "Potential reduction of drawdown by 5-10% by implementing better risk management"
            elif metric == "win_rate":
                suggestion["suggested_refinements"]["win_rate"] = [
                    "Add confirmation indicators",
                    "Implement signal strength threshold",
                    "Add volume confirmation"
                ]
                suggestion["expected_improvements"]["win_rate"] = "Potential improvement of 10-15% in win rate by filtering out weak signals"
        
        # Add summary
        suggestion["summary"] = f"The strategy fails to meet performance thresholds on {len(failing_metrics)} metrics. Automatic refinement can be enabled to attempt to improve these metrics."
        
        return suggestion
    
    def _generate_all_strategies_failed_explanation(self, filter_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an explanation for why all strategies failed.
        
        Parameters:
        -----------
        filter_result : Dict[str, Any]
            Filter result.
            
        Returns:
        --------
        Dict[str, Any]
            Explanation for why all strategies failed.
        """
        self.logger.info("Generating explanation for why all strategies failed")
        
        # Extract information from filter result
        filtered_strategies = filter_result["filtered_strategies"]
        best_strategy = filter_result["best_strategy"]
        best_performance = filter_result["best_performance"]
        
        # Prepare context for Gemma 3
        context = {
            "filtered_strategies": filtered_strategies,
            "best_strategy": best_strategy,
            "best_performance": best_performance
        }
        
        # Generate prompt for all strategies failed explanation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "all_strategies_failed_explanation",
            **context
        )
        
        # Get the appropriate model for all strategies failed explanation
        model = self.gemma_core.model_manager.get_model("all_strategies_failed_explanation")
        
        # Generate all strategies failed explanation using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # In a real implementation, this would parse the result to extract structured explanation
        # For this implementation, we'll create simulated explanation
        
        # Create explanation
        explanation = {
            "num_strategies": len(filtered_strategies),
            "best_performance": best_performance,
            "common_issues": [],
            "market_conditions": {},
            "recommendations": []
        }
        
        # Add common issues
        explanation["common_issues"] = [
            "All strategies show negative total returns in the historical period",
            "Risk-adjusted returns (Sharpe ratios) are consistently below threshold",
            "Drawdowns exceed acceptable limits across all strategies",
            "Win rates are below the minimum threshold"
        ]
        
        # Add market conditions
        explanation["market_conditions"] = {
            "assessment": "The asset may be in a difficult market regime that doesn't align well with the types of strategies being tested",
            "suggestion": "Consider analyzing the current market regime more carefully and designing strategies specifically for these conditions"
        }
        
        # Add recommendations
        explanation["recommendations"] = [
            "Try completely different strategy types not yet explored",
            "Consider fundamental analysis in addition to technical indicators",
            "Explore machine learning approaches for pattern recognition",
            "Adjust performance thresholds temporarily to find a starting point",
            "Analyze successful strategies for this asset from external sources"
        ]
        
        # Add summary
        explanation["summary"] = f"All {len(filtered_strategies)} strategies failed to meet performance thresholds. The best performing strategy still had inadequate performance with a total return of {best_performance['total_return']}% and a Sharpe ratio of {best_performance['sharpe_ratio']}."
        
        return explanation
