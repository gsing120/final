"""
Strategy Generation and Refinement Module for Gemma Advanced Trading System

This module implements strategy generation and refinement capabilities using Gemma 3
to create new trading strategies from scratch and refine existing ones based on
market conditions and performance data.
"""

import os
import logging
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import uuid

# Import Gemma 3 integration architecture
from gemma3_integration.architecture_enhanced import GemmaCore, PromptEngine, ModelManager
from gemma3_integration.adaptive_learning import AdaptiveLearning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class StrategyGenerator:
    """
    Generates new trading strategies using Gemma 3.
    
    This class provides methods for generating new trading strategies based on
    market conditions, asset characteristics, and trading objectives.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the StrategyGenerator.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyGenerator")
        self.gemma_core = gemma_core or GemmaCore()
        
        self.logger.info("Initialized StrategyGenerator")
    
    def generate_strategy(self, asset_type: str, market_conditions: Dict[str, Any],
                        trading_objectives: Dict[str, Any],
                        constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a new trading strategy.
        
        Parameters:
        -----------
        asset_type : str
            Type of asset (e.g., "stock", "forex", "crypto", "futures").
        market_conditions : Dict[str, Any]
            Current market conditions.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target, time horizon).
        constraints : Dict[str, Any], optional
            Constraints for the strategy (e.g., max drawdown, min win rate).
            
        Returns:
        --------
        Dict[str, Any]
            Generated trading strategy.
        """
        self.logger.info(f"Generating strategy for {asset_type} with {market_conditions.get('regime', 'unknown')} market regime")
        
        # Prepare context for Gemma 3
        context = {
            "asset_type": asset_type,
            "market_conditions": market_conditions,
            "trading_objectives": trading_objectives
        }
        
        if constraints:
            context["constraints"] = constraints
        
        # Generate prompt for strategy generation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_generation",
            **context
        )
        
        # Get the appropriate model for strategy generation
        model = self.gemma_core.model_manager.get_model("strategy_generation")
        
        # Generate strategy using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract strategy from chain-of-thought result
        # In a real implementation, this would parse the result to extract a structured strategy
        # For this implementation, we'll create a simulated strategy
        
        # Generate strategy ID
        strategy_id = f"GS-{uuid.uuid4().hex[:8]}"
        
        # Determine strategy type based on market conditions
        market_regime = market_conditions.get("regime", "unknown")
        market_volatility = market_conditions.get("volatility", "moderate")
        
        if market_regime == "trending":
            strategy_type = "trend_following"
        elif market_regime == "mean_reverting":
            strategy_type = "mean_reversion"
        elif market_regime == "volatile":
            strategy_type = "volatility_breakout"
        elif market_regime == "range_bound":
            strategy_type = "range_trading"
        else:
            strategy_type = "adaptive_multi_strategy"
        
        # Determine timeframe based on trading objectives
        time_horizon = trading_objectives.get("time_horizon", "medium")
        
        if time_horizon == "short":
            timeframe = "1h"
        elif time_horizon == "medium":
            timeframe = "4h"
        elif time_horizon == "long":
            timeframe = "1d"
        else:
            timeframe = "4h"  # Default
        
        # Generate strategy name
        strategy_name = f"{strategy_type.replace('_', ' ').title()} for {market_regime.replace('_', ' ').title()} {asset_type.title()} Markets"
        
        # Generate strategy description
        strategy_description = f"A {strategy_type.replace('_', ' ')} strategy designed for {market_regime.replace('_', ' ')} {asset_type} markets with {market_volatility} volatility. "
        strategy_description += f"Optimized for {time_horizon}-term trading on {timeframe} timeframe."
        
        # Generate indicators based on strategy type
        indicators = []
        
        if strategy_type == "trend_following":
            indicators = [
                {
                    "name": "EMA",
                    "parameters": {"period": 20}
                },
                {
                    "name": "EMA",
                    "parameters": {"period": 50}
                },
                {
                    "name": "MACD",
                    "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                },
                {
                    "name": "ADX",
                    "parameters": {"period": 14}
                }
            ]
        elif strategy_type == "mean_reversion":
            indicators = [
                {
                    "name": "RSI",
                    "parameters": {"period": 14}
                },
                {
                    "name": "Bollinger Bands",
                    "parameters": {"period": 20, "std_dev": 2.0}
                },
                {
                    "name": "Stochastic",
                    "parameters": {"k_period": 14, "d_period": 3, "slowing": 3}
                }
            ]
        elif strategy_type == "volatility_breakout":
            indicators = [
                {
                    "name": "ATR",
                    "parameters": {"period": 14}
                },
                {
                    "name": "Bollinger Bands",
                    "parameters": {"period": 20, "std_dev": 2.5}
                },
                {
                    "name": "Keltner Channels",
                    "parameters": {"period": 20, "atr_period": 10, "atr_multiplier": 2.0}
                }
            ]
        elif strategy_type == "range_trading":
            indicators = [
                {
                    "name": "RSI",
                    "parameters": {"period": 14}
                },
                {
                    "name": "Bollinger Bands",
                    "parameters": {"period": 20, "std_dev": 2.0}
                },
                {
                    "name": "Support/Resistance",
                    "parameters": {"lookback_period": 30, "min_touches": 2}
                }
            ]
        else:  # adaptive_multi_strategy
            indicators = [
                {
                    "name": "EMA",
                    "parameters": {"period": 20}
                },
                {
                    "name": "RSI",
                    "parameters": {"period": 14}
                },
                {
                    "name": "ATR",
                    "parameters": {"period": 14}
                },
                {
                    "name": "MACD",
                    "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                },
                {
                    "name": "Bollinger Bands",
                    "parameters": {"period": 20, "std_dev": 2.0}
                }
            ]
        
        # Generate entry conditions based on strategy type
        entry_conditions = []
        
        if strategy_type == "trend_following":
            entry_conditions = [
                "EMA(20) crosses above EMA(50)",
                "MACD line crosses above Signal line",
                "ADX > 25 indicating strong trend"
            ]
        elif strategy_type == "mean_reversion":
            entry_conditions = [
                "RSI < 30 for long entry (oversold)",
                "RSI > 70 for short entry (overbought)",
                "Price touches lower Bollinger Band for long entry",
                "Price touches upper Bollinger Band for short entry"
            ]
        elif strategy_type == "volatility_breakout":
            entry_conditions = [
                "Price breaks above upper Bollinger Band with increased volume",
                "Price breaks below lower Bollinger Band with increased volume",
                "ATR increasing, indicating rising volatility"
            ]
        elif strategy_type == "range_trading":
            entry_conditions = [
                "Price approaches support level with RSI < 30 for long entry",
                "Price approaches resistance level with RSI > 70 for short entry"
            ]
        else:  # adaptive_multi_strategy
            entry_conditions = [
                "For trending market: EMA(20) crosses above EMA(50)",
                "For mean reversion: RSI < 30 for long, RSI > 70 for short",
                "For breakout: Price breaks out of Bollinger Bands with increased ATR"
            ]
        
        # Generate exit conditions based on strategy type
        exit_conditions = []
        
        if strategy_type == "trend_following":
            exit_conditions = [
                "EMA(20) crosses below EMA(50) for long positions",
                "MACD line crosses below Signal line for long positions",
                "Trailing stop at 2 * ATR",
                "Take profit at 3 * risk"
            ]
        elif strategy_type == "mean_reversion":
            exit_conditions = [
                "RSI crosses above 50 for long positions",
                "RSI crosses below 50 for short positions",
                "Price reaches middle Bollinger Band",
                "Stop loss at 1.5 * ATR from entry"
            ]
        elif strategy_type == "volatility_breakout":
            exit_conditions = [
                "Price returns within Bollinger Bands",
                "ATR decreasing, indicating falling volatility",
                "Take profit at 2 * ATR from entry",
                "Stop loss at 1 * ATR from entry"
            ]
        elif strategy_type == "range_trading":
            exit_conditions = [
                "Price approaches resistance level for long positions",
                "Price approaches support level for short positions",
                "RSI crosses above 60 for long positions",
                "RSI crosses below 40 for short positions"
            ]
        else:  # adaptive_multi_strategy
            exit_conditions = [
                "For trending market: Trailing stop at 2 * ATR",
                "For mean reversion: Price reaches middle Bollinger Band",
                "For breakout: Price returns within Bollinger Bands",
                "Universal stop loss at 2% of account equity"
            ]
        
        # Generate risk management parameters based on trading objectives
        risk_tolerance = trading_objectives.get("risk_tolerance", "moderate")
        
        if risk_tolerance == "low":
            risk_per_trade = 0.5
            max_drawdown = 5.0
            position_size = 1.0
        elif risk_tolerance == "moderate":
            risk_per_trade = 1.0
            max_drawdown = 10.0
            position_size = 2.0
        elif risk_tolerance == "high":
            risk_per_trade = 2.0
            max_drawdown = 20.0
            position_size = 3.0
        else:
            risk_per_trade = 1.0
            max_drawdown = 10.0
            position_size = 2.0
        
        risk_management = {
            "risk_per_trade_pct": risk_per_trade,
            "max_drawdown_pct": max_drawdown,
            "position_size_pct": position_size,
            "stop_loss_type": "ATR",
            "stop_loss_atr_multiplier": 1.5,
            "take_profit_type": "risk_multiple",
            "take_profit_risk_multiple": 2.0,
            "max_open_positions": 5,
            "correlation_filter": True
        }
        
        # Create strategy
        strategy = {
            "id": strategy_id,
            "name": strategy_name,
            "description": strategy_description,
            "type": strategy_type,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "indicators": indicators,
            "entry_conditions": entry_conditions,
            "exit_conditions": exit_conditions,
            "risk_management": risk_management,
            "market_conditions": {
                "suitable_regimes": [market_regime],
                "suitable_volatility": [market_volatility],
                "suitable_liquidity": ["high", "moderate"]
            },
            "performance_expectations": {
                "expected_win_rate": 0.55,
                "expected_risk_reward": 1.5,
                "expected_trades_per_month": 15
            },
            "generation_metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "market_conditions": market_conditions,
                "trading_objectives": trading_objectives,
                "constraints": constraints,
                "reasoning": cot_result.get("reasoning_steps", []),
                "conclusion": cot_result.get("conclusion", "")
            }
        }
        
        self.logger.info(f"Generated strategy {strategy_id}: {strategy_name}")
        return strategy
    
    def generate_multi_timeframe_strategy(self, asset_type: str, 
                                        market_conditions: Dict[str, Any],
                                        trading_objectives: Dict[str, Any],
                                        timeframes: List[str]) -> Dict[str, Any]:
        """
        Generate a multi-timeframe trading strategy.
        
        Parameters:
        -----------
        asset_type : str
            Type of asset (e.g., "stock", "forex", "crypto", "futures").
        market_conditions : Dict[str, Any]
            Current market conditions.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target, time horizon).
        timeframes : List[str]
            List of timeframes to include in the strategy.
            
        Returns:
        --------
        Dict[str, Any]
            Generated multi-timeframe trading strategy.
        """
        self.logger.info(f"Generating multi-timeframe strategy for {asset_type} with timeframes {timeframes}")
        
        # Prepare context for Gemma 3
        context = {
            "asset_type": asset_type,
            "market_conditions": market_conditions,
            "trading_objectives": trading_objectives,
            "timeframes": timeframes
        }
        
        # Generate prompt for multi-timeframe strategy generation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "multi_timeframe_strategy_generation",
            **context
        )
        
        # Get the appropriate model for strategy generation
        model = self.gemma_core.model_manager.get_model("strategy_generation")
        
        # Generate strategy using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract strategy from chain-of-thought result
        # In a real implementation, this would parse the result to extract a structured strategy
        # For this implementation, we'll create a simulated strategy
        
        # Generate strategy ID
        strategy_id = f"GMTF-{uuid.uuid4().hex[:8]}"
        
        # Determine strategy type based on market conditions
        market_regime = market_conditions.get("regime", "unknown")
        
        # Generate strategy name
        strategy_name = f"Multi-Timeframe {market_regime.replace('_', ' ').title()} Strategy for {asset_type.title()}"
        
        # Generate strategy description
        strategy_description = f"A multi-timeframe strategy for {asset_type} markets that analyzes {', '.join(timeframes[:-1])} and {timeframes[-1]} timeframes to identify high-probability trading opportunities in {market_regime.replace('_', ' ')} market conditions."
        
        # Create timeframe components
        timeframe_components = []
        
        for tf in timeframes:
            # Determine role based on timeframe position
            if tf == timeframes[0]:  # Highest timeframe
                role = "trend_identification"
            elif tf == timeframes[-1]:  # Lowest timeframe
                role = "entry_timing"
            else:  # Middle timeframes
                role = "confirmation"
            
            # Create indicators for this timeframe
            tf_indicators = []
            
            if role == "trend_identification":
                tf_indicators = [
                    {
                        "name": "EMA",
                        "parameters": {"period": 50}
                    },
                    {
                        "name": "EMA",
                        "parameters": {"period": 200}
                    },
                    {
                        "name": "ADX",
                        "parameters": {"period": 14}
                    }
                ]
            elif role == "confirmation":
                tf_indicators = [
                    {
                        "name": "MACD",
                        "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                    },
                    {
                        "name": "RSI",
                        "parameters": {"period": 14}
                    }
                ]
            else:  # entry_timing
                tf_indicators = [
                    {
                        "name": "Bollinger Bands",
                        "parameters": {"period": 20, "std_dev": 2.0}
                    },
                    {
                        "name": "Stochastic",
                        "parameters": {"k_period": 14, "d_period": 3, "slowing": 3}
                    },
                    {
                        "name": "ATR",
                        "parameters": {"period": 14}
                    }
                ]
            
            # Create conditions for this timeframe
            tf_conditions = []
            
            if role == "trend_identification":
                tf_conditions = [
                    "EMA(50) above EMA(200) for bullish trend",
                    "EMA(50) below EMA(200) for bearish trend",
                    "ADX > 25 for strong trend"
                ]
            elif role == "confirmation":
                tf_conditions = [
                    "MACD line crosses above Signal line for bullish confirmation",
                    "MACD line crosses below Signal line for bearish confirmation",
                    "RSI > 50 for bullish confirmation",
                    "RSI < 50 for bearish confirmation"
                ]
            else:  # entry_timing
                tf_conditions = [
                    "Price pulls back to lower Bollinger Band for bullish entry",
                    "Price pulls back to upper Bollinger Band for bearish entry",
                    "Stochastic crosses above 20 from oversold for bullish entry",
                    "Stochastic crosses below 80 from overbought for bearish entry"
                ]
            
            # Create component for this timeframe
            timeframe_components.append({
                "timeframe": tf,
                "role": role,
                "indicators": tf_indicators,
                "conditions": tf_conditions
            })
        
        # Generate entry logic
        entry_logic = [
            f"Confirm trend direction on {timeframes[0]} timeframe",
            f"Wait for confirmation on {timeframes[1] if len(timeframes) > 2 else timeframes[-1]} timeframe",
            f"Enter on {timeframes[-1]} timeframe when entry conditions are met"
        ]
        
        # Generate exit logic
        exit_logic = [
            f"Exit when {timeframes[-1]} timeframe shows reversal signals",
            f"Exit when {timeframes[1] if len(timeframes) > 2 else timeframes[0]} timeframe shows trend change",
            "Use trailing stop based on ATR on entry timeframe",
            "Take profit at key resistance/support levels"
        ]
        
        # Generate risk management parameters based on trading objectives
        risk_tolerance = trading_objectives.get("risk_tolerance", "moderate")
        
        if risk_tolerance == "low":
            risk_per_trade = 0.5
            max_drawdown = 5.0
            position_size = 1.0
        elif risk_tolerance == "moderate":
            risk_per_trade = 1.0
            max_drawdown = 10.0
            position_size = 2.0
        elif risk_tolerance == "high":
            risk_per_trade = 2.0
            max_drawdown = 20.0
            position_size = 3.0
        else:
            risk_per_trade = 1.0
            max_drawdown = 10.0
            position_size = 2.0
        
        risk_management = {
            "risk_per_trade_pct": risk_per_trade,
            "max_drawdown_pct": max_drawdown,
            "position_size_pct": position_size,
            "stop_loss_type": "ATR",
            "stop_loss_atr_multiplier": 1.5,
            "take_profit_type": "risk_multiple",
            "take_profit_risk_multiple": 2.0,
            "max_open_positions": 5,
            "correlation_filter": True
        }
        
        # Create strategy
        strategy = {
            "id": strategy_id,
            "name": strategy_name,
            "description": strategy_description,
            "type": "multi_timeframe",
            "asset_type": asset_type,
            "timeframe_components": timeframe_components,
            "entry_logic": entry_logic,
            "exit_logic": exit_logic,
            "risk_management": risk_management,
            "market_conditions": {
                "suitable_regimes": [market_regime],
                "suitable_volatility": ["high", "moderate", "low"],
                "suitable_liquidity": ["high", "moderate"]
            },
            "performance_expectations": {
                "expected_win_rate": 0.60,
                "expected_risk_reward": 1.8,
                "expected_trades_per_month": 10
            },
            "generation_metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "market_conditions": market_conditions,
                "trading_objectives": trading_objectives,
                "timeframes": timeframes,
                "reasoning": cot_result.get("reasoning_steps", []),
                "conclusion": cot_result.get("conclusion", "")
            }
        }
        
        self.logger.info(f"Generated multi-timeframe strategy {strategy_id}: {strategy_name}")
        return strategy
    
    def generate_strategy_ensemble(self, asset_type: str, 
                                 market_conditions: Dict[str, Any],
                                 trading_objectives: Dict[str, Any],
                                 strategy_count: int = 3) -> Dict[str, Any]:
        """
        Generate an ensemble of complementary trading strategies.
        
        Parameters:
        -----------
        asset_type : str
            Type of asset (e.g., "stock", "forex", "crypto", "futures").
        market_conditions : Dict[str, Any]
            Current market conditions.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target, time horizon).
        strategy_count : int, optional
            Number of strategies to include in the ensemble. Default is 3.
            
        Returns:
        --------
        Dict[str, Any]
            Generated strategy ensemble.
        """
        self.logger.info(f"Generating strategy ensemble for {asset_type} with {strategy_count} strategies")
        
        # Prepare context for Gemma 3
        context = {
            "asset_type": asset_type,
            "market_conditions": market_conditions,
            "trading_objectives": trading_objectives,
            "strategy_count": strategy_count
        }
        
        # Generate prompt for strategy ensemble generation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_ensemble_generation",
            **context
        )
        
        # Get the appropriate model for strategy generation
        model = self.gemma_core.model_manager.get_model("strategy_generation")
        
        # Generate ensemble using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract ensemble from chain-of-thought result
        # In a real implementation, this would parse the result to extract a structured ensemble
        # For this implementation, we'll create a simulated ensemble
        
        # Generate ensemble ID
        ensemble_id = f"GE-{uuid.uuid4().hex[:8]}"
        
        # Generate ensemble name
        ensemble_name = f"Adaptive Ensemble for {asset_type.title()} Markets"
        
        # Generate ensemble description
        ensemble_description = f"An ensemble of {strategy_count} complementary trading strategies for {asset_type} markets, designed to perform well across different market conditions with emphasis on {market_conditions.get('regime', 'unknown')} regimes."
        
        # Generate component strategies
        component_strategies = []
        
        # Strategy types to include in the ensemble
        strategy_types = ["trend_following", "mean_reversion", "breakout", "volatility", "pattern_based"]
        
        for i in range(min(strategy_count, len(strategy_types))):
            strategy_type = strategy_types[i]
            
            # Generate strategy ID
            strategy_id = f"{ensemble_id}-S{i+1}"
            
            # Generate strategy name
            strategy_name = f"{strategy_type.replace('_', ' ').title()} Component"
            
            # Generate indicators based on strategy type
            indicators = []
            
            if strategy_type == "trend_following":
                indicators = [
                    {
                        "name": "EMA",
                        "parameters": {"period": 20}
                    },
                    {
                        "name": "EMA",
                        "parameters": {"period": 50}
                    },
                    {
                        "name": "MACD",
                        "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
                    }
                ]
            elif strategy_type == "mean_reversion":
                indicators = [
                    {
                        "name": "RSI",
                        "parameters": {"period": 14}
                    },
                    {
                        "name": "Bollinger Bands",
                        "parameters": {"period": 20, "std_dev": 2.0}
                    }
                ]
            elif strategy_type == "breakout":
                indicators = [
                    {
                        "name": "Donchian Channels",
                        "parameters": {"period": 20}
                    },
                    {
                        "name": "ATR",
                        "parameters": {"period": 14}
                    }
                ]
            elif strategy_type == "volatility":
                indicators = [
                    {
                        "name": "Bollinger Bands",
                        "parameters": {"period": 20, "std_dev": 2.0}
                    },
                    {
                        "name": "ATR",
                        "parameters": {"period": 14}
                    },
                    {
                        "name": "Keltner Channels",
                        "parameters": {"period": 20, "atr_period": 10, "atr_multiplier": 2.0}
                    }
                ]
            else:  # pattern_based
                indicators = [
                    {
                        "name": "Candlestick Patterns",
                        "parameters": {"patterns": ["engulfing", "doji", "hammer", "shooting_star"]}
                    },
                    {
                        "name": "Support/Resistance",
                        "parameters": {"lookback_period": 30, "min_touches": 2}
                    }
                ]
            
            # Generate entry conditions based on strategy type
            entry_conditions = []
            
            if strategy_type == "trend_following":
                entry_conditions = [
                    "EMA(20) crosses above EMA(50) for long entry",
                    "MACD line crosses above Signal line for long entry"
                ]
            elif strategy_type == "mean_reversion":
                entry_conditions = [
                    "RSI < 30 for long entry",
                    "Price touches lower Bollinger Band for long entry"
                ]
            elif strategy_type == "breakout":
                entry_conditions = [
                    "Price breaks above upper Donchian Channel for long entry",
                    "Price breaks below lower Donchian Channel for short entry"
                ]
            elif strategy_type == "volatility":
                entry_conditions = [
                    "Price breaks above upper Bollinger Band with ATR increasing for long entry",
                    "Price breaks below lower Bollinger Band with ATR increasing for short entry"
                ]
            else:  # pattern_based
                entry_conditions = [
                    "Bullish engulfing pattern at support level for long entry",
                    "Bearish engulfing pattern at resistance level for short entry"
                ]
            
            # Generate exit conditions based on strategy type
            exit_conditions = []
            
            if strategy_type == "trend_following":
                exit_conditions = [
                    "EMA(20) crosses below EMA(50) for long positions",
                    "MACD line crosses below Signal line for long positions",
                    "Trailing stop at 2 * ATR"
                ]
            elif strategy_type == "mean_reversion":
                exit_conditions = [
                    "RSI crosses above 70 for long positions",
                    "Price reaches middle Bollinger Band for long positions",
                    "Stop loss at 1.5 * ATR from entry"
                ]
            elif strategy_type == "breakout":
                exit_conditions = [
                    "Price returns to breakout level",
                    "Take profit at 2 * ATR from entry",
                    "Stop loss at 1 * ATR from entry"
                ]
            elif strategy_type == "volatility":
                exit_conditions = [
                    "Price returns within Bollinger Bands",
                    "ATR decreasing, indicating falling volatility",
                    "Take profit at 2 * ATR from entry"
                ]
            else:  # pattern_based
                exit_conditions = [
                    "Reversal pattern forms",
                    "Price reaches next resistance level for long positions",
                    "Price reaches next support level for short positions"
                ]
            
            # Create component strategy
            component_strategy = {
                "id": strategy_id,
                "name": strategy_name,
                "type": strategy_type,
                "timeframe": "4h",  # Default timeframe
                "indicators": indicators,
                "entry_conditions": entry_conditions,
                "exit_conditions": exit_conditions,
                "weight": 1.0 / strategy_count  # Equal weight initially
            }
            
            component_strategies.append(component_strategy)
        
        # Generate ensemble logic
        ensemble_logic = {
            "voting_method": "weighted_majority",
            "position_sizing": "proportional_to_confidence",
            "conflict_resolution": "higher_confidence_wins",
            "adaptation_rules": [
                "Increase weight of strategies with higher recent performance",
                "Decrease weight of strategies with lower recent performance",
                "Minimum weight of 0.1 for any strategy",
                "Rebalance weights weekly to sum to 1.0"
            ]
        }
        
        # Generate risk management parameters based on trading objectives
        risk_tolerance = trading_objectives.get("risk_tolerance", "moderate")
        
        if risk_tolerance == "low":
            risk_per_trade = 0.5
            max_drawdown = 5.0
            position_size = 1.0
        elif risk_tolerance == "moderate":
            risk_per_trade = 1.0
            max_drawdown = 10.0
            position_size = 2.0
        elif risk_tolerance == "high":
            risk_per_trade = 2.0
            max_drawdown = 20.0
            position_size = 3.0
        else:
            risk_per_trade = 1.0
            max_drawdown = 10.0
            position_size = 2.0
        
        risk_management = {
            "risk_per_trade_pct": risk_per_trade,
            "max_drawdown_pct": max_drawdown,
            "position_size_pct": position_size,
            "stop_loss_type": "ATR",
            "stop_loss_atr_multiplier": 1.5,
            "take_profit_type": "risk_multiple",
            "take_profit_risk_multiple": 2.0,
            "max_open_positions": 5,
            "correlation_filter": True
        }
        
        # Create ensemble
        ensemble = {
            "id": ensemble_id,
            "name": ensemble_name,
            "description": ensemble_description,
            "type": "ensemble",
            "asset_type": asset_type,
            "component_strategies": component_strategies,
            "ensemble_logic": ensemble_logic,
            "risk_management": risk_management,
            "market_conditions": {
                "suitable_regimes": ["trending", "mean_reverting", "volatile", "range_bound"],
                "suitable_volatility": ["high", "moderate", "low"],
                "suitable_liquidity": ["high", "moderate"]
            },
            "performance_expectations": {
                "expected_win_rate": 0.65,
                "expected_risk_reward": 2.0,
                "expected_trades_per_month": 20
            },
            "generation_metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "market_conditions": market_conditions,
                "trading_objectives": trading_objectives,
                "strategy_count": strategy_count,
                "reasoning": cot_result.get("reasoning_steps", []),
                "conclusion": cot_result.get("conclusion", "")
            }
        }
        
        self.logger.info(f"Generated strategy ensemble {ensemble_id}: {ensemble_name}")
        return ensemble

class StrategyRefiner:
    """
    Refines existing trading strategies using Gemma 3.
    
    This class provides methods for refining existing trading strategies based on
    performance data, market conditions, and trading objectives.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None, 
               adaptive_learning: Optional[AdaptiveLearning] = None):
        """
        Initialize the StrategyRefiner.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        adaptive_learning : AdaptiveLearning, optional
            Instance of AdaptiveLearning for accessing performance data.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyRefiner")
        self.gemma_core = gemma_core or GemmaCore()
        self.adaptive_learning = adaptive_learning or AdaptiveLearning(self.gemma_core)
        
        self.logger.info("Initialized StrategyRefiner")
    
    def refine_strategy(self, strategy: Dict[str, Any], 
                      market_conditions: Dict[str, Any],
                      performance_data: Optional[Dict[str, Any]] = None,
                      refinement_goals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Refine an existing trading strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to refine.
        market_conditions : Dict[str, Any]
            Current market conditions.
        performance_data : Dict[str, Any], optional
            Performance data for the strategy.
            If None, attempts to retrieve from adaptive learning.
        refinement_goals : Dict[str, Any], optional
            Goals for the refinement (e.g., improve win rate, reduce drawdown).
            
        Returns:
        --------
        Dict[str, Any]
            Refined trading strategy.
        """
        strategy_name = strategy.get("name", "unknown")
        self.logger.info(f"Refining strategy {strategy_name}")
        
        # If performance data not provided, try to retrieve from adaptive learning
        if performance_data is None and "name" in strategy:
            try:
                performance_data = self.adaptive_learning.analyze_strategy_performance(strategy["name"])
            except Exception as e:
                self.logger.warning(f"Could not retrieve performance data: {e}")
                performance_data = None
        
        # Prepare context for Gemma 3
        context = {
            "strategy": strategy,
            "market_conditions": market_conditions
        }
        
        if performance_data:
            context["performance_data"] = performance_data
        
        if refinement_goals:
            context["refinement_goals"] = refinement_goals
        
        # Generate prompt for strategy refinement
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_refinement",
            **context
        )
        
        # Get the appropriate model for strategy refinement
        model = self.gemma_core.model_manager.get_model("strategy_refinement")
        
        # Generate refinement using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract refinement from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured refinements
        # For this implementation, we'll create a simulated refinement
        
        # Start with a copy of the original strategy
        refined_strategy = strategy.copy()
        
        # Generate a new ID for the refined strategy
        original_id = strategy.get("id", "unknown")
        refined_strategy["id"] = f"{original_id}-R{uuid.uuid4().hex[:4]}"
        
        # Update name and description
        refined_strategy["name"] = f"{strategy_name} (Refined)"
        
        if "description" in refined_strategy:
            refined_strategy["description"] = f"{refined_strategy['description']} Refined version with improved parameters and conditions."
        
        # Refine indicators if present
        if "indicators" in refined_strategy:
            for indicator in refined_strategy["indicators"]:
                if "parameters" in indicator:
                    # Adjust indicator parameters slightly
                    for param_name, param_value in indicator["parameters"].items():
                        if isinstance(param_value, (int, float)):
                            # Adjust by up to 20% in either direction
                            adjustment = np.random.uniform(-0.2, 0.2)
                            new_value = param_value * (1 + adjustment)
                            
                            # Round to appropriate precision
                            if isinstance(param_value, int):
                                new_value = max(1, int(round(new_value)))
                            else:
                                new_value = round(new_value, 2)
                            
                            indicator["parameters"][param_name] = new_value
        
        # Refine entry conditions if present
        if "entry_conditions" in refined_strategy and refined_strategy["entry_conditions"]:
            # Add a new condition or modify existing ones
            if np.random.random() < 0.5 and len(refined_strategy["entry_conditions"]) > 0:
                # Modify an existing condition
                index = np.random.randint(0, len(refined_strategy["entry_conditions"]))
                condition = refined_strategy["entry_conditions"][index]
                
                # Add a qualifier to the condition
                qualifiers = ["with confirmation from volume", "after consolidation", "with increasing momentum"]
                qualifier = qualifiers[np.random.randint(0, len(qualifiers))]
                
                refined_strategy["entry_conditions"][index] = f"{condition} {qualifier}"
            else:
                # Add a new condition
                new_conditions = [
                    "Volume above 20-day average",
                    "No major news events expected",
                    "Market sentiment aligns with trade direction"
                ]
                new_condition = new_conditions[np.random.randint(0, len(new_conditions))]
                
                refined_strategy["entry_conditions"].append(new_condition)
        
        # Refine exit conditions if present
        if "exit_conditions" in refined_strategy and refined_strategy["exit_conditions"]:
            # Add a new condition or modify existing ones
            if np.random.random() < 0.5 and len(refined_strategy["exit_conditions"]) > 0:
                # Modify an existing condition
                index = np.random.randint(0, len(refined_strategy["exit_conditions"]))
                condition = refined_strategy["exit_conditions"][index]
                
                # Add a qualifier to the condition
                qualifiers = ["with confirmation", "after holding for minimum period", "with decreasing momentum"]
                qualifier = qualifiers[np.random.randint(0, len(qualifiers))]
                
                refined_strategy["exit_conditions"][index] = f"{condition} {qualifier}"
            else:
                # Add a new condition
                new_conditions = [
                    "Exit if correlation with market changes significantly",
                    "Exit partial position at 50% of target",
                    "Exit if volatility increases beyond threshold"
                ]
                new_condition = new_conditions[np.random.randint(0, len(new_conditions))]
                
                refined_strategy["exit_conditions"].append(new_condition)
        
        # Refine risk management if present
        if "risk_management" in refined_strategy:
            risk_management = refined_strategy["risk_management"]
            
            # Adjust risk management parameters
            for param_name, param_value in risk_management.items():
                if isinstance(param_value, (int, float)):
                    # Determine adjustment based on performance data if available
                    adjustment = 0
                    
                    if performance_data:
                        # If win rate is low, reduce risk
                        win_rate = performance_data.get("performance_metrics", {}).get("win_rate", 0.5)
                        
                        if win_rate < 0.4 and "risk" in param_name.lower():
                            adjustment = -0.1  # Reduce risk
                        elif win_rate > 0.6 and "profit" in param_name.lower():
                            adjustment = 0.1  # Increase profit targets
                    else:
                        # Random small adjustment
                        adjustment = np.random.uniform(-0.1, 0.1)
                    
                    # Apply adjustment
                    new_value = param_value * (1 + adjustment)
                    
                    # Round to appropriate precision
                    if isinstance(param_value, int):
                        new_value = max(1, int(round(new_value)))
                    else:
                        new_value = round(new_value, 2)
                    
                    risk_management[param_name] = new_value
        
        # Add refinement metadata
        if "refinement_metadata" not in refined_strategy:
            refined_strategy["refinement_metadata"] = []
        
        refinement_metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "original_strategy_id": original_id,
            "market_conditions": market_conditions,
            "performance_data_used": performance_data is not None,
            "refinement_goals": refinement_goals,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", "")
        }
        
        refined_strategy["refinement_metadata"].append(refinement_metadata)
        
        self.logger.info(f"Refined strategy {strategy_name} to {refined_strategy['name']}")
        return refined_strategy
    
    def adapt_strategy_to_market_conditions(self, strategy: Dict[str, Any],
                                          current_conditions: Dict[str, Any],
                                          target_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a strategy to different market conditions.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to adapt.
        current_conditions : Dict[str, Any]
            Current market conditions the strategy is designed for.
        target_conditions : Dict[str, Any]
            Target market conditions to adapt the strategy to.
            
        Returns:
        --------
        Dict[str, Any]
            Adapted trading strategy.
        """
        strategy_name = strategy.get("name", "unknown")
        current_regime = current_conditions.get("regime", "unknown")
        target_regime = target_conditions.get("regime", "unknown")
        
        self.logger.info(f"Adapting strategy {strategy_name} from {current_regime} to {target_regime} regime")
        
        # Prepare context for Gemma 3
        context = {
            "strategy": strategy,
            "current_conditions": current_conditions,
            "target_conditions": target_conditions
        }
        
        # Generate prompt for strategy adaptation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_adaptation",
            **context
        )
        
        # Get the appropriate model for strategy adaptation
        model = self.gemma_core.model_manager.get_model("strategy_adaptation")
        
        # Generate adaptation using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract adaptation from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured adaptations
        # For this implementation, we'll create a simulated adaptation
        
        # Start with a copy of the original strategy
        adapted_strategy = strategy.copy()
        
        # Generate a new ID for the adapted strategy
        original_id = strategy.get("id", "unknown")
        adapted_strategy["id"] = f"{original_id}-A{uuid.uuid4().hex[:4]}"
        
        # Update name and description
        adapted_strategy["name"] = f"{strategy_name} for {target_regime.replace('_', ' ').title()} Markets"
        
        if "description" in adapted_strategy:
            adapted_strategy["description"] = f"Adapted version of {strategy_name} optimized for {target_regime.replace('_', ' ')} market conditions."
        
        # Adapt indicators based on target regime
        if "indicators" in adapted_strategy:
            # Add or remove indicators based on target regime
            if target_regime == "trending":
                # Add trend indicators
                adapted_strategy["indicators"].append({
                    "name": "ADX",
                    "parameters": {"period": 14}
                })
            elif target_regime == "mean_reverting":
                # Add mean reversion indicators
                adapted_strategy["indicators"].append({
                    "name": "RSI",
                    "parameters": {"period": 14}
                })
            elif target_regime == "volatile":
                # Add volatility indicators
                adapted_strategy["indicators"].append({
                    "name": "ATR",
                    "parameters": {"period": 14}
                })
            elif target_regime == "range_bound":
                # Add range indicators
                adapted_strategy["indicators"].append({
                    "name": "Bollinger Bands",
                    "parameters": {"period": 20, "std_dev": 2.0}
                })
        
        # Adapt entry conditions based on target regime
        if "entry_conditions" in adapted_strategy:
            # Replace or modify entry conditions based on target regime
            if target_regime == "trending":
                adapted_strategy["entry_conditions"] = [
                    "Price above 50-day moving average",
                    "ADX > 25 indicating strong trend",
                    "Entry on pullbacks to moving average"
                ]
            elif target_regime == "mean_reverting":
                adapted_strategy["entry_conditions"] = [
                    "RSI < 30 for long entries",
                    "RSI > 70 for short entries",
                    "Price near historical support/resistance levels"
                ]
            elif target_regime == "volatile":
                adapted_strategy["entry_conditions"] = [
                    "Breakout from consolidation with increased volume",
                    "ATR increasing, indicating rising volatility",
                    "Entry after initial price movement confirms direction"
                ]
            elif target_regime == "range_bound":
                adapted_strategy["entry_conditions"] = [
                    "Price near lower Bollinger Band for long entries",
                    "Price near upper Bollinger Band for short entries",
                    "Entry when price approaches established support/resistance"
                ]
        
        # Adapt exit conditions based on target regime
        if "exit_conditions" in adapted_strategy:
            # Replace or modify exit conditions based on target regime
            if target_regime == "trending":
                adapted_strategy["exit_conditions"] = [
                    "Trailing stop at 2 * ATR",
                    "Exit when trend weakens (ADX < 20)",
                    "Take profit at major resistance/support levels"
                ]
            elif target_regime == "mean_reverting":
                adapted_strategy["exit_conditions"] = [
                    "RSI crosses above 50 for long positions",
                    "RSI crosses below 50 for short positions",
                    "Price reaches middle Bollinger Band"
                ]
            elif target_regime == "volatile":
                adapted_strategy["exit_conditions"] = [
                    "Tight stop loss at 1 * ATR",
                    "Take profit quickly at 1.5 * risk",
                    "Exit when volatility decreases"
                ]
            elif target_regime == "range_bound":
                adapted_strategy["exit_conditions"] = [
                    "Exit when price approaches opposite band",
                    "Exit when price breaks out of range",
                    "Time-based exit after specific holding period"
                ]
        
        # Adapt risk management based on target regime
        if "risk_management" in adapted_strategy:
            risk_management = adapted_strategy["risk_management"]
            
            # Adjust risk management based on target regime
            if target_regime == "trending":
                # Trending markets: larger position sizes, wider stops
                if "position_size_pct" in risk_management:
                    risk_management["position_size_pct"] *= 1.2
                if "stop_loss_atr_multiplier" in risk_management:
                    risk_management["stop_loss_atr_multiplier"] *= 1.2
            elif target_regime == "mean_reverting":
                # Mean reverting markets: moderate position sizes, moderate stops
                if "position_size_pct" in risk_management:
                    risk_management["position_size_pct"] *= 1.0  # No change
                if "stop_loss_atr_multiplier" in risk_management:
                    risk_management["stop_loss_atr_multiplier"] *= 1.0  # No change
            elif target_regime == "volatile":
                # Volatile markets: smaller position sizes, tighter stops
                if "position_size_pct" in risk_management:
                    risk_management["position_size_pct"] *= 0.7
                if "stop_loss_atr_multiplier" in risk_management:
                    risk_management["stop_loss_atr_multiplier"] *= 0.7
            elif target_regime == "range_bound":
                # Range-bound markets: larger position sizes, tighter stops
                if "position_size_pct" in risk_management:
                    risk_management["position_size_pct"] *= 1.2
                if "stop_loss_atr_multiplier" in risk_management:
                    risk_management["stop_loss_atr_multiplier"] *= 0.8
        
        # Update market conditions
        if "market_conditions" in adapted_strategy:
            adapted_strategy["market_conditions"]["suitable_regimes"] = [target_regime]
        else:
            adapted_strategy["market_conditions"] = {
                "suitable_regimes": [target_regime],
                "suitable_volatility": target_conditions.get("volatility", ["moderate"]),
                "suitable_liquidity": ["high", "moderate"]
            }
        
        # Add adaptation metadata
        if "adaptation_metadata" not in adapted_strategy:
            adapted_strategy["adaptation_metadata"] = []
        
        adaptation_metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "original_strategy_id": original_id,
            "from_conditions": current_conditions,
            "to_conditions": target_conditions,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", "")
        }
        
        adapted_strategy["adaptation_metadata"].append(adaptation_metadata)
        
        self.logger.info(f"Adapted strategy {strategy_name} to {target_regime} regime")
        return adapted_strategy
    
    def merge_strategies(self, strategies: List[Dict[str, Any]], 
                       merge_method: str = "best_components") -> Dict[str, Any]:
        """
        Merge multiple strategies into a single strategy.
        
        Parameters:
        -----------
        strategies : List[Dict[str, Any]]
            List of strategies to merge.
        merge_method : str, optional
            Method for merging strategies. Default is "best_components".
            Options: "best_components", "ensemble", "hybrid".
            
        Returns:
        --------
        Dict[str, Any]
            Merged trading strategy.
        """
        if not strategies:
            raise ValueError("No strategies provided for merging")
        
        strategy_names = [s.get("name", "unknown") for s in strategies]
        self.logger.info(f"Merging {len(strategies)} strategies using {merge_method} method")
        
        # Prepare context for Gemma 3
        context = {
            "strategies": strategies,
            "merge_method": merge_method
        }
        
        # Generate prompt for strategy merging
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_merging",
            **context
        )
        
        # Get the appropriate model for strategy merging
        model = self.gemma_core.model_manager.get_model("strategy_merging")
        
        # Generate merged strategy using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract merged strategy from chain-of-thought result
        # In a real implementation, this would parse the result to extract a structured strategy
        # For this implementation, we'll create a simulated merged strategy
        
        # Generate merged strategy ID
        merged_id = f"GM-{uuid.uuid4().hex[:8]}"
        
        # Generate merged strategy name
        merged_name = f"Merged Strategy: {' + '.join(strategy_names[:2])}"
        if len(strategies) > 2:
            merged_name += f" + {len(strategies) - 2} more"
        
        # Generate merged strategy description
        merged_description = f"A merged strategy combining the strengths of {len(strategies)} different strategies: {', '.join(strategy_names)}."
        
        # Determine asset type (use the most common one)
        asset_types = [s.get("asset_type", "unknown") for s in strategies]
        asset_type_counts = {}
        for at in asset_types:
            if at in asset_type_counts:
                asset_type_counts[at] += 1
            else:
                asset_type_counts[at] = 1
        
        asset_type = max(asset_type_counts.items(), key=lambda x: x[1])[0]
        
        # Merge based on specified method
        if merge_method == "best_components":
            # Take the best components from each strategy
            
            # Collect all indicators
            all_indicators = []
            for strategy in strategies:
                if "indicators" in strategy:
                    all_indicators.extend(strategy["indicators"])
            
            # Remove duplicates (simple approach)
            merged_indicators = []
            indicator_names = set()
            
            for indicator in all_indicators:
                name = indicator.get("name", "")
                if name and name not in indicator_names:
                    merged_indicators.append(indicator)
                    indicator_names.add(name)
            
            # Collect all entry conditions
            all_entry_conditions = []
            for strategy in strategies:
                if "entry_conditions" in strategy:
                    all_entry_conditions.extend(strategy["entry_conditions"])
            
            # Remove duplicates and limit to reasonable number
            merged_entry_conditions = list(set(all_entry_conditions))[:5]
            
            # Collect all exit conditions
            all_exit_conditions = []
            for strategy in strategies:
                if "exit_conditions" in strategy:
                    all_exit_conditions.extend(strategy["exit_conditions"])
            
            # Remove duplicates and limit to reasonable number
            merged_exit_conditions = list(set(all_exit_conditions))[:5]
            
            # Use risk management from the first strategy
            merged_risk_management = strategies[0].get("risk_management", {})
            
            # Create merged strategy
            merged_strategy = {
                "id": merged_id,
                "name": merged_name,
                "description": merged_description,
                "type": "merged",
                "asset_type": asset_type,
                "timeframe": strategies[0].get("timeframe", "4h"),  # Use timeframe from first strategy
                "indicators": merged_indicators,
                "entry_conditions": merged_entry_conditions,
                "exit_conditions": merged_exit_conditions,
                "risk_management": merged_risk_management,
                "market_conditions": {
                    "suitable_regimes": ["trending", "mean_reverting", "volatile", "range_bound"],
                    "suitable_volatility": ["high", "moderate", "low"],
                    "suitable_liquidity": ["high", "moderate"]
                },
                "performance_expectations": {
                    "expected_win_rate": 0.60,
                    "expected_risk_reward": 1.8,
                    "expected_trades_per_month": 15
                }
            }
        
        elif merge_method == "ensemble":
            # Create an ensemble strategy
            
            # Create component strategies
            component_strategies = []
            
            for i, strategy in enumerate(strategies):
                component = {
                    "id": f"{merged_id}-C{i+1}",
                    "name": strategy.get("name", f"Component {i+1}"),
                    "type": strategy.get("type", "unknown"),
                    "timeframe": strategy.get("timeframe", "4h"),
                    "indicators": strategy.get("indicators", []),
                    "entry_conditions": strategy.get("entry_conditions", []),
                    "exit_conditions": strategy.get("exit_conditions", []),
                    "weight": 1.0 / len(strategies)  # Equal weight initially
                }
                
                component_strategies.append(component)
            
            # Create ensemble logic
            ensemble_logic = {
                "voting_method": "weighted_majority",
                "position_sizing": "proportional_to_confidence",
                "conflict_resolution": "higher_confidence_wins",
                "adaptation_rules": [
                    "Increase weight of strategies with higher recent performance",
                    "Decrease weight of strategies with lower recent performance",
                    "Minimum weight of 0.1 for any strategy",
                    "Rebalance weights weekly to sum to 1.0"
                ]
            }
            
            # Use risk management from the first strategy
            merged_risk_management = strategies[0].get("risk_management", {})
            
            # Create merged strategy
            merged_strategy = {
                "id": merged_id,
                "name": merged_name,
                "description": merged_description,
                "type": "ensemble",
                "asset_type": asset_type,
                "component_strategies": component_strategies,
                "ensemble_logic": ensemble_logic,
                "risk_management": merged_risk_management,
                "market_conditions": {
                    "suitable_regimes": ["trending", "mean_reverting", "volatile", "range_bound"],
                    "suitable_volatility": ["high", "moderate", "low"],
                    "suitable_liquidity": ["high", "moderate"]
                },
                "performance_expectations": {
                    "expected_win_rate": 0.65,
                    "expected_risk_reward": 2.0,
                    "expected_trades_per_month": 20
                }
            }
        
        else:  # hybrid
            # Create a hybrid strategy with conditional logic
            
            # Collect all indicators
            all_indicators = []
            for strategy in strategies:
                if "indicators" in strategy:
                    all_indicators.extend(strategy["indicators"])
            
            # Remove duplicates (simple approach)
            merged_indicators = []
            indicator_names = set()
            
            for indicator in all_indicators:
                name = indicator.get("name", "")
                if name and name not in indicator_names:
                    merged_indicators.append(indicator)
                    indicator_names.add(name)
            
            # Create conditional logic
            conditional_logic = []
            
            for i, strategy in enumerate(strategies):
                strategy_type = strategy.get("type", "unknown")
                market_conditions = strategy.get("market_conditions", {})
                suitable_regimes = market_conditions.get("suitable_regimes", [])
                
                condition = {
                    "strategy_name": strategy.get("name", f"Strategy {i+1}"),
                    "conditions": [
                        f"Market regime is {regime}" for regime in suitable_regimes
                    ],
                    "entry_conditions": strategy.get("entry_conditions", []),
                    "exit_conditions": strategy.get("exit_conditions", [])
                }
                
                conditional_logic.append(condition)
            
            # Add default condition
            conditional_logic.append({
                "strategy_name": "Default Strategy",
                "conditions": ["No other conditions met"],
                "entry_conditions": strategies[0].get("entry_conditions", []),
                "exit_conditions": strategies[0].get("exit_conditions", [])
            })
            
            # Use risk management from the first strategy
            merged_risk_management = strategies[0].get("risk_management", {})
            
            # Create merged strategy
            merged_strategy = {
                "id": merged_id,
                "name": merged_name,
                "description": merged_description,
                "type": "hybrid",
                "asset_type": asset_type,
                "timeframe": strategies[0].get("timeframe", "4h"),  # Use timeframe from first strategy
                "indicators": merged_indicators,
                "conditional_logic": conditional_logic,
                "risk_management": merged_risk_management,
                "market_conditions": {
                    "suitable_regimes": ["trending", "mean_reverting", "volatile", "range_bound"],
                    "suitable_volatility": ["high", "moderate", "low"],
                    "suitable_liquidity": ["high", "moderate"]
                },
                "performance_expectations": {
                    "expected_win_rate": 0.62,
                    "expected_risk_reward": 1.9,
                    "expected_trades_per_month": 18
                }
            }
        
        # Add merge metadata
        merged_strategy["merge_metadata"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source_strategy_ids": [s.get("id", "unknown") for s in strategies],
            "source_strategy_names": strategy_names,
            "merge_method": merge_method,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", "")
        }
        
        self.logger.info(f"Merged {len(strategies)} strategies into {merged_name}")
        return merged_strategy

class StrategyGenerationAndRefinement:
    """
    Main class for strategy generation and refinement using Gemma 3.
    
    This class provides a unified interface for all strategy generation and refinement
    capabilities, including strategy generation, refinement, adaptation, and merging.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the StrategyGenerationAndRefinement.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyGenerationAndRefinement")
        
        # Create or use provided GemmaCore
        self.gemma_core = gemma_core or GemmaCore()
        
        # Initialize components
        self.adaptive_learning = AdaptiveLearning(self.gemma_core)
        self.strategy_generator = StrategyGenerator(self.gemma_core)
        self.strategy_refiner = StrategyRefiner(self.gemma_core, self.adaptive_learning)
        
        self.logger.info("Initialized StrategyGenerationAndRefinement")
    
    def generate_strategy(self, asset_type: str, market_conditions: Dict[str, Any],
                        trading_objectives: Dict[str, Any],
                        constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a new trading strategy.
        
        Parameters:
        -----------
        asset_type : str
            Type of asset (e.g., "stock", "forex", "crypto", "futures").
        market_conditions : Dict[str, Any]
            Current market conditions.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target, time horizon).
        constraints : Dict[str, Any], optional
            Constraints for the strategy (e.g., max drawdown, min win rate).
            
        Returns:
        --------
        Dict[str, Any]
            Generated trading strategy.
        """
        return self.strategy_generator.generate_strategy(
            asset_type, market_conditions, trading_objectives, constraints
        )
    
    def generate_multi_timeframe_strategy(self, asset_type: str, 
                                        market_conditions: Dict[str, Any],
                                        trading_objectives: Dict[str, Any],
                                        timeframes: List[str]) -> Dict[str, Any]:
        """
        Generate a multi-timeframe trading strategy.
        
        Parameters:
        -----------
        asset_type : str
            Type of asset (e.g., "stock", "forex", "crypto", "futures").
        market_conditions : Dict[str, Any]
            Current market conditions.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target, time horizon).
        timeframes : List[str]
            List of timeframes to include in the strategy.
            
        Returns:
        --------
        Dict[str, Any]
            Generated multi-timeframe trading strategy.
        """
        return self.strategy_generator.generate_multi_timeframe_strategy(
            asset_type, market_conditions, trading_objectives, timeframes
        )
    
    def generate_strategy_ensemble(self, asset_type: str, 
                                 market_conditions: Dict[str, Any],
                                 trading_objectives: Dict[str, Any],
                                 strategy_count: int = 3) -> Dict[str, Any]:
        """
        Generate an ensemble of complementary trading strategies.
        
        Parameters:
        -----------
        asset_type : str
            Type of asset (e.g., "stock", "forex", "crypto", "futures").
        market_conditions : Dict[str, Any]
            Current market conditions.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target, time horizon).
        strategy_count : int, optional
            Number of strategies to include in the ensemble. Default is 3.
            
        Returns:
        --------
        Dict[str, Any]
            Generated strategy ensemble.
        """
        return self.strategy_generator.generate_strategy_ensemble(
            asset_type, market_conditions, trading_objectives, strategy_count
        )
    
    def refine_strategy(self, strategy: Dict[str, Any], 
                      market_conditions: Dict[str, Any],
                      performance_data: Optional[Dict[str, Any]] = None,
                      refinement_goals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Refine an existing trading strategy.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to refine.
        market_conditions : Dict[str, Any]
            Current market conditions.
        performance_data : Dict[str, Any], optional
            Performance data for the strategy.
            If None, attempts to retrieve from adaptive learning.
        refinement_goals : Dict[str, Any], optional
            Goals for the refinement (e.g., improve win rate, reduce drawdown).
            
        Returns:
        --------
        Dict[str, Any]
            Refined trading strategy.
        """
        return self.strategy_refiner.refine_strategy(
            strategy, market_conditions, performance_data, refinement_goals
        )
    
    def adapt_strategy_to_market_conditions(self, strategy: Dict[str, Any],
                                          current_conditions: Dict[str, Any],
                                          target_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a strategy to different market conditions.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to adapt.
        current_conditions : Dict[str, Any]
            Current market conditions the strategy is designed for.
        target_conditions : Dict[str, Any]
            Target market conditions to adapt the strategy to.
            
        Returns:
        --------
        Dict[str, Any]
            Adapted trading strategy.
        """
        return self.strategy_refiner.adapt_strategy_to_market_conditions(
            strategy, current_conditions, target_conditions
        )
    
    def merge_strategies(self, strategies: List[Dict[str, Any]], 
                       merge_method: str = "best_components") -> Dict[str, Any]:
        """
        Merge multiple strategies into a single strategy.
        
        Parameters:
        -----------
        strategies : List[Dict[str, Any]]
            List of strategies to merge.
        merge_method : str, optional
            Method for merging strategies. Default is "best_components".
            Options: "best_components", "ensemble", "hybrid".
            
        Returns:
        --------
        Dict[str, Any]
            Merged trading strategy.
        """
        return self.strategy_refiner.merge_strategies(strategies, merge_method)
    
    def generate_strategy_for_specific_asset(self, ticker: str, 
                                           asset_data: pd.DataFrame,
                                           market_conditions: Dict[str, Any],
                                           trading_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a strategy specifically tailored for a particular asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        asset_data : pd.DataFrame
            Historical data for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target, time horizon).
            
        Returns:
        --------
        Dict[str, Any]
            Generated trading strategy for the specific asset.
        """
        self.logger.info(f"Generating strategy specifically for {ticker}")
        
        # Determine asset type from ticker (simplified)
        asset_type = "stock"
        if "USD" in ticker or "/" in ticker:
            asset_type = "forex"
        elif ticker.endswith("USDT") or ticker.endswith("BTC"):
            asset_type = "crypto"
        
        # Analyze asset characteristics
        asset_characteristics = self._analyze_asset_characteristics(ticker, asset_data)
        
        # Generate base strategy
        base_strategy = self.generate_strategy(
            asset_type, market_conditions, trading_objectives
        )
        
        # Customize strategy for the specific asset
        custom_strategy = self._customize_strategy_for_asset(
            base_strategy, ticker, asset_characteristics
        )
        
        return custom_strategy
    
    def _analyze_asset_characteristics(self, ticker: str, asset_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of a specific asset."""
        # Calculate basic metrics
        returns = asset_data["close"].pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate average volume if available
        avg_volume = None
        if "volume" in asset_data.columns:
            avg_volume = asset_data["volume"].mean()
        
        # Calculate autocorrelation
        autocorr = returns.autocorr(lag=1) if len(returns) > 1 else 0
        
        # Determine if mean-reverting or trending
        if autocorr < -0.1:
            behavior = "mean_reverting"
        elif autocorr > 0.1:
            behavior = "trending"
        else:
            behavior = "random_walk"
        
        # Create asset characteristics
        characteristics = {
            "ticker": ticker,
            "volatility": volatility,
            "avg_volume": avg_volume,
            "autocorrelation": autocorr,
            "behavior": behavior,
            "avg_daily_range_pct": (asset_data["high"] / asset_data["low"] - 1).mean() * 100
        }
        
        return characteristics
    
    def _customize_strategy_for_asset(self, strategy: Dict[str, Any], 
                                    ticker: str, 
                                    asset_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Customize a strategy for a specific asset."""
        # Start with a copy of the base strategy
        custom_strategy = strategy.copy()
        
        # Update ID and name
        custom_strategy["id"] = f"{strategy['id']}-{ticker}"
        custom_strategy["name"] = f"{strategy['name']} for {ticker}"
        
        # Update description
        custom_strategy["description"] = f"Custom strategy for {ticker} based on its specific characteristics. {strategy['description']}"
        
        # Adjust risk management based on asset volatility
        if "risk_management" in custom_strategy:
            volatility = asset_characteristics.get("volatility", 0.2)
            behavior = asset_characteristics.get("behavior", "random_walk")
            
            risk_management = custom_strategy["risk_management"]
            
            # Adjust position size based on volatility
            if "position_size_pct" in risk_management:
                if volatility > 0.3:  # High volatility
                    risk_management["position_size_pct"] *= 0.8
                elif volatility < 0.15:  # Low volatility
                    risk_management["position_size_pct"] *= 1.2
            
            # Adjust stop loss based on behavior
            if "stop_loss_atr_multiplier" in risk_management:
                if behavior == "mean_reverting":
                    risk_management["stop_loss_atr_multiplier"] *= 1.2  # Wider stops for mean reversion
                elif behavior == "trending":
                    risk_management["stop_loss_atr_multiplier"] *= 0.9  # Tighter stops for trending
        
        # Add asset-specific metadata
        custom_strategy["asset_specific_metadata"] = {
            "ticker": ticker,
            "asset_characteristics": asset_characteristics,
            "customization_timestamp": datetime.datetime.now().isoformat()
        }
        
        return custom_strategy
