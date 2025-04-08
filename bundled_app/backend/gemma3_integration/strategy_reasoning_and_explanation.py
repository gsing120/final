"""
Strategy Reasoning and Explanation Module for Gemma Advanced Trading System

This module implements strategy reasoning and explanation capabilities using Gemma 3
to provide detailed reasoning for trade recommendations and explain why particular
strategies are appropriate for current market conditions.
"""

import os
import logging
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Import Gemma 3 integration architecture
from gemma3_integration.architecture_enhanced import GemmaCore, PromptEngine, ChainOfThoughtProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class StrategyReasoning:
    """
    Provides detailed reasoning for trading strategies using Gemma 3.
    
    This class generates explanations for why particular trading strategies are
    appropriate for current market conditions, using chain-of-thought reasoning.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the StrategyReasoning.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyReasoning")
        self.gemma_core = gemma_core or GemmaCore()
        
        self.logger.info("Initialized StrategyReasoning")
    
    def explain_strategy(self, ticker: str, strategy_type: str, strategy: Dict[str, Any],
                       technical_analysis: Dict[str, Any],
                       market_conditions: Dict[str, Any],
                       news_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a detailed explanation for a trading strategy.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        strategy_type : str
            Type of strategy (e.g., "swing", "trend", "mean_reversion").
        strategy : Dict[str, Any]
            Details of the trading strategy.
        technical_analysis : Dict[str, Any]
            Technical analysis for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
        news_analysis : Dict[str, Any], optional
            News analysis for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed explanation of the strategy.
        """
        self.logger.info(f"Explaining {strategy_type} strategy for {ticker}")
        
        # Prepare context for Gemma 3
        context = {
            "ticker": ticker,
            "strategy_type": strategy_type,
            "strategy": strategy,
            "technical_analysis": technical_analysis,
            "market_conditions": market_conditions
        }
        
        if news_analysis:
            context["news_analysis"] = news_analysis
        
        # Generate prompt for strategy explanation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_reasoning",
            **context
        )
        
        # Get the appropriate model for strategy reasoning
        model = self.gemma_core.model_manager.get_model("strategy_reasoning")
        
        # Generate explanation using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract explanation from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured explanation
        # For this implementation, we'll create a structured explanation based on the reasoning steps
        
        reasoning_steps = cot_result.get("reasoning_steps", [])
        conclusion = cot_result.get("conclusion", "")
        
        # Create structured explanation
        explanation = {
            "ticker": ticker,
            "strategy_type": strategy_type,
            "summary": self._generate_summary(strategy_type, ticker, conclusion),
            "key_points": self._extract_key_points(reasoning_steps, conclusion),
            "market_context": self._extract_market_context(market_conditions, technical_analysis, news_analysis),
            "risk_assessment": self._extract_risk_assessment(reasoning_steps, strategy),
            "reasoning_process": reasoning_steps,
            "conclusion": conclusion,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated explanation for {strategy_type} strategy for {ticker}")
        return explanation
    
    def _generate_summary(self, strategy_type: str, ticker: str, conclusion: str) -> str:
        """Generate a summary of the strategy explanation."""
        # In a real implementation, this would extract or generate a summary from the conclusion
        # For this implementation, we'll create a simple summary
        
        strategy_descriptions = {
            "swing": f"a short to medium-term trading approach for {ticker} that aims to capture price movements over several days to weeks",
            "trend": f"a directional trading approach for {ticker} that follows the established market trend",
            "mean_reversion": f"a counter-trend trading approach for {ticker} that capitalizes on price returning to its average",
            "breakout": f"a momentum-based approach for {ticker} that enters when price breaks through significant levels",
            "momentum": f"a strategy for {ticker} that trades in the direction of strong price movements"
        }
        
        description = strategy_descriptions.get(
            strategy_type, 
            f"a trading approach for {ticker} based on {strategy_type} principles"
        )
        
        summary = f"This {strategy_type} strategy is {description}. " + \
                 f"Based on current market conditions and analysis, this strategy is appropriate because it aligns with the observed market behavior and technical indicators."
        
        return summary
    
    def _extract_key_points(self, reasoning_steps: List[str], conclusion: str) -> List[str]:
        """Extract key points from reasoning steps and conclusion."""
        # In a real implementation, this would extract actual key points from the reasoning
        # For this implementation, we'll create simulated key points
        
        # Start with some generic key points
        key_points = [
            "Technical indicators show favorable conditions for this strategy",
            "Current market regime aligns with the strategy's approach",
            "Risk-reward ratio is attractive given current volatility"
        ]
        
        # Add some points based on reasoning steps if available
        if reasoning_steps and len(reasoning_steps) >= 3:
            # Take a subset of reasoning steps and convert them to key points
            for i in range(min(3, len(reasoning_steps))):
                step = reasoning_steps[i]
                # Convert reasoning step to a key point by extracting the main idea
                if len(step) > 20:  # Only use substantive steps
                    key_point = step[:50] + "..." if len(step) > 50 else step
                    key_points.append(key_point)
        
        return key_points
    
    def _extract_market_context(self, market_conditions: Dict[str, Any], 
                              technical_analysis: Dict[str, Any],
                              news_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract market context from provided data."""
        # In a real implementation, this would extract actual market context
        # For this implementation, we'll create a simulated market context
        
        # Extract market regime if available
        market_regime = market_conditions.get("regime", "unknown")
        
        # Extract trend information if available
        trend = technical_analysis.get("trend", "unknown")
        
        # Extract volatility information if available
        volatility = technical_analysis.get("volatility", "unknown")
        
        # Extract sentiment if news analysis is available
        sentiment = "unknown"
        if news_analysis:
            sentiment = news_analysis.get("sentiment", "unknown")
        
        market_context = {
            "market_regime": market_regime,
            "trend": trend,
            "volatility": volatility,
            "sentiment": sentiment,
            "description": f"The market is currently in a {market_regime} regime with {trend} trend and {volatility} volatility. "
                          f"News sentiment is {sentiment}."
        }
        
        return market_context
    
    def _extract_risk_assessment(self, reasoning_steps: List[str], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk assessment from reasoning steps and strategy."""
        # In a real implementation, this would extract actual risk assessment
        # For this implementation, we'll create a simulated risk assessment
        
        # Extract risk management parameters from strategy if available
        stop_loss = strategy.get("risk_management", {}).get("stop_loss", "unknown")
        take_profit = strategy.get("risk_management", {}).get("take_profit", "unknown")
        max_drawdown = strategy.get("risk_management", {}).get("max_drawdown", "unknown")
        
        # Create risk assessment
        risk_level = "moderate"  # Default risk level
        
        # Adjust risk level based on strategy parameters
        if stop_loss != "unknown" and take_profit != "unknown":
            # If we have both stop loss and take profit, calculate risk-reward ratio
            try:
                sl_value = float(stop_loss.split()[0])
                tp_value = float(take_profit.split()[0])
                
                risk_reward = tp_value / sl_value
                
                if risk_reward >= 3:
                    risk_level = "low"
                elif risk_reward >= 2:
                    risk_level = "moderate"
                else:
                    risk_level = "high"
            except:
                # If conversion fails, keep default risk level
                pass
        
        risk_assessment = {
            "risk_level": risk_level,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_drawdown": max_drawdown,
            "description": f"This strategy has a {risk_level} risk level with a stop loss at {stop_loss}, "
                          f"take profit at {take_profit}, and maximum drawdown of {max_drawdown}."
        }
        
        return risk_assessment
    
    def compare_strategies(self, ticker: str, strategies: List[Dict[str, Any]],
                         market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple trading strategies for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        strategies : List[Dict[str, Any]]
            List of trading strategies to compare.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        Dict[str, Any]
            Comparison of the strategies.
        """
        self.logger.info(f"Comparing {len(strategies)} strategies for {ticker}")
        
        # Prepare context for Gemma 3
        context = {
            "ticker": ticker,
            "strategies": strategies,
            "market_conditions": market_conditions
        }
        
        # Generate prompt for strategy comparison
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_comparison",
            **context
        )
        
        # Get the appropriate model for strategy reasoning
        model = self.gemma_core.model_manager.get_model("strategy_reasoning")
        
        # Generate comparison using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract comparison from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured comparison
        # For this implementation, we'll create a structured comparison
        
        # Create individual strategy evaluations
        strategy_evaluations = []
        
        for i, strategy in enumerate(strategies):
            strategy_type = strategy.get("strategy_type", "unknown")
            
            evaluation = {
                "strategy_type": strategy_type,
                "strengths": [
                    f"Strength 1 for {strategy_type}",
                    f"Strength 2 for {strategy_type}"
                ],
                "weaknesses": [
                    f"Weakness 1 for {strategy_type}",
                    f"Weakness 2 for {strategy_type}"
                ],
                "suitability_score": np.random.uniform(0, 10)  # Random score for simulation
            }
            
            strategy_evaluations.append(evaluation)
        
        # Sort strategies by suitability score
        strategy_evaluations.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        # Determine recommended strategy
        recommended_strategy = strategy_evaluations[0]["strategy_type"] if strategy_evaluations else "none"
        
        comparison = {
            "ticker": ticker,
            "strategy_evaluations": strategy_evaluations,
            "recommended_strategy": recommended_strategy,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed strategy comparison for {ticker}")
        return comparison

class TradeExplanation:
    """
    Provides detailed explanations for trade signals using Gemma 3.
    
    This class generates explanations for trade signals, including entry and exit
    decisions, with chain-of-thought reasoning.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the TradeExplanation.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.TradeExplanation")
        self.gemma_core = gemma_core or GemmaCore()
        
        self.logger.info("Initialized TradeExplanation")
    
    def explain_entry_signal(self, ticker: str, signal_type: str,
                           price_data: pd.DataFrame,
                           technical_indicators: Dict[str, Any],
                           market_conditions: Dict[str, Any],
                           strategy: Dict[str, Any],
                           news_sentiment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a detailed explanation for an entry signal.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        signal_type : str
            Type of signal (e.g., "buy", "sell").
        price_data : pd.DataFrame
            Recent price data for the asset.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
        strategy : Dict[str, Any]
            The trading strategy being used.
        news_sentiment : Dict[str, Any], optional
            News sentiment for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed explanation of the entry signal.
        """
        self.logger.info(f"Explaining {signal_type} entry signal for {ticker}")
        
        # Prepare context for Gemma 3
        context = {
            "ticker": ticker,
            "signal_type": signal_type,
            "price_data": str(price_data.tail(10)),  # Simplified for this implementation
            "technical_indicators": technical_indicators,
            "market_conditions": market_conditions,
            "strategy": strategy
        }
        
        if news_sentiment:
            context["news_sentiment"] = news_sentiment
        
        # Generate prompt for entry signal explanation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "entry_signal_explanation",
            **context
        )
        
        # Get the appropriate model for trade explanation
        model = self.gemma_core.model_manager.get_model("trade_analysis")
        
        # Generate explanation using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract explanation from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured explanation
        # For this implementation, we'll create a structured explanation
        
        reasoning_steps = cot_result.get("reasoning_steps", [])
        conclusion = cot_result.get("conclusion", "")
        
        # Create structured explanation
        explanation = {
            "ticker": ticker,
            "signal_type": signal_type,
            "summary": self._generate_signal_summary(signal_type, ticker, conclusion),
            "key_reasons": self._extract_signal_reasons(reasoning_steps, technical_indicators),
            "technical_factors": self._extract_technical_factors(technical_indicators),
            "market_context": self._extract_signal_market_context(market_conditions),
            "risk_assessment": self._extract_signal_risk(strategy, market_conditions),
            "reasoning_process": reasoning_steps,
            "conclusion": conclusion,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated explanation for {signal_type} entry signal for {ticker}")
        return explanation
    
    def explain_exit_signal(self, ticker: str, signal_type: str,
                          entry_price: float, current_price: float,
                          holding_period: int,
                          technical_indicators: Dict[str, Any],
                          market_conditions: Dict[str, Any],
                          strategy: Dict[str, Any],
                          news_sentiment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a detailed explanation for an exit signal.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        signal_type : str
            Type of signal (e.g., "take_profit", "stop_loss", "exit").
        entry_price : float
            Price at which the position was entered.
        current_price : float
            Current price of the asset.
        holding_period : int
            Number of periods the position has been held.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
        strategy : Dict[str, Any]
            The trading strategy being used.
        news_sentiment : Dict[str, Any], optional
            News sentiment for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed explanation of the exit signal.
        """
        self.logger.info(f"Explaining {signal_type} exit signal for {ticker}")
        
        # Calculate return
        return_pct = (current_price - entry_price) / entry_price * 100
        
        # Prepare context for Gemma 3
        context = {
            "ticker": ticker,
            "signal_type": signal_type,
            "entry_price": entry_price,
            "current_price": current_price,
            "return_pct": return_pct,
            "holding_period": holding_period,
            "technical_indicators": technical_indicators,
            "market_conditions": market_conditions,
            "strategy": strategy
        }
        
        if news_sentiment:
            context["news_sentiment"] = news_sentiment
        
        # Generate prompt for exit signal explanation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "exit_signal_explanation",
            **context
        )
        
        # Get the appropriate model for trade explanation
        model = self.gemma_core.model_manager.get_model("trade_analysis")
        
        # Generate explanation using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract explanation from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured explanation
        # For this implementation, we'll create a structured explanation
        
        reasoning_steps = cot_result.get("reasoning_steps", [])
        conclusion = cot_result.get("conclusion", "")
        
        # Create structured explanation
        explanation = {
            "ticker": ticker,
            "signal_type": signal_type,
            "entry_price": entry_price,
            "current_price": current_price,
            "return_pct": return_pct,
            "holding_period": holding_period,
            "summary": self._generate_exit_summary(signal_type, ticker, return_pct, conclusion),
            "key_reasons": self._extract_signal_reasons(reasoning_steps, technical_indicators),
            "technical_factors": self._extract_technical_factors(technical_indicators),
            "market_context": self._extract_signal_market_context(market_conditions),
            "performance_assessment": self._assess_performance(return_pct, holding_period, signal_type),
            "reasoning_process": reasoning_steps,
            "conclusion": conclusion,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated explanation for {signal_type} exit signal for {ticker}")
        return explanation
    
    def _generate_signal_summary(self, signal_type: str, ticker: str, conclusion: str) -> str:
        """Generate a summary of the signal explanation."""
        # In a real implementation, this would extract or generate a summary from the conclusion
        # For this implementation, we'll create a simple summary
        
        if signal_type.lower() == "buy":
            return f"A buy signal has been generated for {ticker} based on favorable technical indicators, market conditions, and strategy alignment."
        elif signal_type.lower() == "sell":
            return f"A sell signal has been generated for {ticker} based on bearish technical indicators, market conditions, and strategy alignment."
        else:
            return f"A {signal_type} signal has been generated for {ticker} based on current technical indicators, market conditions, and strategy alignment."
    
    def _generate_exit_summary(self, signal_type: str, ticker: str, return_pct: float, conclusion: str) -> str:
        """Generate a summary of the exit signal explanation."""
        # In a real implementation, this would extract or generate a summary from the conclusion
        # For this implementation, we'll create a simple summary
        
        return_desc = "profitable" if return_pct > 0 else "unprofitable"
        return_str = f"{return_pct:.2f}%"
        
        if signal_type.lower() == "take_profit":
            return f"A take profit exit signal has been generated for {ticker}, closing a {return_desc} position with a return of {return_str}."
        elif signal_type.lower() == "stop_loss":
            return f"A stop loss exit signal has been generated for {ticker}, closing an {return_desc} position with a return of {return_str}."
        else:
            return f"An exit signal has been generated for {ticker}, closing a {return_desc} position with a return of {return_str}."
    
    def _extract_signal_reasons(self, reasoning_steps: List[str], technical_indicators: Dict[str, Any]) -> List[str]:
        """Extract key reasons from reasoning steps and technical indicators."""
        # In a real implementation, this would extract actual key reasons from the reasoning
        # For this implementation, we'll create simulated key reasons
        
        # Start with some generic reasons
        key_reasons = [
            "Technical indicators have reached signal thresholds",
            "Price action confirms the signal direction",
            "Signal aligns with the strategy's entry/exit criteria"
        ]
        
        # Add some reasons based on technical indicators if available
        for indicator, value in technical_indicators.items():
            if isinstance(value, (int, float)):
                key_reasons.append(f"{indicator.upper()} is at {value}, indicating a potential signal")
            elif isinstance(value, str):
                key_reasons.append(f"{indicator.upper()} shows {value}")
            
            # Limit to 5 reasons
            if len(key_reasons) >= 5:
                break
        
        return key_reasons
    
    def _extract_technical_factors(self, technical_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical factors from technical indicators."""
        # In a real implementation, this would extract actual technical factors
        # For this implementation, we'll create a simulated technical factors summary
        
        # Categorize indicators
        trend_indicators = {}
        momentum_indicators = {}
        volatility_indicators = {}
        other_indicators = {}
        
        # Common trend indicators
        trend_indicator_names = ["sma", "ema", "macd", "adx", "ichimoku", "parabolic_sar"]
        # Common momentum indicators
        momentum_indicator_names = ["rsi", "stochastic", "cci", "williams_r", "mfi", "tsi"]
        # Common volatility indicators
        volatility_indicator_names = ["bollinger", "atr", "keltner", "vix", "standard_deviation"]
        
        for indicator, value in technical_indicators.items():
            indicator_lower = indicator.lower()
            
            if any(name in indicator_lower for name in trend_indicator_names):
                trend_indicators[indicator] = value
            elif any(name in indicator_lower for name in momentum_indicator_names):
                momentum_indicators[indicator] = value
            elif any(name in indicator_lower for name in volatility_indicator_names):
                volatility_indicators[indicator] = value
            else:
                other_indicators[indicator] = value
        
        # Create summary
        technical_factors = {
            "trend_indicators": trend_indicators,
            "momentum_indicators": momentum_indicators,
            "volatility_indicators": volatility_indicators,
            "other_indicators": other_indicators,
            "overall_trend": "bullish" if len(trend_indicators) > 0 else "neutral",
            "overall_momentum": "positive" if len(momentum_indicators) > 0 else "neutral",
            "overall_volatility": "moderate" if len(volatility_indicators) > 0 else "unknown"
        }
        
        return technical_factors
    
    def _extract_signal_market_context(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market context from market conditions."""
        # In a real implementation, this would extract actual market context
        # For this implementation, we'll create a simulated market context
        
        # Extract market regime if available
        market_regime = market_conditions.get("regime", "unknown")
        
        # Extract volatility if available
        volatility = market_conditions.get("volatility", "unknown")
        
        # Extract correlation if available
        correlation = market_conditions.get("correlation", "unknown")
        
        market_context = {
            "market_regime": market_regime,
            "volatility": volatility,
            "correlation": correlation,
            "description": f"The market is currently in a {market_regime} regime with {volatility} volatility."
        }
        
        return market_context
    
    def _extract_signal_risk(self, strategy: Dict[str, Any], market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk assessment from strategy and market conditions."""
        # In a real implementation, this would extract actual risk assessment
        # For this implementation, we'll create a simulated risk assessment
        
        # Extract risk management parameters from strategy if available
        stop_loss = strategy.get("risk_management", {}).get("stop_loss", "unknown")
        take_profit = strategy.get("risk_management", {}).get("take_profit", "unknown")
        
        # Extract volatility from market conditions if available
        volatility = market_conditions.get("volatility", "moderate")
        
        # Determine risk level based on volatility
        risk_level = "moderate"
        
        if volatility == "high":
            risk_level = "high"
        elif volatility == "low":
            risk_level = "low"
        
        # Calculate risk-reward ratio if stop loss and take profit are available
        risk_reward_ratio = "unknown"
        
        if stop_loss != "unknown" and take_profit != "unknown":
            try:
                sl_value = float(stop_loss.split()[0])
                tp_value = float(take_profit.split()[0])
                
                risk_reward_ratio = f"{tp_value / sl_value:.2f}"
            except:
                # If conversion fails, keep default value
                pass
        
        risk_assessment = {
            "risk_level": risk_level,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward_ratio,
            "description": f"This trade has a {risk_level} risk level with a risk-reward ratio of {risk_reward_ratio}."
        }
        
        return risk_assessment
    
    def _assess_performance(self, return_pct: float, holding_period: int, signal_type: str) -> Dict[str, Any]:
        """Assess the performance of a trade."""
        # In a real implementation, this would provide an actual performance assessment
        # For this implementation, we'll create a simulated assessment
        
        # Determine performance rating
        if return_pct > 5:
            rating = "excellent"
        elif return_pct > 2:
            rating = "good"
        elif return_pct > 0:
            rating = "satisfactory"
        elif return_pct > -2:
            rating = "poor"
        else:
            rating = "very poor"
        
        # Determine if the exit was optimal
        optimal_exit = "unknown"
        
        if signal_type.lower() == "take_profit" and return_pct > 0:
            optimal_exit = "yes"
        elif signal_type.lower() == "stop_loss" and return_pct < 0:
            optimal_exit = "yes"
        else:
            optimal_exit = "no"
        
        # Calculate annualized return
        annualized_return = return_pct * (252 / holding_period) if holding_period > 0 else 0
        
        performance_assessment = {
            "return_pct": return_pct,
            "annualized_return": annualized_return,
            "holding_period": holding_period,
            "rating": rating,
            "optimal_exit": optimal_exit,
            "description": f"The trade achieved a {return_pct:.2f}% return over {holding_period} periods, "
                          f"which is considered {rating}. The exit was {optimal_exit} optimal."
        }
        
        return performance_assessment

class StrategyReasoningAndExplanation:
    """
    Main class for strategy reasoning and explanation using Gemma 3.
    
    This class provides a unified interface for all strategy reasoning and explanation
    capabilities, including strategy explanation and trade signal explanation.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the StrategyReasoningAndExplanation.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyReasoningAndExplanation")
        
        # Create or use provided GemmaCore
        self.gemma_core = gemma_core or GemmaCore()
        
        # Initialize components
        self.strategy_reasoning = StrategyReasoning(self.gemma_core)
        self.trade_explanation = TradeExplanation(self.gemma_core)
        
        self.logger.info("Initialized StrategyReasoningAndExplanation")
    
    def explain_strategy(self, ticker: str, strategy_type: str, strategy: Dict[str, Any],
                       technical_analysis: Dict[str, Any],
                       market_conditions: Dict[str, Any],
                       news_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a detailed explanation for a trading strategy.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        strategy_type : str
            Type of strategy (e.g., "swing", "trend", "mean_reversion").
        strategy : Dict[str, Any]
            Details of the trading strategy.
        technical_analysis : Dict[str, Any]
            Technical analysis for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
        news_analysis : Dict[str, Any], optional
            News analysis for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed explanation of the strategy.
        """
        return self.strategy_reasoning.explain_strategy(
            ticker, strategy_type, strategy, technical_analysis, market_conditions, news_analysis
        )
    
    def compare_strategies(self, ticker: str, strategies: List[Dict[str, Any]],
                         market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple trading strategies for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        strategies : List[Dict[str, Any]]
            List of trading strategies to compare.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        Dict[str, Any]
            Comparison of the strategies.
        """
        return self.strategy_reasoning.compare_strategies(ticker, strategies, market_conditions)
    
    def explain_entry_signal(self, ticker: str, signal_type: str,
                           price_data: pd.DataFrame,
                           technical_indicators: Dict[str, Any],
                           market_conditions: Dict[str, Any],
                           strategy: Dict[str, Any],
                           news_sentiment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a detailed explanation for an entry signal.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        signal_type : str
            Type of signal (e.g., "buy", "sell").
        price_data : pd.DataFrame
            Recent price data for the asset.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
        strategy : Dict[str, Any]
            The trading strategy being used.
        news_sentiment : Dict[str, Any], optional
            News sentiment for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed explanation of the entry signal.
        """
        return self.trade_explanation.explain_entry_signal(
            ticker, signal_type, price_data, technical_indicators, 
            market_conditions, strategy, news_sentiment
        )
    
    def explain_exit_signal(self, ticker: str, signal_type: str,
                          entry_price: float, current_price: float,
                          holding_period: int,
                          technical_indicators: Dict[str, Any],
                          market_conditions: Dict[str, Any],
                          strategy: Dict[str, Any],
                          news_sentiment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a detailed explanation for an exit signal.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        signal_type : str
            Type of signal (e.g., "take_profit", "stop_loss", "exit").
        entry_price : float
            Price at which the position was entered.
        current_price : float
            Current price of the asset.
        holding_period : int
            Number of periods the position has been held.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
        strategy : Dict[str, Any]
            The trading strategy being used.
        news_sentiment : Dict[str, Any], optional
            News sentiment for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed explanation of the exit signal.
        """
        return self.trade_explanation.explain_exit_signal(
            ticker, signal_type, entry_price, current_price, holding_period,
            technical_indicators, market_conditions, strategy, news_sentiment
        )
