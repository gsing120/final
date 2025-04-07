"""
Real-Time Signal Analysis Module for Gemma Advanced Trading System

This module implements real-time signal analysis capabilities using Gemma 3
to provide chain-of-thought explanations for trading signals as they occur,
helping traders understand the reasoning behind each recommendation.
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
from gemma3_integration.architecture_enhanced import GemmaCore, PromptEngine, ChainOfThoughtProcessor
from gemma3_integration.strategy_reasoning_and_explanation import StrategyReasoningAndExplanation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class SignalDetector:
    """
    Detects trading signals in real-time data.
    
    This class provides methods for detecting trading signals based on
    strategy rules and current market data.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the SignalDetector.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.SignalDetector")
        self.gemma_core = gemma_core or GemmaCore()
        
        self.logger.info("Initialized SignalDetector")
    
    def detect_signals(self, ticker: str, strategy: Dict[str, Any],
                     market_data: Dict[str, pd.DataFrame],
                     technical_indicators: Dict[str, Any],
                     market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect trading signals based on strategy rules and current market data.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        strategy : Dict[str, Any]
            Trading strategy to use for signal detection.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset, including price and volume data.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        List[Dict[str, Any]]
            Detected trading signals.
        """
        self.logger.info(f"Detecting signals for {ticker} using {strategy.get('name', 'unknown')} strategy")
        
        # Prepare context for Gemma 3
        context = {
            "ticker": ticker,
            "strategy": strategy,
            "market_data": str(market_data.get("price", pd.DataFrame()).tail(10)),  # Simplified for this implementation
            "technical_indicators": technical_indicators,
            "market_conditions": market_conditions
        }
        
        # Generate prompt for signal detection
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "signal_detection",
            **context
        )
        
        # Get the appropriate model for signal detection
        model = self.gemma_core.model_manager.get_model("signal_detection")
        
        # Generate signal detection using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract signals from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured signals
        # For this implementation, we'll create simulated signals
        
        # Check if we have price data
        price_data = market_data.get("price", None)
        if price_data is None or price_data.empty:
            self.logger.warning(f"No price data available for {ticker}")
            return []
        
        # Get latest price
        latest_price = price_data["close"].iloc[-1] if "close" in price_data.columns else price_data.iloc[-1, 0]
        
        # Determine signal type based on strategy and indicators
        strategy_type = strategy.get("type", "unknown")
        entry_conditions = strategy.get("entry_conditions", [])
        
        # Check if any entry conditions are met
        conditions_met = []
        
        # Simulate condition checking based on technical indicators
        for condition in entry_conditions:
            # Check for trend following conditions
            if "crosses above" in condition.lower() and "ema" in condition.lower():
                if technical_indicators.get("ema_crossover", False):
                    conditions_met.append(condition)
            
            # Check for RSI conditions
            elif "rsi" in condition.lower():
                rsi_value = technical_indicators.get("rsi", 50)
                if "< 30" in condition and rsi_value < 30:
                    conditions_met.append(condition)
                elif "> 70" in condition and rsi_value > 70:
                    conditions_met.append(condition)
            
            # Check for MACD conditions
            elif "macd" in condition.lower() and "crosses" in condition.lower():
                if technical_indicators.get("macd_crossover", False):
                    conditions_met.append(condition)
            
            # Check for Bollinger Band conditions
            elif "bollinger" in condition.lower():
                if technical_indicators.get("price_near_lower_band", False) and "lower" in condition.lower():
                    conditions_met.append(condition)
                elif technical_indicators.get("price_near_upper_band", False) and "upper" in condition.lower():
                    conditions_met.append(condition)
        
        # Determine signal direction
        signal_direction = None
        
        if conditions_met:
            # Check if conditions suggest long or short
            long_indicators = sum(1 for c in conditions_met if "long" in c.lower() or 
                                 ("rsi < 30" in c.lower()) or 
                                 ("lower bollinger" in c.lower()))
            
            short_indicators = sum(1 for c in conditions_met if "short" in c.lower() or 
                                  ("rsi > 70" in c.lower()) or 
                                  ("upper bollinger" in c.lower()))
            
            if long_indicators > short_indicators:
                signal_direction = "buy"
            elif short_indicators > long_indicators:
                signal_direction = "sell"
            else:
                # If equal, use market regime to decide
                market_regime = market_conditions.get("regime", "unknown")
                if market_regime == "bullish" or market_regime == "trending_up":
                    signal_direction = "buy"
                elif market_regime == "bearish" or market_regime == "trending_down":
                    signal_direction = "sell"
        
        # If no direction determined, no signal
        if signal_direction is None:
            return []
        
        # Create signal
        signal = {
            "id": f"SIG-{uuid.uuid4().hex[:8]}",
            "ticker": ticker,
            "strategy_id": strategy.get("id", "unknown"),
            "strategy_name": strategy.get("name", "unknown"),
            "signal_type": signal_direction,
            "price": latest_price,
            "timestamp": datetime.datetime.now().isoformat(),
            "confidence": 0.7,  # Simulated confidence
            "conditions_met": conditions_met,
            "technical_indicators": technical_indicators,
            "market_conditions": market_conditions,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", "")
        }
        
        self.logger.info(f"Detected {signal_direction} signal for {ticker}")
        return [signal]
    
    def detect_exit_signals(self, ticker: str, strategy: Dict[str, Any],
                          position: Dict[str, Any],
                          market_data: Dict[str, pd.DataFrame],
                          technical_indicators: Dict[str, Any],
                          market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect exit signals for an existing position.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        strategy : Dict[str, Any]
            Trading strategy to use for signal detection.
        position : Dict[str, Any]
            Current position details.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset, including price and volume data.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        List[Dict[str, Any]]
            Detected exit signals.
        """
        self.logger.info(f"Detecting exit signals for {ticker} position using {strategy.get('name', 'unknown')} strategy")
        
        # Prepare context for Gemma 3
        context = {
            "ticker": ticker,
            "strategy": strategy,
            "position": position,
            "market_data": str(market_data.get("price", pd.DataFrame()).tail(10)),  # Simplified for this implementation
            "technical_indicators": technical_indicators,
            "market_conditions": market_conditions
        }
        
        # Generate prompt for exit signal detection
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "exit_signal_detection",
            **context
        )
        
        # Get the appropriate model for signal detection
        model = self.gemma_core.model_manager.get_model("signal_detection")
        
        # Generate exit signal detection using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract exit signals from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured signals
        # For this implementation, we'll create simulated exit signals
        
        # Check if we have price data
        price_data = market_data.get("price", None)
        if price_data is None or price_data.empty:
            self.logger.warning(f"No price data available for {ticker}")
            return []
        
        # Get latest price
        latest_price = price_data["close"].iloc[-1] if "close" in price_data.columns else price_data.iloc[-1, 0]
        
        # Get position details
        position_type = position.get("position_type", "unknown")
        entry_price = position.get("entry_price", latest_price)
        
        # Calculate current return
        if position_type == "long":
            current_return = (latest_price - entry_price) / entry_price
        else:  # short
            current_return = (entry_price - latest_price) / entry_price
        
        # Determine exit signal type based on strategy and indicators
        exit_conditions = strategy.get("exit_conditions", [])
        
        # Check if any exit conditions are met
        conditions_met = []
        
        # Simulate condition checking based on technical indicators
        for condition in exit_conditions:
            # Check for trend reversal conditions
            if "crosses below" in condition.lower() and "ema" in condition.lower() and position_type == "long":
                if technical_indicators.get("ema_crossover_down", False):
                    conditions_met.append(condition)
            elif "crosses above" in condition.lower() and "ema" in condition.lower() and position_type == "short":
                if technical_indicators.get("ema_crossover_up", False):
                    conditions_met.append(condition)
            
            # Check for RSI conditions
            elif "rsi" in condition.lower():
                rsi_value = technical_indicators.get("rsi", 50)
                if "crosses above 50" in condition.lower() and position_type == "long" and rsi_value > 50:
                    conditions_met.append(condition)
                elif "crosses below 50" in condition.lower() and position_type == "short" and rsi_value < 50:
                    conditions_met.append(condition)
            
            # Check for MACD conditions
            elif "macd" in condition.lower() and "crosses" in condition.lower():
                if (technical_indicators.get("macd_crossover_down", False) and position_type == "long") or \
                   (technical_indicators.get("macd_crossover_up", False) and position_type == "short"):
                    conditions_met.append(condition)
            
            # Check for Bollinger Band conditions
            elif "bollinger" in condition.lower():
                if (technical_indicators.get("price_reaches_middle_band", False) and 
                    "middle" in condition.lower()):
                    conditions_met.append(condition)
            
            # Check for trailing stop conditions
            elif "trailing stop" in condition.lower():
                atr_value = technical_indicators.get("atr", 1.0)
                atr_multiple = float(condition.split("*")[0].split()[-1]) if "*" in condition else 2.0
                
                if position_type == "long":
                    stop_level = latest_price - atr_value * atr_multiple
                    if entry_price > stop_level:
                        conditions_met.append(condition)
                else:  # short
                    stop_level = latest_price + atr_value * atr_multiple
                    if entry_price < stop_level:
                        conditions_met.append(condition)
        
        # Check for take profit or stop loss
        risk_management = strategy.get("risk_management", {})
        
        # Take profit
        take_profit_pct = risk_management.get("take_profit_pct", 0.1)
        if current_return >= take_profit_pct:
            conditions_met.append("Take profit target reached")
        
        # Stop loss
        stop_loss_pct = risk_management.get("stop_loss_pct", 0.05)
        if current_return <= -stop_loss_pct:
            conditions_met.append("Stop loss level reached")
        
        # If no conditions met, no exit signal
        if not conditions_met:
            return []
        
        # Determine exit signal type
        if "Stop loss" in " ".join(conditions_met):
            exit_type = "stop_loss"
        elif "Take profit" in " ".join(conditions_met):
            exit_type = "take_profit"
        else:
            exit_type = "exit"
        
        # Create exit signal
        exit_signal = {
            "id": f"EXIT-{uuid.uuid4().hex[:8]}",
            "ticker": ticker,
            "strategy_id": strategy.get("id", "unknown"),
            "strategy_name": strategy.get("name", "unknown"),
            "signal_type": exit_type,
            "position_type": position_type,
            "entry_price": entry_price,
            "current_price": latest_price,
            "current_return": current_return,
            "timestamp": datetime.datetime.now().isoformat(),
            "confidence": 0.8,  # Simulated confidence
            "conditions_met": conditions_met,
            "technical_indicators": technical_indicators,
            "market_conditions": market_conditions,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", "")
        }
        
        self.logger.info(f"Detected {exit_type} signal for {ticker} {position_type} position")
        return [exit_signal]

class SignalAnalyzer:
    """
    Analyzes trading signals to provide detailed explanations.
    
    This class provides methods for analyzing trading signals and generating
    detailed explanations with chain-of-thought reasoning.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None,
               strategy_reasoning: Optional[StrategyReasoningAndExplanation] = None):
        """
        Initialize the SignalAnalyzer.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        strategy_reasoning : StrategyReasoningAndExplanation, optional
            Instance of StrategyReasoningAndExplanation for strategy explanations.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.SignalAnalyzer")
        self.gemma_core = gemma_core or GemmaCore()
        self.strategy_reasoning = strategy_reasoning or StrategyReasoningAndExplanation(self.gemma_core)
        
        self.logger.info("Initialized SignalAnalyzer")
    
    def analyze_entry_signal(self, signal: Dict[str, Any], 
                           price_data: pd.DataFrame,
                           strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an entry signal and provide a detailed explanation.
        
        Parameters:
        -----------
        signal : Dict[str, Any]
            Entry signal to analyze.
        price_data : pd.DataFrame
            Recent price data for the asset.
        strategy : Dict[str, Any]
            Trading strategy that generated the signal.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed signal analysis and explanation.
        """
        self.logger.info(f"Analyzing {signal.get('signal_type', 'unknown')} signal for {signal.get('ticker', 'unknown')}")
        
        # Extract signal details
        ticker = signal.get("ticker", "unknown")
        signal_type = signal.get("signal_type", "unknown")
        technical_indicators = signal.get("technical_indicators", {})
        market_conditions = signal.get("market_conditions", {})
        
        # Use strategy reasoning to explain entry signal
        explanation = self.strategy_reasoning.explain_entry_signal(
            ticker=ticker,
            signal_type=signal_type,
            price_data=price_data,
            technical_indicators=technical_indicators,
            market_conditions=market_conditions,
            strategy=strategy
        )
        
        # Combine signal and explanation
        analysis = {
            "signal": signal,
            "explanation": explanation,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed analysis for {signal_type} signal for {ticker}")
        return analysis
    
    def analyze_exit_signal(self, signal: Dict[str, Any], 
                          position: Dict[str, Any],
                          strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an exit signal and provide a detailed explanation.
        
        Parameters:
        -----------
        signal : Dict[str, Any]
            Exit signal to analyze.
        position : Dict[str, Any]
            Position details.
        strategy : Dict[str, Any]
            Trading strategy that generated the signal.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed signal analysis and explanation.
        """
        self.logger.info(f"Analyzing {signal.get('signal_type', 'unknown')} signal for {signal.get('ticker', 'unknown')}")
        
        # Extract signal details
        ticker = signal.get("ticker", "unknown")
        signal_type = signal.get("signal_type", "unknown")
        technical_indicators = signal.get("technical_indicators", {})
        market_conditions = signal.get("market_conditions", {})
        
        # Extract position details
        entry_price = position.get("entry_price", 0)
        current_price = signal.get("current_price", 0)
        holding_period = position.get("holding_period", 0)
        
        # Use strategy reasoning to explain exit signal
        explanation = self.strategy_reasoning.explain_exit_signal(
            ticker=ticker,
            signal_type=signal_type,
            entry_price=entry_price,
            current_price=current_price,
            holding_period=holding_period,
            technical_indicators=technical_indicators,
            market_conditions=market_conditions,
            strategy=strategy
        )
        
        # Combine signal and explanation
        analysis = {
            "signal": signal,
            "explanation": explanation,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed analysis for {signal_type} signal for {ticker}")
        return analysis
    
    def analyze_signal_quality(self, signal: Dict[str, Any],
                             historical_signals: List[Dict[str, Any]],
                             market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze the quality of a trading signal.
        
        Parameters:
        -----------
        signal : Dict[str, Any]
            Trading signal to analyze.
        historical_signals : List[Dict[str, Any]]
            Historical signals for comparison.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Signal quality analysis.
        """
        self.logger.info(f"Analyzing quality of {signal.get('signal_type', 'unknown')} signal for {signal.get('ticker', 'unknown')}")
        
        # Prepare context for Gemma 3
        context = {
            "signal": signal,
            "historical_signals": historical_signals[:10],  # Limit to 10 for simplicity
            "market_data": str(market_data.get("price", pd.DataFrame()).tail(10))  # Simplified for this implementation
        }
        
        # Generate prompt for signal quality analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "signal_quality_analysis",
            **context
        )
        
        # Get the appropriate model for signal analysis
        model = self.gemma_core.model_manager.get_model("signal_analysis")
        
        # Generate quality analysis using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract quality analysis from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured analysis
        # For this implementation, we'll create a simulated quality analysis
        
        # Calculate signal quality metrics
        
        # 1. Historical accuracy
        if historical_signals:
            similar_signals = [s for s in historical_signals if s.get("signal_type") == signal.get("signal_type")]
            
            if similar_signals:
                # Assume we have outcome data in historical signals
                successful_signals = sum(1 for s in similar_signals if s.get("outcome", {}).get("successful", False))
                historical_accuracy = successful_signals / len(similar_signals)
            else:
                historical_accuracy = None
        else:
            historical_accuracy = None
        
        # 2. Confidence score
        confidence_score = signal.get("confidence", 0.5)
        
        # 3. Conditions strength
        conditions_met = signal.get("conditions_met", [])
        conditions_strength = min(1.0, len(conditions_met) / 3)  # Normalize to 0-1
        
        # 4. Market alignment
        market_regime = signal.get("market_conditions", {}).get("regime", "unknown")
        signal_type = signal.get("signal_type", "unknown")
        
        if (market_regime in ["bullish", "trending_up"] and signal_type == "buy") or \
           (market_regime in ["bearish", "trending_down"] and signal_type == "sell"):
            market_alignment = "strong"
        elif market_regime in ["neutral", "sideways"]:
            market_alignment = "neutral"
        else:
            market_alignment = "weak"
        
        # Calculate overall quality score
        quality_components = [
            confidence_score,
            conditions_strength
        ]
        
        if historical_accuracy is not None:
            quality_components.append(historical_accuracy)
        
        if market_alignment == "strong":
            quality_components.append(0.8)
        elif market_alignment == "neutral":
            quality_components.append(0.5)
        else:
            quality_components.append(0.2)
        
        overall_quality = sum(quality_components) / len(quality_components)
        
        # Determine quality rating
        if overall_quality >= 0.8:
            quality_rating = "excellent"
        elif overall_quality >= 0.6:
            quality_rating = "good"
        elif overall_quality >= 0.4:
            quality_rating = "moderate"
        else:
            quality_rating = "poor"
        
        # Create quality analysis
        quality_analysis = {
            "signal_id": signal.get("id", "unknown"),
            "ticker": signal.get("ticker", "unknown"),
            "signal_type": signal.get("signal_type", "unknown"),
            "historical_accuracy": historical_accuracy,
            "confidence_score": confidence_score,
            "conditions_strength": conditions_strength,
            "market_alignment": market_alignment,
            "overall_quality": overall_quality,
            "quality_rating": quality_rating,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed quality analysis for signal: {quality_rating}")
        return quality_analysis
    
    def compare_signals(self, signals: List[Dict[str, Any]],
                      market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple trading signals to determine the best one.
        
        Parameters:
        -----------
        signals : List[Dict[str, Any]]
            List of trading signals to compare.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        Dict[str, Any]
            Signal comparison results.
        """
        if not signals:
            return {
                "error": "No signals provided for comparison",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        self.logger.info(f"Comparing {len(signals)} signals")
        
        # Prepare context for Gemma 3
        context = {
            "signals": signals,
            "market_conditions": market_conditions
        }
        
        # Generate prompt for signal comparison
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "signal_comparison",
            **context
        )
        
        # Get the appropriate model for signal comparison
        model = self.gemma_core.model_manager.get_model("signal_comparison")
        
        # Generate comparison using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract comparison from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured comparison
        # For this implementation, we'll create a simulated comparison
        
        # Evaluate each signal
        signal_evaluations = []
        
        for signal in signals:
            # Extract signal details
            signal_id = signal.get("id", "unknown")
            ticker = signal.get("ticker", "unknown")
            signal_type = signal.get("signal_type", "unknown")
            confidence = signal.get("confidence", 0.5)
            conditions_met = signal.get("conditions_met", [])
            
            # Calculate a score for this signal
            score_components = [
                confidence,
                min(1.0, len(conditions_met) / 3)  # Normalize to 0-1
            ]
            
            # Check market alignment
            market_regime = market_conditions.get("regime", "unknown")
            
            if (market_regime in ["bullish", "trending_up"] and signal_type == "buy") or \
               (market_regime in ["bearish", "trending_down"] and signal_type == "sell"):
                score_components.append(0.8)
            elif market_regime in ["neutral", "sideways"]:
                score_components.append(0.5)
            else:
                score_components.append(0.2)
            
            # Calculate overall score
            overall_score = sum(score_components) / len(score_components)
            
            # Create evaluation
            evaluation = {
                "signal_id": signal_id,
                "ticker": ticker,
                "signal_type": signal_type,
                "confidence": confidence,
                "conditions_met_count": len(conditions_met),
                "market_alignment": "aligned" if score_components[-1] >= 0.5 else "misaligned",
                "overall_score": overall_score
            }
            
            signal_evaluations.append(evaluation)
        
        # Sort evaluations by overall score
        signal_evaluations.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Determine best signal
        best_signal = signal_evaluations[0] if signal_evaluations else None
        
        # Create comparison result
        comparison = {
            "signal_evaluations": signal_evaluations,
            "best_signal": best_signal,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed signal comparison, best signal: {best_signal['ticker'] if best_signal else 'none'}")
        return comparison

class RealTimeSignalAnalysis:
    """
    Main class for real-time signal analysis using Gemma 3.
    
    This class provides a unified interface for all real-time signal analysis
    capabilities, including signal detection, analysis, and comparison.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the RealTimeSignalAnalysis.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.RealTimeSignalAnalysis")
        
        # Create or use provided GemmaCore
        self.gemma_core = gemma_core or GemmaCore()
        
        # Initialize components
        self.strategy_reasoning = StrategyReasoningAndExplanation(self.gemma_core)
        self.signal_detector = SignalDetector(self.gemma_core)
        self.signal_analyzer = SignalAnalyzer(self.gemma_core, self.strategy_reasoning)
        
        # Initialize signal history
        self.signal_history = {}
        
        self.logger.info("Initialized RealTimeSignalAnalysis")
    
    def detect_signals(self, ticker: str, strategy: Dict[str, Any],
                     market_data: Dict[str, pd.DataFrame],
                     technical_indicators: Dict[str, Any],
                     market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect trading signals based on strategy rules and current market data.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        strategy : Dict[str, Any]
            Trading strategy to use for signal detection.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset, including price and volume data.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        List[Dict[str, Any]]
            Detected trading signals.
        """
        signals = self.signal_detector.detect_signals(
            ticker, strategy, market_data, technical_indicators, market_conditions
        )
        
        # Store signals in history
        if ticker not in self.signal_history:
            self.signal_history[ticker] = []
        
        self.signal_history[ticker].extend(signals)
        
        return signals
    
    def detect_exit_signals(self, ticker: str, strategy: Dict[str, Any],
                          position: Dict[str, Any],
                          market_data: Dict[str, pd.DataFrame],
                          technical_indicators: Dict[str, Any],
                          market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect exit signals for an existing position.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        strategy : Dict[str, Any]
            Trading strategy to use for signal detection.
        position : Dict[str, Any]
            Current position details.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset, including price and volume data.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        List[Dict[str, Any]]
            Detected exit signals.
        """
        signals = self.signal_detector.detect_exit_signals(
            ticker, strategy, position, market_data, technical_indicators, market_conditions
        )
        
        # Store signals in history
        if ticker not in self.signal_history:
            self.signal_history[ticker] = []
        
        self.signal_history[ticker].extend(signals)
        
        return signals
    
    def analyze_entry_signal(self, signal: Dict[str, Any], 
                           price_data: pd.DataFrame,
                           strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an entry signal and provide a detailed explanation.
        
        Parameters:
        -----------
        signal : Dict[str, Any]
            Entry signal to analyze.
        price_data : pd.DataFrame
            Recent price data for the asset.
        strategy : Dict[str, Any]
            Trading strategy that generated the signal.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed signal analysis and explanation.
        """
        return self.signal_analyzer.analyze_entry_signal(signal, price_data, strategy)
    
    def analyze_exit_signal(self, signal: Dict[str, Any], 
                          position: Dict[str, Any],
                          strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an exit signal and provide a detailed explanation.
        
        Parameters:
        -----------
        signal : Dict[str, Any]
            Exit signal to analyze.
        position : Dict[str, Any]
            Position details.
        strategy : Dict[str, Any]
            Trading strategy that generated the signal.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed signal analysis and explanation.
        """
        return self.signal_analyzer.analyze_exit_signal(signal, position, strategy)
    
    def analyze_signal_quality(self, signal: Dict[str, Any],
                             market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze the quality of a trading signal.
        
        Parameters:
        -----------
        signal : Dict[str, Any]
            Trading signal to analyze.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Signal quality analysis.
        """
        ticker = signal.get("ticker", "unknown")
        historical_signals = self.signal_history.get(ticker, [])
        
        return self.signal_analyzer.analyze_signal_quality(signal, historical_signals, market_data)
    
    def compare_signals(self, signals: List[Dict[str, Any]],
                      market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple trading signals to determine the best one.
        
        Parameters:
        -----------
        signals : List[Dict[str, Any]]
            List of trading signals to compare.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        Dict[str, Any]
            Signal comparison results.
        """
        return self.signal_analyzer.compare_signals(signals, market_conditions)
    
    def get_signal_history(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical signals for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        limit : int, optional
            Maximum number of signals to return. Default is 10.
            
        Returns:
        --------
        List[Dict[str, Any]]
            Historical signals for the ticker.
        """
        signals = self.signal_history.get(ticker, [])
        
        # Sort by timestamp (newest first)
        sorted_signals = sorted(
            signals,
            key=lambda s: s.get("timestamp", ""),
            reverse=True
        )
        
        return sorted_signals[:limit]
    
    def record_signal_outcome(self, signal_id: str, outcome: Dict[str, Any]) -> bool:
        """
        Record the outcome of a signal.
        
        Parameters:
        -----------
        signal_id : str
            ID of the signal.
        outcome : Dict[str, Any]
            Outcome data for the signal.
            
        Returns:
        --------
        bool
            True if the outcome was recorded successfully, False otherwise.
        """
        # Find the signal in history
        for ticker, signals in self.signal_history.items():
            for i, signal in enumerate(signals):
                if signal.get("id") == signal_id:
                    # Add outcome to signal
                    self.signal_history[ticker][i]["outcome"] = outcome
                    self.logger.info(f"Recorded outcome for signal {signal_id}")
                    return True
        
        self.logger.warning(f"Signal {signal_id} not found in history")
        return False
    
    def generate_signal_report(self, ticker: str, 
                             timeframe: str = "1d",
                             limit: int = 5) -> Dict[str, Any]:
        """
        Generate a report of recent signals for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        timeframe : str, optional
            Timeframe for the report. Default is "1d" (1 day).
        limit : int, optional
            Maximum number of signals to include. Default is 5.
            
        Returns:
        --------
        Dict[str, Any]
            Signal report.
        """
        self.logger.info(f"Generating signal report for {ticker} on {timeframe} timeframe")
        
        # Get historical signals
        signals = self.get_signal_history(ticker)
        
        # Filter by timeframe
        cutoff_time = None
        
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        elif timeframe.endswith("d"):
            days = int(timeframe[:-1])
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days)
        elif timeframe.endswith("w"):
            weeks = int(timeframe[:-1])
            cutoff_time = datetime.datetime.now() - datetime.timedelta(weeks=weeks)
        
        if cutoff_time:
            cutoff_time_str = cutoff_time.isoformat()
            filtered_signals = [s for s in signals if s.get("timestamp", "") >= cutoff_time_str]
        else:
            filtered_signals = signals
        
        # Limit number of signals
        limited_signals = filtered_signals[:limit]
        
        # Calculate signal statistics
        buy_signals = [s for s in filtered_signals if s.get("signal_type") == "buy"]
        sell_signals = [s for s in filtered_signals if s.get("signal_type") == "sell"]
        exit_signals = [s for s in filtered_signals if s.get("signal_type") in ["exit", "take_profit", "stop_loss"]]
        
        # Calculate success rate if outcomes are available
        signals_with_outcomes = [s for s in filtered_signals if "outcome" in s]
        successful_signals = [s for s in signals_with_outcomes if s.get("outcome", {}).get("successful", False)]
        
        success_rate = len(successful_signals) / len(signals_with_outcomes) if signals_with_outcomes else None
        
        # Create report
        report = {
            "ticker": ticker,
            "timeframe": timeframe,
            "signal_count": len(filtered_signals),
            "buy_signal_count": len(buy_signals),
            "sell_signal_count": len(sell_signals),
            "exit_signal_count": len(exit_signals),
            "success_rate": success_rate,
            "recent_signals": limited_signals,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated signal report for {ticker}")
        return report
