"""
Central Decision Engine Module for Gemma Advanced Trading System

This module implements the central decision engine that coordinates all Gemma 3
integration components to provide unified trading recommendations and insights.
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
from gemma3_integration.natural_language_market_analysis_enhanced import NaturalLanguageMarketAnalysis
from gemma3_integration.advanced_mathematical_modeling_enhanced import AdvancedMathematicalModeling
from gemma3_integration.strategy_reasoning_and_explanation import StrategyReasoningAndExplanation
from gemma3_integration.adaptive_learning import AdaptiveLearning
from gemma3_integration.strategy_generation_and_refinement import StrategyGenerationAndRefinement
from gemma3_integration.real_time_signal_analysis import RealTimeSignalAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class MarketContext:
    """
    Manages market context information for the central decision engine.
    
    This class provides methods for gathering, updating, and accessing market
    context information, including market conditions, asset data, and news.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None,
               nlp_analysis: Optional[NaturalLanguageMarketAnalysis] = None,
               math_modeling: Optional[AdvancedMathematicalModeling] = None):
        """
        Initialize the MarketContext.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        nlp_analysis : NaturalLanguageMarketAnalysis, optional
            Instance of NaturalLanguageMarketAnalysis for market news analysis.
            If None, creates a new instance.
        math_modeling : AdvancedMathematicalModeling, optional
            Instance of AdvancedMathematicalModeling for market regime detection.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.MarketContext")
        
        # Create or use provided components
        self.gemma_core = gemma_core or GemmaCore()
        self.nlp_analysis = nlp_analysis or NaturalLanguageMarketAnalysis(self.gemma_core)
        self.math_modeling = math_modeling or AdvancedMathematicalModeling(self.gemma_core)
        
        # Initialize market context
        self.global_market_conditions = {}
        self.asset_specific_conditions = {}
        self.market_news = {}
        self.economic_indicators = {}
        self.sentiment_data = {}
        
        # Initialize last update timestamps
        self.last_update_times = {
            "global_market_conditions": None,
            "asset_specific_conditions": {},
            "market_news": {},
            "economic_indicators": None,
            "sentiment_data": {}
        }
        
        self.logger.info("Initialized MarketContext")
    
    def update_global_market_conditions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Update global market conditions.
        
        Parameters:
        -----------
        market_data : Dict[str, pd.DataFrame]
            Market data for major indices and indicators.
            
        Returns:
        --------
        Dict[str, Any]
            Updated global market conditions.
        """
        self.logger.info("Updating global market conditions")
        
        # Use mathematical modeling to detect market regime
        regime_analysis = self.math_modeling.detect_market_regime(market_data)
        
        # Use mathematical modeling to analyze volatility
        volatility_analysis = self.math_modeling.analyze_volatility(market_data)
        
        # Use mathematical modeling to analyze correlations
        correlation_analysis = self.math_modeling.analyze_correlations(market_data)
        
        # Update global market conditions
        self.global_market_conditions = {
            "regime": regime_analysis.get("regime", "unknown"),
            "regime_confidence": regime_analysis.get("confidence", 0.5),
            "regime_description": regime_analysis.get("description", ""),
            "volatility": volatility_analysis.get("volatility_level", "moderate"),
            "volatility_trend": volatility_analysis.get("volatility_trend", "stable"),
            "volatility_percentile": volatility_analysis.get("volatility_percentile", 50),
            "correlations": correlation_analysis.get("correlation_summary", {}),
            "risk_on_off": "risk_on" if regime_analysis.get("regime", "") in ["bullish", "trending_up"] else "risk_off",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Update last update time
        self.last_update_times["global_market_conditions"] = datetime.datetime.now()
        
        self.logger.info(f"Updated global market conditions: {self.global_market_conditions['regime']} regime with {self.global_market_conditions['volatility']} volatility")
        return self.global_market_conditions
    
    def update_asset_specific_conditions(self, ticker: str, 
                                       asset_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Update asset-specific market conditions.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        asset_data : Dict[str, pd.DataFrame]
            Market data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Updated asset-specific conditions.
        """
        self.logger.info(f"Updating market conditions for {ticker}")
        
        # Use mathematical modeling to detect asset-specific regime
        regime_analysis = self.math_modeling.detect_asset_regime(ticker, asset_data)
        
        # Use mathematical modeling to analyze asset volatility
        volatility_analysis = self.math_modeling.analyze_asset_volatility(ticker, asset_data)
        
        # Use mathematical modeling to analyze asset momentum
        momentum_analysis = self.math_modeling.analyze_momentum(ticker, asset_data)
        
        # Use mathematical modeling to detect support/resistance levels
        support_resistance = self.math_modeling.detect_support_resistance(ticker, asset_data)
        
        # Update asset-specific conditions
        asset_conditions = {
            "ticker": ticker,
            "regime": regime_analysis.get("regime", "unknown"),
            "regime_confidence": regime_analysis.get("confidence", 0.5),
            "regime_description": regime_analysis.get("description", ""),
            "volatility": volatility_analysis.get("volatility_level", "moderate"),
            "volatility_trend": volatility_analysis.get("volatility_trend", "stable"),
            "volatility_percentile": volatility_analysis.get("volatility_percentile", 50),
            "momentum": momentum_analysis.get("momentum", "neutral"),
            "momentum_strength": momentum_analysis.get("strength", 0.5),
            "support_levels": support_resistance.get("support_levels", []),
            "resistance_levels": support_resistance.get("resistance_levels", []),
            "relative_strength": "strong" if momentum_analysis.get("relative_strength", 0.5) > 0.6 else "weak",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Store in asset-specific conditions
        self.asset_specific_conditions[ticker] = asset_conditions
        
        # Update last update time
        self.last_update_times["asset_specific_conditions"][ticker] = datetime.datetime.now()
        
        self.logger.info(f"Updated conditions for {ticker}: {asset_conditions['regime']} regime with {asset_conditions['momentum']} momentum")
        return asset_conditions
    
    def update_market_news(self, ticker: str, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update market news analysis for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        news_data : List[Dict[str, Any]]
            News data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Updated market news analysis.
        """
        self.logger.info(f"Updating market news for {ticker}")
        
        # Use NLP analysis to analyze news sentiment
        news_sentiment = self.nlp_analysis.analyze_news_sentiment(ticker, news_data)
        
        # Use NLP analysis to extract key events
        key_events = self.nlp_analysis.extract_key_events(ticker, news_data)
        
        # Use NLP analysis to generate market narrative
        market_narrative = self.nlp_analysis.generate_market_narrative(ticker, news_data)
        
        # Update market news
        news_analysis = {
            "ticker": ticker,
            "sentiment": news_sentiment.get("sentiment", "neutral"),
            "sentiment_score": news_sentiment.get("score", 0.0),
            "key_events": key_events,
            "market_narrative": market_narrative,
            "news_count": len(news_data),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Store in market news
        self.market_news[ticker] = news_analysis
        
        # Update last update time
        self.last_update_times["market_news"][ticker] = datetime.datetime.now()
        
        self.logger.info(f"Updated news for {ticker}: {news_analysis['sentiment']} sentiment with {len(key_events)} key events")
        return news_analysis
    
    def update_economic_indicators(self, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update economic indicators analysis.
        
        Parameters:
        -----------
        economic_data : Dict[str, Any]
            Economic indicator data.
            
        Returns:
        --------
        Dict[str, Any]
            Updated economic indicators analysis.
        """
        self.logger.info("Updating economic indicators")
        
        # Use NLP analysis to analyze economic indicators
        economic_analysis = self.nlp_analysis.analyze_economic_indicators(economic_data)
        
        # Update economic indicators
        self.economic_indicators = {
            "indicators": economic_data,
            "analysis": economic_analysis,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Update last update time
        self.last_update_times["economic_indicators"] = datetime.datetime.now()
        
        self.logger.info("Updated economic indicators")
        return self.economic_indicators
    
    def update_sentiment_data(self, ticker: str, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update sentiment data for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        sentiment_data : Dict[str, Any]
            Sentiment data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Updated sentiment data.
        """
        self.logger.info(f"Updating sentiment data for {ticker}")
        
        # Use NLP analysis to analyze social media sentiment
        social_sentiment = self.nlp_analysis.analyze_social_sentiment(ticker, sentiment_data)
        
        # Update sentiment data
        sentiment_analysis = {
            "ticker": ticker,
            "social_sentiment": social_sentiment.get("sentiment", "neutral"),
            "social_sentiment_score": social_sentiment.get("score", 0.0),
            "social_volume": social_sentiment.get("volume", 0),
            "social_volume_change": social_sentiment.get("volume_change", 0.0),
            "bullish_ratio": social_sentiment.get("bullish_ratio", 0.5),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Store in sentiment data
        self.sentiment_data[ticker] = sentiment_analysis
        
        # Update last update time
        self.last_update_times["sentiment_data"][ticker] = datetime.datetime.now()
        
        self.logger.info(f"Updated sentiment for {ticker}: {sentiment_analysis['social_sentiment']} with score {sentiment_analysis['social_sentiment_score']}")
        return sentiment_analysis
    
    def get_comprehensive_market_context(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive market context for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive market context.
        """
        self.logger.info(f"Getting comprehensive market context for {ticker}")
        
        # Check if we have all required data
        if not self.global_market_conditions:
            self.logger.warning("Global market conditions not available")
        
        if ticker not in self.asset_specific_conditions:
            self.logger.warning(f"Asset-specific conditions for {ticker} not available")
        
        if ticker not in self.market_news:
            self.logger.warning(f"Market news for {ticker} not available")
        
        if not self.economic_indicators:
            self.logger.warning("Economic indicators not available")
        
        if ticker not in self.sentiment_data:
            self.logger.warning(f"Sentiment data for {ticker} not available")
        
        # Create comprehensive market context
        context = {
            "ticker": ticker,
            "global_market_conditions": self.global_market_conditions,
            "asset_specific_conditions": self.asset_specific_conditions.get(ticker, {}),
            "market_news": self.market_news.get(ticker, {}),
            "economic_indicators": self.economic_indicators,
            "sentiment_data": self.sentiment_data.get(ticker, {}),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Retrieved comprehensive market context for {ticker}")
        return context
    
    def is_data_stale(self, data_type: str, ticker: Optional[str] = None, 
                    max_age_minutes: int = 60) -> bool:
        """
        Check if data is stale.
        
        Parameters:
        -----------
        data_type : str
            Type of data to check.
        ticker : str, optional
            Ticker symbol for asset-specific data.
        max_age_minutes : int, optional
            Maximum age in minutes before data is considered stale.
            
        Returns:
        --------
        bool
            True if data is stale, False otherwise.
        """
        now = datetime.datetime.now()
        max_age = datetime.timedelta(minutes=max_age_minutes)
        
        if data_type in ["asset_specific_conditions", "market_news", "sentiment_data"]:
            if ticker is None:
                return True
            
            last_update = self.last_update_times[data_type].get(ticker)
            if last_update is None:
                return True
            
            return (now - last_update) > max_age
        else:
            last_update = self.last_update_times[data_type]
            if last_update is None:
                return True
            
            return (now - last_update) > max_age

class StrategyManager:
    """
    Manages trading strategies for the central decision engine.
    
    This class provides methods for selecting, adapting, and optimizing
    trading strategies based on market conditions and performance data.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None,
               strategy_generation: Optional[StrategyGenerationAndRefinement] = None,
               adaptive_learning: Optional[AdaptiveLearning] = None):
        """
        Initialize the StrategyManager.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        strategy_generation : StrategyGenerationAndRefinement, optional
            Instance of StrategyGenerationAndRefinement for strategy generation.
            If None, creates a new instance.
        adaptive_learning : AdaptiveLearning, optional
            Instance of AdaptiveLearning for strategy optimization.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.StrategyManager")
        
        # Create or use provided components
        self.gemma_core = gemma_core or GemmaCore()
        self.strategy_generation = strategy_generation or StrategyGenerationAndRefinement(self.gemma_core)
        self.adaptive_learning = adaptive_learning or AdaptiveLearning(self.gemma_core)
        
        # Initialize strategy library
        self.strategy_library = {}
        
        # Initialize strategy performance data
        self.strategy_performance = {}
        
        self.logger.info("Initialized StrategyManager")
    
    def add_strategy(self, strategy: Dict[str, Any]) -> str:
        """
        Add a strategy to the library.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to add.
            
        Returns:
        --------
        str
            Strategy ID.
        """
        strategy_id = strategy.get("id", f"S-{uuid.uuid4().hex[:8]}")
        
        if "id" not in strategy:
            strategy["id"] = strategy_id
        
        self.strategy_library[strategy_id] = strategy
        self.logger.info(f"Added strategy {strategy_id} to library: {strategy.get('name', 'unknown')}")
        
        return strategy_id
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a strategy from the library.
        
        Parameters:
        -----------
        strategy_id : str
            ID of the strategy to get.
            
        Returns:
        --------
        Dict[str, Any] or None
            Trading strategy, or None if not found.
        """
        return self.strategy_library.get(strategy_id)
    
    def select_strategy(self, ticker: str, market_context: Dict[str, Any],
                      trading_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best strategy for current market conditions.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        market_context : Dict[str, Any]
            Comprehensive market context.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target).
            
        Returns:
        --------
        Dict[str, Any]
            Selected trading strategy.
        """
        self.logger.info(f"Selecting strategy for {ticker}")
        
        # Extract market conditions
        global_conditions = market_context.get("global_market_conditions", {})
        asset_conditions = market_context.get("asset_specific_conditions", {})
        
        # Combine conditions
        market_conditions = {
            "regime": asset_conditions.get("regime", global_conditions.get("regime", "unknown")),
            "volatility": asset_conditions.get("volatility", global_conditions.get("volatility", "moderate")),
            "momentum": asset_conditions.get("momentum", "neutral"),
            "risk_on_off": global_conditions.get("risk_on_off", "neutral")
        }
        
        # Prepare context for Gemma 3
        context = {
            "ticker": ticker,
            "market_conditions": market_conditions,
            "trading_objectives": trading_objectives,
            "available_strategies": list(self.strategy_library.values())
        }
        
        # Generate prompt for strategy selection
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "strategy_selection",
            **context
        )
        
        # Get the appropriate model for strategy selection
        model = self.gemma_core.model_manager.get_model("strategy_selection")
        
        # Generate strategy selection using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract strategy selection from chain-of-thought result
        # In a real implementation, this would parse the result to extract a structured selection
        # For this implementation, we'll use a simulated selection approach
        
        # Find strategies suitable for current market regime
        suitable_strategies = []
        
        for strategy in self.strategy_library.values():
            # Check if strategy is suitable for current market regime
            suitable_regimes = strategy.get("market_conditions", {}).get("suitable_regimes", [])
            
            if not suitable_regimes or market_conditions["regime"] in suitable_regimes:
                suitable_strategies.append(strategy)
        
        # If no suitable strategies found, generate a new one
        if not suitable_strategies:
            self.logger.info(f"No suitable strategies found for {ticker} in {market_conditions['regime']} regime, generating new strategy")
            
            # Determine asset type from ticker (simplified)
            asset_type = "stock"
            if "USD" in ticker or "/" in ticker:
                asset_type = "forex"
            elif ticker.endswith("USDT") or ticker.endswith("BTC"):
                asset_type = "crypto"
            
            # Generate new strategy
            new_strategy = self.strategy_generation.generate_strategy(
                asset_type=asset_type,
                market_conditions=market_conditions,
                trading_objectives=trading_objectives
            )
            
            # Add to library
            self.add_strategy(new_strategy)
            
            return new_strategy
        
        # Rank strategies by suitability
        ranked_strategies = []
        
        for strategy in suitable_strategies:
            # Calculate suitability score
            score_components = []
            
            # 1. Regime match
            suitable_regimes = strategy.get("market_conditions", {}).get("suitable_regimes", [])
            regime_match = 1.0 if market_conditions["regime"] in suitable_regimes else 0.5
            score_components.append(regime_match)
            
            # 2. Volatility match
            suitable_volatility = strategy.get("market_conditions", {}).get("suitable_volatility", [])
            volatility_match = 1.0 if market_conditions["volatility"] in suitable_volatility else 0.5
            score_components.append(volatility_match)
            
            # 3. Performance (if available)
            strategy_id = strategy.get("id", "")
            if strategy_id in self.strategy_performance:
                performance = self.strategy_performance[strategy_id]
                win_rate = performance.get("win_rate", 0.5)
                score_components.append(win_rate)
            
            # Calculate overall score
            overall_score = sum(score_components) / len(score_components)
            
            ranked_strategies.append((strategy, overall_score))
        
        # Sort by score (descending)
        ranked_strategies.sort(key=lambda x: x[1], reverse=True)
        
        # Select best strategy
        selected_strategy = ranked_strategies[0][0] if ranked_strategies else suitable_strategies[0]
        
        self.logger.info(f"Selected strategy for {ticker}: {selected_strategy.get('name', 'unknown')}")
        return selected_strategy
    
    def adapt_strategy(self, strategy: Dict[str, Any], 
                     market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a strategy to current market conditions.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to adapt.
        market_context : Dict[str, Any]
            Comprehensive market context.
            
        Returns:
        --------
        Dict[str, Any]
            Adapted trading strategy.
        """
        self.logger.info(f"Adapting strategy {strategy.get('name', 'unknown')}")
        
        # Extract market conditions
        global_conditions = market_context.get("global_market_conditions", {})
        asset_conditions = market_context.get("asset_specific_conditions", {})
        
        # Combine conditions
        current_conditions = {
            "regime": asset_conditions.get("regime", global_conditions.get("regime", "unknown")),
            "volatility": asset_conditions.get("volatility", global_conditions.get("volatility", "moderate")),
            "momentum": asset_conditions.get("momentum", "neutral"),
            "risk_on_off": global_conditions.get("risk_on_off", "neutral")
        }
        
        # Get strategy's designed market conditions
        strategy_conditions = {
            "regime": strategy.get("market_conditions", {}).get("suitable_regimes", ["unknown"])[0],
            "volatility": strategy.get("market_conditions", {}).get("suitable_volatility", ["moderate"])[0],
            "momentum": "neutral",
            "risk_on_off": "neutral"
        }
        
        # Check if adaptation is needed
        if current_conditions["regime"] == strategy_conditions["regime"] and \
           current_conditions["volatility"] == strategy_conditions["volatility"]:
            self.logger.info(f"Strategy {strategy.get('name', 'unknown')} already suitable for current conditions")
            return strategy
        
        # Adapt strategy to current conditions
        adapted_strategy = self.strategy_generation.adapt_strategy_to_market_conditions(
            strategy=strategy,
            current_conditions=strategy_conditions,
            target_conditions=current_conditions
        )
        
        # Add to library
        self.add_strategy(adapted_strategy)
        
        self.logger.info(f"Adapted strategy {strategy.get('name', 'unknown')} to {current_conditions['regime']} regime")
        return adapted_strategy
    
    def optimize_strategy(self, strategy_id: str, 
                        performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a strategy based on performance data.
        
        Parameters:
        -----------
        strategy_id : str
            ID of the strategy to optimize.
        performance_data : Dict[str, Any]
            Performance data for the strategy.
            
        Returns:
        --------
        Dict[str, Any]
            Optimized trading strategy.
        """
        self.logger.info(f"Optimizing strategy {strategy_id}")
        
        # Get strategy
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Store performance data
        self.strategy_performance[strategy_id] = performance_data
        
        # Use adaptive learning to optimize strategy
        optimization_insights = self.adaptive_learning.optimize_strategy(
            strategy_name=strategy.get("name", "unknown"),
            performance_data=performance_data
        )
        
        # Use strategy generation to refine strategy
        refined_strategy = self.strategy_generation.refine_strategy(
            strategy=strategy,
            market_conditions=strategy.get("market_conditions", {}),
            performance_data=performance_data,
            refinement_goals=optimization_insights.get("optimization_goals", {})
        )
        
        # Add to library
        self.add_strategy(refined_strategy)
        
        self.logger.info(f"Optimized strategy {strategy.get('name', 'unknown')} to {refined_strategy.get('name', 'unknown')}")
        return refined_strategy
    
    def generate_strategy_report(self, strategy_id: str) -> Dict[str, Any]:
        """
        Generate a report for a strategy.
        
        Parameters:
        -----------
        strategy_id : str
            ID of the strategy.
            
        Returns:
        --------
        Dict[str, Any]
            Strategy report.
        """
        self.logger.info(f"Generating report for strategy {strategy_id}")
        
        # Get strategy
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Get performance data
        performance_data = self.strategy_performance.get(strategy_id, {})
        
        # Create report
        report = {
            "strategy_id": strategy_id,
            "strategy_name": strategy.get("name", "unknown"),
            "strategy_type": strategy.get("type", "unknown"),
            "suitable_regimes": strategy.get("market_conditions", {}).get("suitable_regimes", []),
            "suitable_volatility": strategy.get("market_conditions", {}).get("suitable_volatility", []),
            "performance": performance_data,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated report for strategy {strategy.get('name', 'unknown')}")
        return report

class DecisionEngine:
    """
    Central decision engine for the Gemma Advanced Trading System.
    
    This class coordinates all Gemma 3 integration components to provide
    unified trading recommendations and insights.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the DecisionEngine.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.DecisionEngine")
        
        # Create or use provided GemmaCore
        self.gemma_core = gemma_core or GemmaCore()
        
        # Initialize components
        self.nlp_analysis = NaturalLanguageMarketAnalysis(self.gemma_core)
        self.math_modeling = AdvancedMathematicalModeling(self.gemma_core)
        self.strategy_reasoning = StrategyReasoningAndExplanation(self.gemma_core)
        self.adaptive_learning = AdaptiveLearning(self.gemma_core)
        self.strategy_generation = StrategyGenerationAndRefinement(self.gemma_core)
        self.signal_analysis = RealTimeSignalAnalysis(self.gemma_core)
        
        # Initialize managers
        self.market_context = MarketContext(
            self.gemma_core, self.nlp_analysis, self.math_modeling
        )
        self.strategy_manager = StrategyManager(
            self.gemma_core, self.strategy_generation, self.adaptive_learning
        )
        
        # Initialize decision history
        self.decision_history = []
        
        self.logger.info("Initialized DecisionEngine")
    
    def generate_trading_recommendation(self, ticker: str, 
                                      market_data: Dict[str, pd.DataFrame],
                                      news_data: List[Dict[str, Any]],
                                      sentiment_data: Dict[str, Any],
                                      trading_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading recommendation for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset and global markets.
        news_data : List[Dict[str, Any]]
            News data for the asset.
        sentiment_data : Dict[str, Any]
            Sentiment data for the asset.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target).
            
        Returns:
        --------
        Dict[str, Any]
            Trading recommendation.
        """
        self.logger.info(f"Generating trading recommendation for {ticker}")
        
        # Update market context
        self.market_context.update_global_market_conditions(market_data)
        self.market_context.update_asset_specific_conditions(ticker, market_data)
        self.market_context.update_market_news(ticker, news_data)
        self.market_context.update_sentiment_data(ticker, sentiment_data)
        
        # Get comprehensive market context
        context = self.market_context.get_comprehensive_market_context(ticker)
        
        # Select strategy
        strategy = self.strategy_manager.select_strategy(ticker, context, trading_objectives)
        
        # Adapt strategy if needed
        adapted_strategy = self.strategy_manager.adapt_strategy(strategy, context)
        
        # Extract technical indicators from market data
        technical_indicators = self._extract_technical_indicators(market_data)
        
        # Detect signals
        signals = self.signal_analysis.detect_signals(
            ticker=ticker,
            strategy=adapted_strategy,
            market_data=market_data,
            technical_indicators=technical_indicators,
            market_conditions=context["asset_specific_conditions"]
        )
        
        # If no signals, return no recommendation
        if not signals:
            recommendation = {
                "ticker": ticker,
                "recommendation": "hold",
                "confidence": 0.5,
                "strategy": adapted_strategy,
                "market_context": context,
                "signals": [],
                "reasoning": [
                    "No clear trading signals detected based on current market conditions and selected strategy."
                ],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Add to decision history
            self.decision_history.append(recommendation)
            
            self.logger.info(f"No trading signals for {ticker}, recommending hold")
            return recommendation
        
        # Analyze signals
        analyzed_signals = []
        
        for signal in signals:
            analysis = self.signal_analysis.analyze_entry_signal(
                signal=signal,
                price_data=market_data.get("price", pd.DataFrame()),
                strategy=adapted_strategy
            )
            
            analyzed_signals.append(analysis)
        
        # Evaluate signal quality
        for i, signal in enumerate(signals):
            quality = self.signal_analysis.analyze_signal_quality(
                signal=signal,
                market_data=market_data
            )
            
            analyzed_signals[i]["quality"] = quality
        
        # Compare signals if multiple
        if len(signals) > 1:
            comparison = self.signal_analysis.compare_signals(
                signals=signals,
                market_conditions=context["asset_specific_conditions"]
            )
            
            # Use best signal
            best_signal_id = comparison.get("best_signal", {}).get("signal_id")
            best_signal = next((s for s in signals if s.get("id") == best_signal_id), signals[0])
            
            # Get corresponding analysis
            best_analysis = next((a for a in analyzed_signals if a.get("signal", {}).get("id") == best_signal_id), analyzed_signals[0])
        else:
            best_signal = signals[0]
            best_analysis = analyzed_signals[0]
            comparison = None
        
        # Generate recommendation
        recommendation = {
            "ticker": ticker,
            "recommendation": best_signal.get("signal_type", "hold"),
            "confidence": best_signal.get("confidence", 0.5),
            "price": best_signal.get("price", 0.0),
            "strategy": adapted_strategy,
            "market_context": context,
            "signals": signals,
            "analyzed_signals": analyzed_signals,
            "best_signal": best_signal,
            "best_analysis": best_analysis,
            "signal_comparison": comparison,
            "reasoning": best_analysis.get("explanation", {}).get("reasoning", []),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to decision history
        self.decision_history.append(recommendation)
        
        self.logger.info(f"Generated {recommendation['recommendation']} recommendation for {ticker} with confidence {recommendation['confidence']}")
        return recommendation
    
    def generate_exit_recommendation(self, ticker: str, 
                                   position: Dict[str, Any],
                                   market_data: Dict[str, pd.DataFrame],
                                   news_data: List[Dict[str, Any]],
                                   sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an exit recommendation for an existing position.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        position : Dict[str, Any]
            Current position details.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset and global markets.
        news_data : List[Dict[str, Any]]
            News data for the asset.
        sentiment_data : Dict[str, Any]
            Sentiment data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Exit recommendation.
        """
        self.logger.info(f"Generating exit recommendation for {ticker} position")
        
        # Update market context
        self.market_context.update_global_market_conditions(market_data)
        self.market_context.update_asset_specific_conditions(ticker, market_data)
        self.market_context.update_market_news(ticker, news_data)
        self.market_context.update_sentiment_data(ticker, sentiment_data)
        
        # Get comprehensive market context
        context = self.market_context.get_comprehensive_market_context(ticker)
        
        # Get strategy
        strategy_id = position.get("strategy_id", "")
        strategy = self.strategy_manager.get_strategy(strategy_id)
        
        if not strategy:
            self.logger.warning(f"Strategy {strategy_id} not found, using default exit logic")
            strategy = {
                "id": "default",
                "name": "Default Exit Strategy",
                "exit_conditions": [
                    "Take profit at 2% gain",
                    "Stop loss at 1% loss",
                    "Exit after 5 days"
                ],
                "risk_management": {
                    "take_profit_pct": 0.02,
                    "stop_loss_pct": 0.01
                }
            }
        
        # Extract technical indicators from market data
        technical_indicators = self._extract_technical_indicators(market_data)
        
        # Detect exit signals
        signals = self.signal_analysis.detect_exit_signals(
            ticker=ticker,
            strategy=strategy,
            position=position,
            market_data=market_data,
            technical_indicators=technical_indicators,
            market_conditions=context["asset_specific_conditions"]
        )
        
        # If no signals, return hold recommendation
        if not signals:
            recommendation = {
                "ticker": ticker,
                "recommendation": "hold",
                "confidence": 0.5,
                "position": position,
                "strategy": strategy,
                "market_context": context,
                "signals": [],
                "reasoning": [
                    "No clear exit signals detected based on current market conditions and position status."
                ],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Add to decision history
            self.decision_history.append(recommendation)
            
            self.logger.info(f"No exit signals for {ticker} position, recommending hold")
            return recommendation
        
        # Analyze signals
        analyzed_signals = []
        
        for signal in signals:
            analysis = self.signal_analysis.analyze_exit_signal(
                signal=signal,
                position=position,
                strategy=strategy
            )
            
            analyzed_signals.append(analysis)
        
        # Compare signals if multiple
        if len(signals) > 1:
            comparison = self.signal_analysis.compare_signals(
                signals=signals,
                market_conditions=context["asset_specific_conditions"]
            )
            
            # Use best signal
            best_signal_id = comparison.get("best_signal", {}).get("signal_id")
            best_signal = next((s for s in signals if s.get("id") == best_signal_id), signals[0])
            
            # Get corresponding analysis
            best_analysis = next((a for a in analyzed_signals if a.get("signal", {}).get("id") == best_signal_id), analyzed_signals[0])
        else:
            best_signal = signals[0]
            best_analysis = analyzed_signals[0]
            comparison = None
        
        # Generate recommendation
        recommendation = {
            "ticker": ticker,
            "recommendation": "exit",
            "exit_type": best_signal.get("signal_type", "exit"),
            "confidence": best_signal.get("confidence", 0.5),
            "price": best_signal.get("current_price", 0.0),
            "position": position,
            "strategy": strategy,
            "market_context": context,
            "signals": signals,
            "analyzed_signals": analyzed_signals,
            "best_signal": best_signal,
            "best_analysis": best_analysis,
            "signal_comparison": comparison,
            "reasoning": best_analysis.get("explanation", {}).get("reasoning", []),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to decision history
        self.decision_history.append(recommendation)
        
        self.logger.info(f"Generated {recommendation['exit_type']} recommendation for {ticker} position with confidence {recommendation['confidence']}")
        return recommendation
    
    def generate_market_insights(self, market_data: Dict[str, pd.DataFrame],
                               economic_data: Dict[str, Any],
                               news_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate market insights.
        
        Parameters:
        -----------
        market_data : Dict[str, pd.DataFrame]
            Market data for global markets.
        economic_data : Dict[str, Any]
            Economic indicator data.
        news_data : Dict[str, List[Dict[str, Any]]]
            News data for various markets.
            
        Returns:
        --------
        Dict[str, Any]
            Market insights.
        """
        self.logger.info("Generating market insights")
        
        # Update market context
        self.market_context.update_global_market_conditions(market_data)
        self.market_context.update_economic_indicators(economic_data)
        
        # Update news for major indices
        for ticker, ticker_news in news_data.items():
            self.market_context.update_market_news(ticker, ticker_news)
        
        # Get global market conditions
        global_conditions = self.market_context.global_market_conditions
        
        # Use NLP analysis to generate market narrative
        market_narrative = self.nlp_analysis.generate_global_market_narrative(
            market_data=market_data,
            economic_data=economic_data,
            news_data=news_data
        )
        
        # Use mathematical modeling to identify market themes
        market_themes = self.math_modeling.identify_market_themes(market_data)
        
        # Use mathematical modeling to forecast market direction
        market_forecast = self.math_modeling.forecast_market_direction(market_data)
        
        # Generate insights
        insights = {
            "market_conditions": global_conditions,
            "market_narrative": market_narrative,
            "market_themes": market_themes,
            "market_forecast": market_forecast,
            "economic_indicators": self.market_context.economic_indicators,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info("Generated market insights")
        return insights
    
    def generate_portfolio_recommendations(self, portfolio: Dict[str, Any],
                                         market_data: Dict[str, pd.DataFrame],
                                         trading_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate portfolio recommendations.
        
        Parameters:
        -----------
        portfolio : Dict[str, Any]
            Current portfolio details.
        market_data : Dict[str, pd.DataFrame]
            Market data for assets and global markets.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target).
            
        Returns:
        --------
        Dict[str, Any]
            Portfolio recommendations.
        """
        self.logger.info("Generating portfolio recommendations")
        
        # Update market context
        self.market_context.update_global_market_conditions(market_data)
        
        # Get global market conditions
        global_conditions = self.market_context.global_market_conditions
        
        # Prepare context for Gemma 3
        context = {
            "portfolio": portfolio,
            "market_conditions": global_conditions,
            "trading_objectives": trading_objectives
        }
        
        # Generate prompt for portfolio recommendations
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "portfolio_recommendations",
            **context
        )
        
        # Get the appropriate model for portfolio recommendations
        model = self.gemma_core.model_manager.get_model("portfolio_recommendations")
        
        # Generate portfolio recommendations using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract recommendations from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured recommendations
        # For this implementation, we'll create simulated recommendations
        
        # Analyze portfolio
        positions = portfolio.get("positions", [])
        cash = portfolio.get("cash", 0.0)
        total_value = portfolio.get("total_value", 0.0)
        
        # Calculate current allocation
        allocation = {}
        
        for position in positions:
            ticker = position.get("ticker", "unknown")
            value = position.get("value", 0.0)
            allocation[ticker] = value / total_value
        
        allocation["cash"] = cash / total_value
        
        # Determine market regime
        market_regime = global_conditions.get("regime", "unknown")
        
        # Generate allocation recommendations based on market regime
        recommended_allocation = {}
        
        if market_regime in ["bullish", "trending_up"]:
            # Bullish regime: higher equity allocation
            recommended_allocation = {
                "equity": 0.7,
                "fixed_income": 0.2,
                "cash": 0.1
            }
        elif market_regime in ["bearish", "trending_down"]:
            # Bearish regime: lower equity allocation
            recommended_allocation = {
                "equity": 0.3,
                "fixed_income": 0.5,
                "cash": 0.2
            }
        elif market_regime in ["volatile"]:
            # Volatile regime: more defensive
            recommended_allocation = {
                "equity": 0.4,
                "fixed_income": 0.4,
                "cash": 0.2
            }
        else:
            # Neutral regime: balanced
            recommended_allocation = {
                "equity": 0.6,
                "fixed_income": 0.3,
                "cash": 0.1
            }
        
        # Adjust based on risk tolerance
        risk_tolerance = trading_objectives.get("risk_tolerance", "moderate")
        
        if risk_tolerance == "low":
            recommended_allocation["equity"] *= 0.8
            recommended_allocation["fixed_income"] *= 1.1
            recommended_allocation["cash"] *= 1.5
        elif risk_tolerance == "high":
            recommended_allocation["equity"] *= 1.2
            recommended_allocation["fixed_income"] *= 0.9
            recommended_allocation["cash"] *= 0.5
        
        # Normalize to 100%
        total = sum(recommended_allocation.values())
        for key in recommended_allocation:
            recommended_allocation[key] /= total
        
        # Generate position recommendations
        position_recommendations = []
        
        for position in positions:
            ticker = position.get("ticker", "unknown")
            position_type = position.get("position_type", "long")
            entry_price = position.get("entry_price", 0.0)
            current_price = position.get("current_price", 0.0)
            
            # Calculate return
            if position_type == "long":
                current_return = (current_price - entry_price) / entry_price
            else:  # short
                current_return = (entry_price - current_price) / entry_price
            
            # Determine recommendation
            if current_return > 0.1:  # 10% gain
                action = "take_profit"
                reason = "Position has reached significant profit target"
            elif current_return < -0.05:  # 5% loss
                action = "cut_loss"
                reason = "Position has reached stop loss level"
            elif market_regime in ["bearish", "trending_down"] and position_type == "long":
                action = "reduce"
                reason = "Bearish market conditions suggest reducing long exposure"
            elif market_regime in ["bullish", "trending_up"] and position_type == "short":
                action = "close"
                reason = "Bullish market conditions suggest closing short positions"
            else:
                action = "hold"
                reason = "Position aligns with current market conditions"
            
            position_recommendations.append({
                "ticker": ticker,
                "action": action,
                "reason": reason,
                "current_return": current_return
            })
        
        # Generate new position ideas
        new_position_ideas = []
        
        if market_regime in ["bullish", "trending_up"]:
            new_position_ideas.append({
                "ticker": "SPY",
                "action": "buy",
                "reason": "Broad market exposure in bullish regime"
            })
            new_position_ideas.append({
                "ticker": "QQQ",
                "action": "buy",
                "reason": "Technology exposure in bullish regime"
            })
        elif market_regime in ["bearish", "trending_down"]:
            new_position_ideas.append({
                "ticker": "SH",
                "action": "buy",
                "reason": "Short S&P 500 exposure in bearish regime"
            })
            new_position_ideas.append({
                "ticker": "GLD",
                "action": "buy",
                "reason": "Gold as safe haven in bearish regime"
            })
        elif market_regime in ["volatile"]:
            new_position_ideas.append({
                "ticker": "VXX",
                "action": "buy",
                "reason": "Volatility exposure in volatile regime"
            })
            new_position_ideas.append({
                "ticker": "TLT",
                "action": "buy",
                "reason": "Treasury bonds as safe haven in volatile regime"
            })
        
        # Create recommendations
        recommendations = {
            "current_allocation": allocation,
            "recommended_allocation": recommended_allocation,
            "position_recommendations": position_recommendations,
            "new_position_ideas": new_position_ideas,
            "market_conditions": global_conditions,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info("Generated portfolio recommendations")
        return recommendations
    
    def generate_post_trade_analysis(self, trade: Dict[str, Any],
                                   market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate post-trade analysis.
        
        Parameters:
        -----------
        trade : Dict[str, Any]
            Completed trade details.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Post-trade analysis.
        """
        self.logger.info(f"Generating post-trade analysis for {trade.get('ticker', 'unknown')}")
        
        # Extract trade details
        ticker = trade.get("ticker", "unknown")
        entry_price = trade.get("entry_price", 0.0)
        exit_price = trade.get("exit_price", 0.0)
        entry_date = trade.get("entry_date", "")
        exit_date = trade.get("exit_date", "")
        position_type = trade.get("position_type", "long")
        strategy_id = trade.get("strategy_id", "")
        
        # Calculate trade metrics
        if position_type == "long":
            return_pct = (exit_price - entry_price) / entry_price
        else:  # short
            return_pct = (entry_price - exit_price) / entry_price
        
        # Get strategy
        strategy = self.strategy_manager.get_strategy(strategy_id)
        
        # Use adaptive learning to analyze trade
        trade_analysis = self.adaptive_learning.analyze_trade(
            ticker=ticker,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_date=entry_date,
            exit_date=exit_date,
            position_type=position_type,
            strategy_name=strategy.get("name", "unknown") if strategy else "unknown"
        )
        
        # Use adaptive learning to generate learning insights
        learning_insights = self.adaptive_learning.generate_learning_insights(
            trade=trade,
            market_data=market_data
        )
        
        # Create post-trade analysis
        analysis = {
            "trade": trade,
            "return_pct": return_pct,
            "successful": return_pct > 0,
            "trade_analysis": trade_analysis,
            "learning_insights": learning_insights,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # If strategy exists, optimize it based on this trade
        if strategy:
            # Create simple performance data
            performance_data = {
                "win_rate": 1.0 if return_pct > 0 else 0.0,
                "avg_return": return_pct,
                "max_drawdown": abs(min(0, return_pct)),
                "trade_count": 1
            }
            
            # Optimize strategy
            optimized_strategy = self.strategy_manager.optimize_strategy(
                strategy_id=strategy_id,
                performance_data=performance_data
            )
            
            analysis["optimized_strategy"] = optimized_strategy
        
        self.logger.info(f"Generated post-trade analysis for {ticker} trade with return {return_pct:.2%}")
        return analysis
    
    def generate_decision_explanation(self, decision_id: str) -> Dict[str, Any]:
        """
        Generate a detailed explanation for a trading decision.
        
        Parameters:
        -----------
        decision_id : str
            ID of the decision to explain.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed decision explanation.
        """
        self.logger.info(f"Generating explanation for decision {decision_id}")
        
        # Find decision in history
        decision = None
        
        for d in self.decision_history:
            if d.get("id") == decision_id:
                decision = d
                break
        
        if not decision:
            return {
                "error": f"Decision {decision_id} not found",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Extract decision details
        ticker = decision.get("ticker", "unknown")
        recommendation = decision.get("recommendation", "unknown")
        strategy = decision.get("strategy", {})
        market_context = decision.get("market_context", {})
        signals = decision.get("signals", [])
        
        # Use strategy reasoning to explain decision
        explanation = self.strategy_reasoning.explain_trading_decision(
            ticker=ticker,
            decision_type=recommendation,
            strategy=strategy,
            market_context=market_context,
            signals=signals
        )
        
        # Create detailed explanation
        detailed_explanation = {
            "decision_id": decision_id,
            "ticker": ticker,
            "recommendation": recommendation,
            "confidence": decision.get("confidence", 0.5),
            "strategy": strategy,
            "market_context": market_context,
            "signals": signals,
            "explanation": explanation,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated explanation for {recommendation} decision for {ticker}")
        return detailed_explanation
    
    def _extract_technical_indicators(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Extract technical indicators from market data.
        
        Parameters:
        -----------
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Technical indicators.
        """
        # In a real implementation, this would calculate technical indicators
        # For this implementation, we'll create simulated indicators
        
        # Check if we have price data
        price_data = market_data.get("price", None)
        if price_data is None or price_data.empty:
            return {}
        
        # Get latest prices
        if "close" in price_data.columns:
            latest_close = price_data["close"].iloc[-1]
            latest_open = price_data["open"].iloc[-1] if "open" in price_data.columns else latest_close
            latest_high = price_data["high"].iloc[-1] if "high" in price_data.columns else latest_close
            latest_low = price_data["low"].iloc[-1] if "low" in price_data.columns else latest_close
        else:
            latest_close = price_data.iloc[-1, 0]
            latest_open = latest_close
            latest_high = latest_close
            latest_low = latest_close
        
        # Create simulated indicators
        indicators = {
            "price": latest_close,
            "daily_change_pct": (latest_close - latest_open) / latest_open,
            "rsi": np.random.uniform(30, 70),  # Simulated RSI
            "macd": np.random.uniform(-1, 1),  # Simulated MACD
            "macd_signal": np.random.uniform(-1, 1),  # Simulated MACD Signal
            "macd_histogram": np.random.uniform(-0.5, 0.5),  # Simulated MACD Histogram
            "ema_20": latest_close * (1 + np.random.uniform(-0.05, 0.05)),  # Simulated EMA(20)
            "ema_50": latest_close * (1 + np.random.uniform(-0.1, 0.1)),  # Simulated EMA(50)
            "ema_200": latest_close * (1 + np.random.uniform(-0.2, 0.2)),  # Simulated EMA(200)
            "atr": latest_close * 0.02,  # Simulated ATR (2% of price)
            "bollinger_upper": latest_close * 1.05,  # Simulated Upper Bollinger Band
            "bollinger_middle": latest_close,  # Simulated Middle Bollinger Band
            "bollinger_lower": latest_close * 0.95,  # Simulated Lower Bollinger Band
            "volume": np.random.randint(1000, 10000),  # Simulated Volume
            "volume_sma": np.random.randint(1000, 10000)  # Simulated Volume SMA
        }
        
        # Add derived indicators
        indicators["ema_crossover"] = indicators["ema_20"] > indicators["ema_50"]
        indicators["ema_crossover_up"] = indicators["ema_20"] > indicators["ema_50"] and np.random.random() < 0.5
        indicators["ema_crossover_down"] = indicators["ema_20"] < indicators["ema_50"] and np.random.random() < 0.5
        indicators["macd_crossover"] = indicators["macd"] > indicators["macd_signal"]
        indicators["macd_crossover_up"] = indicators["macd"] > indicators["macd_signal"] and np.random.random() < 0.5
        indicators["macd_crossover_down"] = indicators["macd"] < indicators["macd_signal"] and np.random.random() < 0.5
        indicators["price_near_upper_band"] = latest_close > indicators["bollinger_upper"] * 0.95
        indicators["price_near_lower_band"] = latest_close < indicators["bollinger_lower"] * 1.05
        indicators["price_reaches_middle_band"] = abs(latest_close - indicators["bollinger_middle"]) / indicators["bollinger_middle"] < 0.01
        
        return indicators

class CentralDecisionEngine:
    """
    Main class for the central decision engine of the Gemma Advanced Trading System.
    
    This class provides a unified interface for all decision engine capabilities,
    coordinating all Gemma 3 integration components to provide trading recommendations,
    market insights, and portfolio management.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the CentralDecisionEngine.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.CentralDecisionEngine")
        
        # Create or use provided GemmaCore
        self.gemma_core = gemma_core or GemmaCore()
        
        # Initialize decision engine
        self.decision_engine = DecisionEngine(self.gemma_core)
        
        self.logger.info("Initialized CentralDecisionEngine")
    
    def generate_trading_recommendation(self, ticker: str, 
                                      market_data: Dict[str, pd.DataFrame],
                                      news_data: List[Dict[str, Any]],
                                      sentiment_data: Dict[str, Any],
                                      trading_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading recommendation for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset and global markets.
        news_data : List[Dict[str, Any]]
            News data for the asset.
        sentiment_data : Dict[str, Any]
            Sentiment data for the asset.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target).
            
        Returns:
        --------
        Dict[str, Any]
            Trading recommendation.
        """
        return self.decision_engine.generate_trading_recommendation(
            ticker, market_data, news_data, sentiment_data, trading_objectives
        )
    
    def generate_exit_recommendation(self, ticker: str, 
                                   position: Dict[str, Any],
                                   market_data: Dict[str, pd.DataFrame],
                                   news_data: List[Dict[str, Any]],
                                   sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an exit recommendation for an existing position.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        position : Dict[str, Any]
            Current position details.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset and global markets.
        news_data : List[Dict[str, Any]]
            News data for the asset.
        sentiment_data : Dict[str, Any]
            Sentiment data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Exit recommendation.
        """
        return self.decision_engine.generate_exit_recommendation(
            ticker, position, market_data, news_data, sentiment_data
        )
    
    def generate_market_insights(self, market_data: Dict[str, pd.DataFrame],
                               economic_data: Dict[str, Any],
                               news_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate market insights.
        
        Parameters:
        -----------
        market_data : Dict[str, pd.DataFrame]
            Market data for global markets.
        economic_data : Dict[str, Any]
            Economic indicator data.
        news_data : Dict[str, List[Dict[str, Any]]]
            News data for various markets.
            
        Returns:
        --------
        Dict[str, Any]
            Market insights.
        """
        return self.decision_engine.generate_market_insights(
            market_data, economic_data, news_data
        )
    
    def generate_portfolio_recommendations(self, portfolio: Dict[str, Any],
                                         market_data: Dict[str, pd.DataFrame],
                                         trading_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate portfolio recommendations.
        
        Parameters:
        -----------
        portfolio : Dict[str, Any]
            Current portfolio details.
        market_data : Dict[str, pd.DataFrame]
            Market data for assets and global markets.
        trading_objectives : Dict[str, Any]
            Trading objectives (e.g., risk tolerance, return target).
            
        Returns:
        --------
        Dict[str, Any]
            Portfolio recommendations.
        """
        return self.decision_engine.generate_portfolio_recommendations(
            portfolio, market_data, trading_objectives
        )
    
    def generate_post_trade_analysis(self, trade: Dict[str, Any],
                                   market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate post-trade analysis.
        
        Parameters:
        -----------
        trade : Dict[str, Any]
            Completed trade details.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Post-trade analysis.
        """
        return self.decision_engine.generate_post_trade_analysis(
            trade, market_data
        )
    
    def generate_decision_explanation(self, decision_id: str) -> Dict[str, Any]:
        """
        Generate a detailed explanation for a trading decision.
        
        Parameters:
        -----------
        decision_id : str
            ID of the decision to explain.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed decision explanation.
        """
        return self.decision_engine.generate_decision_explanation(decision_id)
    
    def add_strategy(self, strategy: Dict[str, Any]) -> str:
        """
        Add a strategy to the library.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to add.
            
        Returns:
        --------
        str
            Strategy ID.
        """
        return self.decision_engine.strategy_manager.add_strategy(strategy)
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a strategy from the library.
        
        Parameters:
        -----------
        strategy_id : str
            ID of the strategy to get.
            
        Returns:
        --------
        Dict[str, Any] or None
            Trading strategy, or None if not found.
        """
        return self.decision_engine.strategy_manager.get_strategy(strategy_id)
    
    def generate_strategy_report(self, strategy_id: str) -> Dict[str, Any]:
        """
        Generate a report for a strategy.
        
        Parameters:
        -----------
        strategy_id : str
            ID of the strategy.
            
        Returns:
        --------
        Dict[str, Any]
            Strategy report.
        """
        return self.decision_engine.strategy_manager.generate_strategy_report(strategy_id)
    
    def update_market_context(self, ticker: str,
                            market_data: Dict[str, pd.DataFrame],
                            news_data: List[Dict[str, Any]],
                            sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update market context for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        market_data : Dict[str, pd.DataFrame]
            Market data for the asset and global markets.
        news_data : List[Dict[str, Any]]
            News data for the asset.
        sentiment_data : Dict[str, Any]
            Sentiment data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Updated market context.
        """
        # Update market context
        self.decision_engine.market_context.update_global_market_conditions(market_data)
        self.decision_engine.market_context.update_asset_specific_conditions(ticker, market_data)
        self.decision_engine.market_context.update_market_news(ticker, news_data)
        self.decision_engine.market_context.update_sentiment_data(ticker, sentiment_data)
        
        # Get comprehensive market context
        context = self.decision_engine.market_context.get_comprehensive_market_context(ticker)
        
        return context
    
    def get_market_context(self, ticker: str) -> Dict[str, Any]:
        """
        Get market context for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Market context.
        """
        return self.decision_engine.market_context.get_comprehensive_market_context(ticker)
    
    def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get decision history.
        
        Parameters:
        -----------
        limit : int, optional
            Maximum number of decisions to return. Default is 10.
            
        Returns:
        --------
        List[Dict[str, Any]]
            Decision history.
        """
        return self.decision_engine.decision_history[-limit:]
