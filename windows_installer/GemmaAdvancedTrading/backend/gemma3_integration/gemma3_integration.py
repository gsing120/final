"""
Gemma 3 Integration Module for Gemma Advanced Trading System

This module provides a unified interface for all Gemma 3 integration components,
making it easy to use Gemma 3's capabilities throughout the trading system.
"""

import os
import logging
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Import all Gemma 3 integration components
from backend.gemma3_integration.architecture_enhanced import GemmaCore, PromptEngine, ModelManager, ChainOfThoughtProcessor
from backend.gemma3_integration.natural_language_market_analysis_enhanced import NaturalLanguageMarketAnalysis
from backend.gemma3_integration.advanced_mathematical_modeling_enhanced import AdvancedMathematicalModeling
from backend.gemma3_integration.strategy_reasoning_and_explanation import StrategyReasoningAndExplanation
from backend.gemma3_integration.adaptive_learning import AdaptiveLearning
from backend.gemma3_integration.strategy_generation_and_refinement import StrategyGenerationAndRefinement
from backend.gemma3_integration.real_time_signal_analysis import RealTimeSignalAnalysis
from backend.gemma3_integration.central_decision_engine import CentralDecisionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class Gemma3Integration:
    """
    Main class for Gemma 3 integration in the Gemma Advanced Trading System.
    
    This class provides a unified interface for all Gemma 3 integration components,
    making it easy to use Gemma 3's capabilities throughout the trading system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Gemma3Integration.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file. If None, uses default configuration.
        """
        self.logger = logging.getLogger("GemmaTrading.Gemma3Integration")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize GemmaCore
        self.gemma_core = GemmaCore(config=self.config.get("gemma_core", {}))
        
        # Initialize all components
        self.nlp_analysis = NaturalLanguageMarketAnalysis(self.gemma_core)
        self.math_modeling = AdvancedMathematicalModeling(self.gemma_core)
        self.strategy_reasoning = StrategyReasoningAndExplanation(self.gemma_core)
        self.adaptive_learning = AdaptiveLearning(self.gemma_core)
        self.strategy_generation = StrategyGenerationAndRefinement(self.gemma_core)
        self.signal_analysis = RealTimeSignalAnalysis(self.gemma_core)
        self.decision_engine = CentralDecisionEngine(self.gemma_core)
        
        self.logger.info("Initialized Gemma3Integration")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file. If None, uses default configuration.
            
        Returns:
        --------
        Dict[str, Any]
            Configuration dictionary.
        """
        default_config = {
            "gemma_core": {
                "model_config": {
                    "default_model": "gemma-3-8b",
                    "models": {
                        "gemma-3-8b": {
                            "model_path": "gemma-3-8b",
                            "max_tokens": 8192,
                            "temperature": 0.7
                        },
                        "gemma-3-27b": {
                            "model_path": "gemma-3-27b",
                            "max_tokens": 8192,
                            "temperature": 0.7
                        }
                    }
                },
                "prompt_templates_path": "prompts/",
                "chain_of_thought": {
                    "enabled": True,
                    "max_reasoning_steps": 5
                }
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/gemma3_integration.log"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with default config
                self._merge_configs(default_config, config)
                
                self.logger.info(f"Loaded configuration from {config_path}")
                return default_config
            except Exception as e:
                self.logger.error(f"Error loading configuration from {config_path}: {e}")
                self.logger.info("Using default configuration")
                return default_config
        else:
            self.logger.info("Using default configuration")
            return default_config
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
        """
        Merge override configuration into base configuration.
        
        Parameters:
        -----------
        base_config : Dict[str, Any]
            Base configuration to merge into.
        override_config : Dict[str, Any]
            Override configuration to merge from.
        """
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value
    
    # Natural Language Market Analysis methods
    
    def analyze_news_sentiment(self, ticker: str, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment in financial news.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        news_data : List[Dict[str, Any]]
            News data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            News sentiment analysis.
        """
        return self.nlp_analysis.analyze_news_sentiment(ticker, news_data)
    
    def analyze_earnings_report(self, ticker: str, report_text: str) -> Dict[str, Any]:
        """
        Analyze an earnings report.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the company.
        report_text : str
            Text of the earnings report.
            
        Returns:
        --------
        Dict[str, Any]
            Earnings report analysis.
        """
        return self.nlp_analysis.analyze_earnings_report(ticker, report_text)
    
    def analyze_social_sentiment(self, ticker: str, social_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment in social media.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        social_data : Dict[str, Any]
            Social media data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Social sentiment analysis.
        """
        return self.nlp_analysis.analyze_social_sentiment(ticker, social_data)
    
    def generate_market_narrative(self, ticker: str, news_data: List[Dict[str, Any]]) -> str:
        """
        Generate a narrative explaining market movements.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        news_data : List[Dict[str, Any]]
            News data for the asset.
            
        Returns:
        --------
        str
            Market narrative.
        """
        return self.nlp_analysis.generate_market_narrative(ticker, news_data)
    
    # Advanced Mathematical Modeling methods
    
    def detect_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect current market regime.
        
        Parameters:
        -----------
        market_data : Dict[str, Any]
            Market data for analysis.
            
        Returns:
        --------
        Dict[str, Any]
            Market regime analysis.
        """
        return self.math_modeling.detect_market_regime(market_data)
    
    def forecast_volatility(self, ticker: str, price_data: Any) -> Dict[str, Any]:
        """
        Forecast volatility for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        price_data : Any
            Price data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Volatility forecast.
        """
        return self.math_modeling.forecast_volatility(ticker, price_data)
    
    def analyze_correlations(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlations between assets.
        
        Parameters:
        -----------
        market_data : Dict[str, Any]
            Market data for analysis.
            
        Returns:
        --------
        Dict[str, Any]
            Correlation analysis.
        """
        return self.math_modeling.analyze_correlations(market_data)
    
    def forecast_time_series(self, ticker: str, price_data: Any, horizon: int) -> Dict[str, Any]:
        """
        Forecast future values of a time series.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        price_data : Any
            Price data for the asset.
        horizon : int
            Forecast horizon in periods.
            
        Returns:
        --------
        Dict[str, Any]
            Time series forecast.
        """
        return self.math_modeling.forecast_time_series(ticker, price_data, horizon)
    
    # Strategy Reasoning and Explanation methods
    
    def explain_strategy(self, strategy: Dict[str, Any], market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain why a strategy is appropriate for current market conditions.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Trading strategy to explain.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        Dict[str, Any]
            Strategy explanation.
        """
        return self.strategy_reasoning.explain_strategy(strategy, market_conditions)
    
    def explain_entry_signal(self, ticker: str, signal_type: str, price_data: Any,
                           technical_indicators: Dict[str, Any], market_conditions: Dict[str, Any],
                           strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain an entry signal.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        signal_type : str
            Type of signal (e.g., "buy", "sell").
        price_data : Any
            Recent price data for the asset.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
        strategy : Dict[str, Any]
            Trading strategy that generated the signal.
            
        Returns:
        --------
        Dict[str, Any]
            Signal explanation.
        """
        return self.strategy_reasoning.explain_entry_signal(
            ticker, signal_type, price_data, technical_indicators, market_conditions, strategy
        )
    
    def explain_exit_signal(self, ticker: str, signal_type: str, entry_price: float,
                          current_price: float, holding_period: int,
                          technical_indicators: Dict[str, Any], market_conditions: Dict[str, Any],
                          strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain an exit signal.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        signal_type : str
            Type of signal (e.g., "exit", "take_profit", "stop_loss").
        entry_price : float
            Entry price for the position.
        current_price : float
            Current price of the asset.
        holding_period : int
            Holding period in days.
        technical_indicators : Dict[str, Any]
            Technical indicators for the asset.
        market_conditions : Dict[str, Any]
            Current market conditions.
        strategy : Dict[str, Any]
            Trading strategy that generated the signal.
            
        Returns:
        --------
        Dict[str, Any]
            Signal explanation.
        """
        return self.strategy_reasoning.explain_exit_signal(
            ticker, signal_type, entry_price, current_price, holding_period,
            technical_indicators, market_conditions, strategy
        )
    
    def compare_strategies(self, strategies: List[Dict[str, Any]], 
                         market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple strategies for current market conditions.
        
        Parameters:
        -----------
        strategies : List[Dict[str, Any]]
            List of trading strategies to compare.
        market_conditions : Dict[str, Any]
            Current market conditions.
            
        Returns:
        --------
        Dict[str, Any]
            Strategy comparison.
        """
        return self.strategy_reasoning.compare_strategies(strategies, market_conditions)
    
    # Adaptive Learning methods
    
    def analyze_trade(self, ticker: str, entry_price: float, exit_price: float,
                    entry_date: str, exit_date: str, position_type: str,
                    strategy_name: str) -> Dict[str, Any]:
        """
        Analyze a completed trade.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        entry_price : float
            Entry price for the position.
        exit_price : float
            Exit price for the position.
        entry_date : str
            Entry date for the position.
        exit_date : str
            Exit date for the position.
        position_type : str
            Type of position (e.g., "long", "short").
        strategy_name : str
            Name of the strategy used for the trade.
            
        Returns:
        --------
        Dict[str, Any]
            Trade analysis.
        """
        return self.adaptive_learning.analyze_trade(
            ticker, entry_price, exit_price, entry_date, exit_date, position_type, strategy_name
        )
    
    def analyze_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """
        Analyze the performance of a strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Strategy performance analysis.
        """
        return self.adaptive_learning.analyze_strategy_performance(strategy_name)
    
    def optimize_strategy(self, strategy_name: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a strategy based on performance data.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy to optimize.
        performance_data : Dict[str, Any]
            Performance data for the strategy.
            
        Returns:
        --------
        Dict[str, Any]
            Optimization insights.
        """
        return self.adaptive_learning.optimize_strategy(strategy_name, performance_data)
    
    def generate_learning_insights(self, trade: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate learning insights from a trade.
        
        Parameters:
        -----------
        trade : Dict[str, Any]
            Completed trade details.
        market_data : Dict[str, Any]
            Market data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Learning insights.
        """
        return self.adaptive_learning.generate_learning_insights(trade, market_data)
    
    # Strategy Generation and Refinement methods
    
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
        return self.strategy_generation.generate_strategy(
            asset_type, market_conditions, trading_objectives, constraints
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
        return self.strategy_generation.refine_strategy(
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
        return self.strategy_generation.adapt_strategy_to_market_conditions(
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
        return self.strategy_generation.merge_strategies(strategies, merge_method)
    
    # Real-Time Signal Analysis methods
    
    def detect_signals(self, ticker: str, strategy: Dict[str, Any],
                     market_data: Dict[str, Any],
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
        market_data : Dict[str, Any]
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
        return self.signal_analysis.detect_signals(
            ticker, strategy, market_data, technical_indicators, market_conditions
        )
    
    def detect_exit_signals(self, ticker: str, strategy: Dict[str, Any],
                          position: Dict[str, Any],
                          market_data: Dict[str, Any],
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
        market_data : Dict[str, Any]
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
        return self.signal_analysis.detect_exit_signals(
            ticker, strategy, position, market_data, technical_indicators, market_conditions
        )
    
    def analyze_entry_signal(self, signal: Dict[str, Any], 
                           price_data: Any,
                           strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an entry signal and provide a detailed explanation.
        
        Parameters:
        -----------
        signal : Dict[str, Any]
            Entry signal to analyze.
        price_data : Any
            Recent price data for the asset.
        strategy : Dict[str, Any]
            Trading strategy that generated the signal.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed signal analysis and explanation.
        """
        return self.signal_analysis.analyze_entry_signal(signal, price_data, strategy)
    
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
        return self.signal_analysis.analyze_exit_signal(signal, position, strategy)
    
    # Central Decision Engine methods
    
    def generate_trading_recommendation(self, ticker: str, 
                                      market_data: Dict[str, Any],
                                      news_data: List[Dict[str, Any]],
                                      sentiment_data: Dict[str, Any],
                                      trading_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading recommendation for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        market_data : Dict[str, Any]
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
                                   market_data: Dict[str, Any],
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
        market_data : Dict[str, Any]
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
    
    def generate_market_insights(self, market_data: Dict[str, Any],
                               economic_data: Dict[str, Any],
                               news_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate market insights.
        
        Parameters:
        -----------
        market_data : Dict[str, Any]
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
                                         market_data: Dict[str, Any],
                                         trading_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate portfolio recommendations.
        
        Parameters:
        -----------
        portfolio : Dict[str, Any]
            Current portfolio details.
        market_data : Dict[str, Any]
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
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate post-trade analysis.
        
        Parameters:
        -----------
        trade : Dict[str, Any]
            Completed trade details.
        market_data : Dict[str, Any]
            Market data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Post-trade analysis.
        """
        return self.decision_engine.generate_post_trade_analysis(trade, market_data)
    
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
    
    # Utility methods
    
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
        return self.decision_engine.add_strategy(strategy)
    
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
        return self.decision_engine.get_strategy(strategy_id)
    
    def update_market_context(self, ticker: str,
                            market_data: Dict[str, Any],
                            news_data: List[Dict[str, Any]],
                            sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update market context for an asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        market_data : Dict[str, Any]
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
        return self.decision_engine.update_market_context(
            ticker, market_data, news_data, sentiment_data
        )
    
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
        return self.decision_engine.get_market_context(ticker)
    
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
        return self.decision_engine.get_decision_history(limit)
