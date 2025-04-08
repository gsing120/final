"""
Gemma 3 Integration Architecture for Advanced Trading System

This module defines the core architecture for integrating Gemma 3 AI capabilities
into the trading system. It provides a unified interface for leveraging Gemma 3's
natural language processing, mathematical modeling, and adaptive learning capabilities
across all components of the trading system.

The architecture follows a modular design with the following key components:
1. GemmaCore - Central interface for all Gemma 3 interactions
2. ModelManager - Handles model loading, optimization, and versioning
3. PromptEngine - Manages prompt templates and generation for different use cases
4. ChainOfThoughtProcessor - Implements chain-of-thought reasoning for transparent decision making
5. FeedbackLoop - Manages the learning and adaptation process
6. DataIntegration - Handles integration of various data sources for Gemma 3 analysis
"""

import os
import logging
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class ModelManager:
    """
    Manages Gemma 3 model loading, optimization, and versioning.
    
    This class handles the technical aspects of working with the Gemma 3 model,
    including loading the appropriate model size based on hardware capabilities,
    optimizing for inference speed, and managing model versions.
    """
    
    def __init__(self, model_path: str = None, config: Dict = None):
        """
        Initialize the ModelManager.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the Gemma 3 model files. If None, uses default path.
        config : Dict, optional
            Configuration parameters for model loading and optimization.
        """
        self.logger = logging.getLogger("GemmaTrading.ModelManager")
        self.model_path = model_path or os.environ.get("GEMMA_MODEL_PATH", "./models/gemma3")
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.logger.info(f"Initialized ModelManager with model path: {self.model_path}")
        
    def load_model(self, model_size: str = "7b"):
        """
        Load the Gemma 3 model.
        
        Parameters:
        -----------
        model_size : str
            Size of the model to load ("7b" or "27b").
            
        Returns:
        --------
        bool
            True if model loaded successfully, False otherwise.
        """
        self.logger.info(f"Loading Gemma 3 model (size: {model_size})")
        
        try:
            # Placeholder for actual model loading code
            # In a real implementation, this would use the appropriate library
            # to load the Gemma 3 model based on the specified size
            self.logger.info("Model loaded successfully")
            self.model = f"gemma3-{model_size}"
            self.tokenizer = f"gemma3-tokenizer-{model_size}"
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def optimize_for_inference(self, device: str = "cpu", quantization: bool = True):
        """
        Optimize the model for inference.
        
        Parameters:
        -----------
        device : str
            Device to optimize for ("cpu", "cuda", etc.).
        quantization : bool
            Whether to apply quantization for faster inference.
            
        Returns:
        --------
        bool
            True if optimization was successful, False otherwise.
        """
        self.logger.info(f"Optimizing model for inference on {device} (quantization: {quantization})")
        
        try:
            # Placeholder for actual optimization code
            self.logger.info("Model optimized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to optimize model: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
        --------
        Dict
            Dictionary containing model information.
        """
        return {
            "model_name": self.model,
            "model_path": self.model_path,
            "loaded": self.model is not None,
            "config": self.config
        }


class PromptEngine:
    """
    Manages prompt templates and generation for different use cases.
    
    This class provides a structured way to create effective prompts for
    different trading-related tasks, ensuring consistent and high-quality
    inputs to the Gemma 3 model.
    """
    
    def __init__(self, templates_path: str = None):
        """
        Initialize the PromptEngine.
        
        Parameters:
        -----------
        templates_path : str, optional
            Path to prompt templates. If None, uses default templates.
        """
        self.logger = logging.getLogger("GemmaTrading.PromptEngine")
        self.templates_path = templates_path or "./templates"
        self.templates = self._load_templates()
        self.logger.info(f"Initialized PromptEngine with {len(self.templates)} templates")
    
    def _load_templates(self) -> Dict:
        """
        Load prompt templates from files.
        
        Returns:
        --------
        Dict
            Dictionary of prompt templates.
        """
        # Placeholder for loading actual templates
        # In a real implementation, this would load templates from files
        
        templates = {
            "market_analysis": """
                Analyze the following market data and news for {ticker}:
                
                Technical Data:
                {technical_data}
                
                Recent News:
                {news_data}
                
                Provide a comprehensive analysis including:
                1. Technical analysis insights
                2. News sentiment analysis
                3. Potential catalysts
                4. Market regime assessment
                5. Trading opportunities with risk/reward
                
                Use a chain-of-thought approach to explain your reasoning.
            """,
            
            "strategy_generation": """
                Generate a {strategy_type} trading strategy for {ticker} based on the following data:
                
                Historical Price Data:
                {price_data}
                
                Market Conditions:
                {market_conditions}
                
                Risk Parameters:
                {risk_parameters}
                
                The strategy should include:
                1. Entry conditions with specific indicators and thresholds
                2. Exit conditions including stop loss and take profit levels
                3. Position sizing recommendations
                4. Risk management rules
                5. Expected performance metrics
                
                Explain your reasoning for each component of the strategy.
            """,
            
            "trade_analysis": """
                Analyze the following completed trade:
                
                Trade Details:
                {trade_details}
                
                Market Conditions During Trade:
                {market_conditions}
                
                Original Trade Rationale:
                {trade_rationale}
                
                Provide a detailed post-trade analysis including:
                1. Assessment of the original trade rationale
                2. Execution quality analysis
                3. Identification of what went well and what could be improved
                4. Specific lessons for future trades
                5. Recommendations for strategy adjustments
                
                Use a chain-of-thought approach to explain your reasoning.
            """,
            
            "risk_assessment": """
                Assess the risk for the following portfolio and market conditions:
                
                Portfolio:
                {portfolio_details}
                
                Current Market Conditions:
                {market_conditions}
                
                Economic Indicators:
                {economic_indicators}
                
                Provide a comprehensive risk assessment including:
                1. Portfolio-level risk metrics
                2. Exposure analysis by sector, asset class, and factor
                3. Correlation analysis and diversification assessment
                4. Stress test scenarios and potential drawdowns
                5. Specific risk mitigation recommendations
                
                Explain your reasoning for each risk factor and recommendation.
            """,
            
            "trader_assistance": """
                Answer the following trading question with detailed reasoning:
                
                Question:
                {question}
                
                Relevant Context:
                {context}
                
                Provide a comprehensive answer that:
                1. Directly addresses the question
                2. Explains the underlying concepts
                3. Provides practical implementation advice
                4. Discusses potential risks and considerations
                5. Includes relevant examples or case studies
                
                Use a chain-of-thought approach to explain your reasoning.
            """
        }
        
        return templates
    
    def generate_prompt(self, template_name: str, **kwargs) -> str:
        """
        Generate a prompt using a template.
        
        Parameters:
        -----------
        template_name : str
            Name of the template to use.
        **kwargs
            Variables to fill in the template.
            
        Returns:
        --------
        str
            Generated prompt.
        """
        if template_name not in self.templates:
            self.logger.warning(f"Template '{template_name}' not found, using default")
            return self._generate_default_prompt(**kwargs)
        
        try:
            prompt = self.templates[template_name].format(**kwargs)
            return prompt
        except KeyError as e:
            self.logger.error(f"Missing key in template: {e}")
            return self._generate_default_prompt(**kwargs)
    
    def _generate_default_prompt(self, **kwargs) -> str:
        """
        Generate a default prompt when a template is not available.
        
        Parameters:
        -----------
        **kwargs
            Variables to include in the prompt.
            
        Returns:
        --------
        str
            Generated default prompt.
        """
        prompt_parts = ["Analyze the following trading data:"]
        
        for key, value in kwargs.items():
            prompt_parts.append(f"\n{key.replace('_', ' ').title()}:\n{value}")
        
        prompt_parts.append("\nProvide a detailed analysis with your chain-of-thought reasoning.")
        
        return "\n".join(prompt_parts)
    
    def add_template(self, name: str, template: str) -> bool:
        """
        Add a new prompt template.
        
        Parameters:
        -----------
        name : str
            Name of the template.
        template : str
            Template string.
            
        Returns:
        --------
        bool
            True if template was added successfully, False otherwise.
        """
        if name in self.templates:
            self.logger.warning(f"Overwriting existing template '{name}'")
        
        self.templates[name] = template
        self.logger.info(f"Added template '{name}'")
        return True


class ChainOfThoughtProcessor:
    """
    Implements chain-of-thought reasoning for transparent decision making.
    
    This class handles the generation and processing of chain-of-thought
    reasoning, which provides step-by-step explanations for trading decisions
    and recommendations.
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize the ChainOfThoughtProcessor.
        
        Parameters:
        -----------
        model_manager : ModelManager
            ModelManager instance for accessing the Gemma 3 model.
        """
        self.logger = logging.getLogger("GemmaTrading.ChainOfThoughtProcessor")
        self.model_manager = model_manager
        self.logger.info("Initialized ChainOfThoughtProcessor")
    
    def generate_reasoning(self, prompt: str, max_tokens: int = 1000) -> Dict:
        """
        Generate chain-of-thought reasoning.
        
        Parameters:
        -----------
        prompt : str
            Input prompt for reasoning.
        max_tokens : int
            Maximum number of tokens to generate.
            
        Returns:
        --------
        Dict
            Dictionary containing the reasoning steps and final conclusion.
        """
        self.logger.info("Generating chain-of-thought reasoning")
        
        try:
            # Placeholder for actual reasoning generation
            # In a real implementation, this would use the Gemma 3 model
            # to generate a chain-of-thought reasoning process
            
            # Simulate a chain-of-thought reasoning process
            reasoning = {
                "steps": [
                    "First, I'll analyze the technical indicators to understand the current market context.",
                    "Looking at the price action, I can see a clear uptrend with higher highs and higher lows.",
                    "The RSI is at 65, which indicates bullish momentum but not yet overbought.",
                    "Volume has been increasing on up days, confirming the strength of the trend.",
                    "The 50-day moving average is providing support at $45.20."
                ],
                "conclusion": "Based on the technical analysis, news sentiment, and market context, this appears to be a favorable entry point for a long position with a stop loss at $44.80 and a target of $52.30.",
                "confidence": 0.85,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            return reasoning
        except Exception as e:
            self.logger.error(f"Failed to generate reasoning: {e}")
            return {
                "steps": ["Error generating reasoning"],
                "conclusion": "Unable to provide analysis due to an error",
                "confidence": 0.0,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def extract_decision_factors(self, reasoning: Dict) -> List[Dict]:
        """
        Extract key decision factors from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            List of decision factors with their importance.
        """
        # Placeholder for actual factor extraction
        # In a real implementation, this would analyze the reasoning
        # to identify and extract key decision factors
        
        factors = [
            {"factor": "Uptrend with higher highs and higher lows", "importance": 0.8},
            {"factor": "RSI at 65 (bullish momentum)", "importance": 0.7},
            {"factor": "Increasing volume on up days", "importance": 0.6},
            {"factor": "50-day moving average support at $45.20", "importance": 0.5}
        ]
        
        return factors
    
    def evaluate_reasoning_quality(self, reasoning: Dict) -> Dict:
        """
        Evaluate the quality of chain-of-thought reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        Dict
            Quality metrics for the reasoning.
        """
        # Placeholder for actual quality evaluation
        # In a real implementation, this would assess various aspects
        # of reasoning quality such as logical consistency, evidence usage, etc.
        
        quality = {
            "logical_consistency": 0.9,
            "evidence_usage": 0.85,
            "comprehensiveness": 0.8,
            "clarity": 0.9,
            "overall_quality": 0.86
        }
        
        return quality


class FeedbackLoop:
    """
    Manages the learning and adaptation process.
    
    This class implements the feedback loop for continuous learning and
    improvement of trading strategies based on past performance and outcomes.
    """
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the FeedbackLoop.
        
        Parameters:
        -----------
        storage_path : str, optional
            Path for storing feedback data. If None, uses default path.
        """
        self.logger = logging.getLogger("GemmaTrading.FeedbackLoop")
        self.storage_path = storage_path or "./feedback_data"
        os.makedirs(self.storage_path, exist_ok=True)
        self.logger.info(f"Initialized FeedbackLoop with storage path: {self.storage_path}")
    
    def record_decision(self, decision_id: str, context: Dict, reasoning: Dict, decision: Dict) -> bool:
        """
        Record a trading decision for future evaluation.
        
        Parameters:
        -----------
        decision_id : str
            Unique identifier for the decision.
        context : Dict
            Context in which the decision was made.
        reasoning : Dict
            Chain-of-thought reasoning for the decision.
        decision : Dict
            The actual decision made.
            
        Returns:
        --------
        bool
            True if decision was recorded successfully, False otherwise.
        """
        self.logger.info(f"Recording decision {decision_id}")
        
        try:
            decision_data = {
                "decision_id": decision_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "context": context,
                "reasoning": reasoning,
                "decision": decision,
                "outcomes": None  # To be filled later
            }
            
            # Save decision data to file
            file_path = os.path.join(self.storage_path, f"decision_{decision_id}.json")
            with open(file_path, 'w') as f:
                json.dump(decision_data, f, indent=2)
            
            self.logger.info(f"Decision {decision_id} recorded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to record decision: {e}")
            return False
    
    def record_outcome(self, decision_id: str, outcome: Dict) -> bool:
        """
        Record the outcome of a previously made decision.
        
        Parameters:
        -----------
        decision_id : str
            Unique identifier for the decision.
        outcome : Dict
            Outcome data for the decision.
            
        Returns:
        --------
        bool
            True if outcome was recorded successfully, False otherwise.
        """
        self.logger.info(f"Recording outcome for decision {decision_id}")
        
        try:
            # Load existing decision data
            file_path = os.path.join(self.storage_path, f"decision_{decision_id}.json")
            with open(file_path, 'r') as f:
                decision_data = json.load(f)
            
            # Add outcome data
            decision_data["outcomes"] = outcome
            decision_data["outcome_timestamp"] = datetime.datetime.now().isoformat()
            
            # Save updated decision data
            with open(file_path, 'w') as f:
                json.dump(decision_data, f, indent=2)
            
            self.logger.info(f"Outcome for decision {decision_id} recorded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to record outcome: {e}")
            return False
    
    def analyze_performance(self, time_period: str = "1m") -> Dict:
        """
        Analyze performance of past decisions.
        
        Parameters:
        -----------
        time_period : str
            Time period for analysis ("1d", "1w", "1m", "3m", "6m", "1y").
            
        Returns:
        --------
        Dict
            Performance analysis results.
        """
        self.logger.info(f"Analyzing performance for period: {time_period}")
        
        try:
            # Placeholder for actual performance analysis
            # In a real implementation, this would analyze the outcomes
            # of past decisions to identify patterns and areas for improvement
            
            performance = {
                "total_decisions": 120,
                "successful_decisions": 78,
                "success_rate": 0.65,
                "average_return": 0.028,
                "risk_adjusted_return": 1.2,
                "common_success_factors": [
                    "Strong trend confirmation",
                    "Multiple timeframe alignment",
                    "Volume confirmation",
                    "Clear support/resistance levels"
                ],
                "common_failure_factors": [
                    "Countertrend entries",
                    "Ignoring macro economic factors",
                    "Tight stop losses in volatile markets",
                    "Overtrading during low volatility"
                ],
                "improvement_suggestions": [
                    "Increase weight of volume analysis in decision making",
                    "Add macro economic factor screening",
                    "Adjust stop loss distances based on market volatility",
                    "Implement time-of-day filters for entries"
                ]
            }
            
            return performance
        except Exception as e:
            self.logger.error(f"Failed to analyze performance: {e}")
            return {
                "error": str(e),
                "total_decisions": 0,
                "success_rate": 0.0
            }
    
    def generate_learning_insights(self) -> List[Dict]:
        """
        Generate insights from the learning process.
        
        Returns:
        --------
        List[Dict]
            List of learning insights.
        """
        self.logger.info("Generating learning insights")
        
        try:
            # Placeholder for actual insight generation
            # In a real implementation, this would analyze patterns
            # across multiple decisions and outcomes to generate insights
            
            insights = [
                {
                    "insight": "Entry timing optimization",
                    "description": "Entries made after confirmation of trend on multiple timeframes show 23% higher success rate",
                    "confidence": 0.85,
                    "suggested_action": "Implement multi-timeframe confirmation requirement for entries"
                },
                {
                    "insight": "Stop loss optimization",
                    "description": "Dynamic stop losses based on ATR outperform fixed stop losses by 18% in terms of risk-adjusted return",
                    "confidence": 0.9,
                    "suggested_action": "Replace fixed stop losses with ATR-based stops"
                },
                {
                    "insight": "Sector rotation awareness",
                    "description": "Trades aligned with sector rotation trends have 31% higher average return",
                    "confidence": 0.8,
                    "suggested_action": "Add sector rotation analysis to pre-trade checklist"
                }
            ]
            
            return insights
        except Exception as e:
            self.logger.error(f"Failed to generate learning insights: {e}")
            return [{"error": str(e)}]


class DataIntegration:
    """
    Handles integration of various data sources for Gemma 3 analysis.
    
    This class provides methods for collecting, preprocessing, and integrating
    data from various sources such as market data, news, social media, and
    economic indicators.
    """
    
    def __init__(self, api_keys: Dict = None):
        """
        Initialize the DataIntegration.
        
        Parameters:
        -----------
        api_keys : Dict, optional
            API keys for various data sources.
        """
        self.logger = logging.getLogger("GemmaTrading.DataIntegration")
        self.api_keys = api_keys or {}
        self.logger.info("Initialized DataIntegration")
    
    def fetch_market_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch market data for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        period : str
            Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max").
        interval : str
            Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo").
            
        Returns:
        --------
        pd.DataFrame
            Market data.
        """
        self.logger.info(f"Fetching market data for {ticker} (period: {period}, interval: {interval})")
        
        try:
            # Placeholder for actual data fetching
            # In a real implementation, this would use an API to fetch market data
            
            # Generate synthetic data for demonstration
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
            data = pd.DataFrame({
                'open': np.random.normal(100, 5, 100) * np.linspace(1, 1.2, 100),
                'high': np.random.normal(102, 5, 100) * np.linspace(1, 1.2, 100),
                'low': np.random.normal(98, 5, 100) * np.linspace(1, 1.2, 100),
                'close': np.random.normal(101, 5, 100) * np.linspace(1, 1.2, 100),
                'volume': np.random.normal(1000000, 200000, 100)
            }, index=dates)
            
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
            return pd.DataFrame()
    
    def fetch_news(self, ticker: str, days: int = 7) -> List[Dict]:
        """
        Fetch news articles for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        days : int
            Number of days to look back.
            
        Returns:
        --------
        List[Dict]
            List of news articles.
        """
        self.logger.info(f"Fetching news for {ticker} (days: {days})")
        
        try:
            # Placeholder for actual news fetching
            # In a real implementation, this would use an API to fetch news articles
            
            news = [
                {
                    "title": f"{ticker} Reports Strong Q2 Earnings, Beats Estimates",
                    "date": "2025-04-05",
                    "source": "Financial Times",
                    "url": f"https://example.com/news/{ticker.lower()}-q2-earnings",
                    "summary": f"{ticker} reported Q2 earnings that exceeded analyst expectations, with revenue growing 15% year-over-year."
                },
                {
                    "title": f"{ticker} Announces New Product Line",
                    "date": "2025-04-03",
                    "source": "Bloomberg",
                    "url": f"https://example.com/news/{ticker.lower()}-new-products",
                    "summary": f"{ticker} unveiled a new product line that is expected to open up additional market segments and drive future growth."
                },
                {
                    "title": f"Analyst Upgrades {ticker} to 'Buy'",
                    "date": "2025-04-01",
                    "source": "CNBC",
                    "url": f"https://example.com/news/{ticker.lower()}-upgrade",
                    "summary": f"A leading analyst has upgraded {ticker} from 'Hold' to 'Buy', citing improved growth prospects and competitive positioning."
                }
            ]
            
            return news
        except Exception as e:
            self.logger.error(f"Failed to fetch news: {e}")
            return []
    
    def fetch_social_sentiment(self, ticker: str, days: int = 7) -> Dict:
        """
        Fetch social media sentiment for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        days : int
            Number of days to look back.
            
        Returns:
        --------
        Dict
            Social sentiment data.
        """
        self.logger.info(f"Fetching social sentiment for {ticker} (days: {days})")
        
        try:
            # Placeholder for actual sentiment fetching
            # In a real implementation, this would use an API to fetch social sentiment
            
            sentiment = {
                "overall_sentiment": 0.65,  # -1 to 1 scale
                "sentiment_trend": 0.1,     # Change over period
                "volume": 1250,             # Number of mentions
                "volume_trend": 0.25,       # Change in volume
                "top_topics": [
                    {"topic": "earnings", "count": 320, "sentiment": 0.8},
                    {"topic": "new products", "count": 280, "sentiment": 0.7},
                    {"topic": "competition", "count": 150, "sentiment": 0.2}
                ],
                "sources": {
                    "twitter": {"count": 750, "sentiment": 0.6},
                    "reddit": {"count": 320, "sentiment": 0.7},
                    "stocktwits": {"count": 180, "sentiment": 0.5}
                }
            }
            
            return sentiment
        except Exception as e:
            self.logger.error(f"Failed to fetch social sentiment: {e}")
            return {"overall_sentiment": 0, "error": str(e)}
    
    def fetch_economic_indicators(self) -> Dict:
        """
        Fetch economic indicators.
        
        Returns:
        --------
        Dict
            Economic indicator data.
        """
        self.logger.info("Fetching economic indicators")
        
        try:
            # Placeholder for actual indicator fetching
            # In a real implementation, this would use an API to fetch economic indicators
            
            indicators = {
                "gdp_growth": 2.3,
                "inflation_rate": 2.8,
                "unemployment_rate": 3.6,
                "interest_rate": 4.5,
                "consumer_confidence": 101.2,
                "manufacturing_pmi": 52.8,
                "services_pmi": 54.5,
                "retail_sales_growth": 3.1,
                "housing_starts": 1.45,
                "industrial_production": 2.7
            }
            
            return indicators
        except Exception as e:
            self.logger.error(f"Failed to fetch economic indicators: {e}")
            return {"error": str(e)}
    
    def prepare_data_for_analysis(self, ticker: str) -> Dict:
        """
        Prepare comprehensive data for Gemma 3 analysis.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict
            Comprehensive data for analysis.
        """
        self.logger.info(f"Preparing comprehensive data for {ticker}")
        
        try:
            # Fetch data from various sources
            market_data = self.fetch_market_data(ticker)
            news = self.fetch_news(ticker)
            sentiment = self.fetch_social_sentiment(ticker)
            indicators = self.fetch_economic_indicators()
            
            # Prepare comprehensive data package
            data_package = {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "market_data": market_data.to_dict() if not market_data.empty else {},
                "news": news,
                "social_sentiment": sentiment,
                "economic_indicators": indicators
            }
            
            return data_package
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            return {"error": str(e)}


class GemmaCore:
    """
    Central interface for all Gemma 3 interactions.
    
    This class provides a unified interface for leveraging Gemma 3's capabilities
    across all components of the trading system. It coordinates the interactions
    between the various components of the Gemma 3 integration architecture.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the GemmaCore.
        
        Parameters:
        -----------
        config : Dict, optional
            Configuration parameters.
        """
        self.logger = logging.getLogger("GemmaTrading.GemmaCore")
        self.config = config or {}
        
        # Initialize components
        self.model_manager = ModelManager(
            model_path=self.config.get("model_path"),
            config=self.config.get("model_config")
        )
        self.prompt_engine = PromptEngine(
            templates_path=self.config.get("templates_path")
        )
        self.cot_processor = ChainOfThoughtProcessor(
            model_manager=self.model_manager
        )
        self.feedback_loop = FeedbackLoop(
            storage_path=self.config.get("feedback_storage_path")
        )
        self.data_integration = DataIntegration(
            api_keys=self.config.get("api_keys")
        )
        
        # Load model
        model_size = self.config.get("model_size", "7b")
        self.model_manager.load_model(model_size=model_size)
        self.model_manager.optimize_for_inference(
            device=self.config.get("device", "cpu"),
            quantization=self.config.get("quantization", True)
        )
        
        self.logger.info("Initialized GemmaCore")
    
    def analyze_market(self, ticker: str) -> Dict:
        """
        Perform comprehensive market analysis for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict
            Market analysis results.
        """
        self.logger.info(f"Analyzing market for {ticker}")
        
        try:
            # Prepare data
            data_package = self.data_integration.prepare_data_for_analysis(ticker)
            
            # Generate prompt
            prompt = self.prompt_engine.generate_prompt(
                "market_analysis",
                ticker=ticker,
                technical_data=json.dumps(data_package["market_data"], indent=2),
                news_data=json.dumps(data_package["news"], indent=2)
            )
            
            # Generate reasoning
            reasoning = self.cot_processor.generate_reasoning(prompt)
            
            # Extract decision factors
            factors = self.cot_processor.extract_decision_factors(reasoning)
            
            # Prepare analysis result
            analysis = {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "market_context": {
                    "price_trend": "uptrend",  # Placeholder
                    "volume_trend": "increasing",  # Placeholder
                    "volatility": "moderate",  # Placeholder
                    "market_regime": "bullish"  # Placeholder
                },
                "technical_analysis": {
                    "trend_indicators": {
                        "sma_50_200_cross": "bullish",  # Placeholder
                        "macd": "bullish",  # Placeholder
                        "price_relative_to_moving_averages": "above"  # Placeholder
                    },
                    "momentum_indicators": {
                        "rsi": 65,  # Placeholder
                        "stochastic": "rising",  # Placeholder
                        "cci": 120  # Placeholder
                    },
                    "volatility_indicators": {
                        "bollinger_bands": "expanding",  # Placeholder
                        "atr": 2.5  # Placeholder
                    },
                    "support_resistance": {
                        "support_levels": [45.20, 43.80, 41.50],  # Placeholder
                        "resistance_levels": [48.30, 50.00, 52.30]  # Placeholder
                    }
                },
                "news_sentiment": {
                    "overall_sentiment": data_package["social_sentiment"]["overall_sentiment"],
                    "key_developments": [
                        "Strong Q2 earnings",  # Placeholder
                        "New product line announcement",  # Placeholder
                        "Analyst upgrade"  # Placeholder
                    ]
                },
                "trading_opportunities": {
                    "entry_points": [
                        {
                            "type": "long",
                            "price": 47.50,  # Placeholder
                            "stop_loss": 45.20,  # Placeholder
                            "take_profit": 52.30,  # Placeholder
                            "risk_reward": 2.1,  # Placeholder
                            "confidence": 0.85  # Placeholder
                        }
                    ],
                    "risk_factors": [
                        "Overall market volatility",  # Placeholder
                        "Upcoming economic data releases",  # Placeholder
                        "Sector rotation concerns"  # Placeholder
                    ]
                },
                "reasoning": reasoning,
                "decision_factors": factors
            }
            
            # Record analysis for feedback loop
            analysis_id = f"{ticker}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.feedback_loop.record_decision(
                decision_id=analysis_id,
                context=data_package,
                reasoning=reasoning,
                decision=analysis
            )
            
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze market: {e}")
            return {"error": str(e)}
    
    def generate_trading_strategy(self, ticker: str, strategy_type: str = "swing") -> Dict:
        """
        Generate a trading strategy for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        strategy_type : str
            Type of strategy to generate ("day", "swing", "position", "options").
            
        Returns:
        --------
        Dict
            Generated trading strategy.
        """
        self.logger.info(f"Generating {strategy_type} trading strategy for {ticker}")
        
        try:
            # Prepare data
            data_package = self.data_integration.prepare_data_for_analysis(ticker)
            
            # Generate prompt
            prompt = self.prompt_engine.generate_prompt(
                "strategy_generation",
                ticker=ticker,
                strategy_type=strategy_type,
                price_data=json.dumps(data_package["market_data"], indent=2),
                market_conditions="Bullish trend with moderate volatility",  # Placeholder
                risk_parameters="Max risk per trade: 2%, Risk-reward ratio: 2:1"  # Placeholder
            )
            
            # Generate reasoning
            reasoning = self.cot_processor.generate_reasoning(prompt)
            
            # Prepare strategy result
            strategy = {
                "ticker": ticker,
                "strategy_type": strategy_type,
                "timestamp": datetime.datetime.now().isoformat(),
                "market_context": {
                    "price_trend": "uptrend",  # Placeholder
                    "volatility": "moderate",  # Placeholder
                    "market_regime": "bullish"  # Placeholder
                },
                "entry_conditions": {
                    "primary": [
                        "Price above 50-day moving average",  # Placeholder
                        "RSI between 40 and 60",  # Placeholder
                        "Bullish MACD crossover"  # Placeholder
                    ],
                    "confirmation": [
                        "Increasing volume on up days",  # Placeholder
                        "Price near support level"  # Placeholder
                    ],
                    "filters": [
                        "Avoid entries during first 30 minutes of trading",  # Placeholder
                        "Avoid earnings announcement periods"  # Placeholder
                    ]
                },
                "exit_conditions": {
                    "take_profit": {
                        "method": "Risk multiple",
                        "value": "2R"
                    },
                    "stop_loss": {
                        "method": "ATR multiple",
                        "value": "2 x ATR(14)"
                    },
                    "time_based": {
                        "max_holding_period": "10 trading days"
                    },
                    "technical": [
                        "RSI above 70",  # Placeholder
                        "Price below 20-day moving average"  # Placeholder
                    ]
                },
                "position_sizing": {
                    "method": "Fixed risk percentage",
                    "risk_per_trade": 0.02,
                    "max_position_size": 0.1
                },
                "risk_management": {
                    "max_open_positions": 5,
                    "max_correlated_positions": 2,
                    "max_sector_exposure": 0.3,
                    "daily_stop_loss": 0.03
                },
                "expected_performance": {
                    "win_rate": 0.55,
                    "average_win": 0.02,
                    "average_loss": 0.01,
                    "profit_factor": 1.1,
                    "sharpe_ratio": 1.2
                },
                "parameters": {
                    "sma_short_period": 20,
                    "sma_long_period": 50,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "atr_period": 14,
                    "atr_multiple": 2.0
                },
                "reasoning": reasoning
            }
            
            # Record strategy for feedback loop
            strategy_id = f"{ticker}_{strategy_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.feedback_loop.record_decision(
                decision_id=strategy_id,
                context=data_package,
                reasoning=reasoning,
                decision=strategy
            )
            
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to generate strategy: {e}")
            return {"error": str(e)}
    
    def analyze_trade(self, trade_data: Dict) -> Dict:
        """
        Analyze a completed trade.
        
        Parameters:
        -----------
        trade_data : Dict
            Data for the trade to analyze.
            
        Returns:
        --------
        Dict
            Trade analysis results.
        """
        self.logger.info(f"Analyzing trade for {trade_data.get('ticker', 'unknown')}")
        
        try:
            # Generate prompt
            prompt = self.prompt_engine.generate_prompt(
                "trade_analysis",
                trade_details=json.dumps(trade_data.get("details", {}), indent=2),
                market_conditions=json.dumps(trade_data.get("market_conditions", {}), indent=2),
                trade_rationale=trade_data.get("rationale", "No rationale provided")
            )
            
            # Generate reasoning
            reasoning = self.cot_processor.generate_reasoning(prompt)
            
            # Prepare analysis result
            analysis = {
                "trade_id": trade_data.get("id", "unknown"),
                "ticker": trade_data.get("ticker", "unknown"),
                "timestamp": datetime.datetime.now().isoformat(),
                "performance": {
                    "return": trade_data.get("return", 0.0),
                    "expected_return": trade_data.get("expected_return", 0.0),
                    "deviation": trade_data.get("return", 0.0) - trade_data.get("expected_return", 0.0),
                    "risk_adjusted_return": trade_data.get("risk_adjusted_return", 0.0)
                },
                "execution_quality": {
                    "entry_efficiency": 0.85,  # Placeholder
                    "exit_efficiency": 0.75,  # Placeholder
                    "timing_score": 0.8  # Placeholder
                },
                "strengths": [
                    "Proper identification of trend direction",  # Placeholder
                    "Effective use of support level for entry",  # Placeholder
                    "Appropriate position sizing"  # Placeholder
                ],
                "weaknesses": [
                    "Exit was too early, leaving profit on the table",  # Placeholder
                    "Did not account for sector momentum",  # Placeholder
                    "Stop loss was too tight for market volatility"  # Placeholder
                ],
                "lessons": [
                    "Adjust stop loss based on ATR rather than fixed percentage",  # Placeholder
                    "Consider trailing stops for trending markets",  # Placeholder
                    "Include sector analysis in pre-trade checklist"  # Placeholder
                ],
                "strategy_adjustments": [
                    {
                        "parameter": "stop_loss_method",
                        "current": "Fixed percentage",
                        "recommended": "ATR multiple",
                        "reason": "Better adaptation to market volatility"
                    },
                    {
                        "parameter": "exit_strategy",
                        "current": "Fixed target",
                        "recommended": "Trailing stop",
                        "reason": "Capture more profit in trending markets"
                    }
                ],
                "reasoning": reasoning
            }
            
            # Record analysis for feedback loop
            analysis_id = f"trade_analysis_{trade_data.get('id', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))}"
            self.feedback_loop.record_decision(
                decision_id=analysis_id,
                context=trade_data,
                reasoning=reasoning,
                decision=analysis
            )
            
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze trade: {e}")
            return {"error": str(e)}
    
    def assess_risk(self, portfolio: Dict) -> Dict:
        """
        Assess risk for a portfolio.
        
        Parameters:
        -----------
        portfolio : Dict
            Portfolio data.
            
        Returns:
        --------
        Dict
            Risk assessment results.
        """
        self.logger.info("Assessing portfolio risk")
        
        try:
            # Fetch economic indicators
            indicators = self.data_integration.fetch_economic_indicators()
            
            # Generate prompt
            prompt = self.prompt_engine.generate_prompt(
                "risk_assessment",
                portfolio_details=json.dumps(portfolio, indent=2),
                market_conditions="Bullish trend with increasing volatility",  # Placeholder
                economic_indicators=json.dumps(indicators, indent=2)
            )
            
            # Generate reasoning
            reasoning = self.cot_processor.generate_reasoning(prompt)
            
            # Prepare assessment result
            assessment = {
                "timestamp": datetime.datetime.now().isoformat(),
                "portfolio_metrics": {
                    "total_value": portfolio.get("total_value", 0.0),
                    "cash_percentage": portfolio.get("cash_percentage", 0.0),
                    "beta": 1.2,  # Placeholder
                    "sharpe_ratio": 1.1,  # Placeholder
                    "sortino_ratio": 1.3,  # Placeholder
                    "max_drawdown": 0.15,  # Placeholder
                    "var_95": 0.025,  # Placeholder
                    "cvar_95": 0.035  # Placeholder
                },
                "exposure_analysis": {
                    "sector_exposure": {
                        "technology": 0.35,  # Placeholder
                        "healthcare": 0.25,  # Placeholder
                        "financials": 0.15,  # Placeholder
                        "consumer_discretionary": 0.1,  # Placeholder
                        "other": 0.15  # Placeholder
                    },
                    "factor_exposure": {
                        "market": 1.1,  # Placeholder
                        "size": 0.3,  # Placeholder
                        "value": -0.2,  # Placeholder
                        "momentum": 0.5,  # Placeholder
                        "quality": 0.2  # Placeholder
                    },
                    "concentration_risk": {
                        "top_5_positions": 0.4,  # Placeholder
                        "herfindahl_index": 0.12  # Placeholder
                    }
                },
                "correlation_analysis": {
                    "average_correlation": 0.35,  # Placeholder
                    "diversification_score": 0.65,  # Placeholder
                    "highly_correlated_pairs": [
                        {"ticker1": "AAPL", "ticker2": "MSFT", "correlation": 0.85},  # Placeholder
                        {"ticker1": "JPM", "ticker2": "BAC", "correlation": 0.9}  # Placeholder
                    ]
                },
                "stress_test_scenarios": [
                    {
                        "scenario": "Market correction (-10%)",
                        "portfolio_impact": -0.12,  # Placeholder
                        "most_affected": ["AAPL", "AMZN", "TSLA"]  # Placeholder
                    },
                    {
                        "scenario": "Interest rate hike (+0.5%)",
                        "portfolio_impact": -0.08,  # Placeholder
                        "most_affected": ["JPM", "BAC", "INTC"]  # Placeholder
                    },
                    {
                        "scenario": "Tech sector correction (-15%)",
                        "portfolio_impact": -0.09,  # Placeholder
                        "most_affected": ["AAPL", "MSFT", "NVDA"]  # Placeholder
                    }
                ],
                "risk_mitigation_recommendations": [
                    {
                        "recommendation": "Reduce technology sector exposure",
                        "current": 0.35,
                        "target": 0.25,
                        "priority": "high"
                    },
                    {
                        "recommendation": "Increase defensive positions",
                        "current": 0.1,
                        "target": 0.2,
                        "priority": "medium"
                    },
                    {
                        "recommendation": "Add hedging positions",
                        "current": 0.0,
                        "target": 0.05,
                        "priority": "medium"
                    },
                    {
                        "recommendation": "Diversify within sectors",
                        "current": "Low diversification",
                        "target": "Higher diversification",
                        "priority": "medium"
                    }
                ],
                "reasoning": reasoning
            }
            
            # Record assessment for feedback loop
            assessment_id = f"risk_assessment_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.feedback_loop.record_decision(
                decision_id=assessment_id,
                context={"portfolio": portfolio, "indicators": indicators},
                reasoning=reasoning,
                decision=assessment
            )
            
            return assessment
        except Exception as e:
            self.logger.error(f"Failed to assess risk: {e}")
            return {"error": str(e)}
    
    def answer_trading_question(self, question: str, context: Dict = None) -> Dict:
        """
        Answer a trading-related question.
        
        Parameters:
        -----------
        question : str
            The question to answer.
        context : Dict, optional
            Additional context for the question.
            
        Returns:
        --------
        Dict
            Answer and explanation.
        """
        self.logger.info(f"Answering trading question: {question}")
        
        try:
            # Generate prompt
            prompt = self.prompt_engine.generate_prompt(
                "trader_assistance",
                question=question,
                context=json.dumps(context or {}, indent=2)
            )
            
            # Generate reasoning
            reasoning = self.cot_processor.generate_reasoning(prompt)
            
            # Prepare answer result
            answer = {
                "question": question,
                "timestamp": datetime.datetime.now().isoformat(),
                "answer": reasoning["conclusion"],
                "confidence": reasoning["confidence"],
                "explanation": reasoning["steps"],
                "related_concepts": [
                    "Moving averages",  # Placeholder
                    "Trend identification",  # Placeholder
                    "Support and resistance"  # Placeholder
                ],
                "references": [
                    {
                        "title": "Technical Analysis of the Financial Markets",
                        "author": "John J. Murphy",
                        "relevance": "high"
                    },
                    {
                        "title": "Trading in the Zone",
                        "author": "Mark Douglas",
                        "relevance": "medium"
                    }
                ],
                "reasoning": reasoning
            }
            
            # Record answer for feedback loop
            answer_id = f"question_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.feedback_loop.record_decision(
                decision_id=answer_id,
                context={"question": question, "provided_context": context},
                reasoning=reasoning,
                decision=answer
            )
            
            return answer
        except Exception as e:
            self.logger.error(f"Failed to answer question: {e}")
            return {
                "question": question,
                "answer": "I'm unable to answer this question due to an error.",
                "error": str(e)
            }
    
    def review_backtest(self, backtest_results: Dict) -> Dict:
        """
        Review backtest results.
        
        Parameters:
        -----------
        backtest_results : Dict
            Results of a strategy backtest.
            
        Returns:
        --------
        Dict
            Backtest review and recommendations.
        """
        self.logger.info(f"Reviewing backtest for strategy: {backtest_results.get('strategy_name', 'unknown')}")
        
        try:
            # Generate prompt (using default prompt since there's no specific template)
            prompt = self.prompt_engine._generate_default_prompt(
                backtest_results=json.dumps(backtest_results, indent=2)
            )
            
            # Generate reasoning
            reasoning = self.cot_processor.generate_reasoning(prompt)
            
            # Prepare review result
            review = {
                "strategy_name": backtest_results.get("strategy_name", "unknown"),
                "timestamp": datetime.datetime.now().isoformat(),
                "performance_summary": {
                    "total_return": backtest_results.get("total_return", 0.0),
                    "annualized_return": backtest_results.get("annualized_return", 0.0),
                    "sharpe_ratio": backtest_results.get("sharpe_ratio", 0.0),
                    "max_drawdown": backtest_results.get("max_drawdown", 0.0),
                    "win_rate": backtest_results.get("win_rate", 0.0),
                    "profit_factor": backtest_results.get("profit_factor", 0.0)
                },
                "strengths": [
                    "Consistent performance across different market regimes",  # Placeholder
                    "Good risk-adjusted returns",  # Placeholder
                    "Reasonable drawdowns"  # Placeholder
                ],
                "weaknesses": [
                    "Underperformance in low volatility environments",  # Placeholder
                    "Excessive trading frequency increasing costs",  # Placeholder
                    "Poor performance during market corrections"  # Placeholder
                ],
                "market_regime_analysis": {
                    "bull_market": {
                        "return": 0.18,  # Placeholder
                        "sharpe": 1.4,  # Placeholder
                        "assessment": "Strong performance"  # Placeholder
                    },
                    "bear_market": {
                        "return": -0.08,  # Placeholder
                        "sharpe": 0.6,  # Placeholder
                        "assessment": "Adequate protection"  # Placeholder
                    },
                    "sideways_market": {
                        "return": 0.05,  # Placeholder
                        "sharpe": 0.9,  # Placeholder
                        "assessment": "Average performance"  # Placeholder
                    },
                    "high_volatility": {
                        "return": 0.12,  # Placeholder
                        "sharpe": 1.1,  # Placeholder
                        "assessment": "Good performance"  # Placeholder
                    },
                    "low_volatility": {
                        "return": 0.03,  # Placeholder
                        "sharpe": 0.7,  # Placeholder
                        "assessment": "Underperformance"  # Placeholder
                    }
                },
                "optimization_recommendations": [
                    {
                        "parameter": "entry_threshold",
                        "current": 0.5,
                        "recommended": 0.7,
                        "expected_impact": "Reduce false signals by 15%"
                    },
                    {
                        "parameter": "holding_period",
                        "current": 5,
                        "recommended": 8,
                        "expected_impact": "Improve average profit per trade by 10%"
                    },
                    {
                        "parameter": "stop_loss",
                        "current": "Fixed 2%",
                        "recommended": "ATR-based (2.5x)",
                        "expected_impact": "Better adaptation to volatility"
                    }
                ],
                "reasoning": reasoning
            }
            
            # Record review for feedback loop
            review_id = f"backtest_review_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.feedback_loop.record_decision(
                decision_id=review_id,
                context=backtest_results,
                reasoning=reasoning,
                decision=review
            )
            
            return review
        except Exception as e:
            self.logger.error(f"Failed to review backtest: {e}")
            return {"error": str(e)}
    
    def generate_trading_signal(self, ticker: str, data: pd.DataFrame = None) -> Dict:
        """
        Generate a real-time trading signal.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        data : pd.DataFrame, optional
            Market data. If None, fetches the data.
            
        Returns:
        --------
        Dict
            Trading signal with reasoning.
        """
        self.logger.info(f"Generating trading signal for {ticker}")
        
        try:
            # Fetch data if not provided
            if data is None:
                data = self.data_integration.fetch_market_data(ticker)
            
            # Prepare data package
            data_package = {
                "ticker": ticker,
                "market_data": data.to_dict() if not data.empty else {},
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Generate prompt (using default prompt since there's no specific template)
            prompt = self.prompt_engine._generate_default_prompt(
                ticker=ticker,
                market_data=json.dumps(data_package["market_data"], indent=2),
                task="Generate a trading signal with detailed reasoning"
            )
            
            # Generate reasoning
            reasoning = self.cot_processor.generate_reasoning(prompt)
            
            # Extract decision factors
            factors = self.cot_processor.extract_decision_factors(reasoning)
            
            # Determine signal type from reasoning
            signal_type = "buy"  # Placeholder
            if "bearish" in reasoning["conclusion"].lower() or "sell" in reasoning["conclusion"].lower():
                signal_type = "sell"
            elif "neutral" in reasoning["conclusion"].lower() or "hold" in reasoning["conclusion"].lower():
                signal_type = "hold"
            
            # Prepare signal result
            signal = {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "signal_type": signal_type,
                "confidence": reasoning["confidence"],
                "price": {
                    "current": data["close"].iloc[-1] if not data.empty else None,
                    "entry": data["close"].iloc[-1] if not data.empty else None,  # Placeholder
                    "stop_loss": data["close"].iloc[-1] * 0.95 if not data.empty else None,  # Placeholder
                    "take_profit": data["close"].iloc[-1] * 1.05 if not data.empty else None  # Placeholder
                },
                "time_frame": "short_term",  # Placeholder
                "expected_holding_period": "5-10 days",  # Placeholder
                "risk_reward_ratio": 2.0,  # Placeholder
                "technical_factors": {
                    "trend": "uptrend",  # Placeholder
                    "momentum": "positive",  # Placeholder
                    "volatility": "moderate",  # Placeholder
                    "volume": "above_average"  # Placeholder
                },
                "key_indicators": {
                    "moving_averages": "bullish",  # Placeholder
                    "oscillators": "neutral",  # Placeholder
                    "patterns": "cup_and_handle"  # Placeholder
                },
                "reasoning": reasoning,
                "decision_factors": factors
            }
            
            # Record signal for feedback loop
            signal_id = f"{ticker}_signal_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.feedback_loop.record_decision(
                decision_id=signal_id,
                context=data_package,
                reasoning=reasoning,
                decision=signal
            )
            
            return signal
        except Exception as e:
            self.logger.error(f"Failed to generate signal: {e}")
            return {"error": str(e)}
    
    def learn_from_history(self) -> Dict:
        """
        Learn from historical decisions and outcomes.
        
        Returns:
        --------
        Dict
            Learning results and insights.
        """
        self.logger.info("Learning from historical decisions and outcomes")
        
        try:
            # Analyze performance
            performance = self.feedback_loop.analyze_performance()
            
            # Generate insights
            insights = self.feedback_loop.generate_learning_insights()
            
            # Prepare learning result
            learning = {
                "timestamp": datetime.datetime.now().isoformat(),
                "performance_summary": performance,
                "insights": insights,
                "strategy_adjustments": [
                    {
                        "strategy_type": "swing",
                        "parameter": "confirmation_needed",
                        "current": 2,
                        "recommended": 3,
                        "reason": "Reduce false signals in volatile markets"
                    },
                    {
                        "strategy_type": "day",
                        "parameter": "profit_taking",
                        "current": "Fixed target",
                        "recommended": "Scaled exit",
                        "reason": "Improve average profit per trade"
                    }
                ],
                "risk_adjustments": [
                    {
                        "parameter": "position_sizing",
                        "current": "Fixed percentage",
                        "recommended": "Volatility-adjusted",
                        "reason": "Better risk management in changing market conditions"
                    },
                    {
                        "parameter": "max_correlated_positions",
                        "current": 3,
                        "recommended": 2,
                        "reason": "Reduce portfolio concentration risk"
                    }
                ],
                "new_strategies": [
                    {
                        "name": "Volatility breakout",
                        "description": "Enter on breakouts after low volatility periods",
                        "expected_performance": {
                            "win_rate": 0.6,
                            "profit_factor": 1.3
                        }
                    },
                    {
                        "name": "Sector rotation momentum",
                        "description": "Trade stocks in sectors showing relative strength",
                        "expected_performance": {
                            "win_rate": 0.55,
                            "profit_factor": 1.4
                        }
                    }
                ]
            }
            
            return learning
        except Exception as e:
            self.logger.error(f"Failed to learn from history: {e}")
            return {"error": str(e)}
