"""
Enhanced Gemma 3 Integration Architecture for Advanced Trading System

This module defines the comprehensive architecture for integrating Gemma 3 AI capabilities
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
7. StrategyGenerator - Generates and refines trading strategies
8. SignalAnalyzer - Analyzes market data to generate entry/exit signals with explanations
9. TradeAnalyzer - Analyzes executed trades to identify improvements
10. QualitativeAnalyzer - Analyzes news, social media, and analyst reports
11. RiskEvaluator - Evaluates risk factors combining quantitative and qualitative assessments
12. TraderAssistant - Provides Q&A interface for traders
13. BacktestReviewer - Reviews backtest results and provides insights
14. DecisionEngine - Central engine for generating trade decisions
"""

import os
import logging
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path

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
    including loading the appropriate model version, optimizing for different
    use cases, and managing model updates.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ModelManager.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the Gemma 3 model files. If not provided, uses default path.
        """
        self.logger = logging.getLogger("GemmaTrading.ModelManager")
        
        # Set default model path if not provided
        if model_path is None:
            # Use a mock path for development/testing
            self.model_path = os.path.join(os.path.dirname(__file__), "models")
        else:
            self.model_path = model_path
        
        # Create model path if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize model versions dictionary
        self.model_versions = {
            "default": "gemma3-base",
            "finance": "gemma3-finance",
            "math": "gemma3-math",
            "nlp": "gemma3-nlp"
        }
        
        # Initialize loaded models dictionary
        self.loaded_models = {}
        
        self.logger.info(f"Initialized ModelManager with model path: {self.model_path}")
    
    def load_model(self, model_type: str = "default") -> Any:
        """
        Load a Gemma 3 model of the specified type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to load. Options: "default", "finance", "math", "nlp".
            
        Returns:
        --------
        Any
            Loaded model object.
        """
        self.logger.info(f"Loading model of type: {model_type}")
        
        # Check if model is already loaded
        if model_type in self.loaded_models:
            self.logger.info(f"Model {model_type} already loaded")
            return self.loaded_models[model_type]
        
        # Get model version
        model_version = self.model_versions.get(model_type, self.model_versions["default"])
        
        # In a real implementation, this would load the actual model
        # For now, we'll create a mock model object
        model = {
            "name": model_version,
            "type": model_type,
            "loaded_at": datetime.datetime.now().isoformat()
        }
        
        # Store loaded model
        self.loaded_models[model_type] = model
        
        self.logger.info(f"Loaded model: {model_version}")
        return model
    
    def optimize_model(self, model_type: str, optimization_params: Dict[str, Any]) -> Any:
        """
        Optimize a model for specific use cases.
        
        Parameters:
        -----------
        model_type : str
            Type of model to optimize.
        optimization_params : Dict[str, Any]
            Parameters for optimization.
            
        Returns:
        --------
        Any
            Optimized model object.
        """
        self.logger.info(f"Optimizing model {model_type} with params: {optimization_params}")
        
        # Load model if not already loaded
        if model_type not in self.loaded_models:
            self.load_model(model_type)
        
        # Get model
        model = self.loaded_models[model_type]
        
        # In a real implementation, this would apply optimization techniques
        # For now, we'll just update the mock model object
        model["optimized"] = True
        model["optimization_params"] = optimization_params
        model["optimized_at"] = datetime.datetime.now().isoformat()
        
        self.logger.info(f"Optimized model: {model['name']}")
        return model
    
    def update_model(self, model_type: str) -> Any:
        """
        Update a model to the latest version.
        
        Parameters:
        -----------
        model_type : str
            Type of model to update.
            
        Returns:
        --------
        Any
            Updated model object.
        """
        self.logger.info(f"Updating model: {model_type}")
        
        # In a real implementation, this would check for and download updates
        # For now, we'll just update the mock model object
        
        # Unload existing model if loaded
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
        
        # Load model (which will now be the "updated" version)
        model = self.load_model(model_type)
        
        # Update model metadata
        model["updated"] = True
        model["updated_at"] = datetime.datetime.now().isoformat()
        
        self.logger.info(f"Updated model: {model['name']}")
        return model


class PromptEngine:
    """
    Manages prompt templates and generation for different use cases.
    
    This class provides a structured way to generate prompts for various
    trading-related tasks, ensuring consistency and effectiveness in
    interactions with the Gemma 3 model.
    """
    
    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the PromptEngine.
        
        Parameters:
        -----------
        templates_path : str, optional
            Path to prompt templates. If not provided, uses default templates.
        """
        self.logger = logging.getLogger("GemmaTrading.PromptEngine")
        
        # Set default templates path if not provided
        if templates_path is None:
            # Use a mock path for development/testing
            self.templates_path = os.path.join(os.path.dirname(__file__), "templates")
        else:
            self.templates_path = templates_path
        
        # Create templates path if it doesn't exist
        os.makedirs(self.templates_path, exist_ok=True)
        
        # Load default templates
        self.templates = self._load_default_templates()
        
        self.logger.info(f"Initialized PromptEngine with templates path: {self.templates_path}")
    
    def _load_default_templates(self) -> Dict[str, str]:
        """
        Load default prompt templates.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary of template names to template strings.
        """
        # In a real implementation, this would load templates from files
        # For now, we'll define some basic templates inline
        templates = {
            # Market analysis templates
            "market_analysis": "Analyze the current market conditions for {market}. Consider the following data: {data}",
            "technical_analysis": "Perform technical analysis on {ticker} using the following indicators: {indicators}. Data: {data}",
            "sentiment_analysis": "Analyze the sentiment for {ticker} based on the following news and social media data: {data}",
            
            # Strategy templates
            "strategy_generation": "Generate a trading strategy for {ticker} based on the following market conditions: {conditions}",
            "strategy_optimization": "Optimize the following trading strategy for {ticker}: {strategy}. Consider these performance metrics: {metrics}",
            "strategy_comparison": "Compare the following trading strategies for {ticker}: {strategies}. Which one is most suitable for current market conditions: {conditions}?",
            
            # Signal templates
            "signal_analysis": "Analyze the following trading signal for {ticker}: {signal}. Is this a valid signal given current market conditions: {conditions}?",
            "entry_exit_points": "Determine optimal entry and exit points for {ticker} based on the following strategy: {strategy}. Current market conditions: {conditions}",
            
            # Risk templates
            "risk_assessment": "Assess the risk of trading {ticker} with the following strategy: {strategy}. Consider market conditions: {conditions}",
            "position_sizing": "Determine appropriate position size for {ticker} trade with account size {account_size}, risk tolerance {risk_tolerance}, and strategy: {strategy}",
            
            # News analysis templates
            "news_analysis": "Analyze the following news articles for {ticker}: {articles}. What is the likely impact on price?",
            "earnings_report_analysis": "Analyze the following earnings report for {ticker}: {report_text}. What are the key takeaways and likely market reaction?",
            "breaking_news_analysis": "Analyze this breaking news for {ticker}: {news_text}. What is the immediate trading implication?",
            
            # Social media templates
            "social_media_analysis": "Analyze the following social media posts about {ticker}: {posts}. What is the overall sentiment and potential impact?",
            "social_sentiment_query": "What is the current social media sentiment for {ticker}? Consider the query: {query}",
            "social_anomaly_detection": "Detect any anomalies in social media activity for {ticker}. Historical data: {historical_data}. Current data: {current_data}",
            
            # Narrative templates
            "asset_narrative_generation": "Generate a comprehensive market narrative for {ticker} based on the following data: {data}",
            "market_narrative_generation": "Generate a comprehensive market narrative based on the following data: {data}",
            
            # Integration templates
            "integrated_analysis": "Perform an integrated analysis of {ticker} combining quantitative data: {quantitative_data} and qualitative data: {qualitative_data}",
            
            # Conference call templates
            "conference_call_analysis": "Analyze the following earnings call transcript for {ticker}: {transcript_text}. What are the key insights and management tone?",
            
            # Trader assistance templates
            "trading_qa": "Answer the following trading question: {question}. Consider these market conditions: {conditions}",
            "strategy_explanation": "Explain the following trading strategy in simple terms: {strategy}",
            
            # Backtest templates
            "backtest_analysis": "Analyze the results of this backtest for {strategy}: {results}. What improvements can be made?",
            "performance_attribution": "Attribute the performance of {strategy} to different factors based on these results: {results}",
            
            # Decision templates
            "trading_decision": "Make a trading decision for {ticker} based on the following analysis: {analysis}. Current position: {position}",
            "portfolio_allocation": "Determine optimal portfolio allocation given these assets: {assets} and market conditions: {conditions}"
        }
        
        return templates
    
    def add_template(self, name: str, template: str) -> None:
        """
        Add a new prompt template.
        
        Parameters:
        -----------
        name : str
            Name of the template.
        template : str
            Template string with placeholders.
        """
        self.logger.info(f"Adding template: {name}")
        self.templates[name] = template
    
    def get_template(self, name: str) -> str:
        """
        Get a prompt template by name.
        
        Parameters:
        -----------
        name : str
            Name of the template.
            
        Returns:
        --------
        str
            Template string.
        """
        if name not in self.templates:
            self.logger.warning(f"Template not found: {name}. Using default template.")
            return "Analyze the following data: {data}"
        
        return self.templates[name]
    
    def generate_prompt(self, template_name: str, **kwargs) -> str:
        """
        Generate a prompt using a template and provided values.
        
        Parameters:
        -----------
        template_name : str
            Name of the template to use.
        **kwargs : Any
            Values to fill in the template placeholders.
            
        Returns:
        --------
        str
            Generated prompt.
        """
        self.logger.info(f"Generating prompt using template: {template_name}")
        
        # Get template
        template = self.get_template(template_name)
        
        # Fill in template placeholders
        try:
            prompt = template.format(**kwargs)
        except KeyError as e:
            self.logger.warning(f"Missing value for placeholder: {e}")
            # Replace missing placeholders with a default value
            for key in e.args:
                template = template.replace("{" + key + "}", f"<{key} not provided>")
            prompt = template.format(**kwargs)
        
        return prompt


class ChainOfThoughtProcessor:
    """
    Implements chain-of-thought reasoning for transparent decision making.
    
    This class processes Gemma 3 outputs to extract the reasoning chain,
    ensuring that trading decisions are transparent and explainable.
    """
    
    def __init__(self):
        """Initialize the ChainOfThoughtProcessor."""
        self.logger = logging.getLogger("GemmaTrading.ChainOfThoughtProcessor")
        self.logger.info("Initialized ChainOfThoughtProcessor")
    
    def extract_reasoning_chain(self, response: str) -> List[str]:
        """
        Extract the reasoning chain from a Gemma 3 response.
        
        Parameters:
        -----------
        response : str
            Response from Gemma 3.
            
        Returns:
        --------
        List[str]
            List of reasoning steps.
        """
        self.logger.info("Extracting reasoning chain from response")
        
        # Look for explicit reasoning markers
        if "Reasoning:" in response and "Steps:" in response:
            # Extract the section between "Reasoning:" and the next major section
            reasoning_section = response.split("Reasoning:")[1].split("\n\n")[0]
            
            # Split into steps
            steps = [step.strip() for step in reasoning_section.split("\n") if step.strip()]
            
            if steps:
                return steps
        
        # If no explicit markers, try to identify numbered or bulleted steps
        numbered_steps = []
        
        # Check for numbered steps (e.g., "1. First step")
        numbered_pattern = r'\d+\.\s+(.*)'
        numbered_matches = re.findall(numbered_pattern, response)
        if numbered_matches:
            numbered_steps = numbered_matches
        
        # Check for bulleted steps (e.g., "• First step" or "- First step")
        if not numbered_steps:
            bullet_pattern = r'[•\-\*]\s+(.*)'
            bullet_matches = re.findall(bullet_pattern, response)
            if bullet_matches:
                numbered_steps = bullet_matches
        
        # If still no structured steps found, split by sentences as a fallback
        if not numbered_steps:
            # Simple sentence splitting (not perfect but a reasonable fallback)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
            
            # Filter out very short sentences and questions
            numbered_steps = [s for s in sentences if len(s) > 15 and not s.endswith("?")]
        
        return numbered_steps
    
    def format_reasoning_chain(self, reasoning_chain: List[str], format_type: str = "markdown") -> str:
        """
        Format a reasoning chain for display.
        
        Parameters:
        -----------
        reasoning_chain : List[str]
            List of reasoning steps.
        format_type : str
            Format type: "markdown", "html", or "text".
            
        Returns:
        --------
        str
            Formatted reasoning chain.
        """
        self.logger.info(f"Formatting reasoning chain as {format_type}")
        
        if not reasoning_chain:
            return "No reasoning steps available."
        
        if format_type == "markdown":
            formatted = "## Reasoning Process\n\n"
            for i, step in enumerate(reasoning_chain, 1):
                formatted += f"{i}. {step}\n"
        
        elif format_type == "html":
            formatted = "<h2>Reasoning Process</h2>\n<ol>\n"
            for step in reasoning_chain:
                formatted += f"  <li>{step}</li>\n"
            formatted += "</ol>"
        
        else:  # text
            formatted = "Reasoning Process:\n\n"
            for i, step in enumerate(reasoning_chain, 1):
                formatted += f"{i}. {step}\n"
        
        return formatted
    
    def enhance_prompt_for_cot(self, prompt: str) -> str:
        """
        Enhance a prompt to encourage chain-of-thought reasoning.
        
        Parameters:
        -----------
        prompt : str
            Original prompt.
            
        Returns:
        --------
        str
            Enhanced prompt.
        """
        self.logger.info("Enhancing prompt for chain-of-thought reasoning")
        
        # Add instructions for chain-of-thought reasoning
        cot_instructions = "\n\nPlease think through this step-by-step and explain your reasoning process. First analyze the relevant factors, then evaluate different options, and finally reach a conclusion based on your analysis."
        
        # Add specific format instructions
        format_instructions = "\n\nProvide your reasoning in the following format:\nReasoning:\n1. First consideration\n2. Second consideration\n...\n\nConclusion: Your final answer based on the reasoning above."
        
        # Combine original prompt with instructions
        enhanced_prompt = prompt + cot_instructions + format_instructions
        
        return enhanced_prompt


class FeedbackLoop:
    """
    Manages the learning and adaptation process.
    
    This class implements a feedback loop that allows the system to learn from
    trading outcomes and user feedback, continuously improving its performance.
    """
    
    def __init__(self, feedback_db_path: Optional[str] = None):
        """
        Initialize the FeedbackLoop.
        
        Parameters:
        -----------
        feedback_db_path : str, optional
            Path to the feedback database. If not provided, uses an in-memory database.
        """
        self.logger = logging.getLogger("GemmaTrading.FeedbackLoop")
        
        # Set up feedback storage
        self.feedback_db_path = feedback_db_path
        self.feedback_data = {
            "trade_feedback": [],
            "strategy_feedback": [],
            "user_feedback": []
        }
        
        self.logger.info("Initialized FeedbackLoop")
    
    def record_trade_outcome(self, trade_data: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """
        Record the outcome of a trade for learning.
        
        Parameters:
        -----------
        trade_data : Dict[str, Any]
            Data about the trade.
        outcome : Dict[str, Any]
            Outcome of the trade.
        """
        self.logger.info(f"Recording trade outcome for trade ID: {trade_data.get('id', 'unknown')}")
        
        # Create feedback entry
        feedback_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "trade_data": trade_data,
            "outcome": outcome,
            "processed": False
        }
        
        # Add to feedback data
        self.feedback_data["trade_feedback"].append(feedback_entry)
        
        # In a real implementation, this would also persist to disk
        if self.feedback_db_path:
            self._save_feedback_data()
    
    def record_strategy_performance(self, strategy_data: Dict[str, Any], performance: Dict[str, Any]) -> None:
        """
        Record the performance of a strategy for learning.
        
        Parameters:
        -----------
        strategy_data : Dict[str, Any]
            Data about the strategy.
        performance : Dict[str, Any]
            Performance metrics of the strategy.
        """
        self.logger.info(f"Recording strategy performance for strategy: {strategy_data.get('name', 'unknown')}")
        
        # Create feedback entry
        feedback_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy_data": strategy_data,
            "performance": performance,
            "processed": False
        }
        
        # Add to feedback data
        self.feedback_data["strategy_feedback"].append(feedback_entry)
        
        # In a real implementation, this would also persist to disk
        if self.feedback_db_path:
            self._save_feedback_data()
    
    def record_user_feedback(self, feedback_type: str, content: str, rating: int, context: Dict[str, Any]) -> None:
        """
        Record user feedback for learning.
        
        Parameters:
        -----------
        feedback_type : str
            Type of feedback (e.g., "strategy", "signal", "explanation").
        content : str
            Content that received feedback.
        rating : int
            User rating (1-5).
        context : Dict[str, Any]
            Context in which the feedback was given.
        """
        self.logger.info(f"Recording user feedback of type: {feedback_type}")
        
        # Create feedback entry
        feedback_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "feedback_type": feedback_type,
            "content": content,
            "rating": rating,
            "context": context,
            "processed": False
        }
        
        # Add to feedback data
        self.feedback_data["user_feedback"].append(feedback_entry)
        
        # In a real implementation, this would also persist to disk
        if self.feedback_db_path:
            self._save_feedback_data()
    
    def process_feedback(self) -> Dict[str, Any]:
        """
        Process collected feedback to generate learning insights.
        
        Returns:
        --------
        Dict[str, Any]
            Learning insights derived from feedback.
        """
        self.logger.info("Processing feedback to generate learning insights")
        
        # Count unprocessed feedback
        unprocessed_trade_feedback = sum(1 for f in self.feedback_data["trade_feedback"] if not f["processed"])
        unprocessed_strategy_feedback = sum(1 for f in self.feedback_data["strategy_feedback"] if not f["processed"])
        unprocessed_user_feedback = sum(1 for f in self.feedback_data["user_feedback"] if not f["processed"])
        
        self.logger.info(f"Processing {unprocessed_trade_feedback} trade feedback entries, "
                        f"{unprocessed_strategy_feedback} strategy feedback entries, and "
                        f"{unprocessed_user_feedback} user feedback entries")
        
        # In a real implementation, this would analyze the feedback to extract insights
        # For now, we'll create a simple summary
        
        # Process trade feedback
        trade_insights = self._process_trade_feedback()
        
        # Process strategy feedback
        strategy_insights = self._process_strategy_feedback()
        
        # Process user feedback
        user_insights = self._process_user_feedback()
        
        # Combine insights
        insights = {
            "trade_insights": trade_insights,
            "strategy_insights": strategy_insights,
            "user_insights": user_insights,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Mark feedback as processed
        for feedback_type in self.feedback_data:
            for entry in self.feedback_data[feedback_type]:
                if not entry["processed"]:
                    entry["processed"] = True
        
        # In a real implementation, this would also persist to disk
        if self.feedback_db_path:
            self._save_feedback_data()
        
        return insights
    
    def _process_trade_feedback(self) -> Dict[str, Any]:
        """
        Process trade feedback to generate insights.
        
        Returns:
        --------
        Dict[str, Any]
            Insights from trade feedback.
        """
        # Get unprocessed trade feedback
        unprocessed = [f for f in self.feedback_data["trade_feedback"] if not f["processed"]]
        
        if not unprocessed:
            return {"message": "No trade feedback to process"}
        
        # Calculate success rate
        successful_trades = sum(1 for f in unprocessed if f["outcome"].get("profit", 0) > 0)
        success_rate = successful_trades / len(unprocessed) if unprocessed else 0
        
        # Calculate average profit/loss
        total_profit = sum(f["outcome"].get("profit", 0) for f in unprocessed)
        avg_profit = total_profit / len(unprocessed) if unprocessed else 0
        
        # Identify common factors in successful trades
        successful_trade_factors = {}
        for f in unprocessed:
            if f["outcome"].get("profit", 0) > 0:
                for key, value in f["trade_data"].items():
                    if key not in successful_trade_factors:
                        successful_trade_factors[key] = []
                    successful_trade_factors[key].append(value)
        
        # Simplify to most common values
        common_factors = {}
        for key, values in successful_trade_factors.items():
            if isinstance(values[0], (str, bool, int, float)):
                # Count occurrences
                counter = Counter(values)
                # Get most common value
                most_common = counter.most_common(1)
                if most_common:
                    value, count = most_common[0]
                    # Only include if it appears in at least 50% of successful trades
                    if count / successful_trades >= 0.5:
                        common_factors[key] = value
        
        return {
            "success_rate": success_rate,
            "avg_profit": avg_profit,
            "total_profit": total_profit,
            "common_success_factors": common_factors,
            "processed_count": len(unprocessed)
        }
    
    def _process_strategy_feedback(self) -> Dict[str, Any]:
        """
        Process strategy feedback to generate insights.
        
        Returns:
        --------
        Dict[str, Any]
            Insights from strategy feedback.
        """
        # Get unprocessed strategy feedback
        unprocessed = [f for f in self.feedback_data["strategy_feedback"] if not f["processed"]]
        
        if not unprocessed:
            return {"message": "No strategy feedback to process"}
        
        # Calculate average performance metrics
        avg_metrics = {}
        for metric in ["sharpe_ratio", "win_rate", "profit_factor", "max_drawdown"]:
            values = [f["performance"].get(metric, 0) for f in unprocessed]
            avg_metrics[f"avg_{metric}"] = sum(values) / len(values) if values else 0
        
        # Identify best performing strategies
        strategies = [(f["strategy_data"].get("name", "unknown"), f["performance"].get("sharpe_ratio", 0)) 
                     for f in unprocessed]
        strategies.sort(key=lambda x: x[1], reverse=True)
        best_strategies = strategies[:3] if len(strategies) >= 3 else strategies
        
        return {
            "performance_metrics": avg_metrics,
            "best_strategies": best_strategies,
            "processed_count": len(unprocessed)
        }
    
    def _process_user_feedback(self) -> Dict[str, Any]:
        """
        Process user feedback to generate insights.
        
        Returns:
        --------
        Dict[str, Any]
            Insights from user feedback.
        """
        # Get unprocessed user feedback
        unprocessed = [f for f in self.feedback_data["user_feedback"] if not f["processed"]]
        
        if not unprocessed:
            return {"message": "No user feedback to process"}
        
        # Calculate average ratings by feedback type
        ratings_by_type = {}
        for f in unprocessed:
            feedback_type = f["feedback_type"]
            if feedback_type not in ratings_by_type:
                ratings_by_type[feedback_type] = []
            ratings_by_type[feedback_type].append(f["rating"])
        
        avg_ratings = {
            feedback_type: sum(ratings) / len(ratings) if ratings else 0
            for feedback_type, ratings in ratings_by_type.items()
        }
        
        # Identify low-rated content for improvement
        low_rated = [f for f in unprocessed if f["rating"] <= 2]
        low_rated_types = Counter([f["feedback_type"] for f in low_rated])
        
        return {
            "avg_ratings": avg_ratings,
            "low_rated_types": dict(low_rated_types),
            "overall_avg_rating": sum(f["rating"] for f in unprocessed) / len(unprocessed) if unprocessed else 0,
            "processed_count": len(unprocessed)
        }
    
    def _save_feedback_data(self) -> None:
        """Save feedback data to disk."""
        if not self.feedback_db_path:
            return
        
        try:
            with open(self.feedback_db_path, 'w') as f:
                json.dump(self.feedback_data, f)
            self.logger.info(f"Saved feedback data to {self.feedback_db_path}")
        except Exception as e:
            self.logger.error(f"Failed to save feedback data: {e}")
    
    def _load_feedback_data(self) -> None:
        """Load feedback data from disk."""
        if not self.feedback_db_path or not os.path.exists(self.feedback_db_path):
            return
        
        try:
            with open(self.feedback_db_path, 'r') as f:
                self.feedback_data = json.load(f)
            self.logger.info(f"Loaded feedback data from {self.feedback_db_path}")
        except Exception as e:
            self.logger.error(f"Failed to load feedback data: {e}")


class DataIntegration:
    """
    Handles integration of various data sources for Gemma 3 analysis.
    
    This class provides methods to prepare and format different types of data
    for analysis by Gemma 3, ensuring that the model receives well-structured
    inputs for optimal performance.
    """
    
    def __init__(self):
        """Initialize the DataIntegration."""
        self.logger = logging.getLogger("GemmaTrading.DataIntegration")
        self.logger.info("Initialized DataIntegration")
    
    def prepare_market_data(self, data: pd.DataFrame) -> str:
        """
        Prepare market data for Gemma 3 analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data DataFrame.
            
        Returns:
        --------
        str
            Formatted market data string.
        """
        self.logger.info("Preparing market data for analysis")
        
        # Check if data is empty
        if data.empty:
            return "No market data available."
        
        # Format data as a string
        data_str = "Market Data:\n\n"
        
        # Add summary statistics
        data_str += "Summary Statistics:\n"
        data_str += f"Start Date: {data.index[0].strftime('%Y-%m-%d')}\n"
        data_str += f"End Date: {data.index[-1].strftime('%Y-%m-%d')}\n"
        data_str += f"Number of Trading Days: {len(data)}\n"
        
        if 'Close' in data.columns:
            data_str += f"Starting Price: {data['Close'].iloc[0]:.2f}\n"
            data_str += f"Ending Price: {data['Close'].iloc[-1]:.2f}\n"
            data_str += f"Price Change: {data['Close'].iloc[-1] - data['Close'].iloc[0]:.2f} ({(data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100:.2f}%)\n"
        
        # Add recent data points
        data_str += "\nRecent Data Points:\n"
        recent_data = data.tail(5).reset_index()
        for _, row in recent_data.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], (datetime.datetime, pd.Timestamp)) else str(row['Date'])
            data_str += f"{date_str}: "
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in row:
                    if col == 'Volume':
                        data_str += f"{col}: {row[col]:,.0f} "
                    else:
                        data_str += f"{col}: {row[col]:.2f} "
            data_str += "\n"
        
        return data_str
    
    def prepare_technical_indicators(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> str:
        """
        Prepare technical indicators for Gemma 3 analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data DataFrame.
        indicators : Dict[str, pd.Series]
            Dictionary of indicator names to indicator series.
            
        Returns:
        --------
        str
            Formatted technical indicators string.
        """
        self.logger.info("Preparing technical indicators for analysis")
        
        # Check if data is empty
        if data.empty or not indicators:
            return "No technical indicators available."
        
        # Format indicators as a string
        indicators_str = "Technical Indicators:\n\n"
        
        # Add recent indicator values
        recent_dates = data.index[-5:]
        
        for indicator_name, indicator_series in indicators.items():
            indicators_str += f"{indicator_name}:\n"
            
            for date in recent_dates:
                if date in indicator_series.index:
                    date_str = date.strftime('%Y-%m-%d') if isinstance(date, (datetime.datetime, pd.Timestamp)) else str(date)
                    value = indicator_series[date]
                    indicators_str += f"  {date_str}: {value:.4f}\n"
            
            indicators_str += "\n"
        
        # Add indicator interpretations
        indicators_str += "Indicator Interpretations:\n"
        
        for indicator_name in indicators:
            interpretation = self._get_indicator_interpretation(indicator_name, indicators[indicator_name])
            indicators_str += f"{indicator_name}: {interpretation}\n"
        
        return indicators_str
    
    def prepare_news_data(self, news_items: List[Dict[str, Any]]) -> str:
        """
        Prepare news data for Gemma 3 analysis.
        
        Parameters:
        -----------
        news_items : List[Dict[str, Any]]
            List of news items.
            
        Returns:
        --------
        str
            Formatted news data string.
        """
        self.logger.info("Preparing news data for analysis")
        
        # Check if news items are empty
        if not news_items:
            return "No news data available."
        
        # Format news as a string
        news_str = "News Data:\n\n"
        
        for i, item in enumerate(news_items, 1):
            news_str += f"Article {i}:\n"
            news_str += f"Title: {item.get('title', 'No Title')}\n"
            news_str += f"Date: {item.get('date', 'No Date')}\n"
            news_str += f"Source: {item.get('source', 'No Source')}\n"
            
            # Truncate content if too long
            content = item.get('content', 'No Content')
            if len(content) > 500:
                content = content[:500] + "..."
            
            news_str += f"Content: {content}\n\n"
        
        return news_str
    
    def prepare_social_data(self, social_posts: List[Dict[str, Any]]) -> str:
        """
        Prepare social media data for Gemma 3 analysis.
        
        Parameters:
        -----------
        social_posts : List[Dict[str, Any]]
            List of social media posts.
            
        Returns:
        --------
        str
            Formatted social media data string.
        """
        self.logger.info("Preparing social media data for analysis")
        
        # Check if social posts are empty
        if not social_posts:
            return "No social media data available."
        
        # Format social posts as a string
        social_str = "Social Media Data:\n\n"
        
        for i, post in enumerate(social_posts, 1):
            social_str += f"Post {i}:\n"
            social_str += f"Platform: {post.get('platform', 'Unknown')}\n"
            social_str += f"User: {post.get('user', 'Anonymous')}\n"
            social_str += f"Date: {post.get('date', 'No Date')}\n"
            social_str += f"Content: {post.get('content', 'No Content')}\n"
            social_str += f"Likes: {post.get('likes', 0)}, Shares: {post.get('shares', 0)}, Comments: {post.get('comments', 0)}\n\n"
        
        return social_str
    
    def prepare_earnings_data(self, earnings_data: Dict[str, Any]) -> str:
        """
        Prepare earnings data for Gemma 3 analysis.
        
        Parameters:
        -----------
        earnings_data : Dict[str, Any]
            Earnings data.
            
        Returns:
        --------
        str
            Formatted earnings data string.
        """
        self.logger.info("Preparing earnings data for analysis")
        
        # Check if earnings data is empty
        if not earnings_data:
            return "No earnings data available."
        
        # Format earnings data as a string
        earnings_str = "Earnings Data:\n\n"
        
        # Add basic earnings information
        earnings_str += "Basic Information:\n"
        earnings_str += f"Company: {earnings_data.get('company', 'Unknown')}\n"
        earnings_str += f"Fiscal Quarter: {earnings_data.get('fiscal_quarter', 'Unknown')}\n"
        earnings_str += f"Report Date: {earnings_data.get('report_date', 'Unknown')}\n\n"
        
        # Add key metrics
        earnings_str += "Key Metrics:\n"
        
        metrics = earnings_data.get('metrics', {})
        for metric, value in metrics.items():
            estimate = metrics.get(f"{metric}_estimate", "N/A")
            surprise = metrics.get(f"{metric}_surprise", "N/A")
            
            earnings_str += f"{metric.capitalize()}: {value} (Estimate: {estimate}, Surprise: {surprise})\n"
        
        # Add guidance
        guidance = earnings_data.get('guidance', {})
        if guidance:
            earnings_str += "\nGuidance:\n"
            for metric, value in guidance.items():
                earnings_str += f"{metric.capitalize()}: {value}\n"
        
        # Add highlights
        highlights = earnings_data.get('highlights', [])
        if highlights:
            earnings_str += "\nHighlights:\n"
            for highlight in highlights:
                earnings_str += f"- {highlight}\n"
        
        return earnings_str
    
    def prepare_combined_data(self, data_sources: Dict[str, Any]) -> str:
        """
        Prepare combined data from multiple sources for Gemma 3 analysis.
        
        Parameters:
        -----------
        data_sources : Dict[str, Any]
            Dictionary of data sources.
            
        Returns:
        --------
        str
            Formatted combined data string.
        """
        self.logger.info("Preparing combined data for analysis")
        
        # Format combined data as a string
        combined_str = "Combined Data for Analysis:\n\n"
        
        # Add market data if available
        if 'market_data' in data_sources:
            market_data = data_sources['market_data']
            if isinstance(market_data, pd.DataFrame) and not market_data.empty:
                combined_str += self.prepare_market_data(market_data) + "\n\n"
        
        # Add technical indicators if available
        if 'indicators' in data_sources:
            indicators = data_sources['indicators']
            if indicators and 'market_data' in data_sources:
                combined_str += self.prepare_technical_indicators(data_sources['market_data'], indicators) + "\n\n"
        
        # Add news data if available
        if 'news' in data_sources:
            news_items = data_sources['news']
            if news_items:
                combined_str += self.prepare_news_data(news_items) + "\n\n"
        
        # Add social data if available
        if 'social' in data_sources:
            social_posts = data_sources['social']
            if social_posts:
                combined_str += self.prepare_social_data(social_posts) + "\n\n"
        
        # Add earnings data if available
        if 'earnings' in data_sources:
            earnings_data = data_sources['earnings']
            if earnings_data:
                combined_str += self.prepare_earnings_data(earnings_data) + "\n\n"
        
        return combined_str
    
    def _get_indicator_interpretation(self, indicator_name: str, indicator_series: pd.Series) -> str:
        """
        Get interpretation for a technical indicator.
        
        Parameters:
        -----------
        indicator_name : str
            Name of the indicator.
        indicator_series : pd.Series
            Indicator values.
            
        Returns:
        --------
        str
            Indicator interpretation.
        """
        # Get the most recent value
        if indicator_series.empty:
            return "No data available."
        
        recent_value = indicator_series.iloc[-1]
        
        # Simple interpretations for common indicators
        indicator_name_lower = indicator_name.lower()
        
        if 'rsi' in indicator_name_lower:
            if recent_value > 70:
                return f"Overbought ({recent_value:.2f})"
            elif recent_value < 30:
                return f"Oversold ({recent_value:.2f})"
            else:
                return f"Neutral ({recent_value:.2f})"
        
        elif 'macd' in indicator_name_lower:
            # For MACD, we need both the MACD line and signal line
            # This is a simplified interpretation
            if 'signal' in indicator_name_lower:
                return f"Signal line value: {recent_value:.4f}"
            elif 'histogram' in indicator_name_lower:
                if recent_value > 0:
                    return f"Bullish momentum ({recent_value:.4f})"
                else:
                    return f"Bearish momentum ({recent_value:.4f})"
            else:
                return f"MACD line value: {recent_value:.4f}"
        
        elif 'bollinger' in indicator_name_lower:
            if 'upper' in indicator_name_lower:
                return f"Upper band: {recent_value:.2f}"
            elif 'lower' in indicator_name_lower:
                return f"Lower band: {recent_value:.2f}"
            else:
                return f"Middle band: {recent_value:.2f}"
        
        elif 'ma' in indicator_name_lower or 'sma' in indicator_name_lower or 'ema' in indicator_name_lower:
            # For moving averages, we need price data for comparison
            # This is a simplified interpretation
            return f"Current value: {recent_value:.2f}"
        
        # Default interpretation
        return f"Current value: {recent_value:.4f}"


class GemmaCore:
    """
    Central interface for all Gemma 3 interactions.
    
    This class provides a unified interface for accessing Gemma 3's capabilities,
    managing the underlying model, and coordinating interactions between different
    components of the trading system.
    """
    
    def __init__(self, model_path: Optional[str] = None, templates_path: Optional[str] = None):
        """
        Initialize the GemmaCore.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the Gemma 3 model files. If not provided, uses default path.
        templates_path : str, optional
            Path to prompt templates. If not provided, uses default templates.
        """
        self.logger = logging.getLogger("GemmaTrading.GemmaCore")
        
        # Initialize components
        self.model_manager = ModelManager(model_path)
        self.prompt_engine = PromptEngine(templates_path)
        self.cot_processor = ChainOfThoughtProcessor()
        self.feedback_loop = FeedbackLoop()
        self.data_integration = DataIntegration()
        
        # Load default model
        self.default_model = self.model_manager.load_model()
        
        self.logger.info("Initialized GemmaCore")
    
    def generate_response(self, prompt: str, model_type: str = "default", use_cot: bool = True) -> str:
        """
        Generate a response from Gemma 3.
        
        Parameters:
        -----------
        prompt : str
            Prompt to send to Gemma 3.
        model_type : str
            Type of model to use. Options: "default", "finance", "math", "nlp".
        use_cot : bool
            Whether to use chain-of-thought reasoning.
            
        Returns:
        --------
        str
            Generated response.
        """
        self.logger.info(f"Generating response with model type: {model_type}")
        
        # Enhance prompt for chain-of-thought reasoning if requested
        if use_cot:
            prompt = self.cot_processor.enhance_prompt_for_cot(prompt)
        
        # Load appropriate model
        model = self.model_manager.load_model(model_type)
        
        # In a real implementation, this would call the actual Gemma 3 model
        # For now, we'll generate a mock response
        response = self._generate_mock_response(prompt, model_type)
        
        return response
    
    def analyze_market_data(self, ticker: str, data: pd.DataFrame, indicators: Dict[str, pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze market data using Gemma 3.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        data : pd.DataFrame
            Market data DataFrame.
        indicators : Dict[str, pd.Series], optional
            Dictionary of technical indicators.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results.
        """
        self.logger.info(f"Analyzing market data for {ticker}")
        
        # Prepare data for analysis
        market_data_str = self.data_integration.prepare_market_data(data)
        
        if indicators:
            indicators_str = self.data_integration.prepare_technical_indicators(data, indicators)
            data_str = market_data_str + "\n\n" + indicators_str
        else:
            data_str = market_data_str
        
        # Generate prompt for market analysis
        prompt = self.prompt_engine.generate_prompt(
            "technical_analysis",
            ticker=ticker,
            indicators="various technical indicators" if indicators else "price data only",
            data=data_str
        )
        
        # Get analysis from Gemma 3
        response = self.generate_response(prompt, model_type="finance")
        
        # Extract reasoning chain
        reasoning_chain = self.cot_processor.extract_reasoning_chain(response)
        
        # Parse the response to extract structured information
        analysis = self._parse_market_analysis(response, ticker)
        
        # Add reasoning chain to analysis
        analysis["reasoning"] = reasoning_chain
        
        return analysis
    
    def generate_trading_strategy(self, ticker: str, data: pd.DataFrame, strategy_type: str) -> Dict[str, Any]:
        """
        Generate a trading strategy using Gemma 3.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        data : pd.DataFrame
            Market data DataFrame.
        strategy_type : str
            Type of strategy to generate (e.g., "swing", "day", "position").
            
        Returns:
        --------
        Dict[str, Any]
            Generated strategy.
        """
        self.logger.info(f"Generating {strategy_type} trading strategy for {ticker}")
        
        # Prepare data for strategy generation
        data_str = self.data_integration.prepare_market_data(data)
        
        # Generate prompt for strategy generation
        prompt = self.prompt_engine.generate_prompt(
            "strategy_generation",
            ticker=ticker,
            conditions=f"Market data for {ticker} with strategy type: {strategy_type}\n\n{data_str}"
        )
        
        # Get strategy from Gemma 3
        response = self.generate_response(prompt, model_type="finance")
        
        # Extract reasoning chain
        reasoning_chain = self.cot_processor.extract_reasoning_chain(response)
        
        # Parse the response to extract structured information
        strategy = self._parse_strategy(response, ticker, strategy_type)
        
        # Add reasoning chain to strategy
        strategy["reasoning"] = reasoning_chain
        
        return strategy
    
    def analyze_news_sentiment(self, ticker: str, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze news sentiment using Gemma 3.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        news_items : List[Dict[str, Any]]
            List of news items to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Sentiment analysis results.
        """
        self.logger.info(f"Analyzing news sentiment for {ticker}")
        
        # Prepare news data for analysis
        news_str = self.data_integration.prepare_news_data(news_items)
        
        # Generate prompt for news analysis
        prompt = self.prompt_engine.generate_prompt(
            "news_analysis",
            ticker=ticker,
            articles=news_str
        )
        
        # Get analysis from Gemma 3
        response = self.generate_response(prompt, model_type="nlp")
        
        # Extract reasoning chain
        reasoning_chain = self.cot_processor.extract_reasoning_chain(response)
        
        # Parse the response to extract structured information
        sentiment = self._parse_sentiment_analysis(response, ticker)
        
        # Add reasoning chain to sentiment
        sentiment["reasoning"] = reasoning_chain
        
        return sentiment
    
    def explain_trading_decision(self, ticker: str, decision: str, data: Dict[str, Any]) -> str:
        """
        Generate an explanation for a trading decision using Gemma 3.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        decision : str
            Trading decision to explain (e.g., "BUY", "SELL", "HOLD").
        data : Dict[str, Any]
            Data used to make the decision.
            
        Returns:
        --------
        str
            Explanation of the trading decision.
        """
        self.logger.info(f"Explaining {decision} decision for {ticker}")
        
        # Prepare data for explanation
        data_str = json.dumps(data, indent=2)
        
        # Generate prompt for decision explanation
        prompt = self.prompt_engine.generate_prompt(
            "trading_decision",
            ticker=ticker,
            analysis=f"Decision: {decision}\n\nData: {data_str}",
            position="None"  # Assuming no current position
        )
        
        # Get explanation from Gemma 3
        response = self.generate_response(prompt, model_type="finance")
        
        # Extract reasoning chain
        reasoning_chain = self.cot_processor.extract_reasoning_chain(response)
        
        # Format reasoning chain
        explanation = self.cot_processor.format_reasoning_chain(reasoning_chain, format_type="markdown")
        
        return explanation
    
    def _generate_mock_response(self, prompt: str, model_type: str) -> str:
        """
        Generate a mock response for development/testing.
        
        Parameters:
        -----------
        prompt : str
            Prompt sent to the model.
        model_type : str
            Type of model used.
            
        Returns:
        --------
        str
            Mock response.
        """
        # Extract key information from the prompt
        ticker_match = re.search(r'for\s+([A-Z]+)', prompt)
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"
        
        # Generate different responses based on prompt content
        if "technical_analysis" in prompt or "market_analysis" in prompt:
            return self._generate_mock_technical_analysis(ticker)
        elif "strategy_generation" in prompt:
            return self._generate_mock_strategy(ticker)
        elif "news_analysis" in prompt:
            return self._generate_mock_news_analysis(ticker)
        elif "trading_decision" in prompt:
            return self._generate_mock_trading_decision(ticker)
        else:
            return self._generate_generic_mock_response(ticker)
    
    def _generate_mock_technical_analysis(self, ticker: str) -> str:
        """Generate a mock technical analysis response."""
        return f"""
Reasoning:
1. Examining the recent price action for {ticker}, I notice a clear downtrend over the past 20 trading days.
2. The 20-day SMA is below the 50-day SMA, confirming the bearish trend.
3. The RSI is currently at 35, indicating weak momentum but not yet oversold.
4. The MACD is below the signal line, confirming bearish momentum.
5. Volume has been increasing on down days and decreasing on up days, suggesting selling pressure.
6. Support levels can be identified at $191.39 and $187.22.
7. Resistance levels are at $202.45 and $208.91.

Conclusion: Based on technical analysis, {ticker} is in a bearish trend with weak momentum and increasing selling pressure. The price is likely to test support at $191.39 in the near term. A break below this level could lead to further downside to $187.22. Traders should be cautious and consider bearish strategies.

Technical Analysis Results:
{{
  "ticker": "{ticker}",
  "trend": "bearish",
  "momentum": "weak",
  "support_levels": [191.39, 187.22],
  "resistance_levels": [202.45, 208.91],
  "rsi": 35,
  "macd": "bearish",
  "volume_trend": "increasing on down days",
  "recommendation": "SELL"
}}
"""
    
    def _generate_mock_strategy(self, ticker: str) -> str:
        """Generate a mock trading strategy response."""
        return f"""
Reasoning:
1. Given the bearish trend identified in {ticker}, a swing trading strategy should focus on short positions.
2. The optimal entry point would be after a bounce to resistance around $202.45.
3. A stop loss should be placed above the recent swing high at approximately $208.91.
4. The first target for profit taking would be at the support level of $191.39.
5. The second target would be at the lower support of $187.22.
6. Position sizing should be conservative due to the current market volatility.
7. The strategy should include a trailing stop to lock in profits if the downtrend accelerates.

Conclusion: A bearish swing trading strategy for {ticker} with entry after a bounce to resistance, stop loss above recent swing high, and targets at identified support levels.

Trading Strategy:
{{
  "name": "Bearish Swing Strategy for {ticker}",
  "ticker": "{ticker}",
  "type": "swing",
  "direction": "short",
  "entry": {{
    "condition": "Price bounces to resistance level",
    "price": 202.45
  }},
  "stop_loss": {{
    "condition": "Above recent swing high",
    "price": 208.91,
    "percentage": 3.19
  }},
  "take_profit": [
    {{
      "condition": "First support level",
      "price": 191.39,
      "percentage": 5.46
    }},
    {{
      "condition": "Second support level",
      "price": 187.22,
      "percentage": 7.52
    }}
  ],
  "position_sizing": {{
    "type": "percentage",
    "value": 5
  }},
  "risk_reward_ratio": 1.71,
  "timeframe": "daily",
  "indicators": [
    "20-day SMA",
    "50-day SMA",
    "RSI",
    "MACD"
  ],
  "parameters": {{
    "rsi_threshold": 60,
    "macd_signal": "bearish crossover"
  }}
}}
"""
    
    def _generate_mock_news_analysis(self, ticker: str) -> str:
        """Generate a mock news sentiment analysis response."""
        return f"""
Reasoning:
1. Analyzing the recent news articles for {ticker}, I notice several negative headlines related to product delays.
2. There are also concerns about increasing competition in the company's main market segments.
3. One positive article discusses a new product announcement, but the market reaction was muted.
4. The language used in most articles has a negative tone, with words like "struggles," "challenges," and "pressure."
5. The timing of the negative news coincides with the technical weakness observed in the price chart.
6. Analyst comments quoted in the articles are predominantly cautious or negative.
7. The volume of negative news has increased over the past week compared to the previous period.

Conclusion: The news sentiment for {ticker} is predominantly negative, with concerns about product delays and increasing competition. This aligns with the technical weakness observed in the price action.

News Sentiment Analysis:
{{
  "ticker": "{ticker}",
  "sentiment": "negative",
  "confidence": 0.78,
  "key_topics": [
    "Product delays",
    "Increasing competition",
    "New product announcement",
    "Market pressure"
  ],
  "sentiment_breakdown": {{
    "positive": 0.15,
    "neutral": 0.25,
    "negative": 0.60
  }},
  "key_events": [
    "Announced delay in flagship product release",
    "Competitor launched similar product at lower price point",
    "Announced new product line with limited market impact"
  ],
  "potential_impact": "The negative news sentiment is likely to continue putting downward pressure on the stock price in the near term."
}}
"""
    
    def _generate_mock_trading_decision(self, ticker: str) -> str:
        """Generate a mock trading decision explanation response."""
        return f"""
Reasoning:
1. The technical analysis shows {ticker} is in a clear bearish trend with the 20-day SMA below the 50-day SMA.
2. The RSI at 35 indicates weak momentum, but not yet oversold conditions.
3. The MACD is below the signal line, confirming bearish momentum.
4. News sentiment is predominantly negative, with concerns about product delays and competition.
5. Support levels are identified at $191.39 and $187.22, providing potential targets for a short position.
6. The risk-reward ratio for a short position entered at current levels is favorable at 1.71.
7. Market conditions overall are showing increased volatility, suggesting caution with position sizing.

Conclusion: The decision to SELL {ticker} is supported by both technical analysis showing a bearish trend and negative news sentiment. The strategy includes a clear stop loss and profit targets with a favorable risk-reward ratio.

Trading Decision Explanation:
{{
  "decision": "SELL",
  "ticker": "{ticker}",
  "confidence": 0.82,
  "primary_factors": [
    "Bearish technical trend",
    "Negative news sentiment",
    "Favorable risk-reward ratio"
  ],
  "risk_management": {{
    "stop_loss": 208.91,
    "take_profit": [191.39, 187.22],
    "position_size": "5% of portfolio",
    "max_risk": "1% of portfolio value"
  }},
  "expected_outcome": "Price is expected to decline to first support level at $191.39 within 5-10 trading days."
}}
"""
    
    def _generate_generic_mock_response(self, ticker: str) -> str:
        """Generate a generic mock response."""
        return f"""
Reasoning:
1. Analyzing the provided information for {ticker}.
2. Considering multiple factors including price action, indicators, and market conditions.
3. Evaluating potential scenarios and their probabilities.
4. Weighing risk and reward for different approaches.
5. Considering the broader market context and sector performance.
6. Analyzing historical patterns and their relevance to current conditions.
7. Formulating a balanced assessment based on all available information.

Conclusion: Based on the analysis of available information, {ticker} shows mixed signals with slightly bearish bias in the near term.

Analysis Results:
{{
  "ticker": "{ticker}",
  "overall_assessment": "slightly bearish",
  "confidence": 0.65,
  "key_factors": [
    "Mixed technical signals",
    "Uncertain fundamental outlook",
    "Broader market pressure"
  ],
  "recommendation": "Consider reduced exposure with defined risk parameters"
}}
"""
    
    def _parse_market_analysis(self, response: str, ticker: str) -> Dict[str, Any]:
        """
        Parse market analysis response to extract structured information.
        
        Parameters:
        -----------
        response : str
            Response from Gemma 3.
        ticker : str
            Ticker symbol for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Structured market analysis.
        """
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            try:
                analysis = json.loads(json_match.group(0))
                return analysis
            except json.JSONDecodeError:
                pass
        
        # Fallback to regex extraction if JSON parsing fails
        trend_match = re.search(r'trend[\":]?\s*[\":,]?\s*[\":]?([a-zA-Z]+)[\":]?', response)
        trend = trend_match.group(1) if trend_match else "neutral"
        
        momentum_match = re.search(r'momentum[\":]?\s*[\":,]?\s*[\":]?([a-zA-Z]+)[\":]?', response)
        momentum = momentum_match.group(1) if momentum_match else "neutral"
        
        recommendation_match = re.search(r'recommendation[\":]?\s*[\":,]?\s*[\":]?(BUY|SELL|HOLD)[\":]?', response, re.IGNORECASE)
        recommendation = recommendation_match.group(1).upper() if recommendation_match else "HOLD"
        
        # Extract support levels
        support_levels = []
        support_match = re.search(r'support_levels[\":]?\s*[\":,]?\s*\[([\d\., ]+)\]', response)
        if support_match:
            support_str = support_match.group(1)
            support_levels = [float(level.strip()) for level in support_str.split(',') if level.strip()]
        
        # Extract resistance levels
        resistance_levels = []
        resistance_match = re.search(r'resistance_levels[\":]?\s*[\":,]?\s*\[([\d\., ]+)\]', response)
        if resistance_match:
            resistance_str = resistance_match.group(1)
            resistance_levels = [float(level.strip()) for level in resistance_str.split(',') if level.strip()]
        
        # Construct analysis dictionary
        analysis = {
            "ticker": ticker,
            "trend": trend,
            "momentum": momentum,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "recommendation": recommendation,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return analysis
    
    def _parse_strategy(self, response: str, ticker: str, strategy_type: str) -> Dict[str, Any]:
        """
        Parse strategy response to extract structured information.
        
        Parameters:
        -----------
        response : str
            Response from Gemma 3.
        ticker : str
            Ticker symbol for the asset.
        strategy_type : str
            Type of strategy.
            
        Returns:
        --------
        Dict[str, Any]
            Structured strategy.
        """
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            try:
                strategy = json.loads(json_match.group(0))
                return strategy
            except json.JSONDecodeError:
                pass
        
        # Fallback to regex extraction if JSON parsing fails
        name_match = re.search(r'name[\":]?\s*[\":,]?\s*[\":]?([^\"]+)[\":]?', response)
        name = name_match.group(1) if name_match else f"{strategy_type.capitalize()} Strategy for {ticker}"
        
        direction_match = re.search(r'direction[\":]?\s*[\":,]?\s*[\":]?([a-zA-Z]+)[\":]?', response)
        direction = direction_match.group(1) if direction_match else "long"
        
        # Extract entry condition and price
        entry_condition = "Market price"
        entry_price = 0.0
        
        entry_condition_match = re.search(r'entry[\":]?\s*\{[\s\S]*?condition[\":]?\s*[\":,]?\s*[\":]?([^\"]+)[\":]?', response)
        if entry_condition_match:
            entry_condition = entry_condition_match.group(1)
        
        entry_price_match = re.search(r'entry[\":]?\s*\{[\s\S]*?price[\":]?\s*[\":,]?\s*([0-9.]+)', response)
        if entry_price_match:
            entry_price = float(entry_price_match.group(1))
        
        # Extract stop loss
        stop_loss_price = 0.0
        stop_loss_match = re.search(r'stop_loss[\":]?\s*\{[\s\S]*?price[\":]?\s*[\":,]?\s*([0-9.]+)', response)
        if stop_loss_match:
            stop_loss_price = float(stop_loss_match.group(1))
        
        # Extract take profit
        take_profit_prices = []
        take_profit_match = re.search(r'take_profit[\":]?\s*\[([\s\S]*?)\]', response)
        if take_profit_match:
            take_profit_str = take_profit_match.group(1)
            price_matches = re.findall(r'price[\":]?\s*[\":,]?\s*([0-9.]+)', take_profit_str)
            take_profit_prices = [float(price) for price in price_matches]
        
        # Construct strategy dictionary
        strategy = {
            "name": name,
            "ticker": ticker,
            "type": strategy_type,
            "direction": direction,
            "entry": {
                "condition": entry_condition,
                "price": entry_price
            },
            "stop_loss": {
                "price": stop_loss_price
            },
            "take_profit": [{"price": price} for price in take_profit_prices],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return strategy
    
    def _parse_sentiment_analysis(self, response: str, ticker: str) -> Dict[str, Any]:
        """
        Parse sentiment analysis response to extract structured information.
        
        Parameters:
        -----------
        response : str
            Response from Gemma 3.
        ticker : str
            Ticker symbol for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Structured sentiment analysis.
        """
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            try:
                sentiment = json.loads(json_match.group(0))
                return sentiment
            except json.JSONDecodeError:
                pass
        
        # Fallback to regex extraction if JSON parsing fails
        sentiment_match = re.search(r'sentiment[\":]?\s*[\":,]?\s*[\":]?([a-zA-Z]+)[\":]?', response)
        sentiment_value = sentiment_match.group(1) if sentiment_match else "neutral"
        
        confidence_match = re.search(r'confidence[\":]?\s*[\":,]?\s*([0-9.]+)', response)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        # Extract key topics
        key_topics = []
        topics_match = re.search(r'key_topics[\":]?\s*[\":,]?\s*\[([\s\S]*?)\]', response)
        if topics_match:
            topics_str = topics_match.group(1)
            topic_matches = re.findall(r'[\":]?([^\",:]+)[\":]?', topics_str)
            key_topics = [topic.strip() for topic in topic_matches if topic.strip() and len(topic.strip()) > 2]
        
        # Extract key events
        key_events = []
        events_match = re.search(r'key_events[\":]?\s*[\":,]?\s*\[([\s\S]*?)\]', response)
        if events_match:
            events_str = events_match.group(1)
            event_matches = re.findall(r'[\":]([^\"]+)[\":]', events_str)
            key_events = [event.strip() for event in event_matches if event.strip()]
        
        # Extract potential impact
        impact_match = re.search(r'potential_impact[\":]?\s*[\":,]?\s*[\":]?([^\"]+)[\":]?', response)
        impact = impact_match.group(1) if impact_match else "Neutral impact expected"
        
        # Construct sentiment dictionary
        sentiment = {
            "ticker": ticker,
            "sentiment": sentiment_value,
            "confidence": confidence,
            "key_topics": key_topics,
            "key_events": key_events,
            "potential_impact": impact,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return sentiment
