"""
News Trading Integration Module for Gemma Advanced Trading System.

This module provides functionality for analyzing news and integrating news-based signals
into trading strategies.
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
import requests
import json
import re
import os
from collections import defaultdict
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import concurrent.futures
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NewsTrading")


class NewsTrading:
    """Class for analyzing news and integrating news-based signals into trading strategies."""
    
    def __init__(self, news_provider=None, config=None):
        """
        Initialize the NewsTrading module.
        
        Parameters:
        -----------
        news_provider : object, optional
            News provider object for fetching news data
        config : dict, optional
            Configuration parameters for the module
        """
        self.news_provider = news_provider
        self.config = config or {}
        self.default_config = {
            "cache_news": True,
            "cache_expiry": 3600,  # 1 hour
            "sentiment_threshold_positive": 0.2,
            "sentiment_threshold_negative": -0.2,
            "max_news_age": 86400,  # 24 hours
            "max_news_items": 100,
            "parallel_processing": True,
            "max_workers": 8,
            "nlp_models_dir": "./models/nlp",
            "use_advanced_nlp": False,
            "keywords_file": "./data/keywords.json",
            "company_mappings_file": "./data/company_mappings.json"
        }
        
        # Merge default config with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        # Initialize news cache
        self.news_cache = {}
        self.cache_timestamps = {}
        
        # Initialize sentiment analyzer
        self._initialize_nlp()
        
        # Load keywords and company mappings
        self.keywords = self._load_keywords()
        self.company_mappings = self._load_company_mappings()
        
        logger.info("NewsTrading module initialized")
        
    def set_news_provider(self, news_provider):
        """
        Set the news provider for the module.
        
        Parameters:
        -----------
        news_provider : object
            News provider object for fetching news data
        """
        self.news_provider = news_provider
        logger.info("News provider set")
        
    def _initialize_nlp(self):
        """Initialize NLP components."""
        try:
            # Download NLTK resources if needed
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize advanced NLP if configured
            if self.config["use_advanced_nlp"]:
                # This would typically involve loading more sophisticated NLP models
                # For now, we'll just log that this feature is not fully implemented
                logger.info("Advanced NLP not fully implemented")
                
            logger.info("NLP components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {str(e)}")
            
    def _load_keywords(self):
        """
        Load keywords for news filtering.
        
        Returns:
        --------
        dict
            Dictionary of keywords by category
        """
        keywords = {
            "bullish": ["growth", "beat", "exceed", "positive", "upgrade", "buy", "outperform", "strong", "launch", "partnership"],
            "bearish": ["decline", "miss", "below", "negative", "downgrade", "sell", "underperform", "weak", "layoff", "investigation"],
            "merger_acquisition": ["merger", "acquisition", "takeover", "buyout", "purchase", "acquire", "bid", "offer", "deal", "consolidation"],
            "earnings": ["earnings", "revenue", "profit", "loss", "eps", "quarter", "guidance", "forecast", "outlook", "results"],
            "product": ["product", "launch", "release", "announce", "unveil", "introduce", "new", "innovation", "patent", "approval"],
            "management": ["ceo", "executive", "management", "board", "director", "appoint", "resign", "leave", "join", "hire"],
            "legal": ["lawsuit", "litigation", "settlement", "court", "judge", "legal", "regulatory", "compliance", "investigation", "fine"],
            "economic": ["fed", "interest rate", "inflation", "gdp", "economy", "economic", "unemployment", "jobs", "growth", "recession"]
        }
        
        # Try to load keywords from file if it exists
        if os.path.exists(self.config["keywords_file"]):
            try:
                with open(self.config["keywords_file"], 'r') as f:
                    file_keywords = json.load(f)
                    
                # Merge with default keywords
                for category, words in file_keywords.items():
                    if category in keywords:
                        keywords[category].extend(words)
                    else:
                        keywords[category] = words
                        
                logger.info(f"Loaded keywords from {self.config['keywords_file']}")
                
            except Exception as e:
                logger.error(f"Error loading keywords from file: {str(e)}")
                
        return keywords
        
    def _load_company_mappings(self):
        """
        Load company name to ticker mappings.
        
        Returns:
        --------
        dict
            Dictionary mapping company names to tickers
        """
        mappings = {}
        
        # Try to load mappings from file if it exists
        if os.path.exists(self.config["company_mappings_file"]):
            try:
                with open(self.config["company_mappings_file"], 'r') as f:
                    mappings = json.load(f)
                    
                logger.info(f"Loaded company mappings from {self.config['company_mappings_file']}")
                
            except Exception as e:
                logger.error(f"Error loading company mappings from file: {str(e)}")
                
        return mappings
        
    def save_company_mappings(self):
        """Save company mappings to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config["company_mappings_file"]), exist_ok=True)
            
            with open(self.config["company_mappings_file"], 'w') as f:
                json.dump(self.company_mappings, f, indent=4)
                
            logger.info(f"Saved company mappings to {self.config['company_mappings_file']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving company mappings: {str(e)}")
            return False
            
    def add_company_mapping(self, company_name, ticker):
        """
        Add a company name to ticker mapping.
        
        Parameters:
        -----------
        company_name : str
            Company name
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            self.company_mappings[company_name.lower()] = ticker.upper()
            logger.info(f"Added company mapping: {company_name} -> {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding company mapping: {str(e)}")
            return False
            
    def get_news(self, query=None, tickers=None, categories=None, start_time=None, end_time=None, max_items=None):
        """
        Get news articles based on query parameters.
        
        Parameters:
        -----------
        query : str, optional
            Search query
        tickers : list, optional
            List of ticker symbols
        categories : list, optional
            List of news categories
        start_time : datetime, optional
            Start time for news search
        end_time : datetime, optional
            End time for news search
        max_items : int, optional
            Maximum number of news items to return
            
        Returns:
        --------
        list
            List of news articles
        """
        if self.news_provider is None:
            logger.error("No news provider set")
            return []
            
        max_items = max_items or self.config["max_news_items"]
        
        # Check cache if enabled
        if self.config["cache_news"]:
            cache_key = f"{query}_{tickers}_{categories}_{start_time}_{end_time}"
            current_time = time.time()
            
            if (cache_key in self.news_cache and 
                current_time - self.cache_timestamps.get(cache_key, 0) < self.config["cache_expiry"]):
                logger.debug(f"Using cached news for {cache_key}")
                return self.news_cache[cache_key]
        
        try:
            logger.info(f"Fetching news for query={query}, tickers={tickers}, categories={categories}")
            
            news = self.news_provider.get_news(
                query=query,
                tickers=tickers,
                categories=categories,
                start_time=start_time,
                end_time=end_time,
                max_items=max_items
            )
            
            # Cache the news if enabled
            if self.config["cache_news"]:
                cache_key = f"{query}_{tickers}_{categories}_{start_time}_{end_time}"
                self.news_cache[cache_key] = news
                self.cache_timestamps[cache_key] = time.time()
                
            return news
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
            
    def extract_entities(self, text):
        """
        Extract entities (companies, tickers, etc.) from text.
        
        Parameters:
        -----------
        text : str
            Text to extract entities from
            
        Returns:
        --------
        dict
            Dictionary of extracted entities
        """
        entities = {
            "companies": [],
            "tickers": [],
            "people": [],
            "locations": [],
            "organizations": []
        }
        
        try:
            # Extract tickers (simple regex for now)
            ticker_pattern = r'\$([A-Z]{1,5})'
            tickers = re.findall(ticker_pattern, text)
            entities["tickers"].extend(tickers)
            
            # Extract company names based on mappings
            for company, ticker in self.company_mappings.items():
                if company.lower() in text.lower():
                    entities["companies"].append(company)
                    if ticker not in entities["tickers"]:
                        entities["tickers"].append(ticker)
                        
            # Advanced NLP-based entity extraction would go here
            # For now, we'll just use a simple approach
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return entities
            
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Sentiment analysis results
        """
        try:
            # Use VADER for sentiment analysis
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            # Classify sentiment
            if sentiment["compound"] >= self.config["sentiment_threshold_positive"]:
                classification = "positive"
            elif sentiment["compound"] <= self.config["sentiment_threshold_negative"]:
                classification = "negative"
            else:
                classification = "neutral"
                
            return {
                "compound": sentiment["compound"],
                "positive": sentiment["pos"],
                "negative": sentiment["neg"],
                "neutral": sentiment["neu"],
                "classification": classification
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "compound": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "classification": "neutral"
            }
            
    def categorize_news(self, news_item):
        """
        Categorize a news item based on keywords.
        
        Parameters:
        -----------
        news_item : dict
            News item to categorize
            
        Returns:
        --------
        list
            List of categories
        """
        categories = []
        
        try:
            # Get text to analyze
            text = news_item.get("title", "") + " " + news_item.get("description", "")
            text = text.lower()
            
            # Check each category's keywords
            for category, keywords in self.keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        categories.append(category)
                        break
                        
            return categories
            
        except Exception as e:
            logger.error(f"Error categorizing news: {str(e)}")
            return categories
            
    def process_news_item(self, news_item):
        """
        Process a single news item.
        
        Parameters:
        -----------
        news_item : dict
            News item to process
            
        Returns:
        --------
        dict
            Processed news item with additional analysis
        """
        try:
            # Extract text for analysis
            title = news_item.get("title", "")
            description = news_item.get("description", "")
            content = news_item.get("content", "")
            
            analysis_text = f"{title} {description} {content}"
            
            # Extract entities
            entities = self.extract_entities(analysis_text)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(analysis_text)
            
            # Categorize news
            categories = self.categorize_news(news_item)
            
            # Add analysis to news item
            processed_item = news_item.copy()
            processed_item["entities"] = entities
            processed_item["sentiment"] = sentiment
            processed_item["categories"] = categories
            processed_item["processed_at"] = datetime.now().isoformat()
            
            return processed_item
            
        except Exception as e:
            logger.error(f"Error processing news item: {str(e)}")
            return news_item
            
    def process_news_batch(self, news_items):
        """
        Process a batch of news items.
        
        Parameters:
        -----------
        news_items : list
            List of news items to process
            
        Returns:
        --------
        list
            List of processed news items
        """
        processed_items = []
        
        if self.config["parallel_processing"]:
            # Use parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
                processed_items = list(executor.map(self.process_news_item, news_items))
        else:
            # Use sequential processing
            for item in news_items:
                processed_items.append(self.process_news_item(item))
                
        return processed_items
        
    def aggregate_sentiment(self, processed_news, ticker=None):
        """
        Aggregate sentiment across news items.
        
        Parameters:
        -----------
        processed_news : list
            List of processed news items
        ticker : str, optional
            Ticker symbol to filter by
            
        Returns:
        --------
        dict
            Aggregated sentiment metrics
        """
        if not processed_news:
            return {
                "overall_sentiment": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_count": 0,
                "sentiment_by_category": {},
                "sentiment_trend": []
            }
            
        # Filter by ticker if provided
        if ticker:
            filtered_news = [
                item for item in processed_news 
                if ticker.upper() in [t.upper() for t in item.get("entities", {}).get("tickers", [])]
            ]
        else:
            filtered_news = processed_news
            
        if not filtered_news:
            return {
                "overall_sentiment": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_count": 0,
                "sentiment_by_category": {},
                "sentiment_trend": []
            }
            
        # Calculate overall sentiment
        sentiment_scores = [item.get("sentiment", {}).get("compound", 0) for item in filtered_news]
        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Count by classification
        classifications = [item.get("sentiment", {}).get("classification", "neutral") for item in filtered_news]
        positive_count = classifications.count("positive")
        negative_count = classifications.count("negative")
        neutral_count = classifications.count("neutral")
        
        # Calculate sentiment by category
        sentiment_by_category = defaultdict(list)
        for item in filtered_news:
            for category in item.get("categories", []):
                sentiment_by_category[category].append(item.get("sentiment", {}).get("compound", 0))
                
        # Calculate average sentiment by category
        avg_sentiment_by_category = {
            category: sum(scores) / len(scores) 
            for category, scores in sentiment_by_category.items()
        }
        
        # Calculate sentiment trend over time
        # Sort news by publication date
        sorted_news = sorted(
            filtered_news, 
            key=lambda x: x.get("published_at", datetime.now().isoformat())
        )
        
        # Group by day and calculate average sentiment
        sentiment_trend = []
        day_sentiments = defaultdict(list)
        
        for item in sorted_news:
            published_at = item.get("published_at")
            if published_at:
                try:
                    date = datetime.fromisoformat(published_at).date().isoformat()
                    day_sentiments[date].append(item.get("sentiment", {}).get("compound", 0))
                except (ValueError, TypeError):
                    pass
                    
        for date, scores in day_sentiments.items():
            sentiment_trend.append({
                "date": date,
                "sentiment": sum(scores) / len(scores),
                "count": len(scores)
            })
            
        return {
            "overall_sentiment": overall_sentiment,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "total_count": len(filtered_news),
            "sentiment_by_category": avg_sentiment_by_category,
            "sentiment_trend": sentiment_trend
        }
        
    def generate_news_signals(self, processed_news, ticker=None, threshold=None):
        """
        Generate trading signals based on news sentiment.
        
        Parameters:
        -----------
        processed_news : list
            List of processed news items
        ticker : str, optional
            Ticker symbol to filter by
        threshold : float, optional
            Sentiment threshold for generating signals
            
        Returns:
        --------
        list
            List of trading signals
        """
        signals = []
        
        # Set threshold
        positive_threshold = threshold or self.config["sentiment_threshold_positive"]
        negative_threshold = -positive_threshold if threshold else self.config["sentiment_threshold_negative"]
        
        # Filter by ticker if provided
        if ticker:
            filtered_news = [
                item for item in processed_news 
                if ticker.upper() in [t.upper() for t in item.get("entities", {}).get("tickers", [])]
            ]
        else:
            filtered_news = processed_news
            
        # Generate signals for each news item
        for item in filtered_news:
            # Get sentiment
            sentiment = item.get("sentiment", {}).get("compound", 0)
            
            # Get tickers
            tickers = item.get("entities", {}).get("tickers", [])
            
            # Filter by ticker if provided
            if ticker and ticker.upper() not in [t.upper() for t in tickers]:
                continue
                
            # Generate signal based on sentiment
            if sentiment >= positive_threshold:
                signal_type = "buy"
            elif sentiment <= negative_threshold:
                signal_type = "sell"
            else:
                continue  # No signal for neutral sentiment
                
            # Create signal for each ticker
            for ticker_symbol in tickers:
                signal = {
                    "ticker": ticker_symbol,
                    "signal_type": signal_type,
                    "sentiment_score": sentiment,
                    "news_id": item.get("id"),
                    "news_title": item.get("title"),
                    "news_url": item.get("url"),
                    "published_at": item.get("published_at"),
                    "generated_at": datetime.now().isoformat(),
                    "categories": item.get("categories", []),
                    "confidence": abs(sentiment)  # Higher absolute sentiment = higher confidence
                }
                
                signals.append(signal)
                
        return signals
        
    def filter_signals(self, signals, min_confidence=0.3, categories=None, max_age=None):
        """
        Filter trading signals based on criteria.
        
        Parameters:
        -----------
        signals : list
            List of trading signals
        min_confidence : float, optional
            Minimum confidence level
        categories : list, optional
            List of categories to include
        max_age : int, optional
            Maximum age of signals in seconds
            
        Returns:
        --------
        list
            Filtered trading signals
        """
        filtered_signals = []
        
        max_age = max_age or self.config["max_news_age"]
        current_time = datetime.now()
        
        for signal in signals:
            # Check confidence
            if signal.get("confidence", 0) < min_confidence:
                continue
                
            # Check categories
            if categories and not any(category in signal.get("categories", []) for category in categories):
                continue
                
            # Check age
            if "published_at" in signal:
                try:
                    published_at = datetime.fromisoformat(signal["published_at"])
                    age = (current_time - published_at).total_seconds()
                    
                    if age > max_age:
                        continue
                except (ValueError, TypeError):
                    pass
                    
            filtered_signals.append(signal)
            
        return filtered_signals
        
    def rank_signals(self, signals, ranking_method="confidence"):
        """
        Rank trading signals.
        
        Parameters:
        -----------
        signals : list
            List of trading signals
        ranking_method : str, optional
            Method to use for ranking
            
        Returns:
        --------
        list
            Ranked trading signals
        """
        if not signals:
            return []
            
        if ranking_method == "confidence":
            # Rank by confidence
            return sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)
            
        elif ranking_method == "recency":
            # Rank by recency
            return sorted(
                signals, 
                key=lambda x: datetime.fromisoformat(x.get("published_at", "2000-01-01T00:00:00")) 
                if "published_at" in x else datetime.min,
                reverse=True
            )
            
        elif ranking_method == "combined":
            # Rank by combination of confidence and recency
            for signal in signals:
                if "published_at" in signal:
                    try:
                        published_at = datetime.fromisoformat(signal["published_at"])
                        age_hours = (datetime.now() - published_at).total_seconds() / 3600
                        recency_score = max(0, 1 - (age_hours / 24))  # 0-1 score, 1 for very recent
                        
                        # Combined score: 70% confidence, 30% recency
                        signal["combined_score"] = 0.7 * signal.get("confidence", 0) + 0.3 * recency_score
                    except (ValueError, TypeError):
                        signal["combined_score"] = 0.7 * signal.get("confidence", 0)
                else:
                    signal["combined_score"] = 0.7 * signal.get("confidence", 0)
                    
            return sorted(signals, key=lambda x: x.get("combined_score", 0), reverse=True)
            
        else:
            # Default to confidence ranking
            return sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)
            
    def generate_news_summary(self, processed_news, ticker=None):
        """
        Generate a summary of news for a ticker.
        
        Parameters:
        -----------
        processed_news : list
            List of processed news items
        ticker : str, optional
            Ticker symbol to filter by
            
        Returns:
        --------
        dict
            News summary
        """
        # Filter by ticker if provided
        if ticker:
            filtered_news = [
                item for item in processed_news 
                if ticker.upper() in [t.upper() for t in item.get("entities", {}).get("tickers", [])]
            ]
        else:
            filtered_news = processed_news
            
        if not filtered_news:
            return {
                "ticker": ticker,
                "news_count": 0,
                "sentiment_summary": {},
                "top_positive": [],
                "top_negative": [],
                "recent_news": [],
                "categories": {}
            }
            
        # Get sentiment summary
        sentiment_summary = self.aggregate_sentiment(filtered_news, ticker)
        
        # Get top positive and negative news
        sorted_by_sentiment = sorted(
            filtered_news, 
            key=lambda x: x.get("sentiment", {}).get("compound", 0),
            reverse=True
        )
        
        top_positive = sorted_by_sentiment[:3]
        top_negative = sorted_by_sentiment[-3:] if len(sorted_by_sentiment) >= 3 else []
        
        # Get most recent news
        sorted_by_recency = sorted(
            filtered_news, 
            key=lambda x: x.get("published_at", "2000-01-01T00:00:00") 
            if "published_at" in x else "2000-01-01T00:00:00",
            reverse=True
        )
        
        recent_news = sorted_by_recency[:5]
        
        # Count news by category
        category_counts = defaultdict(int)
        for item in filtered_news:
            for category in item.get("categories", []):
                category_counts[category] += 1
                
        return {
            "ticker": ticker,
            "news_count": len(filtered_news),
            "sentiment_summary": sentiment_summary,
            "top_positive": [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "published_at": item.get("published_at"),
                    "sentiment": item.get("sentiment", {}).get("compound", 0)
                } 
                for item in top_positive
            ],
            "top_negative": [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "published_at": item.get("published_at"),
                    "sentiment": item.get("sentiment", {}).get("compound", 0)
                } 
                for item in top_negative
            ],
            "recent_news": [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "published_at": item.get("published_at"),
                    "sentiment": item.get("sentiment", {}).get("compound", 0)
                } 
                for item in recent_news
            ],
            "categories": dict(category_counts)
        }
        
    def create_news_based_strategy(self, name, description=None):
        """
        Create a news-based trading strategy.
        
        Parameters:
        -----------
        name : str
            Name of the strategy
        description : str, optional
            Description of the strategy
            
        Returns:
        --------
        dict
            Strategy definition
        """
        strategy = {
            "name": name,
            "description": description or f"News-based strategy: {name}",
            "created_at": datetime.now().isoformat(),
            "type": "news_based",
            "parameters": {
                "sentiment_threshold": 0.3,
                "min_confidence": 0.3,
                "max_age": 86400,  # 24 hours
                "categories": [],
                "ranking_method": "combined",
                "position_sizing": "fixed",
                "position_size": 0.02,  # 2% of portfolio
                "stop_loss": 0.02,  # 2% stop loss
                "take_profit": 0.04,  # 4% take profit
                "max_positions": 5,
                "max_positions_per_sector": 2
            },
            "filters": {
                "min_price": 5.0,
                "min_volume": 500000,
                "exclude_tickers": []
            },
            "schedule": {
                "active": True,
                "run_at_market_open": True,
                "run_at_market_close": False,
                "custom_times": [],
                "max_signals_per_day": 10
            }
        }
        
        return strategy
        
    def backtest_news_strategy(self, strategy, historical_news, price_data, start_date=None, end_date=None):
        """
        Backtest a news-based trading strategy.
        
        Parameters:
        -----------
        strategy : dict
            Strategy definition
        historical_news : list
            List of historical news items
        price_data : dict
            Dictionary of price data by ticker
        start_date : datetime, optional
            Start date for backtest
        end_date : datetime, optional
            End date for backtest
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Process historical news
        processed_news = self.process_news_batch(historical_news)
        
        # Filter by date range
        if start_date or end_date:
            filtered_news = []
            for item in processed_news:
                if "published_at" in item:
                    try:
                        published_at = datetime.fromisoformat(item["published_at"])
                        
                        if start_date and published_at < start_date:
                            continue
                            
                        if end_date and published_at > end_date:
                            continue
                            
                        filtered_news.append(item)
                    except (ValueError, TypeError):
                        pass
                        
            processed_news = filtered_news
            
        # Generate signals
        all_signals = self.generate_news_signals(
            processed_news,
            threshold=strategy["parameters"]["sentiment_threshold"]
        )
        
        # Filter signals
        filtered_signals = self.filter_signals(
            all_signals,
            min_confidence=strategy["parameters"]["min_confidence"],
            categories=strategy["parameters"]["categories"],
            max_age=strategy["parameters"]["max_age"]
        )
        
        # Rank signals
        ranked_signals = self.rank_signals(
            filtered_signals,
            ranking_method=strategy["parameters"]["ranking_method"]
        )
        
        # Simulate trading
        portfolio = {
            "cash": 100000,  # Start with $100,000
            "positions": {},
            "trades": [],
            "equity_curve": []
        }
        
        current_date = None
        daily_signals = []
        
        # Sort signals by date
        sorted_signals = sorted(
            ranked_signals,
            key=lambda x: datetime.fromisoformat(x.get("published_at", "2000-01-01T00:00:00"))
            if "published_at" in x else datetime.min
        )
        
        for signal in sorted_signals:
            if "published_at" not in signal:
                continue
                
            try:
                signal_date = datetime.fromisoformat(signal["published_at"]).date()
                
                # If new day, process previous day's signals
                if current_date is not None and signal_date != current_date:
                    self._process_daily_signals(daily_signals, portfolio, price_data, strategy, current_date)
                    daily_signals = []
                    
                current_date = signal_date
                daily_signals.append(signal)
                
            except (ValueError, TypeError):
                pass
                
        # Process last day's signals
        if daily_signals:
            self._process_daily_signals(daily_signals, portfolio, price_data, strategy, current_date)
            
        # Calculate performance metrics
        performance = self._calculate_performance(portfolio)
        
        return {
            "strategy": strategy,
            "performance": performance,
            "trades": portfolio["trades"],
            "equity_curve": portfolio["equity_curve"],
            "final_portfolio": {
                "cash": portfolio["cash"],
                "positions": portfolio["positions"],
                "total_value": portfolio["cash"] + sum(
                    pos["shares"] * self._get_latest_price(ticker, price_data)
                    for ticker, pos in portfolio["positions"].items()
                )
            }
        }
        
    def _process_daily_signals(self, signals, portfolio, price_data, strategy, date):
        """
        Process signals for a single day in backtest.
        
        Parameters:
        -----------
        signals : list
            List of signals for the day
        portfolio : dict
            Portfolio state
        price_data : dict
            Dictionary of price data by ticker
        strategy : dict
            Strategy definition
        date : date
            Current date
        """
        # Limit number of signals per day
        max_signals = strategy["schedule"]["max_signals_per_day"]
        if len(signals) > max_signals:
            signals = signals[:max_signals]
            
        # Process each signal
        for signal in signals:
            ticker = signal["ticker"]
            signal_type = signal["signal_type"]
            
            # Skip if ticker not in price data
            if ticker not in price_data:
                continue
                
            # Get price data for the ticker
            ticker_prices = price_data[ticker]
            
            # Find the closest trading day on or after the signal date
            trading_dates = sorted(ticker_prices.keys())
            signal_date_str = date.isoformat()
            
            execution_date = None
            for trading_date in trading_dates:
                if trading_date >= signal_date_str:
                    execution_date = trading_date
                    break
                    
            if execution_date is None:
                continue  # No future trading days available
                
            # Get execution price (use open price of the next trading day)
            execution_price = ticker_prices[execution_date]["open"]
            
            # Apply price and volume filters
            if execution_price < strategy["filters"]["min_price"]:
                continue
                
            if ticker_prices[execution_date]["volume"] < strategy["filters"]["min_volume"]:
                continue
                
            # Check if ticker is excluded
            if ticker in strategy["filters"]["exclude_tickers"]:
                continue
                
            # Process buy signals
            if signal_type == "buy":
                # Check if we already have a position in this ticker
                if ticker in portfolio["positions"]:
                    continue
                    
                # Check if we've reached the maximum number of positions
                if len(portfolio["positions"]) >= strategy["parameters"]["max_positions"]:
                    continue
                    
                # Calculate position size
                if strategy["parameters"]["position_sizing"] == "fixed":
                    position_value = portfolio["cash"] * strategy["parameters"]["position_size"]
                else:
                    # Other position sizing methods would go here
                    position_value = portfolio["cash"] * strategy["parameters"]["position_size"]
                    
                # Calculate number of shares
                shares = int(position_value / execution_price)
                
                if shares == 0:
                    continue
                    
                # Calculate total cost
                cost = shares * execution_price
                
                # Check if we have enough cash
                if cost > portfolio["cash"]:
                    shares = int(portfolio["cash"] / execution_price)
                    cost = shares * execution_price
                    
                if shares == 0:
                    continue
                    
                # Add position to portfolio
                portfolio["positions"][ticker] = {
                    "shares": shares,
                    "entry_price": execution_price,
                    "entry_date": execution_date,
                    "stop_loss": execution_price * (1 - strategy["parameters"]["stop_loss"]),
                    "take_profit": execution_price * (1 + strategy["parameters"]["take_profit"]),
                    "signal_id": signal.get("news_id")
                }
                
                # Deduct cost from cash
                portfolio["cash"] -= cost
                
                # Record trade
                portfolio["trades"].append({
                    "ticker": ticker,
                    "action": "buy",
                    "shares": shares,
                    "price": execution_price,
                    "date": execution_date,
                    "cost": cost,
                    "signal_id": signal.get("news_id"),
                    "signal_confidence": signal.get("confidence")
                })
                
            # Process sell signals
            elif signal_type == "sell":
                # Check if we have a position in this ticker
                if ticker not in portfolio["positions"]:
                    continue
                    
                # Get position details
                position = portfolio["positions"][ticker]
                shares = position["shares"]
                entry_price = position["entry_price"]
                
                # Calculate proceeds
                proceeds = shares * execution_price
                
                # Add proceeds to cash
                portfolio["cash"] += proceeds
                
                # Calculate profit/loss
                pnl = proceeds - (shares * entry_price)
                pnl_percent = (execution_price / entry_price - 1) * 100
                
                # Remove position from portfolio
                del portfolio["positions"][ticker]
                
                # Record trade
                portfolio["trades"].append({
                    "ticker": ticker,
                    "action": "sell",
                    "shares": shares,
                    "price": execution_price,
                    "date": execution_date,
                    "proceeds": proceeds,
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "signal_id": signal.get("news_id"),
                    "signal_confidence": signal.get("confidence")
                })
                
        # Check stop loss and take profit for existing positions
        positions_to_sell = []
        
        for ticker, position in portfolio["positions"].items():
            if ticker not in price_data:
                continue
                
            ticker_prices = price_data[ticker]
            
            # Find the closest trading day on or after the current date
            trading_dates = sorted(ticker_prices.keys())
            current_date_str = date.isoformat()
            
            execution_date = None
            for trading_date in trading_dates:
                if trading_date >= current_date_str:
                    execution_date = trading_date
                    break
                    
            if execution_date is None:
                continue  # No future trading days available
                
            # Get high and low prices for the day
            high_price = ticker_prices[execution_date]["high"]
            low_price = ticker_prices[execution_date]["low"]
            close_price = ticker_prices[execution_date]["close"]
            
            # Check stop loss
            if low_price <= position["stop_loss"]:
                # Mark position for selling at stop loss price
                positions_to_sell.append({
                    "ticker": ticker,
                    "price": position["stop_loss"],
                    "date": execution_date,
                    "reason": "stop_loss"
                })
                
            # Check take profit
            elif high_price >= position["take_profit"]:
                # Mark position for selling at take profit price
                positions_to_sell.append({
                    "ticker": ticker,
                    "price": position["take_profit"],
                    "date": execution_date,
                    "reason": "take_profit"
                })
                
        # Process positions to sell
        for sell_info in positions_to_sell:
            ticker = sell_info["ticker"]
            execution_price = sell_info["price"]
            execution_date = sell_info["date"]
            reason = sell_info["reason"]
            
            # Get position details
            position = portfolio["positions"][ticker]
            shares = position["shares"]
            entry_price = position["entry_price"]
            
            # Calculate proceeds
            proceeds = shares * execution_price
            
            # Add proceeds to cash
            portfolio["cash"] += proceeds
            
            # Calculate profit/loss
            pnl = proceeds - (shares * entry_price)
            pnl_percent = (execution_price / entry_price - 1) * 100
            
            # Remove position from portfolio
            del portfolio["positions"][ticker]
            
            # Record trade
            portfolio["trades"].append({
                "ticker": ticker,
                "action": "sell",
                "shares": shares,
                "price": execution_price,
                "date": execution_date,
                "proceeds": proceeds,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "reason": reason,
                "signal_id": position.get("signal_id")
            })
            
        # Update equity curve
        portfolio_value = portfolio["cash"]
        for ticker, position in portfolio["positions"].items():
            if ticker in price_data:
                latest_price = self._get_latest_price(ticker, price_data, date)
                portfolio_value += position["shares"] * latest_price
                
        portfolio["equity_curve"].append({
            "date": date.isoformat(),
            "value": portfolio_value
        })
        
    def _get_latest_price(self, ticker, price_data, date=None):
        """
        Get the latest price for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        price_data : dict
            Dictionary of price data by ticker
        date : date, optional
            Date to get price for
            
        Returns:
        --------
        float
            Latest price
        """
        if ticker not in price_data:
            return 0
            
        ticker_prices = price_data[ticker]
        
        if date is None:
            # Get the most recent price
            latest_date = max(ticker_prices.keys())
            return ticker_prices[latest_date]["close"]
            
        # Find the closest trading day on or before the given date
        date_str = date.isoformat()
        trading_dates = sorted(ticker_prices.keys())
        
        closest_date = None
        for trading_date in trading_dates:
            if trading_date <= date_str:
                closest_date = trading_date
            else:
                break
                
        if closest_date is None:
            return 0
            
        return ticker_prices[closest_date]["close"]
        
    def _calculate_performance(self, portfolio):
        """
        Calculate performance metrics for a backtest.
        
        Parameters:
        -----------
        portfolio : dict
            Portfolio state
            
        Returns:
        --------
        dict
            Performance metrics
        """
        if not portfolio["trades"] or not portfolio["equity_curve"]:
            return {
                "total_return": 0,
                "annualized_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "num_trades": 0
            }
            
        # Calculate total return
        initial_value = 100000  # Starting portfolio value
        final_value = portfolio["equity_curve"][-1]["value"]
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate annualized return
        start_date = datetime.fromisoformat(portfolio["equity_curve"][0]["date"])
        end_date = datetime.fromisoformat(portfolio["equity_curve"][-1]["date"])
        days = (end_date - start_date).days
        if days > 0:
            annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        else:
            annualized_return = 0
            
        # Calculate Sharpe ratio
        if len(portfolio["equity_curve"]) > 1:
            returns = []
            for i in range(1, len(portfolio["equity_curve"])):
                prev_value = portfolio["equity_curve"][i-1]["value"]
                curr_value = portfolio["equity_curve"][i]["value"]
                daily_return = curr_value / prev_value - 1
                returns.append(daily_return)
                
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
            
        # Calculate maximum drawdown
        max_drawdown = 0
        peak = initial_value
        
        for point in portfolio["equity_curve"]:
            value = point["value"]
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
                
        # Calculate trade statistics
        num_trades = len([t for t in portfolio["trades"] if t["action"] == "sell"])
        winning_trades = [t for t in portfolio["trades"] if t["action"] == "sell" and t.get("pnl", 0) > 0]
        losing_trades = [t for t in portfolio["trades"] if t["action"] == "sell" and t.get("pnl", 0) <= 0]
        
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        
        total_profit = sum(t.get("pnl", 0) for t in winning_trades)
        total_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "num_trades": num_trades
        }
