"""
Enhanced Natural Language Market Analysis Module for Gemma Advanced Trading System

This module implements comprehensive natural language processing capabilities for analyzing
financial news, earnings reports, social media sentiment, and other textual data
to provide qualitative insights for trading decisions.

The module leverages Gemma 3's advanced NLP capabilities to extract meaningful
information from unstructured text data and integrate it with quantitative analysis.
"""

import os
import logging
import json
import datetime
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from collections import Counter

# Import Gemma 3 integration architecture
from backend.gemma3_integration.architecture_enhanced import GemmaCore, PromptEngine, DataIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class NewsAnalyzer:
    """
    Analyzes financial news articles to extract relevant information and sentiment.
    
    This class processes news articles related to specific tickers or the broader
    market to identify key events, sentiment, and potential trading catalysts.
    """
    
    def __init__(self, gemma_core: GemmaCore):
        """
        Initialize the NewsAnalyzer.
        
        Parameters:
        -----------
        gemma_core : GemmaCore
            Instance of GemmaCore for accessing Gemma 3 capabilities.
        """
        self.logger = logging.getLogger("GemmaTrading.NewsAnalyzer")
        self.gemma_core = gemma_core
        
        self.logger.info("Initialized NewsAnalyzer")
    
    def analyze_articles(self, ticker: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a list of news articles related to a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        articles : List[Dict[str, Any]]
            List of news articles to analyze. Each article should have 'title',
            'content', 'source', and 'date' fields.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including sentiment, key events, and potential impact.
        """
        self.logger.info(f"Analyzing {len(articles)} news articles for {ticker}")
        
        if not articles:
            return {
                "ticker": ticker,
                "sentiment": "neutral",
                "confidence": 0.0,
                "key_events": [],
                "potential_impact": "No news articles to analyze",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Prepare data for Gemma 3
        articles_text = "\n\n".join([
            f"Title: {article.get('title', 'No Title')}\n"
            f"Source: {article.get('source', 'Unknown')}\n"
            f"Date: {article.get('date', 'Unknown')}\n"
            f"Content: {article.get('content', 'No Content')}"
            for article in articles
        ])
        
        # Generate prompt for news analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "news_analysis",
            ticker=ticker,
            articles=articles_text
        )
        
        # Get analysis from Gemma 3
        analysis_result = self.gemma_core.generate_response(prompt)
        
        # Parse the analysis result
        try:
            analysis = json.loads(analysis_result)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemma 3 response as JSON: {analysis_result}")
            
            # Attempt to extract structured information using regex
            sentiment_match = re.search(r"sentiment[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            sentiment = sentiment_match.group(1) if sentiment_match else "neutral"
            
            confidence_match = re.search(r"confidence[\"']?\s*:\s*([0-9.]+)", analysis_result)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # Extract key events using a simple heuristic
            key_events = []
            events_section = re.search(r"key_events[\"']?\s*:\s*\[(.*?)\]", analysis_result, re.DOTALL)
            if events_section:
                events_text = events_section.group(1)
                event_matches = re.findall(r"[\"']([^\"']+)[\"']", events_text)
                key_events = event_matches
            
            impact_match = re.search(r"potential_impact[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            impact = impact_match.group(1) if impact_match else "Unknown"
            
            analysis = {
                "ticker": ticker,
                "sentiment": sentiment,
                "confidence": confidence,
                "key_events": key_events,
                "potential_impact": impact,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return analysis
    
    def analyze_breaking_news(self, ticker: str, news_text: str) -> Dict[str, Any]:
        """
        Analyze breaking news for immediate trading impact.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        news_text : str
            Breaking news text to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including sentiment, urgency, and potential impact.
        """
        self.logger.info(f"Analyzing breaking news for {ticker}")
        
        # Generate prompt for breaking news analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "breaking_news_analysis",
            ticker=ticker,
            news_text=news_text
        )
        
        # Get analysis from Gemma 3
        analysis_result = self.gemma_core.generate_response(prompt)
        
        # Parse the analysis result
        try:
            analysis = json.loads(analysis_result)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemma 3 response as JSON: {analysis_result}")
            
            # Extract structured information using regex
            sentiment_match = re.search(r"sentiment[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            sentiment = sentiment_match.group(1) if sentiment_match else "neutral"
            
            urgency_match = re.search(r"urgency[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            urgency = urgency_match.group(1) if urgency_match else "low"
            
            impact_match = re.search(r"potential_impact[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            impact = impact_match.group(1) if impact_match else "Unknown"
            
            analysis = {
                "ticker": ticker,
                "sentiment": sentiment,
                "urgency": urgency,
                "potential_impact": impact,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return analysis


class EarningsAnalyzer:
    """
    Analyzes earnings reports and conference calls to extract insights.
    
    This class processes earnings reports, conference call transcripts, and
    related financial disclosures to identify key metrics, management sentiment,
    and future guidance.
    """
    
    def __init__(self, gemma_core: GemmaCore):
        """
        Initialize the EarningsAnalyzer.
        
        Parameters:
        -----------
        gemma_core : GemmaCore
            Instance of GemmaCore for accessing Gemma 3 capabilities.
        """
        self.logger = logging.getLogger("GemmaTrading.EarningsAnalyzer")
        self.gemma_core = gemma_core
        
        self.logger.info("Initialized EarningsAnalyzer")
    
    def analyze_earnings_report(self, ticker: str, report_text: str) -> Dict[str, Any]:
        """
        Analyze an earnings report for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        report_text : str
            Text of the earnings report to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including key metrics, sentiment, and guidance.
        """
        self.logger.info(f"Analyzing earnings report for {ticker}")
        
        # Generate prompt for earnings report analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "earnings_report_analysis",
            ticker=ticker,
            report_text=report_text
        )
        
        # Get analysis from Gemma 3
        analysis_result = self.gemma_core.generate_response(prompt)
        
        # Parse the analysis result
        try:
            analysis = json.loads(analysis_result)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemma 3 response as JSON: {analysis_result}")
            
            # Extract structured information using regex
            eps_match = re.search(r"eps[\"']?\s*:\s*([0-9.-]+)", analysis_result)
            eps = float(eps_match.group(1)) if eps_match else None
            
            eps_estimate_match = re.search(r"eps_estimate[\"']?\s*:\s*([0-9.-]+)", analysis_result)
            eps_estimate = float(eps_estimate_match.group(1)) if eps_estimate_match else None
            
            revenue_match = re.search(r"revenue[\"']?\s*:\s*([0-9.]+)", analysis_result)
            revenue = float(revenue_match.group(1)) if revenue_match else None
            
            revenue_estimate_match = re.search(r"revenue_estimate[\"']?\s*:\s*([0-9.]+)", analysis_result)
            revenue_estimate = float(revenue_estimate_match.group(1)) if revenue_estimate_match else None
            
            sentiment_match = re.search(r"management_sentiment[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            sentiment = sentiment_match.group(1) if sentiment_match else "neutral"
            
            guidance_match = re.search(r"guidance[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            guidance = guidance_match.group(1) if guidance_match else "neutral"
            
            analysis = {
                "ticker": ticker,
                "eps": eps,
                "eps_estimate": eps_estimate,
                "revenue": revenue,
                "revenue_estimate": revenue_estimate,
                "management_sentiment": sentiment,
                "guidance": guidance,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return analysis
    
    def analyze_conference_call(self, ticker: str, transcript_text: str) -> Dict[str, Any]:
        """
        Analyze an earnings conference call transcript.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        transcript_text : str
            Text of the conference call transcript to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including management tone, key topics, and insights.
        """
        self.logger.info(f"Analyzing conference call transcript for {ticker}")
        
        # Generate prompt for conference call analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "conference_call_analysis",
            ticker=ticker,
            transcript_text=transcript_text
        )
        
        # Get analysis from Gemma 3
        analysis_result = self.gemma_core.generate_response(prompt)
        
        # Parse the analysis result
        try:
            analysis = json.loads(analysis_result)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemma 3 response as JSON: {analysis_result}")
            
            # Extract structured information using regex
            tone_match = re.search(r"management_tone[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            tone = tone_match.group(1) if tone_match else "neutral"
            
            # Extract key topics using a simple heuristic
            topics = []
            topics_section = re.search(r"key_topics[\"']?\s*:\s*\[(.*?)\]", analysis_result, re.DOTALL)
            if topics_section:
                topics_text = topics_section.group(1)
                topic_matches = re.findall(r"[\"']([^\"']+)[\"']", topics_text)
                topics = topic_matches
            
            insights_match = re.search(r"key_insights[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            insights = insights_match.group(1) if insights_match else "No significant insights"
            
            analysis = {
                "ticker": ticker,
                "management_tone": tone,
                "key_topics": topics,
                "key_insights": insights,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return analysis


class SocialMediaAnalyzer:
    """
    Analyzes social media sentiment and discussions related to financial markets.
    
    This class processes social media posts, comments, and discussions to gauge
    retail investor sentiment, identify trending topics, and detect potential
    market-moving events.
    """
    
    def __init__(self, gemma_core: GemmaCore):
        """
        Initialize the SocialMediaAnalyzer.
        
        Parameters:
        -----------
        gemma_core : GemmaCore
            Instance of GemmaCore for accessing Gemma 3 capabilities.
        """
        self.logger = logging.getLogger("GemmaTrading.SocialMediaAnalyzer")
        self.gemma_core = gemma_core
        
        self.logger.info("Initialized SocialMediaAnalyzer")
    
    def analyze_social_sentiment(self, ticker: str, posts: List[Dict[str, Any]] = None, query: str = None) -> Dict[str, Any]:
        """
        Analyze social media sentiment for a ticker or query.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        posts : List[Dict[str, Any]], optional
            List of social media posts to analyze. Each post should have 'text',
            'source', 'user', and 'timestamp' fields.
        query : str, optional
            Query text to analyze if posts are not provided.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including sentiment, volume, and key topics.
        """
        self.logger.info(f"Analyzing social media sentiment for {ticker}")
        
        if not posts and not query:
            return {
                "ticker": ticker,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "volume": 0,
                "key_topics": [],
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Prepare data for Gemma 3
        if posts:
            posts_text = "\n\n".join([
                f"Source: {post.get('source', 'Unknown')}\n"
                f"User: {post.get('user', 'Anonymous')}\n"
                f"Time: {post.get('timestamp', 'Unknown')}\n"
                f"Text: {post.get('text', 'No Text')}"
                for post in posts
            ])
            
            # Generate prompt for social media analysis
            prompt = self.gemma_core.prompt_engine.generate_prompt(
                "social_media_analysis",
                ticker=ticker,
                posts=posts_text
            )
        else:
            # Generate prompt for query-based social sentiment
            prompt = self.gemma_core.prompt_engine.generate_prompt(
                "social_sentiment_query",
                ticker=ticker,
                query=query
            )
        
        # Get analysis from Gemma 3
        analysis_result = self.gemma_core.generate_response(prompt)
        
        # Parse the analysis result
        try:
            analysis = json.loads(analysis_result)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemma 3 response as JSON: {analysis_result}")
            
            # Extract structured information using regex
            sentiment_match = re.search(r"sentiment[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            sentiment = sentiment_match.group(1) if sentiment_match else "neutral"
            
            score_match = re.search(r"sentiment_score[\"']?\s*:\s*([0-9.-]+)", analysis_result)
            score = float(score_match.group(1)) if score_match else 0.0
            
            volume_match = re.search(r"volume[\"']?\s*:\s*([0-9]+)", analysis_result)
            volume = int(volume_match.group(1)) if volume_match else 0
            
            # Extract key topics using a simple heuristic
            topics = []
            topics_section = re.search(r"key_topics[\"']?\s*:\s*\[(.*?)\]", analysis_result, re.DOTALL)
            if topics_section:
                topics_text = topics_section.group(1)
                topic_matches = re.findall(r"[\"']([^\"']+)[\"']", topics_text)
                topics = topic_matches
            
            analysis = {
                "ticker": ticker,
                "sentiment": sentiment,
                "sentiment_score": score,
                "volume": volume,
                "key_topics": topics,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return analysis
    
    def detect_social_anomalies(self, ticker: str, historical_data: Dict[str, Any], current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in social media activity for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        historical_data : Dict[str, Any]
            Historical social media data for the ticker.
        current_data : Dict[str, Any]
            Current social media data for the ticker.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including anomaly detection and potential causes.
        """
        self.logger.info(f"Detecting social media anomalies for {ticker}")
        
        # Generate prompt for anomaly detection
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "social_anomaly_detection",
            ticker=ticker,
            historical_data=json.dumps(historical_data),
            current_data=json.dumps(current_data)
        )
        
        # Get analysis from Gemma 3
        analysis_result = self.gemma_core.generate_response(prompt)
        
        # Parse the analysis result
        try:
            analysis = json.loads(analysis_result)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemma 3 response as JSON: {analysis_result}")
            
            # Extract structured information using regex
            anomaly_match = re.search(r"anomaly_detected[\"']?\s*:\s*(true|false)", analysis_result, re.IGNORECASE)
            anomaly_detected = anomaly_match.group(1).lower() == "true" if anomaly_match else False
            
            score_match = re.search(r"anomaly_score[\"']?\s*:\s*([0-9.-]+)", analysis_result)
            score = float(score_match.group(1)) if score_match else 0.0
            
            cause_match = re.search(r"potential_cause[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            cause = cause_match.group(1) if cause_match else "Unknown"
            
            analysis = {
                "ticker": ticker,
                "anomaly_detected": anomaly_detected,
                "anomaly_score": score,
                "potential_cause": cause,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return analysis


class MarketNarrativeGenerator:
    """
    Generates comprehensive market narratives based on various data sources.
    
    This class synthesizes information from news, earnings, social media, and
    other sources to create coherent market narratives that explain current
    market conditions and potential future scenarios.
    """
    
    def __init__(self, gemma_core: GemmaCore):
        """
        Initialize the MarketNarrativeGenerator.
        
        Parameters:
        -----------
        gemma_core : GemmaCore
            Instance of GemmaCore for accessing Gemma 3 capabilities.
        """
        self.logger = logging.getLogger("GemmaTrading.MarketNarrativeGenerator")
        self.gemma_core = gemma_core
        
        self.logger.info("Initialized MarketNarrativeGenerator")
    
    def generate_asset_narrative(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a narrative for a specific asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        data : Dict[str, Any]
            Data to use for narrative generation, including news, earnings,
            social sentiment, and technical analysis.
            
        Returns:
        --------
        Dict[str, Any]
            Generated narrative including summary, key factors, and scenarios.
        """
        self.logger.info(f"Generating asset narrative for {ticker}")
        
        # Generate prompt for asset narrative
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "asset_narrative_generation",
            ticker=ticker,
            data=json.dumps(data)
        )
        
        # Get narrative from Gemma 3
        narrative_result = self.gemma_core.generate_response(prompt)
        
        # Parse the narrative result
        try:
            narrative = json.loads(narrative_result)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemma 3 response as JSON: {narrative_result}")
            
            # Extract structured information using regex
            summary_match = re.search(r"summary[\"']?\s*:\s*[\"']([^\"']+)[\"']", narrative_result)
            summary = summary_match.group(1) if summary_match else "No summary available"
            
            # Extract key factors using a simple heuristic
            factors = []
            factors_section = re.search(r"key_factors[\"']?\s*:\s*\[(.*?)\]", narrative_result, re.DOTALL)
            if factors_section:
                factors_text = factors_section.group(1)
                factor_matches = re.findall(r"[\"']([^\"']+)[\"']", factors_text)
                factors = factor_matches
            
            # Extract scenarios using a simple heuristic
            scenarios = []
            scenarios_section = re.search(r"scenarios[\"']?\s*:\s*\[(.*?)\]", narrative_result, re.DOTALL)
            if scenarios_section:
                scenarios_text = scenarios_section.group(1)
                scenario_blocks = re.findall(r"\{(.*?)\}", scenarios_text, re.DOTALL)
                
                for block in scenario_blocks:
                    name_match = re.search(r"name[\"']?\s*:\s*[\"']([^\"']+)[\"']", block)
                    prob_match = re.search(r"probability[\"']?\s*:\s*([0-9.]+)", block)
                    desc_match = re.search(r"description[\"']?\s*:\s*[\"']([^\"']+)[\"']", block)
                    
                    if name_match:
                        scenario = {
                            "name": name_match.group(1),
                            "probability": float(prob_match.group(1)) if prob_match else 0.0,
                            "description": desc_match.group(1) if desc_match else "No description"
                        }
                        scenarios.append(scenario)
            
            narrative = {
                "ticker": ticker,
                "summary": summary,
                "key_factors": factors,
                "scenarios": scenarios,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return narrative
    
    def generate_market_narrative(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a narrative for the overall market.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Data to use for narrative generation, including market indices,
            sector performance, economic indicators, and news.
            
        Returns:
        --------
        Dict[str, Any]
            Generated narrative including summary, key themes, and outlook.
        """
        self.logger.info("Generating market narrative")
        
        # Generate prompt for market narrative
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "market_narrative_generation",
            data=json.dumps(data)
        )
        
        # Get narrative from Gemma 3
        narrative_result = self.gemma_core.generate_response(prompt)
        
        # Parse the narrative result
        try:
            narrative = json.loads(narrative_result)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemma 3 response as JSON: {narrative_result}")
            
            # Extract structured information using regex
            summary_match = re.search(r"summary[\"']?\s*:\s*[\"']([^\"']+)[\"']", narrative_result)
            summary = summary_match.group(1) if summary_match else "No summary available"
            
            # Extract key themes using a simple heuristic
            themes = []
            themes_section = re.search(r"key_themes[\"']?\s*:\s*\[(.*?)\]", narrative_result, re.DOTALL)
            if themes_section:
                themes_text = themes_section.group(1)
                theme_matches = re.findall(r"[\"']([^\"']+)[\"']", themes_text)
                themes = theme_matches
            
            outlook_match = re.search(r"outlook[\"']?\s*:\s*[\"']([^\"']+)[\"']", narrative_result)
            outlook = outlook_match.group(1) if outlook_match else "Neutral outlook"
            
            narrative = {
                "summary": summary,
                "key_themes": themes,
                "outlook": outlook,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return narrative


class NaturalLanguageMarketAnalyzer:
    """
    Main class for natural language market analysis.
    
    This class integrates various natural language analysis components to provide
    comprehensive qualitative insights for trading decisions.
    """
    
    def __init__(self, gemma_core: GemmaCore):
        """
        Initialize the NaturalLanguageMarketAnalyzer.
        
        Parameters:
        -----------
        gemma_core : GemmaCore
            Instance of GemmaCore for accessing Gemma 3 capabilities.
        """
        self.logger = logging.getLogger("GemmaTrading.NLPAnalyzer")
        self.gemma_core = gemma_core
        
        # Initialize component analyzers
        self.news_analyzer = NewsAnalyzer(gemma_core)
        self.earnings_analyzer = EarningsAnalyzer(gemma_core)
        self.social_analyzer = SocialMediaAnalyzer(gemma_core)
        self.narrative_generator = MarketNarrativeGenerator(gemma_core)
        
        self.logger.info("Initialized NaturalLanguageMarketAnalyzer")
    
    def analyze_news(self, news_text: str) -> Dict[str, Any]:
        """
        Analyze news text for trading insights.
        
        Parameters:
        -----------
        news_text : str
            News text to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including sentiment and potential impact.
        """
        # Extract ticker from news if possible
        ticker_match = re.search(r'\$([A-Z]+)', news_text)
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"
        
        # Create a simple article structure
        article = {
            'title': news_text[:100] + ('...' if len(news_text) > 100 else ''),
            'content': news_text,
            'source': 'User Input',
            'date': datetime.datetime.now().strftime('%Y-%m-%d')
        }
        
        # Analyze as breaking news
        return self.news_analyzer.analyze_breaking_news(ticker, news_text)
    
    def analyze_earnings_report(self, report_text: str) -> Dict[str, Any]:
        """
        Analyze an earnings report for trading insights.
        
        Parameters:
        -----------
        report_text : str
            Earnings report text to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including key metrics and management sentiment.
        """
        # Extract ticker from report if possible
        ticker_match = re.search(r'\$([A-Z]+)|([A-Z]{1,5})(?:\s+Inc\.?|\s+Corp\.?|\s+Corporation)', report_text)
        ticker = ticker_match.group(1) if ticker_match and ticker_match.group(1) else \
                ticker_match.group(2) if ticker_match and ticker_match.group(2) else "UNKNOWN"
        
        # Analyze earnings report
        return self.earnings_analyzer.analyze_earnings_report(ticker, report_text)
    
    def analyze_social_sentiment(self, query: str) -> Dict[str, Any]:
        """
        Analyze social media sentiment for a query.
        
        Parameters:
        -----------
        query : str
            Query text to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including sentiment and key topics.
        """
        # Extract ticker from query if possible
        ticker_match = re.search(r'\$([A-Z]+)', query)
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"
        
        # Analyze social sentiment
        return self.social_analyzer.analyze_social_sentiment(ticker, query=query)
    
    def generate_market_narrative(self, market_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive market narrative.
        
        Parameters:
        -----------
        market_data : Dict[str, Any]
            Market data to use for narrative generation.
            
        Returns:
        --------
        str
            Generated market narrative.
        """
        # Generate market narrative
        narrative = self.narrative_generator.generate_market_narrative(market_data)
        
        # Format narrative as text
        narrative_text = f"Market Narrative ({narrative.get('timestamp', 'now')})\n\n"
        narrative_text += f"Summary: {narrative.get('summary', 'No summary available')}\n\n"
        
        narrative_text += "Key Themes:\n"
        for theme in narrative.get('key_themes', []):
            narrative_text += f"- {theme}\n"
        
        narrative_text += f"\nOutlook: {narrative.get('outlook', 'Neutral outlook')}"
        
        return narrative_text
    
    def generate_asset_narrative(self, ticker: str, asset_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive narrative for a specific asset.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        asset_data : Dict[str, Any]
            Asset data to use for narrative generation.
            
        Returns:
        --------
        str
            Generated asset narrative.
        """
        # Generate asset narrative
        narrative = self.narrative_generator.generate_asset_narrative(ticker, asset_data)
        
        # Format narrative as text
        narrative_text = f"{ticker} Narrative ({narrative.get('timestamp', 'now')})\n\n"
        narrative_text += f"Summary: {narrative.get('summary', 'No summary available')}\n\n"
        
        narrative_text += "Key Factors:\n"
        for factor in narrative.get('key_factors', []):
            narrative_text += f"- {factor}\n"
        
        narrative_text += "\nScenarios:\n"
        for scenario in narrative.get('scenarios', []):
            narrative_text += f"- {scenario.get('name', 'Unnamed')} ({scenario.get('probability', 0)*100:.1f}%): {scenario.get('description', 'No description')}\n"
        
        return narrative_text
    
    def integrate_qualitative_analysis(self, ticker: str, quantitative_data: Dict[str, Any], qualitative_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate qualitative and quantitative analysis for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        quantitative_data : Dict[str, Any]
            Quantitative analysis data including technical indicators and price data.
        qualitative_data : Dict[str, Any]
            Qualitative analysis data including news, earnings, and social sentiment.
            
        Returns:
        --------
        Dict[str, Any]
            Integrated analysis with trading recommendations.
        """
        self.logger.info(f"Integrating qualitative and quantitative analysis for {ticker}")
        
        # Generate prompt for integrated analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "integrated_analysis",
            ticker=ticker,
            quantitative_data=json.dumps(quantitative_data),
            qualitative_data=json.dumps(qualitative_data)
        )
        
        # Get analysis from Gemma 3
        analysis_result = self.gemma_core.generate_response(prompt)
        
        # Parse the analysis result
        try:
            analysis = json.loads(analysis_result)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse Gemma 3 response as JSON: {analysis_result}")
            
            # Extract structured information using regex
            recommendation_match = re.search(r"recommendation[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            recommendation = recommendation_match.group(1) if recommendation_match else "HOLD"
            
            confidence_match = re.search(r"confidence[\"']?\s*:\s*([0-9.]+)", analysis_result)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            reasoning_match = re.search(r"reasoning[\"']?\s*:\s*[\"']([^\"']+)[\"']", analysis_result)
            reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
            
            analysis = {
                "ticker": ticker,
                "recommendation": recommendation,
                "confidence": confidence,
                "reasoning": reasoning,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return analysis
