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
from gemma3_integration.architecture_enhanced import GemmaCore, PromptEngine, DataIntegration

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
            news_articles=articles_text
        )
        
        # Get the appropriate model for news analysis
        model = self.gemma_core.model_manager.get_model("news_analysis")
        
        # Generate news analysis using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract analysis from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured analysis
        # For this implementation, we'll simulate the extraction process
        
        # Simulate sentiment analysis
        sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}
        for article in articles:
            content = article.get('content', '').lower()
            title = article.get('title', '').lower()
            
            # Simple keyword-based sentiment analysis (for simulation)
            positive_keywords = ["positive", "growth", "increase", "profit", "beat", "exceed", "bullish", "upgrade"]
            negative_keywords = ["negative", "decline", "decrease", "loss", "miss", "below", "bearish", "downgrade"]
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in content or keyword in title)
            negative_count = sum(1 for keyword in negative_keywords if keyword in content or keyword in title)
            
            if positive_count > negative_count:
                sentiment_scores["positive"] += 1
            elif negative_count > positive_count:
                sentiment_scores["negative"] += 1
            else:
                sentiment_scores["neutral"] += 1
        
        # Determine overall sentiment
        max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
        overall_sentiment = max_sentiment[0]
        confidence = max_sentiment[1] / len(articles) if articles else 0
        
        # Extract key events (simulated)
        key_events = []
        for article in articles:
            title = article.get('title', '')
            if "earnings" in title.lower():
                key_events.append("Earnings report")
            elif "product" in title.lower():
                key_events.append("Product announcement")
            elif "merger" in title.lower() or "acquisition" in title.lower():
                key_events.append("M&A activity")
            elif "analyst" in title.lower():
                key_events.append("Analyst coverage")
        
        # Remove duplicates
        key_events = list(set(key_events))
        
        # Determine potential impact (simulated)
        if overall_sentiment == "positive":
            potential_impact = "The positive news is likely to drive short-term price appreciation."
        elif overall_sentiment == "negative":
            potential_impact = "The negative news may lead to short-term price decline."
        else:
            potential_impact = "The neutral news is unlikely to significantly impact price in the short term."
        
        analysis_result = {
            "ticker": ticker,
            "sentiment": overall_sentiment,
            "confidence": confidence,
            "key_events": key_events,
            "potential_impact": potential_impact,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed news analysis for {ticker}: {overall_sentiment} sentiment")
        return analysis_result
    
    def track_news_trends(self, ticker: str, articles: List[Dict[str, Any]], 
                         days: int = 30) -> Dict[str, Any]:
        """
        Track news trends for a ticker over a specified time period.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        articles : List[Dict[str, Any]]
            List of news articles to analyze.
        days : int, optional
            Number of days to track trends for. Default is 30.
            
        Returns:
        --------
        Dict[str, Any]
            Trend analysis including sentiment over time and topic evolution.
        """
        self.logger.info(f"Tracking news trends for {ticker} over {days} days")
        
        if not articles:
            return {
                "ticker": ticker,
                "sentiment_trend": [],
                "topic_evolution": [],
                "volume_trend": [],
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Filter articles by date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        recent_articles = []
        
        for article in articles:
            article_date_str = article.get('date')
            if not article_date_str:
                continue
                
            try:
                # Try to parse the date string
                article_date = datetime.datetime.fromisoformat(article_date_str)
                if article_date >= cutoff_date:
                    recent_articles.append(article)
            except (ValueError, TypeError):
                # If date parsing fails, include the article anyway
                recent_articles.append(article)
        
        # Group articles by day
        articles_by_day = {}
        for article in recent_articles:
            article_date_str = article.get('date')
            if not article_date_str:
                continue
                
            try:
                article_date = datetime.datetime.fromisoformat(article_date_str)
                day_key = article_date.strftime('%Y-%m-%d')
                
                if day_key not in articles_by_day:
                    articles_by_day[day_key] = []
                    
                articles_by_day[day_key].append(article)
            except (ValueError, TypeError):
                continue
        
        # Analyze sentiment for each day
        sentiment_trend = []
        volume_trend = []
        topic_evolution = []
        
        for day, day_articles in sorted(articles_by_day.items()):
            # Analyze sentiment
            day_analysis = self.analyze_articles(ticker, day_articles)
            
            sentiment_trend.append({
                "date": day,
                "sentiment": day_analysis["sentiment"],
                "confidence": day_analysis["confidence"]
            })
            
            volume_trend.append({
                "date": day,
                "volume": len(day_articles)
            })
            
            # Extract topics (simulated)
            topics = {}
            for article in day_articles:
                content = article.get('content', '').lower()
                title = article.get('title', '').lower()
                
                if "earnings" in content or "earnings" in title:
                    topics["earnings"] = topics.get("earnings", 0) + 1
                if "product" in content or "product" in title:
                    topics["product"] = topics.get("product", 0) + 1
                if "merger" in content or "acquisition" in content or "merger" in title or "acquisition" in title:
                    topics["m&a"] = topics.get("m&a", 0) + 1
                if "analyst" in content or "analyst" in title:
                    topics["analyst"] = topics.get("analyst", 0) + 1
            
            # Get top topics
            top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]
            
            topic_evolution.append({
                "date": day,
                "topics": [{"name": topic, "count": count} for topic, count in top_topics]
            })
        
        trend_result = {
            "ticker": ticker,
            "sentiment_trend": sentiment_trend,
            "topic_evolution": topic_evolution,
            "volume_trend": volume_trend,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed news trend analysis for {ticker}")
        return trend_result

class EarningsAnalyzer:
    """
    Analyzes earnings reports and conference calls to extract insights.
    
    This class processes earnings reports and conference call transcripts to
    identify key metrics, management sentiment, and forward guidance.
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
    
    def analyze_earnings_report(self, ticker: str, report_text: str, 
                              previous_reports: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze an earnings report to extract key metrics and insights.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the company.
        report_text : str
            Text of the earnings report.
        previous_reports : List[Dict[str, Any]], optional
            List of previous earnings reports for comparison.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including key metrics, surprises, and management guidance.
        """
        self.logger.info(f"Analyzing earnings report for {ticker}")
        
        # Prepare data for Gemma 3
        # In a real implementation, this would prepare the data in a format suitable for Gemma 3
        
        # Generate prompt for earnings analysis
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "earnings_analysis",
            ticker=ticker,
            report_text=report_text,
            previous_reports=str(previous_reports) if previous_reports else "No previous reports available"
        )
        
        # Get the appropriate model for earnings analysis
        model = self.gemma_core.model_manager.get_model("news_analysis")
        
        # Generate earnings analysis using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract analysis from chain-of-thought result
        # In a real implementation, this would parse the result to extract structured analysis
        # For this implementation, we'll simulate the extraction process
        
        # Simulate key metrics extraction
        key_metrics = {}
        
        # Look for revenue
        revenue_match = re.search(r'revenue of \$?(\d+\.?\d*)\s*(million|billion)', report_text, re.IGNORECASE)
        if revenue_match:
            amount = float(revenue_match.group(1))
            unit = revenue_match.group(2).lower()
            
            if unit == "billion":
                amount *= 1000
                
            key_metrics["revenue"] = amount
        
        # Look for EPS
        eps_match = re.search(r'EPS of \$?(\d+\.?\d*)', report_text, re.IGNORECASE)
        if eps_match:
            key_metrics["eps"] = float(eps_match.group(1))
        
        # Look for net income
        income_match = re.search(r'net income of \$?(\d+\.?\d*)\s*(million|billion)', report_text, re.IGNORECASE)
        if income_match:
            amount = float(income_match.group(1))
            unit = income_match.group(2).lower()
            
            if unit == "billion":
                amount *= 1000
                
            key_metrics["net_income"] = amount
        
        # Simulate surprises extraction
        surprises = []
        
        if "beat" in report_text.lower() or "exceeded" in report_text.lower():
            surprises.append("Earnings beat expectations")
        elif "miss" in report_text.lower() or "below" in report_text.lower():
            surprises.append("Earnings missed expectations")
            
        if "revenue growth" in report_text.lower():
            surprises.append("Strong revenue growth")
        elif "revenue decline" in report_text.lower():
            surprises.append("Revenue decline")
        
        # Simulate guidance extraction
        guidance = {}
        
        guidance_match = re.search(r'guidance of \$?(\d+\.?\d*)\s*(million|billion)', report_text, re.IGNORECASE)
        if guidance_match:
            amount = float(guidance_match.group(1))
            unit = guidance_match.group(2).lower()
            
            if unit == "billion":
                amount *= 1000
                
            guidance["revenue"] = amount
        
        eps_guidance_match = re.search(r'EPS guidance of \$?(\d+\.?\d*)', report_text, re.IGNORECASE)
        if eps_guidance_match:
            guidance["eps"] = float(eps_guidance_match.group(1))
        
        # Simulate management sentiment
        management_sentiment = "neutral"
        
        positive_phrases = ["optimistic", "confident", "strong", "growth", "positive"]
        negative_phrases = ["challenging", "difficult", "headwinds", "decline", "negative"]
        
        positive_count = sum(1 for phrase in positive_phrases if phrase in report_text.lower())
        negative_count = sum(1 for phrase in negative_phrases if phrase in report_text.lower())
        
        if positive_count > negative_count:
            management_sentiment = "positive"
        elif negative_count > positive_count:
            management_sentiment = "negative"
        
        analysis_result = {
            "ticker": ticker,
            "key_metrics": key_metrics,
            "surprises": surprises,
            "guidance": guidance,
            "management_sentiment": management_sentiment,
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed earnings analysis for {ticker}")
        return analysis_result
    
    def analyze_conference_call(self, ticker: str, transcript: str) -> Dict[str, Any]:
        """
        Analyze a conference call transcript to extract insights.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the company.
        transcript : str
            Text of the conference call transcript.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including management tone, key topics, and analyst sentiment.
        """
        self.logger.info(f"Analyzing conference call transcript for {ticker}")
        
        # In a real implementation, this would use Gemma 3 to analyze the transcript
        # For this implementation, we'll simulate the analysis process
        
        # Simulate management tone analysis
        management_sections = []
        analyst_sections = []
        
        # Split transcript into management and analyst sections (simplified)
        lines = transcript.split('\n')
        current_section = None
        
        for line in lines:
            if "CEO:" in line or "CFO:" in line or "COO:" in line:
                current_section = "management"
                management_sections.append(line)
            elif "Analyst:" in line:
                current_section = "analyst"
                analyst_sections.append(line)
            elif current_section:
                if current_section == "management":
                    management_sections.append(line)
                else:
                    analyst_sections.append(line)
        
        management_text = "\n".join(management_sections)
        analyst_text = "\n".join(analyst_sections)
        
        # Analyze management tone (simulated)
        management_tone = "neutral"
        
        positive_phrases = ["optimistic", "confident", "strong", "growth", "positive"]
        negative_phrases = ["challenging", "difficult", "headwinds", "decline", "negative"]
        
        positive_count = sum(1 for phrase in positive_phrases if phrase in management_text.lower())
        negative_count = sum(1 for phrase in negative_phrases if phrase in management_text.lower())
        
        if positive_count > negative_count:
            management_tone = "positive"
        elif negative_count > positive_count:
            management_tone = "negative"
        
        # Extract key topics (simulated)
        key_topics = []
        
        if "revenue" in transcript.lower():
            key_topics.append("Revenue")
        if "margin" in transcript.lower():
            key_topics.append("Margins")
        if "growth" in transcript.lower():
            key_topics.append("Growth")
        if "product" in transcript.lower():
            key_topics.append("Product")
        if "competition" in transcript.lower():
            key_topics.append("Competition")
        if "market share" in transcript.lower():
            key_topics.append("Market Share")
        
        # Analyze analyst sentiment (simulated)
        analyst_sentiment = "neutral"
        
        positive_phrases = ["great", "impressive", "congratulations", "strong", "positive"]
        negative_phrases = ["concerned", "worried", "disappointing", "weak", "negative"]
        
        positive_count = sum(1 for phrase in positive_phrases if phrase in analyst_text.lower())
        negative_count = sum(1 for phrase in negative_phrases if phrase in analyst_text.lower())
        
        if positive_count > negative_count:
            analyst_sentiment = "positive"
        elif negative_count > positive_count:
            analyst_sentiment = "negative"
        
        # Extract forward-looking statements (simulated)
        forward_looking = []
        
        if "expect" in transcript.lower():
            forward_looking.append("Expectations mentioned")
        if "guidance" in transcript.lower():
            forward_looking.append("Guidance provided")
        if "future" in transcript.lower():
            forward_looking.append("Future plans discussed")
        if "next quarter" in transcript.lower() or "next year" in transcript.lower():
            forward_looking.append("Next period outlook provided")
        
        analysis_result = {
            "ticker": ticker,
            "management_tone": management_tone,
            "key_topics": key_topics,
            "analyst_sentiment": analyst_sentiment,
            "forward_looking_statements": forward_looking,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed conference call analysis for {ticker}")
        return analysis_result

class SocialMediaAnalyzer:
    """
    Analyzes social media content to extract sentiment and trends.
    
    This class processes social media posts related to specific tickers or the
    broader market to identify sentiment, trends, and potential trading catalysts.
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
    
    def analyze_posts(self, ticker: str, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze social media posts related to a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        posts : List[Dict[str, Any]]
            List of social media posts to analyze. Each post should have 'content',
            'source', 'user', and 'date' fields.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including sentiment, key topics, and influencer activity.
        """
        self.logger.info(f"Analyzing {len(posts)} social media posts for {ticker}")
        
        if not posts:
            return {
                "ticker": ticker,
                "sentiment": "neutral",
                "sentiment_breakdown": {"positive": 0, "neutral": 0, "negative": 0},
                "key_topics": [],
                "influencer_activity": [],
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # In a real implementation, this would use Gemma 3 to analyze the posts
        # For this implementation, we'll simulate the analysis process
        
        # Simulate sentiment analysis
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for post in posts:
            content = post.get('content', '').lower()
            
            # Simple keyword-based sentiment analysis (for simulation)
            positive_keywords = ["bullish", "buy", "long", "up", "moon", "rocket", "good", "great"]
            negative_keywords = ["bearish", "sell", "short", "down", "crash", "bad", "terrible"]
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in content)
            negative_count = sum(1 for keyword in negative_keywords if keyword in content)
            
            if positive_count > negative_count:
                sentiment_counts["positive"] += 1
            elif negative_count > positive_count:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1
        
        # Calculate sentiment percentages
        total_posts = len(posts)
        sentiment_breakdown = {
            "positive": sentiment_counts["positive"] / total_posts if total_posts > 0 else 0,
            "neutral": sentiment_counts["neutral"] / total_posts if total_posts > 0 else 0,
            "negative": sentiment_counts["negative"] / total_posts if total_posts > 0 else 0
        }
        
        # Determine overall sentiment
        overall_sentiment = max(sentiment_breakdown.items(), key=lambda x: x[1])[0]
        
        # Extract key topics (simulated)
        topic_counter = Counter()
        
        for post in posts:
            content = post.get('content', '').lower()
            
            # Simple topic extraction (for simulation)
            if "price" in content:
                topic_counter["price"] += 1
            if "earnings" in content:
                topic_counter["earnings"] += 1
            if "buy" in content:
                topic_counter["buy signal"] += 1
            if "sell" in content:
                topic_counter["sell signal"] += 1
            if "news" in content:
                topic_counter["news"] += 1
            if "analyst" in content:
                topic_counter["analyst"] += 1
        
        # Get top topics
        key_topics = [{"topic": topic, "count": count} for topic, count in topic_counter.most_common(5)]
        
        # Analyze influencer activity (simulated)
        user_post_counts = Counter()
        user_engagement = {}
        
        for post in posts:
            user = post.get('user', 'unknown')
            user_post_counts[user] += 1
            
            # Simulate engagement metrics
            engagement = post.get('engagement', {})
            if user not in user_engagement:
                user_engagement[user] = {
                    "likes": 0,
                    "comments": 0,
                    "shares": 0
                }
            
            user_engagement[user]["likes"] += engagement.get('likes', 0)
            user_engagement[user]["comments"] += engagement.get('comments', 0)
            user_engagement[user]["shares"] += engagement.get('shares', 0)
        
        # Identify top influencers
        top_users = user_post_counts.most_common(5)
        
        influencer_activity = [
            {
                "user": user,
                "post_count": count,
                "engagement": user_engagement.get(user, {"likes": 0, "comments": 0, "shares": 0})
            }
            for user, count in top_users
        ]
        
        analysis_result = {
            "ticker": ticker,
            "sentiment": overall_sentiment,
            "sentiment_breakdown": sentiment_breakdown,
            "key_topics": key_topics,
            "influencer_activity": influencer_activity,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed social media analysis for {ticker}")
        return analysis_result
    
    def detect_unusual_activity(self, ticker: str, posts: List[Dict[str, Any]], 
                              baseline_period_days: int = 30) -> Dict[str, Any]:
        """
        Detect unusual social media activity for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        posts : List[Dict[str, Any]]
            List of social media posts to analyze.
        baseline_period_days : int, optional
            Number of days to use for establishing the baseline. Default is 30.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including activity anomalies and sentiment shifts.
        """
        self.logger.info(f"Detecting unusual social media activity for {ticker}")
        
        if not posts:
            return {
                "ticker": ticker,
                "unusual_activity_detected": False,
                "activity_anomalies": [],
                "sentiment_shifts": [],
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # In a real implementation, this would use Gemma 3 to detect unusual activity
        # For this implementation, we'll simulate the detection process
        
        # Group posts by day
        posts_by_day = {}
        
        for post in posts:
            post_date_str = post.get('date')
            if not post_date_str:
                continue
                
            try:
                post_date = datetime.datetime.fromisoformat(post_date_str)
                day_key = post_date.strftime('%Y-%m-%d')
                
                if day_key not in posts_by_day:
                    posts_by_day[day_key] = []
                    
                posts_by_day[day_key].append(post)
            except (ValueError, TypeError):
                continue
        
        # Calculate daily post counts and sentiment
        daily_metrics = {}
        
        for day, day_posts in posts_by_day.items():
            # Count posts
            post_count = len(day_posts)
            
            # Calculate sentiment
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            
            for post in day_posts:
                content = post.get('content', '').lower()
                
                # Simple keyword-based sentiment analysis (for simulation)
                positive_keywords = ["bullish", "buy", "long", "up", "moon", "rocket", "good", "great"]
                negative_keywords = ["bearish", "sell", "short", "down", "crash", "bad", "terrible"]
                
                positive_count = sum(1 for keyword in positive_keywords if keyword in content)
                negative_count = sum(1 for keyword in negative_keywords if keyword in content)
                
                if positive_count > negative_count:
                    sentiment_counts["positive"] += 1
                elif negative_count > positive_count:
                    sentiment_counts["negative"] += 1
                else:
                    sentiment_counts["neutral"] += 1
            
            # Calculate sentiment percentages
            sentiment_breakdown = {
                "positive": sentiment_counts["positive"] / post_count if post_count > 0 else 0,
                "neutral": sentiment_counts["neutral"] / post_count if post_count > 0 else 0,
                "negative": sentiment_counts["negative"] / post_count if post_count > 0 else 0
            }
            
            # Determine overall sentiment
            overall_sentiment = max(sentiment_breakdown.items(), key=lambda x: x[1])[0]
            
            daily_metrics[day] = {
                "post_count": post_count,
                "sentiment": overall_sentiment,
                "sentiment_breakdown": sentiment_breakdown
            }
        
        # Sort days chronologically
        sorted_days = sorted(daily_metrics.keys())
        
        if not sorted_days:
            return {
                "ticker": ticker,
                "unusual_activity_detected": False,
                "activity_anomalies": [],
                "sentiment_shifts": [],
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Calculate baseline metrics
        baseline_end_date = datetime.datetime.fromisoformat(sorted_days[-1])
        baseline_start_date = baseline_end_date - datetime.timedelta(days=baseline_period_days)
        baseline_start_key = baseline_start_date.strftime('%Y-%m-%d')
        
        baseline_days = [day for day in sorted_days if day >= baseline_start_key and day < sorted_days[-1]]
        
        if not baseline_days:
            return {
                "ticker": ticker,
                "unusual_activity_detected": False,
                "activity_anomalies": [],
                "sentiment_shifts": [],
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Calculate baseline post count
        baseline_post_counts = [daily_metrics[day]["post_count"] for day in baseline_days]
        baseline_avg_posts = sum(baseline_post_counts) / len(baseline_post_counts)
        baseline_std_posts = np.std(baseline_post_counts) if len(baseline_post_counts) > 1 else 0
        
        # Calculate baseline sentiment
        baseline_sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for day in baseline_days:
            sentiment = daily_metrics[day]["sentiment"]
            baseline_sentiment_counts[sentiment] += 1
        
        baseline_sentiment = max(baseline_sentiment_counts.items(), key=lambda x: x[1])[0]
        
        # Check for anomalies in the most recent day
        latest_day = sorted_days[-1]
        latest_metrics = daily_metrics[latest_day]
        
        # Check for post count anomaly
        post_count_anomaly = False
        post_count_z_score = 0
        
        if baseline_std_posts > 0:
            post_count_z_score = (latest_metrics["post_count"] - baseline_avg_posts) / baseline_std_posts
            post_count_anomaly = abs(post_count_z_score) > 2  # Z-score threshold of 2
        
        # Check for sentiment shift
        sentiment_shift = latest_metrics["sentiment"] != baseline_sentiment
        
        # Prepare result
        activity_anomalies = []
        sentiment_shifts = []
        
        if post_count_anomaly:
            if post_count_z_score > 0:
                activity_anomalies.append({
                    "type": "increased_activity",
                    "description": f"Post volume increased by {post_count_z_score:.1f} standard deviations above baseline",
                    "baseline": baseline_avg_posts,
                    "current": latest_metrics["post_count"]
                })
            else:
                activity_anomalies.append({
                    "type": "decreased_activity",
                    "description": f"Post volume decreased by {abs(post_count_z_score):.1f} standard deviations below baseline",
                    "baseline": baseline_avg_posts,
                    "current": latest_metrics["post_count"]
                })
        
        if sentiment_shift:
            sentiment_shifts.append({
                "type": "sentiment_shift",
                "description": f"Sentiment shifted from {baseline_sentiment} to {latest_metrics['sentiment']}",
                "baseline": baseline_sentiment,
                "current": latest_metrics["sentiment"]
            })
        
        unusual_activity_detected = post_count_anomaly or sentiment_shift
        
        analysis_result = {
            "ticker": ticker,
            "unusual_activity_detected": unusual_activity_detected,
            "activity_anomalies": activity_anomalies,
            "sentiment_shifts": sentiment_shifts,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed unusual activity detection for {ticker}")
        return analysis_result

class MarketNarrativeGenerator:
    """
    Generates market narratives by combining news, social media, and technical analysis.
    
    This class synthesizes information from various sources to create coherent
    market narratives that explain recent price action and suggest future directions.
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
    
    def generate_narrative(self, ticker: str, 
                          news_analysis: Dict[str, Any],
                          social_media_analysis: Dict[str, Any],
                          technical_analysis: Dict[str, Any],
                          price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a market narrative for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        news_analysis : Dict[str, Any]
            Results of news analysis.
        social_media_analysis : Dict[str, Any]
            Results of social media analysis.
        technical_analysis : Dict[str, Any]
            Results of technical analysis.
        price_data : pd.DataFrame
            Recent price data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            The generated market narrative.
        """
        self.logger.info(f"Generating market narrative for {ticker}")
        
        # Prepare data for Gemma 3
        # In a real implementation, this would prepare the data in a format suitable for Gemma 3
        
        # Generate prompt for narrative generation
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "market_narrative",
            ticker=ticker,
            news_analysis=str(news_analysis),
            social_media_analysis=str(social_media_analysis),
            technical_analysis=str(technical_analysis),
            price_data=str(price_data.tail(10))  # Simplified for this implementation
        )
        
        # Get the appropriate model for narrative generation
        model = self.gemma_core.model_manager.get_model("market_analysis")
        
        # Generate narrative using chain-of-thought reasoning
        cot_result = self.gemma_core.cot_processor.generate_cot(model, prompt)
        
        # Extract narrative from chain-of-thought result
        # In a real implementation, this would parse the result to extract a structured narrative
        # For this implementation, we'll simulate the extraction process
        
        # Simulate narrative generation
        
        # Determine price action
        if len(price_data) >= 2:
            first_price = price_data.iloc[0]['close'] if 'close' in price_data.columns else price_data.iloc[0][0]
            last_price = price_data.iloc[-1]['close'] if 'close' in price_data.columns else price_data.iloc[-1][0]
            price_change_pct = (last_price - first_price) / first_price * 100
            
            if price_change_pct > 5:
                price_action = "strong upward"
            elif price_change_pct > 1:
                price_action = "moderate upward"
            elif price_change_pct > -1:
                price_action = "sideways"
            elif price_change_pct > -5:
                price_action = "moderate downward"
            else:
                price_action = "strong downward"
        else:
            price_action = "unknown"
        
        # Combine news and social media sentiment
        news_sentiment = news_analysis.get("sentiment", "neutral")
        social_sentiment = social_media_analysis.get("sentiment", "neutral")
        
        if news_sentiment == social_sentiment:
            combined_sentiment = news_sentiment
        elif (news_sentiment == "positive" and social_sentiment == "neutral") or (news_sentiment == "neutral" and social_sentiment == "positive"):
            combined_sentiment = "moderately positive"
        elif (news_sentiment == "negative" and social_sentiment == "neutral") or (news_sentiment == "neutral" and social_sentiment == "negative"):
            combined_sentiment = "moderately negative"
        else:
            combined_sentiment = "mixed"
        
        # Get technical trend
        technical_trend = technical_analysis.get("trend", "neutral")
        
        # Generate narrative summary
        if combined_sentiment == "positive" and technical_trend == "bullish":
            narrative_summary = f"{ticker} has shown {price_action} price action, supported by positive news sentiment and bullish technical indicators. The alignment of fundamental and technical factors suggests continued upward momentum."
        elif combined_sentiment == "negative" and technical_trend == "bearish":
            narrative_summary = f"{ticker} has shown {price_action} price action, pressured by negative news sentiment and bearish technical indicators. The alignment of fundamental and technical factors suggests continued downward pressure."
        elif combined_sentiment == "positive" and technical_trend == "bearish":
            narrative_summary = f"{ticker} has shown {price_action} price action, with a divergence between positive news sentiment and bearish technical indicators. This divergence may indicate a potential trend reversal if fundamental factors begin to influence price action."
        elif combined_sentiment == "negative" and technical_trend == "bullish":
            narrative_summary = f"{ticker} has shown {price_action} price action, with a divergence between negative news sentiment and bullish technical indicators. This divergence may indicate a potential trend reversal if fundamental factors begin to influence price action."
        else:
            narrative_summary = f"{ticker} has shown {price_action} price action, with {combined_sentiment} sentiment from news and social media, and {technical_trend} technical indicators. The market appears to be in a transitional phase with no clear directional bias."
        
        # Extract key drivers
        key_drivers = []
        
        # Add news events
        key_events = news_analysis.get("key_events", [])
        if key_events:
            key_drivers.extend(key_events)
        
        # Add social media topics
        social_topics = [topic["topic"] for topic in social_media_analysis.get("key_topics", [])][:2]
        if social_topics:
            key_drivers.extend([f"Social media discussion about {topic}" for topic in social_topics])
        
        # Add technical factors
        if technical_trend == "bullish":
            key_drivers.append("Bullish technical indicators")
        elif technical_trend == "bearish":
            key_drivers.append("Bearish technical indicators")
        
        # Generate trading implications
        if combined_sentiment == "positive" and technical_trend == "bullish":
            trading_implications = "The aligned positive fundamentals and bullish technicals suggest a favorable environment for long positions, with potential for continued upward momentum."
        elif combined_sentiment == "negative" and technical_trend == "bearish":
            trading_implications = "The aligned negative fundamentals and bearish technicals suggest a favorable environment for short positions, with potential for continued downward pressure."
        elif combined_sentiment == "positive" and technical_trend == "bearish":
            trading_implications = "The divergence between positive fundamentals and bearish technicals suggests a cautious approach, potentially waiting for technical confirmation of a trend reversal before establishing long positions."
        elif combined_sentiment == "negative" and technical_trend == "bullish":
            trading_implications = "The divergence between negative fundamentals and bullish technicals suggests a cautious approach, potentially waiting for fundamental improvement or technical breakdown before making trading decisions."
        else:
            trading_implications = "The mixed signals suggest a neutral stance, potentially focusing on range-bound trading strategies or waiting for clearer directional signals before establishing positions."
        
        # Generate future scenarios
        bullish_scenario = f"If {ticker} continues to see positive news catalysts and social media sentiment remains favorable, price could break through technical resistance and establish a new uptrend."
        bearish_scenario = f"If {ticker} faces negative news catalysts or deteriorating social media sentiment, price could break down through technical support and accelerate to the downside."
        
        narrative = {
            "ticker": ticker,
            "summary": narrative_summary,
            "key_drivers": key_drivers,
            "trading_implications": trading_implications,
            "future_scenarios": {
                "bullish": bullish_scenario,
                "bearish": bearish_scenario
            },
            "reasoning": cot_result.get("reasoning_steps", []),
            "conclusion": cot_result.get("conclusion", ""),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed market narrative generation for {ticker}")
        return narrative
    
    def generate_sector_narrative(self, sector: str,
                                 ticker_narratives: List[Dict[str, Any]],
                                 sector_news: List[Dict[str, Any]],
                                 sector_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a sector-wide market narrative.
        
        Parameters:
        -----------
        sector : str
            Name of the sector.
        ticker_narratives : List[Dict[str, Any]]
            List of individual ticker narratives within the sector.
        sector_news : List[Dict[str, Any]]
            News articles related to the sector.
        sector_performance : Dict[str, Any]
            Performance metrics for the sector.
            
        Returns:
        --------
        Dict[str, Any]
            The generated sector narrative.
        """
        self.logger.info(f"Generating sector narrative for {sector}")
        
        # In a real implementation, this would use Gemma 3 to generate a sector narrative
        # For this implementation, we'll simulate the narrative generation process
        
        # Analyze sector performance
        sector_return = sector_performance.get("return", 0)
        
        if sector_return > 5:
            performance_description = "strong outperformance"
        elif sector_return > 1:
            performance_description = "moderate outperformance"
        elif sector_return > -1:
            performance_description = "in-line performance"
        elif sector_return > -5:
            performance_description = "moderate underperformance"
        else:
            performance_description = "significant underperformance"
        
        # Analyze ticker narratives
        ticker_sentiments = [narrative.get("sentiment", "neutral") for narrative in ticker_narratives]
        
        positive_count = sum(1 for sentiment in ticker_sentiments if "positive" in sentiment)
        negative_count = sum(1 for sentiment in ticker_sentiments if "negative" in sentiment)
        neutral_count = len(ticker_sentiments) - positive_count - negative_count
        
        if positive_count > negative_count + neutral_count:
            sector_sentiment = "broadly positive"
        elif negative_count > positive_count + neutral_count:
            sector_sentiment = "broadly negative"
        elif positive_count > negative_count:
            sector_sentiment = "mixed with positive bias"
        elif negative_count > positive_count:
            sector_sentiment = "mixed with negative bias"
        else:
            sector_sentiment = "mixed"
        
        # Analyze sector news
        sector_news_sentiment = "neutral"
        
        if sector_news:
            positive_news = sum(1 for article in sector_news if "positive" in article.get("sentiment", "neutral"))
            negative_news = sum(1 for article in sector_news if "negative" in article.get("sentiment", "neutral"))
            
            if positive_news > negative_news:
                sector_news_sentiment = "positive"
            elif negative_news > positive_news:
                sector_news_sentiment = "negative"
        
        # Generate sector summary
        sector_summary = f"The {sector} sector has shown {performance_description} relative to the broader market, with {sector_sentiment} sentiment across individual stocks and {sector_news_sentiment} news flow."
        
        # Identify sector leaders and laggards
        leaders = []
        laggards = []
        
        for narrative in ticker_narratives:
            ticker = narrative.get("ticker", "")
            if not ticker:
                continue
                
            ticker_return = narrative.get("performance", {}).get("return", 0)
            
            if ticker_return > sector_return + 3:
                leaders.append(ticker)
            elif ticker_return < sector_return - 3:
                laggards.append(ticker)
        
        # Generate sector outlook
        if sector_sentiment == "broadly positive" and sector_news_sentiment == "positive":
            sector_outlook = f"The outlook for the {sector} sector remains favorable, with positive sentiment across individual stocks and supportive news flow. The sector may continue to outperform in the near term."
        elif sector_sentiment == "broadly negative" and sector_news_sentiment == "negative":
            sector_outlook = f"The outlook for the {sector} sector remains challenging, with negative sentiment across individual stocks and unfavorable news flow. The sector may continue to underperform in the near term."
        else:
            sector_outlook = f"The outlook for the {sector} sector is mixed, with varied sentiment across individual stocks and {sector_news_sentiment} news flow. Performance may be stock-specific rather than sector-driven in the near term."
        
        narrative = {
            "sector": sector,
            "summary": sector_summary,
            "performance": performance_description,
            "sentiment": sector_sentiment,
            "news_sentiment": sector_news_sentiment,
            "leaders": leaders,
            "laggards": laggards,
            "outlook": sector_outlook,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed sector narrative generation for {sector}")
        return narrative

class NaturalLanguageMarketAnalysis:
    """
    Main class for natural language market analysis using Gemma 3.
    
    This class provides a unified interface for all natural language market analysis
    capabilities, including news analysis, earnings analysis, social media analysis,
    and market narrative generation.
    """
    
    def __init__(self, gemma_core: Optional[GemmaCore] = None):
        """
        Initialize the NaturalLanguageMarketAnalysis.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.NaturalLanguageMarketAnalysis")
        
        # Create or use provided GemmaCore
        self.gemma_core = gemma_core or GemmaCore()
        
        # Initialize components
        self.news_analyzer = NewsAnalyzer(self.gemma_core)
        self.earnings_analyzer = EarningsAnalyzer(self.gemma_core)
        self.social_media_analyzer = SocialMediaAnalyzer(self.gemma_core)
        self.narrative_generator = MarketNarrativeGenerator(self.gemma_core)
        
        self.logger.info("Initialized NaturalLanguageMarketAnalysis")
    
    def analyze_news(self, ticker: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze news articles for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        articles : List[Dict[str, Any]]
            List of news articles to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            News analysis results.
        """
        return self.news_analyzer.analyze_articles(ticker, articles)
    
    def track_news_trends(self, ticker: str, articles: List[Dict[str, Any]], 
                         days: int = 30) -> Dict[str, Any]:
        """
        Track news trends for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        articles : List[Dict[str, Any]]
            List of news articles to analyze.
        days : int, optional
            Number of days to track trends for. Default is 30.
            
        Returns:
        --------
        Dict[str, Any]
            News trend analysis results.
        """
        return self.news_analyzer.track_news_trends(ticker, articles, days)
    
    def analyze_earnings(self, ticker: str, report_text: str,
                       previous_reports: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze an earnings report for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the company.
        report_text : str
            Text of the earnings report.
        previous_reports : List[Dict[str, Any]], optional
            List of previous earnings reports for comparison.
            
        Returns:
        --------
        Dict[str, Any]
            Earnings analysis results.
        """
        return self.earnings_analyzer.analyze_earnings_report(ticker, report_text, previous_reports)
    
    def analyze_conference_call(self, ticker: str, transcript: str) -> Dict[str, Any]:
        """
        Analyze a conference call transcript for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the company.
        transcript : str
            Text of the conference call transcript.
            
        Returns:
        --------
        Dict[str, Any]
            Conference call analysis results.
        """
        return self.earnings_analyzer.analyze_conference_call(ticker, transcript)
    
    def analyze_social_media(self, ticker: str, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze social media posts for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        posts : List[Dict[str, Any]]
            List of social media posts to analyze.
            
        Returns:
        --------
        Dict[str, Any]
            Social media analysis results.
        """
        return self.social_media_analyzer.analyze_posts(ticker, posts)
    
    def detect_unusual_social_activity(self, ticker: str, posts: List[Dict[str, Any]],
                                     baseline_period_days: int = 30) -> Dict[str, Any]:
        """
        Detect unusual social media activity for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        posts : List[Dict[str, Any]]
            List of social media posts to analyze.
        baseline_period_days : int, optional
            Number of days to use for establishing the baseline. Default is 30.
            
        Returns:
        --------
        Dict[str, Any]
            Unusual activity detection results.
        """
        return self.social_media_analyzer.detect_unusual_activity(ticker, posts, baseline_period_days)
    
    def generate_market_narrative(self, ticker: str,
                                news_analysis: Dict[str, Any],
                                social_media_analysis: Dict[str, Any],
                                technical_analysis: Dict[str, Any],
                                price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a market narrative for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        news_analysis : Dict[str, Any]
            Results of news analysis.
        social_media_analysis : Dict[str, Any]
            Results of social media analysis.
        technical_analysis : Dict[str, Any]
            Results of technical analysis.
        price_data : pd.DataFrame
            Recent price data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            The generated market narrative.
        """
        return self.narrative_generator.generate_narrative(
            ticker, news_analysis, social_media_analysis, technical_analysis, price_data
        )
    
    def generate_sector_narrative(self, sector: str,
                                ticker_narratives: List[Dict[str, Any]],
                                sector_news: List[Dict[str, Any]],
                                sector_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a sector-wide market narrative.
        
        Parameters:
        -----------
        sector : str
            Name of the sector.
        ticker_narratives : List[Dict[str, Any]]
            List of individual ticker narratives within the sector.
        sector_news : List[Dict[str, Any]]
            News articles related to the sector.
        sector_performance : Dict[str, Any]
            Performance metrics for the sector.
            
        Returns:
        --------
        Dict[str, Any]
            The generated sector narrative.
        """
        return self.narrative_generator.generate_sector_narrative(
            sector, ticker_narratives, sector_news, sector_performance
        )
    
    def comprehensive_analysis(self, ticker: str,
                             news_articles: List[Dict[str, Any]],
                             social_media_posts: List[Dict[str, Any]],
                             earnings_report: Optional[str] = None,
                             conference_call: Optional[str] = None,
                             technical_analysis: Optional[Dict[str, Any]] = None,
                             price_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform a comprehensive natural language market analysis for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol for the asset.
        news_articles : List[Dict[str, Any]]
            List of news articles to analyze.
        social_media_posts : List[Dict[str, Any]]
            List of social media posts to analyze.
        earnings_report : str, optional
            Text of the latest earnings report.
        conference_call : str, optional
            Text of the latest conference call transcript.
        technical_analysis : Dict[str, Any], optional
            Results of technical analysis.
        price_data : pd.DataFrame, optional
            Recent price data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive analysis results.
        """
        self.logger.info(f"Performing comprehensive analysis for {ticker}")
        
        # Analyze news
        news_analysis = self.analyze_news(ticker, news_articles)
        
        # Analyze social media
        social_media_analysis = self.analyze_social_media(ticker, social_media_posts)
        
        # Analyze earnings if available
        earnings_analysis = None
        if earnings_report:
            earnings_analysis = self.analyze_earnings(ticker, earnings_report)
        
        # Analyze conference call if available
        conference_call_analysis = None
        if conference_call:
            conference_call_analysis = self.analyze_conference_call(ticker, conference_call)
        
        # Generate market narrative if technical analysis and price data are available
        market_narrative = None
        if technical_analysis and price_data is not None:
            market_narrative = self.generate_market_narrative(
                ticker, news_analysis, social_media_analysis, technical_analysis, price_data
            )
        
        # Detect unusual social media activity
        unusual_activity = self.detect_unusual_social_activity(ticker, social_media_posts)
        
        # Combine all analyses
        comprehensive_result = {
            "ticker": ticker,
            "news_analysis": news_analysis,
            "social_media_analysis": social_media_analysis,
            "earnings_analysis": earnings_analysis,
            "conference_call_analysis": conference_call_analysis,
            "market_narrative": market_narrative,
            "unusual_social_activity": unusual_activity,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed comprehensive analysis for {ticker}")
        return comprehensive_result
