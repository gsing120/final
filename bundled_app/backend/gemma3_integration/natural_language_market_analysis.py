"""
Natural Language Market Analysis Module for Gemma Advanced Trading System

This module implements natural language processing capabilities for analyzing
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
from gemma3_integration.architecture import GemmaCore, PromptEngine, DataIntegration

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
        self.prompt_engine = gemma_core.prompt_engine
        self.data_integration = gemma_core.data_integration
        self.logger.info("Initialized NewsAnalyzer")
    
    def analyze_news_articles(self, ticker: str, days: int = 7) -> Dict:
        """
        Analyze news articles for a specific ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        days : int
            Number of days to look back.
            
        Returns:
        --------
        Dict
            News analysis results.
        """
        self.logger.info(f"Analyzing news articles for {ticker} (days: {days})")
        
        try:
            # Fetch news articles
            news_articles = self.data_integration.fetch_news(ticker, days)
            
            if not news_articles:
                self.logger.warning(f"No news articles found for {ticker}")
                return {
                    "ticker": ticker,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "articles_analyzed": 0,
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "key_events": [],
                    "topics": [],
                    "potential_catalysts": []
                }
            
            # Generate prompt for news analysis
            prompt = self.prompt_engine.generate_prompt(
                "news_analysis",
                ticker=ticker,
                news_articles=json.dumps(news_articles, indent=2)
            ) if "news_analysis" in self.prompt_engine.templates else self.prompt_engine._generate_default_prompt(
                task="Analyze the following news articles",
                ticker=ticker,
                news_articles=json.dumps(news_articles, indent=2)
            )
            
            # Generate reasoning using Gemma 3
            reasoning = self.gemma_core.cot_processor.generate_reasoning(prompt)
            
            # Extract key information from reasoning
            sentiment_score = self._extract_sentiment_score(reasoning)
            sentiment_label = self._sentiment_score_to_label(sentiment_score)
            key_events = self._extract_key_events(reasoning)
            topics = self._extract_topics(reasoning, news_articles)
            potential_catalysts = self._extract_potential_catalysts(reasoning)
            
            # Prepare analysis result
            analysis = {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "articles_analyzed": len(news_articles),
                "overall_sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "key_events": key_events,
                "topics": topics,
                "potential_catalysts": potential_catalysts,
                "reasoning": reasoning
            }
            
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze news articles: {e}")
            return {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _extract_sentiment_score(self, reasoning: Dict) -> float:
        """
        Extract sentiment score from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        float
            Sentiment score (-1.0 to 1.0).
        """
        # Try to extract sentiment score from conclusion
        conclusion = reasoning.get("conclusion", "")
        
        # Look for explicit sentiment mentions
        sentiment_patterns = [
            r"sentiment(?:\s+score)?(?:\s+is)?(?:\s+of)?\s*[:-]?\s*([-+]?\d+\.?\d*)",
            r"([-+]?\d+\.?\d*)(?:\s+sentiment(?:\s+score)?)",
            r"sentiment\s+is\s+(\w+)",
            r"(\w+)\s+sentiment"
        ]
        
        for pattern in sentiment_patterns:
            matches = re.findall(pattern, conclusion.lower())
            if matches:
                match = matches[0]
                if isinstance(match, str) and match.replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).isdigit():
                    # Numeric sentiment
                    score = float(match)
                    # Normalize to -1 to 1 if needed
                    if score > 1.0 or score < -1.0:
                        score = max(-1.0, min(1.0, score / 10.0))
                    return score
                elif isinstance(match, str):
                    # Text sentiment
                    sentiment_map = {
                        "very positive": 0.9,
                        "positive": 0.6,
                        "somewhat positive": 0.3,
                        "neutral": 0.0,
                        "somewhat negative": -0.3,
                        "negative": -0.6,
                        "very negative": -0.9,
                        "bullish": 0.7,
                        "somewhat bullish": 0.4,
                        "bearish": -0.7,
                        "somewhat bearish": -0.4
                    }
                    for sentiment, score in sentiment_map.items():
                        if sentiment in match.lower():
                            return score
        
        # If no explicit sentiment found, use confidence as a proxy
        confidence = reasoning.get("confidence", 0.5)
        
        # Analyze the conclusion text for sentiment words
        positive_words = ["positive", "bullish", "uptrend", "growth", "increase", "higher", "strong", "opportunity", "outperform"]
        negative_words = ["negative", "bearish", "downtrend", "decline", "decrease", "lower", "weak", "risk", "underperform"]
        
        positive_count = sum(1 for word in positive_words if word in conclusion.lower())
        negative_count = sum(1 for word in negative_words if word in conclusion.lower())
        
        if positive_count > negative_count:
            return min(0.8, 0.4 + (positive_count - negative_count) * 0.1) * confidence
        elif negative_count > positive_count:
            return max(-0.8, -0.4 - (negative_count - positive_count) * 0.1) * confidence
        else:
            return 0.0
    
    def _sentiment_score_to_label(self, score: float) -> str:
        """
        Convert sentiment score to label.
        
        Parameters:
        -----------
        score : float
            Sentiment score (-1.0 to 1.0).
            
        Returns:
        --------
        str
            Sentiment label.
        """
        if score >= 0.7:
            return "very positive"
        elif score >= 0.3:
            return "positive"
        elif score >= 0.1:
            return "slightly positive"
        elif score <= -0.7:
            return "very negative"
        elif score <= -0.3:
            return "negative"
        elif score <= -0.1:
            return "slightly negative"
        else:
            return "neutral"
    
    def _extract_key_events(self, reasoning: Dict) -> List[Dict]:
        """
        Extract key events from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            List of key events.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for event patterns
        event_patterns = [
            r"key event(?:s)?:?\s*(.*?)(?:\.|$)",
            r"significant event(?:s)?:?\s*(.*?)(?:\.|$)",
            r"important event(?:s)?:?\s*(.*?)(?:\.|$)",
            r"notable event(?:s)?:?\s*(.*?)(?:\.|$)",
            r"event(?:s)?:?\s*(.*?)(?:\.|$)"
        ]
        
        events = []
        for pattern in event_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 10:  # Minimum length to be meaningful
                    events.append(match.strip())
        
        # If no events found using patterns, extract from reasoning steps
        if not events and reasoning.get("steps"):
            for step in reasoning.get("steps"):
                if "announce" in step.lower() or "report" in step.lower() or "launch" in step.lower() or "release" in step.lower():
                    events.append(step.strip())
        
        # Format events with sentiment
        formatted_events = []
        for i, event in enumerate(events[:5]):  # Limit to top 5 events
            event_sentiment = self._analyze_text_sentiment(event)
            formatted_events.append({
                "description": event,
                "sentiment": self._sentiment_score_to_label(event_sentiment),
                "sentiment_score": event_sentiment,
                "importance": 1.0 - (i * 0.15)  # Decreasing importance
            })
        
        return formatted_events
    
    def _extract_topics(self, reasoning: Dict, news_articles: List[Dict]) -> List[Dict]:
        """
        Extract main topics from reasoning and news articles.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
        news_articles : List[Dict]
            List of news articles.
            
        Returns:
        --------
        List[Dict]
            List of main topics.
        """
        # Combine all text from news articles
        all_news_text = " ".join([article.get("title", "") + " " + article.get("summary", "") for article in news_articles])
        
        # Combine with reasoning
        all_text = all_news_text + " " + " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Define topic categories
        topic_categories = {
            "earnings": ["earnings", "revenue", "profit", "eps", "income", "financial results", "quarterly results"],
            "products": ["product", "launch", "release", "new", "innovation", "feature"],
            "management": ["ceo", "executive", "management", "leadership", "board", "appointed", "resigned"],
            "partnerships": ["partnership", "collaboration", "agreement", "deal", "contract", "alliance"],
            "regulatory": ["regulation", "compliance", "legal", "lawsuit", "settlement", "fine", "investigation"],
            "market_position": ["market share", "competition", "competitor", "industry", "sector", "position"],
            "expansion": ["expansion", "growth", "new market", "international", "global", "acquisition", "merger"],
            "financial_health": ["debt", "cash", "balance sheet", "liquidity", "solvency", "dividend", "buyback"],
            "technology": ["technology", "tech", "innovation", "patent", "r&d", "research", "development"],
            "operations": ["operations", "supply chain", "production", "manufacturing", "logistics", "efficiency"]
        }
        
        # Count topic mentions
        topic_counts = {}
        for category, keywords in topic_categories.items():
            count = sum(all_text.lower().count(keyword) for keyword in keywords)
            if count > 0:
                topic_counts[category] = count
        
        # Sort by count and format
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        formatted_topics = []
        total_mentions = sum(count for _, count in sorted_topics)
        
        for category, count in sorted_topics[:5]:  # Limit to top 5 topics
            # Calculate relevance as percentage of total mentions
            relevance = count / total_mentions if total_mentions > 0 else 0
            
            # Get sentiment for this topic
            topic_sentiment = self._analyze_topic_sentiment(category, reasoning)
            
            formatted_topics.append({
                "category": category.replace("_", " ").title(),
                "relevance": relevance,
                "mention_count": count,
                "sentiment": self._sentiment_score_to_label(topic_sentiment),
                "sentiment_score": topic_sentiment
            })
        
        return formatted_topics
    
    def _analyze_topic_sentiment(self, topic: str, reasoning: Dict) -> float:
        """
        Analyze sentiment for a specific topic.
        
        Parameters:
        -----------
        topic : str
            Topic category.
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        float
            Sentiment score (-1.0 to 1.0).
        """
        # Convert topic to keywords
        topic_keywords = topic.replace("_", " ").split()
        
        # Find sentences containing topic keywords
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        sentences = re.split(r'[.!?]+', all_text)
        
        topic_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in topic_keywords):
                topic_sentences.append(sentence)
        
        if not topic_sentences:
            return 0.0
        
        # Analyze sentiment of topic sentences
        sentiments = [self._analyze_text_sentiment(sentence) for sentence in topic_sentences]
        
        # Return average sentiment
        return sum(sentiments) / len(sentiments)
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a text snippet.
        
        Parameters:
        -----------
        text : str
            Text to analyze.
            
        Returns:
        --------
        float
            Sentiment score (-1.0 to 1.0).
        """
        positive_words = [
            "positive", "bullish", "uptrend", "growth", "increase", "higher", "strong", "opportunity", "outperform",
            "beat", "exceed", "success", "successful", "gain", "improve", "improved", "improving", "improvement",
            "advantage", "advantageous", "profit", "profitable", "win", "winning", "winner", "good", "great",
            "excellent", "outstanding", "remarkable", "impressive", "favorable", "optimistic", "confident"
        ]
        
        negative_words = [
            "negative", "bearish", "downtrend", "decline", "decrease", "lower", "weak", "risk", "underperform",
            "miss", "missed", "failure", "fail", "failed", "failing", "loss", "lose", "losing", "loser",
            "disadvantage", "disadvantageous", "unprofitable", "bad", "poor", "terrible", "disappointing",
            "unfavorable", "pessimistic", "concerned", "concerning", "worry", "worried", "worrying"
        ]
        
        intensifiers = [
            "very", "extremely", "highly", "significantly", "substantially", "considerably", "notably",
            "markedly", "exceptionally", "extraordinarily", "remarkably", "particularly", "especially"
        ]
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in positive_words if word in words)
        negative_count = sum(1 for word in negative_words if word in words)
        
        # Check for intensifiers
        for i, word in enumerate(words[:-1]):
            if word in intensifiers:
                if words[i+1] in positive_words:
                    positive_count += 0.5
                elif words[i+1] in negative_words:
                    negative_count += 0.5
        
        # Check for negations
        negations = ["not", "no", "never", "neither", "nor", "none", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't"]
        for i, word in enumerate(words[:-1]):
            if word in negations:
                if words[i+1] in positive_words:
                    positive_count -= 1
                    negative_count += 1
                elif words[i+1] in negative_words:
                    negative_count -= 1
                    positive_count += 1
        
        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count
    
    def _extract_potential_catalysts(self, reasoning: Dict) -> List[Dict]:
        """
        Extract potential catalysts from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            List of potential catalysts.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for catalyst patterns
        catalyst_patterns = [
            r"catalyst(?:s)?:?\s*(.*?)(?:\.|$)",
            r"potential catalyst(?:s)?:?\s*(.*?)(?:\.|$)",
            r"upcoming catalyst(?:s)?:?\s*(.*?)(?:\.|$)",
            r"future catalyst(?:s)?:?\s*(.*?)(?:\.|$)",
            r"could (?:be|act as|serve as) (?:a|an) catalyst:?\s*(.*?)(?:\.|$)"
        ]
        
        catalysts = []
        for pattern in catalyst_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 10:  # Minimum length to be meaningful
                    catalysts.append(match.strip())
        
        # If no catalysts found using patterns, look for potential future events
        if not catalysts:
            future_patterns = [
                r"upcoming(?:.*?)(?:event|announcement|release|report):?\s*(.*?)(?:\.|$)",
                r"scheduled(?:.*?)(?:event|announcement|release|report):?\s*(.*?)(?:\.|$)",
                r"expected(?:.*?)(?:event|announcement|release|report):?\s*(.*?)(?:\.|$)",
                r"future(?:.*?)(?:event|announcement|release|report):?\s*(.*?)(?:\.|$)"
            ]
            
            for pattern in future_patterns:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                for match in matches:
                    if match and len(match) > 10:  # Minimum length to be meaningful
                        catalysts.append(match.strip())
        
        # Format catalysts with impact assessment
        formatted_catalysts = []
        for i, catalyst in enumerate(catalysts[:3]):  # Limit to top 3 catalysts
            catalyst_sentiment = self._analyze_text_sentiment(catalyst)
            
            # Determine timeframe (short-term, medium-term, long-term)
            timeframe = "medium-term"  # Default
            if "upcoming" in catalyst.lower() or "soon" in catalyst.lower() or "imminent" in catalyst.lower():
                timeframe = "short-term"
            elif "future" in catalyst.lower() or "long" in catalyst.lower() or "eventually" in catalyst.lower():
                timeframe = "long-term"
            
            # Determine potential impact
            impact = "moderate"  # Default
            impact_words = {
                "high": ["significant", "major", "substantial", "considerable", "dramatic", "strong"],
                "low": ["minor", "slight", "small", "minimal", "marginal", "limited"]
            }
            
            for impact_level, words in impact_words.items():
                if any(word in catalyst.lower() for word in words):
                    impact = impact_level
                    break
            
            formatted_catalysts.append({
                "description": catalyst,
                "expected_impact": impact,
                "timeframe": timeframe,
                "sentiment": self._sentiment_score_to_label(catalyst_sentiment),
                "sentiment_score": catalyst_sentiment
            })
        
        return formatted_catalysts


class EarningsAnalyzer:
    """
    Analyzes earnings reports and related information.
    
    This class processes earnings reports, earnings calls transcripts, and
    analyst estimates to extract insights for trading decisions.
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
        self.prompt_engine = gemma_core.prompt_engine
        self.data_integration = gemma_core.data_integration
        self.logger.info("Initialized EarningsAnalyzer")
    
    def analyze_earnings_report(self, ticker: str, report_data: Dict) -> Dict:
        """
        Analyze an earnings report.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        report_data : Dict
            Earnings report data.
            
        Returns:
        --------
        Dict
            Earnings analysis results.
        """
        self.logger.info(f"Analyzing earnings report for {ticker}")
        
        try:
            # Generate prompt for earnings analysis
            prompt = self.prompt_engine.generate_prompt(
                "earnings_analysis",
                ticker=ticker,
                report_data=json.dumps(report_data, indent=2)
            ) if "earnings_analysis" in self.prompt_engine.templates else self.prompt_engine._generate_default_prompt(
                task="Analyze the following earnings report",
                ticker=ticker,
                report_data=json.dumps(report_data, indent=2)
            )
            
            # Generate reasoning using Gemma 3
            reasoning = self.gemma_core.cot_processor.generate_reasoning(prompt)
            
            # Extract key metrics and insights
            key_metrics = self._extract_key_metrics(report_data)
            performance_vs_estimates = self._analyze_performance_vs_estimates(report_data)
            growth_trends = self._analyze_growth_trends(report_data)
            guidance_analysis = self._analyze_guidance(report_data, reasoning)
            management_insights = self._extract_management_insights(reasoning)
            
            # Prepare analysis result
            analysis = {
                "ticker": ticker,
                "report_date": report_data.get("report_date", datetime.datetime.now().isoformat()),
                "timestamp": datetime.datetime.now().isoformat(),
                "key_metrics": key_metrics,
                "performance_vs_estimates": performance_vs_estimates,
                "growth_trends": growth_trends,
                "guidance_analysis": guidance_analysis,
                "management_insights": management_insights,
                "overall_assessment": {
                    "sentiment": self._determine_overall_sentiment(reasoning, performance_vs_estimates),
                    "key_strengths": self._extract_strengths(reasoning),
                    "key_concerns": self._extract_concerns(reasoning),
                    "potential_impact": self._assess_potential_impact(reasoning)
                },
                "reasoning": reasoning
            }
            
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze earnings report: {e}")
            return {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _extract_key_metrics(self, report_data: Dict) -> Dict:
        """
        Extract key metrics from earnings report.
        
        Parameters:
        -----------
        report_data : Dict
            Earnings report data.
            
        Returns:
        --------
        Dict
            Key metrics.
        """
        metrics = {}
        
        # Extract common financial metrics
        metric_keys = [
            "revenue", "earnings", "eps", "net_income", "operating_income",
            "gross_margin", "operating_margin", "net_margin", "ebitda",
            "free_cash_flow", "cash", "debt", "equity"
        ]
        
        for key in metric_keys:
            if key in report_data:
                metrics[key] = report_data[key]
        
        # Add formatted metrics
        if "revenue" in metrics and "prior_revenue" in report_data:
            metrics["revenue_growth"] = (metrics["revenue"] / report_data["prior_revenue"] - 1) * 100
        
        if "eps" in metrics and "prior_eps" in report_data:
            metrics["eps_growth"] = (metrics["eps"] / report_data["prior_eps"] - 1) * 100
        
        return metrics
    
    def _analyze_performance_vs_estimates(self, report_data: Dict) -> Dict:
        """
        Analyze performance versus analyst estimates.
        
        Parameters:
        -----------
        report_data : Dict
            Earnings report data.
            
        Returns:
        --------
        Dict
            Performance versus estimates.
        """
        performance = {}
        
        # Compare actual results to estimates
        metric_comparisons = ["revenue", "eps", "net_income"]
        
        for metric in metric_comparisons:
            actual_key = metric
            estimate_key = f"{metric}_estimate"
            
            if actual_key in report_data and estimate_key in report_data:
                actual = report_data[actual_key]
                estimate = report_data[estimate_key]
                
                if estimate != 0:
                    surprise_pct = (actual / estimate - 1) * 100
                    
                    # Determine beat/miss/meet
                    if surprise_pct > 2:
                        result = "beat"
                    elif surprise_pct < -2:
                        result = "miss"
                    else:
                        result = "meet"
                    
                    performance[metric] = {
                        "actual": actual,
                        "estimate": estimate,
                        "surprise_pct": surprise_pct,
                        "result": result
                    }
        
        return performance
    
    def _analyze_growth_trends(self, report_data: Dict) -> Dict:
        """
        Analyze growth trends from earnings report.
        
        Parameters:
        -----------
        report_data : Dict
            Earnings report data.
            
        Returns:
        --------
        Dict
            Growth trends.
        """
        trends = {}
        
        # Analyze year-over-year and sequential growth
        growth_metrics = ["revenue", "eps", "net_income", "operating_income"]
        
        for metric in growth_metrics:
            current_key = metric
            prior_year_key = f"prior_year_{metric}"
            prior_quarter_key = f"prior_quarter_{metric}"
            
            if current_key in report_data:
                current = report_data[current_key]
                
                # Year-over-year growth
                if prior_year_key in report_data and report_data[prior_year_key] != 0:
                    yoy_growth = (current / report_data[prior_year_key] - 1) * 100
                    
                    # Determine trend
                    if yoy_growth > 15:
                        trend = "strong growth"
                    elif yoy_growth > 5:
                        trend = "moderate growth"
                    elif yoy_growth > 0:
                        trend = "slight growth"
                    elif yoy_growth > -5:
                        trend = "slight decline"
                    else:
                        trend = "significant decline"
                    
                    trends[f"{metric}_yoy"] = {
                        "growth_pct": yoy_growth,
                        "trend": trend
                    }
                
                # Sequential growth
                if prior_quarter_key in report_data and report_data[prior_quarter_key] != 0:
                    qoq_growth = (current / report_data[prior_quarter_key] - 1) * 100
                    
                    # Determine trend
                    if qoq_growth > 10:
                        trend = "strong growth"
                    elif qoq_growth > 3:
                        trend = "moderate growth"
                    elif qoq_growth > 0:
                        trend = "slight growth"
                    elif qoq_growth > -3:
                        trend = "slight decline"
                    else:
                        trend = "significant decline"
                    
                    trends[f"{metric}_qoq"] = {
                        "growth_pct": qoq_growth,
                        "trend": trend
                    }
        
        return trends
    
    def _analyze_guidance(self, report_data: Dict, reasoning: Dict) -> Dict:
        """
        Analyze company guidance from earnings report.
        
        Parameters:
        -----------
        report_data : Dict
            Earnings report data.
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        Dict
            Guidance analysis.
        """
        guidance = {}
        
        # Extract guidance from report data
        if "guidance" in report_data:
            guidance_data = report_data["guidance"]
            
            # Process each guidance metric
            for metric, data in guidance_data.items():
                if isinstance(data, dict):
                    guidance[metric] = data
        
        # If no structured guidance data, extract from reasoning
        if not guidance:
            all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
            
            # Look for guidance patterns
            guidance_patterns = [
                r"guidance(?:.*?)(?:for|on)(?:.*?)(\w+)(?:.*?)(?:is|of|at)(?:.*?)([\d\.]+)(?:.*?)(?:to|-)(?:.*?)([\d\.]+)",
                r"(\w+)(?:.*?)guidance(?:.*?)(?:is|of|at)(?:.*?)([\d\.]+)(?:.*?)(?:to|-)(?:.*?)([\d\.]+)",
                r"guided(?:.*?)(\w+)(?:.*?)(?:of|at)(?:.*?)([\d\.]+)(?:.*?)(?:to|-)(?:.*?)([\d\.]+)"
            ]
            
            for pattern in guidance_patterns:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 3:
                        metric = match[0].lower()
                        try:
                            low = float(match[1])
                            high = float(match[2])
                            
                            guidance[metric] = {
                                "low": low,
                                "high": high,
                                "midpoint": (low + high) / 2
                            }
                        except ValueError:
                            continue
        
        # Analyze guidance versus expectations
        for metric, data in guidance.items():
            estimate_key = f"{metric}_estimate"
            
            if estimate_key in report_data:
                estimate = report_data[estimate_key]
                midpoint = data.get("midpoint", (data.get("low", 0) + data.get("high", 0)) / 2)
                
                if estimate != 0:
                    surprise_pct = (midpoint / estimate - 1) * 100
                    
                    # Determine raised/lowered/maintained
                    if surprise_pct > 2:
                        result = "raised"
                    elif surprise_pct < -2:
                        result = "lowered"
                    else:
                        result = "maintained"
                    
                    guidance[metric]["vs_estimate"] = {
                        "estimate": estimate,
                        "surprise_pct": surprise_pct,
                        "result": result
                    }
        
        return guidance
    
    def _extract_management_insights(self, reasoning: Dict) -> List[Dict]:
        """
        Extract management insights from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            Management insights.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for management comment patterns
        management_patterns = [
            r"management(?:.*?)(?:stated|mentioned|noted|commented|highlighted|emphasized)(?:.*?)that(.*?)(?:\.|$)",
            r"(?:ceo|cfo|executive)(?:.*?)(?:stated|mentioned|noted|commented|highlighted|emphasized)(?:.*?)that(.*?)(?:\.|$)",
            r"(?:according to|per)(?:.*?)management(.*?)(?:\.|$)",
            r"(?:on the|during the)(?:.*?)(?:call|earnings call|conference call)(?:.*?)management(?:.*?)(?:stated|mentioned|noted|commented|highlighted|emphasized)(?:.*?)that(.*?)(?:\.|$)"
        ]
        
        insights = []
        for pattern in management_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 15:  # Minimum length to be meaningful
                    insights.append(match.strip())
        
        # Format insights with topics and sentiment
        formatted_insights = []
        for i, insight in enumerate(insights[:5]):  # Limit to top 5 insights
            # Determine topic
            topics = ["growth", "margins", "competition", "products", "strategy", "outlook", "challenges", "opportunities"]
            insight_topic = "general"
            
            for topic in topics:
                if topic in insight.lower():
                    insight_topic = topic
                    break
            
            # Analyze sentiment
            sentiment_score = self._analyze_text_sentiment(insight)
            
            formatted_insights.append({
                "statement": insight,
                "topic": insight_topic,
                "sentiment": self._sentiment_score_to_label(sentiment_score),
                "sentiment_score": sentiment_score,
                "importance": 1.0 - (i * 0.15)  # Decreasing importance
            })
        
        return formatted_insights
    
    def _determine_overall_sentiment(self, reasoning: Dict, performance: Dict) -> str:
        """
        Determine overall sentiment from reasoning and performance.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
        performance : Dict
            Performance versus estimates.
            
        Returns:
        --------
        str
            Overall sentiment.
        """
        # Extract sentiment from reasoning
        conclusion = reasoning.get("conclusion", "")
        
        # Look for explicit sentiment mentions
        sentiment_patterns = [
            r"overall(?:.*?)(?:sentiment|assessment|outlook)(?:.*?)(?:is|remains)(?:.*?)(\w+)",
            r"(?:sentiment|assessment|outlook)(?:.*?)(?:is|remains)(?:.*?)(\w+)"
        ]
        
        for pattern in sentiment_patterns:
            matches = re.findall(pattern, conclusion.lower())
            if matches:
                sentiment = matches[0]
                
                # Map to standard sentiment labels
                sentiment_map = {
                    "positive": "positive",
                    "very positive": "very positive",
                    "somewhat positive": "slightly positive",
                    "neutral": "neutral",
                    "somewhat negative": "slightly negative",
                    "negative": "negative",
                    "very negative": "very negative",
                    "bullish": "positive",
                    "very bullish": "very positive",
                    "somewhat bullish": "slightly positive",
                    "bearish": "negative",
                    "very bearish": "very negative",
                    "somewhat bearish": "slightly negative",
                    "mixed": "neutral",
                    "strong": "positive",
                    "weak": "negative",
                    "solid": "positive",
                    "poor": "negative",
                    "excellent": "very positive",
                    "good": "positive",
                    "bad": "negative",
                    "terrible": "very negative"
                }
                
                for key, value in sentiment_map.items():
                    if key in sentiment:
                        return value
        
        # If no explicit sentiment found, infer from performance
        if performance:
            beat_count = sum(1 for metric in performance.values() if metric.get("result") == "beat")
            miss_count = sum(1 for metric in performance.values() if metric.get("result") == "miss")
            meet_count = sum(1 for metric in performance.values() if metric.get("result") == "meet")
            
            total_count = beat_count + miss_count + meet_count
            
            if total_count > 0:
                beat_ratio = beat_count / total_count
                miss_ratio = miss_count / total_count
                
                if beat_ratio > 0.7:
                    return "very positive"
                elif beat_ratio > 0.5:
                    return "positive"
                elif beat_ratio > 0.3:
                    return "slightly positive"
                elif miss_ratio > 0.7:
                    return "very negative"
                elif miss_ratio > 0.5:
                    return "negative"
                elif miss_ratio > 0.3:
                    return "slightly negative"
                else:
                    return "neutral"
        
        # Default to neutral if no clear sentiment can be determined
        return "neutral"
    
    def _extract_strengths(self, reasoning: Dict) -> List[str]:
        """
        Extract key strengths from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[str]
            Key strengths.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for strength patterns
        strength_patterns = [
            r"(?:key|main|notable|significant)(?:.*?)strength(?:s)?(?:.*?)(?:include|are|were|was|is)(?:.*?)(.*?)(?:\.|$)",
            r"strength(?:s)?(?:.*?)(?:include|are|were|was|is)(?:.*?)(.*?)(?:\.|$)",
            r"positive(?:.*?)(?:aspect|factor|point|highlight|takeaway)(?:s)?(?:.*?)(?:include|are|were|was|is)(?:.*?)(.*?)(?:\.|$)",
            r"(?:company|firm)(?:.*?)(?:performed well|excelled|outperformed)(?:.*?)(?:in|with|on)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        strengths = []
        for pattern in strength_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 10:  # Minimum length to be meaningful
                    strengths.append(match.strip())
        
        # If no strengths found using patterns, look for positive statements
        if not strengths:
            sentences = re.split(r'[.!?]+', all_text)
            
            positive_words = ["strong", "growth", "increase", "higher", "beat", "exceed", "outperform", "improve", "positive"]
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in positive_words) and len(sentence) > 20:
                    strengths.append(sentence.strip())
        
        # Limit to top 3 strengths and remove duplicates
        unique_strengths = []
        for strength in strengths[:5]:
            # Check if this strength is substantially different from those already included
            if not any(self._text_similarity(strength, existing) > 0.7 for existing in unique_strengths):
                unique_strengths.append(strength)
                if len(unique_strengths) >= 3:
                    break
        
        return unique_strengths
    
    def _extract_concerns(self, reasoning: Dict) -> List[str]:
        """
        Extract key concerns from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[str]
            Key concerns.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for concern patterns
        concern_patterns = [
            r"(?:key|main|notable|significant)(?:.*?)concern(?:s)?(?:.*?)(?:include|are|were|was|is)(?:.*?)(.*?)(?:\.|$)",
            r"concern(?:s)?(?:.*?)(?:include|are|were|was|is)(?:.*?)(.*?)(?:\.|$)",
            r"negative(?:.*?)(?:aspect|factor|point|highlight|takeaway)(?:s)?(?:.*?)(?:include|are|were|was|is)(?:.*?)(.*?)(?:\.|$)",
            r"(?:challenge|risk|headwind|issue|problem)(?:s)?(?:.*?)(?:include|are|were|was|is)(?:.*?)(.*?)(?:\.|$)",
            r"(?:company|firm)(?:.*?)(?:struggled|underperformed|faced challenges)(?:.*?)(?:in|with|on)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        concerns = []
        for pattern in concern_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 10:  # Minimum length to be meaningful
                    concerns.append(match.strip())
        
        # If no concerns found using patterns, look for negative statements
        if not concerns:
            sentences = re.split(r'[.!?]+', all_text)
            
            negative_words = ["weak", "decline", "decrease", "lower", "miss", "below", "underperform", "deteriorate", "negative", "challenge", "risk"]
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in negative_words) and len(sentence) > 20:
                    concerns.append(sentence.strip())
        
        # Limit to top 3 concerns and remove duplicates
        unique_concerns = []
        for concern in concerns[:5]:
            # Check if this concern is substantially different from those already included
            if not any(self._text_similarity(concern, existing) > 0.7 for existing in unique_concerns):
                unique_concerns.append(concern)
                if len(unique_concerns) >= 3:
                    break
        
        return unique_concerns
    
    def _assess_potential_impact(self, reasoning: Dict) -> Dict:
        """
        Assess potential impact of earnings on stock price.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        Dict
            Potential impact assessment.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for impact patterns
        impact_patterns = [
            r"(?:potential|expected|likely|anticipated)(?:.*?)impact(?:.*?)(?:on|to)(?:.*?)(?:stock|price|shares)(?:.*?)(?:is|could be|might be|would be)(?:.*?)(.*?)(?:\.|$)",
            r"(?:stock|price|shares)(?:.*?)(?:could|might|may|should|would)(?:.*?)(?:move|react|respond)(?:.*?)(.*?)(?:\.|$)",
            r"(?:expect|anticipate|forecast|predict)(?:.*?)(?:stock|price|shares)(?:.*?)(?:to|will)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        impact_statements = []
        for pattern in impact_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 10:  # Minimum length to be meaningful
                    impact_statements.append(match.strip())
        
        # Determine direction and magnitude
        direction = "neutral"  # Default
        magnitude = "moderate"  # Default
        
        # Direction keywords
        positive_keywords = ["up", "higher", "increase", "rise", "positive", "upward", "gain", "appreciate", "rally", "bullish"]
        negative_keywords = ["down", "lower", "decrease", "fall", "negative", "downward", "decline", "depreciate", "sell off", "bearish"]
        
        # Magnitude keywords
        strong_keywords = ["significant", "substantial", "considerable", "major", "strong", "sharp", "dramatic", "large"]
        mild_keywords = ["slight", "modest", "minor", "small", "limited", "marginal", "minimal"]
        
        # Analyze impact statements
        for statement in impact_statements:
            statement_lower = statement.lower()
            
            # Determine direction
            if any(keyword in statement_lower for keyword in positive_keywords):
                direction = "positive"
            elif any(keyword in statement_lower for keyword in negative_keywords):
                direction = "negative"
            
            # Determine magnitude
            if any(keyword in statement_lower for keyword in strong_keywords):
                magnitude = "strong"
            elif any(keyword in statement_lower for keyword in mild_keywords):
                magnitude = "mild"
        
        # Prepare impact assessment
        impact = {
            "direction": direction,
            "magnitude": magnitude,
            "timeframe": "short-term",  # Default for earnings impact
            "confidence": reasoning.get("confidence", 0.5),
            "statement": impact_statements[0] if impact_statements else "No specific impact assessment provided."
        }
        
        return impact
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a text snippet.
        
        Parameters:
        -----------
        text : str
            Text to analyze.
            
        Returns:
        --------
        float
            Sentiment score (-1.0 to 1.0).
        """
        positive_words = [
            "positive", "bullish", "uptrend", "growth", "increase", "higher", "strong", "opportunity", "outperform",
            "beat", "exceed", "success", "successful", "gain", "improve", "improved", "improving", "improvement",
            "advantage", "advantageous", "profit", "profitable", "win", "winning", "winner", "good", "great",
            "excellent", "outstanding", "remarkable", "impressive", "favorable", "optimistic", "confident"
        ]
        
        negative_words = [
            "negative", "bearish", "downtrend", "decline", "decrease", "lower", "weak", "risk", "underperform",
            "miss", "missed", "failure", "fail", "failed", "failing", "loss", "lose", "losing", "loser",
            "disadvantage", "disadvantageous", "unprofitable", "bad", "poor", "terrible", "disappointing",
            "unfavorable", "pessimistic", "concerned", "concerning", "worry", "worried", "worrying"
        ]
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in positive_words if word in words)
        negative_count = sum(1 for word in negative_words if word in words)
        
        # Check for negations
        negations = ["not", "no", "never", "neither", "nor", "none", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't"]
        for i, word in enumerate(words[:-1]):
            if word in negations:
                if words[i+1] in positive_words:
                    positive_count -= 1
                    negative_count += 1
                elif words[i+1] in negative_words:
                    negative_count -= 1
                    positive_count += 1
        
        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count
    
    def _sentiment_score_to_label(self, score: float) -> str:
        """
        Convert sentiment score to label.
        
        Parameters:
        -----------
        score : float
            Sentiment score (-1.0 to 1.0).
            
        Returns:
        --------
        str
            Sentiment label.
        """
        if score >= 0.7:
            return "very positive"
        elif score >= 0.3:
            return "positive"
        elif score >= 0.1:
            return "slightly positive"
        elif score <= -0.7:
            return "very negative"
        elif score <= -0.3:
            return "negative"
        elif score <= -0.1:
            return "slightly negative"
        else:
            return "neutral"
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text snippets.
        
        Parameters:
        -----------
        text1 : str
            First text.
        text2 : str
            Second text.
            
        Returns:
        --------
        float
            Similarity score (0.0 to 1.0).
        """
        # Simple word overlap similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


class SocialMediaAnalyzer:
    """
    Analyzes social media sentiment and discussions.
    
    This class processes social media data from platforms like Twitter, Reddit,
    and StockTwits to extract sentiment and identify trending topics related to
    specific tickers or the broader market.
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
        self.prompt_engine = gemma_core.prompt_engine
        self.data_integration = gemma_core.data_integration
        self.logger.info("Initialized SocialMediaAnalyzer")
    
    def analyze_social_sentiment(self, ticker: str, days: int = 7) -> Dict:
        """
        Analyze social media sentiment for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        days : int
            Number of days to look back.
            
        Returns:
        --------
        Dict
            Social sentiment analysis results.
        """
        self.logger.info(f"Analyzing social sentiment for {ticker} (days: {days})")
        
        try:
            # Fetch social sentiment data
            sentiment_data = self.data_integration.fetch_social_sentiment(ticker, days)
            
            if not sentiment_data or "error" in sentiment_data:
                self.logger.warning(f"No social sentiment data found for {ticker}")
                return {
                    "ticker": ticker,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "volume": 0,
                    "trending_topics": [],
                    "key_discussions": []
                }
            
            # Generate prompt for social sentiment analysis
            prompt = self.prompt_engine.generate_prompt(
                "social_sentiment_analysis",
                ticker=ticker,
                sentiment_data=json.dumps(sentiment_data, indent=2)
            ) if "social_sentiment_analysis" in self.prompt_engine.templates else self.prompt_engine._generate_default_prompt(
                task="Analyze the following social media sentiment data",
                ticker=ticker,
                sentiment_data=json.dumps(sentiment_data, indent=2)
            )
            
            # Generate reasoning using Gemma 3
            reasoning = self.gemma_core.cot_processor.generate_reasoning(prompt)
            
            # Extract key information
            overall_sentiment = sentiment_data.get("overall_sentiment", 0.0)
            sentiment_label = self._sentiment_score_to_label(overall_sentiment)
            volume = sentiment_data.get("volume", 0)
            volume_trend = sentiment_data.get("volume_trend", 0.0)
            
            # Extract trending topics
            trending_topics = self._extract_trending_topics(sentiment_data, reasoning)
            
            # Extract key discussions
            key_discussions = self._extract_key_discussions(reasoning)
            
            # Prepare analysis result
            analysis = {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "overall_sentiment": sentiment_label,
                "sentiment_score": overall_sentiment,
                "volume": volume,
                "volume_trend": volume_trend,
                "trending_topics": trending_topics,
                "key_discussions": key_discussions,
                "source_breakdown": sentiment_data.get("sources", {}),
                "unusual_activity": self._detect_unusual_activity(sentiment_data, reasoning),
                "trading_signals": self._extract_trading_signals(reasoning),
                "reasoning": reasoning
            }
            
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze social sentiment: {e}")
            return {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _extract_trending_topics(self, sentiment_data: Dict, reasoning: Dict) -> List[Dict]:
        """
        Extract trending topics from sentiment data and reasoning.
        
        Parameters:
        -----------
        sentiment_data : Dict
            Social sentiment data.
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            Trending topics.
        """
        topics = []
        
        # First try to get topics from sentiment data
        if "top_topics" in sentiment_data and sentiment_data["top_topics"]:
            for topic_data in sentiment_data["top_topics"]:
                topics.append({
                    "topic": topic_data.get("topic", "unknown"),
                    "count": topic_data.get("count", 0),
                    "sentiment": self._sentiment_score_to_label(topic_data.get("sentiment", 0.0)),
                    "sentiment_score": topic_data.get("sentiment", 0.0)
                })
        
        # If no topics found, extract from reasoning
        if not topics:
            # Combine all reasoning steps and conclusion
            all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
            
            # Look for topic patterns
            topic_patterns = [
                r"(?:trending|popular|key|main|hot)(?:.*?)topic(?:s)?(?:.*?)(?:include|are|were|is)(?:.*?)(.*?)(?:\.|$)",
                r"(?:discussion|conversation)(?:s)?(?:.*?)(?:focus|centered|revolve)(?:.*?)(?:on|around)(?:.*?)(.*?)(?:\.|$)",
                r"(?:users|investors|traders)(?:.*?)(?:discussing|talking about|focused on)(?:.*?)(.*?)(?:\.|$)"
            ]
            
            topic_mentions = []
            for pattern in topic_patterns:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                for match in matches:
                    if match and len(match) > 5:  # Minimum length to be meaningful
                        topic_mentions.append(match.strip())
            
            # Extract individual topics from mentions
            if topic_mentions:
                # Split by common separators
                individual_topics = []
                for mention in topic_mentions:
                    # Try to split by common separators
                    splits = re.split(r',|\band\b|\bas well as\b|\balong with\b|;', mention)
                    individual_topics.extend([s.strip() for s in splits if s.strip()])
                
                # Create topic entries
                for i, topic in enumerate(individual_topics[:5]):  # Limit to top 5
                    sentiment_score = self._analyze_text_sentiment(topic)
                    topics.append({
                        "topic": topic,
                        "count": 100 - (i * 20),  # Placeholder count
                        "sentiment": self._sentiment_score_to_label(sentiment_score),
                        "sentiment_score": sentiment_score
                    })
        
        return topics
    
    def _extract_key_discussions(self, reasoning: Dict) -> List[Dict]:
        """
        Extract key discussions from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            Key discussions.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for discussion patterns
        discussion_patterns = [
            r"(?:key|main|notable|significant)(?:.*?)discussion(?:s)?(?:.*?)(?:include|are|were|is)(?:.*?)(.*?)(?:\.|$)",
            r"(?:users|investors|traders)(?:.*?)(?:discussing|talking about|mentioned|highlighted)(?:.*?)(.*?)(?:\.|$)",
            r"(?:popular|common|frequent)(?:.*?)(?:post|comment|mention)(?:s)?(?:.*?)(?:include|are|were|is)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        discussions = []
        for pattern in discussion_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 15:  # Minimum length to be meaningful
                    discussions.append(match.strip())
        
        # Format discussions with sentiment
        formatted_discussions = []
        for i, discussion in enumerate(discussions[:5]):  # Limit to top 5
            sentiment_score = self._analyze_text_sentiment(discussion)
            
            # Try to identify source platform
            source = "unknown"
            platform_keywords = {
                "twitter": ["tweet", "twitter", "tweeted", "retweet"],
                "reddit": ["reddit", "subreddit", "post", "thread", "r/"],
                "stocktwits": ["stocktwits", "twit", "st"]
            }
            
            for platform, keywords in platform_keywords.items():
                if any(keyword in discussion.lower() for keyword in keywords):
                    source = platform
                    break
            
            formatted_discussions.append({
                "content": discussion,
                "source": source,
                "sentiment": self._sentiment_score_to_label(sentiment_score),
                "sentiment_score": sentiment_score,
                "engagement": "high" if i < 2 else "medium" if i < 4 else "low"  # Placeholder
            })
        
        return formatted_discussions
    
    def _detect_unusual_activity(self, sentiment_data: Dict, reasoning: Dict) -> Dict:
        """
        Detect unusual social media activity.
        
        Parameters:
        -----------
        sentiment_data : Dict
            Social sentiment data.
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        Dict
            Unusual activity detection.
        """
        # Check for volume spikes
        volume = sentiment_data.get("volume", 0)
        volume_trend = sentiment_data.get("volume_trend", 0.0)
        
        volume_spike = volume_trend > 0.5  # 50% increase
        
        # Check for sentiment shifts
        sentiment_trend = sentiment_data.get("sentiment_trend", 0.0)
        sentiment_shift = abs(sentiment_trend) > 0.3  # 0.3 shift on -1 to 1 scale
        
        # Look for unusual activity mentions in reasoning
        unusual_activity_mentioned = False
        unusual_activity_description = ""
        
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        unusual_patterns = [
            r"unusual(?:.*?)(?:activity|volume|interest|attention|sentiment)(?:.*?)(.*?)(?:\.|$)",
            r"(?:significant|notable|marked|substantial)(?:.*?)(?:increase|spike|jump|surge|shift|change)(?:.*?)(?:in|of)(?:.*?)(?:activity|volume|interest|attention|sentiment)(?:.*?)(.*?)(?:\.|$)",
            r"(?:abnormal|atypical|unexpected)(?:.*?)(?:activity|volume|interest|attention|sentiment)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        for pattern in unusual_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                unusual_activity_mentioned = True
                unusual_activity_description = matches[0].strip()
                break
        
        # Determine if there's unusual activity
        is_unusual = volume_spike or sentiment_shift or unusual_activity_mentioned
        
        # Prepare result
        unusual_activity = {
            "detected": is_unusual,
            "volume_spike": volume_spike,
            "sentiment_shift": sentiment_shift,
            "description": unusual_activity_description if unusual_activity_mentioned else "No unusual activity detected in social media discussions."
        }
        
        return unusual_activity
    
    def _extract_trading_signals(self, reasoning: Dict) -> List[Dict]:
        """
        Extract potential trading signals from social sentiment.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            Potential trading signals.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for signal patterns
        signal_patterns = [
            r"(?:trading|investment)(?:.*?)(?:signal|opportunity|implication)(?:s)?(?:.*?)(?:include|are|is|suggest|indicate)(?:.*?)(.*?)(?:\.|$)",
            r"(?:sentiment|activity|discussion)(?:.*?)(?:suggest|indicate|point to|imply)(?:.*?)(.*?)(?:\.|$)",
            r"(?:based on|given)(?:.*?)(?:sentiment|activity|discussion)(?:.*?)(?:investor|trader)(?:s)?(?:.*?)(?:should|could|might|may)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        signals = []
        for pattern in signal_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 15:  # Minimum length to be meaningful
                    signals.append(match.strip())
        
        # Format signals
        formatted_signals = []
        for i, signal in enumerate(signals[:3]):  # Limit to top 3
            # Determine signal type
            signal_type = "neutral"  # Default
            if any(word in signal.lower() for word in ["buy", "long", "bullish", "positive", "upside", "increase", "rise"]):
                signal_type = "bullish"
            elif any(word in signal.lower() for word in ["sell", "short", "bearish", "negative", "downside", "decrease", "fall"]):
                signal_type = "bearish"
            
            # Determine timeframe
            timeframe = "medium-term"  # Default
            if any(word in signal.lower() for word in ["immediate", "short-term", "near-term", "day", "intraday"]):
                timeframe = "short-term"
            elif any(word in signal.lower() for word in ["long-term", "longer-term", "extended", "sustained"]):
                timeframe = "long-term"
            
            # Determine strength
            strength = "moderate"  # Default
            if any(word in signal.lower() for word in ["strong", "significant", "substantial", "considerable", "clear", "definite"]):
                strength = "strong"
            elif any(word in signal.lower() for word in ["weak", "slight", "minor", "small", "limited", "potential"]):
                strength = "weak"
            
            formatted_signals.append({
                "description": signal,
                "type": signal_type,
                "timeframe": timeframe,
                "strength": strength,
                "confidence": reasoning.get("confidence", 0.5) * (1.0 - (i * 0.1))  # Decreasing confidence
            })
        
        return formatted_signals
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a text snippet.
        
        Parameters:
        -----------
        text : str
            Text to analyze.
            
        Returns:
        --------
        float
            Sentiment score (-1.0 to 1.0).
        """
        positive_words = [
            "positive", "bullish", "uptrend", "growth", "increase", "higher", "strong", "opportunity", "outperform",
            "beat", "exceed", "success", "successful", "gain", "improve", "improved", "improving", "improvement",
            "advantage", "advantageous", "profit", "profitable", "win", "winning", "winner", "good", "great",
            "excellent", "outstanding", "remarkable", "impressive", "favorable", "optimistic", "confident"
        ]
        
        negative_words = [
            "negative", "bearish", "downtrend", "decline", "decrease", "lower", "weak", "risk", "underperform",
            "miss", "missed", "failure", "fail", "failed", "failing", "loss", "lose", "losing", "loser",
            "disadvantage", "disadvantageous", "unprofitable", "bad", "poor", "terrible", "disappointing",
            "unfavorable", "pessimistic", "concerned", "concerning", "worry", "worried", "worrying"
        ]
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in positive_words if word in words)
        negative_count = sum(1 for word in negative_words if word in words)
        
        # Check for negations
        negations = ["not", "no", "never", "neither", "nor", "none", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't"]
        for i, word in enumerate(words[:-1]):
            if word in negations:
                if words[i+1] in positive_words:
                    positive_count -= 1
                    negative_count += 1
                elif words[i+1] in negative_words:
                    negative_count -= 1
                    positive_count += 1
        
        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count
    
    def _sentiment_score_to_label(self, score: float) -> str:
        """
        Convert sentiment score to label.
        
        Parameters:
        -----------
        score : float
            Sentiment score (-1.0 to 1.0).
            
        Returns:
        --------
        str
            Sentiment label.
        """
        if score >= 0.7:
            return "very positive"
        elif score >= 0.3:
            return "positive"
        elif score >= 0.1:
            return "slightly positive"
        elif score <= -0.7:
            return "very negative"
        elif score <= -0.3:
            return "negative"
        elif score <= -0.1:
            return "slightly negative"
        else:
            return "neutral"


class NaturalLanguageMarketAnalysis:
    """
    Main class for natural language market analysis.
    
    This class coordinates the various natural language analysis components
    and provides a unified interface for analyzing market-related text data.
    """
    
    def __init__(self, gemma_core: GemmaCore = None):
        """
        Initialize the NaturalLanguageMarketAnalysis.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.NaturalLanguageMarketAnalysis")
        
        # Initialize GemmaCore if not provided
        if gemma_core is None:
            from gemma3_integration.architecture import GemmaCore
            gemma_core = GemmaCore()
        
        self.gemma_core = gemma_core
        
        # Initialize analyzers
        self.news_analyzer = NewsAnalyzer(gemma_core)
        self.earnings_analyzer = EarningsAnalyzer(gemma_core)
        self.social_media_analyzer = SocialMediaAnalyzer(gemma_core)
        
        self.logger.info("Initialized NaturalLanguageMarketAnalysis")
    
    def analyze_ticker(self, ticker: str) -> Dict:
        """
        Perform comprehensive natural language analysis for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
            
        Returns:
        --------
        Dict
            Comprehensive analysis results.
        """
        self.logger.info(f"Performing comprehensive analysis for {ticker}")
        
        try:
            # Analyze news
            news_analysis = self.news_analyzer.analyze_news_articles(ticker)
            
            # Analyze social sentiment
            social_analysis = self.social_media_analyzer.analyze_social_sentiment(ticker)
            
            # Fetch earnings data (placeholder)
            # In a real implementation, this would fetch actual earnings data
            earnings_data = {
                "report_date": datetime.datetime.now().isoformat(),
                "revenue": 1250000000,
                "revenue_estimate": 1200000000,
                "eps": 1.25,
                "eps_estimate": 1.15,
                "net_income": 350000000,
                "prior_revenue": 1150000000,
                "prior_eps": 1.10,
                "guidance": {
                    "revenue": {
                        "low": 1280000000,
                        "high": 1320000000
                    },
                    "eps": {
                        "low": 1.30,
                        "high": 1.40
                    }
                }
            }
            
            # Analyze earnings
            earnings_analysis = self.earnings_analyzer.analyze_earnings_report(ticker, earnings_data)
            
            # Integrate analyses
            integrated_analysis = self._integrate_analyses(ticker, news_analysis, social_analysis, earnings_analysis)
            
            return integrated_analysis
        except Exception as e:
            self.logger.error(f"Failed to perform comprehensive analysis: {e}")
            return {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _integrate_analyses(self, ticker: str, news_analysis: Dict, social_analysis: Dict, earnings_analysis: Dict) -> Dict:
        """
        Integrate various analyses into a comprehensive result.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        news_analysis : Dict
            News analysis results.
        social_analysis : Dict
            Social sentiment analysis results.
        earnings_analysis : Dict
            Earnings analysis results.
            
        Returns:
        --------
        Dict
            Integrated analysis results.
        """
        # Generate prompt for integration
        prompt = self.gemma_core.prompt_engine.generate_prompt(
            "integrate_analyses",
            ticker=ticker,
            news_analysis=json.dumps(news_analysis, indent=2),
            social_analysis=json.dumps(social_analysis, indent=2),
            earnings_analysis=json.dumps(earnings_analysis, indent=2)
        ) if "integrate_analyses" in self.gemma_core.prompt_engine.templates else self.gemma_core.prompt_engine._generate_default_prompt(
            task="Integrate the following analyses",
            ticker=ticker,
            news_analysis=json.dumps(news_analysis, indent=2),
            social_analysis=json.dumps(social_analysis, indent=2),
            earnings_analysis=json.dumps(earnings_analysis, indent=2)
        )
        
        # Generate reasoning using Gemma 3
        reasoning = self.gemma_core.cot_processor.generate_reasoning(prompt)
        
        # Calculate overall sentiment
        news_sentiment = news_analysis.get("sentiment_score", 0.0)
        social_sentiment = social_analysis.get("sentiment_score", 0.0)
        earnings_sentiment = 0.0
        
        # Extract earnings sentiment
        if "overall_assessment" in earnings_analysis and "sentiment" in earnings_analysis["overall_assessment"]:
            earnings_sentiment_label = earnings_analysis["overall_assessment"]["sentiment"]
            earnings_sentiment = self._sentiment_label_to_score(earnings_sentiment_label)
        
        # Weight the sentiments (earnings > news > social)
        if "error" not in earnings_analysis:
            overall_sentiment = 0.5 * earnings_sentiment + 0.3 * news_sentiment + 0.2 * social_sentiment
        else:
            overall_sentiment = 0.6 * news_sentiment + 0.4 * social_sentiment
        
        # Prepare integrated result
        integrated_result = {
            "ticker": ticker,
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_sentiment": self._sentiment_score_to_label(overall_sentiment),
            "sentiment_score": overall_sentiment,
            "key_insights": self._extract_key_insights(reasoning),
            "market_narrative": self._extract_market_narrative(reasoning),
            "trading_implications": self._extract_trading_implications(reasoning),
            "risk_factors": self._extract_risk_factors(reasoning),
            "source_analyses": {
                "news": news_analysis,
                "social": social_analysis,
                "earnings": earnings_analysis
            },
            "reasoning": reasoning
        }
        
        return integrated_result
    
    def _extract_key_insights(self, reasoning: Dict) -> List[str]:
        """
        Extract key insights from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[str]
            Key insights.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for insight patterns
        insight_patterns = [
            r"(?:key|main|important|critical|significant)(?:.*?)insight(?:s)?(?:.*?)(?:include|are|is)(?:.*?)(.*?)(?:\.|$)",
            r"insight(?:s)?(?:.*?)(?:include|are|is)(?:.*?)(.*?)(?:\.|$)",
            r"(?:key|main|important|critical|significant)(?:.*?)(?:finding|takeaway|point|observation)(?:s)?(?:.*?)(?:include|are|is)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        insights = []
        for pattern in insight_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 15:  # Minimum length to be meaningful
                    insights.append(match.strip())
        
        # If no insights found using patterns, extract from reasoning steps
        if not insights and reasoning.get("steps"):
            for step in reasoning.get("steps"):
                if len(step) > 30 and ("suggest" in step.lower() or "indicate" in step.lower() or "show" in step.lower() or "reveal" in step.lower()):
                    insights.append(step.strip())
        
        # Limit to top 5 insights and remove duplicates
        unique_insights = []
        for insight in insights:
            # Check if this insight is substantially different from those already included
            if not any(self._text_similarity(insight, existing) > 0.7 for existing in unique_insights):
                unique_insights.append(insight)
                if len(unique_insights) >= 5:
                    break
        
        return unique_insights
    
    def _extract_market_narrative(self, reasoning: Dict) -> str:
        """
        Extract market narrative from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        str
            Market narrative.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for narrative patterns
        narrative_patterns = [
            r"(?:market|overall)(?:.*?)narrative(?:.*?)(?:is|suggests|indicates)(?:.*?)(.*?)(?:\.|$)",
            r"narrative(?:.*?)(?:is|suggests|indicates)(?:.*?)(.*?)(?:\.|$)",
            r"(?:overall|market|combined)(?:.*?)(?:picture|story|assessment)(?:.*?)(?:is|suggests|indicates)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        for pattern in narrative_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # If no narrative found using patterns, use the conclusion
        if reasoning.get("conclusion"):
            return reasoning.get("conclusion")
        
        # Default narrative
        return "No clear market narrative could be extracted from the analysis."
    
    def _extract_trading_implications(self, reasoning: Dict) -> List[Dict]:
        """
        Extract trading implications from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            Trading implications.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for implication patterns
        implication_patterns = [
            r"(?:trading|investment)(?:.*?)implication(?:s)?(?:.*?)(?:include|are|is)(?:.*?)(.*?)(?:\.|$)",
            r"implication(?:s)?(?:.*?)(?:for|on)(?:.*?)(?:trading|investment|investors|traders)(?:.*?)(?:include|are|is)(?:.*?)(.*?)(?:\.|$)",
            r"(?:based on|given)(?:.*?)(?:analysis|sentiment|insights)(?:.*?)(?:investors|traders)(?:.*?)(?:should|could|might|may)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        implications = []
        for pattern in implication_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 15:  # Minimum length to be meaningful
                    implications.append(match.strip())
        
        # Format implications
        formatted_implications = []
        for i, implication in enumerate(implications[:3]):  # Limit to top 3
            # Determine implication type
            implication_type = "neutral"  # Default
            if any(word in implication.lower() for word in ["buy", "long", "bullish", "positive", "upside", "increase", "rise"]):
                implication_type = "bullish"
            elif any(word in implication.lower() for word in ["sell", "short", "bearish", "negative", "downside", "decrease", "fall"]):
                implication_type = "bearish"
            
            # Determine timeframe
            timeframe = "medium-term"  # Default
            if any(word in implication.lower() for word in ["immediate", "short-term", "near-term", "day", "intraday"]):
                timeframe = "short-term"
            elif any(word in implication.lower() for word in ["long-term", "longer-term", "extended", "sustained"]):
                timeframe = "long-term"
            
            # Determine confidence
            confidence = reasoning.get("confidence", 0.5) * (1.0 - (i * 0.1))  # Decreasing confidence
            
            formatted_implications.append({
                "description": implication,
                "type": implication_type,
                "timeframe": timeframe,
                "confidence": confidence
            })
        
        return formatted_implications
    
    def _extract_risk_factors(self, reasoning: Dict) -> List[Dict]:
        """
        Extract risk factors from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            Risk factors.
        """
        # Combine all reasoning steps and conclusion
        all_text = " ".join(reasoning.get("steps", [])) + " " + reasoning.get("conclusion", "")
        
        # Look for risk patterns
        risk_patterns = [
            r"(?:key|main|important|critical|significant)(?:.*?)risk(?:s)?(?:.*?)(?:include|are|is)(?:.*?)(.*?)(?:\.|$)",
            r"risk(?:s)?(?:.*?)(?:include|are|is)(?:.*?)(.*?)(?:\.|$)",
            r"(?:key|main|important|critical|significant)(?:.*?)(?:concern|challenge|headwind|issue)(?:s)?(?:.*?)(?:include|are|is)(?:.*?)(.*?)(?:\.|$)"
        ]
        
        risks = []
        for pattern in risk_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 15:  # Minimum length to be meaningful
                    risks.append(match.strip())
        
        # Format risks
        formatted_risks = []
        for i, risk in enumerate(risks[:3]):  # Limit to top 3
            # Determine risk severity
            severity = "moderate"  # Default
            if any(word in risk.lower() for word in ["significant", "substantial", "major", "critical", "severe", "high"]):
                severity = "high"
            elif any(word in risk.lower() for word in ["minor", "slight", "small", "limited", "low"]):
                severity = "low"
            
            # Determine risk likelihood
            likelihood = "moderate"  # Default
            if any(word in risk.lower() for word in ["likely", "probable", "expected", "anticipated", "high probability"]):
                likelihood = "high"
            elif any(word in risk.lower() for word in ["unlikely", "improbable", "unexpected", "low probability"]):
                likelihood = "low"
            
            formatted_risks.append({
                "description": risk,
                "severity": severity,
                "likelihood": likelihood,
                "importance": 1.0 - (i * 0.2)  # Decreasing importance
            })
        
        return formatted_risks
    
    def _sentiment_score_to_label(self, score: float) -> str:
        """
        Convert sentiment score to label.
        
        Parameters:
        -----------
        score : float
            Sentiment score (-1.0 to 1.0).
            
        Returns:
        --------
        str
            Sentiment label.
        """
        if score >= 0.7:
            return "very positive"
        elif score >= 0.3:
            return "positive"
        elif score >= 0.1:
            return "slightly positive"
        elif score <= -0.7:
            return "very negative"
        elif score <= -0.3:
            return "negative"
        elif score <= -0.1:
            return "slightly negative"
        else:
            return "neutral"
    
    def _sentiment_label_to_score(self, label: str) -> float:
        """
        Convert sentiment label to score.
        
        Parameters:
        -----------
        label : str
            Sentiment label.
            
        Returns:
        --------
        float
            Sentiment score (-1.0 to 1.0).
        """
        label_map = {
            "very positive": 0.8,
            "positive": 0.5,
            "slightly positive": 0.2,
            "neutral": 0.0,
            "slightly negative": -0.2,
            "negative": -0.5,
            "very negative": -0.8
        }
        
        return label_map.get(label.lower(), 0.0)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text snippets.
        
        Parameters:
        -----------
        text1 : str
            First text.
        text2 : str
            Second text.
            
        Returns:
        --------
        float
            Similarity score (0.0 to 1.0).
        """
        # Simple word overlap similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
