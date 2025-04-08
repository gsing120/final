import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import re
import threading
import time
import json
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContinuousResearch")

class ContinuousResearchEngine:
    """
    Engine for continuously researching market news and information
    to inform trading decisions.
    """
    
    def __init__(self, data_dir="research_data"):
        """
        Initialize the continuous research engine.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store research data
        """
        self.data_dir = data_dir
        self.active = False
        self.thread = None
        self.stop_event = threading.Event()
        self.research_interval = 3600  # Default: research every hour
        self.watched_tickers = []
        self.research_results = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize sentiment analyzer
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        logger.info("ContinuousResearchEngine initialized")
    
    def start(self, tickers=None, interval=3600):
        """
        Start continuous research.
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols to research
        interval : int, optional
            Research interval in seconds
        """
        if self.active:
            logger.warning("Continuous research is already active")
            return False
        
        self.active = True
        self.research_interval = interval
        
        if tickers:
            self.watched_tickers = tickers
        
        # Create a new stop event
        self.stop_event = threading.Event()
        
        # Start research thread
        self.thread = threading.Thread(target=self._research_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started continuous research for tickers: {self.watched_tickers}")
        return True
    
    def stop(self):
        """
        Stop continuous research.
        """
        if not self.active:
            logger.warning("Continuous research is not active")
            return False
        
        self.active = False
        self.stop_event.set()
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Stopped continuous research")
        return True
    
    def add_tickers(self, tickers):
        """
        Add tickers to watch list.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols to add
        """
        for ticker in tickers:
            if ticker not in self.watched_tickers:
                self.watched_tickers.append(ticker)
        
        logger.info(f"Added tickers to watch list: {tickers}")
        return True
    
    def remove_tickers(self, tickers):
        """
        Remove tickers from watch list.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols to remove
        """
        for ticker in tickers:
            if ticker in self.watched_tickers:
                self.watched_tickers.remove(ticker)
        
        logger.info(f"Removed tickers from watch list: {tickers}")
        return True
    
    def get_status(self):
        """
        Get current status of continuous research.
        
        Returns:
        --------
        dict
            Status information
        """
        return {
            'active': self.active,
            'watched_tickers': self.watched_tickers,
            'research_interval': self.research_interval,
            'last_update': self.research_results.get('last_update', None),
            'research_count': len(self.research_results.get('news', [])) if 'news' in self.research_results else 0
        }
    
    def get_latest_research(self, ticker=None):
        """
        Get latest research results.
        
        Parameters:
        -----------
        ticker : str, optional
            Ticker symbol to get research for
            
        Returns:
        --------
        dict
            Research results
        """
        if ticker:
            # Filter research results for specific ticker
            if 'news' in self.research_results:
                ticker_news = [n for n in self.research_results['news'] if ticker.upper() in n.get('tickers', [])]
                return {
                    'ticker': ticker,
                    'news': ticker_news,
                    'sentiment': self._calculate_ticker_sentiment(ticker_news),
                    'last_update': self.research_results.get('last_update', None)
                }
            return {
                'ticker': ticker,
                'news': [],
                'sentiment': 0,
                'last_update': None
            }
        
        return self.research_results
    
    def _research_loop(self):
        """
        Main research loop that runs in a separate thread.
        """
        logger.info("Starting research loop")
        
        while not self.stop_event.is_set():
            try:
                # Perform research for all watched tickers
                self._perform_research()
                
                # Save research results
                self._save_research_results()
                
                # Wait for next research interval or until stopped
                self.stop_event.wait(self.research_interval)
            
            except Exception as e:
                logger.error(f"Error in research loop: {e}")
                # Wait a bit before retrying
                self.stop_event.wait(60)
    
    def _perform_research(self):
        """
        Perform research for all watched tickers.
        """
        logger.info(f"Performing research for tickers: {self.watched_tickers}")
        
        # Initialize results if needed
        if not self.research_results:
            self.research_results = {
                'news': [],
                'market_analysis': {},
                'sentiment': {},
                'last_update': None
            }
        
        # Get news for each ticker
        all_news = []
        for ticker in self.watched_tickers:
            try:
                ticker_news = self._get_ticker_news(ticker)
                all_news.extend(ticker_news)
                
                # Get market analysis
                market_analysis = self._get_market_analysis(ticker)
                self.research_results['market_analysis'][ticker] = market_analysis
                
                # Calculate sentiment
                sentiment = self._calculate_ticker_sentiment(ticker_news)
                self.research_results['sentiment'][ticker] = sentiment
                
                logger.info(f"Completed research for {ticker}: {len(ticker_news)} news items")
            
            except Exception as e:
                logger.error(f"Error researching {ticker}: {e}")
        
        # Update news and timestamp
        self.research_results['news'] = all_news
        self.research_results['last_update'] = datetime.now().isoformat()
        
        logger.info(f"Research completed: {len(all_news)} total news items")
    
    def _get_ticker_news(self, ticker):
        """
        Get news for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        list
            List of news items
        """
        news = []
        
        try:
            # Use a financial news API or website
            # For demonstration, we'll use a simple approach
            
            # Example: Get news from Yahoo Finance
            url = f"https://finance.yahoo.com/quote/{ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find news items
                news_items = soup.find_all('li', {'class': 'js-stream-content'})
                
                for item in news_items:
                    try:
                        # Extract news details
                        headline_elem = item.find('h3')
                        link_elem = item.find('a')
                        summary_elem = item.find('p')
                        
                        if headline_elem and link_elem:
                            headline = headline_elem.text.strip()
                            link = link_elem.get('href')
                            if link and not link.startswith('http'):
                                link = f"https://finance.yahoo.com{link}"
                            
                            summary = summary_elem.text.strip() if summary_elem else ""
                            
                            # Calculate sentiment
                            sentiment = self.sentiment_analyzer.polarity_scores(headline + " " + summary)
                            
                            news.append({
                                'ticker': ticker,
                                'tickers': [ticker],
                                'headline': headline,
                                'summary': summary,
                                'url': link,
                                'source': 'Yahoo Finance',
                                'date': datetime.now().isoformat(),
                                'sentiment': sentiment
                            })
                    except Exception as e:
                        logger.error(f"Error parsing news item: {e}")
            
            # Example: Get news from another source
            # Add more news sources as needed
            
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {e}")
        
        return news
    
    def _get_market_analysis(self, ticker):
        """
        Get market analysis for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        dict
            Market analysis
        """
        try:
            # For demonstration, return a simple analysis
            # In a real implementation, this would use more sophisticated analysis
            return {
                'ticker': ticker,
                'analysis_date': datetime.now().isoformat(),
                'market_cap_category': 'Unknown',
                'sector_performance': 'Unknown',
                'relative_strength': 'Unknown',
                'volatility': 'Unknown',
                'trading_volume': 'Unknown'
            }
        
        except Exception as e:
            logger.error(f"Error getting market analysis for {ticker}: {e}")
            return {}
    
    def _calculate_ticker_sentiment(self, news_items):
        """
        Calculate overall sentiment for a ticker based on news.
        
        Parameters:
        -----------
        news_items : list
            List of news items
            
        Returns:
        --------
        float
            Sentiment score (-1 to 1)
        """
        if not news_items:
            return 0
        
        # Calculate weighted average of compound sentiment
        total_sentiment = 0
        for item in news_items:
            if 'sentiment' in item and 'compound' in item['sentiment']:
                total_sentiment += item['sentiment']['compound']
        
        return total_sentiment / len(news_items) if news_items else 0
    
    def _save_research_results(self):
        """
        Save research results to disk.
        """
        try:
            # Create a filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"research_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(self.research_results, f, indent=2)
            
            logger.info(f"Saved research results to {filepath}")
            
            # Also save latest results
            latest_filepath = os.path.join(self.data_dir, "latest_research.json")
            with open(latest_filepath, 'w') as f:
                json.dump(self.research_results, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving research results: {e}")

# Create a global instance for use in the application
continuous_research_engine = ContinuousResearchEngine()
