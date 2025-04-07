"""
Unified Data Access Layer for Gemma Advanced Trading System.

This module provides a unified interface for accessing market data from various sources.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger("GemmaTrading.DataAccess")

class MarketDataProvider:
    """
    Unified interface for accessing market data from various sources.
    
    This class provides a consistent interface for fetching market data
    regardless of the underlying data source (Alpha Vantage, Moomoo, Yahoo Finance, etc.)
    """
    
    def __init__(self, api_key=None, data_source="synthetic"):
        """
        Initialize the market data provider.
        
        Parameters:
        -----------
        api_key : str, optional
            API key for the data source
        data_source : str, optional
            Data source to use ("alpha_vantage", "moomoo", "yahoo", "synthetic")
        """
        self.api_key = api_key
        self.data_source = data_source
        self.logger = logging.getLogger("GemmaTrading.DataAccess")
        self.logger.info(f"Initialized MarketDataProvider with source: {data_source}")
        
        # In a real implementation, we would initialize the appropriate client here
        # based on the data_source parameter
    
    def fetch_historical_data(self, ticker, start_date=None, end_date=None, period="1y", interval="1d"):
        """
        Fetch historical market data for the specified ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to fetch data for
        start_date : str or datetime, optional
            Start date in YYYY-MM-DD format or as datetime object
        end_date : str or datetime, optional
            End date in YYYY-MM-DD format or as datetime object
        period : str, optional
            Period to fetch data for (e.g., "1d", "1mo", "1y")
        interval : str, optional
            Data interval (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
        --------
        pandas.DataFrame
            Market data with columns: Open, High, Low, Close, Volume, Adj Close
        """
        self.logger.info(f"Fetching historical data for {ticker} from {self.data_source}")
        
        if self.data_source == "synthetic":
            return self._generate_synthetic_data(ticker, start_date, end_date, period, interval)
        elif self.data_source == "alpha_vantage":
            # In a real implementation, we would call the Alpha Vantage API here
            self.logger.warning("Alpha Vantage API not implemented yet, using synthetic data")
            return self._generate_synthetic_data(ticker, start_date, end_date, period, interval)
        elif self.data_source == "moomoo":
            # In a real implementation, we would call the Moomoo API here
            self.logger.warning("Moomoo API not implemented yet, using synthetic data")
            return self._generate_synthetic_data(ticker, start_date, end_date, period, interval)
        elif self.data_source == "yahoo":
            # In a real implementation, we would use yfinance here
            self.logger.warning("Yahoo Finance API not implemented yet, using synthetic data")
            return self._generate_synthetic_data(ticker, start_date, end_date, period, interval)
        else:
            self.logger.error(f"Unknown data source: {self.data_source}")
            raise ValueError(f"Unknown data source: {self.data_source}")
    
    def fetch_real_time_data(self, ticker):
        """
        Fetch real-time market data for the specified ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to fetch data for
            
        Returns:
        --------
        dict
            Real-time market data
        """
        self.logger.info(f"Fetching real-time data for {ticker} from {self.data_source}")
        
        if self.data_source == "synthetic":
            # Generate a single data point with current timestamp
            now = datetime.now()
            last_close = 100 + np.random.normal(0, 1)
            return {
                'ticker': ticker,
                'timestamp': now,
                'last_price': last_close,
                'bid': last_close - 0.01,
                'ask': last_close + 0.01,
                'volume': int(np.random.randint(1000, 10000)),
                'change': np.random.normal(0, 0.5),
                'change_percent': np.random.normal(0, 0.5)
            }
        else:
            self.logger.warning(f"Real-time data not implemented for {self.data_source}, using synthetic data")
            now = datetime.now()
            last_close = 100 + np.random.normal(0, 1)
            return {
                'ticker': ticker,
                'timestamp': now,
                'last_price': last_close,
                'bid': last_close - 0.01,
                'ask': last_close + 0.01,
                'volume': int(np.random.randint(1000, 10000)),
                'change': np.random.normal(0, 0.5),
                'change_percent': np.random.normal(0, 0.5)
            }
    
    def fetch_company_info(self, ticker):
        """
        Fetch company information for the specified ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to fetch data for
            
        Returns:
        --------
        dict
            Company information
        """
        self.logger.info(f"Fetching company info for {ticker} from {self.data_source}")
        
        # For demonstration purposes, return synthetic company info
        return {
            'ticker': ticker,
            'name': f"{ticker} Corporation",
            'sector': "Technology",
            'industry': "Software",
            'market_cap': 1000000000,
            'pe_ratio': 20.5,
            'dividend_yield': 1.5,
            'beta': 1.2,
            'description': f"This is a synthetic description for {ticker} Corporation."
        }
    
    def fetch_market_news(self, ticker=None, limit=10):
        """
        Fetch market news for the specified ticker or general market news.
        
        Parameters:
        -----------
        ticker : str, optional
            The ticker symbol to fetch news for, or None for general market news
        limit : int, optional
            Maximum number of news items to return
            
        Returns:
        --------
        list of dict
            News items
        """
        self.logger.info(f"Fetching market news for {ticker if ticker else 'general market'} from {self.data_source}")
        
        # For demonstration purposes, return synthetic news
        news = []
        for i in range(limit):
            news.append({
                'title': f"{'Ticker-specific' if ticker else 'Market'} News Item {i+1}",
                'date': datetime.now() - timedelta(hours=i),
                'source': "Synthetic News Source",
                'url': "https://example.com/news",
                'summary': f"This is a synthetic news item {'about ' + ticker if ticker else 'about the market'}."
            })
        
        return news
    
    def search_tickers(self, query):
        """
        Search for tickers matching the specified query.
        
        Parameters:
        -----------
        query : str
            Search query
            
        Returns:
        --------
        list of dict
            Matching tickers
        """
        self.logger.info(f"Searching for tickers matching '{query}' from {self.data_source}")
        
        # For demonstration purposes, return synthetic search results
        results = []
        for i in range(5):
            results.append({
                'ticker': f"{query.upper()}{i}",
                'name': f"{query.capitalize()} Corporation {i}",
                'exchange': "NYSE" if i % 2 == 0 else "NASDAQ"
            })
        
        return results
    
    def _generate_synthetic_data(self, ticker, start_date=None, end_date=None, period="1y", interval="1d"):
        """
        Generate synthetic market data for demonstration purposes.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to generate data for
        start_date : str or datetime, optional
            Start date in YYYY-MM-DD format or as datetime object
        end_date : str or datetime, optional
            End date in YYYY-MM-DD format or as datetime object
        period : str, optional
            Period to generate data for (e.g., "1d", "1mo", "1y")
        interval : str, optional
            Data interval (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
        --------
        pandas.DataFrame
            Synthetic market data
        """
        self.logger.info(f"Generating synthetic data for {ticker}")
        
        # Parse dates or use defaults
        if not end_date:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        if not start_date:
            if period == "1d":
                start_date = end_date - timedelta(days=1)
            elif period == "1w":
                start_date = end_date - timedelta(days=7)
            elif period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                start_date = end_date - timedelta(days=730)
            elif period == "5y":
                start_date = end_date - timedelta(days=1825)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Determine frequency based on interval
        if interval == "1m":
            freq = "1min"
            # For minute data, only generate for market hours of the most recent day
            if (end_date - start_date).days > 1:
                start_date = end_date.replace(hour=9, minute=30, second=0)
                end_date = end_date.replace(hour=16, minute=0, second=0)
        elif interval == "5m":
            freq = "5min"
            # For minute data, only generate for market hours of the most recent day
            if (end_date - start_date).days > 1:
                start_date = end_date.replace(hour=9, minute=30, second=0)
                end_date = end_date.replace(hour=16, minute=0, second=0)
        elif interval == "15m":
            freq = "15min"
            # For minute data, only generate for market hours of the most recent day
            if (end_date - start_date).days > 1:
                start_date = end_date.replace(hour=9, minute=30, second=0)
                end_date = end_date.replace(hour=16, minute=0, second=0)
        elif interval == "30m":
            freq = "30min"
            # For minute data, only generate for market hours of the most recent day
            if (end_date - start_date).days > 1:
                start_date = end_date.replace(hour=9, minute=30, second=0)
                end_date = end_date.replace(hour=16, minute=0, second=0)
        elif interval == "1h":
            freq = "1H"
            # For hourly data, only generate for market hours
            if (end_date - start_date).days > 1:
                business_days = pd.date_range(start=start_date, end=end_date, freq="B")
                date_range = []
                for day in business_days:
                    for hour in range(9, 17):
                        date_range.append(day.replace(hour=hour, minute=0, second=0))
                date_range = pd.DatetimeIndex(date_range)
            else:
                date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        elif interval == "1d":
            freq = "B"  # Business day frequency
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        elif interval == "1wk":
            freq = "W-FRI"  # Weekly frequency, Friday is the last business day of the week
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        elif interval == "1mo":
            freq = "BM"  # Business month end frequency
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        else:
            freq = "B"  # Default to business day frequency
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate date range if not already created
        if 'date_range' not in locals():
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate synthetic price data with a trend and some volatility
        base_price = 100.0
        trend = np.linspace(0, 20, len(date_range))
        volatility = np.random.normal(0, 1, len(date_range))
        
        # Create price series with some seasonality and momentum
        close_prices = base_price + trend + volatility * 5 + np.sin(np.linspace(0, 10, len(date_range))) * 10
        
        # Create high, low, open prices based on close
        high_prices = close_prices + np.random.uniform(0.5, 2.0, len(date_range))
        low_prices = close_prices - np.random.uniform(0.5, 2.0, len(date_range))
        open_prices = low_prices + np.random.uniform(0, 1, len(date_range)) * (high_prices - low_prices)
        
        # Generate volume data
        volume = np.random.randint(100000, 1000000, len(date_range))
        
        # Create DataFrame
        data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Adj Close': close_prices,
            'Volume': volume
        }, index=date_range)
        
        return data


# Create a singleton instance for easy access
market_data = MarketDataProvider()

def get_provider(api_key=None, data_source="synthetic"):
    """
    Get a configured MarketDataProvider instance.
    
    Parameters:
    -----------
    api_key : str, optional
        API key for the data source
    data_source : str, optional
        Data source to use ("alpha_vantage", "moomoo", "yahoo", "synthetic")
        
    Returns:
    --------
    MarketDataProvider
        Configured market data provider
    """
    return MarketDataProvider(api_key, data_source)

def fetch_historical_data(ticker, start_date=None, end_date=None, period="1y", interval="1d"):
    """
    Fetch historical market data for the specified ticker using the default provider.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to fetch data for
    start_date : str or datetime, optional
        Start date in YYYY-MM-DD format or as datetime object
    end_date : str or datetime, optional
        End date in YYYY-MM-DD format or as datetime object
    period : str, optional
        Period to fetch data for (e.g., "1d", "1mo", "1y")
    interval : str, optional
        Data interval (e.g., "1m", "5m", "1h", "1d")
        
    Returns:
    --------
    pandas.DataFrame
        Market data with columns: Open, High, Low, Close, Volume, Adj Close
    """
    return market_data.fetch_historical_data(ticker, start_date, end_date, period, interval)

def fetch_real_time_data(ticker):
    """
    Fetch real-time market data for the specified ticker using the default provider.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to fetch data for
        
    Returns:
    --------
    dict
        Real-time market data
    """
    return market_data.fetch_real_time_data(ticker)

def fetch_company_info(ticker):
    """
    Fetch company information for the specified ticker using the default provider.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to fetch data for
        
    Returns:
    --------
    dict
        Company information
    """
    return market_data.fetch_company_info(ticker)

def fetch_market_news(ticker=None, limit=10):
    """
    Fetch market news for the specified ticker or general market news using the default provider.
    
    Parameters:
    -----------
    ticker : str, optional
        The ticker symbol to fetch news for, or None for general market news
    limit : int, optional
        Maximum number of news items to return
        
    Returns:
    --------
    list of dict
        News items
    """
    return market_data.fetch_market_news(ticker, limit)

def search_tickers(query):
    """
    Search for tickers matching the specified query using the default provider.
    
    Parameters:
    -----------
    query : str
        Search query
        
    Returns:
    --------
    list of dict
        Matching tickers
    """
    return market_data.search_tickers(query)
