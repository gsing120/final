"""
Data Access Module for Gemma Advanced Trading System.

This module provides functions for accessing market data from various sources.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gemma_trading.log')
    ]
)

logger = logging.getLogger("GemmaTrading.DataAccess")

def get_market_data(ticker, period="1y", interval="1d", proxy=None):
    """
    Get market data for a ticker from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to get data for
    period : str, optional
        Period to fetch data for (e.g., "1d", "1mo", "1y")
    interval : str, optional
        Data interval (e.g., "1m", "5m", "1h", "1d")
    proxy : str, optional
        Proxy server URL
        
    Returns:
    --------
    pandas.DataFrame
        Market data with columns: Open, High, Low, Close, Volume, etc.
    """
    logger.info(f"Getting market data for {ticker} with period={period}, interval={interval}")
    
    try:
        # Get data from Yahoo Finance
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            proxy=proxy,
            progress=False
        )
        
        # Check if data is empty
        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Rename columns
        data.columns = [col if col != 'Date' else 'date' for col in data.columns]
        data.columns = [col if col != 'Adj Close' else 'adj_close' for col in data.columns]
        data.columns = [col.lower() for col in data.columns]
        
        logger.info(f"Got {len(data)} rows of data for {ticker}")
        
        return data
    
    except Exception as e:
        logger.exception(f"Error getting market data for {ticker}: {e}")
        return None

def get_ticker_info(ticker):
    """
    Get information about a ticker from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to get info for
        
    Returns:
    --------
    dict
        Ticker information
    """
    logger.info(f"Getting ticker info for {ticker}")
    
    try:
        # Get ticker info from Yahoo Finance
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        logger.info(f"Got info for {ticker}")
        
        return info
    
    except Exception as e:
        logger.exception(f"Error getting ticker info for {ticker}: {e}")
        return None

def get_market_news(ticker, limit=10):
    """
    Get news for a ticker.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to get news for
    limit : int, optional
        Maximum number of news items to return
        
    Returns:
    --------
    list
        List of news items
    """
    logger.info(f"Getting news for {ticker}")
    
    try:
        # Get ticker news from Yahoo Finance
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news
        
        # Limit number of news items
        if limit and len(news) > limit:
            news = news[:limit]
        
        logger.info(f"Got {len(news)} news items for {ticker}")
        
        return news
    
    except Exception as e:
        logger.exception(f"Error getting news for {ticker}: {e}")
        return []

def save_data(data, filename, directory=None):
    """
    Save data to a file.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to save
    filename : str
        Filename to save to
    directory : str, optional
        Directory to save to
        
    Returns:
    --------
    str
        Path to saved file
    """
    logger.info(f"Saving data to {filename}")
    
    try:
        # Create directory if it doesn't exist
        if directory:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
        else:
            filepath = filename
        
        # Save data based on file extension
        if filename.endswith('.csv'):
            data.to_csv(filepath, index=False)
        elif filename.endswith('.json'):
            data.to_json(filepath, orient='records')
        elif filename.endswith('.xlsx'):
            data.to_excel(filepath, index=False)
        else:
            # Default to CSV
            filepath = f"{filepath}.csv"
            data.to_csv(filepath, index=False)
        
        logger.info(f"Saved data to {filepath}")
        
        return filepath
    
    except Exception as e:
        logger.exception(f"Error saving data to {filename}: {e}")
        return None

def load_data(filepath):
    """
    Load data from a file.
    
    Parameters:
    -----------
    filepath : str
        Path to file to load
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    logger.info(f"Loading data from {filepath}")
    
    try:
        # Load data based on file extension
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            data = pd.read_json(filepath)
        elif filepath.endswith('.xlsx'):
            data = pd.read_excel(filepath)
        else:
            logger.error(f"Unsupported file format: {filepath}")
            return None
        
        logger.info(f"Loaded {len(data)} rows from {filepath}")
        
        return data
    
    except Exception as e:
        logger.exception(f"Error loading data from {filepath}: {e}")
        return None
