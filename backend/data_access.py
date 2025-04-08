import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GemmaTrading.DataAccess")

class YahooFinanceClient:
    """Client for accessing Yahoo Finance data."""
    
    def __init__(self):
        """Initialize the Yahoo Finance client."""
        logger.info("YahooFinanceClient initialized")
        
    def get_market_data(self, ticker, interval='1d', period=None, start_date=None, end_date=None):
        """
        Get market data for a specific ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        interval : str, optional
            Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        period : str, optional
            Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        start_date : str, optional
            Start date in format YYYY-MM-DD
        end_date : str, optional
            End date in format YYYY-MM-DD
            
        Returns:
        --------
        pandas.DataFrame
            Market data with columns: open, high, low, close, volume, etc.
        """
        try:
            logger.info(f"Getting market data for {ticker} with interval={interval}")
            
            # Handle period parameter properly
            if period is not None:
                logger.info(f"Using period={period}")
                data = yf.download(ticker, interval=interval, period=period)
            elif start_date is not None and end_date is not None:
                logger.info(f"Using date range: {start_date} to {end_date}")
                data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
            else:
                # Default to 1 year if no period or date range specified
                logger.info("No period or date range specified, using default period=1y")
                data = yf.download(ticker, interval=interval, period='1y')
            
            # Check if data is empty
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
                
            # Convert column names to lowercase
            # Fix for tuple column names issue
            if isinstance(data.columns, pd.MultiIndex):
                # Handle multi-level columns (common in yfinance output)
                data.columns = [col[0].lower() + '_' + col[1].lower() if isinstance(col, tuple) and len(col) > 1 
                               else col[0].lower() if isinstance(col, tuple) 
                               else col.lower() for col in data.columns]
            else:
                # Handle single-level columns
                data.columns = [col.lower() for col in data.columns]
            
            # Ensure all standard columns exist
            standard_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in standard_columns:
                if col not in data.columns and f'adj_{col}' not in data.columns:
                    # Try to find column with similar name
                    matching_cols = [c for c in data.columns if col in c]
                    if matching_cols:
                        data[col] = data[matching_cols[0]]
                    else:
                        # Create placeholder column if necessary
                        if col == 'volume':
                            data[col] = 0
                        else:
                            # For price columns, use close if available
                            if 'close' in data.columns:
                                data[col] = data['close']
                            elif 'adj_close' in data.columns:
                                data[col] = data['adj_close']
                            else:
                                data[col] = 0
            
            # Add additional calculated columns
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {ticker}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_ticker_info(self, ticker):
        """
        Get information about a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        dict
            Ticker information
        """
        try:
            logger.info(f"Getting ticker info for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            return info
        except Exception as e:
            logger.error(f"Error getting ticker info for {ticker}: {str(e)}")
            return None
    
    def get_historical_dividends(self, ticker, period='5y'):
        """
        Get historical dividends for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        period : str, optional
            Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
        --------
        pandas.Series
            Historical dividends
        """
        try:
            logger.info(f"Getting historical dividends for {ticker} with period={period}")
            ticker_obj = yf.Ticker(ticker)
            dividends = ticker_obj.dividends
            
            # Filter by period if specified
            if period != 'max':
                # Convert period to timedelta
                period_map = {
                    '1d': timedelta(days=1),
                    '5d': timedelta(days=5),
                    '1mo': timedelta(days=30),
                    '3mo': timedelta(days=90),
                    '6mo': timedelta(days=180),
                    '1y': timedelta(days=365),
                    '2y': timedelta(days=365*2),
                    '5y': timedelta(days=365*5),
                    '10y': timedelta(days=365*10),
                    'ytd': datetime(datetime.now().year, 1, 1) - datetime.now()
                }
                
                if period in period_map:
                    start_date = datetime.now() - period_map[period]
                    dividends = dividends[dividends.index >= start_date]
            
            return dividends
        except Exception as e:
            logger.error(f"Error getting historical dividends for {ticker}: {str(e)}")
            return None
    
    def get_historical_splits(self, ticker, period='5y'):
        """
        Get historical stock splits for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        period : str, optional
            Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
        --------
        pandas.Series
            Historical stock splits
        """
        try:
            logger.info(f"Getting historical splits for {ticker} with period={period}")
            ticker_obj = yf.Ticker(ticker)
            splits = ticker_obj.splits
            
            # Filter by period if specified
            if period != 'max':
                # Convert period to timedelta
                period_map = {
                    '1d': timedelta(days=1),
                    '5d': timedelta(days=5),
                    '1mo': timedelta(days=30),
                    '3mo': timedelta(days=90),
                    '6mo': timedelta(days=180),
                    '1y': timedelta(days=365),
                    '2y': timedelta(days=365*2),
                    '5y': timedelta(days=365*5),
                    '10y': timedelta(days=365*10),
                    'ytd': datetime(datetime.now().year, 1, 1) - datetime.now()
                }
                
                if period in period_map:
                    start_date = datetime.now() - period_map[period]
                    splits = splits[splits.index >= start_date]
            
            return splits
        except Exception as e:
            logger.error(f"Error getting historical splits for {ticker}: {str(e)}")
            return None
    
    def get_options_chain(self, ticker):
        """
        Get options chain for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        tuple
            (calls DataFrame, puts DataFrame)
        """
        try:
            logger.info(f"Getting options chain for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = ticker_obj.options
            
            if not expirations:
                logger.warning(f"No options available for {ticker}")
                return None
                
            # Get options for the first expiration date
            expiration = expirations[0]
            options = ticker_obj.option_chain(expiration)
            
            return (options.calls, options.puts)
        except Exception as e:
            logger.error(f"Error getting options chain for {ticker}: {str(e)}")
            return None
