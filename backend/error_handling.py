import logging
import traceback
import sys
import os
import json
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("error_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("GemmaAdvanced")

class ErrorHandler:
    """
    Comprehensive error handling system for the Gemma Advanced Trading System.
    Provides decorators, context managers, and utility functions for robust error handling.
    """
    
    @staticmethod
    def handle_exceptions(func):
        """
        Decorator to catch and handle exceptions in functions.
        
        Parameters:
        -----------
        func : function
            The function to wrap with exception handling
            
        Returns:
        --------
        function
            Wrapped function with exception handling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error with traceback
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return a standardized error response
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "function": func.__name__
                }
        return wrapper
    
    @staticmethod
    def api_error_handler(func):
        """
        Decorator specifically for API endpoints to standardize error responses.
        
        Parameters:
        -----------
        func : function
            The API function to wrap with exception handling
            
        Returns:
        --------
        function
            Wrapped function with standardized API error handling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error with traceback
                logger.error(f"API Error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return a standardized API error response
                return {
                    "status": "error",
                    "message": str(e),
                    "error_type": e.__class__.__name__,
                    "endpoint": func.__name__
                }
        return wrapper
    
    @staticmethod
    def data_source_error_handler(func):
        """
        Decorator for data source functions to handle connection issues and data errors.
        
        Parameters:
        -----------
        func : function
            The data source function to wrap with exception handling
            
        Returns:
        --------
        function
            Wrapped function with data source error handling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_retries = kwargs.pop('max_retries', 3)
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except ConnectionError as e:
                    retry_count += 1
                    logger.warning(f"Connection error in {func.__name__}, retry {retry_count}/{max_retries}: {str(e)}")
                    if retry_count >= max_retries:
                        logger.error(f"Max retries reached for {func.__name__}: {str(e)}")
                        logger.error(traceback.format_exc())
                        return None
                except Exception as e:
                    logger.error(f"Data source error in {func.__name__}: {str(e)}")
                    logger.error(traceback.format_exc())
                    return None
        return wrapper
    
    class ErrorContext:
        """
        Context manager for handling errors in a specific context.
        """
        def __init__(self, context_name, fallback_value=None):
            """
            Initialize the error context.
            
            Parameters:
            -----------
            context_name : str
                Name of the context for logging
            fallback_value : any, optional
                Value to return if an error occurs
            """
            self.context_name = context_name
            self.fallback_value = fallback_value
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                # Log the error with traceback
                logger.error(f"Error in context {self.context_name}: {str(exc_val)}")
                logger.error(traceback.format_exc())
                
                # Return True to suppress the exception
                return True
    
    @staticmethod
    def log_error(error, context=None):
        """
        Utility function to log an error.
        
        Parameters:
        -----------
        error : Exception
            The error to log
        context : str, optional
            Additional context information
        """
        if context:
            logger.error(f"Error in {context}: {str(error)}")
        else:
            logger.error(f"Error: {str(error)}")
        
        logger.error(traceback.format_exc())
    
    @staticmethod
    def save_error_report(error, context=None, file_path=None):
        """
        Save a detailed error report to a file.
        
        Parameters:
        -----------
        error : Exception
            The error to report
        context : str, optional
            Additional context information
        file_path : str, optional
            Path to save the error report, defaults to error_reports directory
        """
        if file_path is None:
            # Create error_reports directory if it doesn't exist
            os.makedirs("error_reports", exist_ok=True)
            file_path = f"error_reports/error_{int(time.time())}.json"
        
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            logger.info(f"Error report saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save error report: {str(e)}")

# Create a global instance for use in the application
error_handler = ErrorHandler()

# Example usage:
if __name__ == "__main__":
    # Example with decorator
    @error_handler.handle_exceptions
    def risky_function(x, y):
        return x / y
    
    # Example with context manager
    with error_handler.ErrorContext("division", fallback_value=0):
        result = 1 / 0
    
    # Example with data source error handler
    @error_handler.data_source_error_handler
    def get_market_data(ticker):
        # Simulate a connection error
        raise ConnectionError("Failed to connect to data source")
    
    # Test the examples
    print(risky_function(10, 0))  # Should return an error dict
    print(get_market_data("AAPL"))  # Should return None after retries
