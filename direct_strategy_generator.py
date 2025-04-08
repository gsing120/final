import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import logging

# Import the new strategy optimization components
from backend.gemma3_integration.strategy_optimization import StrategyOptimizer, PerformanceThresholds, StrategyBacktester
from backend.gemma3_integration.architecture_enhanced import GemmaCore
from backend.gemma3_integration.strategy_generation_and_refinement import StrategyGenerator, StrategyRefiner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('frontend.log')
    ]
)

logger = logging.getLogger("GemmaTrading.DirectStrategyGenerator")

def generate_aapl_strategy():
    """
    Generate an optimized trading strategy for AAPL with positive historical performance.
    
    This function uses the StrategyOptimizer to generate a strategy that meets
    performance thresholds, ensuring only strategies with positive historical
    performance are presented to users.
    
    Returns:
    --------
    Dict[str, Any]
        Optimized trading strategy with positive historical performance.
    """
    logger.info("Generating optimized AAPL strategy")
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    try:
        # Initialize components
        gemma_core = GemmaCore()
        strategy_generator = StrategyGenerator(gemma_core)
        strategy_refiner = StrategyRefiner(gemma_core)
        backtester = StrategyBacktester()
        
        # Set performance thresholds to ensure positive performance
        # Minimum total return must be positive (> 0)
        # Minimum Sharpe ratio should be at least 0.5
        # Maximum drawdown should not exceed -20%
        # Minimum win rate should be at least 50%
        performance_thresholds = PerformanceThresholds(
            min_total_return=0.0,      # Must be positive
            min_sharpe_ratio=0.5,      # Reasonable risk-adjusted return
            max_drawdown=-20.0,        # Limit drawdown to 20%
            min_win_rate=50.0          # At least 50% win rate
        )
        
        # Initialize strategy optimizer with components and configuration
        strategy_optimizer = StrategyOptimizer(
            gemma_core=gemma_core,
            strategy_generator=strategy_generator,
            strategy_refiner=strategy_refiner,
            backtester=backtester,
            performance_thresholds=performance_thresholds,
            max_optimization_iterations=5,    # Try up to 5 iterations to find a good strategy
            num_candidate_strategies=3        # Generate 3 candidates per iteration
        )
        
        # Get current market conditions for AAPL
        market_conditions = get_market_conditions('AAPL')
        
        # Define trading objectives
        trading_objectives = {
            "time_horizon": "medium",      # Medium-term trading (weeks to months)
            "risk_tolerance": "moderate",  # Moderate risk tolerance
            "return_target": 10.0,         # Target 10% return
            "max_drawdown": -15.0          # Willing to accept up to 15% drawdown
        }
        
        # Define constraints
        constraints = {
            "max_trades_per_month": 5,     # Limit trading frequency
            "min_holding_period": 3,       # Minimum holding period in days
            "max_position_size": 10.0      # Maximum position size as percentage of portfolio
        }
        
        # Generate optimized strategy
        optimized_strategy = strategy_optimizer.generate_optimized_strategy(
            ticker='AAPL',
            market_conditions=market_conditions,
            trading_objectives=trading_objectives,
            constraints=constraints
        )
        
        if not optimized_strategy.get("success", True):
            logger.error(f"Failed to generate optimized strategy: {optimized_strategy.get('error', 'Unknown error')}")
            return {"error": "Failed to generate optimized strategy", "success": False}
        
        # Extract key information for the frontend
        strategy_result = {
            "success": True,
            "ticker": "AAPL",
            "strategy_type": optimized_strategy.get("strategy_type", "Swing Trading"),
            "recommendation": optimized_strategy.get("recommendation", "HOLD"),
            "trend": optimized_strategy.get("trend", "Neutral"),
            "momentum": optimized_strategy.get("momentum", "Neutral"),
            "volatility": optimized_strategy.get("volatility", "Moderate"),
            "support": optimized_strategy.get("support", 0.0),
            "resistance": optimized_strategy.get("resistance", 0.0),
            "current_price": optimized_strategy.get("current_price", 0.0),
            "performance": optimized_strategy.get("performance", {}),
            "risk_management": optimized_strategy.get("risk_management", {}),
            "optimization_history": optimized_strategy.get("optimization_history", []),
            "validation_results": optimized_strategy.get("validation_results", {})
        }
        
        # Generate charts
        generate_charts('AAPL')
        
        # Add charts to the strategy result
        strategy_result["charts"] = {
            "price": "/static/AAPL_price_plot.png",
            "rsi": "/static/AAPL_rsi_plot.png",
            "macd": "/static/AAPL_macd_plot.png"
        }
        
        # Generate strategy description
        strategy_result["strategy_description"] = generate_strategy_description(strategy_result)
        
        logger.info(f"Generated optimized AAPL strategy with total return: {strategy_result['performance'].get('total_return', 0.0)}")
        
        return strategy_result
    
    except Exception as e:
        logger.exception(f"Error generating optimized AAPL strategy: {e}")
        return {"error": f"Error generating strategy: {str(e)}", "success": False}

def get_market_conditions(ticker):
    """
    Get current market conditions for a ticker.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol.
        
    Returns:
    --------
    Dict[str, Any]
        Market conditions.
    """
    logger.info(f"Getting market conditions for {ticker}")
    
    try:
        # Download data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return {"regime": "unknown", "volatility": "unknown"}
        
        # Calculate indicators
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Determine trend
        trend = "bullish" if data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1] else "bearish"
        
        # Determine momentum
        momentum = "strong" if data['RSI'].iloc[-1] > 50 else "weak"
        
        # Determine volatility
        volatility_value = data['Close'].pct_change().rolling(window=20).std().iloc[-1] * 100
        volatility = "high" if volatility_value > 2 else "low"
        
        # Create market conditions
        market_conditions = {
            "regime": trend,
            "momentum": momentum,
            "volatility": volatility,
            "volatility_value": float(volatility_value),
            "rsi": float(data['RSI'].iloc[-1]),
            "sma20": float(data['SMA20'].iloc[-1]),
            "sma50": float(data['SMA50'].iloc[-1]),
            "current_price": float(data['Close'].iloc[-1])
        }
        
        logger.info(f"Market conditions for {ticker}: {trend} regime with {momentum} momentum")
        
        return market_conditions
    
    except Exception as e:
        logger.exception(f"Error getting market conditions for {ticker}: {e}")
        return {"regime": "unknown", "volatility": "unknown"}

def generate_charts(ticker):
    """
    Generate charts for a ticker.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol.
    """
    logger.info(f"Generating charts for {ticker}")
    
    try:
        # Download data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return
        
        # Calculate indicators
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Create price plot
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label=f'{ticker} Price')
        plt.plot(data.index, data['SMA20'], label='20-day SMA')
        plt.plot(data.index, data['SMA50'], label='50-day SMA')
        plt.title(f'{ticker} Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.savefig('static/AAPL_price_plot.png')
        
        # Create RSI plot
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['RSI'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        plt.title(f'{ticker} RSI')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.savefig('static/AAPL_rsi_plot.png')
        
        # Create MACD plot
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['MACD'], label='MACD')
        plt.plot(data.index, data['Signal'], label='Signal Line')
        plt.bar(data.index, data['MACD'] - data['Signal'], alpha=0.3)
        plt.title(f'{ticker} MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.savefig('static/AAPL_macd_plot.png')
        
        logger.info(f"Generated charts for {ticker}")
    
    except Exception as e:
        logger.exception(f"Error generating charts for {ticker}: {e}")

def generate_strategy_description(strategy):
    """
    Generate a description for a strategy.
    
    Parameters:
    -----------
    strategy : Dict[str, Any]
        Strategy data.
        
    Returns:
    --------
    str
        Strategy description.
    """
    trend = strategy.get("trend", "Neutral").lower()
    momentum = strategy.get("momentum", "Neutral").lower()
    recommendation = strategy.get("recommendation", "HOLD")
    
    # Get performance metrics
    performance = strategy.get("performance", {})
    total_return = performance.get("total_return", 0.0)
    sharpe_ratio = performance.get("sharpe_ratio", 0.0)
    
    # Get risk management parameters
    risk_management = strategy.get("risk_management", {})
    entry_price = risk_management.get("entry_price", 0.0)
    stop_loss = risk_management.get("stop_loss", 0.0)
    take_profit = risk_management.get("take_profit", 0.0)
    
    # Generate description
    description = f"This {trend} strategy for AAPL is based on a combination of trend following and momentum indicators. "
    description += f"The 20-day SMA is {'above' if trend == 'bullish' else 'below'} the 50-day SMA, indicating a {trend} trend. "
    description += f"The RSI shows {momentum} momentum, and the MACD {'confirms the bullish momentum' if recommendation == 'BUY' else 'confirms the bearish momentum' if recommendation == 'SELL' else 'shows mixed signals'}. "
    
    # Add performance information
    description += f"This strategy has demonstrated a historical return of {total_return}% with a Sharpe ratio of {sharpe_ratio}. "
    
    # Add recommendation
    description += f"The strategy recommends a {recommendation} position "
    
    # Add risk management
    if recommendation in ["BUY", "SELL"]:
        description += f"with a stop loss at {stop_loss} and a take profit at {take_profit}."
    else:
        description += f"as the current market conditions do not present a clear trading opportunity."
    
    # Add optimization information if available
    optimization_history = strategy.get("optimization_history", [])
    if optimization_history:
        num_iterations = len(optimization_history)
        description += f" This strategy was optimized through {num_iterations} iterations to ensure positive historical performance."
    
    return description

if __name__ == "__main__":
    result = generate_aapl_strategy()
    print(json.dumps(result, indent=2))
    
    # Save the result to a file
    with open('aapl_strategy_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("Strategy generated and saved to aapl_strategy_result.json")
