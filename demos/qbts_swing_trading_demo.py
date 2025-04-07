"""
QBTS Swing Trading Strategy Demo

This script demonstrates the QBTS swing trading strategy using the Gemma 3 integration.
"""

import os
import sys
import logging
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add parent directory to path to import modules
sys.path.append('/home/ubuntu/gemma_advanced')

# Import QBTS Swing Trading Strategy
from backend.strategies.qbts_swing_trading import QBTSSwingTradingStrategy
from backend.gemma3_integration.gemma3_integration import Gemma3Integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GemmaTrading.QBTSDemo")

def create_sample_data():
    """Create sample data for the demo."""
    # Create sample market data
    dates = pd.date_range(end=datetime.datetime.now(), periods=100, freq='D')
    
    # Sample price data for SPY
    spy_data = pd.DataFrame({
        'open': np.random.normal(400, 5, 100),
        'high': np.random.normal(405, 5, 100),
        'low': np.random.normal(395, 5, 100),
        'close': np.random.normal(402, 5, 100),
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Add trend
    spy_data['close'] = spy_data['close'].cumsum() / 50 + 380
    spy_data['open'] = spy_data['close'].shift(1, fill_value=spy_data['close'][0])
    spy_data['high'] = spy_data[['open', 'close']].max(axis=1) + np.random.normal(2, 0.5, 100)
    spy_data['low'] = spy_data[['open', 'close']].min(axis=1) - np.random.normal(2, 0.5, 100)
    
    # Sample price data for AAPL
    aapl_data = pd.DataFrame({
        'open': np.random.normal(180, 2, 100),
        'high': np.random.normal(182, 2, 100),
        'low': np.random.normal(178, 2, 100),
        'close': np.random.normal(181, 2, 100),
        'volume': np.random.randint(500000, 5000000, 100)
    }, index=dates)
    
    # Add trend
    aapl_data['close'] = aapl_data['close'].cumsum() / 100 + 170
    aapl_data['open'] = aapl_data['close'].shift(1, fill_value=aapl_data['close'][0])
    aapl_data['high'] = aapl_data[['open', 'close']].max(axis=1) + np.random.normal(1, 0.3, 100)
    aapl_data['low'] = aapl_data[['open', 'close']].min(axis=1) - np.random.normal(1, 0.3, 100)
    
    # Sample price data for MSFT
    msft_data = pd.DataFrame({
        'open': np.random.normal(350, 3, 100),
        'high': np.random.normal(353, 3, 100),
        'low': np.random.normal(347, 3, 100),
        'close': np.random.normal(351, 3, 100),
        'volume': np.random.randint(400000, 4000000, 100)
    }, index=dates)
    
    # Add trend
    msft_data['close'] = msft_data['close'].cumsum() / 80 + 340
    msft_data['open'] = msft_data['close'].shift(1, fill_value=msft_data['close'][0])
    msft_data['high'] = msft_data[['open', 'close']].max(axis=1) + np.random.normal(1.5, 0.4, 100)
    msft_data['low'] = msft_data[['open', 'close']].min(axis=1) - np.random.normal(1.5, 0.4, 100)
    
    # Create market data dictionary
    market_data = {
        'SPY': {
            'price': spy_data
        },
        'AAPL': {
            'price': aapl_data
        },
        'MSFT': {
            'price': msft_data
        },
        'market_indices': {
            'SPY': spy_data
        }
    }
    
    # Sample news data for SPY
    spy_news = [
        {
            'title': 'S&P 500 reaches new all-time high',
            'content': 'The S&P 500 reached a new all-time high today, driven by strong earnings reports from technology companies.',
            'source': 'Financial Times',
            'date': datetime.datetime.now().isoformat(),
            'sentiment': 'positive'
        },
        {
            'title': 'Fed signals potential rate cut',
            'content': 'The Federal Reserve signaled a potential rate cut in the coming months, citing improving inflation data.',
            'source': 'Wall Street Journal',
            'date': (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat(),
            'sentiment': 'positive'
        },
        {
            'title': 'Economic data shows mixed signals',
            'content': 'Recent economic data shows mixed signals, with strong employment but weakening consumer spending.',
            'source': 'Bloomberg',
            'date': (datetime.datetime.now() - datetime.timedelta(days=2)).isoformat(),
            'sentiment': 'neutral'
        }
    ]
    
    # Sample news data for AAPL
    aapl_news = [
        {
            'title': 'Apple announces new iPhone model',
            'content': 'Apple announced a new iPhone model with advanced AI capabilities, expected to drive significant sales growth.',
            'source': 'CNBC',
            'date': datetime.datetime.now().isoformat(),
            'sentiment': 'positive'
        },
        {
            'title': 'Apple services revenue hits record high',
            'content': 'Apple reported record services revenue in the latest quarter, showing strong growth in its high-margin business.',
            'source': 'Reuters',
            'date': (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat(),
            'sentiment': 'positive'
        },
        {
            'title': 'Apple faces regulatory scrutiny in EU',
            'content': 'Apple is facing increased regulatory scrutiny in the European Union over its App Store policies.',
            'source': 'New York Times',
            'date': (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat(),
            'sentiment': 'negative'
        }
    ]
    
    # Sample news data for MSFT
    msft_news = [
        {
            'title': 'Microsoft cloud business continues strong growth',
            'content': 'Microsoft reported strong growth in its cloud business, with Azure revenue up 30% year-over-year.',
            'source': 'CNBC',
            'date': datetime.datetime.now().isoformat(),
            'sentiment': 'positive'
        },
        {
            'title': 'Microsoft expands AI capabilities',
            'content': 'Microsoft announced expanded AI capabilities across its product lineup, leveraging its investment in OpenAI.',
            'source': 'TechCrunch',
            'date': (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat(),
            'sentiment': 'positive'
        },
        {
            'title': 'Microsoft faces antitrust concerns',
            'content': 'Microsoft is facing antitrust concerns related to its recent acquisitions in the gaming industry.',
            'source': 'Wall Street Journal',
            'date': (datetime.datetime.now() - datetime.timedelta(days=2)).isoformat(),
            'sentiment': 'negative'
        }
    ]
    
    # Create news data dictionary
    news_data = {
        'SPY': spy_news,
        'AAPL': aapl_news,
        'MSFT': msft_news
    }
    
    # Sample sentiment data for SPY
    spy_sentiment = {
        'social_media': {
            'twitter': {
                'sentiment_score': 0.65,
                'volume': 5000,
                'bullish_ratio': 0.7
            },
            'reddit': {
                'sentiment_score': 0.58,
                'volume': 3000,
                'bullish_ratio': 0.65
            }
        },
        'analyst_ratings': {
            'buy': 15,
            'hold': 8,
            'sell': 2
        }
    }
    
    # Sample sentiment data for AAPL
    aapl_sentiment = {
        'social_media': {
            'twitter': {
                'sentiment_score': 0.72,
                'volume': 8000,
                'bullish_ratio': 0.75
            },
            'reddit': {
                'sentiment_score': 0.68,
                'volume': 6000,
                'bullish_ratio': 0.7
            }
        },
        'analyst_ratings': {
            'buy': 25,
            'hold': 5,
            'sell': 1
        }
    }
    
    # Sample sentiment data for MSFT
    msft_sentiment = {
        'social_media': {
            'twitter': {
                'sentiment_score': 0.7,
                'volume': 7000,
                'bullish_ratio': 0.73
            },
            'reddit': {
                'sentiment_score': 0.65,
                'volume': 5000,
                'bullish_ratio': 0.68
            }
        },
        'analyst_ratings': {
            'buy': 28,
            'hold': 4,
            'sell': 0
        }
    }
    
    # Create sentiment data dictionary
    sentiment_data = {
        'SPY': spy_sentiment,
        'AAPL': aapl_sentiment,
        'MSFT': msft_sentiment
    }
    
    # Sample economic data
    economic_data = {
        'gdp_growth': 2.8,
        'unemployment_rate': 3.6,
        'inflation_rate': 2.4,
        'interest_rate': 5.25,
        'consumer_confidence': 105.7,
        'manufacturing_pmi': 52.3,
        'services_pmi': 54.1,
        'retail_sales_growth': 3.1,
        'housing_starts': 1.45,
        'date': datetime.datetime.now().isoformat()
    }
    
    # Sample portfolio
    portfolio = {
        'cash': 50000.0,
        'total_value': 200000.0,
        'positions': [
            {
                'ticker': 'SPY',
                'position_type': 'long',
                'quantity': 100,
                'entry_price': 420.0,
                'current_price': 430.0,
                'value': 43000.0,
                'entry_date': (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat(),
                'strategy_id': 'strategy-1'
            },
            {
                'ticker': 'AAPL',
                'position_type': 'long',
                'quantity': 200,
                'entry_price': 180.0,
                'current_price': 190.0,
                'value': 38000.0,
                'entry_date': (datetime.datetime.now() - datetime.timedelta(days=45)).isoformat(),
                'strategy_id': 'strategy-3'
            },
            {
                'ticker': 'MSFT',
                'position_type': 'long',
                'quantity': 100,
                'entry_price': 350.0,
                'current_price': 370.0,
                'value': 37000.0,
                'entry_date': (datetime.datetime.now() - datetime.timedelta(days=15)).isoformat(),
                'strategy_id': 'strategy-4'
            }
        ]
    }
    
    return {
        'market_data': market_data,
        'news_data': news_data,
        'sentiment_data': sentiment_data,
        'economic_data': economic_data,
        'portfolio': portfolio
    }

def demonstrate_qbts_strategy():
    """Demonstrate the QBTS swing trading strategy."""
    logger.info("Starting QBTS Swing Trading Strategy demonstration")
    
    # Initialize Gemma 3 integration
    gemma3 = Gemma3Integration()
    
    # Initialize QBTS strategy
    qbts = QBTSSwingTradingStrategy(gemma3)
    
    # Create sample data
    data = create_sample_data()
    
    # Step 1: Analyze market conditions
    logger.info("Step 1: Analyzing market conditions")
    market_insights = qbts.analyze_market_conditions(
        market_data=data['market_data'],
        economic_data=data['economic_data'],
        news_data=data['news_data']
    )
    
    logger.info(f"Market regime: {qbts.strategy_state['market_regime']}")
    
    # Step 2: Scan for opportunities
    logger.info("Step 2: Scanning for opportunities")
    tickers = ['SPY', 'AAPL', 'MSFT']
    opportunities = qbts.scan_for_opportunities(
        tickers=tickers,
        market_data=data['market_data'],
        news_data=data['news_data'],
        sentiment_data=data['sentiment_data']
    )
    
    if not opportunities:
        logger.info("No trading opportunities found")
        return
    
    logger.info(f"Found {len(opportunities)} trading opportunities")
    
    # Step 3: Generate trade plan for the best opportunity
    logger.info("Step 3: Generating trade plan")
    best_opportunity = opportunities[0]
    trade_plan = qbts.generate_trade_plan(
        opportunity=best_opportunity,
        portfolio=data['portfolio']
    )
    
    logger.info(f"Generated trade plan for {trade_plan['ticker']} {trade_plan['action']}")
    logger.info(f"Entry price: {trade_plan['entry_price']}")
    logger.info(f"Position size: {trade_plan['position_size']}")
    logger.info(f"Stop loss: {trade_plan['stop_loss']['price']}")
    logger.info(f"Take profit: {trade_plan['take_profit']['price']}")
    
    # Step 4: Execute trade plan
    logger.info("Step 4: Executing trade plan")
    executed_trade = qbts.execute_trade_plan(trade_plan)
    
    logger.info(f"Executed {executed_trade['action']} trade for {executed_trade['ticker']}")
    logger.info(f"Entry price: {executed_trade['entry_price']}")
    logger.info(f"Quantity: {executed_trade['quantity']}")
    
    # Step 5: Monitor positions
    logger.info("Step 5: Monitoring positions")
    current_positions = qbts.strategy_state['current_positions']
    exit_recommendations = qbts.monitor_positions(
        current_positions=current_positions,
        market_data=data['market_data'],
        news_data=data['news_data'],
        sentiment_data=data['sentiment_data']
    )
    
    if not exit_recommendations:
        logger.info("No exit recommendations generated")
    else:
        logger.info(f"Generated {len(exit_recommendations)} exit recommendations")
        
        # Step 6: Execute exit for the best exit recommendation
        logger.info("Step 6: Executing exit")
        best_exit = exit_recommendations[0]
        completed_trade = qbts.execute_exit(best_exit)
        
        logger.info(f"Executed exit for {completed_trade['ticker']}")
        logger.info(f"Exit price: {completed_trade['exit_price']}")
        logger.info(f"P&L: {completed_trade['profit_loss']:.2f} ({completed_trade['return_pct']:.2%})")
        logger.info(f"Exit reason: {completed_trade['exit_reason']}")
    
    # Step 7: Generate performance report
    logger.info("Step 7: Generating performance report")
    performance_report = qbts.generate_performance_report()
    
    if 'message' in performance_report:
        logger.info(performance_report['message'])
    else:
        logger.info(f"Total trades: {performance_report['total_trades']}")
        logger.info(f"Win rate: {performance_report['win_rate']:.2%}")
        logger.info(f"Total P&L: {performance_report['total_profit_loss']:.2f}")
        logger.info(f"Profit factor: {performance_report['profit_factor']:.2f}")
    
    logger.info("QBTS Swing Trading Strategy demonstration completed")

if __name__ == "__main__":
    demonstrate_qbts_strategy()
