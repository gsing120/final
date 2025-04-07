"""
Test script for Gemma 3 Integration in the Gemma Advanced Trading System

This script tests the Gemma 3 integration to ensure all components work together seamlessly.
"""

import os
import sys
import logging
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Import Gemma 3 integration
from backend.gemma3_integration.gemma3_integration import Gemma3Integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GemmaTrading.Test")

def create_sample_market_data() -> Dict[str, pd.DataFrame]:
    """Create sample market data for testing."""
    # Create sample price data
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
    
    # Sample price data for QQQ
    qqq_data = pd.DataFrame({
        'open': np.random.normal(380, 7, 100),
        'high': np.random.normal(385, 7, 100),
        'low': np.random.normal(375, 7, 100),
        'close': np.random.normal(382, 7, 100),
        'volume': np.random.randint(800000, 8000000, 100)
    }, index=dates)
    
    # Add trend
    qqq_data['close'] = qqq_data['close'].cumsum() / 40 + 360
    qqq_data['open'] = qqq_data['close'].shift(1, fill_value=qqq_data['close'][0])
    qqq_data['high'] = qqq_data[['open', 'close']].max(axis=1) + np.random.normal(3, 0.7, 100)
    qqq_data['low'] = qqq_data[['open', 'close']].min(axis=1) - np.random.normal(3, 0.7, 100)
    
    # Create market data dictionary
    market_data = {
        'SPY': {
            'price': spy_data
        },
        'QQQ': {
            'price': qqq_data
        },
        'market_indices': {
            'SPY': spy_data,
            'QQQ': qqq_data
        }
    }
    
    return market_data

def create_sample_news_data() -> Dict[str, List[Dict[str, Any]]]:
    """Create sample news data for testing."""
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
    
    # Sample news data for QQQ
    qqq_news = [
        {
            'title': 'Tech stocks rally on AI optimism',
            'content': 'Technology stocks rallied today as investors remain optimistic about the potential of artificial intelligence.',
            'source': 'CNBC',
            'date': datetime.datetime.now().isoformat(),
            'sentiment': 'positive'
        },
        {
            'title': 'Major tech earnings beat expectations',
            'content': 'Major technology companies reported earnings that beat analyst expectations, driving the Nasdaq higher.',
            'source': 'Reuters',
            'date': (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat(),
            'sentiment': 'positive'
        },
        {
            'title': 'Concerns about tech regulation emerge',
            'content': 'Concerns about potential technology regulation have emerged, with lawmakers discussing new antitrust measures.',
            'source': 'New York Times',
            'date': (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat(),
            'sentiment': 'negative'
        }
    ]
    
    # Create news data dictionary
    news_data = {
        'SPY': spy_news,
        'QQQ': qqq_news
    }
    
    return news_data

def create_sample_sentiment_data() -> Dict[str, Dict[str, Any]]:
    """Create sample sentiment data for testing."""
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
    
    # Sample sentiment data for QQQ
    qqq_sentiment = {
        'social_media': {
            'twitter': {
                'sentiment_score': 0.72,
                'volume': 6000,
                'bullish_ratio': 0.75
            },
            'reddit': {
                'sentiment_score': 0.68,
                'volume': 4000,
                'bullish_ratio': 0.7
            }
        },
        'analyst_ratings': {
            'buy': 18,
            'hold': 5,
            'sell': 1
        }
    }
    
    # Create sentiment data dictionary
    sentiment_data = {
        'SPY': spy_sentiment,
        'QQQ': qqq_sentiment
    }
    
    return sentiment_data

def create_sample_economic_data() -> Dict[str, Any]:
    """Create sample economic data for testing."""
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
    
    return economic_data

def create_sample_trading_objectives() -> Dict[str, Any]:
    """Create sample trading objectives for testing."""
    trading_objectives = {
        'risk_tolerance': 'moderate',
        'return_target': 'high',
        'time_horizon': 'medium',
        'max_drawdown': 10.0,
        'min_win_rate': 0.55,
        'preferred_asset_classes': ['stocks', 'etfs'],
        'excluded_sectors': ['energy']
    }
    
    return trading_objectives

def create_sample_portfolio() -> Dict[str, Any]:
    """Create sample portfolio for testing."""
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
                'ticker': 'QQQ',
                'position_type': 'long',
                'quantity': 80,
                'entry_price': 380.0,
                'current_price': 400.0,
                'value': 32000.0,
                'entry_date': (datetime.datetime.now() - datetime.timedelta(days=20)).isoformat(),
                'strategy_id': 'strategy-2'
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
    
    return portfolio

def create_sample_trade() -> Dict[str, Any]:
    """Create sample completed trade for testing."""
    trade = {
        'ticker': 'AAPL',
        'position_type': 'long',
        'quantity': 50,
        'entry_price': 170.0,
        'exit_price': 190.0,
        'entry_date': (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat(),
        'exit_date': datetime.datetime.now().isoformat(),
        'profit_loss': 1000.0,
        'return_pct': 11.76,
        'strategy_id': 'strategy-3',
        'strategy_name': 'Momentum Strategy'
    }
    
    return trade

def test_gemma3_integration():
    """Test the Gemma 3 integration."""
    logger.info("Starting Gemma 3 integration test")
    
    # Initialize Gemma 3 integration
    gemma3 = Gemma3Integration()
    
    # Create sample data
    market_data = create_sample_market_data()
    news_data = create_sample_news_data()
    sentiment_data = create_sample_sentiment_data()
    economic_data = create_sample_economic_data()
    trading_objectives = create_sample_trading_objectives()
    portfolio = create_sample_portfolio()
    trade = create_sample_trade()
    
    # Test Natural Language Market Analysis
    logger.info("Testing Natural Language Market Analysis")
    
    # Test news sentiment analysis
    news_sentiment = gemma3.analyze_news_sentiment('SPY', news_data['SPY'])
    logger.info(f"News sentiment for SPY: {news_sentiment.get('sentiment', 'unknown')}")
    
    # Test social sentiment analysis
    social_sentiment = gemma3.analyze_social_sentiment('SPY', sentiment_data['SPY'])
    logger.info(f"Social sentiment for SPY: {social_sentiment.get('sentiment', 'unknown')}")
    
    # Test market narrative generation
    market_narrative = gemma3.generate_market_narrative('SPY', news_data['SPY'])
    logger.info(f"Market narrative for SPY: {market_narrative[:100]}...")
    
    # Test Advanced Mathematical Modeling
    logger.info("Testing Advanced Mathematical Modeling")
    
    # Test market regime detection
    market_regime = gemma3.detect_market_regime(market_data)
    logger.info(f"Detected market regime: {market_regime.get('regime', 'unknown')}")
    
    # Test volatility forecasting
    volatility_forecast = gemma3.forecast_volatility('SPY', market_data['SPY']['price'])
    logger.info(f"Volatility forecast for SPY: {volatility_forecast.get('forecast', 'unknown')}")
    
    # Test correlation analysis
    correlation_analysis = gemma3.analyze_correlations(market_data)
    logger.info(f"Correlation analysis: {correlation_analysis.get('correlation_summary', {})}")
    
    # Test Strategy Generation and Refinement
    logger.info("Testing Strategy Generation and Refinement")
    
    # Test strategy generation
    strategy = gemma3.generate_strategy(
        asset_type='stock',
        market_conditions={'regime': market_regime.get('regime', 'unknown')},
        trading_objectives=trading_objectives
    )
    logger.info(f"Generated strategy: {strategy.get('name', 'unknown')}")
    
    # Add strategy to library
    strategy_id = gemma3.add_strategy(strategy)
    logger.info(f"Added strategy to library with ID: {strategy_id}")
    
    # Test strategy refinement
    refined_strategy = gemma3.refine_strategy(
        strategy=strategy,
        market_conditions={'regime': market_regime.get('regime', 'unknown')},
        refinement_goals={'improve_win_rate': True}
    )
    logger.info(f"Refined strategy: {refined_strategy.get('name', 'unknown')}")
    
    # Test Real-Time Signal Analysis
    logger.info("Testing Real-Time Signal Analysis")
    
    # Create technical indicators
    technical_indicators = {
        'rsi': 45.0,
        'macd': 0.5,
        'macd_signal': 0.3,
        'ema_20': 425.0,
        'ema_50': 420.0,
        'ema_crossover': True,
        'bollinger_upper': 435.0,
        'bollinger_middle': 425.0,
        'bollinger_lower': 415.0
    }
    
    # Test signal detection
    signals = gemma3.detect_signals(
        ticker='SPY',
        strategy=strategy,
        market_data={'price': market_data['SPY']['price']},
        technical_indicators=technical_indicators,
        market_conditions={'regime': market_regime.get('regime', 'unknown')}
    )
    
    if signals:
        logger.info(f"Detected {len(signals)} signals for SPY")
        
        # Test signal analysis
        signal_analysis = gemma3.analyze_entry_signal(
            signal=signals[0],
            price_data=market_data['SPY']['price'],
            strategy=strategy
        )
        logger.info(f"Signal analysis: {signal_analysis.get('explanation', {}).get('summary', 'unknown')}")
    else:
        logger.info("No signals detected for SPY")
    
    # Test Central Decision Engine
    logger.info("Testing Central Decision Engine")
    
    # Test trading recommendation
    recommendation = gemma3.generate_trading_recommendation(
        ticker='SPY',
        market_data=market_data,
        news_data=news_data['SPY'],
        sentiment_data=sentiment_data['SPY'],
        trading_objectives=trading_objectives
    )
    logger.info(f"Trading recommendation for SPY: {recommendation.get('recommendation', 'unknown')}")
    
    # Test portfolio recommendations
    portfolio_recommendations = gemma3.generate_portfolio_recommendations(
        portfolio=portfolio,
        market_data=market_data,
        trading_objectives=trading_objectives
    )
    logger.info(f"Portfolio recommendations: {portfolio_recommendations.get('recommended_allocation', {})}")
    
    # Test post-trade analysis
    post_trade_analysis = gemma3.generate_post_trade_analysis(
        trade=trade,
        market_data={'price': market_data['SPY']['price']}  # Using SPY data as a substitute
    )
    logger.info(f"Post-trade analysis: {post_trade_analysis.get('successful', False)}")
    
    # Test market insights
    market_insights = gemma3.generate_market_insights(
        market_data=market_data,
        economic_data=economic_data,
        news_data=news_data
    )
    logger.info(f"Market insights: {market_insights.get('market_narrative', '')[:100]}...")
    
    logger.info("Gemma 3 integration test completed successfully")
    return True

if __name__ == "__main__":
    test_gemma3_integration()
