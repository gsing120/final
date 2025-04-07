import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.nlp.news_trading import (
    NewsAnalyzer,
    SentimentAnalysis,
    EntityExtraction,
    NewsImpactPredictor
)

class TestNewsTrading:
    """
    Unit tests for the News Trading module.
    """
    
    @pytest.fixture
    def mock_gemma_model(self):
        """Create a mock Gemma model for testing."""
        mock_model = MagicMock()
        
        # Configure the mock to return sample responses
        mock_model.analyze_news_sentiment.return_value = {
            "sentiment_score": 0.75,
            "sentiment_label": "positive",
            "confidence": 0.85,
            "key_phrases": ["strong earnings", "revenue growth", "market expansion"]
        }
        
        mock_model.extract_entities.return_value = {
            "companies": [
                {"name": "Apple Inc.", "ticker": "AAPL", "relevance": 0.9},
                {"name": "Microsoft Corporation", "ticker": "MSFT", "relevance": 0.4}
            ],
            "people": [
                {"name": "Tim Cook", "role": "CEO", "company": "Apple Inc."},
                {"name": "Satya Nadella", "role": "CEO", "company": "Microsoft Corporation"}
            ],
            "locations": [
                {"name": "Cupertino", "country": "USA", "relevance": 0.8}
            ],
            "products": [
                {"name": "iPhone", "company": "Apple Inc.", "relevance": 0.85},
                {"name": "Azure", "company": "Microsoft Corporation", "relevance": 0.3}
            ]
        }
        
        mock_model.predict_news_impact.return_value = {
            "price_impact": 0.03,  # 3% expected price movement
            "volume_impact": 0.5,   # 50% expected volume increase
            "volatility_impact": 0.2,  # 20% expected volatility increase
            "duration": "short-term",  # Impact expected to be short-term
            "confidence": 0.7,
            "affected_sectors": [
                {"name": "Technology", "impact": 0.04},
                {"name": "Consumer Electronics", "impact": 0.03}
            ]
        }
        
        return mock_model
    
    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data for testing."""
        news_data = [
            {
                "title": "Apple Reports Record Quarterly Revenue and EPS",
                "content": "Apple today announced financial results for its fiscal 2023 first quarter ended December 31, 2022. The Company posted quarterly revenue of $97.3 billion, up 9 percent year over year, and quarterly earnings per diluted share of $1.52, up 11 percent year over year.",
                "source": "Company Press Release",
                "published_at": "2023-01-26T13:30:00Z",
                "url": "https://www.apple.com/newsroom/2023/01/apple-reports-first-quarter-results/"
            },
            {
                "title": "Microsoft Announces Layoffs Amid Economic Uncertainty",
                "content": "Microsoft Corp said on Wednesday it would cut 10,000 jobs by the end of the third quarter of fiscal 2023, the latest sign that layoffs were accelerating in the U.S. technology sector as companies brace for an economic downturn.",
                "source": "Reuters",
                "published_at": "2023-01-18T09:15:00Z",
                "url": "https://www.reuters.com/technology/microsoft-cut-10000-jobs-2023-01-18/"
            },
            {
                "title": "Federal Reserve Raises Interest Rates by 25 Basis Points",
                "content": "The Federal Reserve raised its target interest rate by a quarter of a percentage point on Wednesday, yet continued to promise 'ongoing increases' in borrowing costs as part of its still unresolved battle against inflation.",
                "source": "Financial Times",
                "published_at": "2023-02-01T18:00:00Z",
                "url": "https://www.ft.com/content/federal-reserve-raises-rates"
            }
        ]
        
        return news_data
    
    def test_news_analyzer_initialization(self, mock_gemma_model):
        """Test initialization of NewsAnalyzer class."""
        # Initialize the class
        news_analyzer = NewsAnalyzer(model=mock_gemma_model)
        
        # Verify initialization
        assert news_analyzer.model == mock_gemma_model
        assert isinstance(news_analyzer.sentiment_analyzer, SentimentAnalysis)
        assert isinstance(news_analyzer.entity_extractor, EntityExtraction)
        assert isinstance(news_analyzer.impact_predictor, NewsImpactPredictor)
    
    def test_sentiment_analysis(self, mock_gemma_model, sample_news_data):
        """Test sentiment analysis functionality."""
        # Initialize the class
        news_analyzer = NewsAnalyzer(model=mock_gemma_model)
        
        # Perform sentiment analysis on a single news item
        sentiment_result = news_analyzer.analyze_sentiment(sample_news_data[0])
        
        # Verify the result
        assert isinstance(sentiment_result, dict)
        assert "sentiment_score" in sentiment_result
        assert "sentiment_label" in sentiment_result
        assert "confidence" in sentiment_result
        assert "key_phrases" in sentiment_result
        assert "news_title" in sentiment_result
        assert "timestamp" in sentiment_result
        
        assert sentiment_result["sentiment_score"] == 0.75
        assert sentiment_result["sentiment_label"] == "positive"
        assert sentiment_result["confidence"] == 0.85
        assert len(sentiment_result["key_phrases"]) == 3
        
        # Verify the model was called with the correct data
        mock_gemma_model.analyze_news_sentiment.assert_called_once()
        
        # Test batch sentiment analysis
        batch_results = news_analyzer.batch_analyze_sentiment(sample_news_data)
        
        # Verify the results
        assert isinstance(batch_results, list)
        assert len(batch_results) == 3
        
        # Verify each result has the expected structure
        for result in batch_results:
            assert "sentiment_score" in result
            assert "sentiment_label" in result
            assert "news_title" in result
        
        # Verify the model was called for each news item
        assert mock_gemma_model.analyze_news_sentiment.call_count == 4  # 1 from previous test + 3 from batch
    
    def test_entity_extraction(self, mock_gemma_model, sample_news_data):
        """Test entity extraction functionality."""
        # Initialize the class
        news_analyzer = NewsAnalyzer(model=mock_gemma_model)
        
        # Perform entity extraction on a single news item
        entity_result = news_analyzer.extract_entities(sample_news_data[0])
        
        # Verify the result
        assert isinstance(entity_result, dict)
        assert "companies" in entity_result
        assert "people" in entity_result
        assert "locations" in entity_result
        assert "products" in entity_result
        assert "news_title" in entity_result
        assert "timestamp" in entity_result
        
        assert len(entity_result["companies"]) == 2
        assert entity_result["companies"][0]["ticker"] == "AAPL"
        assert len(entity_result["people"]) == 2
        assert len(entity_result["products"]) == 2
        
        # Verify the model was called with the correct data
        mock_gemma_model.extract_entities.assert_called_once()
        
        # Test batch entity extraction
        batch_results = news_analyzer.batch_extract_entities(sample_news_data)
        
        # Verify the results
        assert isinstance(batch_results, list)
        assert len(batch_results) == 3
        
        # Verify each result has the expected structure
        for result in batch_results:
            assert "companies" in result
            assert "people" in result
            assert "news_title" in result
        
        # Verify the model was called for each news item
        assert mock_gemma_model.extract_entities.call_count == 4  # 1 from previous test + 3 from batch
    
    def test_news_impact_prediction(self, mock_gemma_model, sample_news_data):
        """Test news impact prediction functionality."""
        # Initialize the class
        news_analyzer = NewsAnalyzer(model=mock_gemma_model)
        
        # Perform impact prediction on a single news item
        impact_result = news_analyzer.predict_impact(sample_news_data[0], target_ticker="AAPL")
        
        # Verify the result
        assert isinstance(impact_result, dict)
        assert "price_impact" in impact_result
        assert "volume_impact" in impact_result
        assert "volatility_impact" in impact_result
        assert "duration" in impact_result
        assert "confidence" in impact_result
        assert "affected_sectors" in impact_result
        assert "news_title" in impact_result
        assert "target_ticker" in impact_result
        assert "timestamp" in impact_result
        
        assert impact_result["price_impact"] == 0.03
        assert impact_result["volume_impact"] == 0.5
        assert impact_result["duration"] == "short-term"
        assert impact_result["target_ticker"] == "AAPL"
        assert len(impact_result["affected_sectors"]) == 2
        
        # Verify the model was called with the correct data
        mock_gemma_model.predict_news_impact.assert_called_once_with(
            news=sample_news_data[0],
            target_ticker="AAPL"
        )
        
        # Test batch impact prediction
        target_tickers = ["AAPL", "MSFT", "SPY"]
        batch_results = news_analyzer.batch_predict_impact(sample_news_data, target_tickers)
        
        # Verify the results
        assert isinstance(batch_results, dict)
        assert len(batch_results) == 3  # One entry per ticker
        
        # Verify each ticker has results for each news item
        for ticker in target_tickers:
            assert ticker in batch_results
            assert isinstance(batch_results[ticker], list)
            assert len(batch_results[ticker]) == 3  # Three news items
        
        # Verify the model was called for each news-ticker combination
        assert mock_gemma_model.predict_news_impact.call_count == 10  # 1 from previous test + 9 from batch
    
    def test_news_categorization(self, mock_gemma_model, sample_news_data):
        """Test news categorization functionality."""
        # Initialize the class
        news_analyzer = NewsAnalyzer(model=mock_gemma_model)
        
        # Configure mock for categorization
        mock_gemma_model.categorize_news.return_value = {
            "primary_category": "Earnings",
            "secondary_categories": ["Financial Performance", "Corporate News"],
            "relevance_score": 0.95,
            "market_relevance": "high"
        }
        
        # Perform news categorization
        category_result = news_analyzer.categorize_news(sample_news_data[0])
        
        # Verify the result
        assert isinstance(category_result, dict)
        assert "primary_category" in category_result
        assert "secondary_categories" in category_result
        assert "relevance_score" in category_result
        assert "market_relevance" in category_result
        assert "news_title" in category_result
        assert "timestamp" in category_result
        
        assert category_result["primary_category"] == "Earnings"
        assert len(category_result["secondary_categories"]) == 2
        assert category_result["market_relevance"] == "high"
        
        # Verify the model was called with the correct data
        mock_gemma_model.categorize_news.assert_called_once_with(news=sample_news_data[0])
    
    def test_trading_signal_generation(self, mock_gemma_model, sample_news_data):
        """Test trading signal generation functionality."""
        # Initialize the class
        news_analyzer = NewsAnalyzer(model=mock_gemma_model)
        
        # Configure mock for signal generation
        mock_gemma_model.generate_trading_signal.return_value = {
            "action": "buy",
            "ticker": "AAPL",
            "confidence": 0.8,
            "time_horizon": "short-term",
            "entry_price_range": {"min": 150.0, "max": 155.0},
            "target_price": 165.0,
            "stop_loss": 145.0,
            "position_size_recommendation": "moderate",
            "reasoning": [
                "Strong earnings report exceeding expectations",
                "Positive forward guidance",
                "Increased dividend announced"
            ]
        }
        
        # Perform trading signal generation
        signal_result = news_analyzer.generate_trading_signal(
            news=sample_news_data[0],
            ticker="AAPL",
            current_price=152.0,
            risk_profile="moderate"
        )
        
        # Verify the result
        assert isinstance(signal_result, dict)
        assert "action" in signal_result
        assert "ticker" in signal_result
        assert "confidence" in signal_result
        assert "time_horizon" in signal_result
        assert "entry_price_range" in signal_result
        assert "target_price" in signal_result
        assert "stop_loss" in signal_result
        assert "position_size_recommendation" in signal_result
        assert "reasoning" in signal_result
        assert "news_title" in signal_result
        assert "timestamp" in signal_result
        
        assert signal_result["action"] == "buy"
        assert signal_result["ticker"] == "AAPL"
        assert signal_result["confidence"] == 0.8
        assert len(signal_result["reasoning"]) == 3
        
        # Verify the model was called with the correct parameters
        mock_gemma_model.generate_trading_signal.assert_called_once_with(
            news=sample_news_data[0],
            ticker="AAPL",
            current_price=152.0,
            risk_profile="moderate"
        )
    
    def test_news_backtesting(self, mock_gemma_model):
        """Test news backtesting functionality."""
        # Initialize the class
        news_analyzer = NewsAnalyzer(model=mock_gemma_model)
        
        # Create historical news and price data
        historical_news = [
            {
                "title": "Apple Announces New iPhone",
                "content": "Apple today announced the new iPhone with revolutionary features.",
                "published_at": "2022-09-07T13:00:00Z"
            },
            {
                "title": "Apple Reports Strong Earnings",
                "content": "Apple reported earnings that beat analyst expectations.",
                "published_at": "2022-10-27T13:30:00Z"
            },
            {
                "title": "Apple Faces Supply Chain Issues",
                "content": "Apple is experiencing supply chain constraints for its products.",
                "published_at": "2022-11-15T09:00:00Z"
            }
        ]
        
        # Create price data (dates aligned with news events)
        dates = pd.date_range(start='2022-09-01', end='2022-12-01', freq='D')
        prices = pd.Series(np.linspace(150, 170, len(dates)) + np.random.normal(0, 2, len(dates)), index=dates)
        
        # Configure mock for backtesting
        mock_gemma_model.generate_trading_signal.side_effect = [
            {"action": "buy", "confidence": 0.8},
            {"action": "buy", "confidence": 0.9},
            {"action": "sell", "confidence": 0.7}
        ]
        
        # Perform news backtesting
        backtest_result = news_analyzer.backtest_news_strategy(
            historical_news=historical_news,
            price_data=prices,
            ticker="AAPL",
            holding_period=5,
            confidence_threshold=0.7
        )
        
        # Verify the result
        assert isinstance(backtest_result, dict)
        assert "trades" in backtest_result
        assert "performance_metrics" in backtest_result
        assert "summary" in backtest_result
        
        assert isinstance(backtest_result["trades"], list)
        assert len(backtest_result["trades"]) == 3  # One trade per news item
        
        assert "total_return" in backtest_result["performance_metrics"]
        assert "win_rate" in backtest_result["performance_metrics"]
        assert "average_return" in backtest_result["performance_metrics"]
        
        # Verify the model was called for each news item
        assert mock_gemma_model.generate_trading_signal.call_count == 3
