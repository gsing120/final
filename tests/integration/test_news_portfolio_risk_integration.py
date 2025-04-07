import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.nlp.news_trading import NewsAnalyzer
from backend.risk_management.core import Portfolio, RiskManager
from backend.portfolio_optimization import PortfolioOptimizer

class TestNewsPortfolioRiskIntegration:
    """
    Integration tests for the interaction between news analysis, portfolio management, and risk management.
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
        
        mock_model.generate_trading_signal.return_value = {
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
    
    @pytest.fixture
    def news_analyzer(self, mock_gemma_model):
        """Create a NewsAnalyzer instance for testing."""
        return NewsAnalyzer(model=mock_gemma_model)
    
    @pytest.fixture
    def portfolio(self):
        """Create a Portfolio instance for testing."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        # Add positions
        portfolio.add_position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            entry_date="2023-01-15",
            stop_loss=140.0,
            take_profit=170.0,
            position_type="long"
        )
        
        portfolio.add_position(
            symbol="MSFT",
            quantity=50,
            entry_price=250.0,
            entry_date="2023-01-16",
            stop_loss=240.0,
            take_profit=270.0,
            position_type="long"
        )
        
        portfolio.add_position(
            symbol="GOOGL",
            quantity=20,
            entry_price=2000.0,
            entry_date="2023-01-17",
            stop_loss=1900.0,
            take_profit=2200.0,
            position_type="long"
        )
        
        return portfolio
    
    @pytest.fixture
    def risk_manager(self, portfolio):
        """Create a RiskManager instance for testing."""
        return RiskManager(
            portfolio=portfolio,
            max_portfolio_risk=5.0,  # 5% max portfolio risk
            max_position_risk=2.0,   # 2% max position risk
            max_sector_exposure=30.0, # 30% max sector exposure
            max_correlation=0.7       # 0.7 max correlation
        )
    
    @pytest.fixture
    def portfolio_optimizer(self):
        """Create a PortfolioOptimizer instance for testing."""
        return PortfolioOptimizer()
    
    def test_news_impact_on_risk_management(self, news_analyzer, portfolio, risk_manager, sample_news_data):
        """Test how news analysis impacts risk management decisions."""
        # Get current portfolio risk status
        current_prices = {"AAPL": 155.0, "MSFT": 255.0, "GOOGL": 2050.0}
        initial_risk_status = risk_manager.check_risk_limits(current_prices)
        
        # Analyze news impact for portfolio holdings
        news_impacts = {}
        for position in portfolio.positions:
            symbol = position.symbol
            for news in sample_news_data:
                if symbol.lower() in news["title"].lower() or symbol.lower() in news["content"].lower():
                    impact = news_analyzer.predict_impact(news, target_ticker=symbol)
                    if symbol not in news_impacts:
                        news_impacts[symbol] = []
                    news_impacts[symbol].append(impact)
        
        # Adjust risk management based on news impact
        adjusted_stop_losses = {}
        for symbol, impacts in news_impacts.items():
            # Find position
            position_index = next((i for i, pos in enumerate(portfolio.positions) if pos.symbol == symbol), None)
            if position_index is not None:
                position = portfolio.positions[position_index]
                
                # Calculate average expected price impact
                avg_price_impact = sum(impact["price_impact"] for impact in impacts) / len(impacts)
                avg_volatility_impact = sum(impact["volatility_impact"] for impact in impacts) / len(impacts)
                
                # Adjust stop loss based on news impact
                current_price = current_prices[symbol]
                current_stop_loss = position.stop_loss
                
                if avg_price_impact < -0.02:  # Negative news
                    # Tighten stop loss
                    new_stop_loss = current_price * (1 - 0.03)  # Tighter stop loss
                    adjusted_stop_losses[symbol] = max(new_stop_loss, current_stop_loss)
                elif avg_volatility_impact > 0.15:  # High volatility expected
                    # Widen stop loss to avoid getting stopped out by volatility
                    volatility_adjusted_stop = current_price * (1 - 0.07)  # Wider stop loss
                    adjusted_stop_losses[symbol] = min(volatility_adjusted_stop, current_stop_loss)
        
        # Apply adjusted stop losses
        for symbol, new_stop_loss in adjusted_stop_losses.items():
            position_index = next((i for i, pos in enumerate(portfolio.positions) if pos.symbol == symbol), None)
            if position_index is not None:
                portfolio.positions[position_index].stop_loss = new_stop_loss
        
        # Check risk status after adjustments
        adjusted_risk_status = risk_manager.check_risk_limits(current_prices)
        
        # Verify risk adjustments
        assert initial_risk_status != adjusted_risk_status
        
        # Verify that stop losses were adjusted based on news
        for symbol, new_stop_loss in adjusted_stop_losses.items():
            position_index = next((i for i, pos in enumerate(portfolio.positions) if pos.symbol == symbol), None)
            assert portfolio.positions[position_index].stop_loss == new_stop_loss
    
    def test_news_based_trading_signals_with_risk_constraints(self, news_analyzer, portfolio, risk_manager, sample_news_data):
        """Test generating trading signals from news with risk management constraints."""
        # Current portfolio state
        current_prices = {"AAPL": 155.0, "MSFT": 255.0, "GOOGL": 2050.0}
        portfolio_value = portfolio.total_value(current_prices)
        
        # Generate trading signals from news
        trading_signals = []
        for news in sample_news_data:
            # Extract potential ticker mentions from news
            potential_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            for ticker in potential_tickers:
                if ticker.lower() in news["title"].lower() or ticker.lower() in news["content"].lower():
                    # Generate trading signal
                    signal = news_analyzer.generate_trading_signal(
                        news=news,
                        ticker=ticker,
                        current_price=current_prices.get(ticker, 100.0),  # Default price if not in our current prices
                        risk_profile="moderate"
                    )
                    trading_signals.append(signal)
        
        # Filter signals based on confidence
        high_confidence_signals = [signal for signal in trading_signals if signal["confidence"] > 0.7]
        
        # Apply risk constraints to signals
        executable_signals = []
        for signal in high_confidence_signals:
            ticker = signal["ticker"]
            action = signal["action"]
            
            if action == "buy":
                # Check if we already have a position
                existing_position = next((pos for pos in portfolio.positions if pos.symbol == ticker), None)
                
                if existing_position:
                    # We already have a position, check if we should add to it
                    current_exposure = existing_position.current_value(current_prices[ticker]) / portfolio_value
                    
                    # Check sector exposure
                    sector = "Technology"  # In a real implementation, we would look this up
                    sector_exposure = sum(pos.current_value(current_prices.get(pos.symbol, 100.0)) / portfolio_value 
                                         for pos in portfolio.positions 
                                         if pos.symbol in ["AAPL", "MSFT", "GOOGL"])  # Tech stocks
                    
                    if current_exposure < 0.1 and sector_exposure < risk_manager.max_sector_exposure / 100:
                        # We can add to the position
                        executable_signals.append(signal)
                else:
                    # New position, calculate position size based on risk
                    entry_price = signal["entry_price_range"]["max"]
                    stop_loss = signal["stop_loss"]
                    risk_per_share = entry_price - stop_loss
                    
                    # Calculate position size based on risk per trade
                    risk_per_trade = portfolio_value * (risk_manager.max_position_risk / 100)
                    max_shares = int(risk_per_trade / risk_per_share)
                    
                    if max_shares > 0:
                        # Add position size to signal
                        signal["position_size"] = max_shares
                        executable_signals.append(signal)
            
            elif action == "sell":
                # Check if we have a position to sell
                existing_position = next((pos for pos in portfolio.positions if pos.symbol == ticker), None)
                
                if existing_position:
                    # We have a position to sell
                    signal["position_size"] = existing_position.quantity
                    executable_signals.append(signal)
        
        # Verify that we have executable signals
        assert len(executable_signals) > 0
        
        # Verify that each executable signal has a position size
        for signal in executable_signals:
            assert "position_size" in signal
            assert signal["position_size"] > 0
            
            # Verify that buy signals have stop loss and take profit
            if signal["action"] == "buy":
                assert "stop_loss" in signal
                assert "target_price" in signal
    
    def test_portfolio_optimization_with_news_sentiment(self, news_analyzer, portfolio_optimizer, portfolio, sample_news_data):
        """Test portfolio optimization incorporating news sentiment."""
        # Current portfolio state
        current_prices = {"AAPL": 155.0, "MSFT": 255.0, "GOOGL": 2050.0}
        
        # Analyze sentiment for all news
        sentiments = {}
        for news in sample_news_data:
            sentiment = news_analyzer.analyze_sentiment(news)
            
            # Extract ticker mentions
            potential_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            for ticker in potential_tickers:
                if ticker.lower() in news["title"].lower() or ticker.lower() in news["content"].lower():
                    if ticker not in sentiments:
                        sentiments[ticker] = []
                    sentiments[ticker].append(sentiment)
        
        # Calculate average sentiment score for each ticker
        avg_sentiments = {}
        for ticker, ticker_sentiments in sentiments.items():
            avg_sentiments[ticker] = sum(s["sentiment_score"] for s in ticker_sentiments) / len(ticker_sentiments)
        
        # Create expected returns based on sentiment
        expected_returns = {}
        for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]:
            if ticker in avg_sentiments:
                # Map sentiment score (0-1) to expected return (-0.1 to 0.2)
                expected_returns[ticker] = -0.1 + avg_sentiments[ticker] * 0.3
            else:
                # Default expected return if no news
                expected_returns[ticker] = 0.05
        
        # Create risk model (correlation matrix)
        correlation_matrix = pd.DataFrame({
            "AAPL": [1.0, 0.7, 0.6, 0.5, 0.4],
            "MSFT": [0.7, 1.0, 0.7, 0.6, 0.5],
            "GOOGL": [0.6, 0.7, 1.0, 0.7, 0.6],
            "AMZN": [0.5, 0.6, 0.7, 1.0, 0.7],
            "META": [0.4, 0.5, 0.6, 0.7, 1.0]
        }, index=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
        
        # Volatility estimates
        volatilities = {
            "AAPL": 0.25,
            "MSFT": 0.28,
            "GOOGL": 0.30,
            "AMZN": 0.35,
            "META": 0.40
        }
        
        # Mock the optimize_portfolio method
        with patch.object(portfolio_optimizer, 'optimize_portfolio') as mock_optimize:
            # Configure mock to return sample optimization results
            mock_optimize.return_value = {
                "optimal_weights": {
                    "AAPL": 0.3,
                    "MSFT": 0.25,
                    "GOOGL": 0.2,
                    "AMZN": 0.15,
                    "META": 0.1
                },
                "expected_return": 0.12,
                "expected_risk": 0.18,
                "sharpe_ratio": 0.67
            }
            
            # Run portfolio optimization
            optimization_result = portfolio_optimizer.optimize_portfolio(
                tickers=list(expected_returns.keys()),
                expected_returns=expected_returns,
                correlation_matrix=correlation_matrix,
                volatilities=volatilities,
                risk_tolerance=0.5,
                constraints={
                    "min_weight": 0.05,
                    "max_weight": 0.4
                }
            )
            
            # Verify optimization results
            assert isinstance(optimization_result, dict)
            assert "optimal_weights" in optimization_result
            assert "expected_return" in optimization_result
            assert "expected_risk" in optimization_result
            assert "sharpe_ratio" in optimization_result
            
            # Verify weights sum to 1
            assert sum(optimization_result["optimal_weights"].values()) == pytest.approx(1.0)
            
            # Verify the optimize_portfolio method was called with the correct parameters
            mock_optimize.assert_called_once_with(
                tickers=list(expected_returns.keys()),
                expected_returns=expected_returns,
                correlation_matrix=correlation_matrix,
                volatilities=volatilities,
                risk_tolerance=0.5,
                constraints={
                    "min_weight": 0.05,
                    "max_weight": 0.4
                }
            )
            
            # Calculate rebalancing trades
            current_weights = {}
            portfolio_value = portfolio.total_value(current_prices)
            
            for position in portfolio.positions:
                symbol = position.symbol
                position_value = position.current_value(current_prices[symbol])
                current_weights[symbol] = position_value / portfolio_value
            
            # Add zero weights for tickers not in portfolio
            for ticker in expected_returns.keys():
                if ticker not in current_weights:
                    current_weights[ticker] = 0.0
            
            # Calculate target values
            target_values = {ticker: weight * portfolio_value for ticker, weight in optimization_result["optimal_weights"].items()}
            
            # Calculate current values
            current_values = {ticker: weight * portfolio_value for ticker, weight in current_weights.items()}
            
            # Calculate trades
            trades = {}
            for ticker in expected_returns.keys():
                trade_value = target_values.get(ticker, 0) - current_values.get(ticker, 0)
                if abs(trade_value) > 100:  # Ignore small trades
                    price = current_prices.get(ticker, 100.0)
                    shares = int(trade_value / price)
                    if shares != 0:
                        trades[ticker] = shares
            
            # Verify we have trades
            assert len(trades) > 0
            
            # Verify trades are consistent with optimization
            for ticker, shares in trades.items():
                # If target weight > current weight, we should buy
                if optimization_result["optimal_weights"].get(ticker, 0) > current_weights.get(ticker, 0):
                    assert shares > 0
                # If target weight < current weight, we should sell
                elif optimization_result["optimal_weights"].get(ticker, 0) < current_weights.get(ticker, 0):
                    assert shares < 0
