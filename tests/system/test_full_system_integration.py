import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main system components
from backend.central_logic_engine import CentralLogicEngine
from backend.market_data import MarketDataManager
from backend.strategy_engine import StrategyEngine
from backend.ai.gemma_quantitative_analysis import GemmaQuantitativeAnalysis
from backend.risk_management.core import RiskManager
from backend.market_scanner import MarketScanner
from backend.nlp.news_trading import NewsAnalyzer
from backend.distributed.backtesting import DistributedBacktester
from backend.performance_tracking import PerformanceTracker

class TestFullSystemIntegration:
    """
    System tests for the full Gemma Advanced Trading System.
    """
    
    @pytest.fixture
    def mock_gemma_model(self):
        """Create a mock Gemma model for testing."""
        mock_model = MagicMock()
        
        # Configure the mock to return sample responses for various methods
        mock_model.analyze_market_data.return_value = {
            "market_regime": "bullish",
            "confidence": 0.85,
            "key_factors": ["strong momentum", "positive breadth", "supportive macro"]
        }
        
        mock_model.analyze_correlations.return_value = {
            "correlation_matrix": pd.DataFrame({
                "SPY": [1.0, 0.8, 0.6, -0.3],
                "QQQ": [0.8, 1.0, 0.7, -0.4],
                "IWM": [0.6, 0.7, 1.0, -0.2],
                "TLT": [-0.3, -0.4, -0.2, 1.0]
            }, index=["SPY", "QQQ", "IWM", "TLT"]),
            "insights": ["Tech and small caps highly correlated", "Bonds showing negative correlation to equities"]
        }
        
        mock_model.generate_trading_strategy.return_value = {
            "strategy_name": "Momentum Breakout",
            "description": "Identifies stocks breaking out of consolidation patterns with strong volume",
            "market_regime": "bullish",
            "indicators": [
                {"name": "SMA", "period": 20},
                {"name": "RSI", "period": 14},
                {"name": "Volume SMA", "period": 50}
            ],
            "entry_conditions": [
                "price > SMA(20)",
                "RSI(14) > 50",
                "volume > Volume SMA(50) * 1.5"
            ],
            "exit_conditions": [
                "price < SMA(20)",
                "RSI(14) < 30"
            ],
            "risk_management": {
                "position_size_percent": 2.0,
                "stop_loss_percent": 5.0,
                "take_profit_percent": 15.0,
                "max_positions": 5
            }
        }
        
        mock_model.analyze_news_sentiment.return_value = {
            "sentiment_score": 0.75,
            "sentiment_label": "positive",
            "confidence": 0.85,
            "key_phrases": ["strong earnings", "revenue growth", "market expansion"]
        }
        
        return mock_model
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        # Create a dictionary of DataFrames for different assets
        data = {}
        
        # Generate sample data for multiple stocks
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY", "QQQ", "IWM", "TLT"]
        
        for symbol in symbols:
            # Use symbol hash for reproducible randomness
            np.random.seed(hash(symbol) % 2**32)
            
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            
            # Base price varies by symbol
            base_price = 100 + hash(symbol) % 400
            
            # Generate price data with some randomness
            close_prices = base_price + np.random.normal(0, base_price * 0.01, 100).cumsum()
            
            # Add some trend based on the symbol to create variety
            if symbol in ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]:
                # Uptrend
                close_prices += np.linspace(0, base_price * 0.2, 100)
            elif symbol in ["AMZN", "IWM"]:
                # Downtrend
                close_prices += np.linspace(base_price * 0.1, -base_price * 0.1, 100)
            # META and TLT remain random walk
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': close_prices - np.random.uniform(0, base_price * 0.01, 100),
                'high': close_prices + np.random.uniform(0, base_price * 0.01, 100),
                'low': close_prices - np.random.uniform(0, base_price * 0.01, 100),
                'close': close_prices,
                'volume': np.random.randint(base_price * 100, base_price * 1000, 100)
            }, index=dates)
            
            data[symbol] = df
        
        return data
    
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
    def central_logic_engine(self, mock_gemma_model, sample_market_data):
        """Create a CentralLogicEngine instance for testing."""
        # Create mock dependencies
        market_data_manager = MagicMock(spec=MarketDataManager)
        market_data_manager.get_historical_data.return_value = sample_market_data
        market_data_manager.get_current_prices.return_value = {
            symbol: data['close'].iloc[-1] for symbol, data in sample_market_data.items()
        }
        
        strategy_engine = MagicMock(spec=StrategyEngine)
        gemma_analysis = GemmaQuantitativeAnalysis(model=mock_gemma_model)
        risk_manager = MagicMock(spec=RiskManager)
        market_scanner = MagicMock(spec=MarketScanner)
        news_analyzer = NewsAnalyzer(model=mock_gemma_model)
        backtester = MagicMock(spec=DistributedBacktester)
        performance_tracker = MagicMock(spec=PerformanceTracker)
        
        # Create central logic engine
        engine = CentralLogicEngine(
            market_data_manager=market_data_manager,
            strategy_engine=strategy_engine,
            gemma_analysis=gemma_analysis,
            risk_manager=risk_manager,
            market_scanner=market_scanner,
            news_analyzer=news_analyzer,
            backtester=backtester,
            performance_tracker=performance_tracker
        )
        
        return engine
    
    def test_end_to_end_trading_workflow(self, central_logic_engine, sample_market_data, sample_news_data):
        """Test the complete end-to-end trading workflow."""
        # Mock the necessary methods
        with patch.object(central_logic_engine.market_data_manager, 'get_historical_data') as mock_get_data, \
             patch.object(central_logic_engine.market_data_manager, 'get_current_prices') as mock_get_prices, \
             patch.object(central_logic_engine.strategy_engine, 'register_strategy') as mock_register, \
             patch.object(central_logic_engine.strategy_engine, 'execute_strategy') as mock_execute, \
             patch.object(central_logic_engine.risk_manager, 'check_risk_limits') as mock_check_risk, \
             patch.object(central_logic_engine.risk_manager, 'adjust_position_size') as mock_adjust_size, \
             patch.object(central_logic_engine.market_scanner, 'execute_scan') as mock_scan, \
             patch.object(central_logic_engine.backtester, 'run_backtest') as mock_backtest, \
             patch.object(central_logic_engine.performance_tracker, 'add_backtest_result') as mock_track:
            
            # Configure mocks
            mock_get_data.return_value = sample_market_data
            mock_get_prices.return_value = {
                symbol: data['close'].iloc[-1] for symbol, data in sample_market_data.items()
            }
            
            mock_scan.return_value = MagicMock(
                matching_symbols=["AAPL", "MSFT", "GOOGL"]
            )
            
            mock_check_risk.return_value = {
                "within_limits": True,
                "portfolio_risk": 4.5,
                "max_portfolio_risk": 5.0,
                "position_risks": {
                    "AAPL": 1.5,
                    "MSFT": 1.2,
                    "GOOGL": 1.8
                }
            }
            
            mock_adjust_size.return_value = 100  # Adjusted position size
            
            mock_execute.return_value = {
                "action": "buy",
                "symbol": "AAPL",
                "quantity": 100,
                "price": 155.0,
                "timestamp": pd.Timestamp("2023-03-15 10:30:00"),
                "strategy_id": "AI_Generated_Strategy_1"
            }
            
            mock_backtest.return_value = {
                "trades": [
                    {"entry_date": "2023-01-15", "exit_date": "2023-02-01", "return": 0.08}
                ],
                "performance_metrics": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": 0.05
                }
            }
            
            # 1. Initialize the system
            central_logic_engine.initialize()
            
            # 2. Analyze market regime
            market_regime = central_logic_engine.analyze_market_regime()
            assert market_regime["market_regime"] == "bullish"
            
            # 3. Generate AI strategy
            ai_strategy = central_logic_engine.generate_ai_strategy(
                strategy_type="momentum",
                risk_profile="moderate"
            )
            assert ai_strategy["strategy_name"] == "Momentum Breakout"
            
            # 4. Register the strategy
            central_logic_engine.register_strategy(ai_strategy)
            mock_register.assert_called_once()
            
            # 5. Scan for trading opportunities
            scan_result = central_logic_engine.scan_for_opportunities()
            assert "AAPL" in scan_result.matching_symbols
            mock_scan.assert_called_once()
            
            # 6. Analyze news for potential trades
            news_analysis = central_logic_engine.analyze_news(sample_news_data)
            assert len(news_analysis) > 0
            assert "AAPL" in news_analysis
            
            # 7. Check risk limits
            risk_status = central_logic_engine.check_risk_limits()
            assert risk_status["within_limits"] == True
            mock_check_risk.assert_called_once()
            
            # 8. Execute trading strategy
            trade_result = central_logic_engine.execute_strategy("AI_Generated_Strategy_1", "AAPL")
            assert trade_result["action"] == "buy"
            assert trade_result["symbol"] == "AAPL"
            mock_execute.assert_called_once()
            
            # 9. Backtest strategy
            backtest_result = central_logic_engine.backtest_strategy(
                strategy_id="AI_Generated_Strategy_1",
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-03-01"
            )
            assert backtest_result["performance_metrics"]["total_return"] == 0.15
            mock_backtest.assert_called_once()
            
            # 10. Track performance
            central_logic_engine.track_performance("AI_Generated_Strategy_1", backtest_result)
            mock_track.assert_called_once()
    
    def test_market_regime_based_strategy_adaptation(self, central_logic_engine, mock_gemma_model):
        """Test adapting strategies based on market regime changes."""
        # Mock the analyze_market_data method to return different regimes
        with patch.object(mock_gemma_model, 'analyze_market_data') as mock_analyze, \
             patch.object(central_logic_engine.strategy_engine, 'register_strategy') as mock_register, \
             patch.object(central_logic_engine.strategy_engine, 'update_strategy') as mock_update:
            
            # First return bullish regime
            mock_analyze.return_value = {
                "market_regime": "bullish",
                "confidence": 0.85,
                "key_factors": ["strong momentum", "positive breadth", "supportive macro"]
            }
            
            # Analyze market regime (bullish)
            bullish_regime = central_logic_engine.analyze_market_regime()
            assert bullish_regime["market_regime"] == "bullish"
            
            # Generate strategy for bullish regime
            bullish_strategy = central_logic_engine.generate_ai_strategy(
                strategy_type="momentum",
                risk_profile="moderate"
            )
            
            # Register the strategy
            central_logic_engine.register_strategy(bullish_strategy)
            mock_register.assert_called_once()
            
            # Now change to bearish regime
            mock_analyze.return_value = {
                "market_regime": "bearish",
                "confidence": 0.80,
                "key_factors": ["negative momentum", "deteriorating breadth", "economic concerns"]
            }
            
            # Analyze market regime again (now bearish)
            bearish_regime = central_logic_engine.analyze_market_regime()
            assert bearish_regime["market_regime"] == "bearish"
            
            # Adapt strategy for bearish regime
            central_logic_engine.adapt_strategies_to_market_regime(bearish_regime)
            
            # Verify strategy was updated
            mock_update.assert_called_once()
    
    def test_risk_management_integration(self, central_logic_engine):
        """Test risk management integration with trading decisions."""
        # Mock the necessary methods
        with patch.object(central_logic_engine.risk_manager, 'check_risk_limits') as mock_check_risk, \
             patch.object(central_logic_engine.risk_manager, 'adjust_position_size') as mock_adjust_size, \
             patch.object(central_logic_engine.strategy_engine, 'execute_strategy') as mock_execute:
            
            # Configure mocks for different scenarios
            
            # Scenario 1: Within risk limits
            mock_check_risk.return_value = {
                "within_limits": True,
                "portfolio_risk": 4.5,
                "max_portfolio_risk": 5.0,
                "position_risks": {
                    "AAPL": 1.5,
                    "MSFT": 1.2,
                    "GOOGL": 1.8
                }
            }
            
            mock_adjust_size.return_value = 100  # Adjusted position size
            
            mock_execute.return_value = {
                "action": "buy",
                "symbol": "AAPL",
                "quantity": 100,
                "price": 155.0,
                "timestamp": pd.Timestamp("2023-03-15 10:30:00"),
                "strategy_id": "AI_Generated_Strategy_1"
            }
            
            # Execute trade with risk management
            trade_result = central_logic_engine.execute_with_risk_management(
                strategy_id="AI_Generated_Strategy_1",
                symbol="AAPL",
                action="buy",
                quantity=150,  # Original quantity before risk adjustment
                price=155.0
            )
            
            # Verify risk was checked and position size was adjusted
            mock_check_risk.assert_called_once()
            mock_adjust_size.assert_called_once()
            assert trade_result["quantity"] == 100  # Adjusted quantity
            
            # Reset mocks
            mock_check_risk.reset_mock()
            mock_adjust_size.reset_mock()
            mock_execute.reset_mock()
            
            # Scenario 2: Exceeding risk limits
            mock_check_risk.return_value = {
                "within_limits": False,
                "portfolio_risk": 5.5,
                "max_portfolio_risk": 5.0,
                "position_risks": {
                    "AAPL": 1.5,
                    "MSFT": 1.2,
                    "GOOGL": 2.8  # Exceeding individual position risk
                }
            }
            
            # Execute trade with risk management (should be rejected)
            with pytest.raises(Exception) as excinfo:
                central_logic_engine.execute_with_risk_management(
                    strategy_id="AI_Generated_Strategy_1",
                    symbol="GOOGL",
                    action="buy",
                    quantity=100,
                    price=2050.0
                )
            
            assert "Risk limits exceeded" in str(excinfo.value)
            mock_check_risk.assert_called_once()
            mock_execute.assert_not_called()  # Trade should not be executed
    
    def test_news_based_trading_integration(self, central_logic_engine, sample_news_data):
        """Test integration of news analysis with trading decisions."""
        # Mock the necessary methods
        with patch.object(central_logic_engine.news_analyzer, 'analyze_sentiment') as mock_sentiment, \
             patch.object(central_logic_engine.news_analyzer, 'predict_impact') as mock_impact, \
             patch.object(central_logic_engine.news_analyzer, 'generate_trading_signal') as mock_signal, \
             patch.object(central_logic_engine.strategy_engine, 'execute_strategy') as mock_execute, \
             patch.object(central_logic_engine.risk_manager, 'check_risk_limits') as mock_check_risk:
            
            # Configure mocks
            mock_sentiment.return_value = {
                "sentiment_score": 0.75,
                "sentiment_label": "positive",
                "confidence": 0.85,
                "key_phrases": ["strong earnings", "revenue growth", "market expansion"]
            }
            
            mock_impact.return_value = {
                "price_impact": 0.03,  # 3% expected price movement
                "volume_impact": 0.5,   # 50% expected volume increase
                "volatility_impact": 0.2,  # 20% expected volatility increase
                "duration": "short-term",  # Impact expected to be short-term
                "confidence": 0.7
            }
            
            mock_signal.return_value = {
                "action": "buy",
                "ticker": "AAPL",
                "confidence": 0.8,
                "time_horizon": "short-term",
                "entry_price_range": {"min": 150.0, "max": 155.0},
                "target_price": 165.0,
                "stop_loss": 145.0,
                "position_size_recommendation": "moderate"
            }
            
            mock_check_risk.return_value = {
                "within_limits": True,
                "portfolio_risk": 4.5,
                "max_portfolio_risk": 5.0
            }
            
            mock_execute.return_value = {
                "action": "buy",
                "symbol": "AAPL",
                "quantity": 100,
                "price": 155.0,
                "timestamp": pd.Timestamp("2023-03-15 10:30:00"),
                "strategy_id": "News_Based_Strategy"
            }
            
            # Process news and generate trading signals
            news_signals = central_logic_engine.process_news_for_trading(sample_news_data)
            
            # Verify signals were generated
            assert len(news_signals) > 0
            
            # Execute a news-based trade
            trade_result = central_logic_engine.execute_news_based_trade(
                news_signals[0],
                current_price=155.0
            )
            
            # Verify trade was executed
            assert trade_result["action"] == "buy"
            assert trade_result["symbol"] == "AAPL"
            mock_execute.assert_called_once()
    
    def test_performance_analysis_integration(self, central_logic_engine):
        """Test integration of performance analysis with trading system."""
        # Mock the necessary methods
        with patch.object(central_logic_engine.performance_tracker, 'get_performance_metrics') as mock_metrics, \
             patch.object(central_logic_engine.performance_tracker, 'get_equity_curve') as mock_equity, \
             patch.object(central_logic_engine.performance_tracker, 'get_drawdowns') as mock_drawdowns, \
             patch.object(central_logic_engine.performance_tracker, 'compare_strategies') as mock_compare:
            
            # Configure mocks
            mock_metrics.return_value = {
                "total_return": 0.15,
                "annualized_return": 0.65,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.2,
                "max_drawdown": 0.05,
                "win_rate": 0.75
            }
            
            mock_equity.return_value = pd.Series([10000, 10200, 10400, 10600, 10800, 11000, 11500], 
                                               index=pd.date_range(start='2023-01-01', periods=7, freq='W'))
            
            mock_drawdowns.return_value = [
                {"start_date": "2023-02-15", "end_date": "2023-02-20", "recovery_date": "2023-02-25", 
                 "depth": 0.03, "length": 5, "recovery_time": 5}
            ]
            
            mock_compare.return_value = [
                ("AI_Generated_Strategy_1", 0.15),
                ("News_Based_Strategy", 0.12),
                ("Technical_Strategy", 0.08)
            ]
            
            # Get performance metrics
            metrics = central_logic_engine.get_strategy_performance("AI_Generated_Strategy_1")
            assert metrics["total_return"] == 0.15
            assert metrics["sharpe_ratio"] == 1.5
            
            # Get equity curve
            equity_curve = central_logic_engine.get_strategy_equity_curve("AI_Generated_Strategy_1")
            assert len(equity_curve) == 7
            assert equity_curve.iloc[-1] == 11500
            
            # Get drawdowns
            drawdowns = central_logic_engine.get_strategy_drawdowns("AI_Generated_Strategy_1")
            assert len(drawdowns) == 1
            assert drawdowns[0]["depth"] == 0.03
            
            # Compare strategies
            comparison = central_logic_engine.compare_strategy_performance()
            assert len(comparison) == 3
            assert comparison[0][0] == "AI_Generated_Strategy_1"  # Best strategy
            assert comparison[0][1] == 0.15  # Best return
