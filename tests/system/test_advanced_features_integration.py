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
from backend.paper_trading_mode import PaperTradingEngine
from backend.alert_system import AlertSystem

class TestAdvancedFeaturesIntegration:
    """
    System tests for advanced features of the Gemma Advanced Trading System.
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
        news_analyzer = MagicMock(spec=NewsAnalyzer)
        backtester = MagicMock(spec=DistributedBacktester)
        performance_tracker = MagicMock(spec=PerformanceTracker)
        paper_trading_engine = MagicMock(spec=PaperTradingEngine)
        alert_system = MagicMock(spec=AlertSystem)
        
        # Create central logic engine
        engine = CentralLogicEngine(
            market_data_manager=market_data_manager,
            strategy_engine=strategy_engine,
            gemma_analysis=gemma_analysis,
            risk_manager=risk_manager,
            market_scanner=market_scanner,
            news_analyzer=news_analyzer,
            backtester=backtester,
            performance_tracker=performance_tracker,
            paper_trading_engine=paper_trading_engine,
            alert_system=alert_system
        )
        
        return engine
    
    @pytest.fixture
    def paper_trading_engine(self, sample_market_data):
        """Create a PaperTradingEngine instance for testing."""
        # Create mock dependencies
        market_data_manager = MagicMock(spec=MarketDataManager)
        market_data_manager.get_historical_data.return_value = sample_market_data
        market_data_manager.get_current_prices.return_value = {
            symbol: data['close'].iloc[-1] for symbol, data in sample_market_data.items()
        }
        
        strategy_engine = MagicMock(spec=StrategyEngine)
        risk_manager = MagicMock(spec=RiskManager)
        
        # Create paper trading engine
        engine = PaperTradingEngine(
            market_data_manager=market_data_manager,
            strategy_engine=strategy_engine,
            risk_manager=risk_manager,
            initial_capital=100000.0
        )
        
        return engine
    
    @pytest.fixture
    def alert_system(self):
        """Create an AlertSystem instance for testing."""
        return AlertSystem()
    
    def test_paper_trading_integration(self, central_logic_engine, paper_trading_engine):
        """Test paper trading integration with the main system."""
        # Mock the necessary methods
        with patch.object(central_logic_engine.paper_trading_engine, 'initialize') as mock_init, \
             patch.object(central_logic_engine.paper_trading_engine, 'execute_trade') as mock_execute, \
             patch.object(central_logic_engine.paper_trading_engine, 'get_portfolio_status') as mock_status, \
             patch.object(central_logic_engine.paper_trading_engine, 'get_trade_history') as mock_history:
            
            # Configure mocks
            mock_status.return_value = {
                "total_value": 105000.0,
                "cash": 50000.0,
                "equity": 55000.0,
                "positions": [
                    {"symbol": "AAPL", "quantity": 100, "entry_price": 150.0, "current_price": 155.0, "market_value": 15500.0, "unrealized_pnl": 500.0},
                    {"symbol": "MSFT", "quantity": 50, "entry_price": 250.0, "current_price": 260.0, "market_value": 13000.0, "unrealized_pnl": 500.0},
                    {"symbol": "GOOGL", "quantity": 10, "entry_price": 2500.0, "current_price": 2650.0, "market_value": 26500.0, "unrealized_pnl": 1500.0}
                ],
                "performance": {
                    "total_return": 0.05,
                    "daily_pnl": 1200.0,
                    "open_positions": 3
                }
            }
            
            mock_history.return_value = [
                {"timestamp": pd.Timestamp("2023-03-01 10:30:00"), "action": "buy", "symbol": "AAPL", "quantity": 100, "price": 150.0, "strategy_id": "AI_Strategy_1"},
                {"timestamp": pd.Timestamp("2023-03-02 11:15:00"), "action": "buy", "symbol": "MSFT", "quantity": 50, "price": 250.0, "strategy_id": "AI_Strategy_1"},
                {"timestamp": pd.Timestamp("2023-03-03 14:45:00"), "action": "buy", "symbol": "GOOGL", "quantity": 10, "price": 2500.0, "strategy_id": "AI_Strategy_2"}
            ]
            
            mock_execute.return_value = {
                "success": True,
                "trade_id": "trade_004",
                "timestamp": pd.Timestamp("2023-03-15 10:30:00"),
                "action": "buy",
                "symbol": "SPY",
                "quantity": 20,
                "price": 420.0,
                "strategy_id": "AI_Strategy_3",
                "status": "executed"
            }
            
            # Initialize paper trading
            central_logic_engine.initialize_paper_trading(initial_capital=100000.0)
            mock_init.assert_called_once_with(initial_capital=100000.0)
            
            # Execute paper trade
            trade_result = central_logic_engine.execute_paper_trade(
                strategy_id="AI_Strategy_3",
                symbol="SPY",
                action="buy",
                quantity=20,
                price=420.0
            )
            
            assert trade_result["success"] == True
            assert trade_result["symbol"] == "SPY"
            mock_execute.assert_called_once()
            
            # Get paper trading portfolio status
            portfolio_status = central_logic_engine.get_paper_trading_status()
            assert portfolio_status["total_value"] == 105000.0
            assert len(portfolio_status["positions"]) == 3
            mock_status.assert_called_once()
            
            # Get paper trading history
            trade_history = central_logic_engine.get_paper_trading_history()
            assert len(trade_history) == 3
            assert trade_history[0]["symbol"] == "AAPL"
            mock_history.assert_called_once()
    
    def test_alert_system_integration(self, central_logic_engine, alert_system):
        """Test alert system integration with the main system."""
        # Mock the necessary methods
        with patch.object(central_logic_engine.alert_system, 'add_price_alert') as mock_add_price, \
             patch.object(central_logic_engine.alert_system, 'add_technical_alert') as mock_add_technical, \
             patch.object(central_logic_engine.alert_system, 'add_news_alert') as mock_add_news, \
             patch.object(central_logic_engine.alert_system, 'check_alerts') as mock_check, \
             patch.object(central_logic_engine.alert_system, 'get_triggered_alerts') as mock_get_triggered:
            
            # Configure mocks
            mock_add_price.return_value = "alert_001"
            mock_add_technical.return_value = "alert_002"
            mock_add_news.return_value = "alert_003"
            
            mock_check.return_value = ["alert_001", "alert_002"]
            
            mock_get_triggered.return_value = [
                {"alert_id": "alert_001", "type": "price", "symbol": "AAPL", "condition": "above", "target": 160.0, "current_value": 162.5, "triggered_at": pd.Timestamp("2023-03-15 10:30:00")},
                {"alert_id": "alert_002", "type": "technical", "symbol": "MSFT", "indicator": "RSI", "condition": "above", "target": 70.0, "current_value": 75.3, "triggered_at": pd.Timestamp("2023-03-15 10:35:00")}
            ]
            
            # Add price alert
            alert_id = central_logic_engine.add_price_alert(
                symbol="AAPL",
                condition="above",
                target_price=160.0
            )
            assert alert_id == "alert_001"
            mock_add_price.assert_called_once()
            
            # Add technical alert
            alert_id = central_logic_engine.add_technical_alert(
                symbol="MSFT",
                indicator="RSI",
                condition="above",
                target_value=70.0,
                params={"period": 14}
            )
            assert alert_id == "alert_002"
            mock_add_technical.assert_called_once()
            
            # Add news alert
            alert_id = central_logic_engine.add_news_alert(
                keywords=["earnings", "beat", "Apple"],
                sentiment="positive",
                min_confidence=0.7
            )
            assert alert_id == "alert_003"
            mock_add_news.assert_called_once()
            
            # Check alerts
            triggered_alerts = central_logic_engine.check_all_alerts()
            assert len(triggered_alerts) == 2
            mock_check.assert_called_once()
            
            # Get triggered alerts
            alert_details = central_logic_engine.get_triggered_alert_details()
            assert len(alert_details) == 2
            assert alert_details[0]["symbol"] == "AAPL"
            assert alert_details[1]["indicator"] == "RSI"
            mock_get_triggered.assert_called_once()
    
    def test_automated_strategy_generation(self, central_logic_engine, mock_gemma_model):
        """Test automated strategy generation based on market conditions."""
        # Mock the necessary methods
        with patch.object(mock_gemma_model, 'analyze_market_data') as mock_analyze, \
             patch.object(mock_gemma_model, 'generate_trading_strategy') as mock_generate, \
             patch.object(central_logic_engine.strategy_engine, 'register_strategy') as mock_register, \
             patch.object(central_logic_engine.backtester, 'run_backtest') as mock_backtest:
            
            # Configure mocks
            mock_analyze.return_value = {
                "market_regime": "bullish",
                "confidence": 0.85,
                "key_factors": ["strong momentum", "positive breadth", "supportive macro"],
                "sector_performance": {
                    "Technology": 0.12,
                    "Healthcare": 0.08,
                    "Financials": 0.05,
                    "Energy": -0.03,
                    "Utilities": -0.02
                },
                "volatility_regime": "moderate",
                "correlation_regime": "normal"
            }
            
            # Configure mock to return different strategies based on input
            def generate_strategy_side_effect(market_data, strategy_type, risk_profile):
                if strategy_type == "momentum":
                    return {
                        "strategy_name": "AI Momentum Strategy",
                        "description": "Momentum strategy for bullish markets",
                        "indicators": [{"name": "RSI", "params": {"period": 14}}],
                        "entry_conditions": [{"type": "greater_than", "left": "RSI", "right": 50}],
                        "exit_conditions": [{"type": "less_than", "left": "RSI", "right": 30}]
                    }
                elif strategy_type == "mean_reversion":
                    return {
                        "strategy_name": "AI Mean Reversion Strategy",
                        "description": "Mean reversion strategy for overbought conditions",
                        "indicators": [{"name": "RSI", "params": {"period": 14}}],
                        "entry_conditions": [{"type": "less_than", "left": "RSI", "right": 30}],
                        "exit_conditions": [{"type": "greater_than", "left": "RSI", "right": 50}]
                    }
                else:
                    return {
                        "strategy_name": "AI Trend Following Strategy",
                        "description": "Trend following strategy using moving averages",
                        "indicators": [
                            {"name": "SMA", "params": {"period": 20}},
                            {"name": "SMA", "params": {"period": 50}}
                        ],
                        "entry_conditions": [{"type": "cross_above", "left": "SMA_20", "right": "SMA_50"}],
                        "exit_conditions": [{"type": "cross_below", "left": "SMA_20", "right": "SMA_50"}]
                    }
            
            mock_generate.side_effect = generate_strategy_side_effect
            
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
            
            # Generate strategies based on market analysis
            strategies = central_logic_engine.generate_strategies_for_market_conditions()
            
            # Verify strategies were generated
            assert len(strategies) > 0
            
            # Verify different types of strategies were generated
            strategy_types = [s["strategy_name"] for s in strategies]
            assert "AI Momentum Strategy" in strategy_types
            assert "AI Mean Reversion Strategy" in strategy_types
            assert "AI Trend Following Strategy" in strategy_types
            
            # Verify strategies were registered
            assert mock_register.call_count >= 3
            
            # Verify strategies were backtested
            assert mock_backtest.call_count >= 3
    
    def test_multi_timeframe_analysis(self, central_logic_engine, sample_market_data):
        """Test multi-timeframe analysis for trading decisions."""
        # Mock the necessary methods
        with patch.object(central_logic_engine.market_data_manager, 'get_historical_data') as mock_get_data, \
             patch.object(central_logic_engine.gemma_analysis, 'analyze_market_regime') as mock_analyze:
            
            # Configure mocks
            mock_get_data.return_value = sample_market_data
            
            # Configure mock to return different analysis for different timeframes
            def analyze_regime_side_effect(data, **kwargs):
                timeframe = kwargs.get('timeframe', 'daily')
                if timeframe == 'daily':
                    return {
                        "market_regime": "bullish",
                        "confidence": 0.85,
                        "key_factors": ["strong momentum", "positive breadth"]
                    }
                elif timeframe == 'weekly':
                    return {
                        "market_regime": "neutral",
                        "confidence": 0.70,
                        "key_factors": ["consolidation", "mixed signals"]
                    }
                elif timeframe == 'monthly':
                    return {
                        "market_regime": "bullish",
                        "confidence": 0.90,
                        "key_factors": ["long-term uptrend", "economic growth"]
                    }
                else:
                    return {
                        "market_regime": "unknown",
                        "confidence": 0.50,
                        "key_factors": []
                    }
            
            mock_analyze.side_effect = analyze_regime_side_effect
            
            # Perform multi-timeframe analysis
            analysis = central_logic_engine.perform_multi_timeframe_analysis(
                symbol="SPY",
                timeframes=["daily", "weekly", "monthly"]
            )
            
            # Verify analysis for each timeframe
            assert "daily" in analysis
            assert "weekly" in analysis
            assert "monthly" in analysis
            
            assert analysis["daily"]["market_regime"] == "bullish"
            assert analysis["weekly"]["market_regime"] == "neutral"
            assert analysis["monthly"]["market_regime"] == "bullish"
            
            # Verify confidence levels
            assert analysis["daily"]["confidence"] == 0.85
            assert analysis["weekly"]["confidence"] == 0.70
            assert analysis["monthly"]["confidence"] == 0.90
            
            # Get consensus analysis
            consensus = central_logic_engine.get_multi_timeframe_consensus(analysis)
            
            # Verify consensus calculation
            assert "consensus_regime" in consensus
            assert "confidence" in consensus
            assert "alignment" in consensus
            
            # Since 2 out of 3 timeframes are bullish, consensus should be bullish
            assert consensus["consensus_regime"] == "bullish"
            
            # Alignment should be partial (not all timeframes agree)
            assert consensus["alignment"] == "partial"
    
    def test_portfolio_optimization_integration(self, central_logic_engine):
        """Test portfolio optimization integration with the main system."""
        # Mock the necessary methods
        with patch.object(central_logic_engine, 'get_portfolio_holdings') as mock_get_holdings, \
             patch.object(central_logic_engine.gemma_analysis, 'analyze_correlations') as mock_correlations, \
             patch.object(central_logic_engine.gemma_analysis, 'optimize_portfolio') as mock_optimize:
            
            # Configure mocks
            mock_get_holdings.return_value = [
                {"symbol": "AAPL", "quantity": 100, "market_value": 15500.0, "weight": 0.28},
                {"symbol": "MSFT", "quantity": 50, "market_value": 13000.0, "weight": 0.24},
                {"symbol": "GOOGL", "quantity": 10, "market_value": 26500.0, "weight": 0.48}
            ]
            
            mock_correlations.return_value = {
                "correlation_matrix": pd.DataFrame({
                    "AAPL": [1.0, 0.7, 0.6],
                    "MSFT": [0.7, 1.0, 0.7],
                    "GOOGL": [0.6, 0.7, 1.0]
                }, index=["AAPL", "MSFT", "GOOGL"]),
                "insights": ["High correlation between tech stocks"]
            }
            
            mock_optimize.return_value = {
                "optimal_weights": {
                    "AAPL": 0.25,
                    "MSFT": 0.25,
                    "GOOGL": 0.20,
                    "SPY": 0.15,
                    "TLT": 0.15
                },
                "expected_return": 0.12,
                "expected_volatility": 0.15,
                "sharpe_ratio": 0.8
            }
            
            # Run portfolio optimization
            optimization_result = central_logic_engine.optimize_current_portfolio(
                risk_tolerance=0.5,
                constraints={"min_weight": 0.05, "max_weight": 0.3}
            )
            
            # Verify optimization result
            assert "optimal_weights" in optimization_result
            assert "expected_return" in optimization_result
            assert "expected_volatility" in optimization_result
            assert "sharpe_ratio" in optimization_result
            
            # Verify weights sum to 1
            assert sum(optimization_result["optimal_weights"].values()) == pytest.approx(1.0)
            
            # Calculate rebalancing trades
            rebalancing_trades = central_logic_engine.calculate_rebalancing_trades(
                current_holdings=mock_get_holdings.return_value,
                target_weights=optimization_result["optimal_weights"],
                portfolio_value=55000.0
            )
            
            # Verify rebalancing trades
            assert isinstance(rebalancing_trades, list)
            
            # Check for new positions to add (SPY and TLT)
            spy_trade = next((t for t in rebalancing_trades if t["symbol"] == "SPY"), None)
            tlt_trade = next((t for t in rebalancing_trades if t["symbol"] == "TLT"), None)
            
            assert spy_trade is not None
            assert tlt_trade is not None
            assert spy_trade["action"] == "buy"
            assert tlt_trade["action"] == "buy"
            
            # Check for position adjustments
            googl_trade = next((t for t in rebalancing_trades if t["symbol"] == "GOOGL"), None)
            assert googl_trade is not None
            assert googl_trade["action"] == "sell"  # Should reduce from 0.48 to 0.20
