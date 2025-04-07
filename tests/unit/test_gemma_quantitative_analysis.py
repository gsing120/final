import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.ai.gemma_quantitative_analysis import (
    GemmaQuantitativeAnalysis,
    MarketRegimeAnalysis,
    CorrelationAnalysis,
    FactorAnalysis
)

class TestGemmaQuantitativeAnalysis:
    """
    Unit tests for the Gemma Quantitative Analysis module.
    """
    
    @pytest.fixture
    def mock_gemma_model(self):
        """Create a mock Gemma model for testing."""
        mock_model = MagicMock()
        
        # Configure the mock to return sample responses
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
        
        mock_model.analyze_factors.return_value = {
            "factor_exposures": {
                "value": 0.2,
                "growth": 0.5,
                "momentum": 0.7,
                "quality": 0.4,
                "size": -0.1,
                "volatility": -0.3
            },
            "factor_performance": {
                "value": -0.5,
                "growth": 1.2,
                "momentum": 0.8,
                "quality": 0.3,
                "size": -0.2,
                "volatility": -0.4
            },
            "recommendations": ["Reduce value exposure", "Maintain growth and momentum exposure"]
        }
        
        return mock_model
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        # Create a dictionary of DataFrames for different assets
        data = {}
        
        # Generate sample data for SPY
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        spy_data = pd.DataFrame({
            'open': np.linspace(400, 450, 100) + np.random.normal(0, 2, 100),
            'high': np.linspace(405, 455, 100) + np.random.normal(0, 2, 100),
            'low': np.linspace(395, 445, 100) + np.random.normal(0, 2, 100),
            'close': np.linspace(400, 450, 100) + np.random.normal(0, 2, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        data['SPY'] = spy_data
        
        # Generate sample data for QQQ (more volatile)
        qqq_data = pd.DataFrame({
            'open': np.linspace(300, 380, 100) + np.random.normal(0, 4, 100),
            'high': np.linspace(305, 385, 100) + np.random.normal(0, 4, 100),
            'low': np.linspace(295, 375, 100) + np.random.normal(0, 4, 100),
            'close': np.linspace(300, 380, 100) + np.random.normal(0, 4, 100),
            'volume': np.random.randint(2000000, 15000000, 100)
        }, index=dates)
        data['QQQ'] = qqq_data
        
        # Generate sample data for IWM (small caps)
        iwm_data = pd.DataFrame({
            'open': np.linspace(180, 200, 100) + np.random.normal(0, 3, 100),
            'high': np.linspace(185, 205, 100) + np.random.normal(0, 3, 100),
            'low': np.linspace(175, 195, 100) + np.random.normal(0, 3, 100),
            'close': np.linspace(180, 200, 100) + np.random.normal(0, 3, 100),
            'volume': np.random.randint(1500000, 12000000, 100)
        }, index=dates)
        data['IWM'] = iwm_data
        
        # Generate sample data for TLT (bonds, negative correlation)
        tlt_data = pd.DataFrame({
            'open': np.linspace(100, 90, 100) + np.random.normal(0, 1, 100),
            'high': np.linspace(102, 92, 100) + np.random.normal(0, 1, 100),
            'low': np.linspace(98, 88, 100) + np.random.normal(0, 1, 100),
            'close': np.linspace(100, 90, 100) + np.random.normal(0, 1, 100),
            'volume': np.random.randint(1000000, 8000000, 100)
        }, index=dates)
        data['TLT'] = tlt_data
        
        return data
    
    def test_gemma_quantitative_analysis_initialization(self, mock_gemma_model):
        """Test initialization of GemmaQuantitativeAnalysis class."""
        # Initialize the class
        gemma_quant = GemmaQuantitativeAnalysis(model=mock_gemma_model)
        
        # Verify initialization
        assert gemma_quant.model == mock_gemma_model
        assert isinstance(gemma_quant.market_regime_analyzer, MarketRegimeAnalysis)
        assert isinstance(gemma_quant.correlation_analyzer, CorrelationAnalysis)
        assert isinstance(gemma_quant.factor_analyzer, FactorAnalysis)
    
    def test_market_regime_analysis(self, mock_gemma_model, sample_market_data):
        """Test market regime analysis functionality."""
        # Initialize the class
        gemma_quant = GemmaQuantitativeAnalysis(model=mock_gemma_model)
        
        # Perform market regime analysis
        regime_analysis = gemma_quant.analyze_market_regime(sample_market_data)
        
        # Verify the result
        assert isinstance(regime_analysis, dict)
        assert "market_regime" in regime_analysis
        assert "confidence" in regime_analysis
        assert "key_factors" in regime_analysis
        assert "timestamp" in regime_analysis
        
        assert regime_analysis["market_regime"] == "bullish"
        assert regime_analysis["confidence"] == 0.85
        assert len(regime_analysis["key_factors"]) == 3
        
        # Verify the model was called with the correct data
        mock_gemma_model.analyze_market_data.assert_called_once()
        
        # Test with specific time period
        start_date = "2023-01-15"
        end_date = "2023-01-30"
        
        regime_analysis_period = gemma_quant.analyze_market_regime(
            sample_market_data,
            start_date=start_date,
            end_date=end_date
        )
        
        assert regime_analysis_period["market_regime"] == "bullish"
        
        # Verify the model was called with filtered data
        assert mock_gemma_model.analyze_market_data.call_count == 2
    
    def test_correlation_analysis(self, mock_gemma_model, sample_market_data):
        """Test correlation analysis functionality."""
        # Initialize the class
        gemma_quant = GemmaQuantitativeAnalysis(model=mock_gemma_model)
        
        # Perform correlation analysis
        correlation_analysis = gemma_quant.analyze_correlations(sample_market_data)
        
        # Verify the result
        assert isinstance(correlation_analysis, dict)
        assert "correlation_matrix" in correlation_analysis
        assert "insights" in correlation_analysis
        assert "timestamp" in correlation_analysis
        
        assert isinstance(correlation_analysis["correlation_matrix"], pd.DataFrame)
        assert correlation_analysis["correlation_matrix"].shape == (4, 4)
        assert len(correlation_analysis["insights"]) == 2
        
        # Verify the model was called with the correct data
        mock_gemma_model.analyze_correlations.assert_called_once()
        
        # Test with specific assets
        assets = ["SPY", "QQQ"]
        
        correlation_analysis_subset = gemma_quant.analyze_correlations(
            sample_market_data,
            assets=assets
        )
        
        assert correlation_analysis_subset["correlation_matrix"].shape == (4, 4)
        
        # Verify the model was called with filtered data
        assert mock_gemma_model.analyze_correlations.call_count == 2
    
    def test_factor_analysis(self, mock_gemma_model, sample_market_data):
        """Test factor analysis functionality."""
        # Initialize the class
        gemma_quant = GemmaQuantitativeAnalysis(model=mock_gemma_model)
        
        # Perform factor analysis
        factor_analysis = gemma_quant.analyze_factors(sample_market_data)
        
        # Verify the result
        assert isinstance(factor_analysis, dict)
        assert "factor_exposures" in factor_analysis
        assert "factor_performance" in factor_analysis
        assert "recommendations" in factor_analysis
        assert "timestamp" in factor_analysis
        
        assert isinstance(factor_analysis["factor_exposures"], dict)
        assert len(factor_analysis["factor_exposures"]) == 6
        assert len(factor_analysis["recommendations"]) == 2
        
        # Verify the model was called with the correct data
        mock_gemma_model.analyze_factors.assert_called_once()
        
        # Test with specific factors
        factors = ["value", "growth", "momentum"]
        
        factor_analysis_subset = gemma_quant.analyze_factors(
            sample_market_data,
            factors=factors
        )
        
        assert len(factor_analysis_subset["factor_exposures"]) == 6
        
        # Verify the model was called with filtered data
        assert mock_gemma_model.analyze_factors.call_count == 2
    
    def test_portfolio_optimization(self, mock_gemma_model, sample_market_data):
        """Test portfolio optimization functionality."""
        # Initialize the class
        gemma_quant = GemmaQuantitativeAnalysis(model=mock_gemma_model)
        
        # Configure mock for portfolio optimization
        mock_gemma_model.optimize_portfolio.return_value = {
            "optimal_weights": {
                "SPY": 0.4,
                "QQQ": 0.3,
                "IWM": 0.2,
                "TLT": 0.1
            },
            "expected_return": 0.12,
            "expected_volatility": 0.15,
            "sharpe_ratio": 0.8,
            "efficient_frontier": [
                {"return": 0.08, "volatility": 0.10, "weights": {"SPY": 0.2, "QQQ": 0.1, "IWM": 0.1, "TLT": 0.6}},
                {"return": 0.10, "volatility": 0.12, "weights": {"SPY": 0.3, "QQQ": 0.2, "IWM": 0.1, "TLT": 0.4}},
                {"return": 0.12, "volatility": 0.15, "weights": {"SPY": 0.4, "QQQ": 0.3, "IWM": 0.2, "TLT": 0.1}},
                {"return": 0.14, "volatility": 0.20, "weights": {"SPY": 0.5, "QQQ": 0.4, "IWM": 0.1, "TLT": 0.0}}
            ]
        }
        
        # Perform portfolio optimization
        optimization_result = gemma_quant.optimize_portfolio(
            sample_market_data,
            risk_tolerance=0.5,
            constraints={"min_weight": 0.05, "max_weight": 0.5}
        )
        
        # Verify the result
        assert isinstance(optimization_result, dict)
        assert "optimal_weights" in optimization_result
        assert "expected_return" in optimization_result
        assert "expected_volatility" in optimization_result
        assert "sharpe_ratio" in optimization_result
        assert "efficient_frontier" in optimization_result
        assert "timestamp" in optimization_result
        
        assert isinstance(optimization_result["optimal_weights"], dict)
        assert len(optimization_result["optimal_weights"]) == 4
        assert len(optimization_result["efficient_frontier"]) == 4
        
        # Verify the model was called with the correct parameters
        mock_gemma_model.optimize_portfolio.assert_called_once_with(
            market_data=sample_market_data,
            risk_tolerance=0.5,
            constraints={"min_weight": 0.05, "max_weight": 0.5}
        )
    
    def test_strategy_generation(self, mock_gemma_model, sample_market_data):
        """Test trading strategy generation functionality."""
        # Initialize the class
        gemma_quant = GemmaQuantitativeAnalysis(model=mock_gemma_model)
        
        # Configure mock for strategy generation
        mock_gemma_model.generate_trading_strategy.return_value = {
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
        
        # Perform strategy generation
        strategy = gemma_quant.generate_trading_strategy(
            sample_market_data,
            strategy_type="momentum",
            risk_profile="moderate"
        )
        
        # Verify the result
        assert isinstance(strategy, dict)
        assert "strategy_name" in strategy
        assert "description" in strategy
        assert "market_regime" in strategy
        assert "indicators" in strategy
        assert "entry_conditions" in strategy
        assert "exit_conditions" in strategy
        assert "risk_management" in strategy
        assert "timestamp" in strategy
        
        assert strategy["strategy_name"] == "Momentum Breakout"
        assert len(strategy["indicators"]) == 3
        assert len(strategy["entry_conditions"]) == 3
        assert len(strategy["exit_conditions"]) == 2
        
        # Verify the model was called with the correct parameters
        mock_gemma_model.generate_trading_strategy.assert_called_once_with(
            market_data=sample_market_data,
            strategy_type="momentum",
            risk_profile="moderate"
        )
