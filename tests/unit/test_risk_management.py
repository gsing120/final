import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from backend.risk_management.core import (
    Position,
    Portfolio,
    RiskManager
)

class TestRiskManagementCore:
    """
    Unit tests for the core risk management module.
    """
    
    def test_position_creation(self):
        """Test Position class creation and properties."""
        # Create a basic position
        position = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            entry_date="2023-01-15",
            stop_loss=140.0,
            take_profit=170.0,
            position_type="long"
        )
        
        # Verify basic properties
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert position.entry_date == "2023-01-15"
        assert position.stop_loss == 140.0
        assert position.take_profit == 170.0
        assert position.position_type == "long"
        assert position.exit_price is None
        assert position.exit_date is None
        assert position.status == "open"
        
        # Test position value calculation
        assert position.initial_value == 15000.0  # 100 * 150.0
        
        # Test current value calculation
        current_price = 160.0
        assert position.current_value(current_price) == 16000.0  # 100 * 160.0
        
        # Test profit/loss calculation
        assert position.unrealized_pnl(current_price) == 1000.0  # (160.0 - 150.0) * 100
        assert position.unrealized_pnl_percent(current_price) == pytest.approx(6.67, 0.01)  # (160.0 - 150.0) / 150.0 * 100
        
        # Test position closing
        position.close(exit_price=165.0, exit_date="2023-01-20")
        assert position.exit_price == 165.0
        assert position.exit_date == "2023-01-20"
        assert position.status == "closed"
        assert position.realized_pnl() == 1500.0  # (165.0 - 150.0) * 100
        assert position.realized_pnl_percent() == pytest.approx(10.0)  # (165.0 - 150.0) / 150.0 * 100
        
        # Test short position
        short_position = Position(
            symbol="MSFT",
            quantity=50,
            entry_price=250.0,
            entry_date="2023-01-15",
            stop_loss=260.0,
            take_profit=230.0,
            position_type="short"
        )
        
        assert short_position.initial_value == 12500.0  # 50 * 250.0
        
        # Test short position profit/loss calculation
        short_current_price = 240.0
        assert short_position.unrealized_pnl(short_current_price) == 500.0  # (250.0 - 240.0) * 50
        assert short_position.unrealized_pnl_percent(short_current_price) == pytest.approx(4.0)  # (250.0 - 240.0) / 250.0 * 100
        
        # Test short position closing
        short_position.close(exit_price=235.0, exit_date="2023-01-20")
        assert short_position.realized_pnl() == 750.0  # (250.0 - 235.0) * 50
        assert short_position.realized_pnl_percent() == pytest.approx(6.0)  # (250.0 - 235.0) / 250.0 * 100
    
    def test_portfolio_management(self):
        """Test Portfolio class functionality."""
        # Create a portfolio
        portfolio = Portfolio(initial_capital=100000.0)
        
        # Verify initial state
        assert portfolio.initial_capital == 100000.0
        assert portfolio.cash == 100000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.closed_positions) == 0
        
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
        
        # Verify positions were added
        assert len(portfolio.positions) == 2
        assert portfolio.positions[0].symbol == "AAPL"
        assert portfolio.positions[1].symbol == "MSFT"
        
        # Verify cash was reduced
        expected_cash = 100000.0 - (100 * 150.0) - (50 * 250.0)
        assert portfolio.cash == expected_cash
        
        # Test portfolio value calculation
        current_prices = {"AAPL": 160.0, "MSFT": 260.0}
        portfolio_value = portfolio.total_value(current_prices)
        expected_value = expected_cash + (100 * 160.0) + (50 * 260.0)
        assert portfolio_value == expected_value
        
        # Test portfolio profit/loss calculation
        portfolio_pnl = portfolio.total_pnl(current_prices)
        expected_pnl = (100 * (160.0 - 150.0)) + (50 * (260.0 - 250.0))
        assert portfolio_pnl == expected_pnl
        
        # Test closing a position
        portfolio.close_position(0, exit_price=165.0, exit_date="2023-01-20")
        assert len(portfolio.positions) == 1
        assert len(portfolio.closed_positions) == 1
        assert portfolio.positions[0].symbol == "MSFT"
        assert portfolio.closed_positions[0].symbol == "AAPL"
        
        # Verify cash was increased after closing
        expected_cash += 100 * 165.0
        assert portfolio.cash == expected_cash
        
        # Test portfolio performance metrics
        performance = portfolio.get_performance_metrics(current_prices)
        assert "total_value" in performance
        assert "total_pnl" in performance
        assert "total_pnl_percent" in performance
        assert "open_positions_value" in performance
        assert "cash" in performance
        
        # Test position sizing
        max_position_size = portfolio.calculate_position_size("GOOGL", 1500.0, risk_percent=2.0, stop_loss_percent=5.0)
        # Expected shares = (portfolio_value * risk_percent) / (entry_price * stop_loss_percent)
        expected_shares = int((portfolio_value * 0.02) / (1500.0 * 0.05))
        assert max_position_size == expected_shares
    
    def test_risk_manager(self):
        """Test RiskManager class functionality."""
        # Create a portfolio
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
        
        # Create a risk manager
        risk_manager = RiskManager(
            portfolio=portfolio,
            max_portfolio_risk=5.0,  # 5% max portfolio risk
            max_position_risk=2.0,   # 2% max position risk
            max_sector_exposure=30.0, # 30% max sector exposure
            max_correlation=0.7       # 0.7 max correlation
        )
        
        # Test risk parameters
        assert risk_manager.max_portfolio_risk == 5.0
        assert risk_manager.max_position_risk == 2.0
        assert risk_manager.max_sector_exposure == 30.0
        assert risk_manager.max_correlation == 0.7
        
        # Test position risk calculation
        position_risk = risk_manager.calculate_position_risk(
            portfolio.positions[0],
            current_price=155.0
        )
        
        # Expected risk = (entry_price - stop_loss) * quantity / portfolio_value
        expected_risk = (150.0 - 140.0) * 100 / portfolio.total_value({"AAPL": 155.0, "MSFT": 255.0})
        assert position_risk == pytest.approx(expected_risk * 100, 0.01)  # Convert to percentage
        
        # Test portfolio risk calculation
        portfolio_risk = risk_manager.calculate_portfolio_risk({"AAPL": 155.0, "MSFT": 255.0})
        assert isinstance(portfolio_risk, float)
        
        # Test risk checks
        risk_status = risk_manager.check_risk_limits({"AAPL": 155.0, "MSFT": 255.0})
        assert isinstance(risk_status, dict)
        assert "portfolio_risk_status" in risk_status
        assert "position_risk_status" in risk_status
        assert "sector_exposure_status" in risk_status
        
        # Test position sizing recommendation
        position_size = risk_manager.recommend_position_size(
            symbol="GOOGL",
            entry_price=1500.0,
            stop_loss=1425.0,
            current_prices={"AAPL": 155.0, "MSFT": 255.0}
        )
        
        assert isinstance(position_size, int)
        assert position_size >= 0
        
        # Test stop loss adjustment
        new_stop_loss = risk_manager.adjust_trailing_stop(
            portfolio.positions[0],
            current_price=160.0,
            trailing_percent=5.0
        )
        
        # Expected new stop loss = max(current_stop_loss, current_price * (1 - trailing_percent/100))
        expected_stop = max(140.0, 160.0 * 0.95)
        assert new_stop_loss == expected_stop
