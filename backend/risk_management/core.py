"""
Core Risk Management Module for Gemma Advanced Trading System.

This module provides the foundation for risk management functionality.
"""

import numpy as np
import pandas as pd
from datetime import datetime


class RiskManager:
    """Core risk management class for the Gemma Advanced Trading System."""
    
    def __init__(self, portfolio_value=100000.0, max_portfolio_risk=0.02, max_position_risk=0.01):
        """
        Initialize the RiskManager.
        
        Parameters:
        -----------
        portfolio_value : float
            Total portfolio value in base currency
        max_portfolio_risk : float
            Maximum acceptable portfolio risk as a decimal (e.g., 0.02 = 2%)
        max_position_risk : float
            Maximum acceptable risk per position as a decimal (e.g., 0.01 = 1%)
        """
        self.portfolio_value = portfolio_value
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.positions = {}  # Dictionary to store current positions
        self.risk_metrics = {}  # Dictionary to store risk metrics
        self.historical_data = {}  # Dictionary to store historical data for risk calculations
        self.account_balance = portfolio_value  # Initialize account balance to match portfolio value
        
    def set_portfolio_value(self, value):
        """
        Set the current portfolio value.
        
        Parameters:
        -----------
        value : float
            Current portfolio value in base currency
        """
        self.portfolio_value = value
        
    def set_account_balance(self, balance):
        """
        Set the current account balance.
        
        Parameters:
        -----------
        balance : float
            Current account balance in base currency
        """
        self.account_balance = balance
        
    def get_account_balance(self):
        """
        Get the current account balance.
        
        Returns:
        --------
        float
            Current account balance in base currency
        """
        return self.account_balance
        
    def update_account_balance(self, amount, transaction_type):
        """
        Update the account balance based on a transaction.
        
        Parameters:
        -----------
        amount : float
            Transaction amount
        transaction_type : str
            Type of transaction ('deposit', 'withdrawal', 'trade_profit', 'trade_loss', 'fee')
            
        Returns:
        --------
        float
            Updated account balance
        """
        if transaction_type in ['deposit', 'trade_profit']:
            self.account_balance += amount
        elif transaction_type in ['withdrawal', 'trade_loss', 'fee']:
            self.account_balance -= amount
        
        return self.account_balance
        
    def set_risk_limits(self, max_portfolio_risk, max_position_risk):
        """
        Set risk limits for the portfolio.
        
        Parameters:
        -----------
        max_portfolio_risk : float
            Maximum acceptable portfolio risk as a decimal
        max_position_risk : float
            Maximum acceptable risk per position as a decimal
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        
    def add_position(self, symbol, quantity, entry_price, stop_loss=None):
        """
        Add a position to the risk management system.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        quantity : float
            Position quantity
        entry_price : float
            Entry price per unit
        stop_loss : float, optional
            Stop loss price per unit
            
        Returns:
        --------
        bool
            True if position was added successfully, False otherwise
        """
        position_value = quantity * entry_price
        
        # Calculate position risk if stop loss is provided
        if stop_loss is not None:
            if quantity > 0:  # Long position
                risk_per_unit = entry_price - stop_loss
            else:  # Short position
                risk_per_unit = stop_loss - entry_price
                
            position_risk = abs(quantity) * risk_per_unit
            risk_percentage = position_risk / self.portfolio_value
            
            # Check if position risk exceeds maximum allowed
            if risk_percentage > self.max_position_risk:
                return False
        
        # Add position to the dictionary
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'entry_time': datetime.now()
        }
        
        return True
        
    def update_position(self, symbol, quantity=None, current_price=None, stop_loss=None):
        """
        Update an existing position.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        quantity : float, optional
            New position quantity
        current_price : float, optional
            Current market price
        stop_loss : float, optional
            New stop loss price
            
        Returns:
        --------
        bool
            True if position was updated successfully, False otherwise
        """
        if symbol not in self.positions:
            return False
            
        position = self.positions[symbol]
        
        if quantity is not None:
            position['quantity'] = quantity
            
        if current_price is not None:
            position['current_price'] = current_price
            position['position_value'] = position['quantity'] * current_price
            
            # Calculate unrealized P&L
            position['unrealized_pnl'] = position['quantity'] * (current_price - position['entry_price'])
            position['unrealized_pnl_pct'] = position['unrealized_pnl'] / (position['quantity'] * position['entry_price'])
            
        if stop_loss is not None:
            position['stop_loss'] = stop_loss
            
        return True
        
    def remove_position(self, symbol):
        """
        Remove a position from the risk management system.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
            
        Returns:
        --------
        bool
            True if position was removed successfully, False otherwise
        """
        if symbol in self.positions:
            del self.positions[symbol]
            return True
        return False
        
    def calculate_portfolio_exposure(self):
        """
        Calculate the current portfolio exposure.
        
        Returns:
        --------
        float
            Total portfolio exposure as a percentage of portfolio value
        """
        total_exposure = 0.0
        
        for symbol, position in self.positions.items():
            if 'position_value' in position:
                total_exposure += abs(position['position_value'])
                
        exposure_percentage = total_exposure / self.portfolio_value
        
        return exposure_percentage
        
    def calculate_position_risk(self, symbol):
        """
        Calculate the risk for a specific position.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
            
        Returns:
        --------
        tuple
            (Risk amount, Risk percentage)
        """
        if symbol not in self.positions:
            return (0.0, 0.0)
            
        position = self.positions[symbol]
        
        if 'stop_loss' not in position or position['stop_loss'] is None:
            return (0.0, 0.0)
            
        # Calculate risk based on stop loss
        if position['quantity'] > 0:  # Long position
            risk_per_unit = position['entry_price'] - position['stop_loss']
        else:  # Short position
            risk_per_unit = position['stop_loss'] - position['entry_price']
            
        risk_amount = abs(position['quantity']) * risk_per_unit
        risk_percentage = risk_amount / self.portfolio_value
        
        return (risk_amount, risk_percentage)
        
    def calculate_total_portfolio_risk(self):
        """
        Calculate the total portfolio risk based on current positions.
        
        Returns:
        --------
        tuple
            (Total risk amount, Total risk percentage)
        """
        total_risk = 0.0
        
        for symbol in self.positions:
            risk_amount, _ = self.calculate_position_risk(symbol)
            total_risk += risk_amount
            
        risk_percentage = total_risk / self.portfolio_value
        
        return (total_risk, risk_percentage)
        
    def is_risk_acceptable(self):
        """
        Check if the current portfolio risk is within acceptable limits.
        
        Returns:
        --------
        bool
            True if risk is acceptable, False otherwise
        """
        _, risk_percentage = self.calculate_total_portfolio_risk()
        
        return risk_percentage <= self.max_portfolio_risk
        
    def get_portfolio_summary(self):
        """
        Get a summary of the current portfolio.
        
        Returns:
        --------
        dict
            Portfolio summary including value, exposure, risk, and positions
        """
        exposure = self.calculate_portfolio_exposure()
        total_risk, risk_percentage = self.calculate_total_portfolio_risk()
        
        summary = {
            'portfolio_value': self.portfolio_value,
            'account_balance': self.account_balance,
            'exposure': exposure,
            'exposure_percentage': exposure / self.portfolio_value,
            'total_risk': total_risk,
            'risk_percentage': risk_percentage,
            'position_count': len(self.positions),
            'positions': self.positions
        }
        
        return summary
        
    def suggest_position_size(self, symbol, entry_price, stop_loss, risk_percentage=None):
        """
        Suggest position size based on risk parameters.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        entry_price : float
            Planned entry price
        stop_loss : float
            Planned stop loss price
        risk_percentage : float, optional
            Risk percentage for this position (defaults to max_position_risk)
            
        Returns:
        --------
        float
            Suggested position size in quantity
        """
        if risk_percentage is None:
            risk_percentage = self.max_position_risk
            
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
            
        # Calculate risk amount
        risk_amount = self.portfolio_value * risk_percentage
        
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        return position_size
        
    def add_historical_data(self, symbol, data):
        """
        Add historical price data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        data : pandas.DataFrame
            Historical price data with at least 'close' column
        """
        self.historical_data[symbol] = data
        
    def get_correlation_matrix(self):
        """
        Calculate correlation matrix for all symbols in the portfolio.
        
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix of returns
        """
        # Extract returns for each symbol
        returns_dict = {}
        
        for symbol, data in self.historical_data.items():
            if 'close' in data.columns:
                returns_dict[symbol] = data['close'].pct_change().dropna()
                
        # Create a DataFrame with all returns
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
