"""
Position sizing utility functions for Gemma Advanced Trading System.
"""

import math
from .position_sizing import PositionSizer

def calculate_position_size(account_balance, entry_price, stop_loss, risk_per_trade=0.02, position_sizing_method="fixed_risk"):
    """
    Calculate position size based on risk parameters.
    
    Parameters:
    -----------
    account_balance : float
        Total account balance in base currency
    entry_price : float
        Entry price per share
    stop_loss : float
        Stop loss price per share
    risk_per_trade : float, optional
        Maximum risk per trade as a decimal (e.g., 0.02 = 2%)
    position_sizing_method : str, optional
        Method to use for position sizing ('fixed_risk', 'fixed_percentage', etc.)
        
    Returns:
    --------
    dict
        Position sizing results including position size, value, and risk metrics
    """
    # Create position sizer
    position_sizer = PositionSizer(account_balance=account_balance, max_risk_per_trade=risk_per_trade)
    
    # Calculate position size based on method
    if position_sizing_method == "fixed_risk":
        # Determine if long or short position
        is_long = entry_price > stop_loss
        
        # Calculate position size
        result = position_sizer.fixed_risk(
            entry_price=entry_price,
            stop_loss=stop_loss,
            entry_price_long=is_long,
            shares=True
        )
    elif position_sizing_method == "fixed_percentage":
        result = position_sizer.fixed_percentage(
            percentage=risk_per_trade,
            entry_price=entry_price,
            shares=True
        )
    elif position_sizing_method == "volatility_based":
        # For volatility-based sizing, we need historical prices
        # Since we don't have them here, we'll use a simplified approach
        # Assume stop distance is 2% of entry price
        stop_distance = entry_price * 0.02
        risk_amount = account_balance * risk_per_trade
        num_shares = math.floor(risk_amount / stop_distance)
        position_value = num_shares * entry_price
        
        result = {
            'position_value': position_value,
            'percentage_of_account': position_value / account_balance,
            'shares': num_shares,
            'risk_amount': risk_amount,
            'method': 'volatility_based_simplified'
        }
    else:
        # Default to fixed percentage
        result = position_sizer.fixed_percentage(
            percentage=risk_per_trade,
            entry_price=entry_price,
            shares=True
        )
    
    return result
