"""
Position Sizing Module for Gemma Advanced Trading System.

This module provides functionality for determining optimal position sizes based on various strategies.
"""

import numpy as np
import pandas as pd
import math


class PositionSizer:
    """Class for calculating optimal position sizes using various strategies."""
    
    def __init__(self, account_balance=100000.0, max_risk_per_trade=0.02, max_portfolio_risk=0.05):
        """
        Initialize the PositionSizer.
        
        Parameters:
        -----------
        account_balance : float
            Total account balance in base currency
        max_risk_per_trade : float
            Maximum risk per trade as a decimal (e.g., 0.02 = 2%)
        max_portfolio_risk : float
            Maximum portfolio risk as a decimal (e.g., 0.05 = 5%)
        """
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        
    def fixed_dollar_amount(self, amount):
        """
        Calculate position size based on a fixed dollar amount.
        
        Parameters:
        -----------
        amount : float
            Fixed dollar amount to allocate
            
        Returns:
        --------
        dict
            Position sizing results
        """
        # Check if amount exceeds account balance
        if amount > self.account_balance:
            amount = self.account_balance
            
        results = {
            'position_value': amount,
            'percentage_of_account': amount / self.account_balance,
            'method': 'fixed_dollar_amount'
        }
        
        return results
    
    def fixed_percentage(self, percentage, entry_price=None, shares=False):
        """
        Calculate position size based on a fixed percentage of account balance.
        
        Parameters:
        -----------
        percentage : float
            Percentage of account balance to allocate (as a decimal)
        entry_price : float, optional
            Entry price per share (required if shares=True)
        shares : bool
            Whether to return the number of shares
            
        Returns:
        --------
        dict
            Position sizing results
        """
        # Ensure percentage is within bounds
        percentage = min(percentage, 1.0)
        
        # Calculate position value
        position_value = self.account_balance * percentage
        
        results = {
            'position_value': position_value,
            'percentage_of_account': percentage,
            'method': 'fixed_percentage'
        }
        
        # Calculate number of shares if requested
        if shares and entry_price is not None and entry_price > 0:
            results['shares'] = math.floor(position_value / entry_price)
            results['actual_position_value'] = results['shares'] * entry_price
            results['actual_percentage'] = results['actual_position_value'] / self.account_balance
        
        return results
    
    def fixed_risk(self, entry_price, stop_loss, entry_price_long=True, shares=False):
        """
        Calculate position size based on fixed risk per trade.
        
        Parameters:
        -----------
        entry_price : float
            Entry price per share
        stop_loss : float
            Stop loss price per share
        entry_price_long : bool
            Whether the entry is a long position (True) or short position (False)
        shares : bool
            Whether to return the number of shares
            
        Returns:
        --------
        dict
            Position sizing results
        """
        # Calculate risk per share
        if entry_price_long:
            risk_per_share = entry_price - stop_loss
        else:
            risk_per_share = stop_loss - entry_price
            
        # Ensure risk per share is positive
        risk_per_share = abs(risk_per_share)
        
        if risk_per_share == 0:
            return {
                'position_value': 0,
                'percentage_of_account': 0,
                'shares': 0,
                'risk_amount': 0,
                'method': 'fixed_risk',
                'error': 'Risk per share is zero'
            }
        
        # Calculate risk amount
        risk_amount = self.account_balance * self.max_risk_per_trade
        
        # Calculate number of shares
        num_shares = math.floor(risk_amount / risk_per_share)
        
        # Calculate position value
        position_value = num_shares * entry_price
        
        results = {
            'position_value': position_value,
            'percentage_of_account': position_value / self.account_balance,
            'risk_amount': risk_amount,
            'risk_per_share': risk_per_share,
            'method': 'fixed_risk'
        }
        
        if shares:
            results['shares'] = num_shares
        
        return results
    
    def volatility_based(self, entry_price, historical_prices, atr_periods=14, atr_multiplier=2.0, shares=False):
        """
        Calculate position size based on volatility (ATR).
        
        Parameters:
        -----------
        entry_price : float
            Entry price per share
        historical_prices : pandas.DataFrame
            DataFrame with historical prices (must have 'high', 'low', 'close' columns)
        atr_periods : int
            Number of periods for ATR calculation
        atr_multiplier : float
            Multiplier for ATR to determine stop distance
        shares : bool
            Whether to return the number of shares
            
        Returns:
        --------
        dict
            Position sizing results
        """
        # Calculate ATR
        atr = self._calculate_atr(historical_prices, atr_periods)
        
        if atr == 0:
            return {
                'position_value': 0,
                'percentage_of_account': 0,
                'shares': 0,
                'risk_amount': 0,
                'method': 'volatility_based',
                'error': 'ATR is zero'
            }
        
        # Calculate stop distance
        stop_distance = atr * atr_multiplier
        
        # Calculate risk amount
        risk_amount = self.account_balance * self.max_risk_per_trade
        
        # Calculate number of shares
        num_shares = math.floor(risk_amount / stop_distance)
        
        # Calculate position value
        position_value = num_shares * entry_price
        
        results = {
            'position_value': position_value,
            'percentage_of_account': position_value / self.account_balance,
            'risk_amount': risk_amount,
            'atr': atr,
            'stop_distance': stop_distance,
            'method': 'volatility_based'
        }
        
        if shares:
            results['shares'] = num_shares
        
        return results
    
    def kelly_criterion(self, win_rate, win_loss_ratio, entry_price=None, shares=False, fraction=1.0):
        """
        Calculate position size based on the Kelly Criterion.
        
        Parameters:
        -----------
        win_rate : float
            Probability of winning (as a decimal)
        win_loss_ratio : float
            Ratio of average win to average loss
        entry_price : float, optional
            Entry price per share (required if shares=True)
        shares : bool
            Whether to return the number of shares
        fraction : float
            Fraction of Kelly to use (1.0 = full Kelly, 0.5 = half Kelly)
            
        Returns:
        --------
        dict
            Position sizing results
        """
        # Calculate Kelly percentage
        kelly_percentage = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply fraction
        kelly_percentage = kelly_percentage * fraction
        
        # Ensure Kelly percentage is within bounds
        kelly_percentage = max(0, min(kelly_percentage, self.max_risk_per_trade))
        
        # Calculate position value
        position_value = self.account_balance * kelly_percentage
        
        results = {
            'position_value': position_value,
            'percentage_of_account': kelly_percentage,
            'kelly_percentage': kelly_percentage,
            'method': 'kelly_criterion'
        }
        
        # Calculate number of shares if requested
        if shares and entry_price is not None and entry_price > 0:
            results['shares'] = math.floor(position_value / entry_price)
            results['actual_position_value'] = results['shares'] * entry_price
            results['actual_percentage'] = results['actual_position_value'] / self.account_balance
        
        return results
    
    def optimal_f(self, historical_trades, entry_price=None, shares=False, fraction=1.0):
        """
        Calculate position size based on the Optimal F method.
        
        Parameters:
        -----------
        historical_trades : list
            List of historical trade results (profits/losses)
        entry_price : float, optional
            Entry price per share (required if shares=True)
        shares : bool
            Whether to return the number of shares
        fraction : float
            Fraction of Optimal F to use (1.0 = full Optimal F, 0.5 = half Optimal F)
            
        Returns:
        --------
        dict
            Position sizing results
        """
        if not historical_trades:
            return {
                'position_value': 0,
                'percentage_of_account': 0,
                'method': 'optimal_f',
                'error': 'No historical trades provided'
            }
        
        # Find the largest losing trade
        largest_loss = abs(min(historical_trades))
        
        if largest_loss == 0:
            return {
                'position_value': 0,
                'percentage_of_account': 0,
                'method': 'optimal_f',
                'error': 'Largest loss is zero'
            }
        
        # Calculate Optimal F
        optimal_f = self._calculate_optimal_f(historical_trades, largest_loss)
        
        # Apply fraction
        optimal_f = optimal_f * fraction
        
        # Ensure Optimal F is within bounds
        optimal_f = max(0, min(optimal_f, self.max_risk_per_trade))
        
        # Calculate position value
        position_value = self.account_balance * optimal_f
        
        results = {
            'position_value': position_value,
            'percentage_of_account': optimal_f,
            'optimal_f': optimal_f,
            'method': 'optimal_f'
        }
        
        # Calculate number of shares if requested
        if shares and entry_price is not None and entry_price > 0:
            results['shares'] = math.floor(position_value / entry_price)
            results['actual_position_value'] = results['shares'] * entry_price
            results['actual_percentage'] = results['actual_position_value'] / self.account_balance
        
        return results
    
    def portfolio_heat(self, current_positions, entry_price, stop_loss, entry_price_long=True, shares=False):
        """
        Calculate position size based on portfolio heat (total portfolio risk).
        
        Parameters:
        -----------
        current_positions : list
            List of dictionaries with current positions (each with 'risk_amount' key)
        entry_price : float
            Entry price per share
        stop_loss : float
            Stop loss price per share
        entry_price_long : bool
            Whether the entry is a long position (True) or short position (False)
        shares : bool
            Whether to return the number of shares
            
        Returns:
        --------
        dict
            Position sizing results
        """
        # Calculate current portfolio risk
        current_risk = sum(position.get('risk_amount', 0) for position in current_positions)
        
        # Calculate remaining risk budget
        remaining_risk = self.account_balance * self.max_portfolio_risk - current_risk
        
        # Ensure remaining risk is positive
        remaining_risk = max(0, remaining_risk)
        
        # Calculate risk per share
        if entry_price_long:
            risk_per_share = entry_price - stop_loss
        else:
            risk_per_share = stop_loss - entry_price
            
        # Ensure risk per share is positive
        risk_per_share = abs(risk_per_share)
        
        if risk_per_share == 0:
            return {
                'position_value': 0,
                'percentage_of_account': 0,
                'shares': 0,
                'risk_amount': 0,
                'method': 'portfolio_heat',
                'error': 'Risk per share is zero'
            }
        
        # Calculate number of shares
        num_shares = math.floor(remaining_risk / risk_per_share)
        
        # Calculate position value
        position_value = num_shares * entry_price
        
        results = {
            'position_value': position_value,
            'percentage_of_account': position_value / self.account_balance,
            'risk_amount': remaining_risk,
            'risk_per_share': risk_per_share,
            'current_portfolio_risk': current_risk,
            'max_portfolio_risk': self.account_balance * self.max_portfolio_risk,
            'method': 'portfolio_heat'
        }
        
        if shares:
            results['shares'] = num_shares
        
        return results
    
    def _calculate_atr(self, historical_prices, periods=14):
        """
        Calculate Average True Range (ATR).
        
        Parameters:
        -----------
        historical_prices : pandas.DataFrame
            DataFrame with historical prices (must have 'high', 'low', 'close' columns)
        periods : int
            Number of periods for ATR calculation
            
        Returns:
        --------
        float
            ATR value
        """
        if len(historical_prices) < periods:
            return 0
        
        # Calculate True Range
        tr1 = historical_prices['high'] - historical_prices['low']
        tr2 = abs(historical_prices['high'] - historical_prices['close'].shift(1))
        tr3 = abs(historical_prices['low'] - historical_prices['close'].shift(1))
        
        # Get the maximum of the three true ranges
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=periods).mean().iloc[-1]
        
        return atr
    
    def _calculate_optimal_f(self, historical_trades, largest_loss):
        """
        Calculate Optimal F using an iterative approach.
        
        Parameters:
        -----------
        historical_trades : list
            List of historical trade results (profits/losses)
        largest_loss : float
            Absolute value of the largest loss
            
        Returns:
        --------
        float
            Optimal F value
        """
        # Define the objective function (terminal wealth relative)
        def twr(f):
            product = 1.0
            for trade in historical_trades:
                product *= (1 + f * trade / largest_loss)
            return product
        
        # Find the maximum TWR using a grid search
        f_values = np.linspace(0.01, 1.0, 100)
        twr_values = [twr(f) for f in f_values]
        
        # Find the f value that maximizes TWR
        optimal_f = f_values[np.argmax(twr_values)]
        
        return optimal_f
