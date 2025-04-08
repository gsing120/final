"""
Value at Risk (VaR) Calculations for Gemma Advanced Trading System.

This module provides various methods for calculating Value at Risk.
"""

import numpy as np
import pandas as pd
from scipy import stats


class VaRCalculator:
    """Class for calculating Value at Risk using different methodologies."""
    
    def __init__(self, confidence_level=0.95, time_horizon=1):
        """
        Initialize the VaRCalculator.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for VaR calculation (e.g., 0.95 for 95%)
        time_horizon : int
            Time horizon in days for VaR calculation
        """
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        
    def historical_var(self, returns, portfolio_value):
        """
        Calculate VaR using the Historical Simulation method.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Historical returns series
        portfolio_value : float
            Current portfolio value
            
        Returns:
        --------
        float
            Value at Risk (absolute value)
        """
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Find the index corresponding to the confidence level
        index = int((1 - self.confidence_level) * len(sorted_returns))
        
        # Get the return at the specified confidence level
        var_return = sorted_returns[index]
        
        # Scale VaR to the time horizon
        var_return = var_return * np.sqrt(self.time_horizon)
        
        # Convert to absolute value
        var_absolute = portfolio_value * abs(var_return)
        
        return var_absolute
    
    def parametric_var(self, returns, portfolio_value):
        """
        Calculate VaR using the Parametric (Variance-Covariance) method.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Historical returns series
        portfolio_value : float
            Current portfolio value
            
        Returns:
        --------
        float
            Value at Risk (absolute value)
        """
        # Calculate mean and standard deviation of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Calculate Z-score for the given confidence level
        z_score = stats.norm.ppf(1 - self.confidence_level)
        
        # Calculate VaR
        var_return = mean_return + z_score * std_return
        
        # Scale VaR to the time horizon
        var_return = var_return * np.sqrt(self.time_horizon)
        
        # Convert to absolute value
        var_absolute = portfolio_value * abs(var_return)
        
        return var_absolute
    
    def monte_carlo_var(self, returns, portfolio_value, num_simulations=10000):
        """
        Calculate VaR using Monte Carlo simulation.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Historical returns series
        portfolio_value : float
            Current portfolio value
        num_simulations : int
            Number of Monte Carlo simulations to run
            
        Returns:
        --------
        float
            Value at Risk (absolute value)
        """
        # Calculate mean and standard deviation of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate random returns using normal distribution
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        # Scale returns to the time horizon
        simulated_returns = simulated_returns * np.sqrt(self.time_horizon)
        
        # Sort simulated returns
        sorted_returns = np.sort(simulated_returns)
        
        # Find the index corresponding to the confidence level
        index = int((1 - self.confidence_level) * num_simulations)
        
        # Get the return at the specified confidence level
        var_return = sorted_returns[index]
        
        # Convert to absolute value
        var_absolute = portfolio_value * abs(var_return)
        
        return var_absolute
    
    def conditional_var(self, returns, portfolio_value, method='historical'):
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Historical returns series
        portfolio_value : float
            Current portfolio value
        method : str
            Method to use for VaR calculation ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
        --------
        float
            Conditional Value at Risk (absolute value)
        """
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Find the index corresponding to the confidence level
        index = int((1 - self.confidence_level) * len(sorted_returns))
        
        # Calculate the average of returns beyond VaR
        cvar_return = np.mean(sorted_returns[:index])
        
        # Scale CVaR to the time horizon
        cvar_return = cvar_return * np.sqrt(self.time_horizon)
        
        # Convert to absolute value
        cvar_absolute = portfolio_value * abs(cvar_return)
        
        return cvar_absolute
    
    def portfolio_var(self, returns_dict, weights, portfolio_value):
        """
        Calculate VaR for a portfolio of assets.
        
        Parameters:
        -----------
        returns_dict : dict
            Dictionary of returns series for each asset
        weights : dict
            Dictionary of portfolio weights for each asset
        portfolio_value : float
            Current portfolio value
            
        Returns:
        --------
        float
            Portfolio Value at Risk (absolute value)
        """
        # Create a DataFrame with all returns
        returns_df = pd.DataFrame(returns_dict)
        
        # Create a weights array
        assets = list(weights.keys())
        weights_array = np.array([weights[asset] for asset in assets])
        
        # Calculate portfolio returns
        portfolio_returns = returns_df[assets].dot(weights_array)
        
        # Calculate VaR using historical method
        var = self.historical_var(portfolio_returns, portfolio_value)
        
        return var
    
    def component_var(self, returns_dict, weights, portfolio_value):
        """
        Calculate Component VaR for each asset in the portfolio.
        
        Parameters:
        -----------
        returns_dict : dict
            Dictionary of returns series for each asset
        weights : dict
            Dictionary of portfolio weights for each asset
        portfolio_value : float
            Current portfolio value
            
        Returns:
        --------
        dict
            Component VaR for each asset
        """
        # Create a DataFrame with all returns
        returns_df = pd.DataFrame(returns_dict)
        
        # Create a weights array
        assets = list(weights.keys())
        weights_array = np.array([weights[asset] for asset in assets])
        
        # Calculate portfolio returns
        portfolio_returns = returns_df[assets].dot(weights_array)
        
        # Calculate portfolio VaR
        portfolio_var = self.historical_var(portfolio_returns, portfolio_value)
        
        # Calculate covariance matrix
        cov_matrix = returns_df[assets].cov()
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(weights_array.T.dot(cov_matrix).dot(weights_array))
        
        # Calculate marginal VaR for each asset
        marginal_var = {}
        component_var = {}
        
        for i, asset in enumerate(assets):
            # Calculate marginal contribution to risk
            marginal_contribution = (cov_matrix[asset].dot(weights_array)) / portfolio_vol
            
            # Calculate marginal VaR
            marginal_var[asset] = marginal_contribution * portfolio_var / portfolio_vol
            
            # Calculate component VaR
            component_var[asset] = marginal_var[asset] * weights[asset]
        
        return component_var
    
    def incremental_var(self, returns_dict, weights, portfolio_value, new_asset_returns, new_asset_weight):
        """
        Calculate Incremental VaR for adding a new asset to the portfolio.
        
        Parameters:
        -----------
        returns_dict : dict
            Dictionary of returns series for each asset
        weights : dict
            Dictionary of portfolio weights for each asset
        portfolio_value : float
            Current portfolio value
        new_asset_returns : pandas.Series
            Returns series for the new asset
        new_asset_weight : float
            Proposed weight for the new asset
            
        Returns:
        --------
        float
            Incremental VaR (change in portfolio VaR)
        """
        # Calculate current portfolio VaR
        current_var = self.portfolio_var(returns_dict, weights, portfolio_value)
        
        # Add new asset to returns dictionary and adjust weights
        new_returns_dict = returns_dict.copy()
        new_weights = weights.copy()
        
        # Add new asset
        new_asset_name = f"new_asset_{len(weights)}"
        new_returns_dict[new_asset_name] = new_asset_returns
        
        # Adjust weights
        scale_factor = 1 - new_asset_weight
        for asset in new_weights:
            new_weights[asset] *= scale_factor
        
        new_weights[new_asset_name] = new_asset_weight
        
        # Calculate new portfolio VaR
        new_var = self.portfolio_var(new_returns_dict, new_weights, portfolio_value)
        
        # Calculate incremental VaR
        incremental_var = new_var - current_var
        
        return incremental_var
