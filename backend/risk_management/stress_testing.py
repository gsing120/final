"""
Stress Testing Module for Gemma Advanced Trading System.

This module provides functionality for stress testing portfolios under various scenarios.
"""

import numpy as np
import pandas as pd
from datetime import datetime


class StressTester:
    """Class for performing stress tests on portfolios."""
    
    def __init__(self):
        """Initialize the StressTester."""
        self.historical_scenarios = {}
        self.custom_scenarios = {}
        
    def add_historical_scenario(self, name, start_date, end_date, description=None):
        """
        Add a historical scenario for stress testing.
        
        Parameters:
        -----------
        name : str
            Name of the scenario
        start_date : datetime or str
            Start date of the scenario
        end_date : datetime or str
            End date of the scenario
        description : str, optional
            Description of the scenario
        """
        self.historical_scenarios[name] = {
            'start_date': pd.to_datetime(start_date),
            'end_date': pd.to_datetime(end_date),
            'description': description
        }
        
    def add_custom_scenario(self, name, asset_shocks, correlation_adjustments=None, description=None):
        """
        Add a custom scenario for stress testing.
        
        Parameters:
        -----------
        name : str
            Name of the scenario
        asset_shocks : dict
            Dictionary of asset price shocks (percentage changes)
        correlation_adjustments : dict, optional
            Dictionary of correlation adjustments between assets
        description : str, optional
            Description of the scenario
        """
        self.custom_scenarios[name] = {
            'asset_shocks': asset_shocks,
            'correlation_adjustments': correlation_adjustments,
            'description': description
        }
        
    def run_historical_stress_test(self, portfolio, historical_data, scenario_name):
        """
        Run a stress test using a historical scenario.
        
        Parameters:
        -----------
        portfolio : dict
            Dictionary of portfolio positions with symbols and quantities
        historical_data : dict
            Dictionary of historical price data for each symbol
        scenario_name : str
            Name of the historical scenario to use
            
        Returns:
        --------
        dict
            Stress test results including portfolio impact
        """
        if scenario_name not in self.historical_scenarios:
            raise ValueError(f"Historical scenario '{scenario_name}' not found")
            
        scenario = self.historical_scenarios[scenario_name]
        start_date = scenario['start_date']
        end_date = scenario['end_date']
        
        # Initialize results
        results = {
            'scenario_name': scenario_name,
            'description': scenario['description'],
            'start_date': start_date,
            'end_date': end_date,
            'portfolio_impact': {},
            'total_impact': 0.0
        }
        
        # Calculate impact for each position
        for symbol, position in portfolio.items():
            if symbol not in historical_data:
                continue
                
            data = historical_data[symbol]
            
            # Filter data for the scenario period
            scenario_data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            if len(scenario_data) == 0:
                continue
                
            # Calculate price change during the scenario
            start_price = scenario_data['close'].iloc[0]
            end_price = scenario_data['close'].iloc[-1]
            price_change = (end_price - start_price) / start_price
            
            # Calculate impact on position
            position_value = position['quantity'] * position['current_price']
            position_impact = position_value * price_change
            
            # Add to results
            results['portfolio_impact'][symbol] = {
                'price_change': price_change,
                'position_impact': position_impact
            }
            
            results['total_impact'] += position_impact
            
        # Calculate percentage impact on portfolio
        portfolio_value = sum(position['quantity'] * position['current_price'] for position in portfolio.values())
        results['percentage_impact'] = results['total_impact'] / portfolio_value if portfolio_value > 0 else 0
        
        return results
        
    def run_custom_stress_test(self, portfolio, scenario_name):
        """
        Run a stress test using a custom scenario.
        
        Parameters:
        -----------
        portfolio : dict
            Dictionary of portfolio positions with symbols and quantities
        scenario_name : str
            Name of the custom scenario to use
            
        Returns:
        --------
        dict
            Stress test results including portfolio impact
        """
        if scenario_name not in self.custom_scenarios:
            raise ValueError(f"Custom scenario '{scenario_name}' not found")
            
        scenario = self.custom_scenarios[scenario_name]
        asset_shocks = scenario['asset_shocks']
        
        # Initialize results
        results = {
            'scenario_name': scenario_name,
            'description': scenario['description'],
            'portfolio_impact': {},
            'total_impact': 0.0
        }
        
        # Calculate impact for each position
        for symbol, position in portfolio.items():
            if symbol not in asset_shocks:
                continue
                
            # Get shock for this asset
            price_change = asset_shocks[symbol]
            
            # Calculate impact on position
            position_value = position['quantity'] * position['current_price']
            position_impact = position_value * price_change
            
            # Add to results
            results['portfolio_impact'][symbol] = {
                'price_change': price_change,
                'position_impact': position_impact
            }
            
            results['total_impact'] += position_impact
            
        # Calculate percentage impact on portfolio
        portfolio_value = sum(position['quantity'] * position['current_price'] for position in portfolio.values())
        results['percentage_impact'] = results['total_impact'] / portfolio_value if portfolio_value > 0 else 0
        
        return results
        
    def run_monte_carlo_stress_test(self, portfolio, returns_dict, num_simulations=1000, time_horizon=20):
        """
        Run a Monte Carlo stress test.
        
        Parameters:
        -----------
        portfolio : dict
            Dictionary of portfolio positions with symbols and quantities
        returns_dict : dict
            Dictionary of historical returns for each symbol
        num_simulations : int
            Number of Monte Carlo simulations to run
        time_horizon : int
            Time horizon in days for the simulation
            
        Returns:
        --------
        dict
            Stress test results including worst-case scenarios
        """
        # Create a DataFrame with all returns
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate mean and covariance of returns
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Initialize results
        results = {
            'worst_case_loss': 0.0,
            'worst_case_percentage': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'expected_shortfall_95': 0.0,
            'expected_shortfall_99': 0.0,
            'simulations': []
        }
        
        # Calculate portfolio value
        portfolio_value = sum(position['quantity'] * position['current_price'] for position in portfolio.values())
        
        # Create portfolio weights
        weights = {}
        for symbol, position in portfolio.items():
            position_value = position['quantity'] * position['current_price']
            weights[symbol] = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Run Monte Carlo simulations
        portfolio_returns = np.zeros(num_simulations)
        
        for i in range(num_simulations):
            # Generate random returns using multivariate normal distribution
            random_returns = np.random.multivariate_normal(
                mean_returns.values, 
                cov_matrix.values, 
                time_horizon
            )
            
            # Calculate cumulative returns for each asset
            cumulative_returns = np.cumprod(1 + random_returns, axis=0) - 1
            
            # Calculate final portfolio return
            final_returns = cumulative_returns[-1]
            portfolio_return = 0
            
            for j, symbol in enumerate(returns_df.columns):
                if symbol in weights:
                    portfolio_return += weights[symbol] * final_returns[j]
            
            portfolio_returns[i] = portfolio_return
            
            # Store simulation result
            results['simulations'].append(portfolio_return)
        
        # Sort portfolio returns
        sorted_returns = np.sort(portfolio_returns)
        
        # Calculate worst-case loss
        worst_case_return = sorted_returns[0]
        results['worst_case_loss'] = portfolio_value * worst_case_return
        results['worst_case_percentage'] = worst_case_return
        
        # Calculate VaR
        var_95_index = int(0.05 * num_simulations)
        var_99_index = int(0.01 * num_simulations)
        
        results['var_95'] = portfolio_value * abs(sorted_returns[var_95_index])
        results['var_99'] = portfolio_value * abs(sorted_returns[var_99_index])
        
        # Calculate Expected Shortfall
        results['expected_shortfall_95'] = portfolio_value * abs(np.mean(sorted_returns[:var_95_index]))
        results['expected_shortfall_99'] = portfolio_value * abs(np.mean(sorted_returns[:var_99_index]))
        
        return results
        
    def run_sensitivity_analysis(self, portfolio, risk_factors, shock_range=(-0.2, 0.2), num_steps=10):
        """
        Run a sensitivity analysis on the portfolio.
        
        Parameters:
        -----------
        portfolio : dict
            Dictionary of portfolio positions with symbols and quantities
        risk_factors : dict
            Dictionary mapping risk factors to affected symbols
        shock_range : tuple
            Range of shocks to apply (min, max)
        num_steps : int
            Number of steps in the shock range
            
        Returns:
        --------
        dict
            Sensitivity analysis results
        """
        # Initialize results
        results = {
            'risk_factors': {},
            'portfolio_value': 0.0
        }
        
        # Calculate portfolio value
        portfolio_value = sum(position['quantity'] * position['current_price'] for position in portfolio.values())
        results['portfolio_value'] = portfolio_value
        
        # Generate shock levels
        shock_levels = np.linspace(shock_range[0], shock_range[1], num_steps)
        
        # Analyze each risk factor
        for factor, affected_symbols in risk_factors.items():
            factor_results = {
                'affected_symbols': affected_symbols,
                'shocks': [],
                'impacts': []
            }
            
            for shock in shock_levels:
                total_impact = 0.0
                
                for symbol in affected_symbols:
                    if symbol in portfolio:
                        position = portfolio[symbol]
                        position_value = position['quantity'] * position['current_price']
                        position_impact = position_value * shock
                        total_impact += position_impact
                
                factor_results['shocks'].append(shock)
                factor_results['impacts'].append(total_impact)
            
            results['risk_factors'][factor] = factor_results
        
        return results
        
    def run_correlation_stress_test(self, portfolio, returns_dict, correlation_adjustments):
        """
        Run a stress test with adjusted correlations.
        
        Parameters:
        -----------
        portfolio : dict
            Dictionary of portfolio positions with symbols and quantities
        returns_dict : dict
            Dictionary of historical returns for each symbol
        correlation_adjustments : dict
            Dictionary of correlation adjustments between pairs of assets
            
        Returns:
        --------
        dict
            Stress test results with adjusted correlations
        """
        # Create a DataFrame with all returns
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate original correlation matrix
        original_corr = returns_df.corr()
        
        # Create a copy for adjustments
        adjusted_corr = original_corr.copy()
        
        # Apply correlation adjustments
        for pair, adjustment in correlation_adjustments.items():
            asset1, asset2 = pair
            if asset1 in adjusted_corr.columns and asset2 in adjusted_corr.columns:
                adjusted_corr.loc[asset1, asset2] = adjustment
                adjusted_corr.loc[asset2, asset1] = adjustment
        
        # Ensure the correlation matrix is positive semi-definite
        eigenvalues = np.linalg.eigvals(adjusted_corr)
        if np.any(eigenvalues < 0):
            # If not positive semi-definite, use nearest positive semi-definite matrix
            # This is a simplified approach; in practice, more sophisticated methods might be used
            eigenvalues[eigenvalues < 0] = 0
            eigenvectors = np.linalg.eig(adjusted_corr)[1]
            adjusted_corr = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
            
            # Ensure diagonal is 1
            for i in range(len(adjusted_corr)):
                adjusted_corr.iloc[i, i] = 1.0
        
        # Calculate standard deviations
        std_dev = returns_df.std()
        
        # Calculate adjusted covariance matrix
        adjusted_cov = pd.DataFrame(
            np.outer(std_dev, std_dev) * adjusted_corr,
            index=returns_df.columns,
            columns=returns_df.columns
        )
        
        # Calculate portfolio value
        portfolio_value = sum(position['quantity'] * position['current_price'] for position in portfolio.values())
        
        # Create portfolio weights
        weights = {}
        for symbol, position in portfolio.items():
            position_value = position['quantity'] * position['current_price']
            weights[symbol] = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate portfolio volatility with original correlations
        original_portfolio_vol = self._calculate_portfolio_volatility(weights, original_corr, std_dev)
        
        # Calculate portfolio volatility with adjusted correlations
        adjusted_portfolio_vol = self._calculate_portfolio_volatility(weights, adjusted_corr, std_dev)
        
        # Calculate VaR with original and adjusted correlations (assuming normal distribution)
        z_score_95 = 1.645  # 95% confidence level
        original_var_95 = portfolio_value * original_portfolio_vol * z_score_95
        adjusted_var_95 = portfolio_value * adjusted_portfolio_vol * z_score_95
        
        # Prepare results
        results = {
            'original_correlation': original_corr.to_dict(),
            'adjusted_correlation': adjusted_corr.to_dict(),
            'original_portfolio_volatility': original_portfolio_vol,
            'adjusted_portfolio_volatility': adjusted_portfolio_vol,
            'original_var_95': original_var_95,
            'adjusted_var_95': adjusted_var_95,
            'var_change': adjusted_var_95 - original_var_95,
            'var_change_percentage': (adjusted_var_95 - original_var_95) / original_var_95 if original_var_95 > 0 else 0
        }
        
        return results
    
    def _calculate_portfolio_volatility(self, weights, correlation, std_dev):
        """
        Calculate portfolio volatility.
        
        Parameters:
        -----------
        weights : dict
            Dictionary of portfolio weights
        correlation : pandas.DataFrame
            Correlation matrix
        std_dev : pandas.Series
            Standard deviations of returns
            
        Returns:
        --------
        float
            Portfolio volatility
        """
        portfolio_var = 0.0
        
        for asset1, weight1 in weights.items():
            for asset2, weight2 in weights.items():
                if asset1 in correlation.index and asset2 in correlation.columns:
                    corr = correlation.loc[asset1, asset2]
                    portfolio_var += weight1 * weight2 * std_dev[asset1] * std_dev[asset2] * corr
        
        return np.sqrt(portfolio_var)
