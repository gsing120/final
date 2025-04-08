"""
Portfolio Optimization Module for Gemma Advanced Trading System.

This module provides functionality for optimizing portfolio allocations.
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco


class PortfolioOptimizer:
    """Class for optimizing portfolio allocations using various strategies."""
    
    def __init__(self, risk_free_rate=0.0):
        """
        Initialize the PortfolioOptimizer.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free rate used in optimization calculations
        """
        self.risk_free_rate = risk_free_rate
        
    def mean_variance_optimization(self, returns_df, target_return=None, target_risk=None, 
                                  constraints=None, bounds=None):
        """
        Perform mean-variance optimization to find the optimal portfolio weights.
        
        Parameters:
        -----------
        returns_df : pandas.DataFrame
            DataFrame of asset returns with assets as columns
        target_return : float, optional
            Target portfolio return (if None, maximize Sharpe ratio)
        target_risk : float, optional
            Target portfolio risk (if None, maximize Sharpe ratio)
        constraints : list, optional
            List of additional constraints for the optimization
        bounds : tuple or list, optional
            Bounds for asset weights (default is (0, 1) for each asset)
            
        Returns:
        --------
        dict
            Optimization results including optimal weights and metrics
        """
        # Get number of assets
        n_assets = len(returns_df.columns)
        assets = returns_df.columns
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Set default constraints if not provided
        if constraints is None:
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Define the objective function based on the optimization goal
        if target_return is not None:
            # Minimize risk for a given target return
            objective_function = lambda weights: self._portfolio_volatility(weights, cov_matrix)
            constraints.append({'type': 'eq', 
                               'fun': lambda weights: self._portfolio_return(weights, mean_returns) - target_return})
        elif target_risk is not None:
            # Maximize return for a given target risk
            objective_function = lambda weights: -self._portfolio_return(weights, mean_returns)
            constraints.append({'type': 'eq', 
                               'fun': lambda weights: self._portfolio_volatility(weights, cov_matrix) - target_risk})
        else:
            # Maximize Sharpe ratio
            objective_function = lambda weights: -self._portfolio_sharpe_ratio(weights, mean_returns, cov_matrix)
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Run the optimization
        optimization_result = sco.minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract optimal weights
        optimal_weights = optimization_result['x']
        
        # Calculate portfolio metrics with optimal weights
        portfolio_return = self._portfolio_return(optimal_weights, mean_returns)
        portfolio_volatility = self._portfolio_volatility(optimal_weights, cov_matrix)
        portfolio_sharpe = self._portfolio_sharpe_ratio(optimal_weights, mean_returns, cov_matrix)
        
        # Create results dictionary
        results = {
            'weights': {asset: weight for asset, weight in zip(assets, optimal_weights)},
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe,
            'success': optimization_result['success'],
            'message': optimization_result['message']
        }
        
        return results
    
    def efficient_frontier(self, returns_df, n_points=20, bounds=None):
        """
        Calculate the efficient frontier.
        
        Parameters:
        -----------
        returns_df : pandas.DataFrame
            DataFrame of asset returns with assets as columns
        n_points : int
            Number of points on the efficient frontier
        bounds : tuple or list, optional
            Bounds for asset weights (default is (0, 1) for each asset)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with efficient frontier points (return, volatility, sharpe_ratio, weights)
        """
        # Get number of assets
        n_assets = len(returns_df.columns)
        assets = returns_df.columns
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Find minimum volatility portfolio
        min_vol_result = self.mean_variance_optimization(returns_df, bounds=bounds)
        min_vol_return = min_vol_result['return']
        
        # Find maximum return portfolio
        max_return_weights = np.zeros(n_assets)
        max_return_asset_idx = np.argmax(mean_returns)
        max_return_weights[max_return_asset_idx] = 1
        max_return = mean_returns.iloc[max_return_asset_idx]
        
        # Generate target returns for efficient frontier
        target_returns = np.linspace(min_vol_return, max_return, n_points)
        
        # Calculate efficient frontier
        efficient_frontier_points = []
        
        for target_return in target_returns:
            result = self.mean_variance_optimization(returns_df, target_return=target_return, bounds=bounds)
            
            if result['success']:
                point = {
                    'return': result['return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['weights']
                }
                efficient_frontier_points.append(point)
        
        # Convert to DataFrame
        ef_df = pd.DataFrame(efficient_frontier_points)
        
        return ef_df
    
    def risk_parity_optimization(self, returns_df, risk_budget=None, bounds=None):
        """
        Perform risk parity optimization.
        
        Parameters:
        -----------
        returns_df : pandas.DataFrame
            DataFrame of asset returns with assets as columns
        risk_budget : list, optional
            Target risk contribution for each asset (if None, equal risk contribution)
        bounds : tuple or list, optional
            Bounds for asset weights (default is (0, 1) for each asset)
            
        Returns:
        --------
        dict
            Optimization results including optimal weights and metrics
        """
        # Get number of assets
        n_assets = len(returns_df.columns)
        assets = returns_df.columns
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov()
        
        # Set default risk budget if not provided (equal risk contribution)
        if risk_budget is None:
            risk_budget = np.array([1.0 / n_assets] * n_assets)
        else:
            # Normalize risk budget to sum to 1
            risk_budget = np.array(risk_budget) / np.sum(risk_budget)
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Define the objective function for risk parity
        def risk_parity_objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_contributions = weights * (np.dot(cov_matrix, weights)) / portfolio_vol
            risk_contributions = risk_contributions / np.sum(risk_contributions)
            return np.sum((risk_contributions - risk_budget)**2)
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Run the optimization
        optimization_result = sco.minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract optimal weights
        optimal_weights = optimization_result['x']
        
        # Calculate portfolio metrics with optimal weights
        mean_returns = returns_df.mean()
        portfolio_return = self._portfolio_return(optimal_weights, mean_returns)
        portfolio_volatility = self._portfolio_volatility(optimal_weights, cov_matrix)
        portfolio_sharpe = self._portfolio_sharpe_ratio(optimal_weights, mean_returns, cov_matrix)
        
        # Calculate risk contributions
        risk_contributions = optimal_weights * (np.dot(cov_matrix, optimal_weights)) / portfolio_volatility
        risk_contributions = risk_contributions / np.sum(risk_contributions)
        
        # Create results dictionary
        results = {
            'weights': {asset: weight for asset, weight in zip(assets, optimal_weights)},
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe,
            'risk_contributions': {asset: rc for asset, rc in zip(assets, risk_contributions)},
            'success': optimization_result['success'],
            'message': optimization_result['message']
        }
        
        return results
    
    def minimum_correlation_algorithm(self, returns_df, bounds=None):
        """
        Perform portfolio optimization using the Minimum Correlation Algorithm.
        
        Parameters:
        -----------
        returns_df : pandas.DataFrame
            DataFrame of asset returns with assets as columns
        bounds : tuple or list, optional
            Bounds for asset weights (default is (0, 1) for each asset)
            
        Returns:
        --------
        dict
            Optimization results including optimal weights and metrics
        """
        # Get number of assets
        n_assets = len(returns_df.columns)
        assets = returns_df.columns
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Define the objective function for minimum correlation
        def min_correlation_objective(weights):
            weights = np.array(weights)
            portfolio_corr = np.dot(weights.T, np.dot(corr_matrix, weights))
            return portfolio_corr
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Run the optimization
        optimization_result = sco.minimize(
            min_correlation_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract optimal weights
        optimal_weights = optimization_result['x']
        
        # Calculate portfolio metrics with optimal weights
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        portfolio_return = self._portfolio_return(optimal_weights, mean_returns)
        portfolio_volatility = self._portfolio_volatility(optimal_weights, cov_matrix)
        portfolio_sharpe = self._portfolio_sharpe_ratio(optimal_weights, mean_returns, cov_matrix)
        
        # Create results dictionary
        results = {
            'weights': {asset: weight for asset, weight in zip(assets, optimal_weights)},
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe,
            'success': optimization_result['success'],
            'message': optimization_result['message']
        }
        
        return results
    
    def hierarchical_risk_parity(self, returns_df):
        """
        Perform portfolio optimization using the Hierarchical Risk Parity algorithm.
        
        Parameters:
        -----------
        returns_df : pandas.DataFrame
            DataFrame of asset returns with assets as columns
            
        Returns:
        --------
        dict
            Optimization results including optimal weights and metrics
        """
        # Get assets
        assets = returns_df.columns
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov().values
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr().values
        
        # Calculate distance matrix
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        # Perform hierarchical clustering
        import scipy.cluster.hierarchy as sch
        linkage = sch.linkage(distance_matrix, method='single')
        
        # Get the order of assets based on clustering
        cluster_order = sch.leaves_list(linkage)
        
        # Reorder covariance matrix and assets
        ordered_cov = cov_matrix[cluster_order][:, cluster_order]
        ordered_assets = [assets[i] for i in cluster_order]
        
        # Calculate inverse-variance weights
        ivp_weights = 1 / np.diag(ordered_cov)
        ivp_weights = ivp_weights / np.sum(ivp_weights)
        
        # Recursive bisection and weight allocation
        def recursive_bisection(cov, weights, indices):
            n = len(indices)
            if n == 1:
                return weights
            
            # Bisect the cluster
            mid = n // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]
            
            # Calculate sub-cluster variances
            left_var = np.sum(np.outer(weights[left_indices], weights[left_indices]) * 
                             cov[np.ix_(left_indices, left_indices)])
            right_var = np.sum(np.outer(weights[right_indices], weights[right_indices]) * 
                              cov[np.ix_(right_indices, right_indices)])
            
            # Adjust weights to equalize risk
            left_allocation = 1 - right_var / (left_var + right_var)
            right_allocation = 1 - left_allocation
            
            # Scale weights
            weights[left_indices] *= left_allocation
            weights[right_indices] *= right_allocation
            
            # Recursive calls
            weights = recursive_bisection(cov, weights, left_indices)
            weights = recursive_bisection(cov, weights, right_indices)
            
            return weights
        
        # Apply recursive bisection
        indices = list(range(len(ordered_assets)))
        hrp_weights = recursive_bisection(ordered_cov, ivp_weights.copy(), indices)
        
        # Map weights back to original asset order
        weights_dict = {ordered_assets[i]: hrp_weights[i] for i in range(len(ordered_assets))}
        
        # Calculate portfolio metrics
        mean_returns = returns_df.mean()
        weights_array = np.array([weights_dict[asset] for asset in assets])
        portfolio_return = self._portfolio_return(weights_array, mean_returns)
        portfolio_volatility = self._portfolio_volatility(weights_array, cov_matrix)
        portfolio_sharpe = self._portfolio_sharpe_ratio(weights_array, mean_returns, cov_matrix)
        
        # Create results dictionary
        results = {
            'weights': weights_dict,
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe
        }
        
        return results
    
    def _portfolio_return(self, weights, mean_returns):
        """
        Calculate portfolio return.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Array of portfolio weights
        mean_returns : pandas.Series
            Series of mean returns for each asset
            
        Returns:
        --------
        float
            Portfolio return
        """
        return np.sum(weights * mean_returns)
    
    def _portfolio_volatility(self, weights, cov_matrix):
        """
        Calculate portfolio volatility.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Array of portfolio weights
        cov_matrix : pandas.DataFrame
            Covariance matrix of returns
            
        Returns:
        --------
        float
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def _portfolio_sharpe_ratio(self, weights, mean_returns, cov_matrix):
        """
        Calculate portfolio Sharpe ratio.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Array of portfolio weights
        mean_returns : pandas.Series
            Series of mean returns for each asset
        cov_matrix : pandas.DataFrame
            Covariance matrix of returns
            
        Returns:
        --------
        float
            Portfolio Sharpe ratio
        """
        portfolio_return = self._portfolio_return(weights, mean_returns)
        portfolio_volatility = self._portfolio_volatility(weights, cov_matrix)
        
        if portfolio_volatility == 0:
            return 0
        
        return (portfolio_return - self.risk_free_rate) / portfolio_volatility
