"""
Risk Metrics Module for Gemma Advanced Trading System.

This module provides various risk metrics for portfolio and performance analysis.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import math


class RiskMetrics:
    """Class for calculating various risk metrics for portfolio analysis."""
    
    def __init__(self, risk_free_rate=0.0):
        """
        Initialize the RiskMetrics calculator.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate as a decimal
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
        
    def sharpe_ratio(self, returns, annualized=True, period=252):
        """
        Calculate Sharpe Ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        annualized : bool
            Whether to annualize the result
        period : int
            Number of periods in a year for annualization
            
        Returns:
        --------
        float
            Sharpe Ratio
        """
        if len(returns) == 0:
            return 0
            
        # Calculate excess returns
        excess_returns = returns - self.daily_risk_free_rate
        
        # Calculate mean and standard deviation
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)
        
        if std_excess_return == 0:
            return 0
            
        # Calculate Sharpe Ratio
        sharpe = mean_excess_return / std_excess_return
        
        # Annualize if requested
        if annualized:
            sharpe = sharpe * np.sqrt(period)
            
        return sharpe
    
    def sortino_ratio(self, returns, annualized=True, period=252, target_return=0):
        """
        Calculate Sortino Ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        annualized : bool
            Whether to annualize the result
        period : int
            Number of periods in a year for annualization
        target_return : float
            Target return for downside deviation calculation
            
        Returns:
        --------
        float
            Sortino Ratio
        """
        if len(returns) == 0:
            return 0
            
        # Calculate excess returns
        excess_returns = returns - self.daily_risk_free_rate
        
        # Calculate mean excess return
        mean_excess_return = np.mean(excess_returns)
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < target_return]
        if len(downside_returns) == 0:
            return float('inf')  # No downside returns
            
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return 0
            
        # Calculate Sortino Ratio
        sortino = mean_excess_return / downside_deviation
        
        # Annualize if requested
        if annualized:
            sortino = sortino * np.sqrt(period)
            
        return sortino
    
    def calmar_ratio(self, returns, annualized=True, period=252):
        """
        Calculate Calmar Ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        annualized : bool
            Whether to annualize the result
        period : int
            Number of periods in a year for annualization
            
        Returns:
        --------
        float
            Calmar Ratio
        """
        if len(returns) == 0:
            return 0
            
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # Calculate annualized return
        total_return = cumulative_returns.iloc[-1] if hasattr(returns, 'iloc') else cumulative_returns[-1]
        num_periods = len(returns)
        annualized_return = (1 + total_return) ** (period / num_periods) - 1
        
        # Calculate maximum drawdown
        max_drawdown = self.maximum_drawdown(returns)
        
        if max_drawdown == 0:
            return 0
            
        # Calculate Calmar Ratio
        calmar = annualized_return / abs(max_drawdown)
            
        return calmar
    
    def information_ratio(self, returns, benchmark_returns, annualized=True, period=252):
        """
        Calculate Information Ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        benchmark_returns : pandas.Series or numpy.ndarray
            Series of benchmark returns
        annualized : bool
            Whether to annualize the result
        period : int
            Number of periods in a year for annualization
            
        Returns:
        --------
        float
            Information Ratio
        """
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0
            
        # Calculate tracking error
        tracking_error = returns - benchmark_returns
        
        # Calculate mean and standard deviation of tracking error
        mean_tracking_error = np.mean(tracking_error)
        std_tracking_error = np.std(tracking_error, ddof=1)
        
        if std_tracking_error == 0:
            return 0
            
        # Calculate Information Ratio
        information_ratio = mean_tracking_error / std_tracking_error
        
        # Annualize if requested
        if annualized:
            information_ratio = information_ratio * np.sqrt(period)
            
        return information_ratio
    
    def maximum_drawdown(self, returns):
        """
        Calculate Maximum Drawdown.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
            
        Returns:
        --------
        float
            Maximum Drawdown as a positive decimal
        """
        if len(returns) == 0:
            return 0
            
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Calculate maximum drawdown
        max_drawdown = np.min(drawdown)
        
        return abs(max_drawdown)
    
    def drawdown_periods(self, returns, threshold=0.0):
        """
        Identify drawdown periods.
        
        Parameters:
        -----------
        returns : pandas.Series
            Series of returns
        threshold : float
            Minimum drawdown to consider
            
        Returns:
        --------
        list
            List of drawdown periods (start date, end date, drawdown)
        """
        if len(returns) == 0:
            return []
            
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Identify drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        max_drawdown = 0
        
        for date, dd in drawdown.items():
            if not in_drawdown and dd < -threshold:
                # Start of drawdown period
                in_drawdown = True
                start_date = date
                max_drawdown = dd
            elif in_drawdown:
                if dd < max_drawdown:
                    # Update maximum drawdown
                    max_drawdown = dd
                elif dd == 0:
                    # End of drawdown period
                    in_drawdown = False
                    drawdown_periods.append((start_date, date, abs(max_drawdown)))
                    start_date = None
                    max_drawdown = 0
        
        # Check if still in drawdown at the end
        if in_drawdown:
            drawdown_periods.append((start_date, returns.index[-1], abs(max_drawdown)))
        
        return drawdown_periods
    
    def value_at_risk(self, returns, confidence_level=0.95, method='historical'):
        """
        Calculate Value at Risk (VaR).
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        confidence_level : float
            Confidence level for VaR calculation
        method : str
            Method to use ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
        --------
        float
            Value at Risk as a positive decimal
        """
        if len(returns) == 0:
            return 0
            
        if method == 'historical':
            # Historical VaR
            var_percentile = np.percentile(returns, 100 * (1 - confidence_level))
            return abs(var_percentile)
            
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean + z_score * std
            return abs(var)
            
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            simulated_returns = np.random.normal(mean, std, 10000)
            var_percentile = np.percentile(simulated_returns, 100 * (1 - confidence_level))
            return abs(var_percentile)
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def conditional_value_at_risk(self, returns, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        confidence_level : float
            Confidence level for CVaR calculation
            
        Returns:
        --------
        float
            Conditional Value at Risk as a positive decimal
        """
        if len(returns) == 0:
            return 0
            
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Find the index corresponding to VaR
        var_index = int(len(sorted_returns) * (1 - confidence_level))
        
        # Calculate CVaR as the average of returns beyond VaR
        cvar = np.mean(sorted_returns[:var_index])
        
        return abs(cvar)
    
    def omega_ratio(self, returns, threshold=0.0):
        """
        Calculate Omega Ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        threshold : float
            Return threshold
            
        Returns:
        --------
        float
            Omega Ratio
        """
        if len(returns) == 0:
            return 0
            
        # Separate returns above and below threshold
        returns_above = returns[returns > threshold]
        returns_below = returns[returns <= threshold]
        
        if len(returns_below) == 0:
            return float('inf')  # No returns below threshold
            
        # Calculate Omega Ratio
        omega = (np.sum(returns_above - threshold) / len(returns)) / (np.sum(threshold - returns_below) / len(returns))
        
        return omega
    
    def gain_to_pain_ratio(self, returns):
        """
        Calculate Gain to Pain Ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
            
        Returns:
        --------
        float
            Gain to Pain Ratio
        """
        if len(returns) == 0:
            return 0
            
        # Separate positive and negative returns
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # No negative returns
            
        # Calculate Gain to Pain Ratio
        gain_to_pain = np.sum(positive_returns) / abs(np.sum(negative_returns))
        
        return gain_to_pain
    
    def ulcer_index(self, returns, period=14):
        """
        Calculate Ulcer Index.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        period : int
            Calculation period
            
        Returns:
        --------
        float
            Ulcer Index
        """
        if len(returns) == 0:
            return 0
            
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate percentage drawdown
        drawdown = ((cumulative_returns - running_max) / running_max) * 100
        
        # Calculate squared drawdown
        squared_drawdown = drawdown ** 2
        
        # Calculate Ulcer Index
        ulcer_index = np.sqrt(np.mean(squared_drawdown))
        
        return ulcer_index
    
    def kappa_ratio(self, returns, threshold=0.0, moments=3):
        """
        Calculate Kappa Ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        threshold : float
            Return threshold
        moments : int
            Number of moments to consider (2 for Sortino, 3 for Kappa-3)
            
        Returns:
        --------
        float
            Kappa Ratio
        """
        if len(returns) == 0:
            return 0
            
        # Calculate excess returns
        excess_returns = returns - threshold
        
        # Calculate mean excess return
        mean_excess_return = np.mean(excess_returns)
        
        # Calculate lower partial moments
        lower_partial_moment = np.mean(np.maximum(0, -excess_returns) ** moments) ** (1 / moments)
        
        if lower_partial_moment == 0:
            return float('inf')  # No returns below threshold
            
        # Calculate Kappa Ratio
        kappa = mean_excess_return / lower_partial_moment
        
        return kappa
    
    def tail_ratio(self, returns, percentile=5):
        """
        Calculate Tail Ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        percentile : int
            Percentile to use for tail calculation
            
        Returns:
        --------
        float
            Tail Ratio
        """
        if len(returns) == 0:
            return 0
            
        # Calculate upper and lower tails
        upper_tail = np.percentile(returns, 100 - percentile)
        lower_tail = np.percentile(returns, percentile)
        
        if lower_tail == 0:
            return float('inf')  # Lower tail is zero
            
        # Calculate Tail Ratio
        tail_ratio = abs(upper_tail / lower_tail)
        
        return tail_ratio
    
    def downside_potential(self, returns, threshold=0.0):
        """
        Calculate Downside Potential.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        threshold : float
            Return threshold
            
        Returns:
        --------
        float
            Downside Potential
        """
        if len(returns) == 0:
            return 0
            
        # Calculate downside returns
        downside_returns = returns[returns < threshold]
        
        if len(downside_returns) == 0:
            return 0  # No downside returns
            
        # Calculate Downside Potential
        downside_potential = np.mean(threshold - downside_returns)
        
        return downside_potential
    
    def upside_potential(self, returns, threshold=0.0):
        """
        Calculate Upside Potential.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        threshold : float
            Return threshold
            
        Returns:
        --------
        float
            Upside Potential
        """
        if len(returns) == 0:
            return 0
            
        # Calculate upside returns
        upside_returns = returns[returns > threshold]
        
        if len(upside_returns) == 0:
            return 0  # No upside returns
            
        # Calculate Upside Potential
        upside_potential = np.mean(upside_returns - threshold)
        
        return upside_potential
    
    def upside_potential_ratio(self, returns, threshold=0.0):
        """
        Calculate Upside Potential Ratio.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Series of returns
        threshold : float
            Return threshold
            
        Returns:
        --------
        float
            Upside Potential Ratio
        """
        # Calculate upside and downside potential
        upside = self.upside_potential(returns, threshold)
        downside = self.downside_potential(returns, threshold)
        
        if downside == 0:
            return float('inf')  # No downside potential
            
        # Calculate Upside Potential Ratio
        upr = upside / downside
        
        return upr
    
    def calculate_all_metrics(self, returns, benchmark_returns=None):
        """
        Calculate all risk metrics.
        
        Parameters:
        -----------
        returns : pandas.Series
            Series of returns
        benchmark_returns : pandas.Series, optional
            Series of benchmark returns
            
        Returns:
        --------
        dict
            Dictionary of all risk metrics
        """
        metrics = {
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(returns),
            'maximum_drawdown': self.maximum_drawdown(returns),
            'value_at_risk_95': self.value_at_risk(returns, 0.95),
            'conditional_value_at_risk_95': self.conditional_value_at_risk(returns, 0.95),
            'omega_ratio': self.omega_ratio(returns),
            'gain_to_pain_ratio': self.gain_to_pain_ratio(returns),
            'ulcer_index': self.ulcer_index(returns),
            'kappa_3_ratio': self.kappa_ratio(returns, moments=3),
            'tail_ratio': self.tail_ratio(returns),
            'upside_potential_ratio': self.upside_potential_ratio(returns)
        }
        
        # Add benchmark-dependent metrics if benchmark is provided
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.information_ratio(returns, benchmark_returns)
            metrics['tracking_error'] = np.std(returns - benchmark_returns, ddof=1)
            metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            metrics['alpha'] = np.mean(returns) - metrics['beta'] * np.mean(benchmark_returns)
        
        return metrics
