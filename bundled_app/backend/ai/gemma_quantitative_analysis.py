"""
Gemma Quantitative Analysis Module for Gemma Advanced Trading System.

This module integrates Gemma 3 AI for quantitative analysis of financial markets.
"""

import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
import requests
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemma_quant_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GemmaQuantitativeAnalyzer")


class GemmaQuantitativeAnalyzer:
    """Class for performing quantitative analysis using Gemma 3 AI."""
    
    def __init__(self, model_path=None, config=None):
        """
        Initialize the GemmaQuantitativeAnalyzer.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the Gemma 3 model
        config : dict, optional
            Configuration parameters for the analyzer
        """
        self.model_path = model_path or os.environ.get("GEMMA_MODEL_PATH", "./models/gemma-3")
        self.config = config or {}
        self.default_config = {
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 0.9,
            "context_window": 8192,
            "use_gpu": True,
            "precision": "float16",
            "cache_dir": "./cache"
        }
        
        # Merge default config with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        # Initialize model
        self.model = None
        self.model_loaded = False
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.config["cache_dir"], exist_ok=True)
        
        logger.info("GemmaQuantitativeAnalyzer initialized")
        
    def load_model(self):
        """
        Load the Gemma 3 model.
        
        Returns:
        --------
        bool
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading Gemma 3 model from {self.model_path}")
            
            # This is a placeholder for the actual model loading code
            # In a real implementation, this would use the appropriate library
            # to load the Gemma 3 model based on the runtime environment
            
            # Example with a hypothetical Gemma library:
            # from gemma import GemmaModel
            # self.model = GemmaModel.from_pretrained(
            #     self.model_path,
            #     use_gpu=self.config["use_gpu"],
            #     precision=self.config["precision"]
            # )
            
            # For now, we'll simulate model loading
            time.sleep(2)  # Simulate loading time
            self.model_loaded = True
            logger.info("Gemma 3 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Gemma 3 model: {str(e)}")
            self.model_loaded = False
            return False
            
    def _ensure_model_loaded(self):
        """
        Ensure the model is loaded before making predictions.
        
        Returns:
        --------
        bool
            True if model is loaded, False otherwise
        """
        if not self.model_loaded:
            return self.load_model()
        return True
        
    def _format_prompt(self, prompt_template, **kwargs):
        """
        Format a prompt template with the provided arguments.
        
        Parameters:
        -----------
        prompt_template : str
            Prompt template with placeholders
        **kwargs : dict
            Arguments to fill in the template
            
        Returns:
        --------
        str
            Formatted prompt
        """
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing key in prompt template: {str(e)}")
            return prompt_template
            
    def _call_model(self, prompt, system_prompt=None):
        """
        Call the Gemma 3 model with the provided prompt.
        
        Parameters:
        -----------
        prompt : str
            Input prompt for the model
        system_prompt : str, optional
            System prompt to guide the model's behavior
            
        Returns:
        --------
        str
            Model response
        """
        if not self._ensure_model_loaded():
            return "Error: Model not loaded"
            
        try:
            logger.info("Calling Gemma 3 model")
            
            # This is a placeholder for the actual model inference code
            # In a real implementation, this would use the appropriate library
            # to call the Gemma 3 model
            
            # Example with a hypothetical Gemma library:
            # response = self.model.generate(
            #     prompt=prompt,
            #     system_prompt=system_prompt,
            #     temperature=self.config["temperature"],
            #     max_tokens=self.config["max_tokens"],
            #     top_p=self.config["top_p"]
            # )
            
            # For now, we'll simulate model response
            # In a real implementation, this would be replaced with actual model inference
            time.sleep(1)  # Simulate inference time
            
            # Simulate a response based on the prompt
            if "correlation" in prompt.lower():
                response = "Based on my analysis of the correlation matrix, I observe strong positive correlation (0.85) between assets A and B, suggesting they move together. Assets C and D show negative correlation (-0.62), indicating potential diversification benefits. I recommend reducing exposure to either A or B to avoid concentration risk."
            elif "volatility" in prompt.lower():
                response = "The volatility analysis shows increasing market turbulence with GARCH(1,1) parameters α=0.15 and β=0.78, indicating high persistence. The annualized volatility forecast for the next 5 days is 28.4%, which exceeds the 75th percentile of historical volatility. I recommend reducing position sizes by 25% and implementing tighter stop-losses."
            elif "regression" in prompt.lower():
                response = "The multiple regression analysis yields the following equation: Y = 2.34 + 0.78X₁ - 0.42X₂ + 0.15X₃ with R² = 0.67 and adjusted R² = 0.64. The p-values for X₁ and X₂ are statistically significant (p < 0.01), while X₃ is not (p = 0.23). The Durbin-Watson statistic of 1.97 suggests no autocorrelation in residuals."
            elif "optimization" in prompt.lower():
                response = "The portfolio optimization using mean-variance analysis suggests an allocation of 35% to asset A, 25% to asset B, 20% to asset C, and 20% to asset D. This allocation achieves an expected return of 8.2% with volatility of 12.5%, resulting in a Sharpe ratio of 0.656. The efficient frontier analysis shows this is near the optimal risk-adjusted return point."
            elif "forecast" in prompt.lower():
                response = "My time series forecast using ARIMA(2,1,2) predicts an upward trend with 85% confidence interval of [102.3, 108.7] for the next period. The model shows good fit with AIC=276.4 and BIC=289.1. Residual analysis confirms no significant autocorrelation (Ljung-Box p=0.78). The forecast suggests a potential buying opportunity with a price target of 105.5."
            else:
                response = "Based on my quantitative analysis, I've identified several key patterns in the data. The primary factors driving recent price movements appear to be momentum (37% contribution), volatility regime shifts (28%), and correlation changes (18%). I recommend adjusting your strategy to account for these factors by implementing a dynamic allocation approach that increases exposure during positive momentum phases while maintaining strict risk controls."
            
            logger.info("Gemma 3 model response generated")
            return response
            
        except Exception as e:
            logger.error(f"Error calling Gemma 3 model: {str(e)}")
            return f"Error: {str(e)}"
            
    def analyze_correlation_matrix(self, correlation_matrix, asset_names=None):
        """
        Analyze a correlation matrix using Gemma 3.
        
        Parameters:
        -----------
        correlation_matrix : pandas.DataFrame or numpy.ndarray
            Correlation matrix to analyze
        asset_names : list, optional
            List of asset names corresponding to the correlation matrix
            
        Returns:
        --------
        dict
            Analysis results including insights and recommendations
        """
        # Convert numpy array to DataFrame if necessary
        if isinstance(correlation_matrix, np.ndarray):
            if asset_names is None:
                asset_names = [f"Asset_{i+1}" for i in range(correlation_matrix.shape[0])]
            correlation_matrix = pd.DataFrame(
                correlation_matrix, 
                index=asset_names, 
                columns=asset_names
            )
            
        # Format the correlation matrix as a string
        corr_str = correlation_matrix.round(2).to_string()
        
        # Create prompt for Gemma 3
        prompt_template = """
        I need you to analyze this correlation matrix between different assets:
        
        {correlation_matrix}
        
        Please provide:
        1. Identification of highly correlated asset pairs (both positive and negative)
        2. Clusters of assets that move together
        3. Diversification opportunities
        4. Potential risk concentrations
        5. Specific recommendations for portfolio construction based on this correlation structure
        
        Focus on quantitative insights and actionable recommendations.
        """
        
        prompt = self._format_prompt(prompt_template, correlation_matrix=corr_str)
        
        # Call Gemma 3 model
        response = self._call_model(prompt)
        
        # Process the response
        # In a real implementation, we might parse the response into structured data
        results = {
            "raw_analysis": response,
            "timestamp": datetime.now().isoformat(),
            "correlation_matrix": correlation_matrix.to_dict() if isinstance(correlation_matrix, pd.DataFrame) else correlation_matrix.tolist()
        }
        
        return results
        
    def analyze_volatility(self, price_data, window_sizes=None, forecast_horizon=5):
        """
        Analyze volatility patterns using Gemma 3.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame or pandas.Series
            Price data to analyze
        window_sizes : list, optional
            List of window sizes for volatility calculation
        forecast_horizon : int
            Number of periods to forecast volatility
            
        Returns:
        --------
        dict
            Analysis results including insights and recommendations
        """
        if window_sizes is None:
            window_sizes = [5, 10, 20, 60]
            
        # Calculate returns
        if isinstance(price_data, pd.DataFrame) and 'close' in price_data.columns:
            returns = price_data['close'].pct_change().dropna()
        else:
            returns = price_data.pct_change().dropna()
            
        # Calculate volatility for different window sizes
        volatility = {}
        for window in window_sizes:
            vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            volatility[f"{window}_day"] = vol.dropna().tolist()
            
        # Calculate some basic volatility statistics
        current_vol = volatility[f"{window_sizes[0]}_day"][-1]
        avg_vol = np.mean(volatility[f"{window_sizes[-1]}_day"])
        vol_percentile = np.percentile(volatility[f"{window_sizes[-1]}_day"], [25, 50, 75, 90])
        
        # Create prompt for Gemma 3
        prompt_template = """
        I need you to analyze volatility patterns in financial data with the following characteristics:
        
        Current short-term volatility (annualized): {current_vol:.2f}%
        Average long-term volatility (annualized): {avg_vol:.2f}%
        Volatility percentiles (25th, 50th, 75th, 90th): {vol_percentile}
        
        Recent volatility trends:
        {recent_vol_trend}
        
        Please provide:
        1. Analysis of the current volatility regime
        2. Identification of volatility patterns and cycles
        3. Volatility forecast for the next {forecast_horizon} periods
        4. Risk management recommendations based on volatility analysis
        5. Suggested position sizing adjustments
        
        Focus on quantitative insights and actionable trading recommendations.
        """
        
        # Format recent volatility trend
        recent_vol = volatility[f"{window_sizes[0]}_day"][-10:]
        recent_vol_trend = ", ".join([f"{v:.2f}%" for v in recent_vol])
        
        prompt = self._format_prompt(
            prompt_template, 
            current_vol=current_vol*100, 
            avg_vol=avg_vol*100, 
            vol_percentile=vol_percentile*100, 
            recent_vol_trend=recent_vol_trend,
            forecast_horizon=forecast_horizon
        )
        
        # Call Gemma 3 model
        response = self._call_model(prompt)
        
        # Process the response
        results = {
            "raw_analysis": response,
            "timestamp": datetime.now().isoformat(),
            "volatility_data": volatility,
            "current_volatility": current_vol,
            "average_volatility": avg_vol,
            "volatility_percentiles": vol_percentile.tolist()
        }
        
        return results
        
    def perform_regression_analysis(self, X, y, feature_names=None):
        """
        Perform regression analysis using Gemma 3.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Independent variables
        y : pandas.Series or numpy.ndarray
            Dependent variable
        feature_names : list, optional
            Names of the features in X
            
        Returns:
        --------
        dict
            Analysis results including insights and recommendations
        """
        # Convert numpy arrays to pandas objects if necessary
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"X{i+1}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name="y")
            
        # Perform basic regression analysis
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Get coefficients and statistics
        coefficients = model.coef_
        intercept = model.intercept_
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Create a summary of the regression
        coef_summary = "\n".join([f"{name}: {coef:.4f}" for name, coef in zip(X.columns, coefficients)])
        
        # Create prompt for Gemma 3
        prompt_template = """
        I need you to analyze the results of a multiple regression analysis:
        
        Dependent variable: {y_name}
        Independent variables: {x_names}
        
        Regression coefficients:
        Intercept: {intercept:.4f}
        {coef_summary}
        
        Model statistics:
        R-squared: {r2:.4f}
        RMSE: {rmse:.4f}
        
        Please provide:
        1. Interpretation of the regression coefficients
        2. Assessment of model fit and statistical significance
        3. Identification of the most important factors
        4. Potential issues with the regression (multicollinearity, etc.)
        5. Recommendations for improving the model
        6. Actionable insights based on this regression analysis
        
        Focus on quantitative insights and practical applications of this regression model.
        """
        
        prompt = self._format_prompt(
            prompt_template, 
            y_name=y.name,
            x_names=", ".join(X.columns),
            intercept=intercept,
            coef_summary=coef_summary,
            r2=r2,
            rmse=rmse
        )
        
        # Call Gemma 3 model
        response = self._call_model(prompt)
        
        # Process the response
        results = {
            "raw_analysis": response,
            "timestamp": datetime.now().isoformat(),
            "coefficients": {name: float(coef) for name, coef in zip(X.columns, coefficients)},
            "intercept": float(intercept),
            "r2": float(r2),
            "rmse": float(rmse)
        }
        
        return results
        
    def optimize_portfolio(self, returns_df, risk_free_rate=0.0):
        """
        Optimize portfolio allocation using Gemma 3.
        
        Parameters:
        -----------
        returns_df : pandas.DataFrame
            DataFrame of asset returns with assets as columns
        risk_free_rate : float
            Risk-free rate used in optimization calculations
            
        Returns:
        --------
        dict
            Analysis results including optimal weights and insights
        """
        # Calculate basic portfolio statistics
        mean_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # Perform basic portfolio optimization
        from scipy.optimize import minimize
        
        # Function to calculate portfolio statistics
        def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
            return portfolio_return, portfolio_std_dev, sharpe_ratio
        
        # Function to minimize negative Sharpe Ratio
        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            return -portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]
        
        # Constraints and bounds
        num_assets = len(returns_df.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize portfolio
        optimal_result = minimize(
            neg_sharpe_ratio, 
            initial_weights, 
            args=(mean_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Get optimal weights
        optimal_weights = optimal_result['x']
        
        # Calculate portfolio statistics with optimal weights
        optimal_return, optimal_std_dev, optimal_sharpe = portfolio_stats(
            optimal_weights, mean_returns, cov_matrix, risk_free_rate
        )
        
        # Create a summary of the optimization
        weights_summary = "\n".join([
            f"{asset}: {weight:.2%}" for asset, weight in zip(returns_df.columns, optimal_weights)
        ])
        
        # Create prompt for Gemma 3
        prompt_template = """
        I need you to analyze the results of a portfolio optimization:
        
        Assets: {assets}
        
        Optimal portfolio weights:
        {weights_summary}
        
        Portfolio statistics:
        Expected annual return: {return:.2%}
        Annual volatility: {std_dev:.2%}
        Sharpe ratio: {sharpe:.3f}
        
        Please provide:
        1. Analysis of the optimal portfolio allocation
        2. Assessment of the risk-return characteristics
        3. Potential concerns or limitations of this optimization
        4. Recommendations for portfolio implementation
        5. Suggestions for alternative allocations or constraints to consider
        
        Focus on quantitative insights and practical portfolio management recommendations.
        """
        
        prompt = self._format_prompt(
            prompt_template, 
            assets=", ".join(returns_df.columns),
            weights_summary=weights_summary,
            return=optimal_return,
            std_dev=optimal_std_dev,
            sharpe=optimal_sharpe
        )
        
        # Call Gemma 3 model
        response = self._call_model(prompt)
        
        # Process the response
        results = {
            "raw_analysis": response,
            "timestamp": datetime.now().isoformat(),
            "optimal_weights": {asset: float(weight) for asset, weight in zip(returns_df.columns, optimal_weights)},
            "expected_return": float(optimal_return),
            "volatility": float(optimal_std_dev),
            "sharpe_ratio": float(optimal_sharpe)
        }
        
        return results
        
    def forecast_time_series(self, time_series, forecast_periods=5, include_confidence_intervals=True):
        """
        Forecast time series data using Gemma 3.
        
        Parameters:
        -----------
        time_series : pandas.Series
            Time series data to forecast
        forecast_periods : int
            Number of periods to forecast
        include_confidence_intervals : bool
            Whether to include confidence intervals in the forecast
            
        Returns:
        --------
        dict
            Analysis results including forecast and insights
        """
        # Perform basic time series analysis
        # Calculate statistics
        mean = time_series.mean()
        std = time_series.std()
        min_val = time_series.min()
        max_val = time_series.max()
        
        # Calculate trend
        from scipy import stats
        x = np.arange(len(time_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_series)
        
        # Calculate simple forecast
        last_value = time_series.iloc[-1]
        trend_forecast = [last_value + slope * (i+1) for i in range(forecast_periods)]
        
        # Calculate confidence intervals
        if include_confidence_intervals:
            ci_lower = [forecast - 1.96 * std for forecast in trend_forecast]
            ci_upper = [forecast + 1.96 * std for forecast in trend_forecast]
        else:
            ci_lower = None
            ci_upper = None
        
        # Create a summary of the time series
        recent_values = time_series.tail(10).tolist()
        recent_values_str = ", ".join([f"{v:.2f}" for v in recent_values])
        
        # Create prompt for Gemma 3
        prompt_template = """
        I need you to analyze and forecast this time series:
        
        Recent values: {recent_values}
        
        Statistics:
        Mean: {mean:.2f}
        Standard deviation: {std:.2f}
        Min: {min_val:.2f}
        Max: {max_val:.2f}
        
        Trend analysis:
        Slope: {slope:.4f} per period
        R-squared: {r_squared:.4f}
        P-value: {p_value:.4f}
        
        Simple forecast for next {forecast_periods} periods:
        {forecast_values}
        
        Please provide:
        1. Analysis of the time series patterns and characteristics
        2. Interpretation of the trend and statistical significance
        3. Refined forecast for the next {forecast_periods} periods
        4. Confidence assessment of the forecast
        5. Recommendations for trading or investment decisions based on this forecast
        
        Focus on quantitative insights and actionable recommendations.
        """
        
        forecast_values_str = ", ".join([f"{v:.2f}" for v in trend_forecast])
        if include_confidence_intervals:
            ci_str = ", ".join([f"[{l:.2f}, {u:.2f}]" for l, u in zip(ci_lower, ci_upper)])
            forecast_values_str += f"\nConfidence intervals (95%): {ci_str}"
        
        prompt = self._format_prompt(
            prompt_template, 
            recent_values=recent_values_str,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            slope=slope,
            r_squared=r_value**2,
            p_value=p_value,
            forecast_periods=forecast_periods,
            forecast_values=forecast_values_str
        )
        
        # Call Gemma 3 model
        response = self._call_model(prompt)
        
        # Process the response
        results = {
            "raw_analysis": response,
            "timestamp": datetime.now().isoformat(),
            "forecast": trend_forecast,
            "statistics": {
                "mean": float(mean),
                "std": float(std),
                "min": float(min_val),
                "max": float(max_val)
            },
            "trend": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value)
            }
        }
        
        if include_confidence_intervals:
            results["confidence_intervals"] = {
                "lower": ci_lower,
                "upper": ci_upper
            }
        
        return results
        
    def analyze_trading_strategy(self, strategy_returns, benchmark_returns=None, strategy_description=None):
        """
        Analyze trading strategy performance using Gemma 3.
        
        Parameters:
        -----------
        strategy_returns : pandas.Series
            Returns of the trading strategy
        benchmark_returns : pandas.Series, optional
            Returns of the benchmark for comparison
        strategy_description : str, optional
            Description of the trading strategy
            
        Returns:
        --------
        dict
            Analysis results including performance metrics and insights
        """
        # Calculate performance metrics
        cumulative_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate drawdowns
        cum_returns = (1 + strategy_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns)
        
        # Compare to benchmark if provided
        benchmark_comparison = ""
        if benchmark_returns is not None:
            benchmark_cum_return = (1 + benchmark_returns).cumprod().iloc[-1] - 1
            benchmark_ann_return = (1 + benchmark_cum_return) ** (252 / len(benchmark_returns)) - 1
            benchmark_vol = benchmark_returns.std() * np.sqrt(252)
            benchmark_sharpe = benchmark_ann_return / benchmark_vol if benchmark_vol > 0 else 0
            
            # Calculate alpha and beta
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            variance = np.var(benchmark_returns)
            beta = covariance / variance if variance > 0 else 0
            alpha = annualized_return - beta * benchmark_ann_return
            
            benchmark_comparison = f"""
            Benchmark comparison:
            Strategy vs Benchmark returns: {annualized_return:.2%} vs {benchmark_ann_return:.2%}
            Strategy vs Benchmark volatility: {volatility:.2%} vs {benchmark_vol:.2%}
            Strategy vs Benchmark Sharpe: {sharpe_ratio:.2f} vs {benchmark_sharpe:.2f}
            Alpha: {alpha:.2%}
            Beta: {beta:.2f}
            """
        
        # Create prompt for Gemma 3
        prompt_template = """
        I need you to analyze the performance of a trading strategy:
        
        {strategy_description}
        
        Performance metrics:
        Cumulative return: {cum_return:.2%}
        Annualized return: {ann_return:.2%}
        Annualized volatility: {volatility:.2%}
        Sharpe ratio: {sharpe:.2f}
        Maximum drawdown: {max_drawdown:.2%}
        Win rate: {win_rate:.2%}
        
        {benchmark_comparison}
        
        Please provide:
        1. Comprehensive analysis of the strategy performance
        2. Assessment of risk-adjusted returns
        3. Identification of strengths and weaknesses
        4. Recommendations for strategy improvement
        5. Evaluation of strategy robustness and potential future performance
        
        Focus on quantitative insights and actionable recommendations for strategy enhancement.
        """
        
        strategy_desc = strategy_description or "Trading strategy performance analysis"
        
        prompt = self._format_prompt(
            prompt_template, 
            strategy_description=strategy_desc,
            cum_return=cumulative_return,
            ann_return=annualized_return,
            volatility=volatility,
            sharpe=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            benchmark_comparison=benchmark_comparison
        )
        
        # Call Gemma 3 model
        response = self._call_model(prompt)
        
        # Process the response
        results = {
            "raw_analysis": response,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": {
                "cumulative_return": float(cumulative_return),
                "annualized_return": float(annualized_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate)
            }
        }
        
        if benchmark_returns is not None:
            results["benchmark_comparison"] = {
                "alpha": float(alpha),
                "beta": float(beta),
                "benchmark_return": float(benchmark_ann_return),
                "benchmark_volatility": float(benchmark_vol),
                "benchmark_sharpe": float(benchmark_sharpe)
            }
        
        return results
        
    def generate_trading_strategy(self, market_data, constraints=None, objectives=None):
        """
        Generate a trading strategy using Gemma 3.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary containing market data statistics
        constraints : dict, optional
            Dictionary of constraints for the strategy
        objectives : dict, optional
            Dictionary of objectives for the strategy
            
        Returns:
        --------
        dict
            Generated trading strategy and insights
        """
        # Format market data
        market_summary = []
        for asset, data in market_data.items():
            summary = f"Asset: {asset}\n"
            for key, value in data.items():
                if isinstance(value, float):
                    summary += f"  {key}: {value:.2f}\n"
                else:
                    summary += f"  {key}: {value}\n"
            market_summary.append(summary)
        
        market_data_str = "\n".join(market_summary)
        
        # Format constraints
        constraints_str = ""
        if constraints:
            constraints_list = []
            for key, value in constraints.items():
                constraints_list.append(f"{key}: {value}")
            constraints_str = "Strategy constraints:\n" + "\n".join(constraints_list)
        
        # Format objectives
        objectives_str = ""
        if objectives:
            objectives_list = []
            for key, value in objectives.items():
                objectives_list.append(f"{key}: {value}")
            objectives_str = "Strategy objectives:\n" + "\n".join(objectives_list)
        
        # Create prompt for Gemma 3
        prompt_template = """
        I need you to generate a quantitative trading strategy based on the following market data:
        
        {market_data}
        
        {constraints}
        
        {objectives}
        
        Please provide:
        1. A complete trading strategy with specific entry and exit rules
        2. Position sizing recommendations
        3. Risk management parameters
        4. Expected performance characteristics
        5. Potential weaknesses and how to address them
        6. Implementation guidelines
        
        The strategy should be quantitative, rule-based, and implementable in code.
        Focus on creating a strategy with a positive expected value and robust risk management.
        """
        
        prompt = self._format_prompt(
            prompt_template, 
            market_data=market_data_str,
            constraints=constraints_str,
            objectives=objectives_str
        )
        
        # Call Gemma 3 model
        response = self._call_model(prompt)
        
        # Process the response
        results = {
            "strategy": response,
            "timestamp": datetime.now().isoformat(),
            "market_data_summary": market_data,
            "constraints": constraints,
            "objectives": objectives
        }
        
        return results
        
    def analyze_market_regime(self, market_data, lookback_periods=None):
        """
        Analyze market regime using Gemma 3.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary containing market data for different assets
        lookback_periods : list, optional
            List of lookback periods for analysis
            
        Returns:
        --------
        dict
            Analysis results including regime identification and insights
        """
        if lookback_periods is None:
            lookback_periods = [20, 60, 120]
            
        # Format market data
        market_summary = []
        for asset, data in market_data.items():
            if 'returns' in data:
                returns = data['returns']
                # Calculate statistics for different lookback periods
                for period in lookback_periods:
                    if len(returns) >= period:
                        recent_returns = returns[-period:]
                        mean_return = np.mean(recent_returns)
                        volatility = np.std(recent_returns)
                        sharpe = mean_return / volatility if volatility > 0 else 0
                        market_summary.append(f"Asset: {asset}, Period: {period}")
                        market_summary.append(f"  Mean return: {mean_return:.2%}")
                        market_summary.append(f"  Volatility: {volatility:.2%}")
                        market_summary.append(f"  Sharpe ratio: {sharpe:.2f}")
        
        market_data_str = "\n".join(market_summary)
        
        # Create prompt for Gemma 3
        prompt_template = """
        I need you to analyze the current market regime based on the following market data:
        
        {market_data}
        
        Please provide:
        1. Identification of the current market regime (e.g., bull market, bear market, high volatility, low volatility, etc.)
        2. Analysis of regime characteristics and key drivers
        3. Assessment of regime stability and potential transition signals
        4. Recommended trading strategies suitable for the current regime
        5. Risk management adjustments appropriate for this regime
        
        Focus on quantitative insights and actionable recommendations for navigating the current market environment.
        """
        
        prompt = self._format_prompt(
            prompt_template, 
            market_data=market_data_str
        )
        
        # Call Gemma 3 model
        response = self._call_model(prompt)
        
        # Process the response
        results = {
            "raw_analysis": response,
            "timestamp": datetime.now().isoformat(),
            "market_data_summary": market_data,
            "lookback_periods": lookback_periods
        }
        
        return results
        
    def analyze_factor_exposures(self, returns, factor_returns):
        """
        Analyze factor exposures using Gemma 3.
        
        Parameters:
        -----------
        returns : pandas.Series
            Returns of the portfolio or strategy
        factor_returns : pandas.DataFrame
            Returns of various factors
            
        Returns:
        --------
        dict
            Analysis results including factor exposures and insights
        """
        # Perform factor regression
        import statsmodels.api as sm
        
        # Add constant to factor returns
        X = sm.add_constant(factor_returns)
        
        # Fit the model
        model = sm.OLS(returns, X).fit()
        
        # Get factor exposures and statistics
        factor_exposures = model.params[1:]  # Skip the constant
        t_values = model.tvalues[1:]
        p_values = model.pvalues[1:]
        r_squared = model.rsquared
        
        # Format factor exposures
        factor_summary = []
        for factor, exposure, t_val, p_val in zip(factor_returns.columns, factor_exposures, t_values, p_values):
            significance = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            factor_summary.append(f"{factor}: {exposure:.4f} (t={t_val:.2f}{significance})")
        
        factor_exposures_str = "\n".join(factor_summary)
        
        # Create prompt for Gemma 3
        prompt_template = """
        I need you to analyze the factor exposures of a portfolio or trading strategy:
        
        Factor exposures:
        {factor_exposures}
        
        Model fit:
        R-squared: {r_squared:.4f}
        
        Please provide:
        1. Interpretation of the factor exposures and their statistical significance
        2. Assessment of the overall factor model fit
        3. Identification of dominant factor exposures and their implications
        4. Recommendations for adjusting factor exposures
        5. Potential risks associated with the current factor profile
        
        Focus on quantitative insights and actionable recommendations for optimizing factor exposures.
        """
        
        prompt = self._format_prompt(
            prompt_template, 
            factor_exposures=factor_exposures_str,
            r_squared=r_squared
        )
        
        # Call Gemma 3 model
        response = self._call_model(prompt)
        
        # Process the response
        results = {
            "raw_analysis": response,
            "timestamp": datetime.now().isoformat(),
            "factor_exposures": {factor: float(exposure) for factor, exposure in zip(factor_returns.columns, factor_exposures)},
            "t_values": {factor: float(t_val) for factor, t_val in zip(factor_returns.columns, t_values)},
            "p_values": {factor: float(p_val) for factor, p_val in zip(factor_returns.columns, p_values)},
            "r_squared": float(r_squared)
        }
        
        return results
