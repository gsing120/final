"""
Advanced Mathematical Modeling Module for Gemma Advanced Trading System

This module implements advanced mathematical modeling capabilities using Gemma 3
for volatility forecasting, correlation analysis, regime detection, and other
quantitative analysis tasks.
"""

import os
import logging
import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import Gemma 3 integration architecture
from gemma3_integration.architecture import GemmaCore, PromptEngine, DataIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class AdvancedMathematicalModeling:
    """
    Main class for advanced mathematical modeling using Gemma 3.
    
    This class provides methods for volatility forecasting, correlation analysis,
    regime detection, and other advanced quantitative analysis tasks.
    """
    
    def __init__(self, gemma_core: GemmaCore = None):
        """
        Initialize the AdvancedMathematicalModeling.
        
        Parameters:
        -----------
        gemma_core : GemmaCore, optional
            Instance of GemmaCore for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.AdvancedMathematicalModeling")
        
        # Initialize GemmaCore if not provided
        if gemma_core is None:
            from gemma3_integration.architecture import GemmaCore
            gemma_core = GemmaCore()
        
        self.gemma_core = gemma_core
        self.data_integration = gemma_core.data_integration
        
        self.logger.info("Initialized AdvancedMathematicalModeling")
    
    def forecast_volatility(self, ticker: str, days: int = 30, forecast_horizon: int = 10) -> Dict:
        """
        Forecast volatility for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        days : int
            Number of days of historical data to use.
        forecast_horizon : int
            Number of days to forecast.
            
        Returns:
        --------
        Dict
            Volatility forecast results.
        """
        self.logger.info(f"Forecasting volatility for {ticker} (days: {days}, horizon: {forecast_horizon})")
        
        try:
            # Fetch historical data
            historical_data = self.data_integration.fetch_historical_data(ticker, days)
            
            if not historical_data or len(historical_data) < days * 0.8:  # Allow for some missing days
                self.logger.warning(f"Insufficient historical data for {ticker}")
                return {
                    "ticker": ticker,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error": "Insufficient historical data"
                }
            
            # Calculate historical volatility
            returns = np.log(historical_data['close'] / historical_data['close'].shift(1)).dropna()
            historical_volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Generate prompt for volatility forecasting
            prompt = self.gemma_core.prompt_engine.generate_prompt(
                "volatility_forecasting",
                ticker=ticker,
                historical_data=historical_data.to_json(orient='records'),
                historical_volatility=historical_volatility,
                forecast_horizon=forecast_horizon
            )
            
            # Generate reasoning using Gemma 3
            reasoning = self.gemma_core.cot_processor.generate_reasoning(prompt)
            
            # Extract forecast from reasoning
            forecast_values = self._extract_volatility_forecast(reasoning, historical_volatility, forecast_horizon)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(forecast_values, historical_volatility)
            
            # Identify volatility regimes
            volatility_regime = self._identify_volatility_regime(forecast_values, historical_volatility)
            
            # Prepare forecast result
            forecast_result = {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "historical_volatility": historical_volatility,
                "forecast_horizon": forecast_horizon,
                "forecast_values": forecast_values,
                "confidence_intervals": confidence_intervals,
                "volatility_regime": volatility_regime,
                "reasoning": reasoning
            }
            
            return forecast_result
        except Exception as e:
            self.logger.error(f"Failed to forecast volatility: {e}")
            return {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _extract_volatility_forecast(self, reasoning: Dict, historical_volatility: float, forecast_horizon: int) -> List[float]:
        """
        Extract volatility forecast from reasoning.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
        historical_volatility : float
            Historical volatility.
        forecast_horizon : int
            Number of days to forecast.
            
        Returns:
        --------
        List[float]
            Forecast values.
        """
        # Try to extract forecast values from reasoning
        if "forecast_values" in reasoning:
            return reasoning["forecast_values"]
        
        # If not explicitly provided, generate based on reasoning
        conclusion = reasoning.get("conclusion", "")
        
        # Look for volatility trend indicators
        increasing = any(word in conclusion.lower() for word in ["increase", "rise", "higher", "growing", "upward"])
        decreasing = any(word in conclusion.lower() for word in ["decrease", "fall", "lower", "declining", "downward"])
        
        # Generate forecast based on trend
        if increasing:
            # Increasing volatility
            max_increase = 0.5  # Maximum 50% increase
            increase_factor = np.linspace(0, max_increase, forecast_horizon)
            forecast = [historical_volatility * (1 + factor) for factor in increase_factor]
        elif decreasing:
            # Decreasing volatility
            max_decrease = 0.3  # Maximum 30% decrease
            decrease_factor = np.linspace(0, max_decrease, forecast_horizon)
            forecast = [historical_volatility * (1 - factor) for factor in decrease_factor]
        else:
            # Stable volatility with small random variations
            forecast = [historical_volatility * (1 + np.random.normal(0, 0.05)) for _ in range(forecast_horizon)]
        
        return forecast
    
    def _calculate_confidence_intervals(self, forecast_values: List[float], historical_volatility: float) -> Dict:
        """
        Calculate confidence intervals for volatility forecast.
        
        Parameters:
        -----------
        forecast_values : List[float]
            Forecast values.
        historical_volatility : float
            Historical volatility.
            
        Returns:
        --------
        Dict
            Confidence intervals.
        """
        # Calculate 95% confidence intervals
        lower_95 = [value * 0.7 for value in forecast_values]
        upper_95 = [value * 1.3 for value in forecast_values]
        
        # Calculate 68% confidence intervals
        lower_68 = [value * 0.85 for value in forecast_values]
        upper_68 = [value * 1.15 for value in forecast_values]
        
        return {
            "lower_95": lower_95,
            "upper_95": upper_95,
            "lower_68": lower_68,
            "upper_68": upper_68
        }
    
    def _identify_volatility_regime(self, forecast_values: List[float], historical_volatility: float) -> Dict:
        """
        Identify volatility regime.
        
        Parameters:
        -----------
        forecast_values : List[float]
            Forecast values.
        historical_volatility : float
            Historical volatility.
            
        Returns:
        --------
        Dict
            Volatility regime.
        """
        # Calculate average forecast volatility
        avg_forecast = sum(forecast_values) / len(forecast_values)
        
        # Determine trend
        if avg_forecast > historical_volatility * 1.2:
            trend = "increasing"
        elif avg_forecast < historical_volatility * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Determine regime
        if avg_forecast > 0.4:  # 40% annualized volatility
            regime = "high"
        elif avg_forecast > 0.2:  # 20% annualized volatility
            regime = "moderate"
        else:
            regime = "low"
        
        return {
            "regime": regime,
            "trend": trend,
            "avg_forecast": avg_forecast,
            "historical": historical_volatility
        }
    
    def analyze_correlations(self, tickers: List[str], days: int = 30) -> Dict:
        """
        Analyze correlations between multiple tickers.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols.
        days : int
            Number of days of historical data to use.
            
        Returns:
        --------
        Dict
            Correlation analysis results.
        """
        self.logger.info(f"Analyzing correlations for {tickers} (days: {days})")
        
        try:
            # Fetch historical data for all tickers
            all_data = {}
            for ticker in tickers:
                data = self.data_integration.fetch_historical_data(ticker, days)
                if data is not None and len(data) > 0:
                    all_data[ticker] = data['close']
            
            if not all_data or len(all_data) < 2:
                self.logger.warning(f"Insufficient data for correlation analysis")
                return {
                    "tickers": tickers,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error": "Insufficient data for correlation analysis"
                }
            
            # Create DataFrame with close prices
            df = pd.DataFrame(all_data)
            
            # Calculate returns
            returns = np.log(df / df.shift(1)).dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            # Generate prompt for correlation analysis
            prompt = self.gemma_core.prompt_engine.generate_prompt(
                "correlation_analysis",
                tickers=tickers,
                correlation_matrix=correlation_matrix.to_json()
            )
            
            # Generate reasoning using Gemma 3
            reasoning = self.gemma_core.cot_processor.generate_reasoning(prompt)
            
            # Extract key insights
            key_insights = self._extract_correlation_insights(reasoning, correlation_matrix)
            
            # Identify clusters
            clusters = self._identify_correlation_clusters(correlation_matrix)
            
            # Calculate rolling correlations for key pairs
            rolling_correlations = self._calculate_rolling_correlations(returns, key_insights)
            
            # Prepare analysis result
            analysis_result = {
                "tickers": tickers,
                "timestamp": datetime.datetime.now().isoformat(),
                "correlation_matrix": correlation_matrix.to_dict(),
                "key_insights": key_insights,
                "clusters": clusters,
                "rolling_correlations": rolling_correlations,
                "reasoning": reasoning
            }
            
            return analysis_result
        except Exception as e:
            self.logger.error(f"Failed to analyze correlations: {e}")
            return {
                "tickers": tickers,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _extract_correlation_insights(self, reasoning: Dict, correlation_matrix: pd.DataFrame) -> List[Dict]:
        """
        Extract key insights from correlation analysis.
        
        Parameters:
        -----------
        reasoning : Dict
            Chain-of-thought reasoning.
        correlation_matrix : pd.DataFrame
            Correlation matrix.
            
        Returns:
        --------
        List[Dict]
            Key insights.
        """
        insights = []
        
        # Extract from reasoning if available
        if "key_insights" in reasoning:
            return reasoning["key_insights"]
        
        # Find highest correlations
        corr_values = correlation_matrix.unstack()
        corr_values = corr_values[corr_values < 1.0]  # Remove self-correlations
        highest_corrs = corr_values.sort_values(ascending=False)[:5]  # Top 5 highest
        
        for (ticker1, ticker2), corr in highest_corrs.items():
            insights.append({
                "type": "high_correlation",
                "tickers": [ticker1, ticker2],
                "value": corr,
                "description": f"Strong positive correlation between {ticker1} and {ticker2} ({corr:.2f})"
            })
        
        # Find lowest correlations
        lowest_corrs = corr_values.sort_values()[:5]  # Top 5 lowest
        
        for (ticker1, ticker2), corr in lowest_corrs.items():
            insights.append({
                "type": "low_correlation",
                "tickers": [ticker1, ticker2],
                "value": corr,
                "description": f"Weak or negative correlation between {ticker1} and {ticker2} ({corr:.2f})"
            })
        
        # Find average correlations for each ticker
        avg_corrs = correlation_matrix.mean()
        highest_avg = avg_corrs.sort_values(ascending=False)[:3]  # Top 3 highest average
        lowest_avg = avg_corrs.sort_values()[:3]  # Top 3 lowest average
        
        for ticker, avg_corr in highest_avg.items():
            insights.append({
                "type": "high_avg_correlation",
                "ticker": ticker,
                "value": avg_corr,
                "description": f"{ticker} has high average correlation with other tickers ({avg_corr:.2f})"
            })
        
        for ticker, avg_corr in lowest_avg.items():
            insights.append({
                "type": "low_avg_correlation",
                "ticker": ticker,
                "value": avg_corr,
                "description": f"{ticker} has low average correlation with other tickers ({avg_corr:.2f})"
            })
        
        return insights
    
    def _identify_correlation_clusters(self, correlation_matrix: pd.DataFrame) -> List[Dict]:
        """
        Identify clusters of correlated tickers.
        
        Parameters:
        -----------
        correlation_matrix : pd.DataFrame
            Correlation matrix.
            
        Returns:
        --------
        List[Dict]
            Correlation clusters.
        """
        # Simple clustering based on correlation thresholds
        clusters = []
        
        # Convert to numpy array for easier manipulation
        corr_array = correlation_matrix.values
        tickers = correlation_matrix.index.tolist()
        
        # Track which tickers have been assigned to clusters
        assigned = set()
        
        # Find clusters with high internal correlation
        for i in range(len(tickers)):
            if tickers[i] in assigned:
                continue
            
            # Find tickers highly correlated with this one
            cluster_tickers = [tickers[i]]
            assigned.add(tickers[i])
            
            for j in range(len(tickers)):
                if i == j or tickers[j] in assigned:
                    continue
                
                # Check if correlation is high enough
                if corr_array[i, j] > 0.7:  # Threshold for high correlation
                    cluster_tickers.append(tickers[j])
                    assigned.add(tickers[j])
            
            # Only add as cluster if it has multiple tickers
            if len(cluster_tickers) > 1:
                # Calculate average internal correlation
                internal_corrs = []
                for idx1, t1 in enumerate(cluster_tickers):
                    for idx2, t2 in enumerate(cluster_tickers):
                        if idx1 < idx2:  # Avoid duplicates and self-correlations
                            internal_corrs.append(correlation_matrix.loc[t1, t2])
                
                avg_internal_corr = sum(internal_corrs) / len(internal_corrs) if internal_corrs else 0
                
                clusters.append({
                    "tickers": cluster_tickers,
                    "avg_internal_correlation": avg_internal_corr,
                    "size": len(cluster_tickers)
                })
        
        # Add remaining tickers as individual "clusters"
        for ticker in tickers:
            if ticker not in assigned:
                clusters.append({
                    "tickers": [ticker],
                    "avg_internal_correlation": 1.0,
                    "size": 1
                })
        
        return clusters
    
    def _calculate_rolling_correlations(self, returns: pd.DataFrame, key_insights: List[Dict]) -> Dict:
        """
        Calculate rolling correlations for key ticker pairs.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Returns data.
        key_insights : List[Dict]
            Key correlation insights.
            
        Returns:
        --------
        Dict
            Rolling correlations.
        """
        rolling_corrs = {}
        
        # Extract ticker pairs from high and low correlation insights
        pairs = []
        for insight in key_insights:
            if insight["type"] in ["high_correlation", "low_correlation"] and "tickers" in insight:
                ticker_pair = tuple(insight["tickers"])
                if ticker_pair not in pairs and ticker_pair[::-1] not in pairs:
                    pairs.append(ticker_pair)
        
        # Calculate 10-day rolling correlations for each pair
        window = min(10, len(returns) // 2)  # Ensure window is not too large
        
        for ticker1, ticker2 in pairs[:5]:  # Limit to top 5 pairs
            if ticker1 in returns.columns and ticker2 in returns.columns:
                # Calculate rolling correlation
                rolling_corr = returns[ticker1].rolling(window=window).corr(returns[ticker2])
                
                # Convert to list for JSON serialization
                rolling_corrs[f"{ticker1}_{ticker2}"] = rolling_corr.dropna().tolist()
        
        return rolling_corrs
    
    def detect_market_regimes(self, ticker: str, days: int = 90) -> Dict:
        """
        Detect market regimes for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        days : int
            Number of days of historical data to use.
            
        Returns:
        --------
        Dict
            Market regime detection results.
        """
        self.logger.info(f"Detecting market regimes for {ticker} (days: {days})")
        
        try:
            # Fetch historical data
            historical_data = self.data_integration.fetch_historical_data(ticker, days)
            
            if not historical_data or len(historical_data) < days * 0.8:  # Allow for some missing days
                self.logger.warning(f"Insufficient historical data for {ticker}")
                return {
                    "ticker": ticker,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error": "Insufficient historical data"
                }
            
            # Calculate returns
            returns = np.log(historical_data['close'] / historical_data['close'].shift(1)).dropna()
            
            # Calculate volatility
            volatility = returns.rolling(window=20).std() * np.sqrt(252)  # 20-day rolling volatility, annualized
            
            # Calculate trend
            sma_short = historical_data['close'].rolling(window=20).mean()  # 20-day SMA
            sma_long = historical_data['close'].rolling(window=50).mean()  # 50-day SMA
            
            # Generate prompt for regime detection
            prompt = self.gemma_core.prompt_engine.generate_prompt(
                "regime_detection",
                ticker=ticker,
                historical_data=historical_data.to_json(orient='records'),
                returns=returns.to_json(),
                volatility=volatility.to_json()
            )
            
            # Generate reasoning using Gemma 3
            reasoning = self.gemma_core.cot_processor.generate_reasoning(prompt)
            
            # Identify regimes
            regimes = self._identify_regimes(historical_data, returns, volatility, sma_short, sma_long, reasoning)
            
            # Analyze regime transitions
            transitions = self._analyze_regime_transitions(regimes)
            
            # Forecast future regime
            future_regime = self._forecast_future_regime(regimes, reasoning)
            
            # Prepare detection result
            detection_result = {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "regimes": regimes,
                "transitions": transitions,
                "future_regime": future_regime,
                "reasoning": reasoning
            }
            
            return detection_result
        except Exception as e:
            self.logger.error(f"Failed to detect market regimes: {e}")
            return {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _identify_regimes(self, historical_data: pd.DataFrame, returns: pd.Series, 
                         volatility: pd.Series, sma_short: pd.Series, sma_long: pd.Series,
                         reasoning: Dict) -> List[Dict]:
        """
        Identify market regimes.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical price data.
        returns : pd.Series
            Returns data.
        volatility : pd.Series
            Volatility data.
        sma_short : pd.Series
            Short-term moving average.
        sma_long : pd.Series
            Long-term moving average.
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            Identified regimes.
        """
        # Extract from reasoning if available
        if "regimes" in reasoning:
            return reasoning["regimes"]
        
        regimes = []
        
        # Prepare data for regime identification
        data = pd.DataFrame({
            'close': historical_data['close'],
            'returns': returns,
            'volatility': volatility,
            'sma_short': sma_short,
            'sma_long': sma_long
        }).dropna()
        
        if len(data) < 10:
            return regimes
        
        # Define regime characteristics
        regime_defs = {
            'bull_trend': {
                'condition': (data['sma_short'] > data['sma_long']) & (data['volatility'] < data['volatility'].mean()),
                'description': 'Bullish trend with low volatility'
            },
            'bull_volatile': {
                'condition': (data['sma_short'] > data['sma_long']) & (data['volatility'] >= data['volatility'].mean()),
                'description': 'Bullish trend with high volatility'
            },
            'bear_trend': {
                'condition': (data['sma_short'] < data['sma_long']) & (data['volatility'] < data['volatility'].mean()),
                'description': 'Bearish trend with low volatility'
            },
            'bear_volatile': {
                'condition': (data['sma_short'] < data['sma_long']) & (data['volatility'] >= data['volatility'].mean()),
                'description': 'Bearish trend with high volatility'
            },
            'sideways_low_vol': {
                'condition': (abs(data['sma_short'] - data['sma_long']) / data['sma_long'] < 0.02) & 
                            (data['volatility'] < data['volatility'].mean()),
                'description': 'Sideways market with low volatility'
            },
            'sideways_high_vol': {
                'condition': (abs(data['sma_short'] - data['sma_long']) / data['sma_long'] < 0.02) & 
                            (data['volatility'] >= data['volatility'].mean()),
                'description': 'Sideways market with high volatility'
            }
        }
        
        # Identify regimes
        current_regime = None
        regime_start = None
        
        for date, row in data.iterrows():
            # Determine regime for this day
            day_regime = None
            for regime_name, regime_def in regime_defs.items():
                if regime_def['condition'].loc[date]:
                    day_regime = regime_name
                    break
            
            # Default to sideways if no regime identified
            if day_regime is None:
                day_regime = 'sideways_low_vol' if row['volatility'] < data['volatility'].mean() else 'sideways_high_vol'
            
            # Check if regime has changed
            if day_regime != current_regime:
                # Record previous regime if it exists
                if current_regime is not None and regime_start is not None:
                    regimes.append({
                        'regime': current_regime,
                        'description': regime_defs[current_regime]['description'],
                        'start_date': regime_start.isoformat(),
                        'end_date': date.isoformat(),
                        'duration_days': (date - regime_start).days
                    })
                
                # Start new regime
                current_regime = day_regime
                regime_start = date
        
        # Add final regime
        if current_regime is not None and regime_start is not None:
            regimes.append({
                'regime': current_regime,
                'description': regime_defs[current_regime]['description'],
                'start_date': regime_start.isoformat(),
                'end_date': data.index[-1].isoformat(),
                'duration_days': (data.index[-1] - regime_start).days
            })
        
        return regimes
    
    def _analyze_regime_transitions(self, regimes: List[Dict]) -> Dict:
        """
        Analyze regime transitions.
        
        Parameters:
        -----------
        regimes : List[Dict]
            Identified regimes.
            
        Returns:
        --------
        Dict
            Regime transition analysis.
        """
        if not regimes or len(regimes) < 2:
            return {
                "transitions": [],
                "avg_regime_duration": 0,
                "most_common_regime": None,
                "most_common_transition": None
            }
        
        # Calculate average regime duration
        durations = [regime['duration_days'] for regime in regimes]
        avg_duration = sum(durations) / len(durations)
        
        # Find most common regime
        regime_counts = {}
        for regime in regimes:
            regime_type = regime['regime']
            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1
        
        most_common = max(regime_counts.items(), key=lambda x: x[1])
        
        # Analyze transitions
        transitions = []
        transition_counts = {}
        
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]['regime']
            to_regime = regimes[i + 1]['regime']
            transition_key = f"{from_regime}_to_{to_regime}"
            
            transition = {
                "from_regime": from_regime,
                "to_regime": to_regime,
                "from_date": regimes[i]['end_date'],
                "to_date": regimes[i + 1]['start_date']
            }
            
            transitions.append(transition)
            transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
        
        # Find most common transition
        most_common_transition = max(transition_counts.items(), key=lambda x: x[1]) if transition_counts else None
        
        return {
            "transitions": transitions,
            "avg_regime_duration": avg_duration,
            "most_common_regime": {
                "regime": most_common[0],
                "count": most_common[1]
            },
            "most_common_transition": {
                "transition": most_common_transition[0],
                "count": most_common_transition[1]
            } if most_common_transition else None
        }
    
    def _forecast_future_regime(self, regimes: List[Dict], reasoning: Dict) -> Dict:
        """
        Forecast future market regime.
        
        Parameters:
        -----------
        regimes : List[Dict]
            Identified regimes.
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        Dict
            Future regime forecast.
        """
        # Extract from reasoning if available
        if "future_regime" in reasoning:
            return reasoning["future_regime"]
        
        if not regimes:
            return {
                "regime": "unknown",
                "description": "Unable to forecast future regime due to insufficient data",
                "confidence": 0.0
            }
        
        # Get current regime
        current_regime = regimes[-1]['regime']
        
        # Extract forecast from reasoning conclusion
        conclusion = reasoning.get("conclusion", "")
        
        # Look for explicit regime mentions
        regime_keywords = {
            "bull_trend": ["bull trend", "bullish trend", "uptrend", "bull market"],
            "bull_volatile": ["volatile bull", "bullish volatile", "volatile uptrend"],
            "bear_trend": ["bear trend", "bearish trend", "downtrend", "bear market"],
            "bear_volatile": ["volatile bear", "bearish volatile", "volatile downtrend"],
            "sideways_low_vol": ["sideways low", "range-bound low", "consolidation low"],
            "sideways_high_vol": ["sideways high", "range-bound high", "consolidation high"]
        }
        
        forecast_regime = None
        for regime, keywords in regime_keywords.items():
            if any(keyword in conclusion.lower() for keyword in keywords):
                forecast_regime = regime
                break
        
        # If no explicit mention, use current regime as default
        if forecast_regime is None:
            forecast_regime = current_regime
        
        # Determine confidence based on reasoning
        confidence = reasoning.get("confidence", 0.5)
        
        # Get description
        regime_descriptions = {
            'bull_trend': 'Bullish trend with low volatility',
            'bull_volatile': 'Bullish trend with high volatility',
            'bear_trend': 'Bearish trend with low volatility',
            'bear_volatile': 'Bearish trend with high volatility',
            'sideways_low_vol': 'Sideways market with low volatility',
            'sideways_high_vol': 'Sideways market with high volatility'
        }
        
        description = regime_descriptions.get(forecast_regime, "Unknown regime")
        
        return {
            "regime": forecast_regime,
            "description": description,
            "confidence": confidence,
            "current_regime": current_regime,
            "forecast_reasoning": conclusion
        }
    
    def perform_factor_analysis(self, ticker: str, factors: List[str] = None, days: int = 90) -> Dict:
        """
        Perform factor analysis for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        factors : List[str], optional
            List of factors to analyze. If None, uses default factors.
        days : int
            Number of days of historical data to use.
            
        Returns:
        --------
        Dict
            Factor analysis results.
        """
        self.logger.info(f"Performing factor analysis for {ticker} (days: {days})")
        
        try:
            # Use default factors if none provided
            if factors is None:
                factors = ["SPY", "QQQ", "IWM", "VIX", "TLT", "XLF", "XLK", "XLE"]
            
            # Ensure ticker is not in factors
            factors = [f for f in factors if f != ticker]
            
            # Fetch historical data for ticker and factors
            ticker_data = self.data_integration.fetch_historical_data(ticker, days)
            
            if not ticker_data or len(ticker_data) < days * 0.8:
                self.logger.warning(f"Insufficient historical data for {ticker}")
                return {
                    "ticker": ticker,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error": "Insufficient historical data"
                }
            
            # Fetch factor data
            factor_data = {}
            for factor in factors:
                data = self.data_integration.fetch_historical_data(factor, days)
                if data is not None and len(data) > 0:
                    factor_data[factor] = data['close']
            
            if not factor_data:
                self.logger.warning(f"Insufficient factor data")
                return {
                    "ticker": ticker,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error": "Insufficient factor data"
                }
            
            # Create DataFrame with ticker and factor prices
            df = pd.DataFrame(factor_data)
            df[ticker] = ticker_data['close']
            
            # Calculate returns
            returns = np.log(df / df.shift(1)).dropna()
            
            # Perform regression analysis
            regression_results = self._perform_factor_regression(returns, ticker, factors)
            
            # Generate prompt for factor analysis
            prompt = self.gemma_core.prompt_engine.generate_prompt(
                "factor_analysis",
                ticker=ticker,
                factors=factors,
                regression_results=regression_results
            )
            
            # Generate reasoning using Gemma 3
            reasoning = self.gemma_core.cot_processor.generate_reasoning(prompt)
            
            # Extract key factor exposures
            factor_exposures = self._extract_factor_exposures(regression_results, reasoning)
            
            # Analyze factor stability
            factor_stability = self._analyze_factor_stability(returns, ticker, factors)
            
            # Prepare analysis result
            analysis_result = {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "factors": factors,
                "regression_results": regression_results,
                "factor_exposures": factor_exposures,
                "factor_stability": factor_stability,
                "reasoning": reasoning
            }
            
            return analysis_result
        except Exception as e:
            self.logger.error(f"Failed to perform factor analysis: {e}")
            return {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _perform_factor_regression(self, returns: pd.DataFrame, ticker: str, factors: List[str]) -> Dict:
        """
        Perform factor regression analysis.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Returns data.
        ticker : str
            Ticker symbol.
        factors : List[str]
            List of factors.
            
        Returns:
        --------
        Dict
            Regression results.
        """
        # Prepare data for regression
        y = returns[ticker]
        X = returns[factors]
        
        # Add constant for intercept
        X = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
        
        # Perform regression
        model = sm.OLS(y, X).fit()
        
        # Extract results
        coefficients = model.params.to_dict()
        p_values = model.pvalues.to_dict()
        t_values = model.tvalues.to_dict()
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        
        # Format results
        factor_results = {}
        for factor in ['const'] + factors:
            factor_results[factor] = {
                "coefficient": coefficients.get(factor, 0),
                "p_value": p_values.get(factor, 1),
                "t_value": t_values.get(factor, 0),
                "significant": p_values.get(factor, 1) < 0.05
            }
        
        return {
            "factor_results": factor_results,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "num_observations": len(y)
        }
    
    def _extract_factor_exposures(self, regression_results: Dict, reasoning: Dict) -> List[Dict]:
        """
        Extract key factor exposures.
        
        Parameters:
        -----------
        regression_results : Dict
            Regression results.
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            Key factor exposures.
        """
        exposures = []
        
        # Extract from reasoning if available
        if "factor_exposures" in reasoning:
            return reasoning["factor_exposures"]
        
        # Extract from regression results
        factor_results = regression_results.get("factor_results", {})
        
        for factor, result in factor_results.items():
            if factor == 'const':
                continue
            
            coefficient = result.get("coefficient", 0)
            p_value = result.get("p_value", 1)
            significant = result.get("significant", False)
            
            # Only include significant factors
            if significant:
                # Determine exposure type
                if coefficient > 0.8:
                    exposure_type = "very high positive"
                elif coefficient > 0.4:
                    exposure_type = "high positive"
                elif coefficient > 0.1:
                    exposure_type = "moderate positive"
                elif coefficient > 0:
                    exposure_type = "slight positive"
                elif coefficient > -0.1:
                    exposure_type = "slight negative"
                elif coefficient > -0.4:
                    exposure_type = "moderate negative"
                elif coefficient > -0.8:
                    exposure_type = "high negative"
                else:
                    exposure_type = "very high negative"
                
                exposures.append({
                    "factor": factor,
                    "coefficient": coefficient,
                    "p_value": p_value,
                    "exposure_type": exposure_type,
                    "significance": "high" if p_value < 0.01 else "moderate"
                })
        
        # Sort by absolute coefficient value
        exposures.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
        
        return exposures
    
    def _analyze_factor_stability(self, returns: pd.DataFrame, ticker: str, factors: List[str]) -> Dict:
        """
        Analyze stability of factor exposures over time.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Returns data.
        ticker : str
            Ticker symbol.
        factors : List[str]
            List of factors.
            
        Returns:
        --------
        Dict
            Factor stability analysis.
        """
        # Ensure sufficient data
        if len(returns) < 60:  # Need at least 60 days for meaningful analysis
            return {
                "stability_score": 0,
                "rolling_coefficients": {},
                "most_stable_factor": None,
                "least_stable_factor": None
            }
        
        # Calculate rolling regressions
        window = min(30, len(returns) // 3)  # 30-day window or 1/3 of data, whichever is smaller
        
        rolling_coefficients = {}
        coefficient_stds = {}
        
        for factor in factors:
            # Simple rolling regression for each factor individually
            y = returns[ticker]
            X = returns[factor]
            
            # Add constant
            X = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
            
            # Rolling regression
            rolling_coefs = pd.Series(index=returns.index[window-1:])
            
            for i in range(window, len(returns) + 1):
                y_window = y.iloc[i-window:i]
                X_window = X.iloc[i-window:i]
                
                try:
                    model = sm.OLS(y_window, X_window).fit()
                    rolling_coefs.iloc[i-window] = model.params[factor]
                except:
                    rolling_coefs.iloc[i-window] = np.nan
            
            # Store rolling coefficients and calculate standard deviation
            rolling_coefficients[factor] = rolling_coefs.dropna().tolist()
            coefficient_stds[factor] = rolling_coefs.std()
        
        # Determine most and least stable factors
        if coefficient_stds:
            most_stable_factor = min(coefficient_stds.items(), key=lambda x: x[1])
            least_stable_factor = max(coefficient_stds.items(), key=lambda x: x[1])
            
            # Calculate overall stability score (inverse of average std)
            avg_std = sum(coefficient_stds.values()) / len(coefficient_stds)
            stability_score = 1 / (1 + avg_std)  # Normalize to 0-1 range
            
            return {
                "stability_score": stability_score,
                "rolling_coefficients": rolling_coefficients,
                "most_stable_factor": {
                    "factor": most_stable_factor[0],
                    "std": most_stable_factor[1]
                },
                "least_stable_factor": {
                    "factor": least_stable_factor[0],
                    "std": least_stable_factor[1]
                }
            }
        else:
            return {
                "stability_score": 0,
                "rolling_coefficients": {},
                "most_stable_factor": None,
                "least_stable_factor": None
            }
    
    def analyze_time_series(self, ticker: str, days: int = 90, forecast_days: int = 10) -> Dict:
        """
        Perform time series analysis and forecasting.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        days : int
            Number of days of historical data to use.
        forecast_days : int
            Number of days to forecast.
            
        Returns:
        --------
        Dict
            Time series analysis results.
        """
        self.logger.info(f"Performing time series analysis for {ticker} (days: {days}, forecast: {forecast_days})")
        
        try:
            # Fetch historical data
            historical_data = self.data_integration.fetch_historical_data(ticker, days)
            
            if not historical_data or len(historical_data) < days * 0.8:
                self.logger.warning(f"Insufficient historical data for {ticker}")
                return {
                    "ticker": ticker,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error": "Insufficient historical data"
                }
            
            # Perform decomposition
            decomposition = self._decompose_time_series(historical_data)
            
            # Generate prompt for time series analysis
            prompt = self.gemma_core.prompt_engine.generate_prompt(
                "time_series_analysis",
                ticker=ticker,
                historical_data=historical_data.to_json(orient='records'),
                decomposition=decomposition
            )
            
            # Generate reasoning using Gemma 3
            reasoning = self.gemma_core.cot_processor.generate_reasoning(prompt)
            
            # Generate forecast
            forecast = self._generate_forecast(historical_data, decomposition, forecast_days, reasoning)
            
            # Identify patterns
            patterns = self._identify_time_series_patterns(historical_data, decomposition, reasoning)
            
            # Calculate forecast confidence
            confidence_intervals = self._calculate_forecast_confidence(historical_data, forecast, decomposition)
            
            # Prepare analysis result
            analysis_result = {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "decomposition": decomposition,
                "forecast": forecast,
                "patterns": patterns,
                "confidence_intervals": confidence_intervals,
                "reasoning": reasoning
            }
            
            return analysis_result
        except Exception as e:
            self.logger.error(f"Failed to perform time series analysis: {e}")
            return {
                "ticker": ticker,
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _decompose_time_series(self, historical_data: pd.DataFrame) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical price data.
            
        Returns:
        --------
        Dict
            Decomposition results.
        """
        # Ensure sufficient data
        if len(historical_data) < 30:
            return {
                "trend": historical_data['close'].tolist(),
                "seasonal": [0] * len(historical_data),
                "residual": [0] * len(historical_data)
            }
        
        # Convert to time series
        ts = historical_data['close']
        
        # Simple moving average for trend
        window = min(10, len(ts) // 4)  # 10-day window or 1/4 of data, whichever is smaller
        trend = ts.rolling(window=window, center=True).mean()
        
        # Fill NaN values at edges
        trend = trend.fillna(method='bfill').fillna(method='ffill')
        
        # Detrended series
        detrended = ts - trend
        
        # Simple seasonal component (assuming 5-day business week)
        seasonal = pd.Series(index=ts.index)
        
        # Group by day of week and calculate average deviation from trend
        if len(detrended) >= 10:  # Need at least 2 weeks of data
            day_of_week_avg = detrended.groupby(detrended.index.dayofweek).mean()
            
            for i, idx in enumerate(detrended.index):
                day_of_week = idx.dayofweek
                seasonal.iloc[i] = day_of_week_avg.get(day_of_week, 0)
        else:
            seasonal = pd.Series(0, index=ts.index)
        
        # Residual component
        residual = detrended - seasonal
        
        return {
            "trend": trend.tolist(),
            "seasonal": seasonal.tolist(),
            "residual": residual.tolist()
        }
    
    def _generate_forecast(self, historical_data: pd.DataFrame, decomposition: Dict, 
                          forecast_days: int, reasoning: Dict) -> Dict:
        """
        Generate time series forecast.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical price data.
        decomposition : Dict
            Time series decomposition.
        forecast_days : int
            Number of days to forecast.
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        Dict
            Forecast results.
        """
        # Extract from reasoning if available
        if "forecast" in reasoning:
            return reasoning["forecast"]
        
        # Get last price
        last_price = historical_data['close'].iloc[-1]
        last_date = historical_data.index[-1]
        
        # Extract trend direction from reasoning
        conclusion = reasoning.get("conclusion", "")
        
        trend_up = any(word in conclusion.lower() for word in ["upward", "bullish", "positive", "increase", "rising"])
        trend_down = any(word in conclusion.lower() for word in ["downward", "bearish", "negative", "decrease", "falling"])
        
        # Calculate recent trend
        recent_trend = 0
        if len(historical_data) >= 10:
            recent_prices = historical_data['close'].iloc[-10:]
            if recent_prices.iloc[-1] > recent_prices.iloc[0]:
                recent_trend = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1) / 10  # Daily rate
            else:
                recent_trend = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1) / 10  # Daily rate
        
        # Adjust trend based on reasoning
        if trend_up:
            trend_factor = max(0.001, recent_trend)  # Ensure positive
        elif trend_down:
            trend_factor = min(-0.001, recent_trend)  # Ensure negative
        else:
            trend_factor = recent_trend
        
        # Generate forecast dates
        last_date = pd.to_datetime(last_date)
        forecast_dates = []
        current_date = last_date
        
        for _ in range(forecast_days):
            current_date = current_date + pd.Timedelta(days=1)
            # Skip weekends
            while current_date.dayofweek > 4:  # 5 = Saturday, 6 = Sunday
                current_date = current_date + pd.Timedelta(days=1)
            forecast_dates.append(current_date)
        
        # Generate forecast values
        forecast_values = []
        current_price = last_price
        
        for i in range(forecast_days):
            # Apply trend
            current_price = current_price * (1 + trend_factor)
            
            # Add seasonal component if available
            day_of_week = forecast_dates[i].dayofweek
            seasonal_component = 0
            
            # Find average seasonal component for this day of week
            if len(historical_data) >= 10:
                seasonal_values = []
                for j, idx in enumerate(historical_data.index):
                    if idx.dayofweek == day_of_week and j < len(decomposition["seasonal"]):
                        seasonal_values.append(decomposition["seasonal"][j])
                
                if seasonal_values:
                    seasonal_component = sum(seasonal_values) / len(seasonal_values)
            
            # Add seasonal component
            current_price += seasonal_component
            
            # Ensure price is positive
            current_price = max(0.01, current_price)
            
            forecast_values.append(current_price)
        
        return {
            "dates": [date.isoformat() for date in forecast_dates],
            "values": forecast_values,
            "trend_factor": trend_factor,
            "last_price": last_price
        }
    
    def _identify_time_series_patterns(self, historical_data: pd.DataFrame, 
                                      decomposition: Dict, reasoning: Dict) -> List[Dict]:
        """
        Identify patterns in time series data.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical price data.
        decomposition : Dict
            Time series decomposition.
        reasoning : Dict
            Chain-of-thought reasoning.
            
        Returns:
        --------
        List[Dict]
            Identified patterns.
        """
        # Extract from reasoning if available
        if "patterns" in reasoning:
            return reasoning["patterns"]
        
        patterns = []
        
        # Ensure sufficient data
        if len(historical_data) < 20:
            return patterns
        
        # Calculate returns
        returns = np.log(historical_data['close'] / historical_data['close'].shift(1)).dropna()
        
        # Check for trend
        trend_window = min(20, len(historical_data) // 3)
        sma = historical_data['close'].rolling(window=trend_window).mean()
        
        if sma.iloc[-1] > sma.iloc[-trend_window] * 1.05:
            # Uptrend (5% increase over window)
            patterns.append({
                "pattern": "uptrend",
                "description": f"Uptrend detected over the last {trend_window} days",
                "strength": "strong" if sma.iloc[-1] > sma.iloc[-trend_window] * 1.1 else "moderate"
            })
        elif sma.iloc[-1] < sma.iloc[-trend_window] * 0.95:
            # Downtrend (5% decrease over window)
            patterns.append({
                "pattern": "downtrend",
                "description": f"Downtrend detected over the last {trend_window} days",
                "strength": "strong" if sma.iloc[-1] < sma.iloc[-trend_window] * 0.9 else "moderate"
            })
        else:
            # Sideways
            patterns.append({
                "pattern": "sideways",
                "description": f"Sideways movement detected over the last {trend_window} days",
                "strength": "moderate"
            })
        
        # Check for seasonality
        if len(historical_data) >= 20:
            # Check day-of-week effect
            day_of_week_returns = returns.groupby(returns.index.dayofweek).mean()
            max_day = day_of_week_returns.idxmax()
            min_day = day_of_week_returns.idxmin()
            
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            
            if day_of_week_returns[max_day] > 0 and day_of_week_returns[max_day] > 2 * returns.mean():
                patterns.append({
                    "pattern": "day_of_week_effect",
                    "description": f"Positive returns tend to occur on {day_names[max_day]}",
                    "strength": "moderate"
                })
            
            if day_of_week_returns[min_day] < 0 and day_of_week_returns[min_day] < 2 * returns.mean():
                patterns.append({
                    "pattern": "day_of_week_effect",
                    "description": f"Negative returns tend to occur on {day_names[min_day]}",
                    "strength": "moderate"
                })
        
        # Check for mean reversion
        if len(returns) >= 20:
            # Calculate autocorrelation
            autocorr = returns.autocorr(lag=1)
            
            if autocorr < -0.2:
                patterns.append({
                    "pattern": "mean_reversion",
                    "description": "Mean reversion pattern detected (negative autocorrelation)",
                    "strength": "strong" if autocorr < -0.3 else "moderate"
                })
            elif autocorr > 0.2:
                patterns.append({
                    "pattern": "momentum",
                    "description": "Momentum pattern detected (positive autocorrelation)",
                    "strength": "strong" if autocorr > 0.3 else "moderate"
                })
        
        # Check for volatility clustering
        if len(returns) >= 30:
            # Calculate squared returns
            squared_returns = returns ** 2
            
            # Calculate autocorrelation of squared returns
            vol_autocorr = squared_returns.autocorr(lag=1)
            
            if vol_autocorr > 0.2:
                patterns.append({
                    "pattern": "volatility_clustering",
                    "description": "Volatility clustering detected (periods of high/low volatility tend to persist)",
                    "strength": "strong" if vol_autocorr > 0.3 else "moderate"
                })
        
        return patterns
    
    def _calculate_forecast_confidence(self, historical_data: pd.DataFrame, 
                                      forecast: Dict, decomposition: Dict) -> Dict:
        """
        Calculate confidence intervals for forecast.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical price data.
        forecast : Dict
            Forecast results.
        decomposition : Dict
            Time series decomposition.
            
        Returns:
        --------
        Dict
            Confidence intervals.
        """
        # Ensure sufficient data
        if len(historical_data) < 20 or "values" not in forecast:
            return {
                "lower_95": forecast.get("values", []),
                "upper_95": forecast.get("values", [])
            }
        
        # Calculate historical volatility
        returns = np.log(historical_data['close'] / historical_data['close'].shift(1)).dropna()
        volatility = returns.std()
        
        # Calculate confidence intervals
        forecast_values = forecast["values"]
        lower_95 = []
        upper_95 = []
        
        for i, value in enumerate(forecast_values):
            # Wider intervals for longer horizons
            horizon_factor = np.sqrt(i + 1)
            interval_width = value * volatility * 1.96 * horizon_factor
            
            lower_95.append(max(0.01, value - interval_width))
            upper_95.append(value + interval_width)
        
        return {
            "lower_95": lower_95,
            "upper_95": upper_95,
            "volatility": volatility
        }
