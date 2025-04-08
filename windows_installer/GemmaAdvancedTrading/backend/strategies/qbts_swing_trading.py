"""
QBTS (Quantitative Behavioral Trading Strategy) Swing Trading Strategy

This module implements a QBTS swing trading strategy using the Gemma 3 integration.
The strategy combines technical analysis, sentiment analysis, and market regime detection
to identify high-probability swing trading opportunities.
"""

import os
import sys
import logging
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Import Gemma 3 integration
from backend.gemma3_integration.gemma3_integration import Gemma3Integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class QBTSSwingTradingStrategy:
    """
    QBTS (Quantitative Behavioral Trading Strategy) Swing Trading Strategy
    
    This class implements a swing trading strategy that combines quantitative analysis
    with behavioral finance principles, enhanced by Gemma 3's capabilities.
    """
    
    def __init__(self, gemma3: Optional[Gemma3Integration] = None):
        """
        Initialize the QBTS Swing Trading Strategy.
        
        Parameters:
        -----------
        gemma3 : Gemma3Integration, optional
            Instance of Gemma3Integration for accessing Gemma 3 capabilities.
            If None, creates a new instance.
        """
        self.logger = logging.getLogger("GemmaTrading.QBTSSwingStrategy")
        
        # Create or use provided Gemma3Integration
        self.gemma3 = gemma3 or Gemma3Integration()
        
        # Initialize strategy parameters
        self.strategy_params = {
            "name": "QBTS Swing Trading Strategy",
            "description": "A swing trading strategy that combines quantitative analysis with behavioral finance principles",
            "time_frame": "daily",
            "holding_period": {"min": 3, "max": 15, "target": 7},
            "position_sizing": {
                "default_risk_per_trade": 0.01,  # 1% risk per trade
                "max_position_size": 0.05,  # 5% of portfolio
                "position_sizing_method": "volatility_adjusted"
            },
            "entry_conditions": {
                "technical": {
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "ema_fast_period": 20,
                    "ema_slow_period": 50,
                    "volume_surge_threshold": 1.5,  # 1.5x average volume
                    "atr_period": 14,
                    "bollinger_period": 20,
                    "bollinger_std": 2.0
                },
                "sentiment": {
                    "min_news_sentiment": 0.6,  # Minimum news sentiment score for long
                    "max_news_sentiment": 0.4,  # Maximum news sentiment score for short
                    "min_social_sentiment": 0.55,  # Minimum social sentiment score for long
                    "max_social_sentiment": 0.45  # Maximum social sentiment score for short
                },
                "market_regime": {
                    "suitable_regimes_long": ["bullish", "trending_up", "recovery"],
                    "suitable_regimes_short": ["bearish", "trending_down", "distribution"]
                }
            },
            "exit_conditions": {
                "take_profit": {
                    "default": 0.03,  # 3% profit target
                    "trailing_stop": True,
                    "trailing_stop_activation": 0.02,  # Activate trailing stop after 2% profit
                    "trailing_stop_distance": 0.015  # 1.5% trailing stop
                },
                "stop_loss": {
                    "default": 0.02,  # 2% stop loss
                    "atr_multiple": 2.0,  # Alternative: 2x ATR
                    "max_loss": 0.03  # Maximum loss 3%
                },
                "time_based": {
                    "max_holding_days": 15,
                    "min_holding_days": 3
                }
            }
        }
        
        # Initialize strategy state
        self.strategy_state = {
            "current_positions": {},
            "historical_trades": [],
            "market_regime": "unknown",
            "last_analysis_time": None
        }
        
        self.logger.info("Initialized QBTS Swing Trading Strategy")
    
    def analyze_market_conditions(self, market_data: Dict[str, Any],
                                economic_data: Dict[str, Any],
                                news_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze current market conditions.
        
        Parameters:
        -----------
        market_data : Dict[str, Any]
            Market data for global markets and assets.
        economic_data : Dict[str, Any]
            Economic indicator data.
        news_data : Dict[str, List[Dict[str, Any]]]
            News data for various markets.
            
        Returns:
        --------
        Dict[str, Any]
            Market conditions analysis.
        """
        self.logger.info("Analyzing market conditions")
        
        # Use Gemma 3 to generate market insights
        market_insights = self.gemma3.generate_market_insights(
            market_data=market_data,
            economic_data=economic_data,
            news_data=news_data
        )
        
        # Extract market regime
        market_regime = market_insights.get("market_conditions", {}).get("regime", "unknown")
        
        # Update strategy state
        self.strategy_state["market_regime"] = market_regime
        self.strategy_state["last_analysis_time"] = datetime.datetime.now().isoformat()
        
        self.logger.info(f"Current market regime: {market_regime}")
        return market_insights
    
    def scan_for_opportunities(self, tickers: List[str],
                             market_data: Dict[str, Any],
                             news_data: Dict[str, List[Dict[str, Any]]],
                             sentiment_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scan for swing trading opportunities.
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols to scan.
        market_data : Dict[str, Any]
            Market data for assets.
        news_data : Dict[str, List[Dict[str, Any]]]
            News data for assets.
        sentiment_data : Dict[str, Dict[str, Any]]
            Sentiment data for assets.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of trading opportunities.
        """
        self.logger.info(f"Scanning {len(tickers)} tickers for swing trading opportunities")
        
        # Define trading objectives
        trading_objectives = {
            "risk_tolerance": "moderate",
            "return_target": "high",
            "time_horizon": "medium",
            "strategy_type": "swing_trading"
        }
        
        opportunities = []
        
        for ticker in tickers:
            self.logger.info(f"Analyzing {ticker}")
            
            # Skip if we don't have data for this ticker
            if ticker not in market_data or ticker not in news_data or ticker not in sentiment_data:
                self.logger.warning(f"Missing data for {ticker}, skipping")
                continue
            
            # Generate trading recommendation using Gemma 3
            recommendation = self.gemma3.generate_trading_recommendation(
                ticker=ticker,
                market_data=market_data,
                news_data=news_data[ticker],
                sentiment_data=sentiment_data[ticker],
                trading_objectives=trading_objectives
            )
            
            # Check if recommendation is actionable
            if recommendation.get("recommendation") in ["buy", "sell"]:
                # Extract key information
                opportunity = {
                    "ticker": ticker,
                    "action": recommendation.get("recommendation"),
                    "confidence": recommendation.get("confidence", 0.0),
                    "price": recommendation.get("price", 0.0),
                    "signals": recommendation.get("signals", []),
                    "reasoning": recommendation.get("reasoning", []),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Add technical analysis
                if "best_analysis" in recommendation:
                    opportunity["technical_analysis"] = recommendation["best_analysis"].get("technical_factors", {})
                
                # Add sentiment analysis
                opportunity["sentiment"] = {
                    "news": recommendation.get("market_context", {}).get("market_news", {}).get("sentiment", "neutral"),
                    "social": recommendation.get("market_context", {}).get("sentiment_data", {}).get("social_sentiment", "neutral")
                }
                
                # Add market regime
                opportunity["market_regime"] = recommendation.get("market_context", {}).get("global_market_conditions", {}).get("regime", "unknown")
                
                # Calculate risk parameters
                opportunity["risk_parameters"] = self._calculate_risk_parameters(
                    ticker=ticker,
                    action=opportunity["action"],
                    price=opportunity["price"],
                    market_data=market_data[ticker]
                )
                
                opportunities.append(opportunity)
                self.logger.info(f"Found {opportunity['action']} opportunity for {ticker} with confidence {opportunity['confidence']}")
        
        # Sort opportunities by confidence
        opportunities.sort(key=lambda x: x["confidence"], reverse=True)
        
        self.logger.info(f"Found {len(opportunities)} swing trading opportunities")
        return opportunities
    
    def generate_trade_plan(self, opportunity: Dict[str, Any],
                          portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed trade plan for an opportunity.
        
        Parameters:
        -----------
        opportunity : Dict[str, Any]
            Trading opportunity.
        portfolio : Dict[str, Any]
            Current portfolio details.
            
        Returns:
        --------
        Dict[str, Any]
            Detailed trade plan.
        """
        self.logger.info(f"Generating trade plan for {opportunity['ticker']} {opportunity['action']}")
        
        # Extract key information
        ticker = opportunity["ticker"]
        action = opportunity["action"]
        entry_price = opportunity["price"]
        risk_parameters = opportunity["risk_parameters"]
        
        # Calculate position size
        position_size = self._calculate_position_size(
            ticker=ticker,
            action=action,
            entry_price=entry_price,
            stop_loss_price=risk_parameters["stop_loss_price"],
            portfolio=portfolio
        )
        
        # Generate trade plan
        trade_plan = {
            "ticker": ticker,
            "action": action,
            "entry_price": entry_price,
            "entry_price_range": {
                "min": entry_price * 0.99,  # 1% below entry price
                "max": entry_price * 1.01   # 1% above entry price
            },
            "position_size": position_size,
            "stop_loss": {
                "price": risk_parameters["stop_loss_price"],
                "type": "fixed",
                "risk_amount": risk_parameters["risk_amount"]
            },
            "take_profit": {
                "price": risk_parameters["take_profit_price"],
                "type": "trailing",
                "activation_price": risk_parameters["trailing_activation_price"],
                "trailing_distance": risk_parameters["trailing_distance"]
            },
            "time_frame": {
                "min_holding_days": self.strategy_params["holding_period"]["min"],
                "max_holding_days": self.strategy_params["holding_period"]["max"],
                "target_holding_days": self.strategy_params["holding_period"]["target"]
            },
            "expected_return": {
                "profit_target": risk_parameters["profit_target"],
                "risk_reward_ratio": risk_parameters["risk_reward_ratio"]
            },
            "reasoning": opportunity["reasoning"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Use Gemma 3 to explain the trade plan
        trade_explanation = self.gemma3.explain_entry_signal(
            ticker=ticker,
            signal_type=action,
            price_data=None,  # Not needed for explanation
            technical_indicators=opportunity.get("technical_analysis", {}),
            market_conditions={"regime": opportunity["market_regime"]},
            strategy=self.strategy_params
        )
        
        # Add explanation to trade plan
        trade_plan["explanation"] = trade_explanation.get("explanation", {})
        
        self.logger.info(f"Generated trade plan for {ticker} with position size {position_size}")
        return trade_plan
    
    def execute_trade_plan(self, trade_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade plan (simulated).
        
        Parameters:
        -----------
        trade_plan : Dict[str, Any]
            Detailed trade plan.
            
        Returns:
        --------
        Dict[str, Any]
            Executed trade details.
        """
        self.logger.info(f"Executing trade plan for {trade_plan['ticker']} {trade_plan['action']}")
        
        # Extract key information
        ticker = trade_plan["ticker"]
        action = trade_plan["action"]
        entry_price = trade_plan["entry_price"]
        position_size = trade_plan["position_size"]
        stop_loss_price = trade_plan["stop_loss"]["price"]
        take_profit_price = trade_plan["take_profit"]["price"]
        
        # Simulate trade execution
        executed_trade = {
            "ticker": ticker,
            "action": action,
            "entry_price": entry_price,
            "position_size": position_size,
            "quantity": int(position_size / entry_price),
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "entry_time": datetime.datetime.now().isoformat(),
            "status": "open",
            "trade_id": f"QBTS-{ticker}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "trade_plan": trade_plan
        }
        
        # Update strategy state
        self.strategy_state["current_positions"][ticker] = executed_trade
        
        self.logger.info(f"Executed {action} trade for {ticker} at {entry_price} with quantity {executed_trade['quantity']}")
        return executed_trade
    
    def monitor_positions(self, current_positions: Dict[str, Dict[str, Any]],
                        market_data: Dict[str, Any],
                        news_data: Dict[str, List[Dict[str, Any]]],
                        sentiment_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Monitor open positions and generate exit recommendations.
        
        Parameters:
        -----------
        current_positions : Dict[str, Dict[str, Any]]
            Dictionary of current positions.
        market_data : Dict[str, Any]
            Market data for assets.
        news_data : Dict[str, List[Dict[str, Any]]]
            News data for assets.
        sentiment_data : Dict[str, Dict[str, Any]]
            Sentiment data for assets.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of exit recommendations.
        """
        self.logger.info(f"Monitoring {len(current_positions)} open positions")
        
        exit_recommendations = []
        
        for ticker, position in current_positions.items():
            self.logger.info(f"Analyzing position for {ticker}")
            
            # Skip if we don't have data for this ticker
            if ticker not in market_data or ticker not in news_data or ticker not in sentiment_data:
                self.logger.warning(f"Missing data for {ticker}, skipping")
                continue
            
            # Generate exit recommendation using Gemma 3
            recommendation = self.gemma3.generate_exit_recommendation(
                ticker=ticker,
                position=position,
                market_data=market_data,
                news_data=news_data[ticker],
                sentiment_data=sentiment_data[ticker]
            )
            
            # Check if exit is recommended
            if recommendation.get("recommendation") == "exit":
                # Extract key information
                exit_rec = {
                    "ticker": ticker,
                    "action": "exit",
                    "exit_type": recommendation.get("exit_type", "normal"),
                    "confidence": recommendation.get("confidence", 0.0),
                    "price": recommendation.get("price", 0.0),
                    "signals": recommendation.get("signals", []),
                    "reasoning": recommendation.get("reasoning", []),
                    "position": position,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                exit_recommendations.append(exit_rec)
                self.logger.info(f"Generated exit recommendation for {ticker} with reason {exit_rec['exit_type']}")
        
        # Sort exit recommendations by confidence
        exit_recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        self.logger.info(f"Generated {len(exit_recommendations)} exit recommendations")
        return exit_recommendations
    
    def execute_exit(self, exit_recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an exit recommendation (simulated).
        
        Parameters:
        -----------
        exit_recommendation : Dict[str, Any]
            Exit recommendation.
            
        Returns:
        --------
        Dict[str, Any]
            Completed trade details.
        """
        self.logger.info(f"Executing exit for {exit_recommendation['ticker']}")
        
        # Extract key information
        ticker = exit_recommendation["ticker"]
        exit_price = exit_recommendation["price"]
        position = exit_recommendation["position"]
        
        # Calculate profit/loss
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        
        if position["action"] == "buy":
            profit_loss = (exit_price - entry_price) * quantity
            return_pct = (exit_price - entry_price) / entry_price
        else:  # sell
            profit_loss = (entry_price - exit_price) * quantity
            return_pct = (entry_price - exit_price) / entry_price
        
        # Create completed trade
        completed_trade = {
            "ticker": ticker,
            "action": position["action"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "entry_time": position["entry_time"],
            "exit_time": datetime.datetime.now().isoformat(),
            "holding_period_days": (datetime.datetime.now() - datetime.datetime.fromisoformat(position["entry_time"])).days,
            "profit_loss": profit_loss,
            "return_pct": return_pct,
            "exit_reason": exit_recommendation["exit_type"],
            "trade_id": position["trade_id"],
            "status": "closed",
            "original_position": position,
            "exit_recommendation": exit_recommendation
        }
        
        # Update strategy state
        if ticker in self.strategy_state["current_positions"]:
            del self.strategy_state["current_positions"][ticker]
        
        self.strategy_state["historical_trades"].append(completed_trade)
        
        # Generate post-trade analysis using Gemma 3
        post_trade_analysis = self.gemma3.generate_post_trade_analysis(
            trade=completed_trade,
            market_data={"price": None}  # Not needed for analysis
        )
        
        # Add analysis to completed trade
        completed_trade["post_trade_analysis"] = post_trade_analysis
        
        self.logger.info(f"Executed exit for {ticker} at {exit_price} with P&L {profit_loss:.2f} ({return_pct:.2%})")
        return completed_trade
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report for the strategy.
        
        Returns:
        --------
        Dict[str, Any]
            Performance report.
        """
        self.logger.info("Generating performance report")
        
        # Get historical trades
        trades = self.strategy_state["historical_trades"]
        
        if not trades:
            return {
                "message": "No historical trades to analyze",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t["profit_loss"] > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit_loss = sum(t["profit_loss"] for t in trades)
        avg_profit_loss = total_profit_loss / total_trades if total_trades > 0 else 0
        
        avg_winning_trade = sum(t["profit_loss"] for t in trades if t["profit_loss"] > 0) / winning_trades if winning_trades > 0 else 0
        avg_losing_trade = sum(t["profit_loss"] for t in trades if t["profit_loss"] <= 0) / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(sum(t["profit_loss"] for t in trades if t["profit_loss"] > 0) / sum(t["profit_loss"] for t in trades if t["profit_loss"] < 0)) if sum(t["profit_loss"] for t in trades if t["profit_loss"] < 0) != 0 else float('inf')
        
        avg_holding_period = sum(t["holding_period_days"] for t in trades) / total_trades if total_trades > 0 else 0
        
        # Create performance report
        report = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit_loss": total_profit_loss,
            "avg_profit_loss": avg_profit_loss,
            "avg_winning_trade": avg_winning_trade,
            "avg_losing_trade": avg_losing_trade,
            "profit_factor": profit_factor,
            "avg_holding_period": avg_holding_period,
            "trades_by_ticker": self._group_trades_by_ticker(trades),
            "trades_by_market_regime": self._group_trades_by_market_regime(trades),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated performance report: {total_trades} trades with {win_rate:.2%} win rate")
        return report
    
    def _calculate_risk_parameters(self, ticker: str, action: str, price: float,
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk parameters for a trade.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        action : str
            Trade action ("buy" or "sell").
        price : float
            Entry price.
        market_data : Dict[str, Any]
            Market data for the asset.
            
        Returns:
        --------
        Dict[str, Any]
            Risk parameters.
        """
        # Get price data
        price_data = market_data.get("price", None)
        
        if price_data is None or price_data.empty:
            # Use default risk parameters if no price data
            stop_loss_pct = self.strategy_params["exit_conditions"]["stop_loss"]["default"]
            take_profit_pct = self.strategy_params["exit_conditions"]["take_profit"]["default"]
            
            if action == "buy":
                stop_loss_price = price * (1 - stop_loss_pct)
                take_profit_price = price * (1 + take_profit_pct)
            else:  # sell
                stop_loss_price = price * (1 + stop_loss_pct)
                take_profit_price = price * (1 - take_profit_pct)
            
            risk_amount = abs(price - stop_loss_price)
            reward_amount = abs(take_profit_price - price)
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            return {
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "risk_amount": risk_amount,
                "reward_amount": reward_amount,
                "risk_reward_ratio": risk_reward_ratio,
                "profit_target": take_profit_pct,
                "trailing_activation_price": price * (1 + self.strategy_params["exit_conditions"]["take_profit"]["trailing_stop_activation"]) if action == "buy" else price * (1 - self.strategy_params["exit_conditions"]["take_profit"]["trailing_stop_activation"]),
                "trailing_distance": price * self.strategy_params["exit_conditions"]["take_profit"]["trailing_stop_distance"]
            }
        
        # Calculate ATR
        atr_period = self.strategy_params["entry_conditions"]["technical"]["atr_period"]
        high = price_data["high"].values
        low = price_data["low"].values
        close = price_data["close"].values
        
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
        atr = np.mean(tr[-atr_period:])
        
        # Calculate stop loss based on ATR
        atr_multiple = self.strategy_params["exit_conditions"]["stop_loss"]["atr_multiple"]
        stop_loss_distance = atr * atr_multiple
        
        if action == "buy":
            stop_loss_price = price - stop_loss_distance
            take_profit_price = price + (stop_loss_distance * 2)  # 2:1 reward-to-risk ratio
        else:  # sell
            stop_loss_price = price + stop_loss_distance
            take_profit_price = price - (stop_loss_distance * 2)  # 2:1 reward-to-risk ratio
        
        # Calculate risk parameters
        risk_amount = abs(price - stop_loss_price)
        reward_amount = abs(take_profit_price - price)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        profit_target = reward_amount / price
        
        # Calculate trailing stop parameters
        trailing_activation_pct = self.strategy_params["exit_conditions"]["take_profit"]["trailing_stop_activation"]
        trailing_distance_pct = self.strategy_params["exit_conditions"]["take_profit"]["trailing_stop_distance"]
        
        if action == "buy":
            trailing_activation_price = price * (1 + trailing_activation_pct)
            trailing_distance = price * trailing_distance_pct
        else:  # sell
            trailing_activation_price = price * (1 - trailing_activation_pct)
            trailing_distance = price * trailing_distance_pct
        
        return {
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "risk_amount": risk_amount,
            "reward_amount": reward_amount,
            "risk_reward_ratio": risk_reward_ratio,
            "profit_target": profit_target,
            "trailing_activation_price": trailing_activation_price,
            "trailing_distance": trailing_distance,
            "atr": atr
        }
    
    def _calculate_position_size(self, ticker: str, action: str, entry_price: float,
                               stop_loss_price: float, portfolio: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol.
        action : str
            Trade action ("buy" or "sell").
        entry_price : float
            Entry price.
        stop_loss_price : float
            Stop loss price.
        portfolio : Dict[str, Any]
            Portfolio details.
            
        Returns:
        --------
        float
            Position size in currency.
        """
        # Get portfolio value
        portfolio_value = portfolio.get("total_value", 100000.0)  # Default to $100,000 if not provided
        
        # Calculate risk per trade
        risk_per_trade = portfolio_value * self.strategy_params["position_sizing"]["default_risk_per_trade"]
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            self.logger.warning(f"Risk per share is zero for {ticker}, using default position size")
            return portfolio_value * 0.02  # Default to 2% of portfolio
        
        # Calculate number of shares
        num_shares = risk_per_trade / risk_per_share
        
        # Calculate position size
        position_size = num_shares * entry_price
        
        # Cap position size
        max_position_size = portfolio_value * self.strategy_params["position_sizing"]["max_position_size"]
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def _group_trades_by_ticker(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group trades by ticker and calculate performance metrics.
        
        Parameters:
        -----------
        trades : List[Dict[str, Any]]
            List of completed trades.
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Trades grouped by ticker with performance metrics.
        """
        result = {}
        
        for trade in trades:
            ticker = trade["ticker"]
            
            if ticker not in result:
                result[ticker] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_profit_loss": 0.0,
                    "trades": []
                }
            
            result[ticker]["total_trades"] += 1
            result[ticker]["total_profit_loss"] += trade["profit_loss"]
            
            if trade["profit_loss"] > 0:
                result[ticker]["winning_trades"] += 1
            else:
                result[ticker]["losing_trades"] += 1
            
            result[ticker]["trades"].append(trade)
        
        # Calculate win rate and average profit/loss
        for ticker, data in result.items():
            data["win_rate"] = data["winning_trades"] / data["total_trades"] if data["total_trades"] > 0 else 0
            data["avg_profit_loss"] = data["total_profit_loss"] / data["total_trades"] if data["total_trades"] > 0 else 0
        
        return result
    
    def _group_trades_by_market_regime(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group trades by market regime and calculate performance metrics.
        
        Parameters:
        -----------
        trades : List[Dict[str, Any]]
            List of completed trades.
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Trades grouped by market regime with performance metrics.
        """
        result = {}
        
        for trade in trades:
            # Extract market regime from original position or use unknown
            market_regime = trade.get("original_position", {}).get("trade_plan", {}).get("market_regime", "unknown")
            
            if market_regime not in result:
                result[market_regime] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_profit_loss": 0.0,
                    "trades": []
                }
            
            result[market_regime]["total_trades"] += 1
            result[market_regime]["total_profit_loss"] += trade["profit_loss"]
            
            if trade["profit_loss"] > 0:
                result[market_regime]["winning_trades"] += 1
            else:
                result[market_regime]["losing_trades"] += 1
            
            result[market_regime]["trades"].append(trade)
        
        # Calculate win rate and average profit/loss
        for regime, data in result.items():
            data["win_rate"] = data["winning_trades"] / data["total_trades"] if data["total_trades"] > 0 else 0
            data["avg_profit_loss"] = data["total_profit_loss"] / data["total_trades"] if data["total_trades"] > 0 else 0
        
        return result
