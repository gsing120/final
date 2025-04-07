#!/usr/bin/env python3
"""
Gemma Advanced Trading System - Main Application

This is the main entry point for the Gemma Advanced Trading System.
It provides a unified interface to access all system functionality.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import system components
from indicators.trend_indicators import (
    simple_moving_average, exponential_moving_average, 
    moving_average_convergence_divergence, bollinger_bands
)
from indicators.momentum_indicators import (
    relative_strength_index, stochastic_oscillator,
    commodity_channel_index, williams_percent_r
)
from indicators.volatility_indicators import (
    average_true_range, standard_deviation
)
from indicators.volume_indicators import (
    on_balance_volume, volume_weighted_average_price
)
from risk_management.core import RiskManager
from risk_management.position_sizing import calculate_position_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemma_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GemmaTrading")

class MarketDataFetcher:
    """Fetches market data for specified tickers."""
    
    def __init__(self):
        """Initialize the market data fetcher."""
        self.logger = logging.getLogger("GemmaTrading.MarketDataFetcher")
    
    def fetch_data(self, ticker, start_date=None, end_date=None, period="1y"):
        """
        Fetch market data for the specified ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to fetch data for
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
        period : str, optional
            Period to fetch data for (e.g., "1d", "1mo", "1y")
            
        Returns:
        --------
        pandas.DataFrame
            Market data with columns: Open, High, Low, Close, Volume, Adj Close
        """
        self.logger.info(f"Fetching market data for {ticker}")
        
        try:
            # For demonstration purposes, generate synthetic data
            # In a real implementation, this would use yfinance, Alpha Vantage, or another data source
            if not start_date:
                end_date = datetime.now()
                if period == "1y":
                    start_date = end_date - timedelta(days=365)
                elif period == "6mo":
                    start_date = end_date - timedelta(days=182)
                elif period == "3mo":
                    start_date = end_date - timedelta(days=91)
                elif period == "1mo":
                    start_date = end_date - timedelta(days=30)
                else:
                    start_date = end_date - timedelta(days=365)
            else:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
                if end_date:
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                else:
                    end_date = datetime.now()
            
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate synthetic price data with a trend and some volatility
            base_price = 100.0
            trend = np.linspace(0, 20, len(date_range))
            volatility = np.random.normal(0, 1, len(date_range))
            
            # Create price series with some seasonality and momentum
            close_prices = base_price + trend + volatility * 5 + np.sin(np.linspace(0, 10, len(date_range))) * 10
            
            # Create high, low, open prices based on close
            high_prices = close_prices + np.random.uniform(0.5, 2.0, len(date_range))
            low_prices = close_prices - np.random.uniform(0.5, 2.0, len(date_range))
            open_prices = low_prices + np.random.uniform(0, 1, len(date_range)) * (high_prices - low_prices)
            
            # Generate volume data
            volume = np.random.randint(100000, 1000000, len(date_range))
            
            # Create DataFrame
            data = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Adj Close': close_prices,
                'Volume': volume
            }, index=date_range)
            
            self.logger.info(f"Successfully fetched data for {ticker}: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise

class StrategyBuilder:
    """Builds trading strategies based on specified parameters."""
    
    def __init__(self):
        """Initialize the strategy builder."""
        self.logger = logging.getLogger("GemmaTrading.StrategyBuilder")
        self.market_data_fetcher = MarketDataFetcher()
    
    def generate_swing_trading_strategy(self, ticker, parameters=None):
        """
        Generate a swing trading strategy for the specified ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to generate a strategy for
        parameters : dict, optional
            Strategy parameters
            
        Returns:
        --------
        dict
            Strategy definition and analysis
        """
        self.logger.info(f"Generating swing trading strategy for {ticker}")
        
        # Set default parameters if not provided
        if parameters is None:
            parameters = {
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'sma_short_period': 20,
                'sma_long_period': 50,
                'atr_period': 14,
                'risk_per_trade': 0.02,  # 2% risk per trade
                'risk_reward_ratio': 2.0,  # 1:2 risk-reward ratio
                'stop_loss_atr_multiple': 2.0,  # Stop loss at 2x ATR
            }
        
        # Fetch market data
        data = self.market_data_fetcher.fetch_data(ticker, period="1y")
        
        # Calculate indicators
        data['SMA_Short'] = simple_moving_average(data['Close'], period=parameters['sma_short_period'])
        data['SMA_Long'] = simple_moving_average(data['Close'], period=parameters['sma_long_period'])
        data['RSI'] = relative_strength_index(data['Close'], period=parameters['rsi_period'])
        data['ATR'] = average_true_range(data['High'], data['Low'], data['Close'], period=parameters['atr_period'])
        
        # Generate strategy signals
        data['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Buy signal: RSI crosses above oversold level AND short SMA crosses above long SMA
        buy_condition = (
            (data['RSI'] > parameters['rsi_oversold']) & 
            (data['RSI'].shift(1) <= parameters['rsi_oversold']) &
            (data['SMA_Short'] > data['SMA_Long']) & 
            (data['SMA_Short'].shift(1) <= data['SMA_Long'].shift(1))
        )
        data.loc[buy_condition, 'Signal'] = 1
        
        # Sell signal: RSI crosses below overbought level OR short SMA crosses below long SMA
        sell_condition = (
            (data['RSI'] < parameters['rsi_overbought']) & 
            (data['RSI'].shift(1) >= parameters['rsi_overbought']) |
            (data['SMA_Short'] < data['SMA_Long']) & 
            (data['SMA_Short'].shift(1) >= data['SMA_Long'].shift(1))
        )
        data.loc[sell_condition, 'Signal'] = -1
        
        # Calculate stop loss and take profit levels for each buy signal
        data['Stop_Loss'] = np.nan
        data['Take_Profit'] = np.nan
        
        for i in range(1, len(data)):
            if data['Signal'].iloc[i] == 1:  # Buy signal
                entry_price = data['Close'].iloc[i]
                atr_value = data['ATR'].iloc[i]
                
                # Set stop loss at entry_price - (ATR * multiple)
                stop_loss = entry_price - (atr_value * parameters['stop_loss_atr_multiple'])
                
                # Set take profit based on risk-reward ratio
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * parameters['risk_reward_ratio'])
                
                data.loc[data.index[i], 'Stop_Loss'] = stop_loss
                data.loc[data.index[i], 'Take_Profit'] = take_profit
        
        # Backtest the strategy
        initial_capital = 10000.0
        position = 0
        capital = initial_capital
        trades = []
        
        for i in range(1, len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            signal = data['Signal'].iloc[i]
            
            if position == 0 and signal == 1:  # Enter long position
                # Calculate position size based on risk
                stop_loss = data['Stop_Loss'].iloc[i]
                risk_amount = initial_capital * parameters['risk_per_trade']
                risk_per_share = current_price - stop_loss
                position_size = risk_amount / risk_per_share
                position_value = position_size * current_price
                
                # Record trade entry
                trades.append({
                    'type': 'entry',
                    'date': current_date,
                    'price': current_price,
                    'position_size': position_size,
                    'position_value': position_value,
                    'stop_loss': stop_loss,
                    'take_profit': data['Take_Profit'].iloc[i]
                })
                
                position = 1
                
            elif position == 1:  # Check for exit conditions
                entry_trade = trades[-1]
                stop_loss = entry_trade['stop_loss']
                take_profit = entry_trade['take_profit']
                position_size = entry_trade['position_size']
                
                # Check if stop loss or take profit was hit
                if current_price <= stop_loss:  # Stop loss hit
                    # Record trade exit
                    trades.append({
                        'type': 'exit',
                        'date': current_date,
                        'price': stop_loss,
                        'position_size': position_size,
                        'position_value': position_size * stop_loss,
                        'exit_reason': 'stop_loss',
                        'profit_loss': position_size * (stop_loss - entry_trade['price'])
                    })
                    position = 0
                    capital += position_size * (stop_loss - entry_trade['price'])
                    
                elif current_price >= take_profit:  # Take profit hit
                    # Record trade exit
                    trades.append({
                        'type': 'exit',
                        'date': current_date,
                        'price': take_profit,
                        'position_size': position_size,
                        'position_value': position_size * take_profit,
                        'exit_reason': 'take_profit',
                        'profit_loss': position_size * (take_profit - entry_trade['price'])
                    })
                    position = 0
                    capital += position_size * (take_profit - entry_trade['price'])
                    
                elif signal == -1:  # Sell signal
                    # Record trade exit
                    trades.append({
                        'type': 'exit',
                        'date': current_date,
                        'price': current_price,
                        'position_size': position_size,
                        'position_value': position_size * current_price,
                        'exit_reason': 'signal',
                        'profit_loss': position_size * (current_price - entry_trade['price'])
                    })
                    position = 0
                    capital += position_size * (current_price - entry_trade['price'])
        
        # Calculate strategy performance metrics
        if trades:
            # Filter for completed trades (entry and exit pairs)
            completed_trades = []
            for i in range(0, len(trades), 2):
                if i + 1 < len(trades):
                    entry = trades[i]
                    exit = trades[i + 1]
                    completed_trades.append({
                        'entry_date': entry['date'],
                        'exit_date': exit['date'],
                        'entry_price': entry['price'],
                        'exit_price': exit['price'],
                        'position_size': entry['position_size'],
                        'profit_loss': exit['profit_loss'],
                        'exit_reason': exit['exit_reason'],
                        'duration': (exit['date'] - entry['date']).days
                    })
            
            # Calculate performance metrics
            total_trades = len(completed_trades)
            winning_trades = sum(1 for trade in completed_trades if trade['profit_loss'] > 0)
            losing_trades = sum(1 for trade in completed_trades if trade['profit_loss'] <= 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = sum(trade['profit_loss'] for trade in completed_trades if trade['profit_loss'] > 0)
            total_loss = sum(trade['profit_loss'] for trade in completed_trades if trade['profit_loss'] <= 0)
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            
            average_win = total_profit / winning_trades if winning_trades > 0 else 0
            average_loss = total_loss / losing_trades if losing_trades > 0 else 0
            
            final_capital = capital
            total_return = (final_capital - initial_capital) / initial_capital * 100
            
            # Calculate drawdown
            equity_curve = [initial_capital]
            for trade in completed_trades:
                equity_curve.append(equity_curve[-1] + trade['profit_loss'])
            
            peak = equity_curve[0]
            max_drawdown = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            performance = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_win': average_win,
                'average_loss': average_loss,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'final_capital': final_capital
            }
        else:
            performance = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_win': 0,
                'average_loss': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'final_capital': initial_capital
            }
        
        # Generate strategy summary
        strategy = {
            'ticker': ticker,
            'strategy_type': 'Swing Trading',
            'parameters': parameters,
            'signals': data[['Close', 'SMA_Short', 'SMA_Long', 'RSI', 'ATR', 'Signal', 'Stop_Loss', 'Take_Profit']].dropna(),
            'trades': trades,
            'completed_trades': completed_trades if trades else [],
            'performance': performance,
            'description': f"""
                Swing Trading Strategy for {ticker}
                
                Entry Conditions:
                - RSI crosses above {parameters['rsi_oversold']} (oversold level)
                - {parameters['sma_short_period']}-day SMA crosses above {parameters['sma_long_period']}-day SMA
                
                Exit Conditions:
                - RSI crosses below {parameters['rsi_overbought']} (overbought level)
                - {parameters['sma_short_period']}-day SMA crosses below {parameters['sma_long_period']}-day SMA
                - Stop Loss: Entry Price - ({parameters['stop_loss_atr_multiple']} x ATR)
                - Take Profit: Entry Price + (Risk x {parameters['risk_reward_ratio']})
                
                Risk Management:
                - Risk per trade: {parameters['risk_per_trade'] * 100}% of capital
                - Risk-Reward Ratio: 1:{parameters['risk_reward_ratio']}
            """
        }
        
        self.logger.info(f"Successfully generated swing trading strategy for {ticker}")
        return strategy

class GemmaAdvancedTradingSystem:
    """Main application class for the Gemma Advanced Trading System."""
    
    def __init__(self):
        """Initialize the Gemma Advanced Trading System."""
        self.logger = logging.getLogger("GemmaTrading.Main")
        self.strategy_builder = StrategyBuilder()
        self.logger.info("Gemma Advanced Trading System initialized")
    
    def generate_strategy(self, ticker, strategy_type, parameters=None):
        """
        Generate a trading strategy for the specified ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to generate a strategy for
        strategy_type : str
            Type of strategy to generate (e.g., "swing", "day", "options")
        parameters : dict, optional
            Strategy parameters
            
        Returns:
        --------
        dict
            Strategy definition and analysis
        """
        self.logger.info(f"Generating {strategy_type} strategy for {ticker}")
        
        if strategy_type.lower() == "swing":
            return self.strategy_builder.generate_swing_trading_strategy(ticker, parameters)
        else:
            raise ValueError(f"Strategy type '{strategy_type}' not supported")
    
    def run(self, args):
        """
        Run the Gemma Advanced Trading System with the specified arguments.
        
        Parameters:
        -----------
        args : argparse.Namespace
            Command-line arguments
        """
        self.logger.info(f"Running Gemma Advanced Trading System with args: {args}")
        
        if args.command == "generate":
            strategy = self.generate_strategy(args.ticker, args.strategy_type)
            
            # Print strategy summary
            print("\n" + "="*80)
            print(f"GEMMA ADVANCED TRADING SYSTEM - {args.strategy_type.upper()} STRATEGY FOR {args.ticker}")
            print("="*80)
            print(strategy['description'])
            print("\nPERFORMANCE METRICS:")
            print(f"Total Trades: {strategy['performance']['total_trades']}")
            print(f"Win Rate: {strategy['performance']['win_rate']*100:.2f}%")
            print(f"Profit Factor: {strategy['performance']['profit_factor']:.2f}")
            print(f"Total Return: {strategy['performance']['total_return']:.2f}%")
            print(f"Max Drawdown: {strategy['performance']['max_drawdown']:.2f}%")
            print(f"Final Capital: ${strategy['performance']['final_capital']:.2f}")
            
            # Print recent signals
            print("\nRECENT SIGNALS:")
            recent_signals = strategy['signals'].tail(10)
            for date, row in recent_signals.iterrows():
                signal_type = "BUY" if row['Signal'] == 1 else "SELL" if row['Signal'] == -1 else "HOLD"
                print(f"{date.date()} - {signal_type} - Close: ${row['Close']:.2f}, RSI: {row['RSI']:.2f}")
                if row['Signal'] == 1:
                    print(f"  Stop Loss: ${row['Stop_Loss']:.2f}, Take Profit: ${row['Take_Profit']:.2f}")
            
            # Print recent trades
            if strategy['completed_trades']:
                print("\nRECENT TRADES:")
                recent_trades = strategy['completed_trades'][-5:]
                for trade in recent_trades:
                    profit_loss = trade['profit_loss']
                    result = "WIN" if profit_loss > 0 else "LOSS"
                    print(f"{trade['entry_date'].date()} to {trade['exit_date'].date()} - {result} - ${profit_loss:.2f}")
                    print(f"  Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}, Reason: {trade['exit_reason']}")
            
            print("\nSTRATEGY RECOMMENDATIONS:")
            if strategy['performance']['win_rate'] > 0.5 and strategy['performance']['profit_factor'] > 1.5:
                print("✅ This strategy shows promising results and could be considered for implementation.")
                if strategy['performance']['max_drawdown'] > 20:
                    print("⚠️ However, the maximum drawdown is high. Consider reducing position sizes.")
            elif strategy['performance']['win_rate'] > 0.4 and strategy['performance']['profit_factor'] > 1.0:
                print("⚠️ This strategy shows moderate results. Consider further optimization before implementation.")
            else:
                print("❌ This strategy does not show promising results. Further optimization is recommended.")
            
            print("="*80)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Gemma Advanced Trading System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Generate strategy command
    generate_parser = subparsers.add_parser("generate", help="Generate a trading strategy")
    generate_parser.add_argument("ticker", help="Ticker symbol")
    generate_parser.add_argument("--strategy-type", "-s", default="swing", choices=["swing", "day", "options"],
                                help="Type of strategy to generate")
    
    return parser.parse_args()

def main():
    """Main entry point for the Gemma Advanced Trading System."""
    args = parse_args()
    
    if not args.command:
        print("Error: No command specified. Use --help for usage information.")
        return 1
    
    app = GemmaAdvancedTradingSystem()
    app.run(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())
