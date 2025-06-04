"""
Market Scanner Module for Gemma Advanced Trading System.

This module provides functionality for scanning markets to identify trading opportunities.
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
import concurrent.futures
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketScanner")


class MarketScanner:
    """Class for scanning markets to identify trading opportunities."""
    
    def __init__(self, data_provider=None, indicators=None, config=None):
        """
        Initialize the MarketScanner.
        
        Parameters:
        -----------
        data_provider : object, optional
            Data provider object for fetching market data
        indicators : dict, optional
            Dictionary of indicator objects to use for scanning
        config : dict, optional
            Configuration parameters for the scanner
        """
        self.data_provider = data_provider
        self.indicators = indicators or {}
        self.config = config or {}
        self.default_config = {
            "max_symbols": 100,
            "lookback_periods": 100,
            "parallel_processing": True,
            "max_workers": 8,
            "cache_data": True,
            "cache_expiry": 300,  # 5 minutes
            "default_timeframe": "1d",
            "scan_interval": 3600,  # 1 hour
            "min_volume": 100000,
            "min_price": 1.0
        }
        
        # Merge default config with provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
        # Initialize data cache
        self.data_cache = {}
        self.cache_timestamps = {}
        
        logger.info("MarketScanner initialized")
        
    def set_data_provider(self, data_provider):
        """
        Set the data provider for the scanner.
        
        Parameters:
        -----------
        data_provider : object
            Data provider object for fetching market data
        """
        self.data_provider = data_provider
        logger.info("Data provider set")
        
    def add_indicator(self, name, indicator):
        """
        Add an indicator to the scanner.
        
        Parameters:
        -----------
        name : str
            Name of the indicator
        indicator : object
            Indicator object
        """
        self.indicators[name] = indicator
        logger.info(f"Added indicator: {name}")
        
    def get_market_data(self, symbol, timeframe=None, lookback_periods=None):
        """
        Get market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        timeframe : str, optional
            Timeframe for the data (e.g., "1d", "1h", "15m")
        lookback_periods : int, optional
            Number of periods to look back
            
        Returns:
        --------
        pandas.DataFrame
            Market data for the symbol
        """
        if self.data_provider is None:
            logger.error("No data provider set")
            return None
            
        timeframe = timeframe or self.config["default_timeframe"]
        lookback_periods = lookback_periods or self.config["lookback_periods"]
        
        # Check cache if enabled
        if self.config["cache_data"]:
            cache_key = f"{symbol}_{timeframe}_{lookback_periods}"
            current_time = time.time()
            
            if (cache_key in self.data_cache and 
                current_time - self.cache_timestamps.get(cache_key, 0) < self.config["cache_expiry"]):
                logger.debug(f"Using cached data for {symbol}")
                return self.data_cache[cache_key]
        
        try:
            logger.info(f"Fetching data for {symbol} ({timeframe}, {lookback_periods} periods)")
            data = self.data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=lookback_periods
            )
            
            # Cache the data if enabled
            if self.config["cache_data"]:
                cache_key = f"{symbol}_{timeframe}_{lookback_periods}"
                self.data_cache[cache_key] = data
                self.cache_timestamps[cache_key] = time.time()
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
            
    def scan_symbol(self, symbol, scan_criteria, timeframe=None):
        """
        Scan a single symbol for trading opportunities.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        scan_criteria : dict
            Dictionary of scan criteria
        timeframe : str, optional
            Timeframe for the data
            
        Returns:
        --------
        dict
            Scan results for the symbol
        """
        timeframe = timeframe or self.config["default_timeframe"]
        
        # Get market data
        data = self.get_market_data(symbol, timeframe)
        if data is None or len(data) < 2:
            logger.warning(f"Insufficient data for {symbol}")
            return None
            
        # Apply scan criteria
        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "criteria_met": [],
            "criteria_failed": [],
            "score": 0,
            "signals": {}
        }
        
        try:
            # Check minimum requirements
            if "close" in data.columns and "volume" in data.columns:
                current_price = data["close"].iloc[-1]
                current_volume = data["volume"].iloc[-1]
                
                if current_price < self.config["min_price"]:
                    logger.debug(f"{symbol} price below minimum: {current_price}")
                    results["criteria_failed"].append("min_price")
                    return results
                    
                if current_volume < self.config["min_volume"]:
                    logger.debug(f"{symbol} volume below minimum: {current_volume}")
                    results["criteria_failed"].append("min_volume")
                    return results
            
            # Apply each scan criterion
            for criterion_name, criterion in scan_criteria.items():
                criterion_type = criterion.get("type")
                
                if criterion_type == "indicator":
                    # Apply indicator-based criterion
                    indicator_name = criterion.get("indicator")
                    if indicator_name in self.indicators:
                        indicator = self.indicators[indicator_name]
                        params = criterion.get("params", {})
                        
                        # Call the indicator function
                        if hasattr(indicator, criterion.get("method", "")):
                            method = getattr(indicator, criterion.get("method"))
                            
                            # Prepare arguments based on required inputs
                            args = {}
                            for param_name, column_name in params.items():
                                if column_name in data.columns:
                                    args[param_name] = data[column_name]
                            
                            # Call the method with unpacked arguments
                            indicator_result = method(**args)
                            
                            # Apply the condition
                            condition = criterion.get("condition", {})
                            condition_type = condition.get("type")
                            
                            if condition_type == "crossover":
                                # Check for crossover
                                value1 = indicator_result
                                value2 = condition.get("value")
                                
                                if isinstance(value2, str) and value2 in data.columns:
                                    value2 = data[value2]
                                    
                                if len(value1) >= 2 and (
                                    (value1.iloc[-2] < value2 and value1.iloc[-1] > value2) or
                                    (condition.get("direction") == "down" and 
                                     value1.iloc[-2] > value2 and value1.iloc[-1] < value2)
                                ):
                                    results["criteria_met"].append(criterion_name)
                                    results["score"] += criterion.get("weight", 1)
                                    results["signals"][criterion_name] = {
                                        "value": float(value1.iloc[-1]),
                                        "crossed": float(value2) if not isinstance(value2, pd.Series) else float(value2.iloc[-1]),
                                        "direction": "up" if value1.iloc[-1] > value2 else "down"
                                    }
                                else:
                                    results["criteria_failed"].append(criterion_name)
                                    
                            elif condition_type == "threshold":
                                # Check for threshold
                                value = indicator_result.iloc[-1]
                                threshold = condition.get("value")
                                operator = condition.get("operator", ">")
                                
                                if (operator == ">" and value > threshold) or \
                                   (operator == "<" and value < threshold) or \
                                   (operator == ">=" and value >= threshold) or \
                                   (operator == "<=" and value <= threshold) or \
                                   (operator == "==" and value == threshold):
                                    results["criteria_met"].append(criterion_name)
                                    results["score"] += criterion.get("weight", 1)
                                    results["signals"][criterion_name] = {
                                        "value": float(value),
                                        "threshold": threshold,
                                        "operator": operator
                                    }
                                else:
                                    results["criteria_failed"].append(criterion_name)
                                    
                            elif condition_type == "pattern":
                                # Check for pattern
                                pattern_result = indicator_result.iloc[-1]
                                if pattern_result == 1:
                                    results["criteria_met"].append(criterion_name)
                                    results["score"] += criterion.get("weight", 1)
                                    results["signals"][criterion_name] = {
                                        "pattern_detected": True
                                    }
                                else:
                                    results["criteria_failed"].append(criterion_name)
                
                elif criterion_type == "price_action":
                    # Apply price action criterion
                    action_type = criterion.get("action_type")
                    
                    if action_type == "gap":
                        # Check for gap
                        if len(data) >= 2:
                            prev_close = data["close"].iloc[-2]
                            current_open = data["open"].iloc[-1]
                            gap_percent = (current_open - prev_close) / prev_close * 100
                            
                            min_gap = criterion.get("min_gap", 1.0)
                            direction = criterion.get("direction", "up")
                            
                            if (direction == "up" and gap_percent > min_gap) or \
                               (direction == "down" and gap_percent < -min_gap):
                                results["criteria_met"].append(criterion_name)
                                results["score"] += criterion.get("weight", 1)
                                results["signals"][criterion_name] = {
                                    "gap_percent": float(gap_percent),
                                    "direction": direction
                                }
                            else:
                                results["criteria_failed"].append(criterion_name)
                                
                    elif action_type == "breakout":
                        # Check for breakout
                        periods = criterion.get("periods", 20)
                        if len(data) >= periods:
                            lookback_data = data.iloc[-periods:]
                            
                            if criterion.get("direction", "up") == "up":
                                resistance = lookback_data["high"].max()
                                current_price = data["close"].iloc[-1]
                                
                                if current_price > resistance:
                                    results["criteria_met"].append(criterion_name)
                                    results["score"] += criterion.get("weight", 1)
                                    results["signals"][criterion_name] = {
                                        "breakout_level": float(resistance),
                                        "current_price": float(current_price),
                                        "direction": "up"
                                    }
                                else:
                                    results["criteria_failed"].append(criterion_name)
                            else:
                                support = lookback_data["low"].min()
                                current_price = data["close"].iloc[-1]
                                
                                if current_price < support:
                                    results["criteria_met"].append(criterion_name)
                                    results["score"] += criterion.get("weight", 1)
                                    results["signals"][criterion_name] = {
                                        "breakout_level": float(support),
                                        "current_price": float(current_price),
                                        "direction": "down"
                                    }
                                else:
                                    results["criteria_failed"].append(criterion_name)
                
                elif criterion_type == "fundamental":
                    # Apply fundamental criterion
                    # This would typically require additional data from the data provider
                    # For now, we'll just log that this criterion type is not implemented
                    logger.warning(f"Fundamental criterion {criterion_name} not implemented")
                    results["criteria_failed"].append(criterion_name)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {str(e)}")
            return None
            
    def scan_market(self, symbols, scan_criteria, timeframe=None):
        """
        Scan multiple symbols for trading opportunities.
        
        Parameters:
        -----------
        symbols : list
            List of trading symbols
        scan_criteria : dict
            Dictionary of scan criteria
        timeframe : str, optional
            Timeframe for the data
            
        Returns:
        --------
        list
            List of scan results for each symbol
        """
        timeframe = timeframe or self.config["default_timeframe"]
        
        # Limit the number of symbols if necessary
        if len(symbols) > self.config["max_symbols"]:
            logger.warning(f"Limiting scan to {self.config['max_symbols']} symbols")
            symbols = symbols[:self.config["max_symbols"]]
            
        logger.info(f"Scanning {len(symbols)} symbols with {len(scan_criteria)} criteria")
        
        results = []
        
        if self.config["parallel_processing"]:
            # Use parallel processing
            scan_func = partial(self.scan_symbol, scan_criteria=scan_criteria, timeframe=timeframe)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
                futures = {executor.submit(scan_func, symbol): symbol for symbol in symbols}
                
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error scanning {symbol}: {str(e)}")
        else:
            # Use sequential processing
            for symbol in symbols:
                result = self.scan_symbol(symbol, scan_criteria, timeframe)
                if result is not None:
                    results.append(result)
                    
        # Sort results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Scan completed. Found {len(results)} results.")
        
        return results
        
    def filter_scan_results(self, scan_results, min_score=1, required_criteria=None):
        """
        Filter scan results based on criteria.
        
        Parameters:
        -----------
        scan_results : list
            List of scan results
        min_score : int
            Minimum score required
        required_criteria : list, optional
            List of criteria that must be met
            
        Returns:
        --------
        list
            Filtered scan results
        """
        filtered_results = []
        
        for result in scan_results:
            if result["score"] < min_score:
                continue
                
            if required_criteria is not None:
                if not all(criterion in result["criteria_met"] for criterion in required_criteria):
                    continue
                    
            filtered_results.append(result)
            
        logger.info(f"Filtered {len(scan_results)} results to {len(filtered_results)}")
        
        return filtered_results
        
    def create_scan_definition(self, name, description=None):
        """
        Create a new scan definition.
        
        Parameters:
        -----------
        name : str
            Name of the scan
        description : str, optional
            Description of the scan
            
        Returns:
        --------
        dict
            Scan definition template
        """
        scan_definition = {
            "name": name,
            "description": description or f"Scan definition for {name}",
            "created_at": datetime.now().isoformat(),
            "criteria": {},
            "min_score": 1,
            "required_criteria": [],
            "timeframe": self.config["default_timeframe"]
        }
        
        return scan_definition
        
    def add_indicator_criterion(self, scan_definition, name, indicator, method, params, condition, weight=1):
        """
        Add an indicator-based criterion to a scan definition.
        
        Parameters:
        -----------
        scan_definition : dict
            Scan definition
        name : str
            Name of the criterion
        indicator : str
            Name of the indicator
        method : str
            Method of the indicator to call
        params : dict
            Parameters for the indicator method
        condition : dict
            Condition to check
        weight : int
            Weight of the criterion
            
        Returns:
        --------
        dict
            Updated scan definition
        """
        scan_definition["criteria"][name] = {
            "type": "indicator",
            "indicator": indicator,
            "method": method,
            "params": params,
            "condition": condition,
            "weight": weight
        }
        
        return scan_definition
        
    def add_price_action_criterion(self, scan_definition, name, action_type, params, weight=1):
        """
        Add a price action criterion to a scan definition.
        
        Parameters:
        -----------
        scan_definition : dict
            Scan definition
        name : str
            Name of the criterion
        action_type : str
            Type of price action
        params : dict
            Parameters for the price action
        weight : int
            Weight of the criterion
            
        Returns:
        --------
        dict
            Updated scan definition
        """
        scan_definition["criteria"][name] = {
            "type": "price_action",
            "action_type": action_type,
            "weight": weight,
            **params
        }
        
        return scan_definition
        
    def add_fundamental_criterion(self, scan_definition, name, metric, condition, weight=1):
        """
        Add a fundamental criterion to a scan definition.
        
        Parameters:
        -----------
        scan_definition : dict
            Scan definition
        name : str
            Name of the criterion
        metric : str
            Fundamental metric
        condition : dict
            Condition to check
        weight : int
            Weight of the criterion
            
        Returns:
        --------
        dict
            Updated scan definition
        """
        scan_definition["criteria"][name] = {
            "type": "fundamental",
            "metric": metric,
            "condition": condition,
            "weight": weight
        }
        
        return scan_definition
        
    def save_scan_definition(self, scan_definition, filename):
        """
        Save a scan definition to a file.
        
        Parameters:
        -----------
        scan_definition : dict
            Scan definition
        filename : str
            Filename to save to
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        import json
        
        try:
            with open(filename, 'w') as f:
                json.dump(scan_definition, f, indent=4)
            logger.info(f"Saved scan definition to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving scan definition: {str(e)}")
            return False
            
    def load_scan_definition(self, filename):
        """
        Load a scan definition from a file.
        
        Parameters:
        -----------
        filename : str
            Filename to load from
            
        Returns:
        --------
        dict
            Scan definition
        """
        import json
        
        try:
            with open(filename, 'r') as f:
                scan_definition = json.load(f)
            logger.info(f"Loaded scan definition from {filename}")
            return scan_definition
        except Exception as e:
            logger.error(f"Error loading scan definition: {str(e)}")
            return None
            
    def schedule_scan(self, scan_definition, symbols, interval=None, start_time=None, end_time=None):
        """
        Schedule a recurring market scan.
        
        Parameters:
        -----------
        scan_definition : dict
            Scan definition
        symbols : list
            List of symbols to scan
        interval : int, optional
            Interval between scans in seconds
        start_time : datetime, optional
            Start time for the scan schedule
        end_time : datetime, optional
            End time for the scan schedule
            
        Returns:
        --------
        dict
            Scan schedule
        """
        interval = interval or self.config["scan_interval"]
        
        scan_schedule = {
            "scan_definition": scan_definition,
            "symbols": symbols,
            "interval": interval,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "next_scan": datetime.now().isoformat(),
            "last_scan": None,
            "scan_count": 0,
            "active": True
        }
        
        logger.info(f"Scheduled scan {scan_definition['name']} with {len(symbols)} symbols")
        
        return scan_schedule
        
    def run_scheduled_scan(self, scan_schedule):
        """
        Run a scheduled market scan.
        
        Parameters:
        -----------
        scan_schedule : dict
            Scan schedule
            
        Returns:
        --------
        list
            Scan results
        """
        # Check if scan is active
        if not scan_schedule.get("active", True):
            logger.info(f"Scan {scan_schedule['scan_definition']['name']} is inactive")
            return []
            
        # Check if it's time to run the scan
        next_scan = datetime.fromisoformat(scan_schedule["next_scan"])
        if datetime.now() < next_scan:
            logger.info(f"Not time for scan {scan_schedule['scan_definition']['name']} yet")
            return []
            
        # Run the scan
        scan_definition = scan_schedule["scan_definition"]
        symbols = scan_schedule["symbols"]
        
        logger.info(f"Running scheduled scan {scan_definition['name']}")
        
        results = self.scan_market(
            symbols=symbols,
            scan_criteria=scan_definition["criteria"],
            timeframe=scan_definition.get("timeframe", self.config["default_timeframe"])
        )
        
        # Filter results
        filtered_results = self.filter_scan_results(
            scan_results=results,
            min_score=scan_definition.get("min_score", 1),
            required_criteria=scan_definition.get("required_criteria")
        )
        
        # Update scan schedule
        scan_schedule["last_scan"] = datetime.now().isoformat()
        scan_schedule["next_scan"] = (datetime.now() + timedelta(seconds=scan_schedule["interval"])).isoformat()
        scan_schedule["scan_count"] += 1
        
        logger.info(f"Completed scheduled scan {scan_definition['name']}. Found {len(filtered_results)} results.")
        
        return filtered_results
        
    def create_predefined_scans(self):
        """
        Create a set of predefined scan definitions.
        
        Returns:
        --------
        dict
            Dictionary of predefined scan definitions
        """
        predefined_scans = {}
        
        # Bullish Momentum Scan
        bullish_momentum = self.create_scan_definition(
            name="Bullish Momentum",
            description="Scan for stocks with strong bullish momentum"
        )
        
        self.add_indicator_criterion(
            scan_definition=bullish_momentum,
            name="rsi_bullish",
            indicator="momentum_indicators",
            method="rsi",
            params={"data": "close"},
            condition={"type": "threshold", "operator": ">", "value": 50},
            weight=1
        )
        
        self.add_indicator_criterion(
            scan_definition=bullish_momentum,
            name="macd_bullish",
            indicator="trend_indicators",
            method="macd",
            params={"data": "close"},
            condition={"type": "crossover", "value": 0, "direction": "up"},
            weight=2
        )
        
        self.add_price_action_criterion(
            scan_definition=bullish_momentum,
            name="price_above_ma",
            action_type="ma_relation",
            params={"ma_period": 50, "direction": "above"},
            weight=1
        )
        
        predefined_scans["bullish_momentum"] = bullish_momentum
        
        # Breakout Scan
        breakout = self.create_scan_definition(
            name="Breakout",
            description="Scan for stocks breaking out of resistance levels"
        )
        
        self.add_price_action_criterion(
            scan_definition=breakout,
            name="resistance_breakout",
            action_type="breakout",
            params={"periods": 20, "direction": "up"},
            weight=2
        )
        
        self.add_indicator_criterion(
            scan_definition=breakout,
            name="volume_surge",
            indicator="volume_indicators",
            method="on_balance_volume",
            params={"close": "close", "volume": "volume"},
            condition={"type": "threshold", "operator": ">", "value": 0},
            weight=1
        )
        
        predefined_scans["breakout"] = breakout
        
        # Oversold Bounce Scan
        oversold_bounce = self.create_scan_definition(
            name="Oversold Bounce",
            description="Scan for oversold stocks showing signs of reversal"
        )
        
        self.add_indicator_criterion(
            scan_definition=oversold_bounce,
            name="rsi_oversold",
            indicator="momentum_indicators",
            method="rsi",
            params={"data": "close"},
            condition={"type": "threshold", "operator": "<", "value": 30},
            weight=2
        )
        
        self.add_indicator_criterion(
            scan_definition=oversold_bounce,
            name="stoch_oversold",
            indicator="momentum_indicators",
            method="stochastic",
            params={"high": "high", "low": "low", "close": "close"},
            condition={"type": "threshold", "operator": "<", "value": 20},
            weight=1
        )
        
        predefined_scans["oversold_bounce"] = oversold_bounce
        
        # Volume Breakout Scan
        volume_breakout = self.create_scan_definition(
            name="Volume Breakout",
            description="Scan for stocks with unusual volume and price movement"
        )
        
        self.add_indicator_criterion(
            scan_definition=volume_breakout,
            name="volume_surge",
            indicator="volume_indicators",
            method="volume_oscillator",
            params={"volume": "volume"},
            condition={"type": "threshold", "operator": ">", "value": 20},
            weight=2
        )
        
        self.add_price_action_criterion(
            scan_definition=volume_breakout,
            name="price_change",
            action_type="price_change",
            params={"min_change": 2.0, "direction": "up"},
            weight=1
        )
        
        predefined_scans["volume_breakout"] = volume_breakout
        
        logger.info(f"Created {len(predefined_scans)} predefined scan definitions")
        
        return predefined_scans
