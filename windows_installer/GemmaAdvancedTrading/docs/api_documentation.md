# Gemma Advanced Trading System API Documentation

## Overview

The Gemma Advanced Trading System provides a comprehensive set of API endpoints for generating trading strategies, managing portfolios, analyzing market data, and controlling the system's memory modules. This documentation describes how to use these endpoints to interact with the system.

## Base URL

All API endpoints are accessible at the base URL: `http://localhost:5000`

## Authentication

Currently, the API does not require authentication for local development. In a production environment, proper authentication would be implemented.

## API Endpoints

### 1. Strategy Generation

#### Generate Trading Strategy

Generates a comprehensive trading strategy for a specified ticker symbol.

**Endpoint:** `/generate_strategy`

**Method:** POST

**Parameters:**
- `ticker` (string, required): The ticker symbol (e.g., "AAPL", "MSFT")
- `strategy_type` (string, optional): The type of trading strategy ("swing", "day", "position"). Default: "swing"

**Example Request:**
```bash
curl -X POST -F "ticker=AAPL" -F "strategy_type=swing" http://localhost:5000/generate_strategy
```

**Example Response:**
```json
{
  "success": true,
  "ticker": "AAPL",
  "prediction": {
    "date": "2025-04-09",
    "trend": "bearish",
    "momentum": "weak",
    "volatility": "high",
    "support": 191.39,
    "resistance": 237.13,
    "recommendation": "sell"
  },
  "performance": {
    "total_return": "8.23%",
    "sharpe_ratio": "0.43",
    "max_drawdown": "-29.87%",
    "win_rate": "100.00%",
    "volatility": "27.66%"
  },
  "trades": [
    {
      "date": "2024-05-15",
      "type": "BUY",
      "price": "$175.25",
      "shares": 100,
      "pnl": ""
    },
    {
      "date": "2025-04-07",
      "type": "SELL",
      "price": "$189.65",
      "shares": 100,
      "pnl": "8.23%"
    }
  ],
  "plot_urls": {
    "price_chart": "/static/img/AAPL_price_chart.png",
    "indicators_chart": "/static/img/AAPL_indicators_chart.png"
  },
  "description": "AAPL is currently in a bearish trend with weak momentum and high volatility. The 20-day SMA is below the 50-day SMA, indicating a downtrend. The RSI is below 50, showing weak momentum. The MACD is below the signal line, confirming bearish momentum. Based on these indicators, the recommendation is to SELL with a target of $191.39 (support level) and a stop loss at $237.13 (resistance level)."
}
```

### 2. Market Scanner

#### Get Scanner Results

Retrieves the latest market scanner results showing trading opportunities across multiple assets.

**Endpoint:** `/get_scanner_results`

**Method:** GET

**Parameters:** None

**Example Request:**
```bash
curl -X GET http://localhost:5000/get_scanner_results
```

**Example Response:**
```json
{
  "success": true,
  "scanner_results": [
    {
      "ticker": "AAPL",
      "signal": "SELL",
      "strength": "Strong",
      "pattern": "Double Top",
      "volume": "+25%"
    },
    {
      "ticker": "MSFT",
      "signal": "HOLD",
      "strength": "Neutral",
      "pattern": "Consolidation",
      "volume": "+5%"
    },
    {
      "ticker": "GOOGL",
      "signal": "BUY",
      "strength": "Moderate",
      "pattern": "Breakout",
      "volume": "+15%"
    }
  ]
}
```

### 3. Portfolio Management

#### Get Portfolio

Retrieves the current portfolio information including positions and watchlist.

**Endpoint:** `/get_portfolio`

**Method:** GET

**Parameters:** None

**Example Request:**
```bash
curl -X GET http://localhost:5000/get_portfolio
```

**Example Response:**
```json
{
  "success": true,
  "portfolio": {
    "cash": 100000.00,
    "positions": [
      {
        "ticker": "AAPL",
        "shares": 100,
        "entry_price": 175.25,
        "current_price": 182.50,
        "value": 18250.00,
        "profit": 725.00,
        "profit_percent": 4.14
      },
      {
        "ticker": "MSFT",
        "shares": 50,
        "entry_price": 310.75,
        "current_price": 325.30,
        "value": 16265.00,
        "profit": 727.50,
        "profit_percent": 4.68
      }
    ],
    "watchlist": ["AMZN", "NVDA", "TSLA", "META", "AMD"],
    "total_value": 134515.00,
    "total_profit": 1452.50,
    "total_profit_percent": 1.09
  }
}
```

### 4. Trade Journal

#### Get Trade Journal

Retrieves the trade journal entries.

**Endpoint:** `/get_trade_journal`

**Method:** GET

**Parameters:** None

**Example Request:**
```bash
curl -X GET http://localhost:5000/get_trade_journal
```

**Example Response:**
```json
{
  "success": true,
  "trade_journal": [
    {
      "date": "2025-04-01",
      "ticker": "AAPL",
      "action": "BUY",
      "price": 170.25,
      "shares": 100,
      "reason": "Bullish breakout pattern with increasing volume",
      "result": "Open"
    },
    {
      "date": "2025-03-25",
      "ticker": "MSFT",
      "action": "BUY",
      "price": 310.75,
      "shares": 50,
      "reason": "Pullback to support with RSI oversold",
      "result": "Open"
    }
  ]
}
```

#### Add Journal Entry

Adds a new entry to the trade journal.

**Endpoint:** `/add_journal_entry`

**Method:** POST

**Parameters:**
- `date` (string, required): Date of the trade (YYYY-MM-DD)
- `ticker` (string, required): Ticker symbol
- `action` (string, required): "BUY" or "SELL"
- `price` (float, required): Trade price
- `shares` (integer, required): Number of shares
- `reason` (string, required): Reason for the trade
- `result` (string, optional): Result of the trade (e.g., "Open", "Closed (+5.2%)")

**Example Request:**
```bash
curl -X POST \
  -F "date=2025-04-08" \
  -F "ticker=NVDA" \
  -F "action=BUY" \
  -F "price=950.75" \
  -F "shares=10" \
  -F "reason=Breakout from consolidation with high volume" \
  -F "result=Open" \
  http://localhost:5000/add_journal_entry
```

**Example Response:**
```json
{
  "success": true,
  "entry": {
    "date": "2025-04-08",
    "ticker": "NVDA",
    "action": "BUY",
    "price": 950.75,
    "shares": 10,
    "reason": "Breakout from consolidation with high volume",
    "result": "Open"
  }
}
```

### 5. Backtesting

#### Run Backtest

Runs a backtest for a specified strategy and ticker.

**Endpoint:** `/run_backtest`

**Method:** POST

**Parameters:**
- `ticker` (string, required): Ticker symbol
- `strategy_type` (string, required): Strategy type ("swing", "day", "position")
- `start_date` (string, required): Start date for backtest (YYYY-MM-DD)
- `end_date` (string, required): End date for backtest (YYYY-MM-DD)
- `initial_capital` (float, optional): Initial capital. Default: 100000.0

**Example Request:**
```bash
curl -X POST \
  -F "ticker=AAPL" \
  -F "strategy_type=swing" \
  -F "start_date=2024-01-01" \
  -F "end_date=2025-01-01" \
  -F "initial_capital=100000" \
  http://localhost:5000/run_backtest
```

**Example Response:**
```json
{
  "success": true,
  "backtest_results": {
    "ticker": "AAPL",
    "strategy_type": "swing",
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
    "initial_capital": 100000.0,
    "final_capital": 112500.0,
    "total_return": "12.50%",
    "annualized_return": "12.50%",
    "sharpe_ratio": 0.85,
    "max_drawdown": "-8.75%",
    "win_rate": "65.00%",
    "profit_factor": 1.75,
    "trades": [
      {
        "date_entry": "2024-01-15",
        "date_exit": "2024-02-10",
        "type": "LONG",
        "entry_price": 175.25,
        "exit_price": 182.50,
        "shares": 100,
        "pnl": "4.14%",
        "pnl_dollar": "$725.00"
      }
    ]
  }
}
```

### 6. Memory Module Settings

#### Get Memory Settings

Retrieves the current memory module settings.

**Endpoint:** `/get_memory_settings`

**Method:** GET

**Parameters:** None

**Example Request:**
```bash
curl -X GET http://localhost:5000/get_memory_settings
```

**Example Response:**
```json
{
  "success": true,
  "memory_settings": {
    "trade_history_depth": 50,
    "market_pattern_recognition": 75,
    "sentiment_analysis_weight": 60,
    "technical_indicator_memory": 80,
    "adaptive_learning_rate": 65
  }
}
```

#### Update Memory Settings

Updates the memory module settings.

**Endpoint:** `/update_memory_settings`

**Method:** POST

**Parameters:**
- `trade_history_depth` (integer, optional): Depth of trade history to consider (0-100)
- `market_pattern_recognition` (integer, optional): Weight for market pattern recognition (0-100)
- `sentiment_analysis_weight` (integer, optional): Weight for sentiment analysis (0-100)
- `technical_indicator_memory` (integer, optional): Memory for technical indicators (0-100)
- `adaptive_learning_rate` (integer, optional): Rate of adaptive learning (0-100)

**Example Request:**
```bash
curl -X POST \
  -F "trade_history_depth=60" \
  -F "market_pattern_recognition=80" \
  -F "sentiment_analysis_weight=70" \
  -F "technical_indicator_memory=85" \
  -F "adaptive_learning_rate=75" \
  http://localhost:5000/update_memory_settings
```

**Example Response:**
```json
{
  "success": true,
  "message": "Memory settings updated successfully",
  "memory_settings": {
    "trade_history_depth": 60,
    "market_pattern_recognition": 80,
    "sentiment_analysis_weight": 70,
    "technical_indicator_memory": 85,
    "adaptive_learning_rate": 75
  }
}
```

## Error Handling

All API endpoints return a JSON response with a `success` field indicating whether the request was successful. If an error occurs, the response will include an `error` field with a description of the error.

**Example Error Response:**
```json
{
  "success": false,
  "error": "Error generating strategy: Invalid ticker symbol"
}
```

## Frontend Integration

The Gemma Advanced Trading System provides a comprehensive web interface that uses these API endpoints to deliver a user-friendly trading experience. The frontend includes:

1. **Dashboard** - Overview of portfolio, market conditions, and recent trades
2. **Strategy Generator** - Interface for generating trading strategies
3. **Market Scanner** - Tool for finding trading opportunities
4. **Backtesting** - Interface for testing strategies against historical data
5. **Trade Journal** - Log of trades and performance
6. **Settings** - Configuration options including memory module controls

## Memory Module Control

The memory module control is a key feature that allows users to adjust how the system learns from past data and adapts to changing market conditions. The control panel includes sliders for:

1. **Trade History Depth** - Controls how far back the system looks at past trades
2. **Market Pattern Recognition** - Adjusts the weight given to recognized market patterns
3. **Sentiment Analysis Weight** - Controls the influence of sentiment data on strategies
4. **Technical Indicator Memory** - Adjusts how long the system remembers indicator signals
5. **Adaptive Learning Rate** - Controls how quickly the system adapts to new market conditions

These settings can be adjusted through the Settings page in the frontend or directly via the API endpoints described above.

## Example: Generating an AAPL Strategy

Here's a complete example of how to generate a trading strategy for AAPL using the API:

```bash
curl -X POST -F "ticker=AAPL" -F "strategy_type=swing" http://localhost:5000/generate_strategy
```

This will return a comprehensive trading strategy for AAPL including:
- Current market trend analysis
- Support and resistance levels
- Entry and exit recommendations
- Risk management guidelines
- Performance metrics
- Historical trade analysis

The strategy can then be implemented manually or through an automated trading system.
