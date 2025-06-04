"""Simplified paper trading engine for unit tests."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    action: str
    symbol: str
    quantity: int
    price: float
    timestamp: datetime
    strategy_id: str | None = None


class PaperTradingEngine:
    """Execute trades in-memory without touching real markets."""

    def __init__(self, market_data_manager=None, strategy_engine=None, risk_manager=None, initial_capital=100000.0):
        self.market_data_manager = market_data_manager
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.cash = float(initial_capital)
        self.positions = {}
        self.trades: list[Trade] = []

    def initialize(self):
        """Reset portfolio to initial state."""
        self.cash = float(self.cash)
        self.positions = {}
        self.trades.clear()

    def execute_trade(self, action, symbol, quantity, price=None, strategy_id=None):
        """Record a trade and update cash/positions."""
        if price is None:
            price = self.market_data_manager.get_current_prices([symbol]).get(symbol, 0)

        if action == "buy":
            cost = quantity * price
            if cost > self.cash:
                quantity = int(self.cash // price)
                cost = quantity * price
            if quantity <= 0:
                return {"success": False, "reason": "insufficient_cash"}
            self.cash -= cost
            pos = self.positions.setdefault(symbol, {"quantity": 0, "avg_price": 0})
            pos_qty = pos["quantity"] + quantity
            pos["avg_price"] = ((pos["avg_price"] * pos_qty) + cost) / pos_qty
            pos["quantity"] = pos_qty
        elif action == "sell":
            if symbol not in self.positions or self.positions[symbol]["quantity"] < quantity:
                return {"success": False, "reason": "no_position"}
            proceeds = quantity * price
            self.cash += proceeds
            self.positions[symbol]["quantity"] -= quantity
            if self.positions[symbol]["quantity"] == 0:
                del self.positions[symbol]
        else:
            return {"success": False, "reason": "invalid_action"}

        trade = Trade(action, symbol, quantity, price, datetime.now(), strategy_id)
        self.trades.append(trade)
        return {"success": True, "trade_id": len(self.trades)}

    def get_portfolio_status(self):
        """Return cash balance and open positions."""
        prices = self.market_data_manager.get_current_prices(self.positions.keys()) if self.market_data_manager else {}
        value = self.cash
        pos_list = []
        for sym, info in self.positions.items():
            current_price = prices.get(sym, info["avg_price"])
            market_val = current_price * info["quantity"]
            value += market_val
            pos_list.append({
                "symbol": sym,
                "quantity": info["quantity"],
                "entry_price": info["avg_price"],
                "current_price": current_price,
                "market_value": market_val,
            })
        return {"cash": self.cash, "positions": pos_list, "total_value": value}

    def get_trade_history(self):
        """Return a list of executed trades."""
        return [trade.__dict__ for trade in self.trades]
