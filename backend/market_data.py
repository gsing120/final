"""Simple market data access using :class:`FMPClient`."""

from .data_access import FMPClient


class MarketDataManager:
    """Fetch historical and real-time data for tests."""

    def __init__(self, api_key=None):
        self.client = FMPClient(api_key=api_key)

    def get_historical_data(self, symbols, interval="1d", period="1y"):
        """Return a dictionary of dataframes keyed by symbol."""
        data = {}
        for symbol in symbols:
            df = self.client.get_market_data(symbol, interval=interval, period=period)
            data[symbol] = df
        return data

    def get_current_prices(self, symbols):
        """Return the last close price for each symbol."""
        prices = {}
        for symbol in symbols:
            df = self.client.get_market_data(symbol, interval="1d", period="5d")
            if df is not None and not df.empty:
                prices[symbol] = float(df["close"].iloc[-1])
            else:
                prices[symbol] = 0.0
        return prices

    def get_universe(self):
        """Return a small default universe of major ETFs."""
        return ["SPY", "QQQ", "IWM", "TLT"]
