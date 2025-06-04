class MarketDataManager:
    """Placeholder market data manager."""

    def get_historical_data(self, *args, **kwargs):
        return {}

    def get_current_prices(self, symbols):
        return {s: 0 for s in symbols}

    def get_universe(self):
        return []
