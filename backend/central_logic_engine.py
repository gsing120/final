"""Co-ordinates major components for integration tests."""

from datetime import datetime


class CentralLogicEngine:
    """Minimal orchestration layer connecting system components."""

    def __init__(self, **components):
        self.components = components

    def run_cycle(self, symbols):
        """Execute a basic data/strategy/evaluation cycle."""
        mdm = self.components.get("market_data_manager")
        se = self.components.get("strategy_engine")
        tracker = self.components.get("performance_tracker")
        backtester = self.components.get("backtester")

        if not all([mdm, se, tracker, backtester]):
            return None

        data = mdm.get_historical_data(symbols)
        strategies = list(se.strategies.keys())
        results = {}
        for strat_name in strategies:
            result = backtester.run_backtest(strategy_name=strat_name, data=data)
            tracker.add_backtest_result(strat_name, result)
            results[strat_name] = result

        return {
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }
