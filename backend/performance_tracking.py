class PerformanceTracker:
    """Minimal tracker for storing backtest results."""

    def __init__(self):
        self.results = {}

    def add_backtest_result(self, strategy_id, result):
        self.results[strategy_id] = result

    def get_tracked_strategies(self):
        return list(self.results.keys())

    def get_performance_metrics(self, strategy_id):
        return self.results[strategy_id]["performance_metrics"]

    def get_equity_curve(self, strategy_id):
        return self.results[strategy_id]["equity_curve"]

    def get_trades(self, strategy_id):
        return self.results[strategy_id]["trades"]

    def compare_strategies(self, metric="total_return"):
        comparison = []
        for sid, res in self.results.items():
            comparison.append((sid, res["performance_metrics"].get(metric, 0)))
        return sorted(comparison, key=lambda x: x[1], reverse=True)
