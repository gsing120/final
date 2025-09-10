import itertools
import pandas as pd
from typing import Dict, List, Callable, Any


class Strategy:
    """Simple strategy container used for tests."""

    def __init__(self, name: str, description: str = "", parameters: Dict[str, Any] | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.indicators: List[Dict[str, Any]] = []
        self.entry_conditions: List[Dict[str, Any]] = []
        self.exit_conditions: List[Dict[str, Any]] = []

    def add_indicator(self, name: str, function: Callable, params: Dict[str, Any] | None = None, input_data: Any = "close"):
        self.indicators.append({"name": name, "function": function, "params": params or {}, "input": input_data})

    def add_entry_condition(self, condition_type: str, left_operand: Any, right_operand: Any):
        self.entry_conditions.append({"type": condition_type, "left": left_operand, "right": right_operand})

    def add_exit_condition(self, condition_type: str, left_operand: Any, right_operand: Any):
        self.exit_conditions.append({"type": condition_type, "left": left_operand, "right": right_operand})


class StrategyEngine:
    """Very small strategy engine used for unit tests."""

    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}

    def register_strategy(self, strategy: Strategy):
        self.strategies[strategy.name] = strategy

    def get_calculated_indicators(self, strategy_name: str, data: pd.DataFrame) -> Dict[str, pd.Series]:
        strategy = self.strategies[strategy_name]
        results = {}
        for spec in strategy.indicators:
            params = {k: (strategy.parameters.get(v, v) if isinstance(v, str) else v) for k, v in spec["params"].items()}
            inputs = spec["input"]
            if isinstance(inputs, list):
                arg_data = [data[i] for i in inputs]
            else:
                arg_data = [data[inputs]]
            results[spec["name"]] = spec["function"](*arg_data, **params)
        return results

    def backtest(self, strategy_name: str, data: pd.DataFrame, **kwargs):
        # Return a minimal result structure
        return {"trades": [], "performance_metrics": {}, "equity_curve": pd.Series(dtype=float)}

    def optimize_parameters(self, strategy_name: str, data: pd.DataFrame, param_grid: Dict[str, List[Any]], **kwargs):
        combinations = list(itertools.product(*param_grid.values()))
        all_results = []
        keys = list(param_grid.keys())
        for combo in combinations:
            params = dict(zip(keys, combo))
            all_results.append({"parameters": params, "performance": {"total_return": 0.0}})
        return {"best_parameters": all_results[0]["parameters"], "performance": all_results[0]["performance"], "all_results": all_results}
