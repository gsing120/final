from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
import time


@dataclass
class ScanCriteria:
    name: str
    description: str
    indicator_conditions: List[Dict[str, Any]]
    price_conditions: List[Dict[str, Any]]
    fundamental_conditions: List[Dict[str, Any]]
    time_conditions: List[Dict[str, Any]]

    def validate(self) -> bool:
        for cond in self.indicator_conditions:
            if cond.get("indicator") is None:
                return False
        return True


@dataclass
class ScanTemplate:
    name: str
    description: str
    criteria: List[ScanCriteria]
    universe_filter: Dict[str, Any]
    sort_by: Dict[str, Any]
    max_results: int

    def add_criteria(self, criteria: ScanCriteria):
        self.criteria.append(criteria)

    def remove_criteria(self, name: str):
        self.criteria = [c for c in self.criteria if c.name != name]


@dataclass
class ScanResult:
    criteria_name: str
    matching_symbols: List[str]
    execution_time: float
    total_symbols_scanned: int


class MarketScanner:
    def __init__(self, market_data_client=None):
        self.market_data_client = market_data_client
        self.templates: List[ScanTemplate] = []

    def add_template(self, template: ScanTemplate):
        self.templates.append(template)

    def execute_scan(self, criteria: ScanCriteria) -> ScanResult:
        start = time.time()
        symbols = self.market_data_client.get_universe()
        data = self.market_data_client.get_historical_data_bulk(symbols)
        matching = list(data.keys())  # simplified: all symbols match
        return ScanResult(
            criteria_name=criteria.name,
            matching_symbols=matching,
            execution_time=time.time() - start,
            total_symbols_scanned=len(symbols),
        )

    def execute_template_scan(self, template_name: str):
        template = next(t for t in self.templates if t.name == template_name)
        results = [self.execute_scan(c) for c in template.criteria]
        return results

    def get_combined_template_results(self, template_name: str, results: List[ScanResult]):
        all_symbols = [set(r.matching_symbols) for r in results]
        if not all_symbols:
            return {"matching_symbols": [], "all_criteria_matched": [], "any_criteria_matched": []}
        all_matched = set.intersection(*all_symbols)
        any_matched = set.union(*all_symbols)
        return {
            "matching_symbols": list(any_matched),
            "all_criteria_matched": list(all_matched),
            "any_criteria_matched": list(any_matched),
        }
