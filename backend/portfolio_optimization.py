"""Basic mean-variance portfolio optimizer for unit tests."""

import numpy as np
import pandas as pd


class PortfolioOptimizer:
    """Compute optimal weights using a mean-variance approach."""

    def optimize(self, price_data: dict, risk_free_rate=0.0):
        """Return weights that maximize the Sharpe ratio."""
        if not price_data:
            return {}

        # Build returns DataFrame
        returns = []
        symbols = []
        for symbol, df in price_data.items():
            if "close" in df.columns:
                returns.append(df["close"].pct_change().dropna())
                symbols.append(symbol)

        if not returns:
            return {}

        returns_df = pd.concat(returns, axis=1)
        returns_df.columns = symbols

        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        num_assets = len(symbols)
        weights = np.ones(num_assets) / num_assets

        # Simple gradient ascent on Sharpe ratio
        lr = 0.01
        for _ in range(200):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            grad = (mean_returns * portfolio_vol - portfolio_return * np.dot(cov_matrix, weights) / portfolio_vol)
            weights += lr * grad
            weights = np.clip(weights, 0, 1)
            weights /= weights.sum()

        return {symbol: float(w) for symbol, w in zip(symbols, weights)}
