from flask import Flask, render_template
from backend.data_access import FMPClient

app = Flask(__name__)
fmp = FMPClient()

sample_portfolio = {
    "value": 125000.0,
    "performance": 12.5,
    "cash": 25000.0,
    "positions": [
        {"ticker": "AAPL", "shares": 10, "entry_price": 150.0, "current_price": 170.0},
        {"ticker": "MSFT", "shares": 5, "entry_price": 300.0, "current_price": 320.0}
    ]
}

@app.route('/')
def dashboard():
    for p in sample_portfolio["positions"]:
        data = fmp.get_market_data(p["ticker"], period="5d")
        if data is not None:
            p["current_price"] = float(data["close"].iloc[-1])
        p["value"] = p["shares"] * p["current_price"]
        p["profit"] = p["shares"] * (p["current_price"] - p["entry_price"])
        p["profit_percent"] = ((p["current_price"] - p["entry_price"]) / p["entry_price"]) * 100
    return render_template('simple_dashboard.html', portfolio=sample_portfolio)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
