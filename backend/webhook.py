import os
import json
import requests
from dotenv import load_dotenv
from optimizer import load_returns, predict_expected, optimize_portfolio

# Load environment variables from .env
load_dotenv()

# Your webhook URL (fallback to user-provided URL if env var is missing)
WEBHOOK_URL = os.getenv(
    "TV_WEBHOOK_URL",
    "https://webhook.site/04dc12f1-08e6-4728-bc7d-daa068bb6028"
)


def run_webhook(symbols=None):
    """
    Executes the full pipeline:
      1) Load historical returns
      2) Predict expected returns via ML models
      3) Optimize portfolio weights
      4) Send JSON payload via HTTP POST to the webhook URL
    """
    symbols = symbols or ["SPY"]

    # 1) Historical returns
    returns_df = load_returns(symbols)

    # 2) Expected returns
    mu = predict_expected(symbols)

    # 3) Portfolio optimization
    weights = optimize_portfolio(mu, returns_df)

    # 4) Prepare and send payload
    payload = {"message": "Rebalance Signal"}
    for sym in symbols:
        payload[f"{sym}_weight"] = float(weights[sym])

    response = requests.post(WEBHOOK_URL, json=payload)
    print(f"Using webhook URL: {WEBHOOK_URL}")
    print(f"Webhook sent: {response.status_code}")


if __name__ == "__main__":
    run_webhook()
