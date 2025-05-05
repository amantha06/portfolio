import os
import joblib
import numpy as np
import pandas as pd

import cvxpy as cp
from sklearn.covariance import LedoitWolf

from typing import List
from features import load_data, add_features

# List of engineered features—must match your features.py output
FEATURE_COLS = [
    "r1", "r5", "r21", "vol21",
    "rsi", "macd", "macd_signal",
    "bb_upper", "bb_lower",
    "tr", "atr"
]

def load_returns(symbols: List[str]) -> pd.DataFrame:
    """
    Load raw OHLCV CSVs via load_data, compute daily percentage-change returns,
    and align into a single DataFrame.
    """
    rets = {}
    for sym in symbols:
        df = load_data(f"data/{sym}.csv")
        rets[sym] = df["Close"].pct_change().dropna()
    return pd.DataFrame(rets).dropna()


def predict_expected(symbols: List[str], model_dir: str = "models") -> pd.Series:
    """
    For each symbol:
      - compute features on the full price history, take the latest row
      - load corresponding KNN and Kernel Ridge models
      - predict next-day return for each and average them
    Returns a Series of expected returns indexed by symbol.
    """
    mu = {}
    for sym in symbols:
        # Compute features and select the latest feature vector
        df_raw = load_data(f"data/{sym}.csv")
        df_feat = add_features(df_raw)
        X_latest = df_feat.iloc[[-1]][FEATURE_COLS].values  # shape (1, n_features)

        # Load trained models
        knn = joblib.load(os.path.join(model_dir, f"{sym}_knn.pkl"))
        kr = joblib.load(os.path.join(model_dir, f"{sym}_kr.pkl"))

        # Predict and average
        p_knn = knn.predict(X_latest)[0]
        p_kr = kr.predict(X_latest)[0]
        mu[sym] = np.mean([p_knn, p_kr])

    return pd.Series(mu)


def optimize_portfolio(mu: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """
    Solve the Markowitz optimization:
      maximize  mu^T w - 0.5 * w^T Sigma w
      subject to sum(w) == 1 and w >= 0
    Returns optimal weights as a Series indexed by symbol.
    """
    # Estimate covariance with shrinkage
    cov = LedoitWolf().fit(returns).covariance_
    n = mu.shape[0]
    w = cp.Variable(n)

    # Define objective and constraints
    objective = mu.values @ w - 0.5 * cp.quad_form(w, cov)
    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.SCS)

    return pd.Series(w.value, index=mu.index)


if __name__ == "__main__":
    symbols = ["SPY", "QQQ", "IWM"]
    #symbols = ["SPY"] 
    # 1) Load and compute historical returns
    returns = load_returns(symbols)

    # 2) Predict expected returns via ML models
    mu = predict_expected(symbols)

    # 3) Compute optimal weights
    w_opt = optimize_portfolio(mu, returns)

    # 4) Display
    print("Optimal portfolio weights:")
    print(w_opt.apply(lambda x: f"{x:.2%}"))
