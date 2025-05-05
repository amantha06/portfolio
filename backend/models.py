import os
import joblib
import numpy as np
from typing import List

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.kernel_ridge    import KernelRidge

from features import load_data, add_features

# Symbols to train
SYMBOLS    = ["SPY", "QQQ", "IWM"]
MODEL_DIR  = "models"

# Feature columns (must match those in add_features)
FEATURE_COLS = [
    "r1", "r5", "r21", "vol21",
    "rsi", "macd", "macd_signal",
    "bb_upper", "bb_lower", "tr", "atr"
]

# Hyperparameter grids
KNN_PARAMS = {
    "n_neighbors": [5, 10, 20],
    "weights":     ["uniform", "distance"]
}
KR_PARAMS  = {
    "alpha":  [1, 10, 100],
    "kernel": ["rbf", "poly"],
    "gamma":  [0.1, 1]
}


def train_models(symbols: List[str] = SYMBOLS, model_dir: str = MODEL_DIR):
    """
    Train KNN and Kernel Ridge for each symbol in `symbols`.
    Saves models as {symbol}_knn.pkl and {symbol}_kr.pkl in `model_dir`.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Time-series CV splitter
    tscv = TimeSeriesSplit(n_splits=5)

    for sym in symbols:
        # 1) Load raw data & compute features
        df_raw  = load_data(f"data/{sym}.csv")
        df_feat = add_features(df_raw)

        # 2) Build next-day return target
        df_feat["target"] = df_feat["r1"].shift(-1)
        df_feat.dropna(inplace=True)

        X = df_feat[FEATURE_COLS].values
        y = df_feat["target"].values

        # 3) Train KNN
        knn_cv = GridSearchCV(
            KNeighborsRegressor(), KNN_PARAMS,
            cv=tscv, n_jobs=-1, error_score="raise"
        )
        knn_cv.fit(X, y)
        best_knn = knn_cv.best_estimator_
        joblib.dump(best_knn, os.path.join(model_dir, f"{sym}_knn.pkl"))

        # 4) Train Kernel Ridge
        kr_cv = GridSearchCV(
            KernelRidge(), KR_PARAMS,
            cv=tscv, n_jobs=-1, error_score="raise"
        )
        kr_cv.fit(X, y)
        best_kr = kr_cv.best_estimator_
        joblib.dump(best_kr, os.path.join(model_dir, f"{sym}_kr.pkl"))

        print(f"Trained {sym}: KNN {knn_cv.best_params_}, KR {kr_cv.best_params_}")


if __name__ == "__main__":
    train_models()
