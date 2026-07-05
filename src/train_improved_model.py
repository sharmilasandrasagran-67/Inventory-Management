"""Improved demand model (Q2c): RandomForest + engineered features.

Baseline (train_model.py) is LinearRegression on raw features. This adds three
engineered features and a non-linear model, then reports the improvement.
Saves metrics + feature importances to model/improved_model.json.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN = BASE_DIR / "data" / "inventory_demand_training.csv"
OUT = BASE_DIR / "model" / "improved_model.json"
BASE_FEATURES = ["SalesPrice", "Volume", "Classification", "Store", "day_of_week", "day_of_month"]


def engineer(df):
    df = df.copy()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)          # feature creation
    df["price_per_ml"] = (df["SalesPrice"] / df["Volume"].replace(0, np.nan))
    df["price_per_ml"] = df["price_per_ml"].fillna(df["price_per_ml"].median())
    freq = df["Store"].value_counts(normalize=True)                 # frequency encoding
    df["store_freq"] = df["Store"].map(freq)
    return df, BASE_FEATURES + ["is_weekend", "price_per_ml", "store_freq"]


def main():
    df, feats = engineer(pd.read_csv(TRAIN))
    X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["SalesQuantity"],
                                              test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, max_depth=16,
                                  min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "model": "RandomForestRegressor",
        "features": feats,
        "validation_mae": float(mean_absolute_error(y_te, pred)),
        "validation_rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "validation_r2": float(r2_score(y_te, pred)),
        "feature_importance": {f: float(i) for f, i in
                               sorted(zip(feats, model.feature_importances_),
                                      key=lambda t: -t[1])},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(OUT, "w"), indent=2)
    print(f"Improved model: MAE={metrics['validation_mae']:.4f} "
          f"RMSE={metrics['validation_rmse']:.4f} R2={metrics['validation_r2']:.4f}")
    print("Saved ->", OUT)
    return metrics


if __name__ == "__main__":
    main()
