"""Train the three inventory-decision models on the consolidated store table.

Models (all exported to model/inventory_models.json so inference needs no sklearn):
  * profit_regressor       LinearRegression  -> predicts gross_profit (revenue earned)
  * reorder_classifier     LogisticRegression -> P(needs_reorder)  (replenishment)
  * discontinue_classifier LogisticRegression -> P(discontinue)    (low/no demand)

The reorder/discontinue labels are produced by the transparent business rules in
inventory_rules.py; the classifiers learn those policies from the KPI features so
they can score new or partially-known SKUs. Standardisation stats are saved so the
sigmoid can be reproduced from JSON alone.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score)

BASE_DIR = Path(__file__).resolve().parent.parent
STORE_PATH = BASE_DIR / "data" / "inventory_features_store.csv"
MODELS_PATH = BASE_DIR / "model" / "inventory_models.json"

PROFIT_FEATURES = ["units_sold", "unit_sell", "unit_cost", "Volume", "Classification"]
PROFIT_TARGET = "gross_profit"

REORDER_FEATURES = ["daily_demand", "end_onhand", "inventory_turnover",
                    "sell_through_pct", "days_of_supply"]
REORDER_TARGET = "needs_reorder"

DISCONTINUE_FEATURES = ["units_sold", "end_onhand", "days_of_supply",
                        "sell_through_pct", "inventory_turnover", "gross_margin_pct"]
DISCONTINUE_TARGET = "discontinue_candidate"

# Large but finite cap for "infinite" days-of-supply (no sales) so models are stable.
DOS_CAP = 9999.0


def _clean(df):
    df = df.copy()
    df["days_of_supply"] = df["days_of_supply"].replace([np.inf, -np.inf], np.nan).fillna(DOS_CAP)
    for c in ["inventory_turnover", "sell_through_pct", "gross_margin_pct", "daily_demand"]:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Fill remaining model inputs; price gaps -> 0 so rows are usable.
    for c in ["units_sold", "unit_sell", "unit_cost", "Volume", "Classification",
              "end_onhand", "gross_profit", "gross_margin_pct"]:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _fit_regressor(df):
    X = df[PROFIT_FEATURES].astype(float)
    y = df[PROFIT_TARGET].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    pred = m.predict(Xte)
    return {
        "type": "linear_regression",
        "features": PROFIT_FEATURES,
        "target": PROFIT_TARGET,
        "intercept": float(m.intercept_),
        "coefficients": {f: float(c) for f, c in zip(PROFIT_FEATURES, m.coef_)},
        "validation_mae": float(mean_absolute_error(yte, pred)),
        "validation_rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "validation_r2": float(r2_score(yte, pred)),
    }


def _fit_classifier(df, features, target):
    X = df[features].astype(float)
    y = df[target].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y if y.nunique() > 1 else None)
    scaler = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
    pred = m.predict(scaler.transform(Xte))
    return {
        "type": "logistic_regression",
        "features": features,
        "target": target,
        "intercept": float(m.intercept_[0]),
        "coefficients": {f: float(c) for f, c in zip(features, m.coef_[0])},
        "scaler_mean": {f: float(v) for f, v in zip(features, scaler.mean_)},
        "scaler_scale": {f: float(v) for f, v in zip(features, scaler.scale_)},
        "validation_accuracy": float(accuracy_score(yte, pred)),
        "validation_precision": float(precision_score(yte, pred, zero_division=0)),
        "validation_recall": float(recall_score(yte, pred, zero_division=0)),
        "positive_rate": float(y.mean()),
    }


def train_and_save_models():
    if not STORE_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {STORE_PATH}. "
                                "Run inventory_features.py first.")
    df = _clean(pd.read_csv(STORE_PATH, low_memory=False))

    models = {
        "profit_regressor": _fit_regressor(df),
        "reorder_classifier": _fit_classifier(df, REORDER_FEATURES, REORDER_TARGET),
        "discontinue_classifier": _fit_classifier(df, DISCONTINUE_FEATURES, DISCONTINUE_TARGET),
        "dos_cap": DOS_CAP,
        "n_rows": int(len(df)),
    }

    MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_PATH, "w") as f:
        json.dump(models, f, indent=2)

    print("Inventory models trained.")
    print(f"  profit_regressor       R2={models['profit_regressor']['validation_r2']:.3f} "
          f"MAE={models['profit_regressor']['validation_mae']:.2f}")
    print(f"  reorder_classifier     acc={models['reorder_classifier']['validation_accuracy']:.3f} "
          f"recall={models['reorder_classifier']['validation_recall']:.3f}")
    print(f"  discontinue_classifier acc={models['discontinue_classifier']['validation_accuracy']:.3f} "
          f"recall={models['discontinue_classifier']['validation_recall']:.3f}")
    print(f"  saved -> {MODELS_PATH}")
    return models


if __name__ == "__main__":
    train_and_save_models()
