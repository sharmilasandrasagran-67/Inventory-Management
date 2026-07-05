"""Train a baseline demand-forecasting model for the wine & spirits retailer.

The model predicts SalesQuantity (units sold) per sales record using a small set
of non-leaky numeric features. It mirrors the simple, transparent build of the
reference project: a LinearRegression whose parameters are exported to JSON so
that prediction can be reproduced anywhere without re-loading scikit-learn.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_DATA_PATH = BASE_DIR / "data" / "inventory_demand_training.csv"
MODEL_PARAMS_PATH = BASE_DIR / "model" / "model_params.json"

# Features deliberately exclude SalesDollars and ExciseTax because both scale
# directly with SalesQuantity and would leak the target into the model.
FEATURES = [
    "SalesPrice",
    "Volume",
    "Classification",
    "Store",
    "day_of_week",
    "day_of_month",
]

TARGET = "SalesQuantity"


def train_and_save_model():
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")

    df = pd.read_csv(TRAINING_DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df.dropna(subset=required_columns)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    model_params = {
        "target": TARGET,
        "features": FEATURES,
        "intercept": float(model.intercept_),
        "coefficients": {
            feature: float(coef)
            for feature, coef in zip(FEATURES, model.coef_)
        },
        "validation_mae": float(mae),
        "validation_rmse": float(rmse),
        "n_training_rows": int(len(X_train)),
    }

    MODEL_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PARAMS_PATH, "w") as f:
        json.dump(model_params, f, indent=2)

    print("Model trained successfully.")
    print(f"Validation MAE : {mae:.4f} units")
    print(f"Validation RMSE: {rmse:.4f} units")
    print(f"Saved model parameters to: {MODEL_PARAMS_PATH}")

    return model_params


if __name__ == "__main__":
    train_and_save_model()
