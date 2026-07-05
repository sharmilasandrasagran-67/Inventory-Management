"""Prediction utilities.

Re-implements the linear model purely from the exported JSON parameters so that
inference needs no scikit-learn dependency and is fully reproducible.
"""

from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PARAMS_PATH = BASE_DIR / "model" / "model_params.json"


def load_model_params():
    if not MODEL_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Model parameter file not found: {MODEL_PARAMS_PATH}")

    with open(MODEL_PARAMS_PATH, "r") as f:
        model_params = json.load(f)

    return model_params


def predict_sales_quantity(input_data, model_params=None):
    """Predict SalesQuantity (units) for a single record.

    input_data: dict mapping each feature name to a numeric value.
    """
    if model_params is None:
        model_params = load_model_params()

    prediction = model_params["intercept"]

    for feature in model_params["features"]:
        prediction += float(input_data[feature]) * float(
            model_params["coefficients"][feature]
        )

    # Demand cannot be negative; floor the linear output at zero.
    return float(max(prediction, 0.0))
