"""Inference for the inventory-decision models, using only the exported JSON.

No scikit-learn required at inference time. Logistic models reproduce the sigmoid
after applying the saved standardisation (mean / scale).
"""

from pathlib import Path
import json
import math

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_PATH = BASE_DIR / "model" / "inventory_models.json"


def load_inventory_models():
    if not MODELS_PATH.exists():
        raise FileNotFoundError(f"Models file not found: {MODELS_PATH}. "
                                "Run train_inventory_models.py first.")
    with open(MODELS_PATH, "r") as f:
        return json.load(f)


def _sigmoid(z):
    if z < 0:
        return math.exp(z) / (1.0 + math.exp(z))
    return 1.0 / (1.0 + math.exp(-z))


def predict_profit(input_data, models=None):
    """Predict gross_profit (revenue earned minus cost) for one product/SKU."""
    models = models or load_inventory_models()
    m = models["profit_regressor"]
    pred = m["intercept"]
    for f in m["features"]:
        pred += float(input_data[f]) * m["coefficients"][f]
    return float(pred)


def _logistic_proba(model, input_data):
    z = model["intercept"]
    for f in model["features"]:
        x = float(input_data[f])
        x = (x - model["scaler_mean"][f]) / model["scaler_scale"][f]
        z += x * model["coefficients"][f]
    return _sigmoid(z)


def predict_reorder_proba(input_data, models=None):
    """Probability that a SKU needs replenishment."""
    models = models or load_inventory_models()
    return _logistic_proba(models["reorder_classifier"], input_data)


def predict_discontinue_proba(input_data, models=None):
    """Probability that a SKU should be discontinued (no / very weak demand)."""
    models = models or load_inventory_models()
    return _logistic_proba(models["discontinue_classifier"], input_data)


def recommend(input_data, models=None, threshold=0.5):
    """Return all three decisions for one SKU as a dict."""
    models = models or load_inventory_models()
    reorder_p = predict_reorder_proba(input_data, models)
    discont_p = predict_discontinue_proba(input_data, models)
    profit = predict_profit(input_data, models)
    return {
        "predicted_gross_profit": profit,
        "reorder_probability": reorder_p,
        "needs_reorder": int(reorder_p >= threshold),
        "discontinue_probability": discont_p,
        "discontinue_candidate": int(discont_p >= threshold),
        "profitable": int(profit > 0),
    }
