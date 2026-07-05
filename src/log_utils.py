"""Logging utilities for monitoring deployed demand predictions.

Each call records the inputs, the model prediction, the actual observed demand,
the resulting error, latency, and manual business feedback. These logs power the
monitoring dashboard and the agile retrospective view.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "data" / "monitoring_logs.csv"


def log_prediction(
    sales_price,
    volume,
    classification,
    store,
    day_of_week,
    day_of_month,
    prediction,
    actual_quantity,
    latency_ms,
    feedback_score,
    feedback_text,
):
    error = float(actual_quantity) - float(prediction)
    absolute_error = abs(error)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "SalesPrice": sales_price,
        "Volume": volume,
        "Classification": classification,
        "Store": store,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "prediction": float(prediction),
        "actual_quantity": float(actual_quantity),
        "error": error,
        "absolute_error": absolute_error,
        "latency_ms": float(latency_ms),
        "feedback_score": int(feedback_score),
        "feedback_text": feedback_text or "",
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])

    if LOG_PATH.exists():
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_PATH, index=False)

    return row


def load_monitoring_logs():
    if not LOG_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
