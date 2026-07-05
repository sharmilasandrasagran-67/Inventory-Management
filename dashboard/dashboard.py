import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict_utils import load_model_params, predict_sales_quantity
from log_utils import log_prediction, load_monitoring_logs

TRAINING_DATA_PATH = BASE_DIR / "data" / "inventory_demand_training.csv"
PRODUCTION_DATA_PATH = BASE_DIR / "data" / "inventory_demand_production.csv"

st.set_page_config(
    page_title="Inventory Demand Monitoring Dashboard",
    layout="wide",
)

st.title("Inventory Demand Prediction and Monitoring Dashboard")

st.write(
    "This dashboard demonstrates how a deployed demand-forecasting model can be "
    "monitored using actual outcomes, latency, manual business feedback, and input "
    "drift indicators. The model predicts SalesQuantity (units sold) for a wine & "
    "spirits retailer."
)


@st.cache_data
def load_training_data():
    return pd.read_csv(TRAINING_DATA_PATH)


@st.cache_data
def load_production_data():
    return pd.read_csv(PRODUCTION_DATA_PATH)


@st.cache_data
def load_params():
    return load_model_params()


training_data = load_training_data()
production_data = load_production_data()
model_params = load_params()

tab1, tab2, tab3 = st.tabs([
    "Prediction and Manual Feedback",
    "Monitoring Dashboard",
    "Agile Retrospective",
])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.header("Prediction and Manual Feedback")

    st.write(
        "Simulate a business user generating a demand prediction and manually "
        "providing feedback after comparing the prediction with actual units sold."
    )

    st.sidebar.header("Product / Sale Profile")

    sales_price = st.sidebar.number_input(
        "Sales Price (per unit)", min_value=0.0, value=12.99, step=0.5
    )
    volume = st.sidebar.number_input(
        "Volume (mL)", min_value=0.0, value=750.0, step=50.0
    )
    classification = st.sidebar.selectbox("Classification", [1, 2], index=0)
    store = st.sidebar.slider("Store", 1, 80, 1)
    day_of_week = st.sidebar.slider("Day of week (0=Mon)", 0, 6, 2)
    day_of_month = st.sidebar.slider("Day of month", 1, 31, 15)

    input_data = {
        "SalesPrice": sales_price,
        "Volume": volume,
        "Classification": classification,
        "Store": store,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
    }

    start = time.time()
    prediction = predict_sales_quantity(input_data, model_params)
    latency_ms = (time.time() - start) * 1000.0

    st.metric("Predicted SalesQuantity (units)", f"{prediction:.2f}")
    st.caption(f"Prediction latency: {latency_ms:.2f} ms")

    st.subheader("Provide Manual Feedback")

    actual_quantity = st.number_input(
        "Actual units sold (observed)", min_value=0.0, value=1.0, step=1.0
    )
    feedback_score = st.slider("Business feedback score (1-5)", 1, 5, 3)
    feedback_text = st.text_input("Feedback notes", "")

    if st.button("Log this prediction"):
        row = log_prediction(
            sales_price=sales_price,
            volume=volume,
            classification=classification,
            store=store,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            prediction=prediction,
            actual_quantity=actual_quantity,
            latency_ms=latency_ms,
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )
        st.success(
            f"Logged. Absolute error = {row['absolute_error']:.2f} units, "
            f"feedback score = {row['feedback_score']}."
        )

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.header("Monitoring Dashboard")

    logs = load_monitoring_logs()

    if logs.empty:
        st.info("No monitoring logs yet. Log a prediction in the first tab to begin.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Predictions logged", len(logs))
        col2.metric("Mean absolute error", f"{logs['absolute_error'].mean():.2f}")
        col3.metric("Mean latency (ms)", f"{logs['latency_ms'].mean():.2f}")

        st.subheader("Absolute error over time")
        st.line_chart(logs.set_index("timestamp")["absolute_error"])

        st.subheader("Average business feedback score")
        st.metric("Feedback score (avg)", f"{logs['feedback_score'].mean():.2f}")

        st.subheader("Input drift check (production vs training)")
        drift_rows = []
        for feature in model_params["features"]:
            if feature in training_data.columns and feature in production_data.columns:
                train_mean = float(training_data[feature].mean())
                prod_mean = float(production_data[feature].mean())
                pct = ((prod_mean - train_mean) / train_mean * 100.0) if train_mean else np.nan
                drift_rows.append({
                    "feature": feature,
                    "training_mean": round(train_mean, 3),
                    "production_mean": round(prod_mean, 3),
                    "drift_pct": round(pct, 2),
                })
        st.dataframe(pd.DataFrame(drift_rows))

        st.subheader("Recent logs")
        st.dataframe(logs.tail(20))

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.header("Agile Retrospective")

    st.write(
        "A short retrospective view supporting the agile workflow: what the team "
        "shipped this sprint, what monitoring reveals, and what to improve next."
    )

    st.markdown(
        "- **What went well:** an end-to-end MVP (train -> JSON params -> predict "
        "-> dashboard -> CI/CD) is deployed and monitored.\n"
        "- **What to watch:** baseline LinearRegression has high error on raw "
        "transaction-level demand; monitor MAE and input drift.\n"
        "- **Next iteration:** aggregate demand to product-store-week, add "
        "lag/seasonality features, and trial tree-based or time-series models."
    )
