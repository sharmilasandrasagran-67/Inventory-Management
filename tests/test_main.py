import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_utils import load_model_params, predict_sales_quantity
from log_utils import log_prediction


def test_model_params_exist():
    params = load_model_params()

    assert "intercept" in params
    assert "coefficients" in params
    assert "features" in params


def test_prediction_output_is_numeric():
    params = load_model_params()

    input_data = {
        "SalesPrice": 12.99,
        "Volume": 750,
        "Classification": 1,
        "Store": 1,
        "day_of_week": 2,
        "day_of_month": 15,
    }

    prediction = predict_sales_quantity(input_data, params)

    assert isinstance(prediction, float)


def test_prediction_is_non_negative():
    params = load_model_params()

    input_data = {
        "SalesPrice": 399.99,
        "Volume": 18000,
        "Classification": 2,
        "Store": 80,
        "day_of_week": 6,
        "day_of_month": 31,
    }

    prediction = predict_sales_quantity(input_data, params)

    assert prediction >= 0.0


def test_log_prediction_creates_correct_error():
    row = log_prediction(
        sales_price=12.99,
        volume=750,
        classification=1,
        store=1,
        day_of_week=2,
        day_of_month=15,
        prediction=5.0,
        actual_quantity=3.0,
        latency_ms=12.5,
        feedback_score=4,
        feedback_text="Test log",
    )

    assert row["absolute_error"] == 2.0
    assert row["feedback_score"] == 4
