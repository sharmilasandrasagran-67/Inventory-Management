import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import normalize_sales_date
from inventory_rules import add_kpis, add_rule_flags, RULE_CONFIG
from inventory_predict_utils import load_inventory_models, recommend


def test_normalize_sales_date_mixed_formats():
    s = pd.Series(["1/13/2016", "2016-05-13", "2016-12-31 00:00:00", "42503", "12/1/2016"])
    out = normalize_sales_date(s)
    assert out.notna().all()
    # 1/13/2016 -> January, 42503 -> 2016-05-13
    assert out.iloc[0].month == 1
    assert str(out.iloc[3].date()) == "2016-05-13"
    # full-year coverage is preserved (not collapsed to Jan/Feb)
    assert set(out.dt.month) == {1, 5, 12}


def test_normalize_sales_date_bad_value_is_nat():
    out = normalize_sales_date(pd.Series(["not-a-date"]))
    assert out.isna().all()


def _sample_frame():
    return pd.DataFrame({
        "beg_onhand": [10, 5, 0],
        "end_onhand": [8, 0, 20],
        "units_sold": [40, 12, 0],
        "units_purchased": [38, 7, 0],
        "sales_revenue": [800.0, 240.0, 0.0],
        "unit_cost": [14.0, 12.0, 9.0],
        "unit_sell": [20.0, 20.0, 15.0],
        "sales_span_days": [60, 60, 60],
    })


def test_kpis_and_profit():
    df = add_kpis(_sample_frame())
    # gross_profit = revenue - units_sold*unit_cost
    assert round(df["gross_profit"].iloc[0], 2) == round(800.0 - 40 * 14.0, 2)
    assert df["daily_demand"].iloc[0] > 0


def test_rule_flags_dead_stock_and_reorder():
    df = add_rule_flags(add_kpis(_sample_frame()), RULE_CONFIG)
    # Row 2 (index 2): zero sales but 20 on hand -> dead stock / discontinue
    assert df["dead_stock"].iloc[2] == 1
    assert df["discontinue_candidate"].iloc[2] == 1
    # Flags are 0/1 integers
    for col in ["needs_reorder", "discontinue_candidate", "profitable"]:
        assert set(df[col].unique()).issubset({0, 1})


def test_recommend_returns_all_decisions():
    models = load_inventory_models()
    row = {"units_sold": 50, "unit_sell": 20.0, "unit_cost": 14.0, "Volume": 750,
           "Classification": 1, "end_onhand": 3, "daily_demand": 0.8,
           "inventory_turnover": 4.0, "sell_through_pct": 60.0, "days_of_supply": 4.0,
           "gross_margin_pct": 30.0}
    rec = recommend(row, models)
    for key in ["predicted_gross_profit", "reorder_probability", "needs_reorder",
                "discontinue_probability", "discontinue_candidate", "profitable"]:
        assert key in rec
    assert 0.0 <= rec["reorder_probability"] <= 1.0
    assert 0.0 <= rec["discontinue_probability"] <= 1.0
