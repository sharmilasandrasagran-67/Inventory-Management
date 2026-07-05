import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validate_data import validate


def test_validate_passes_clean_frame():
    df = pd.DataFrame({
        "InventoryId": ["1_A_58"], "Store": [1], "Brand": [58], "SalesQuantity": [2],
        "SalesDollars": [25.0], "SalesPrice": [12.5], "SalesDate": ["1/1/2016"], "Volume": [750],
    })
    passed, results = validate(df)
    assert passed
    assert all(ok for _, ok, _ in results)


def test_validate_flags_negative_and_missing():
    df = pd.DataFrame({
        "InventoryId": ["1_A_58", "1_A_58"], "Store": [1, 1], "Brand": [58, 58],
        "SalesQuantity": [-1, None], "SalesDollars": [25.0, 25.0], "SalesPrice": [12.5, 12.5],
        "SalesDate": ["1/1/2016", "1/1/2016"], "Volume": [750, 750],
    })
    passed, results = validate(df)
    assert not passed
