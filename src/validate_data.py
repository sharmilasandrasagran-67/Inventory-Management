"""Automated data validation (Agile automation, Q3a).

Runs a set of checks against a dataset and exits non-zero if any critical check
fails, so it can gate a CI/CD pipeline. Checks: required columns, missing values,
duplicate rows, data types, and value ranges.

Usage:
    python src/validate_data.py            # validates the demand training CSV
    python src/validate_data.py <csv_path>
"""

from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = BASE_DIR / "data" / "inventory_demand_training.csv"

REQUIRED_COLUMNS = ["InventoryId", "Store", "Brand", "SalesQuantity",
                    "SalesDollars", "SalesPrice", "SalesDate"]
NON_NEGATIVE = ["SalesQuantity", "SalesDollars", "SalesPrice", "Volume"]


def validate(df):
    """Return (passed: bool, results: list of (check, status, detail))."""
    results = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    results.append(("required_columns", not missing_cols,
                    "missing: " + str(missing_cols) if missing_cols else "all present"))

    null_total = int(df.isnull().sum().sum())
    results.append(("missing_values", null_total == 0, f"{null_total} nulls"))

    dup = int(df.duplicated().sum())
    results.append(("duplicate_rows", dup == 0, f"{dup} duplicates"))

    if "SalesQuantity" in df.columns:
        numeric = pd.api.types.is_numeric_dtype(df["SalesQuantity"])
        results.append(("dtype_SalesQuantity_numeric", numeric, str(df["SalesQuantity"].dtype)))

    bad_range = 0
    for c in NON_NEGATIVE:
        if c in df.columns:
            bad_range += int((df[c] < 0).sum())
    results.append(("value_ranges_non_negative", bad_range == 0, f"{bad_range} negative values"))

    passed = all(ok for _, ok, _ in results)
    return passed, results


def main(path=DEFAULT_CSV):
    df = pd.read_csv(path, low_memory=False)
    passed, results = validate(df)
    print(f"Validation of {Path(path).name} ({len(df):,} rows)")
    for check, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {check}: {detail}")
    if not passed:
        print("VALIDATION FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)
