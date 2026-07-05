"""Data preprocessing helpers.

The raw SalesFINAL file has an inconsistent SalesDate column: January/February
values are stored as TEXT in M/D/YYYY form (e.g. "1/13/2016"), while later months
were saved by Excel as real date values (and may appear as ISO strings such as
"2016-05-13" or even as Excel serial numbers like "42503"). A naive parse keeps
only the text dates and silently drops or misreads the rest.

`normalize_sales_date` aligns all of these to a single pandas datetime so the full
year can be used downstream.
"""

import numpy as np
import pandas as pd

# Excel's day-zero is 1899-12-30 (Excel incorrectly treats 1900 as a leap year).
EXCEL_EPOCH = "1899-12-30"


def normalize_sales_date(series):
    """Return a pandas datetime Series, handling mixed SalesDate formats.

    Strategy (each pass only fills values the previous pass could not parse):
      1. US text dates            -> "%m/%d/%Y"  (e.g. 1/13/2016)
      2. Any remaining strings    -> per-element parser (e.g. 2016-05-13, with time)
      3. Purely numeric leftovers -> treated as Excel serial day numbers
    """
    raw = series.astype(str).str.strip()

    # Pass 1: explicit US text format (covers the Jan/Feb text values).
    out = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")

    # Pass 2: per-element parser for ISO / real-date strings (covers Mar-Dec),
    # including values that carry a time suffix like "2016-12-31 00:00:00".
    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(
            raw[mask], errors="coerce", format="mixed", dayfirst=False
        )

    # Pass 3: Excel serial numbers (e.g. "42503") -> calendar dates.
    mask = out.isna()
    if mask.any():
        nums = pd.to_numeric(raw[mask], errors="coerce")
        out.loc[mask] = pd.to_datetime(
            nums, unit="D", origin=EXCEL_EPOCH, errors="coerce"
        )

    return out


def load_sales(path, usecols=None):
    """Load a sales CSV, normalise SalesDate, and add calendar features.

    Adds: SalesDate (datetime), SalesDateISO (YYYY-MM-DD text), month,
    day_of_week, day_of_month. Rows whose date cannot be parsed are reported
    and dropped so they do not corrupt downstream aggregation.
    """
    df = pd.read_csv(path, usecols=usecols, dtype={"SalesDate": str})

    df["SalesDate"] = normalize_sales_date(df["SalesDate"])

    n_bad = int(df["SalesDate"].isna().sum())
    if n_bad:
        print(f"[preprocess] dropped {n_bad} rows with unparseable SalesDate")
        df = df.dropna(subset=["SalesDate"])

    df["SalesDateISO"] = df["SalesDate"].dt.strftime("%Y-%m-%d")
    df["month"] = df["SalesDate"].dt.month
    df["day_of_week"] = df["SalesDate"].dt.dayofweek
    df["day_of_month"] = df["SalesDate"].dt.day

    return df
