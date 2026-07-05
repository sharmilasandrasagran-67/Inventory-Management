"""Build consolidated inventory decision tables from the six raw Kaggle files.

Outputs two CSVs into data/:
  * inventory_features_store.csv   -> one row per product-store (InventoryId)
  * inventory_features_product.csv -> one row per product (Brand) rollup

Each row carries KPIs (velocity, days-of-supply, sell-through, turnover, margin,
gross profit) and the three decision flags (needs_reorder, discontinue_candidate,
profitable) produced by inventory_rules.

Run AFTER downloading the raw files (e.g. via kagglehub). Point RAW_DIR at the
folder that contains the six CSVs. The large SalesFINAL / PurchasesFINAL files are
read in chunks to keep memory modest.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from preprocess import load_sales, normalize_sales_date
from inventory_rules import build_decision_table, RULE_CONFIG

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data"

SALES_FILE = "SalesFINAL12312016.csv"
PURCH_FILE = "PurchasesFINAL12312016.csv"
BEG_FILE = "BegInvFINAL12312016.csv"
END_FILE = "EndInvFINAL12312016.csv"
PRICE_FILE = "2017PurchasePricesDec.csv"


def aggregate_sales(raw_dir):
    cols = ["InventoryId", "Brand", "Store", "Description", "Size",
            "SalesQuantity", "SalesDollars", "SalesPrice", "SalesDate",
            "Volume", "Classification", "VendorNo"]
    # Robust mixed-format date handling happens inside load_sales().
    s = load_sales(Path(raw_dir) / SALES_FILE, usecols=cols)
    span_days = (s["SalesDate"].max() - s["SalesDate"].min()).days + 1
    agg = s.groupby("InventoryId").agg(
        Brand=("Brand", "first"), Store=("Store", "first"),
        Description=("Description", "first"), Size=("Size", "first"),
        Volume=("Volume", "first"), Classification=("Classification", "first"),
        VendorNo=("VendorNo", "first"),
        units_sold=("SalesQuantity", "sum"),
        sales_revenue=("SalesDollars", "sum"),
        avg_sales_price=("SalesPrice", "mean"),
        n_sale_days=("SalesDate", "nunique"),
        last_sale=("SalesDate", "max"),
    ).reset_index()
    agg["sales_span_days"] = span_days
    print(f"[features] sales: {len(agg)} product-stores over {span_days} days")
    return agg


def aggregate_purchases(raw_dir, chunksize=400000):
    cols = ["InventoryId", "Quantity", "Dollars", "PurchasePrice"]
    parts = []
    for chunk in pd.read_csv(Path(raw_dir) / PURCH_FILE, usecols=cols, chunksize=chunksize):
        parts.append(chunk.groupby("InventoryId").agg(
            units_purchased=("Quantity", "sum"),
            purchase_cost=("Dollars", "sum"),
            sum_pp=("PurchasePrice", "sum"),
            n_pp=("PurchasePrice", "count"),
        ))
    res = pd.concat(parts).groupby("InventoryId").agg(
        units_purchased=("units_purchased", "sum"),
        purchase_cost=("purchase_cost", "sum"),
        sum_pp=("sum_pp", "sum"),
        n_pp=("n_pp", "sum"),
    )
    res["avg_purchase_price"] = res["sum_pp"] / res["n_pp"]
    res = res.drop(columns=["sum_pp", "n_pp"]).reset_index()
    print(f"[features] purchases: {len(res)} product-stores")
    return res


def build_store_table(raw_dir, config=RULE_CONFIG):
    raw_dir = Path(raw_dir)
    sales = aggregate_sales(raw_dir)
    purch = aggregate_purchases(raw_dir)

    beg = pd.read_csv(raw_dir / BEG_FILE,
                      usecols=["InventoryId", "Store", "City", "Brand",
                               "Description", "Size", "onHand", "Price"])
    beg = beg.rename(columns={"onHand": "beg_onhand", "Price": "unit_price"})
    end = pd.read_csv(raw_dir / END_FILE, usecols=["InventoryId", "onHand"])
    end = end.rename(columns={"onHand": "end_onhand"})
    end = end.groupby("InventoryId", as_index=False)["end_onhand"].sum()

    prices = pd.read_csv(raw_dir / PRICE_FILE,
                         usecols=["Brand", "Price", "PurchasePrice"])
    prices = prices.rename(columns={"Price": "ref_retail_price",
                                    "PurchasePrice": "ref_purchase_price"})
    prices = prices.groupby("Brand", as_index=False).first()

    ids = pd.Index(beg["InventoryId"]).union(end["InventoryId"]) \
        .union(sales["InventoryId"]).union(purch["InventoryId"])
    df = pd.DataFrame({"InventoryId": ids})
    df = df.merge(beg, on="InventoryId", how="left")
    df = df.merge(end, on="InventoryId", how="left")
    df = df.merge(sales, on="InventoryId", how="left", suffixes=("", "_s"))
    df = df.merge(purch, on="InventoryId", how="left")
    for c in ["Brand", "Store", "Description", "Size"]:
        if c + "_s" in df:
            df[c] = df[c].fillna(df[c + "_s"])
    df = df.merge(prices, on="Brand", how="left")

    for c in ["beg_onhand", "end_onhand", "units_sold", "sales_revenue",
              "units_purchased", "purchase_cost"]:
        df[c] = df[c].fillna(0)
    df["sales_span_days"] = df["sales_span_days"].fillna(df["sales_span_days"].max())

    # Unit cost: purchase average -> price-table cost -> on-hand price.
    df["unit_cost"] = (df["avg_purchase_price"]
                       .fillna(df["ref_purchase_price"]).fillna(df["unit_price"]))
    # Unit sell price: realised average -> price-table retail -> on-hand price.
    df["unit_sell"] = (df["avg_sales_price"]
                       .fillna(df["ref_retail_price"]).fillna(df["unit_price"]))

    df = build_decision_table(df, config)
    return df


def rollup_to_product(store_df, config=RULE_CONFIG):
    prod = store_df.groupby("Brand").agg(
        Description=("Description", "first"), Size=("Size", "first"),
        Classification=("Classification", "first"), VendorNo=("VendorNo", "first"),
        n_store_skus=("InventoryId", "nunique"),
        beg_onhand=("beg_onhand", "sum"), end_onhand=("end_onhand", "sum"),
        units_sold=("units_sold", "sum"), units_purchased=("units_purchased", "sum"),
        sales_revenue=("sales_revenue", "sum"),
        unit_cost=("unit_cost", "mean"), unit_sell=("unit_sell", "mean"),
        sales_span_days=("sales_span_days", "max"),
    ).reset_index()
    prod = build_decision_table(prod, config)
    return prod


STORE_COLS = ["InventoryId", "Store", "City", "Brand", "Description", "Size",
              "Volume", "Classification", "VendorNo", "beg_onhand", "end_onhand",
              "units_sold", "units_purchased", "sales_revenue", "unit_cost",
              "unit_sell", "cogs", "gross_profit", "gross_margin_pct",
              "margin_per_unit", "daily_demand", "inventory_turnover",
              "days_of_supply", "sell_through_pct", "reorder_point",
              "sales_span_days", "needs_reorder", "dead_stock", "slow_mover",
              "discontinue_candidate", "profitable", "high_margin"]


def main(raw_dir):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store = build_store_table(raw_dir)
    store_out = store[[c for c in STORE_COLS if c in store.columns]].copy()
    store_out.to_csv(OUT_DIR / "inventory_features_store.csv", index=False)

    prod = rollup_to_product(store)
    prod.to_csv(OUT_DIR / "inventory_features_product.csv", index=False)

    print(f"[features] wrote {len(store_out)} store rows and {len(prod)} product rows")
    print(f"[features] needs_reorder={int(store['needs_reorder'].sum())} "
          f"discontinue={int(store['discontinue_candidate'].sum())} "
          f"profitable={int(store['profitable'].sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(BASE_DIR / "raw"),
                        help="Folder containing the six raw Kaggle CSV files")
    args = parser.parse_args()
    main(args.raw_dir)
