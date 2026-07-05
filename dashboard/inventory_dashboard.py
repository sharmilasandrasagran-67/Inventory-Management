"""Inventory decision dashboard: replenishment, dead stock, and profitability.

Reads the consolidated tables built by src/inventory_features.py and the models
trained by src/train_inventory_models.py. Run with:

    streamlit run dashboard/inventory_dashboard.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from inventory_predict_utils import recommend, load_inventory_models

STORE_PATH = BASE_DIR / "data" / "inventory_features_store.csv"
PRODUCT_PATH = BASE_DIR / "data" / "inventory_features_product.csv"

st.set_page_config(page_title="Inventory Decisions Dashboard", layout="wide")
st.title("Inventory Decisions: Replenish, Discontinue, Profitability")
st.caption(
    "Built from all six source files (sales, purchases, beginning/ending inventory, "
    "invoice, prices). Decisions combine transparent business rules with ML models."
)


@st.cache_data
def load_store():
    return pd.read_csv(STORE_PATH, low_memory=False)


@st.cache_data
def load_product():
    return pd.read_csv(PRODUCT_PATH, low_memory=False)


@st.cache_data
def load_models():
    return load_inventory_models()


store = load_store()
product = load_product()

level = st.radio("Analysis level", ["Product-store (SKU)", "Product (Brand)"], horizontal=True)
df = store if level.startswith("Product-store") else product

tab1, tab2, tab3, tab4 = st.tabs([
    "Replenishment", "Dead Stock / Discontinue", "Profitability", "Score a SKU",
])

# ------------------------------------------------------------ Replenishment
with tab1:
    st.header("Replenishment — what to reorder now")
    st.write("Items whose current stock is at or below the reorder point while demand continues.")
    reorder = df[df["needs_reorder"] == 1].copy()
    c1, c2 = st.columns(2)
    c1.metric("Items needing reorder", f"{len(reorder):,}")
    c2.metric("Share of catalogue", f"{100*len(reorder)/max(len(df),1):.1f}%")
    show = ["Description", "end_onhand", "daily_demand", "days_of_supply",
            "reorder_point", "units_sold", "gross_margin_pct"]
    if level.startswith("Product-store"):
        show = ["InventoryId", "Store"] + show
    show = [c for c in show if c in reorder.columns]
    st.dataframe(reorder.sort_values("daily_demand", ascending=False)[show].head(200))

# ------------------------------------------------------------ Dead stock
with tab2:
    st.header("Dead stock / discontinue candidates")
    st.write("No sales but stock on hand (dead), or more than ~6 months of supply (slow mover).")
    dead = df[df["discontinue_candidate"] == 1].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Discontinue candidates", f"{len(dead):,}")
    if "dead_stock" in dead:
        c2.metric("Dead (zero sales)", f"{int(dead['dead_stock'].sum()):,}")
    if "slow_mover" in dead:
        c3.metric("Slow movers", f"{int(dead['slow_mover'].sum()):,}")
    show = ["Description", "end_onhand", "units_sold", "days_of_supply", "sell_through_pct"]
    if level.startswith("Product-store"):
        show = ["InventoryId", "Store"] + show
    show = [c for c in show if c in dead.columns]
    st.dataframe(dead.sort_values("end_onhand", ascending=False)[show].head(200))

# ------------------------------------------------------------ Profitability
with tab3:
    st.header("Profitability — revenue earned vs cost")
    total_rev = df["sales_revenue"].sum()
    total_profit = df["gross_profit"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total revenue", f"${total_rev:,.0f}")
    c2.metric("Total gross profit", f"${total_profit:,.0f}")
    c3.metric("Overall margin", f"{100*total_profit/max(total_rev,1):.1f}%")

    st.subheader("Top profitable items")
    show = ["Description", "units_sold", "sales_revenue", "gross_profit", "gross_margin_pct"]
    if level.startswith("Product-store"):
        show = ["InventoryId", "Store"] + show
    show = [c for c in show if c in df.columns]
    st.dataframe(df.sort_values("gross_profit", ascending=False)[show].head(50))

    st.subheader("Loss-making / unprofitable items")
    loss = df[df["gross_profit"] <= 0]
    st.metric("Unprofitable items", f"{len(loss):,}")
    st.dataframe(loss.sort_values("gross_profit")[show].head(50))

# ------------------------------------------------------------ SKU scorer
with tab4:
    st.header("Score a single SKU with the models")
    st.write("Enter attributes to get model-based reorder, discontinue, and profit predictions.")
    col = st.columns(3)
    units_sold = col[0].number_input("Units sold (period)", 0.0, value=50.0)
    unit_sell = col[1].number_input("Unit sell price", 0.0, value=20.0)
    unit_cost = col[2].number_input("Unit cost", 0.0, value=14.0)
    volume = col[0].number_input("Volume (mL)", 0.0, value=750.0)
    classification = col[1].selectbox("Classification", [1, 2])
    end_onhand = col[2].number_input("End on-hand units", 0.0, value=5.0)
    daily_demand = col[0].number_input("Daily demand", 0.0, value=0.8)
    inv_turnover = col[1].number_input("Inventory turnover", 0.0, value=4.0)
    sell_through = col[2].number_input("Sell-through %", 0.0, value=60.0)
    dos = col[0].number_input("Days of supply", 0.0, value=6.0)
    margin_pct = col[1].number_input("Gross margin %", -100.0, 100.0, value=30.0)

    row = {"units_sold": units_sold, "unit_sell": unit_sell, "unit_cost": unit_cost,
           "Volume": volume, "Classification": classification, "end_onhand": end_onhand,
           "daily_demand": daily_demand, "inventory_turnover": inv_turnover,
           "sell_through_pct": sell_through, "days_of_supply": dos,
           "gross_margin_pct": margin_pct}
    if st.button("Score SKU"):
        rec = recommend(row, load_models())
        a, b, c = st.columns(3)
        a.metric("Reorder?", "YES" if rec["needs_reorder"] else "no",
                 f"p={rec['reorder_probability']:.2f}")
        b.metric("Discontinue?", "YES" if rec["discontinue_candidate"] else "no",
                 f"p={rec['discontinue_probability']:.2f}")
        c.metric("Predicted gross profit", f"${rec['predicted_gross_profit']:,.2f}")
