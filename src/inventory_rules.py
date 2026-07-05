"""KPIs and business rules for inventory decisions.

These are pure functions that operate on an aggregated dataframe (one row per
InventoryId or per Brand). They power three decisions:

  * needs_reorder         -> replenish based on sales velocity vs current stock
  * discontinue_candidate -> stop products with no / very weak demand (dead or slow stock)
  * profitable            -> revenue earned vs cost of goods sold

Thresholds live in RULE_CONFIG so they can be tuned without touching the logic.
"""

import numpy as np
import pandas as pd

RULE_CONFIG = {
    "lead_time_days": 7,     # supplier lead time
    "safety_days": 7,        # extra buffer beyond lead time
    "slow_days_of_supply": 180,   # > 6 months of stock = slow mover
    "min_margin_pct": 15.0,  # threshold for "high margin"
}


def add_kpis(df, config=RULE_CONFIG):
    """Add inventory + profitability KPIs. Expects these columns:
    beg_onhand, end_onhand, units_sold, units_purchased, sales_revenue,
    unit_cost, unit_sell, sales_span_days.
    """
    df = df.copy()

    df["cogs"] = df["units_sold"] * df["unit_cost"]
    df["gross_profit"] = df["sales_revenue"] - df["cogs"]
    df["gross_margin_pct"] = np.where(
        df["sales_revenue"] > 0, df["gross_profit"] / df["sales_revenue"] * 100, np.nan
    )
    df["margin_per_unit"] = df["unit_sell"] - df["unit_cost"]

    df["daily_demand"] = df["units_sold"] / df["sales_span_days"].replace(0, np.nan)
    avg_inv = (df["beg_onhand"] + df["end_onhand"]) / 2
    df["inventory_turnover"] = np.where(avg_inv > 0, df["units_sold"] / avg_inv, np.nan)
    df["days_of_supply"] = np.where(
        df["daily_demand"] > 0, df["end_onhand"] / df["daily_demand"], np.nan
    )
    available = df["beg_onhand"] + df["units_purchased"]
    df["sell_through_pct"] = np.where(
        available > 0, df["units_sold"] / available * 100, np.nan
    )
    return df


def add_rule_flags(df, config=RULE_CONFIG):
    """Add the three decision flags from the KPIs."""
    df = df.copy()
    lead = config["lead_time_days"]
    safety = config["safety_days"]
    slow = config["slow_days_of_supply"]
    min_margin = config["min_margin_pct"]

    # Replenishment: stock at/under the reorder point for products that still sell.
    df["reorder_point"] = df["daily_demand"].fillna(0) * (lead + safety)
    df["needs_reorder"] = (
        (df["daily_demand"].fillna(0) > 0) & (df["end_onhand"] <= df["reorder_point"])
    ).astype(int)

    # Discontinue: dead stock (no sales but holding units) or very slow movers.
    df["dead_stock"] = (
        (df["units_sold"] == 0) & (df["end_onhand"] > 0)
    ).astype(int)
    df["slow_mover"] = (
        (df["units_sold"] > 0) & (df["days_of_supply"] > slow)
    ).astype(int)
    df["discontinue_candidate"] = (
        (df["dead_stock"] == 1) | (df["slow_mover"] == 1)
    ).astype(int)

    # Profitability.
    df["profitable"] = (df["gross_profit"] > 0).astype(int)
    df["high_margin"] = (df["gross_margin_pct"] >= min_margin).astype(int)
    return df


def build_decision_table(df, config=RULE_CONFIG):
    """Convenience: KPIs + flags in one call."""
    return add_rule_flags(add_kpis(df, config), config)
