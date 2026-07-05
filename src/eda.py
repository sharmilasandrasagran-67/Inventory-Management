"""Exploratory Data Analysis (Q1b): distribution, relationship, categorical.

Saves three charts to reports/. Run: python src/eda.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data" / "inventory_demand_training.csv"
OUT = BASE_DIR / "reports"
BLUE, NAVY, GREEN = "#2E75B6", "#1F3864", "#548235"


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)

    # 1. Distribution
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(df["SalesQuantity"].clip(upper=15), bins=15, color=BLUE, edgecolor="white")
    ax[0].set_title("SalesQuantity distribution"); ax[0].set_xlabel("Units sold")
    ax[1].boxplot(df["SalesQuantity"], patch_artist=True,
                  boxprops=dict(facecolor="#D9E2F3", color=BLUE))
    ax[1].set_title("SalesQuantity boxplot")
    fig.tight_layout(); fig.savefig(OUT / "eda_distribution.png", dpi=140); plt.close(fig)

    # 2. Relationship / correlation
    numcols = ["SalesQuantity", "SalesPrice", "Volume", "ExciseTax",
               "Classification", "Store", "day_of_week", "day_of_month"]
    corr = df[numcols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numcols))); ax.set_xticklabels(numcols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(numcols))); ax.set_yticklabels(numcols, fontsize=7)
    fig.colorbar(im, fraction=0.046); ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig(OUT / "eda_relationship.png", dpi=140); plt.close(fig)

    # 3. Categorical
    fig, ax = plt.subplots(figsize=(7, 3.8))
    top = df.groupby("Store")["SalesQuantity"].sum().sort_values(ascending=False).head(10)
    ax.bar(top.index.astype(str), top.values, color=BLUE)
    ax.set_title("Top 10 stores by total units sold"); ax.set_xlabel("Store")
    fig.tight_layout(); fig.savefig(OUT / "eda_categorical.png", dpi=140); plt.close(fig)

    print("EDA charts written to", OUT)


if __name__ == "__main__":
    main()
