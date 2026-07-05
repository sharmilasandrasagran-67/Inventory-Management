"""Data drift & model performance degradation (Q5b).

Compares training (January) vs production (February) feature distributions using
the Population Stability Index (PSI), and checks whether the demand model's MAE
degrades from the training holdout to production. Saves a chart to reports/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN = BASE_DIR / "data" / "inventory_demand_training.csv"
PROD = BASE_DIR / "data" / "inventory_demand_production.csv"
OUT = BASE_DIR / "reports"
FEATURES = ["SalesPrice", "Volume", "Classification", "Store", "day_of_week", "day_of_month"]


def psi(expected, actual, bins=10):
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, bins=qs)[0] / len(expected), 1e-4, None)
    a = np.clip(np.histogram(actual, bins=qs)[0] / len(actual), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    OUT.mkdir(exist_ok=True)
    train, prod = pd.read_csv(TRAIN), pd.read_csv(PROD)
    drift_feats = ["SalesPrice", "Volume", "SalesQuantity", "day_of_month"]
    psis = {f: psi(train[f].values, prod[f].values) for f in drift_feats}

    Xtr, Xte, ytr, yte = train_test_split(train[FEATURES], train["SalesQuantity"],
                                          test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    mae_train = mean_absolute_error(yte, m.predict(Xte))
    mae_prod = mean_absolute_error(prod["SalesQuantity"], m.predict(prod[FEATURES]))

    print("PSI (>=0.2 = significant drift):")
    for f, v in psis.items():
        flag = "DRIFT" if v >= 0.2 else ("moderate" if v >= 0.1 else "stable")
        print(f"  {f}: {v:.3f} ({flag})")
    print(f"MAE train holdout={mae_train:.3f}  production={mae_prod:.3f}  "
          f"change={100*(mae_prod-mae_train)/mae_train:+.1f}%")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cols = ["#C00000" if v >= 0.2 else "#BF9000" if v >= 0.1 else "#2E75B6" for v in psis.values()]
    ax[0].bar(list(psis.keys()), list(psis.values()), color=cols)
    ax[0].axhline(0.1, ls="--", color="#BF9000"); ax[0].axhline(0.2, ls="--", color="#C00000")
    ax[0].set_title("Feature drift (PSI)")
    ax[1].bar(["train", "production"], [mae_train, mae_prod], color=["#2E75B6", "#C00000"])
    ax[1].set_title("Model MAE: train vs production")
    fig.tight_layout(); fig.savefig(OUT / "drift.png", dpi=140); plt.close(fig)
    print("Chart written to", OUT / "drift.png")


if __name__ == "__main__":
    main()
