# Inventory Demand & Decisions Monitoring

An agile MVP for a wine & spirits retailer (Kaggle: *Inventory Analysis Case
Study* by Bhanu Pratap Biswas). It combines **all six source files** to support
four decisions, with a Streamlit front end and a GitHub Actions CI/CD pipeline.

## Decisions supported

1. **Demand forecast** — predict `SalesQuantity` (units) per sales record.
2. **Replenishment** — when to reorder, from sales velocity vs current stock.
3. **Discontinue** — which SKUs to stop (dead stock or very slow movers).
4. **Profitability** — revenue earned vs cost of goods, to rank profitable products.

## How the six files combine

`src/inventory_features.py` joins them on `InventoryId` / `Brand`:

| File | Contributes |
|------|-------------|
| SalesFINAL | units sold, revenue, sales velocity |
| PurchasesFINAL | units purchased, purchase cost, unit cost |
| BegInvFINAL / EndInvFINAL | beginning & ending on-hand stock |
| 2017PurchasePrices | fallback cost/retail price per product |
| InvoicePurchases | vendor invoice context |

It outputs two tables into `data/`:

- `inventory_features_store.csv` — one row per product-store (InventoryId)
- `inventory_features_product.csv` — one row per product (Brand) rollup

Each carries KPIs (daily demand, days-of-supply, sell-through, turnover, gross
margin, gross profit) and rule flags (`needs_reorder`, `discontinue_candidate`,
`profitable`) from `src/inventory_rules.py`.

## Date preprocessing (important)

In the full `SalesFINAL` file, `SalesDate` is **mixed-type**: January/February are
text (`1/13/2016`), while March–December are real Excel dates (and can appear as
ISO strings or Excel serial numbers). `src/preprocess.normalize_sales_date`
aligns all of these to a single date so the full year is usable. The copy commonly
downloaded via Excel is **truncated to Jan–Feb** (1,048,575 rows = Excel's limit);
re-download the untruncated file from Kaggle for full-year results.

## Models (exported to JSON, no sklearn needed at inference)

- `train_model.py` -> `model/model_params.json` (demand: LinearRegression)
- `train_inventory_models.py` -> `model/inventory_models.json`:
  - `profit_regressor` (LinearRegression) — predicts gross profit
  - `reorder_classifier` (LogisticRegression) — P(needs reorder)
  - `discontinue_classifier` (LogisticRegression) — P(discontinue)

The reorder/discontinue labels come from the transparent business rules; the
classifiers learn those policies so they can score new or partially-known SKUs.

## Structure

```
inventory-demand-monitoring/
├── data/                         # prepared CSVs (logs gitignored)
├── src/
│   ├── preprocess.py             # robust mixed-format SalesDate handling
│   ├── train_model.py            # demand model
│   ├── predict_utils.py, log_utils.py
│   ├── inventory_rules.py        # KPIs + business-rule flags
│   ├── inventory_features.py     # builds consolidated tables from 6 raw files
│   ├── train_inventory_models.py # reorder / discontinue / profit models
│   └── inventory_predict_utils.py
├── dashboard/
│   ├── dashboard.py              # demand prediction & monitoring
│   └── inventory_dashboard.py    # replenish / discontinue / profitability
├── tests/                        # pytest (test_main.py, test_inventory.py)
├── model/                        # generated JSON params
├── .github/workflows/ci_cd_pipeline.yml
├── requirements.txt, runtime.txt, .gitignore
```

## Run locally

```bash
pip install -r requirements.txt
python src/inventory_features.py --raw-dir path/to/kaggle/files   # rebuild tables (optional)
python src/train_model.py
python src/train_inventory_models.py
pytest tests/ -v
streamlit run dashboard/inventory_dashboard.py
```

## Notes & limitations

- The committed `inventory_features_store.csv` is a 60k-row sample so the repo stays
  light; `inventory_features.py` regenerates the full table from the raw files.
- Because the Excel copy of sales is Jan–Feb only, velocity/days-of-supply are based
  on ~2 months; verify against the full-year file before acting operationally.
- Reorder/discontinue classifiers reproduce rule-based labels (policy learning),
  so treat their high accuracy as automation of the rules, not independent proof.

## Security

Never commit access tokens. Authenticate Git pushes with a token entered at runtime
(`getpass`) or a credential manager.
