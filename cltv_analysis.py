# Customer Lifetime Value (CLTV) Prediction — Olist E-Commerce
#
# This plain Python script contains the full CLTV analysis. It is generated from
# the notebook cells so GitHub can always preview the project even when its
# notebook renderer is unreliable.

try:
    from IPython.display import display
except ImportError:  # Allows the script to run outside Jupyter, too.
    def display(obj):
        print(obj)


# %% [markdown]
# # Customer Lifetime Value (CLTV) Prediction — Olist E-Commerce
# 
# ---
# 
# **Author:** `<Zofia Pempera>`
# 
# ## Goal
# 
# The goal of this project is to build a predictive framework that estimates **Customer Lifetime Value
# (CLTV)** using **two complementary approaches**:
# 1. **Probabilistic (BTYD)** models — BG/NBD + Gamma-Gamma
# 2. **Machine Learning** regression — predicting holdout-period revenue
# 
# ## Dataset
# **Brazilian E-Commerce Public Dataset by Olist** (Kaggle: `olistbr/brazilian-ecommerce`).
# A set of relational tables (orders, customers, payments, items, …) covering ~100k orders placed
# between **2016 and 2018**. These tables are aggregated into a single **customer-level transactional
# dataset** suitable for CLTV analysis.
# 
# - **Calibration period:** used to *train* the models.
# - **Holdout period:** used to *validate* the predictions against real future behavior.
# 
# ## Methodology Overview
# | Step | Description |
# |------|-------------|
# | **A** | Data Audit & Cleaning — duplicates, missing values, returns/cancellations, outliers |
# | **B** | EDA + RFM Feature Engineering (Recency, Frequency, Monetary, Tenure *T*) |
# | **C** | Modeling — BG/NBD, Gamma-Gamma, 90-day CLTV, ML regression, model comparison |
# | **D** | Marketing recommendations (presentation deliverable) |
# 
# ## Key Deliverables
# - Estimated **total value of the customer base** (financial planning, macro view).
# - A targeted **"win-back" campaign** prioritizing customers by predicted CLTV and *P(alive)* (micro view).
# 
# ---
# *Tools: Python · pandas · lifetimes (BG/NBD + Gamma-Gamma) · scikit-learn · XGBoost · matplotlib/seaborn*

# %%
# SETUP
# Standard Jupyter notebook setup.
# Install dependencies before opening/running the notebook:
#     pip install -r requirements.txt

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import (
    summary_data_from_transaction_data,
    calibration_and_holdout_data,
)
from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
    plot_period_transactions,
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 5)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Download the public Olist dataset from Kaggle.
# If this fails, run `pip install -r requirements.txt` again and make sure your
# environment has internet access.
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
dataset_path = Path(path)
print("Dataset downloaded to:", dataset_path)

print("\nFiles in dataset:")
for file in sorted(dataset_path.iterdir()):
    print("  -", file.name)

print("\n✓ Setup complete. Libraries imported, data ready.")

# %% [markdown]
# ## Step 1 — Loading the Olist Relational Tables
# 
# The Olist dataset is **not a single file** — it is a set of **9 relational tables** that
# mirror a real e-commerce database. To build a customer-level dataset for CLTV, we need to
# understand how they connect:
# 
# | Table | Role | Key columns |
# |-------|------|-------------|
# | `orders` | One row per order (status, timestamps) | `order_id`, `customer_id` |
# | `order_items` | One row per item in an order (price, freight) | `order_id`, `product_id`, `price` |
# | `order_payments` | Payment info per order | `order_id`, `payment_value` |
# | `customers` | Links each order to a *unique* person | `customer_id`, `customer_unique_id` |
# | `products` / `reviews` / `sellers` / `geolocation` / `category_translation` | Context (not core to CLTV) | — |
# 
# ### Critical detail: `customer_id` vs `customer_unique_id`
# In Olist, **`customer_id` is a per-order identifier** — the *same* person gets a *new*
# `customer_id` for every order they place. The column that actually identifies a real,
# recurring customer is **`customer_unique_id`**.
# 
# This matters enormously for CLTV: if we used `customer_id`, **every customer would look like
# a one-time buyer** and our "Buy Till You Die" models would be meaningless. We must join through
# `customers` and aggregate on `customer_unique_id`.

# %%
# Helper: build full path to a CSV inside the downloaded dataset folder
def load(name):
    return pd.read_csv(dataset_path / name)

# Load the tables we need for CLTV
orders     = load("olist_orders_dataset.csv")
items      = load("olist_order_items_dataset.csv")
payments   = load("olist_order_payments_dataset.csv")
customers  = load("olist_customers_dataset.csv")

# Quick look at the shape and head of each core table
for name, df in [("orders", orders), ("items", items),
                 ("payments", payments), ("customers", customers)]:
    print(f"\n{'='*60}\n{name.upper()}  —  shape: {df.shape}\n{'='*60}")
    display(df.head(3))

# Parse order timestamps as datetime (needed for the calibration/holdout split later)
date_cols = [c for c in orders.columns if "timestamp" in c or "date" in c]
for c in date_cols:
    orders[c] = pd.to_datetime(orders[c], errors="coerce")

print("\nParsed date columns in `orders`:", date_cols)

# %% [markdown]
# ## Data Audit & Cleaning
# 
# Before any modeling, we assess data integrity and build a cleaning pipeline. Our objectives:
# 
# 1. **Audit** — quantify missing values, duplicates, and order-status anomalies.
# 2. **Filter invalid transactions** — keep only orders that represent real, completed revenue.
# 3. **Handle returns / cancellations** — exclude `canceled` and `unavailable` orders.
# 4. **Treat outliers** — cap extreme order values that would distort the monetary models.
# 
# ### Why order status matters for CLTV
# CLTV is built on delivered, paid-for purchases. The `orders` table contains statuses like
# `delivered`, `shipped`, `canceled`, `unavailable`, `invoiced`, etc. Including canceled or
# unavailable orders would inflate frequency and monetary value for revenue that never materialized.
# We therefore restrict the analysis to **`delivered`** orders — the only status that guarantees a
# completed transaction.

# %%
# Missing values
def missing_report(df, name):
    """Return a tidy table of missing values per column (only columns with NaNs)."""
    miss = df.isnull().sum()
    pct = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": miss, "missing_pct": pct})
    report = report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)
    report.insert(0, "table", name)
    return report

tables = {"orders": orders, "items": items, "payments": payments, "customers": customers}

missing_all = pd.concat([missing_report(df, n) for n, df in tables.items()])

print("MISSING VALUES — only columns that have any (others are 100% complete)\n")
if len(missing_all):
    display(missing_all.reset_index().rename(columns={"index": "column"}))
else:
    print("No missing values in any core table.")

print("\nTables with NO missing values:",
      [n for n, df in tables.items() if df.isnull().sum().sum() == 0])

# %%
# Duplicate rows

dup_report = pd.DataFrame({
    "table": list(tables.keys()),
    "total_rows": [len(df) for df in tables.values()],
    "duplicate_rows": [df.duplicated().sum() for df in tables.values()],
})
dup_report["duplicate_pct"] = (dup_report["duplicate_rows"] /
                               dup_report["total_rows"] * 100).round(3)

print("DUPLICATE ROW CHECK\n")
display(dup_report)

# %%
# Order status distribution

status = orders["order_status"].value_counts().rename_axis("order_status").reset_index(name="count")
status["pct"] = (status["count"] / len(orders) * 100).round(2)
status["counts_as_revenue?"] = np.where(status["order_status"] == "delivered", "yes", "no")

print("ORDER STATUS DISTRIBUTION\n")
display(status)

# Visual: bar chart of statuses (log scale so small categories are visible)
plt.figure(figsize=(9, 4))
sns.barplot(data=status, x="order_status", y="count",
            hue="order_status", palette="viridis", legend=False)
plt.yscale("log")
plt.title("Order Status Distribution (log scale)")
plt.xlabel(""); plt.ylabel("Number of orders (log)")
plt.xticks(rotation=30)
plt.tight_layout(); plt.show()

# %%
# Outlier inspection on ORDER VALUE
_items_agg = (items.groupby("order_id")
                   .agg(order_value=("price", "sum"),
                        freight=("freight_value", "sum"))
                   .reset_index())
_items_agg["order_value"] = _items_agg["order_value"] + _items_agg["freight"]

ov = _items_agg["order_value"]

# Numeric summary with key percentiles
summary = ov.describe(percentiles=[.25, .5, .75, .90, .95, .99]).to_frame("order_value")
summary.loc["IQR"] = summary.loc["75%", "order_value"] - summary.loc["25%", "order_value"]
print("ORDER VALUE — numeric summary (R$)\n")
display(summary.round(2))

# IQR-based outlier flagging (standard 1.5 * IQR rule)
Q1, Q3 = ov.quantile(.25), ov.quantile(.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
n_out = (ov > upper_fence).sum()

outlier_tbl = pd.DataFrame({
    "metric": ["Q1 (25%)", "Q3 (75%)", "IQR", "Upper fence (Q3 + 1.5·IQR)",
               "99th percentile", "Max", "Orders above upper fence", "% above upper fence"],
    "value": [Q1, Q3, IQR, upper_fence, ov.quantile(.99), ov.max(),
              n_out, round(n_out / len(ov) * 100, 2)],
})
print("OUTLIER FLAGGING (IQR rule)\n")
display(outlier_tbl.round(2))

# Visual: boxplot + histogram so you can SEE the skew and the tail
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(x=ov, ax=axes[0], color="salmon")
axes[0].set_title("Order Value — Boxplot (raw)")
axes[0].set_xlabel("Order value (R$)")

sns.histplot(ov, bins=80, ax=axes[1], color="steelblue")
axes[1].set_title("Order Value — Distribution (raw)")
axes[1].set_xlabel("Order value (R$)")
axes[1].axvline(upper_fence, color="red", ls="--", label=f"Upper fence ≈ R$ {upper_fence:,.0f}")
axes[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Audit Findings & Cleaning Decisions
# 
# - **0 duplicate rows** in all core tables = No action needed
# - **Missing values only in `orders`**: `order_approved_at` (160), `order_delivered_carrier_date` (1 783), `order_delivered_customer_date` (2 965) | **No imputation** - we don't use them
# - **Order status**: 96 478 `delivered` vs ~3 100 non-delivered (canceled, unavailable, in-progress) **Keep only `delivered`**
# - **Date range**: 2016-09-04 → 2018-10-17 (~25 months)
# - **Order value** right-skewed with extreme tail | **Capped at 99th percentile** | Preserves genuine high-value customers (critical for CLTV) while preventing a few extreme orders from distorting the Gamma-Gamma monetary model. |
# 
# **Key takeaway:** The dataset is structurally clean. The main "cleaning" work is conceptual —
# correctly identifying which orders count as revenue (`delivered`) and which customer identifier
# represents a real recurring person (`customer_unique_id`).

# %%
# CLEANING

# Keep only DELIVERED orders
n0 = len(orders)
orders_clean = orders[orders["order_status"] == "delivered"].copy()
orders_clean = orders_clean.dropna(subset=["order_purchase_timestamp"])
print(f"[1] Orders: {n0:,} -> {len(orders_clean):,} (kept only 'delivered')")

# Order-level monetary value (item price + freight)
items_agg = (items.groupby("order_id")
                  .agg(order_value=("price", "sum"),
                       freight=("freight_value", "sum"))
                  .reset_index())
items_agg["order_value"] = items_agg["order_value"] + items_agg["freight"]
print(f"[2] Built order_value for {len(items_agg):,} orders")

#Merge orders -> items -> customers
# Aggregate on customer_unique_id (the REAL recurring person), not customer_id.
df = (orders_clean[["order_id", "customer_id", "order_purchase_timestamp"]]
      .merge(items_agg[["order_id", "order_value"]], on="order_id", how="inner")
      .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left"))

df = df.rename(columns={"customer_unique_id": "customer",
                        "order_purchase_timestamp": "date"})
df = df[["customer", "order_id", "date", "order_value"]]
print(f"[3] Merged transactional rows: {len(df):,}")

# Defensive: drop rows where the customer link failed (should be ~0 in Olist)
n_nocust = df["customer"].isnull().sum()
if n_nocust:
    df = df.dropna(subset=["customer"])
    print(f"    Dropped {n_nocust:,} rows with no customer_unique_id link")

# --- Step 4: Outlier capping at 99th percentile + remove non-positive --------
cap = df["order_value"].quantile(0.99)
n_capped = (df["order_value"] > cap).sum()
df["order_value"] = df["order_value"].clip(upper=cap)
df = df[df["order_value"] > 0].copy()
print(f"[4] Capped {n_capped:,} orders above 99th pct (R$ {cap:,.2f}); "
      f"removed non-positive values")

print(f"\n✓ CLEAN transactional dataset: {len(df):,} rows, "
      f"{df['customer'].nunique():,} unique customers")

# %%
# ==============================================================================
# TASK A — POST-CLEANING SANITY CHECK
# ==============================================================================

checks = pd.DataFrame({
    "check": [
        "Transactions (rows)",
        "Unique customers",
        "Unique orders",
        "Missing values remaining",
        "Non-positive order values",
        "Date range start",
        "Date range end",
        "Mean order value (R$)",
        "Median order value (R$)",
    ],
    "result": [
        f"{len(df):,}",
        f"{df['customer'].nunique():,}",
        f"{df['order_id'].nunique():,}",
        f"{df.isnull().sum().sum()}",
        f"{(df['order_value'] <= 0).sum()}",
        f"{df['date'].min().date()}",
        f"{df['date'].max().date()}",
        f"{df['order_value'].mean():,.2f}",
        f"{df['order_value'].median():,.2f}",
    ],
})
print("SANITY CHECK\n")
display(checks)

# Repeat-purchase behavior — the single most important number for BTYD models
orders_per_cust = df.groupby("customer")["order_id"].nunique()
repeat = (orders_per_cust > 1).sum()
one_time = (orders_per_cust == 1).sum()

rp = pd.DataFrame({
    "segment": ["One-time buyers", "Repeat buyers (2+ orders)", "Total customers"],
    "count": [one_time, repeat, len(orders_per_cust)],
    "pct": [round(one_time/len(orders_per_cust)*100, 1),
            round(repeat/len(orders_per_cust)*100, 1), 100.0],
})
print("\nREPEAT-PURCHASE BEHAVIOR (critical for BG/NBD)\n")
display(rp)
print(f"Max orders by a single customer: {orders_per_cust.max()}")

# %% [markdown]
# ## Exploratory Data Analysis & RFM Feature Engineering
# 
# 1. **Visualize** the distribution of purchase **frequency**, **recency**, and **customer spend**,
#    and highlight what they reveal about customer retention.
# 2. **Engineer RFM + T features** — transform the transaction log into a *customer-level* table
#    containing **Recency**, **Frequency**, **Monetary value**, and **T** (customer tenure / age
#    in the dataset). These four features are the foundation of the BG/NBD and Gamma-Gamma models.

# %%
# EDA: distribution of frequency, recency, and spend

# Customer-level aggregates for EDA (descriptive, pre-model)
cust = df.groupby("customer").agg(
    n_orders=("order_id", "nunique"),
    total_spend=("order_value", "sum"),
    first_purchase=("date", "min"),
    last_purchase=("date", "max"),
).reset_index()

study_end = df["date"].max()
cust["recency_days"] = (cust["last_purchase"] - cust["first_purchase"]).dt.days
cust["days_since_last"] = (study_end - cust["last_purchase"]).dt.days

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Frequency (orders per customer) — capped view for readability
sns.histplot(cust["n_orders"].clip(upper=6), bins=range(1, 8),
             ax=axes[0], color="steelblue", discrete=True)
axes[0].set_title("Purchase Frequency (orders per customer)")
axes[0].set_xlabel("Number of orders (capped at 6)")

# Recency (days since last purchase)
sns.histplot(cust["days_since_last"], bins=50, ax=axes[1], color="seagreen")
axes[1].set_title("Recency (days since last purchase)")
axes[1].set_xlabel("Days since last order")

# Monetary (total spend per customer) — log scale due to skew
sns.histplot(cust["total_spend"], bins=60, ax=axes[2], color="indianred")
axes[2].set_yscale("log")
axes[2].set_title("Customer Spend (total, log-count)")
axes[2].set_xlabel("Total spend (R$)")

plt.tight_layout(); plt.show()

# Retention insight: share of revenue from repeat vs one-time buyers
repeat_mask = cust["n_orders"] > 1
insight = pd.DataFrame({
    "segment": ["One-time buyers", "Repeat buyers"],
    "customers": [(~repeat_mask).sum(), repeat_mask.sum()],
    "revenue_R$": [cust.loc[~repeat_mask, "total_spend"].sum(),
                   cust.loc[repeat_mask, "total_spend"].sum()],
})
insight["% of customers"] = (insight["customers"]/insight["customers"].sum()*100).round(1)
insight["% of revenue"]  = (insight["revenue_R$"]/insight["revenue_R$"].sum()*100).round(1)
print("RETENTION INSIGHT — one-time vs repeat buyers\n")
display(insight)

# %% [markdown]
# ###RFM Feature Engineering & Calibration / Holdout Split
# 
# To train BTYD models we transform the transaction log into a **customer-level summary**
# with four features required by the BG/NBD model:
# 
# | Feature | Meaning | Why it matters |
# |---------|---------|----------------|
# | **Frequency** | Number of *repeat* purchases (= total orders − 1) | The model needs to see returning behavior |
# | **Recency** | Age of the customer at their last purchase (days from first to last order) | Tells the model *when* the last purchase happened relative to their lifetime |
# | **T** | Customer tenure: time from first purchase to end of observation | The "opportunity window" — how long they *could* have purchased |
# | **Monetary** | Average value per *repeat* transaction | Input to the Gamma-Gamma model |
# 
# ### Why we split time into Calibration and Holdout
# We cannot evaluate a predictive model on the same data we trained it on. The standard CLTV
# protocol is **time-based**, not random:
# 
# - **Calibration period** — earlier slice of time. Used to *fit* the model (learn each
#   customer's RFM profile).
# - **Holdout period** — later slice. Used to *check* whether the model's predictions for
#   this future window match what customers actually did.
# 
# ### Choosing the split point
# Our data spans 2016-09-04 - 2018-10-17 (around 25 months). The earliest months are sparse
# (Olist was just launching), so a naive midpoint would put too little signal in calibration.
# We use a ~18-month calibration / ~7-month holdout split, with the cut on 2018-03-01.
# This gives the model enough history to learn from and a meaningful future window to validate against.

# %%
# Define the time periods
observation_end = df["date"].max()           # 2018-10-17
calibration_end = pd.Timestamp("2018-03-01") # split point
print(f"Observation period : {df['date'].min().date()}  ->  {observation_end.date()}")
print(f"Calibration period : {df['date'].min().date()}  ->  {calibration_end.date()}")
print(f"Holdout period     : {calibration_end.date()}  ->  {observation_end.date()}")

# lifetimes provides a built-in helper that returns one row per customer with:
#   frequency_cal, recency_cal, T_cal, monetary_value_cal  (from calibration period)
#   frequency_holdout, monetary_value_holdout              (from holdout period)
#   duration_holdout                                       (length of holdout window)
summary_cal_holdout = calibration_and_holdout_data(
    transactions=df,
    customer_id_col="customer",
    datetime_col="date",
    monetary_value_col="order_value",
    calibration_period_end=calibration_end,
    observation_period_end=observation_end,
    freq="D",  # daily granularity
)

print(f"\nCustomers active in calibration: {len(summary_cal_holdout):,}")
display(summary_cal_holdout.head())

# Quick descriptive look at the calibration-period RFM features
print("\nCalibration-period feature summary:\n")
display(summary_cal_holdout[
    ["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal"]
].describe().round(2))

# %% [markdown]
# ##Modeling & Evaluation
# 
# ###Probabilistic Model: BG/NBD (Beta-Geometric / Negative Binomial Distribution)
# 
# 1. **While active**, a customer places purchases according to a Poisson process with rate λ.
# 2. **After every purchase**, the customer becomes permanently inactive ("dies") with probability p.
# 
# Both λ and p vary across the population — λ ~ Gamma(r, α), p ~ Beta(a, b). The model fits
# these four heterogeneity parameters via Maximum Likelihood from the RFM-T summary.
# 
# What we get out of a fitted BG/NBD model:
# - **Expected number of purchases** in any future window (e.g. the next 90 days).
# - **P(alive)** — the probability that a given customer is still active *right now*.
# 
# > **Why this model fits our setting.** Olist is a **non-contractual** business (no
# > subscriptions, no formal "cancel" event) with **continuous** purchase opportunities.
# > BG/NBD is designed exactly for this case.
# 
# ### Reality check before fitting
# Our calibration data shows ~97% of customers with `frequency_cal = 0` (one-time buyers).
# The BG/NBD model handles this — zeros carry real information ("had opportunity, did not return")
# — but its predictions will be **conservative**: most customers will be predicted to make
# near-zero future purchases, which is genuinely what the data says. The model's value lies in
# **separating the ~3% repeat customers into risk tiers** by their *P(alive)*.

# %%
# Fit the BG/NBD model on the calibration period

# Penalizer adds light L2 regularization — recommended when the data is sparse
# (few repeat purchases), which is exactly our case.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(
    frequency = summary_cal_holdout["frequency_cal"],
    recency   = summary_cal_holdout["recency_cal"],
    T         = summary_cal_holdout["T_cal"],
)

print("BG/NBD fitted parameters")
print("="*50)
print(bgf.summary.round(4))

# %%
#Goodness-of-fit diagnostics

# Does the model reproduce the observed distribution of repeat purchases in
# calibration? If actual vs predicted bars roughly align, the model fits well.

fig, ax = plt.subplots(figsize=(9, 4.5))
plot_period_transactions(bgf, max_frequency=6, ax=ax)
ax.set_title("BG/NBD goodness-of-fit: actual vs predicted "
             "frequency distribution (calibration)")
plt.tight_layout(); plt.show()

print("""
How to read this chart:
- Blue bars  = ACTUAL number of customers with that many repeat purchases
- Orange bars = PREDICTED by the BG/NBD model
- Close alignment across the full range = the model is faithful to the data.
- A common pattern in sparse data: tiny under/over-prediction at frequency 0 and 1,
  with very tight agreement on higher frequencies.
""")

# %%
# Per-customer predictions: P(alive) and expected 90-day purchases

HORIZON_DAYS = 90  # planning horizon for the Marketing team

# Predicted number of purchases in the next `HORIZON_DAYS`
summary_cal_holdout["pred_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t        = HORIZON_DAYS,
    frequency = summary_cal_holdout["frequency_cal"],
    recency   = summary_cal_holdout["recency_cal"],
    T         = summary_cal_holdout["T_cal"],
)

# Probability the customer is still "alive" at the end of the calibration period
summary_cal_holdout["p_alive"] = bgf.conditional_probability_alive(
    frequency = summary_cal_holdout["frequency_cal"],
    recency   = summary_cal_holdout["recency_cal"],
    T         = summary_cal_holdout["T_cal"],
)

print("Per-customer predictions — distribution\n")
display(summary_cal_holdout[["pred_purchases_90d", "p_alive"]].describe().round(3))

# Top customers by expected future purchases — these are the ones Marketing
# should care about the most.
print("\nTop 10 customers by predicted 90-day purchases\n")
top = summary_cal_holdout.sort_values("pred_purchases_90d", ascending=False).head(10)
display(top[["frequency_cal", "recency_cal", "T_cal",
             "pred_purchases_90d", "p_alive"]].round(3))

# %% [markdown]
# ### Probabilistic Model: Gamma-Gamma (Monetary Value)
# 
# The BG/NBD model predicts how many times a customer will buy — but not how much they spend.
# For that we use the Gamma-Gamma model (Fader & Hardie, The Gamma-Gamma Model of Monetary Value).
# It estimates the expected average transaction value for each customer, accounting for the fact
# that customers with few observed purchases have noisier average spend.
# 
# Combining BG/NBD (frequency) × Gamma-Gamma (monetary value) gives us full CLTV.
# 
# ### Critical assumption to verify FIRST
# The Gamma-Gamma model is only valid if there is no significant correlation between purchase
# frequency and monetary value. If frequent buyers systematically spent more (or less) per order,
# the model's independence assumption would be violated and its estimates unreliable.
# 
# We check this with the Pearson correlation between `frequency` and `monetary_value`, using only
# repeat customers (frequency ≥ 1), since the model is defined only for them. A correlation near
# 0 (roughly |r| < 0.1) means the assumption holds.

# %%
# Verify the Gamma-Gamma independence assumption

repeat_customers = summary_cal_holdout[
    (summary_cal_holdout["frequency_cal"] > 0) &
    (summary_cal_holdout["monetary_value_cal"] > 0)
].copy()

print(f"Repeat customers usable for Gamma-Gamma: {len(repeat_customers):,}")

# Pearson correlation between frequency and monetary value
corr = repeat_customers[["frequency_cal", "monetary_value_cal"]].corr().iloc[0, 1]

print("\nGAMMA-GAMMA INDEPENDENCE CHECK")
print("="*50)
print(f"Pearson correlation (frequency vs monetary): {corr:.4f}")
verdict = ("PASS — correlation is negligible, assumption holds."
           if abs(corr) < 0.1 else
           "CAUTION — non-trivial correlation; interpret monetary estimates carefully.")
print(verdict)

# Visual confirmation
plt.figure(figsize=(7, 5))
sns.scatterplot(data=repeat_customers, x="frequency_cal",
                y="monetary_value_cal", alpha=0.3, color="indianred")
plt.title(f"Frequency vs Monetary value (repeat customers)\nPearson r = {corr:.4f}")
plt.xlabel("Frequency (repeat purchases)")
plt.ylabel("Average monetary value per order (R$)")
plt.tight_layout(); plt.show()

# %%
# Fit the Gamma-Gamma model and estimate average order value

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(
    frequency      = repeat_customers["frequency_cal"],
    monetary_value = repeat_customers["monetary_value_cal"],
)

print("Gamma-Gamma fitted parameters")
print("="*50)
print(ggf.summary.round(4))

# Predicted average transaction value for the repeat customers
repeat_customers["exp_avg_value"] = ggf.conditional_expected_average_profit(
    frequency      = repeat_customers["frequency_cal"],
    monetary_value = repeat_customers["monetary_value_cal"],
)

print("\nValidation — model vs empirical average order value (repeat customers):")
print(f"  Model-predicted mean avg value : R$ {repeat_customers['exp_avg_value'].mean():,.2f}")
print(f"  Empirical mean avg value       : R$ {repeat_customers['monetary_value_cal'].mean():,.2f}")

print("\nDistribution of predicted average order value (R$):\n")
display(repeat_customers["exp_avg_value"].describe().round(2))

# %% [markdown]
# ### Predicted 90-Day CLTV (BG/NBD × Gamma-Gamma)
# 
# We now combine the two probabilistic models into a single monetary forecast:
# 
# **CLTV = (expected number of future purchases) × (expected value per purchase)**
# 
# The `lifetimes` library does this in one step via `customer_lifetime_value()`, which internally:
# 1. uses **BG/NBD** to project the expected transactions over the horizon,
# 2. uses **Gamma-Gamma** to assign an expected monetary value to those transactions,
# 3. optionally discounts future cash flows to present value.
# 
# ### Scope of the CLTV estimate
# The Gamma-Gamma model is defined only for the **987 repeat customers** (frequency ≥ 1). These
# are the customers who carry genuine future repeat value. For the ~97% one-time buyers, the
# predicted future repeat-purchase rate is effectively zero, so their forward-looking repeat CLTV
# is ≈ 0 — this is not a data gap, it is what the behavior implies. We therefore compute precise
# CLTV for the repeat segment and treat the rest as the low-value, hard-to-reactivate majority.
# 
# > **Horizon:** 90 days. **Discount:** a small monthly discount rate makes future revenue
# > comparable to present value (standard in financial planning).

# %%
# Compute 90-day predicted CLTV for repeat customers

# Monthly discount rate (~12.7% annual -> 1% monthly is a common default).
MONTHLY_DISCOUNT = 0.01
HORIZON_MONTHS = 3   # 90 days ≈ 3 months

repeat_customers["cltv_90d"] = ggf.customer_lifetime_value(
    transaction_prediction_model = bgf,                       # BG/NBD for frequency
    frequency      = repeat_customers["frequency_cal"],
    recency        = repeat_customers["recency_cal"],
    T              = repeat_customers["T_cal"],
    monetary_value = repeat_customers["monetary_value_cal"],
    time           = HORIZON_MONTHS,        # months
    freq           = "D",                   # data is in days
    discount_rate  = MONTHLY_DISCOUNT,
)

print("90-DAY PREDICTED CLTV — repeat customers (R$)")
print("="*50)
display(repeat_customers["cltv_90d"].describe().round(2))

print(f"\nTotal predicted 90-day value of the repeat base: "
      f"R$ {repeat_customers['cltv_90d'].sum():,.2f}")

# Top customers by predicted CLTV — the priority list for Marketing
print("\nTop 10 customers by predicted 90-day CLTV\n")
top_cltv = repeat_customers.sort_values("cltv_90d", ascending=False).head(10)
display(top_cltv[["frequency_cal", "recency_cal", "T_cal",
                  "p_alive", "exp_avg_value", "cltv_90d"]].round(2))

# %%
# Visualize CLTV distribution & segment the repeat base

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Distribution of predicted CLTV
sns.histplot(repeat_customers["cltv_90d"], bins=50, ax=axes[0], color="purple")
axes[0].set_title("Distribution of 90-day predicted CLTV (repeat customers)")
axes[0].set_xlabel("Predicted CLTV (R$)")

# CLTV vs P(alive): the strategic 2x2 view
sc = axes[1].scatter(repeat_customers["p_alive"],
                     repeat_customers["cltv_90d"],
                     c=repeat_customers["cltv_90d"], cmap="viridis", alpha=0.6)
axes[1].set_title("CLTV vs P(alive) — the action map")
axes[1].set_xlabel("P(alive)")
axes[1].set_ylabel("Predicted 90-day CLTV (R$)")
plt.colorbar(sc, ax=axes[1], label="CLTV (R$)")
plt.tight_layout(); plt.show()

# Concentration of value: what share of CLTV comes from the top 10% / 20%?
sorted_cltv = repeat_customers["cltv_90d"].sort_values(ascending=False)
total = sorted_cltv.sum()
for pct in [0.10, 0.20, 0.50]:
    n = int(len(sorted_cltv) * pct)
    share = sorted_cltv.head(n).sum() / total * 100
    print(f"Top {int(pct*100):>2}% of repeat customers ({n:>3}) generate "
          f"{share:5.1f}% of predicted repeat CLTV")

# %% [markdown]
# ### Machine Learning Approach (Regression)
# 
# The BTYD models are *probabilistic* — they assume a specific buying process. The ML approach is
# *data-driven* and assumption-free: we treat CLTV prediction as a **supervised regression problem**.
# 
# **Setup (proper time-based validation):**
# - **Features (X)** — each customer's behavior measured in the **calibration period only**:
#   frequency, recency, T, monetary value (the same RFM-T inputs the BTYD models used).
# - **Target (y)** — the customer's **actual total revenue in the holdout period**
#   (`frequency_holdout × monetary_value_holdout`).
# 
# The model learns the mapping *calibration behavior → future revenue* on a training split, then
# predicts on an unseen test split. Because features come strictly *before* the target window, there
# is **no data leakage** — this mirrors how the model would be used in production.
# 
# We train two regressors and keep the better one for the final comparison: **Random Forest** and
# **XGBoost** (both handle non-linear relationships and skewed targets well).

# %%
#Build features (X) and target (y) for the ML model

ml = summary_cal_holdout.copy()

# Target: actual revenue generated by each customer in the HOLDOUT period
ml["holdout_revenue"] = ml["frequency_holdout"] * ml["monetary_value_holdout"]

# Features: behavior observed in the CALIBRATION period only
feature_cols = ["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal"]
X = ml[feature_cols]
y = ml["holdout_revenue"]

print("Feature matrix and target")
print("="*50)
print(f"X shape: {X.shape}   |   y shape: {y.shape}")
print(f"\nCustomers with non-zero holdout revenue: "
      f"{(y > 0).sum():,} ({(y > 0).mean()*100:.2f}%)")
print(f"Mean holdout revenue (all):      R$ {y.mean():,.2f}")
print(f"Mean holdout revenue (buyers):   R$ {y[y > 0].mean():,.2f}")

# Train / test split (time-based features, so a random split of customers is fine here)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape[0]:,} customers   |   Test: {X_test.shape[0]:,} customers")

# %%
#Train Random Forest and XGBoost regressors

# --- Random Forest -----------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# --- XGBoost -----------------------------------------------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# --- Evaluate both -----------------------------------------------------------
def eval_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse}

ml_results = pd.DataFrame([
    eval_model("Random Forest", y_test, rf_pred),
    eval_model("XGBoost",       y_test, xgb_pred),
])
print("ML MODEL PERFORMANCE (holdout revenue prediction)")
print("="*50)
display(ml_results.round(3))

# Feature importance from the (usually stronger) XGBoost model
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nFeature importance (XGBoost):\n")
display(importance.round(3))

# %% [markdown]
# ### Model Evaluation & Comparison: BTYD vs Machine Learning
# 
# We now put both approaches on a **level playing field** and judge them by the same standard:
# how well do they predict each customer's **actual revenue during the holdout period**?
# 
# **Fair-comparison protocol:**
# - Both models predict **holdout-period revenue** (≈181 days), evaluated against the *same*
#   ground truth (`frequency_holdout × monetary_value_holdout`).
# - We evaluate on the **same test customers** used for the ML model, so neither approach gets
#   an easier subset.
# - **BTYD prediction** = BG/NBD expected purchases over the holdout window × Gamma-Gamma expected
#   value per purchase.
# - Metrics: **MAE, MSE, RMSE** (lower is better) — exactly as required by the assignment.
# 
# **What each metric tells us:**
# - **MAE** — average error in R$. Dominated by the many zero-revenue customers, so it rewards
#   "predict near-zero for everyone."
# - **RMSE** — penalizes large misses heavily. This is where the models are truly tested:
#   can they catch the few high-value customers?

# %%

# Length of the holdout window (in days) — same for every customer
holdout_days = int(summary_cal_holdout["duration_holdout"].iloc[0])
print(f"Holdout window length: {holdout_days} days")

btyd = summary_cal_holdout.copy()

# 1) Expected number of purchases over the holdout window (BG/NBD)
btyd["btyd_exp_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t         = holdout_days,
    frequency = btyd["frequency_cal"],
    recency   = btyd["recency_cal"],
    T         = btyd["T_cal"],
)

# 2) Expected value per purchase (Gamma-Gamma).
#    GG is only defined for repeat customers; for the rest we fall back to the
#    population mean order value (a reasonable, transparent default).
pop_mean_value = repeat_customers["monetary_value_cal"].mean()

mask_repeat = (btyd["frequency_cal"] > 0) & (btyd["monetary_value_cal"] > 0)
btyd["btyd_exp_value"] = pop_mean_value  # default
btyd.loc[mask_repeat, "btyd_exp_value"] = ggf.conditional_expected_average_profit(
    frequency      = btyd.loc[mask_repeat, "frequency_cal"],
    monetary_value = btyd.loc[mask_repeat, "monetary_value_cal"],
)

# 3) BTYD predicted holdout revenue = expected purchases × expected value
btyd["btyd_pred_revenue"] = btyd["btyd_exp_purchases"] * btyd["btyd_exp_value"]

print("BTYD predicted holdout revenue — summary (R$):")
display(btyd["btyd_pred_revenue"].describe().round(3))

# %%
# Head-to-head comparison on the SAME test customers

# Align BTYD predictions to the exact test set used for the ML model
test_idx = X_test.index
y_true_test = ml.loc[test_idx, "holdout_revenue"]

btyd_pred_test = btyd.loc[test_idx, "btyd_pred_revenue"]
rf_pred_test   = rf_pred      # already predicted on X_test
xgb_pred_test  = xgb_pred

comparison = pd.DataFrame([
    eval_model("BTYD (BG/NBD × Gamma-Gamma)", y_true_test, btyd_pred_test),
    eval_model("Random Forest",               y_true_test, rf_pred_test),
    eval_model("XGBoost",                      y_true_test, xgb_pred_test),
]).sort_values("RMSE")

print("FINAL MODEL COMPARISON — holdout revenue (same test set)")
print("="*60)
display(comparison.round(3))

# Visual comparison of the three models across metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric in zip(axes, ["MAE", "RMSE", "MSE"]):
    sns.barplot(data=comparison, x="Model", y=metric, ax=ax,
                hue="Model", palette="mako", legend=False)
    ax.set_title(metric); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.show()

# %%
# Beyond averages: who catches the high-value customers?

# Aggregate-level view: does each model recover the TOTAL holdout revenue,
# and how well does it rank customers? (Ranking matters for targeting.)

print("AGGREGATE HOLDOUT REVENUE (test set)")
print("="*55)
print(f"  Actual total:        R$ {y_true_test.sum():,.2f}")
print(f"  BTYD predicted:      R$ {btyd_pred_test.sum():,.2f}")
print(f"  Random Forest:       R$ {rf_pred_test.sum():,.2f}")
print(f"  XGBoost:             R$ {xgb_pred_test.sum():,.2f}")

# Ranking quality: take each model's top-50 predicted customers and measure
# how much ACTUAL holdout revenue those customers really generated.
def captured_revenue(pred, k=50):
    top_customers = pd.Series(np.asarray(pred), index=test_idx).nlargest(k).index
    return y_true_test.loc[top_customers].sum()

actual_top50 = y_true_test.nlargest(50).sum()  # theoretical best
print(f"\nACTUAL revenue captured by each model's TOP-50 predicted customers")
print("="*55)
print(f"  Perfect targeting (ceiling): R$ {actual_top50:,.2f}")
print(f"  BTYD top-50:                 R$ {captured_revenue(btyd_pred_test):,.2f}")
print(f"  Random Forest top-50:        R$ {captured_revenue(rf_pred_test):,.2f}")
print(f"  XGBoost top-50:              R$ {captured_revenue(xgb_pred_test):,.2f}")

# %% [markdown]
# ### C.4 — Conclusions: Which Model, and When?
# 
# | Criterion | Winner | Evidence |
# |-----------|--------|----------|
# | **Point accuracy** (MAE / RMSE) | Random Forest (marginal) | RF MAE 3.73 / RMSE 27.43 vs BTYD 3.99 / 28.76 — differences are small |
# | **Aggregate revenue accuracy** | **BTYD** | Actual R$ 27,375 → BTYD R$ 30,167 (+10%); ML models **under-predict** (RF −6%, XGB −4%) |
# | **Customer ranking / targeting** (top-50 captured) | **BTYD** | BTYD R$ 1,615 vs RF R$ 1,199 vs XGB R$ 1,457 |
# | **Interpretability & extra outputs** | **BTYD** | Provides P(alive) and a transparent buying-process story; ML is a black box |
# 
# **Key insight — the metrics disagree on purpose.** In data where ~97% of customers generate zero
# future revenue, "predict near-zero for everyone" wins on MAE — which is why Random Forest edges
# ahead there. But that same conservatism makes ML **systematically under-forecast the total**,
# which is dangerous for budget planning (you'd allocate too little). BTYD, as a generative model of
# the purchase process, recovers the aggregate value far better and ranks high-value customers
# better — *despite* a slightly worse RMSE.
# 
# **Recommendation:**
# - **For long-term budget forecasting (macro):** use **BTYD**. It estimates total customer-base
#   value more faithfully and comes with interpretable parameters and P(alive).
# - **For point-level scoring (micro):** ML is competitive and easy to extend with extra features,
#   but here it offers no decisive edge and lacks P(alive).
# 
# **Honest limitation.** All models capture only ~9% of the theoretical "perfect-targeting" revenue.
# This is a property of the **data**, not a modeling failure: individual future purchases in a
# non-contractual marketplace with 97% one-time buyers are inherently hard to predict. The models'
# real value is **prioritization** — concentrating limited marketing budget on the small,
# identifiable high-value segment.
