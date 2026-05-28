# Customer Lifetime Value (CLTV) Prediction — Olist E-Commerce

Predicting customer lifetime value for the Olist Brazilian e-commerce marketplace,
combining **probabilistic (Buy-Till-You-Die)** models with a **machine-learning regression**
approach, and translating the results into actionable marketing strategy.

---

## Project Overview

Olist's marketing team needs to forecast the future value of its customer base to plan budget
allocation and loyalty campaigns. This project builds an end-to-end CLTV framework:

1. **Data audit & cleaning** of 9 relational Olist tables (~100k orders, 2016–2018).
2. **RFM-T feature engineering** with a time-based **calibration / holdout** split.
3. **Probabilistic modeling** — BG/NBD (purchase frequency + P(alive)) and Gamma-Gamma
   (monetary value) → 90-day CLTV.
4. **Machine-learning modeling** — Random Forest & XGBoost regression on holdout revenue.
5. **Model comparison** (MAE / MSE / RMSE) and **marketing recommendations**.

## Key Findings

- ~97% of Olist customers are **one-time buyers**; the ~3% repeat segment drives nearly all
  predictable future value.
- Value is highly concentrated: the **top 10% of repeat customers generate ~62%** of predicted
  repeat CLTV.
- **BTYD** estimates aggregate base value more faithfully (better for budget forecasting),
  while **ML** is marginally better on point-level error metrics.
- Output feeds a **CLTV × P(alive) action map** that prioritizes customers for loyalty vs.
  win-back campaigns.

## Tech Stack

`Python` · `pandas` · `lifetimes` (BG/NBD + Gamma-Gamma) · `scikit-learn` · `XGBoost` ·
`matplotlib` / `seaborn`

## Repository Structure

\`\`\`
notebooks/      Jupyter notebook with the full analysis
presentation/   2-slide executive summary (PDF)
images/         key visualizations
\`\`\`

## Reproducing the Analysis

The Olist dataset is downloaded at runtime via the Kaggle API (not committed to the repo).

1. Install dependencies: `pip install -r requirements.txt`
2. Open `notebooks/cltv_analysis.ipynb` (designed for Google Colab).
3. Provide a Kaggle API token when prompted — the notebook downloads the dataset automatically.

## Methodology Notes

- **Order status filtering:** only `delivered` orders count toward CLTV (completed revenue).
- **Customer identity:** aggregation uses `customer_unique_id`, not the per-order `customer_id`.
- **Outliers:** order values capped at the 99th percentile (winsorization) to stabilize the
  Gamma-Gamma model without dropping high-value customers.

## Limitations

Individual future purchases in a non-contractual marketplace with 97% one-time buyers are
inherently hard to predict; all models capture only a fraction of theoretical "perfect targeting."
The framework's real value is **prioritization**, not perfect per-customer prediction.
