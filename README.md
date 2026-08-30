# TAP Into DS Hackathon 2026
## Ecosystem-Level Demand Forecasting and Market Share Intelligence

**Team:** Shirley Edward  
**Organization:** Genentech / Roche Commercial Analytics  
**Hackathon:** Signal and Share Challenge 2026  

**Live AI Dashboard:** https://tap-into-ds-hackathon-2026-dashboard.streamlit.app/

---

## Project Summary

This project delivers two forecasting tasks for 8 GNE pharmaceutical brands across 80 sales ecosystems for January to June 2025.

- **Task A:** Monthly demand forecast in equivalent units per brand per zone (3,840 rows)
- **Task B:** Implied market share per brand within its therapeutic area per zone (3,840 rows)

The solution uses a hybrid ensemble of TiDE (Time-series Dense Encoder, Google Research 2023) and LightGBM, achieving a portfolio WAPE of 1.85% versus the TM1 internal planning baseline of 14.16% — an 87% reduction in forecast error.

---

## Repository Structure

```
tap-into-ds-hackathon-2026-shirleyedward/
|
|- 01_input/
|  |- raw/                         Raw data files from the hackathon committee
|  |  |- fact_sales_monthly.csv    Historical sales by brand, zone, month
|  |  |- test_features.csv         Feature rows for the 2025 forecast horizon
|  |  |- fact_internal_forecast.csv   TM1 baseline forecast (benchmark)
|  |  |- fact_payer_access_monthly.csv
|  |  |- fact_promotion_monthly.csv
|  |  |- fact_price_monthly.csv
|  |  |- fact_epidemiology_yearly.csv
|  |  |- sample_submission.csv     Submission format template
|  |
|  |- processed/
|     |- master_train_v7.csv       Final training dataset with all engineered features
|     |- master_test_v7.csv        Final test dataset ready for inference
|
|- 02_notebooks/                   Jupyter notebooks covering the full modelling journey
|  |- 01_exploratory_data_analysis.ipynb       EDA: sales trends, seasonality, payer patterns
|  |- 02_data_preparation_imputation.ipynb     Cleaning, imputation, train/test split
|  |- 03_feature_engineering_analysis.ipynb   Feature construction and incremental value testing
|  |- 04_model_tide_v5_final_base_model.ipynb  TiDE architecture, training and validation
|  |- 05_model_comparison_all_architectures.ipynb  TFT vs TiDE vs LightGBM vs Prophet
|  |- 07_market_share_analysis.ipynb           Competitor volume estimation and share methodology
|  |- 09_final_hybrid_submission.ipynb         Ensemble blending and submission assembly
|  |- 10_validation_backtest_full_year_2024.ipynb  Full 2024 hold-out validation and leakage audit
|  |- 11_genai_taskc.ipynb                     GenAI narrative generation (Task C)
|
|- 03_scripts/                     Python scripts for the modelling pipeline
|  |- data_prep.py                 Data loading, joining, and cleaning
|  |- utils.py                     Shared helper functions
|  |- evaluate.py                  WAPE, RMSE, sMAPE, Bias metric utilities
|  |- run_features_v7.py           Feature engineering pipeline (final version)
|  |- run_tide_v5.py               TiDE model training for stable brands
|  |- run_lgbm_brand.py            LightGBM per-brand training
|  |- run_lgbm_v2.py               LightGBM v2 with tuned hyperparameters
|  |- run_smart_hybrid.py          Ensemble blending logic
|  |- run_final_model.py           Production model build
|  |- run_market_share_v3.py       Market share and competitor forecast calculation
|  |- run_leakage_check.py         17-point data leakage validation
|  |- run_prophet_ets.py           Prophet and ETS models for erratic brands
|  |- run_backtest_h1_2024.py      Full-year 2024 hold-out backtest
|  |- run_pipeline.py              Main wrapper script (run this to reproduce outputs)
|
|- 04_outputs/                     Final submission files
|  |- submission.csv               Task A: demand forecast (3,840 rows)
|  |- share_submission.csv         Task B: market share forecast (3,840 rows)
|
|- 05_documents/
|  |- Signal_and_Share_Challenge_Brief.docx        Original hackathon brief
|  |- Forecasting_Intelligence_AI_Agent_Prompts.docx  Chatbot prompt guide
|
|- dashboard.py                    Streamlit AI forecasting dashboard (Task C)
|- requirements.txt                Python package dependencies
|- README.md                       This file
```

---

## Submission Files

### Task A: `04_outputs/submission.csv`

| Column | Description |
|--------|-------------|
| row_id | Unique identifier matching the test set |
| date_year_month | Forecast month (YYYYMM format) |
| ecosystem_id | Sales zone identifier |
| product_brand_id | Brand identifier |
| product_brand_name | Brand name |
| forecast_units_eqv | Forecast demand in equivalent units |

Total: 3,840 rows (8 brands x 80 zones x 6 months)

### Task B: `04_outputs/share_submission.csv`

| Column | Description |
|--------|-------------|
| ecosystem_id | Sales zone identifier |
| product_brand_id | Brand identifier |
| product_brand_name | Brand name |
| date_year_month | Forecast month (YYYYMM format) |
| forecast_share | Predicted market share (0 to 1) |

Total: 3,840 rows (8 brands x 80 zones x 6 months)

Market share formula: GNE brand forecast / (GNE portfolio forecast + competitor forecast) per zone per month.

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Reproduce submission files

```bash
python 03_scripts/run_pipeline.py
```

This trains LightGBM for the 5 erratic brands, loads pre-saved TiDE predictions for the 3 stable brands, assembles both submission files, and prints the full WAPE vs TM1 comparison table.

---

## WAPE vs TM1 Baseline Results

Validated on H2 2024 hold-out (July to December 2024, 6 months unseen during training).

| Brand | Therapeutic Area | Model | Our WAPE | TM1 WAPE | Improvement |
|-------|-----------------|-------|----------|----------|-------------|
| Xolarin | RESP | TiDE v5 | 0.66% | 13.90% | +13.24pp |
| Hemvia | HEM | TiDE v5 | 0.80% | 14.21% | +13.41pp |
| Ocretiva | MS | TiDE v5 | 0.93% | 16.40% | +15.47pp |
| Perjenta | ONC | LightGBM | 1.94% | 12.04% | +10.10pp |
| Kadcynex | ONC | LightGBM | 2.29% | 10.90% | +8.61pp |
| Retivue | OPH | LightGBM | 2.32% | 10.93% | +8.61pp |
| Phesgrox | ONC | LightGBM | 3.22% | 11.67% | +8.45pp |
| Vabyseal | OPH | LightGBM | 4.76% | 21.12% | +16.36pp |
| **Portfolio** | **ALL** | **Ensemble** | **1.85%** | **14.16%** | **+12.31pp** |

Portfolio forecast error reduced by 87% versus TM1 baseline.

---

## Model Architecture

Brands were split into two groups based on demand pattern:

**Stable brands (Hemvia, Xolarin, Ocretiva):** TiDE v5 (Time-series Dense Encoder, Google Research, arXiv:2304.08424). These brands have high volume and consistent seasonal patterns that TiDE captures well using its dense encoder architecture and RevIN normalization.

**Erratic brands (Perjenta, Phesgrox, Kadcynex, Retivue, Vabyseal):** LightGBM with per-brand hyperparameter tuning. Lower-volume brands with launch curves and competitive disruptions respond better to gradient boosting on explicit lag and feature interactions.

---

## Feature Engineering

| Group | Features |
|-------|---------|
| Lag features | lag_1, lag_2, lag_3, lag_6, lag_12 |
| Rolling averages | roll_mean_3, roll_mean_6 |
| Payer access | pct_lives_covered, pct_preferred, pct_prior_auth_required, access_burden |
| Promotion | rep_calls_adstock, digital_adstock, marketing_spend_usd, copay_redemptions |
| Price | effective_net_price_per_unit |
| Seasonality | fourier_sin_1, fourier_cos_1, fourier_sin_2, fourier_cos_2, brand_seasonal_index, is_h2 |
| Trend | sales_momentum, yoy_growth, month_of_year |

All lag and rolling features use only past observations. No future data enters any feature window.

---

## AI Forecasting Dashboard (Task C)

**Live:** https://tap-into-ds-hackathon-2026-dashboard.streamlit.app/

The dashboard is an AI-powered forecasting intelligence tool built with Streamlit. It provides:

- Market share and volume analysis by brand, zone, and therapeutic area
- Supply planning with RMSE-based buffer stock recommendations
- Territory prioritisation and zone risk scoring
- Forecast accuracy breakdown: WAPE, RMSE, sMAPE, Bias vs TM1 baseline
- Role-aware answers tailored to Territory Account Manager, Brand Manager, Data Scientist, and Data Analyst
- Hybrid chatbot: rule-based data engine + curated knowledge base + Groq LLM (Meta Llama 3.3-70B) fallback

To run the dashboard locally:

```bash
streamlit run dashboard.py
```

Then open http://localhost:8501 in your browser.

---

## Validation Approach

- Training period: January 2021 to June 2024 (42 months)
- Validation period: July to December 2024 (6 months, fully held out)
- Forecast horizon: January to June 2025 (submitted to committee)
- 17 data leakage checks passed (all rolling windows computed on lagged values only)
- No zone above 8% WAPE in the validation period
- Forecast error stable across all 6 forecast months with no step degradation
