"""
data_prep.py
Load and join all raw CSV files into one clean master table.
Called by: 02_notebooks/01_eda.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "01_input" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "01_input" / "processed"


def load_raw_files():
    """Load all raw CSVs into a dictionary of DataFrames."""
    files = {
        "sales":        "fact_sales_monthly.csv",
        "payer":        "fact_payer_access_monthly.csv",
        "promotion":    "fact_promotion_monthly.csv",
        "price":        "fact_price_monthly.csv",
        "epidemiology": "fact_epidemiology_yearly.csv",
        "forecast":     "fact_internal_forecast.csv",
        "geo":          "dim_geography.csv",
        "product":      "dim_product.csv",
        "indication":   "dim_indication.csv",
        "prod_ind":     "dim_product_indication.csv",
        "payer_bob":    "dim_payer_bob.csv",
        "test":         "test_features.csv",
        "submission":   "sample_submission.csv",
    }
    data = {}
    for key, fname in files.items():
        data[key] = pd.read_csv(RAW_DIR / fname)
        print(f"Loaded {key}: {data[key].shape[0]:,} rows")
    return data


def fix_data_quality(sales_df):
    """Fix all known data quality issues in the sales table."""
    df = sales_df.copy()

    # Fix 1: Standardize data_provider_name casing
    df["data_provider_name"] = df["data_provider_name"].str.upper().str.strip()

    # Fix 2: Remove duplicate rows (keep first)
    before = len(df)
    df = df.drop_duplicates(
        subset=["date_year_month", "ecosystem_id", "product_brand_id",
                "data_provider_name", "sale_channel_type"]
    )
    print(f"Removed {before - len(df)} duplicate rows")

    # Fix 3: Flag negative dollar rows (don't remove — qty_eqv may be valid)
    df["flag_negative_dollars"] = (df["iqvia_sales_dollars"] < 0).astype(int)
    print(f"Flagged {df['flag_negative_dollars'].sum()} negative dollar rows")

    # Fix 4: Flag last 2 months of DDD data as under-reported (~15% lag)
    ddd_months = df[df["data_provider_name"] == "DDD"]["date_year_month"].max()
    lag_months = sorted(df[df["data_provider_name"] == "DDD"]["date_year_month"].unique())[-2:]
    df["flag_ddd_lag"] = (
        (df["data_provider_name"] == "DDD") &
        (df["date_year_month"].isin(lag_months))
    ).astype(int)
    print(f"Flagged {df['flag_ddd_lag'].sum()} DDD lag rows (months: {lag_months})")

    return df


def build_master_table(data):
    """
    Join all tables into one flat master training table.
    Grain: date_year_month x ecosystem_id x product_brand_id
    """
    # Start with sales (training data)
    master = data["sales"].copy()

    # Attach price (month x brand)
    master = master.merge(
        data["price"][["date_year_month", "product_brand_id",
                        "list_wac_per_unit", "gross_to_net_ratio",
                        "effective_net_price_per_unit"]],
        on=["date_year_month", "product_brand_id"],
        how="left"
    )

    # Attach payer access (month x ecosystem x brand)
    master = master.merge(
        data["payer"],
        on=["date_year_month", "ecosystem_id", "product_brand_id"],
        how="left",
        suffixes=("", "_payer")
    )

    # Attach promotion — GNE brands only (month x ecosystem x brand)
    master = master.merge(
        data["promotion"],
        on=["date_year_month", "ecosystem_id", "product_brand_id"],
        how="left"
    )

    # Attach geography static info
    geo_cols = ["ecosystem_id", "region", "total_population", "pct_65_plus",
                "pct_rural", "median_household_income", "commercial_lives_share"]
    master = master.merge(data["geo"][geo_cols], on="ecosystem_id", how="left")

    # Attach product static info
    prod_cols = ["product_brand_id", "manufacturer", "product_benefit_type",
                 "flag_generic", "launch_date", "list_wac_per_unit"]
    master = master.merge(
        data["product"][prod_cols].rename(columns={"list_wac_per_unit": "wac_list_static"}),
        on="product_brand_id",
        how="left"
    )

    # Attach epidemiology (yearly — forward fill to monthly)
    master["date_year"] = master["date_year_month"] // 100
    epi_agg = data["epidemiology"].groupby(
        ["ecosystem_id", "year"]
    )[["treated_patient_volume", "prevalence_rate_per_100k"]].sum().reset_index()
    epi_agg.rename(columns={"year": "date_year"}, inplace=True)
    master = master.merge(epi_agg, on=["ecosystem_id", "date_year"], how="left")

    # Clean up duplicate columns created by merges
    master = master.rename(columns={"product_brand_name_x": "product_brand_name",
                                     "commercial_lives_share_x": "commercial_lives_share"})
    drop_cols = [c for c in master.columns if c.endswith(("_y", "_payer"))
                 and c not in ["date_year_month"]]
    master = master.drop(columns=drop_cols, errors="ignore")

    print(f"Master table shape: {master.shape}")
    return master


def build_test_table(data, master):
    """
    Build the test table (Jan-Jun 2025) with all features except sales target.
    Same structure as master but no iqvia_sales_qty_eqv.
    """
    test = data["test"].copy()

    # Attach all the same features (payer, promo, price extend through horizon)
    test = test.merge(
        data["price"][["date_year_month", "product_brand_id",
                        "list_wac_per_unit", "gross_to_net_ratio",
                        "effective_net_price_per_unit"]],
        on=["date_year_month", "product_brand_id"], how="left"
    )
    test = test.merge(data["payer"], on=["date_year_month", "ecosystem_id", "product_brand_id"], how="left")
    test = test.merge(data["promotion"], on=["date_year_month", "ecosystem_id", "product_brand_id"], how="left")

    geo_cols = ["ecosystem_id", "region", "total_population", "pct_65_plus",
                "pct_rural", "median_household_income", "commercial_lives_share"]
    test = test.merge(data["geo"][geo_cols], on="ecosystem_id", how="left")

    prod_cols = ["product_brand_id", "manufacturer", "product_benefit_type",
                 "flag_generic", "launch_date"]
    test = test.merge(data["product"][prod_cols], on="product_brand_id", how="left")

    test["date_year"] = test["date_year_month"] // 100
    epi_agg = data["epidemiology"].groupby(
        ["ecosystem_id", "year"]
    )[["treated_patient_volume", "prevalence_rate_per_100k"]].sum().reset_index()
    epi_agg.rename(columns={"year": "date_year"}, inplace=True)
    test = test.merge(epi_agg, on=["ecosystem_id", "date_year"], how="left")

    print(f"Test table shape: {test.shape}")
    return test


def save_processed(master, test):
    PROCESSED_DIR.mkdir(exist_ok=True)
    master.to_csv(PROCESSED_DIR / "master_train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "master_test.csv", index=False)
    print(f"Saved master_train.csv and master_test.csv to {PROCESSED_DIR}")


if __name__ == "__main__":
    data = load_raw_files()
    data["sales"] = fix_data_quality(data["sales"])
    master = build_master_table(data)
    test = build_test_table(data, master)
    save_processed(master, test)
