"""
utils.py
Small helper functions used across multiple notebooks.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR       = ROOT / "01_input" / "raw"
PROCESSED_DIR = ROOT / "01_input" / "processed"
OUTPUT_TFT    = ROOT / "04_outputs" / "tft"
OUTPUT_TIDE   = ROOT / "04_outputs" / "tide"
OUTPUT_FINAL  = ROOT / "04_outputs" / "final"


def yyyymm_to_date(yyyymm_series):
    """Convert integer YYYYMM column to pandas datetime."""
    return pd.to_datetime(yyyymm_series.astype(str), format="%Y%m")


def make_submission(row_ids, predictions, filepath):
    """
    Save predictions in the required submission format.
    Validates: no negatives, no NaNs, correct row count.
    """
    sub = pd.DataFrame({
        "row_id": row_ids,
        "forecast_units_eqv": predictions
    })
    assert sub["forecast_units_eqv"].isna().sum() == 0, "ERROR: NaN values in predictions"
    assert (sub["forecast_units_eqv"] >= 0).all(),      "ERROR: Negative values in predictions"
    assert len(sub) == 3840,                             f"ERROR: Expected 3840 rows, got {len(sub)}"
    sub.to_csv(filepath, index=False)
    print(f"Submission saved: {filepath}  ({len(sub)} rows, min={sub['forecast_units_eqv'].min():.2f})")
    return sub


def print_data_summary(df, name="DataFrame"):
    """Quick summary of a DataFrame: shape, nulls, dtypes."""
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if len(null_cols):
        print(f"Columns with nulls:\n{null_cols.to_string()}")
    else:
        print("No null values found.")
