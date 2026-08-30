"""
evaluate.py
Scoring functions used to compare TFT vs TiDE vs TM1 baseline.
"""

import pandas as pd
import numpy as np


def wape(y_true, y_pred):
    """
    WAPE: Weighted Absolute Percentage Error (primary hackathon metric).
    Lower is better. Range: 0 (perfect) to 1+ (bad).
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true) + 1e-8)


def macro_wape(y_true, y_pred, group_ids):
    """
    MACRO-WAPE: Average WAPE across all individual series.
    Penalizes models that are good at large brands but bad at small ones.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group_ids})
    per_group = df.groupby("group").apply(
        lambda g: wape(g["y_true"].values, g["y_pred"].values)
    )
    return per_group.mean()


def smape(y_true, y_pred):
    """Symmetric MAPE — less sensitive to near-zero actuals."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return np.mean(np.abs(y_true - y_pred) / denom)


def bias(y_true, y_pred):
    """
    Forecast bias: positive = over-predicting, negative = under-predicting.
    """
    return np.mean(y_pred - y_true)


def full_scorecard(y_true, y_pred, brand_ids, eco_ids, horizon_steps):
    """
    Run all metrics and return a summary DataFrame.
    Used to build the TFT vs TiDE comparison table.
    """
    results = {
        "WAPE (overall)":       wape(y_true, y_pred),
        "MACRO-WAPE (by brand)": macro_wape(y_true, y_pred, brand_ids),
        "sMAPE":                 smape(y_true, y_pred),
        "Bias (avg)":            bias(y_true, y_pred),
    }

    # WAPE per brand
    df = pd.DataFrame({
        "y_true": y_true, "y_pred": y_pred,
        "brand": brand_ids, "eco": eco_ids, "step": horizon_steps
    })
    brand_wapes = df.groupby("brand").apply(
        lambda g: wape(g["y_true"].values, g["y_pred"].values)
    ).rename("WAPE")
    step_wapes = df.groupby("step").apply(
        lambda g: wape(g["y_true"].values, g["y_pred"].values)
    ).rename("WAPE")

    print("=== Overall Scores ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== WAPE by Brand ===")
    print(brand_wapes.to_string())

    print("\n=== WAPE by Horizon Step (month 1..6) ===")
    print(step_wapes.to_string())

    return results, brand_wapes, step_wapes


def compare_models(tft_pred, tide_pred, tm1_pred, y_true, brand_ids):
    """
    Print a side-by-side comparison table of all three models.
    """
    rows = []
    for name, pred in [("TM1 Baseline", tm1_pred), ("TFT", tft_pred), ("TiDE", tide_pred)]:
        rows.append({
            "Model": name,
            "WAPE": round(wape(y_true, pred), 4),
            "MACRO-WAPE": round(macro_wape(y_true, pred, brand_ids), 4),
            "sMAPE": round(smape(y_true, pred), 4),
            "Bias": round(bias(y_true, pred), 2),
        })
    comparison = pd.DataFrame(rows).set_index("Model")
    print(comparison.to_string())
    return comparison
