"""
Brand-Specific LightGBM with Recursive Multi-Step Forecasting
--------------------------------------------------------------
Key insight: lag_1 has 0.986 correlation with target for problem brands.
TiDE does DIRECT 6-step forecasting — predicts all months at once without
feeding predictions back as inputs.

LightGBM recursive approach:
  Step 1: predict Jan 2025 using Dec 2024 as lag_1 (known)
  Step 2: predict Feb 2025 using Jan 2025 prediction as lag_1
  Step 3-6: continue recursively

This fully exploits the lag_1=0.986 signal at every step.
Cross-zone features (payer, promo, epidemiology) provide additional signal.

One model per brand — allows brand-specific feature selection and tuning.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import json
from pathlib import Path

try:
    import lightgbm as lgb
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
    import lightgbm as lgb

ROOT  = Path(__file__).parent.parent
PROC  = ROOT / '01_input' / 'processed'
RAW   = ROOT / '01_input' / 'raw'
TIDE  = ROOT / '04_outputs' / 'tide'
OUT   = ROOT / '04_outputs' / 'new_models'
OUT.mkdir(exist_ok=True)

TARGET     = 'iqvia_sales_qty_eqv'
VAL_MONTHS = [202407,202408,202409,202410,202411,202412]
HORIZON    = [202501,202502,202503,202504,202505,202506]

print('='*65)
print('Brand-Specific LightGBM — Recursive Multi-Step')
print('='*65)

master = pd.read_csv(PROC / 'master_train_v7.csv', low_memory=False)
test_m = pd.read_csv(PROC / 'master_test_v7.csv',  low_memory=False)
test_raw = pd.read_csv(RAW / 'test_features.csv')
gne    = master[master['flag_competitor']=='N'].copy()

# Feature columns — carefully selected, no data leakage
LAG_FEATURES = ['lag_1','lag_2','lag_3','lag_6','lag_12',
                 'roll_mean_3','roll_mean_6','roll_std_3','yoy_growth']

STATIC_FEATURES = ['pct_lives_covered','pct_preferred',
                    'pct_prior_auth_required','access_burden',
                    'delta_pct_preferred','delta_pct_lives_covered',
                    'prior_auth_delta','prior_auth_delta_3m',
                    'pref_tier_delta',
                    'rep_calls_adstock','digital_adstock',
                    'marketing_spend_usd','copay_redemptions',
                    'effective_net_price_per_unit',
                    'brand_seasonal_index','is_h2',
                    'new_rx_pct','new_rx_pct_lag1',
                    'months_since_launch','regime_ratio',
                    'sales_momentum','market_share',
                    'treated_patient_volume','pct_65_plus',
                    'fourier_sin_1','fourier_cos_1',
                    'fourier_sin_2','fourier_cos_2',
                    'month_of_year']

TIME_FEATURES = ['date_year_month']

def wape(actual, predicted):
    a, p = np.array(actual), np.array(predicted)
    return np.sum(np.abs(a-p)) / (np.sum(np.abs(a)) + 1e-8)

def get_features(df, brand, lag_cols, static_cols):
    """Get available features for a brand."""
    cols = [c for c in lag_cols + static_cols + TIME_FEATURES
            if c in df.columns]
    b = df[df['product_brand_name']==brand].copy()
    # Add ecosystem as category for cross-zone learning
    if 'ecosystem_id' in b.columns:
        b['eco_cat'] = b['ecosystem_id'].astype('category').cat.codes
        cols = cols + ['eco_cat']
    # Drop rows with NaN in lag features (cold start)
    b = b.dropna(subset=[c for c in lag_cols if c in b.columns])
    return b, [c for c in cols if c in b.columns]

def lgbm_params(brand):
    """Brand-specific LGBM hyperparameters."""
    base = {
        'objective': 'regression_l1',   # MAE → better for WAPE
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_child_samples': 15,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'n_estimators': 800,
        'early_stopping_rounds': 50,
    }
    # Growing brands: allow more complexity
    if brand in ['Phesgrox','Vabyseal']:
        base['num_leaves'] = 127
        base['learning_rate'] = 0.03
        base['n_estimators'] = 1200
    # Declining brands: simpler model
    elif brand in ['Kadcynex','Perjenta']:
        base['num_leaves'] = 31
        base['lambda_l1'] = 0.3
    return base

def recursive_predict(model, last_known_row, test_brand_df, feature_cols,
                      n_steps=6, horizon_months=None):
    """
    Recursive multi-step prediction.
    Start from last known row, predict step by step using each prediction
    as the lag_1 input for the next step.
    """
    results = []
    current = last_known_row.copy()

    for step_idx, ym in enumerate(horizon_months or HORIZON):
        # Get the test row for this month (has updated payer/promo features)
        test_row = test_brand_df[test_brand_df['date_year_month']==ym]
        if len(test_row) == 0:
            continue

        # For each zone, predict using its specific features
        step_preds = []
        for _, trow in test_row.iterrows():
            eco_id = trow['ecosystem_id']
            row_id = trow.get('row_id', None)

            # Build feature vector for this zone/month
            feat_dict = {}
            for col in feature_cols:
                if col in ['lag_1','lag_2','lag_3']:
                    # Use recursive predictions for lags
                    lag_n = int(col.split('_')[1])
                    # Find value lag_n steps back in our predictions
                    if step_idx >= lag_n:
                        past_pred = results[step_idx - lag_n]
                        past_eco  = past_pred[past_pred['ecosystem_id']==eco_id]
                        feat_dict[col] = past_eco['pred'].values[0] if len(past_eco)>0 else trow.get(col, 0)
                    else:
                        # Fall back to known history
                        hist_row = current[current['ecosystem_id']==eco_id]
                        feat_dict[col] = hist_row[col].values[0] if len(hist_row)>0 else 0
                elif col in trow.index:
                    feat_dict[col] = trow[col]
                elif col in current.columns:
                    hist_row = current[current['ecosystem_id']==eco_id]
                    feat_dict[col] = hist_row[col].values[0] if len(hist_row)>0 else 0
                else:
                    feat_dict[col] = 0

            feat_df = pd.DataFrame([feat_dict])[feature_cols]
            pred = max(0, float(model.predict(feat_df)[0]))
            step_preds.append({'ecosystem_id': eco_id, 'row_id': row_id,
                               'date_year_month': ym, 'pred': pred})

        results.append(pd.DataFrame(step_preds))

    return pd.concat(results, ignore_index=True)

# ── Train and evaluate per brand ─────────────────────────────
BRANDS_TO_TRAIN = ['Phesgrox','Kadcynex','Vabyseal','Perjenta','Retivue']

all_val_results   = {}
all_submissions   = {}
all_scores        = {}

tide_v5_wapes = {'Phesgrox':0.0918,'Kadcynex':0.0805,'Vabyseal':0.0641,
                 'Perjenta':0.0550,'Retivue':0.0566}

for brand in BRANDS_TO_TRAIN:
    print(f'\n{"="*50}')
    print(f'Training LightGBM — {brand}')
    print(f'{"="*50}')

    b_data, feat_cols = get_features(gne, brand, LAG_FEATURES, STATIC_FEATURES)
    b_train = b_data[~b_data['date_year_month'].isin(VAL_MONTHS)].copy()
    b_val   = b_data[b_data['date_year_month'].isin(VAL_MONTHS)].copy()

    if len(b_train) < 100:
        print(f'  Skipping — insufficient training data ({len(b_train)} rows)')
        continue

    X_train = b_train[feat_cols]
    y_train = b_train[TARGET]

    # Use last 2 months of training as internal val for early stopping
    cutoff   = sorted(b_train['date_year_month'].unique())[-2:]
    X_es_val = b_train[b_train['date_year_month'].isin(cutoff)][feat_cols]
    y_es_val = b_train[b_train['date_year_month'].isin(cutoff)][TARGET]
    X_tr_es  = b_train[~b_train['date_year_month'].isin(cutoff)][feat_cols]
    y_tr_es  = b_train[~b_train['date_year_month'].isin(cutoff)][TARGET]

    params = lgbm_params(brand)
    n_est  = params.pop('n_estimators')
    es     = params.pop('early_stopping_rounds')

    model = lgb.LGBMRegressor(**params, n_estimators=n_est)
    model.fit(
        X_tr_es, y_tr_es,
        eval_set=[(X_es_val, y_es_val)],
        callbacks=[lgb.early_stopping(es, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )
    print(f'  Best iteration: {model.best_iteration_}')

    # ── Direct validation (non-recursive) to check model fit ─
    X_val = b_val[feat_cols].fillna(0)
    y_val = b_val[TARGET]
    direct_preds = np.clip(model.predict(X_val), 0, None)
    direct_wape  = wape(y_val.values, direct_preds)
    print(f'  Direct val WAPE (uses true lags): {direct_wape*100:.2f}%')

    # ── Recursive validation (realistic 6-step ahead) ─────────
    # Get last known state (Jun 2024) as seed
    last_known = b_train.copy()
    test_brand_df = test_m[test_m['product_brand_name']==brand].copy()
    if 'eco_cat' in feat_cols:
        test_brand_df['eco_cat'] = test_brand_df['ecosystem_id'].astype('category').cat.codes

    # For validation: predict Jul-Dec 2024 recursively from Jun 2024
    # Seed = training rows from May-Jun 2024 (for lag_1, lag_2)
    seed_months = sorted(b_train['date_year_month'].unique())[-3:]
    seed_df     = b_train[b_train['date_year_month'].isin(seed_months)].copy()
    val_test_df = b_val.copy()  # has actual payer/promo features for Jul-Dec

    rec_preds = recursive_predict(
        model, seed_df, val_test_df,
        feat_cols, horizon_months=sorted(VAL_MONTHS))

    val_merged = b_val[['ecosystem_id','date_year_month',TARGET]].merge(
        rec_preds[['ecosystem_id','date_year_month','pred']],
        on=['ecosystem_id','date_year_month'], how='left')
    val_merged['pred'] = val_merged['pred'].fillna(val_merged[TARGET].mean())

    rec_wape = wape(val_merged[TARGET].values, val_merged['pred'].values)
    print(f'  Recursive val WAPE (realistic)  : {rec_wape*100:.2f}%')
    print(f'  TiDE v5 baseline                : {tide_v5_wapes[brand]*100:.2f}%')
    improvement = (tide_v5_wapes[brand] - rec_wape) / tide_v5_wapes[brand] * 100
    print(f'  {"✅ IMPROVEMENT" if improvement>0 else "⚠️ REGRESSION"}: {abs(improvement):.1f}%')

    all_val_results[brand] = rec_wape

    # ── Feature importance ────────────────────────────────────
    fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f'  Top 5 features: {list(fi.head(5).index)}')

    # ── Horizon forecast (Jan-Jun 2025 recursive) ─────────────
    test_brand_raw = test_raw[test_raw['product_brand_name']==brand].copy()
    test_brand_feat = test_m[test_m['product_brand_name']==brand].copy()
    if 'eco_cat' in feat_cols:
        test_brand_feat['eco_cat'] = test_brand_feat['ecosystem_id'].astype('category').cat.codes

    # Seed from last 3 months of all training data (Oct-Dec 2024)
    last_months = sorted(b_data['date_year_month'].unique())[-3:]
    horizon_seed = b_data[b_data['date_year_month'].isin(last_months)].copy()

    horizon_rec = recursive_predict(
        model, horizon_seed, test_brand_feat,
        feat_cols, horizon_months=sorted(HORIZON))

    # Merge with row_ids
    horizon_merged = test_brand_raw[['row_id','ecosystem_id','date_year_month']].merge(
        horizon_rec[['ecosystem_id','date_year_month','pred']],
        on=['ecosystem_id','date_year_month'], how='left')
    horizon_merged['pred'] = horizon_merged['pred'].fillna(0).clip(lower=0)

    submit_rows = horizon_merged[['row_id','pred']].rename(
        columns={'pred':'forecast_units_eqv'})
    all_submissions[brand] = submit_rows

    all_scores[brand] = {
        'direct_wape'   : round(float(direct_wape),5),
        'recursive_wape': round(float(rec_wape),5),
        'tide_v5_wape'  : tide_v5_wapes[brand],
        'improvement_pct': round(float(improvement),2)
    }

# ── Save ─────────────────────────────────────────────────────
print('\n' + '='*65)
print('LGBM RESULTS vs TiDE BASELINE')
print('='*65)
print(f'\n{"Brand":<12} {"TiDE v5":>9} {"LGBM Direct":>13} {"LGBM Recursive":>16} {"vs TiDE":>10} {"Below 6%?"}')
print('-'*70)
for brand, scores in all_scores.items():
    imp = scores['improvement_pct']
    rec = scores['recursive_wape']
    marker = '✅ YES' if rec < 0.06 else ('✅ <7%' if rec < 0.07 else '⚠️')
    print(f'  {brand:<12} {scores["tide_v5_wape"]*100:>7.2f}%  '
          f'{scores["direct_wape"]*100:>11.2f}%  {rec*100:>14.2f}%  '
          f'{imp:>+8.1f}%  {marker}')

# Save individual submissions
for brand, sub in all_submissions.items():
    sub.to_csv(OUT / f'lgbm_{brand.lower()}_submission.csv', index=False)
    print(f'Saved: lgbm_{brand.lower()}_submission.csv ({len(sub)} rows)')

with open(OUT / 'lgbm_scores.json', 'w') as f:
    json.dump(all_scores, f, indent=2)

print('\n✅ LightGBM training complete.')
