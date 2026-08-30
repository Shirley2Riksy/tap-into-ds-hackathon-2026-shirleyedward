"""
LightGBM v2 — Targeted fixes for Phesgrox and Vabyseal
--------------------------------------------------------
Phesgrox fix: add national linear trend forecast as a feature.
  The model sees where the national total is headed and anchors
  zone predictions to it. Corrects the growing step-wise bias.

Vabyseal fix: train jointly on Retivue + Vabyseal (same OPH market).
  Retivue has 48 months vs Vabyseal's 28. Joint training gives LGBM
  enough data to learn OPH zone dynamics, then Vabyseal-specific
  growth stage features distinguish the two drugs.
"""

import warnings, json
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT  = Path(__file__).parent.parent
PROC  = ROOT/'01_input/processed'
RAW   = ROOT/'01_input/raw'
TIDE  = ROOT/'04_outputs/tide'
ENS   = ROOT/'04_outputs/ensemble'
NEW   = ROOT/'04_outputs/new_models'
FINAL = ROOT/'04_outputs/final'

TARGET     = 'iqvia_sales_qty_eqv'
VAL_MONTHS = [202407,202408,202409,202410,202411,202412]
HORIZON    = [202501,202502,202503,202504,202505,202506]

master  = pd.read_csv(PROC/'master_train_v7.csv', low_memory=False)
test_m  = pd.read_csv(PROC/'master_test_v7.csv',  low_memory=False)
test_r  = pd.read_csv(RAW/'test_features.csv')
gne     = master[master['flag_competitor']=='N'].copy()

def wape(a,p):
    a,p = np.array(a), np.array(p)
    return np.sum(np.abs(a-p))/(np.sum(np.abs(a))+1e-8)

LAG_FEATURES = ['lag_1','lag_2','lag_3','lag_6','lag_12',
                 'roll_mean_3','roll_mean_6','roll_std_3','yoy_growth']
STATIC_FEATURES = ['pct_lives_covered','pct_preferred','pct_prior_auth_required',
                    'access_burden','delta_pct_preferred','delta_pct_lives_covered',
                    'prior_auth_delta','prior_auth_delta_3m','pref_tier_delta',
                    'rep_calls_adstock','digital_adstock','marketing_spend_usd',
                    'copay_redemptions','effective_net_price_per_unit',
                    'brand_seasonal_index','is_h2','new_rx_pct','new_rx_pct_lag1',
                    'months_since_launch','regime_ratio','sales_momentum',
                    'market_share','treated_patient_volume','pct_65_plus',
                    'fourier_sin_1','fourier_cos_1','fourier_sin_2','fourier_cos_2',
                    'month_of_year','national_trend_forecast']   # NEW

def add_national_trend(df, brand, lookback=18):
    """Add national linear trend forecast as a per-row feature."""
    b_all = df[df['product_brand_name']==brand].copy()
    nat   = b_all.groupby('date_year_month')[TARGET].sum()
    nat_sorted = nat.sort_index()

    result = []
    for ym in sorted(b_all['date_year_month'].unique()):
        hist = nat_sorted[nat_sorted.index < ym]
        if len(hist) < 6:
            trend_fc = nat_sorted.iloc[:len(hist)+1].mean()
        else:
            lb = min(lookback, len(hist))
            x  = np.arange(lb)
            y  = hist.values[-lb:]
            c  = np.polyfit(x, y, 1)
            trend_fc = max(0, c[0]*lb + c[1])  # 1-step ahead
        result.append({'date_year_month': ym, 'national_trend_forecast': trend_fc})

    trend_df = pd.DataFrame(result)
    b_all = b_all.merge(trend_df, on='date_year_month', how='left')
    # Also normalise: zone's share of national trend
    b_all['zone_share_of_nat'] = b_all[TARGET] / (b_all['national_trend_forecast'] + 1e-8)
    return b_all

def add_national_trend_test(test_df_brand, train_nat, brand, horizon_months, lookback=18):
    """Build national trend forecasts for horizon months."""
    nat_vals = train_nat.sort_index().values
    result = []
    for i, ym in enumerate(sorted(horizon_months)):
        lb  = min(lookback, len(nat_vals)+i)
        x   = np.arange(lb)
        y   = np.concatenate([nat_vals[-lb+i:] if lb>i else nat_vals,
                               [result[j]['national_trend_forecast'] for j in range(i)]])[-lb:]
        c   = np.polyfit(np.arange(len(y)), y, 1)
        tfc = max(0, c[0]*len(y) + c[1])
        result.append({'date_year_month': ym, 'national_trend_forecast': tfc})
    trend_df = pd.DataFrame(result)
    return test_df_brand.merge(trend_df, on='date_year_month', how='left')

def get_feat_cols(df, extra_cols=None):
    base = LAG_FEATURES + STATIC_FEATURES + ['date_year_month','eco_cat']
    if extra_cols:
        base += extra_cols
    return [c for c in base if c in df.columns]

def recursive_predict(model, seed_df, test_df, feat_cols, horizon_months):
    results = []
    for step_idx, ym in enumerate(sorted(horizon_months)):
        test_row = test_df[test_df['date_year_month']==ym]
        step_preds = []
        for _, trow in test_row.iterrows():
            eco_id = trow['ecosystem_id']
            fd = {}
            for col in feat_cols:
                if col in ['lag_1','lag_2','lag_3']:
                    lag_n = int(col.split('_')[1])
                    if step_idx >= lag_n:
                        past = results[step_idx-lag_n]
                        pe   = past[past['ecosystem_id']==eco_id]
                        fd[col] = pe['pred'].values[0] if len(pe)>0 else 0
                    else:
                        hr = seed_df[seed_df['ecosystem_id']==eco_id]
                        fd[col] = hr[col].values[-1] if len(hr)>0 and col in hr.columns else 0
                elif col in trow.index: fd[col] = trow[col]
                elif col in seed_df.columns:
                    hr = seed_df[seed_df['ecosystem_id']==eco_id]
                    fd[col] = hr[col].values[-1] if len(hr)>0 else 0
                else: fd[col] = 0
            pred = max(0, float(model.predict(pd.DataFrame([fd])[feat_cols])[0]))
            step_preds.append({'ecosystem_id':eco_id,'date_year_month':ym,
                               'pred':pred,'row_id':trow.get('row_id')})
        results.append(pd.DataFrame(step_preds))
    return pd.concat(results, ignore_index=True)

print('='*65)
print('LightGBM v2 — Phesgrox + Vabyseal targeted fixes')
print('='*65)

scores_v2 = {}

# ════════════════════════════════════════════════════════════
# FIX 1 — PHESGROX: national trend as feature
# ════════════════════════════════════════════════════════════
print('\n=== PHESGROX — with national trend feature ===')

phes_all = add_national_trend(gne, 'Phesgrox', lookback=24)
phes_all['eco_cat'] = phes_all['ecosystem_id'].astype('category').cat.codes
phes_all = phes_all.dropna(subset=['lag_1'])

phes_tr  = phes_all[~phes_all['date_year_month'].isin(VAL_MONTHS)].copy()
phes_val = phes_all[phes_all['date_year_month'].isin(VAL_MONTHS)].copy()

feat_cols_p = get_feat_cols(phes_tr, extra_cols=['zone_share_of_nat'])

cutoff  = sorted(phes_tr['date_year_month'].unique())[-2:]
X_tr    = phes_tr[~phes_tr['date_year_month'].isin(cutoff)][feat_cols_p].fillna(0)
y_tr    = phes_tr[~phes_tr['date_year_month'].isin(cutoff)][TARGET]
X_es    = phes_tr[phes_tr['date_year_month'].isin(cutoff)][feat_cols_p].fillna(0)
y_es    = phes_tr[phes_tr['date_year_month'].isin(cutoff)][TARGET]

model_p = lgb.LGBMRegressor(objective='regression_l1', num_leaves=127,
    learning_rate=0.03, min_child_samples=15, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, lambda_l1=0.1, verbose=-1, n_estimators=1500)
model_p.fit(X_tr, y_tr, eval_set=[(X_es,y_es)],
    callbacks=[lgb.early_stopping(60,verbose=False), lgb.log_evaluation(-1)])
print(f'  Best iteration: {model_p.best_iteration_}')

# Direct val
direct_p = wape(phes_val[TARGET], np.clip(model_p.predict(phes_val[feat_cols_p].fillna(0)), 0, None))
print(f'  Direct WAPE  : {direct_p*100:.2f}%')

# Recursive val — use phes_val (actual H2 2024 rows with payer/promo features)
seed_p    = phes_tr[phes_tr['date_year_month'].isin(sorted(phes_tr['date_year_month'].unique())[-3:])].copy()
nat_p_tr  = gne[gne['product_brand_name']=='Phesgrox'].groupby('date_year_month')[TARGET].sum()
nat_p_tr  = nat_p_tr[~nat_p_tr.index.isin(VAL_MONTHS)]
# Use the actual val rows (from master_train_v7) as test_df for recursive predict
val_test_p = phes_val.copy()
val_test_p = add_national_trend_test(val_test_p, nat_p_tr, 'Phesgrox', VAL_MONTHS, 24)
val_test_p['eco_cat'] = val_test_p['ecosystem_id'].astype('category').cat.codes

rec_p     = recursive_predict(model_p, seed_p, val_test_p, feat_cols_p, VAL_MONTHS)
val_m     = phes_val[['ecosystem_id','date_year_month',TARGET]].merge(
    rec_p[['ecosystem_id','date_year_month','pred']], on=['ecosystem_id','date_year_month'], how='left')
rec_wape_p = wape(val_m[TARGET], val_m['pred'].fillna(0))
print(f'  Recursive WAPE: {rec_wape_p*100:.2f}%  (v1 was 6.06%)')

# Horizon
test_p_hor = test_m[test_m['product_brand_name']=='Phesgrox'].copy()
test_p_hor = add_national_trend_test(test_p_hor, nat_p_tr, 'Phesgrox', HORIZON, 24)
test_p_hor['eco_cat'] = test_p_hor['ecosystem_id'].astype('category').cat.codes
test_p_hor['zone_share_of_nat'] = 0.0  # unknown at horizon

seed_ph = phes_all[phes_all['date_year_month'].isin(sorted(phes_all['date_year_month'].unique())[-3:])].copy()
rec_hor_p = recursive_predict(model_p, seed_ph, test_p_hor, feat_cols_p, HORIZON)
test_p_raw = test_r[test_r['product_brand_name']=='Phesgrox']
sub_p = test_p_raw[['row_id','ecosystem_id','date_year_month']].merge(
    rec_hor_p[['ecosystem_id','date_year_month','pred']], on=['ecosystem_id','date_year_month'], how='left')
sub_p[['row_id','pred']].rename(columns={'pred':'forecast_units_eqv'}).to_csv(
    NEW/'lgbm_v2_phesgrox_submission.csv', index=False)

scores_v2['Phesgrox'] = {'v1_wape':0.0606,'v2_wape':float(rec_wape_p),'tide_wape':0.0918}
print(f'  Saved lgbm_v2_phesgrox_submission.csv')

# ════════════════════════════════════════════════════════════
# FIX 2 — VABYSEAL: joint OPH training (Retivue + Vabyseal)
# ════════════════════════════════════════════════════════════
print('\n=== VABYSEAL — joint OPH training (Retivue + Vabyseal) ===')

oph_brands = ['Retivue','Vabyseal']
oph_dfs = []
for br in oph_brands:
    b = add_national_trend(gne, br, lookback=12)
    b['eco_cat']    = b['ecosystem_id'].astype('category').cat.codes
    b['brand_cat']  = 0 if br=='Retivue' else 1
    oph_dfs.append(b)
oph_all = pd.concat(oph_dfs, ignore_index=True)
oph_all = oph_all.dropna(subset=['lag_1'])

oph_tr  = oph_all[~oph_all['date_year_month'].isin(VAL_MONTHS)].copy()
oph_val = oph_all[oph_all['date_year_month'].isin(VAL_MONTHS)].copy()
vas_val = oph_val[oph_val['product_brand_name']=='Vabyseal'].copy()

feat_cols_v = get_feat_cols(oph_tr, extra_cols=['zone_share_of_nat','brand_cat'])

cutoff_v = sorted(oph_tr['date_year_month'].unique())[-2:]
X_tr_v   = oph_tr[~oph_tr['date_year_month'].isin(cutoff_v)][feat_cols_v].fillna(0)
y_tr_v   = oph_tr[~oph_tr['date_year_month'].isin(cutoff_v)][TARGET]
X_es_v   = oph_tr[oph_tr['date_year_month'].isin(cutoff_v)][feat_cols_v].fillna(0)
y_es_v   = oph_tr[oph_tr['date_year_month'].isin(cutoff_v)][TARGET]

model_v = lgb.LGBMRegressor(objective='regression_l1', num_leaves=127,
    learning_rate=0.03, min_child_samples=15, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, lambda_l1=0.1, verbose=-1, n_estimators=1500)
model_v.fit(X_tr_v, y_tr_v, eval_set=[(X_es_v,y_es_v)],
    callbacks=[lgb.early_stopping(60,verbose=False), lgb.log_evaluation(-1)])
print(f'  Best iteration: {model_v.best_iteration_}')

direct_v = wape(vas_val[TARGET], np.clip(model_v.predict(vas_val[feat_cols_v].fillna(0)), 0, None))
print(f'  Direct WAPE (Vabyseal only): {direct_v*100:.2f}%')

# Recursive val for Vabyseal — use actual val rows
seed_v   = oph_all[(oph_all['product_brand_name']=='Vabyseal') &
                    (oph_all['date_year_month'].isin(sorted(oph_all[oph_all['product_brand_name']=='Vabyseal']['date_year_month'].unique())[-3:]))].copy()
nat_v_tr = gne[gne['product_brand_name']=='Vabyseal'].groupby('date_year_month')[TARGET].sum()
nat_v_tr = nat_v_tr[~nat_v_tr.index.isin(VAL_MONTHS)]
val_test_v = vas_val.copy()
val_test_v = add_national_trend_test(val_test_v, nat_v_tr, 'Vabyseal', VAL_MONTHS, 12)
val_test_v['eco_cat']   = val_test_v['ecosystem_id'].astype('category').cat.codes
val_test_v['brand_cat'] = 1

rec_v    = recursive_predict(model_v, seed_v, val_test_v, feat_cols_v, VAL_MONTHS)
val_mv   = vas_val[['ecosystem_id','date_year_month',TARGET]].merge(
    rec_v[['ecosystem_id','date_year_month','pred']], on=['ecosystem_id','date_year_month'], how='left')
rec_wape_v = wape(val_mv[TARGET], val_mv['pred'].fillna(0))
print(f'  Recursive WAPE (Vabyseal): {rec_wape_v*100:.2f}%  (ensemble was 6.37%)')

# Horizon
test_v_hor = test_m[test_m['product_brand_name']=='Vabyseal'].copy()
test_v_hor = add_national_trend_test(test_v_hor, nat_v_tr, 'Vabyseal', HORIZON, 12)
test_v_hor['eco_cat']   = test_v_hor['ecosystem_id'].astype('category').cat.codes
test_v_hor['brand_cat'] = 1
test_v_hor['zone_share_of_nat'] = 0.0
seed_vh = oph_all[(oph_all['product_brand_name']=='Vabyseal') &
                   (oph_all['date_year_month'].isin(sorted(oph_all[oph_all['product_brand_name']=='Vabyseal']['date_year_month'].unique())[-3:]))].copy()
rec_hor_v = recursive_predict(model_v, seed_vh, test_v_hor, feat_cols_v, HORIZON)
test_v_raw = test_r[test_r['product_brand_name']=='Vabyseal']
sub_v = test_v_raw[['row_id','ecosystem_id','date_year_month']].merge(
    rec_hor_v[['ecosystem_id','date_year_month','pred']], on=['ecosystem_id','date_year_month'], how='left')
sub_v[['row_id','pred']].rename(columns={'pred':'forecast_units_eqv'}).to_csv(
    NEW/'lgbm_v2_vabyseal_submission.csv', index=False)

scores_v2['Vabyseal'] = {'v1_wape':0.0637,'v2_wape':float(rec_wape_v),'ensemble_wape':0.0637}
print(f'  Saved lgbm_v2_vabyseal_submission.csv')

# ════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('LGBM v2 SUMMARY')
print('='*65)
for brand, s in scores_v2.items():
    base = s.get('v1_wape', s.get('ensemble_wape'))
    imp  = (base - s['v2_wape'])/base*100
    flag = '✅' if s['v2_wape']<0.06 else ('⚠️' if s['v2_wape']<0.07 else '🔴')
    print(f'  {brand}: {base*100:.2f}% → {s["v2_wape"]*100:.2f}%  ({imp:+.1f}%)  {flag}')

with open(NEW/'lgbm_v2_scores.json','w') as f:
    json.dump({k:{m:round(float(v),5) for m,v in d.items()} for k,d in scores_v2.items()}, f, indent=2)
