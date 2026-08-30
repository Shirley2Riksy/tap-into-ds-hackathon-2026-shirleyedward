"""
run_prophet_ets.py
Prophet (Phesgrox/Vabyseal) + ETS (Kadcynex) — brand-level then zone allocation.

Approach:
  1. Aggregate to brand national total (much cleaner signal)
  2. Fit Prophet/ETS on the national total
  3. Distribute to zones using historical stable zone shares
  4. Blend with TiDE v5 zone predictions using optimal alpha

Usage: python3 03_scripts/run_prophet_ets.py
"""

import sys, warnings, json
sys.path.append('03_scripts')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path
from evaluate import wape, full_scorecard
from utils import make_submission
from darts import TimeSeries
from darts.models import Prophet, ExponentialSmoothing

OUTPUT = Path('04_outputs/ensemble')
OUTPUT.mkdir(exist_ok=True)

TARGET = 'iqvia_sales_qty_eqv'
VAL    = 6
STEP_TO_MONTH = {1:202407,2:202408,3:202409,4:202410,5:202411,6:202412}
VAL_MONTHS    = list(STEP_TO_MONTH.values())
TEST_MONTHS   = [202501,202502,202503,202504,202505,202506]
TRN_MONTHS_ALL= [202101,202102,202103,202104,202105,202106,
                  202107,202108,202109,202110,202111,202112,
                  202201,202202,202203,202204,202205,202206,
                  202207,202208,202209,202210,202211,202212,
                  202301,202302,202303,202304,202305,202306,
                  202307,202308,202309,202310,202311,202312,
                  202401,202402,202403,202404,202405,202406]

print('='*60)
print('Prophet + ETS — Brand-level forecast + Zone allocation')
print('='*60)

# ── Load data ──────────────────────────────────────────────────
master = pd.read_csv('01_input/processed/master_train_v4.csv', low_memory=False)
diag   = pd.read_csv('04_outputs/tide/tide_v5_diagnostic.csv')
prod   = pd.read_csv('01_input/raw/dim_product.csv')[['product_brand_id','product_brand_name']]
test_m = pd.read_csv('01_input/raw/test_features.csv')
# product_brand_name already exists in test_features.csv
tide_sub = pd.read_csv('04_outputs/tide/tide_v5_submission.csv')
tide_sub = tide_sub.merge(test_m[['row_id','product_brand_name','ecosystem_id','date_year_month']],
                           on='row_id', how='left')

gne = master[master['flag_competitor']=='N'].copy()
gne['date'] = pd.to_datetime(gne['date_year_month'].astype(str), format='%Y%m')
diag['date_year_month'] = diag['step'].map(STEP_TO_MONTH)

TIDE_V5_WAPE = {'Kadcynex':0.0805,'Phesgrox':0.0918,'Vabyseal':0.0641}

CONFIG = {
    'Phesgrox': 'Prophet',
    'Vabyseal': 'Prophet',
    'Kadcynex': 'ETS',
}

all_results = {}

for brand, model_type in CONFIG.items():
    print(f'\n{"="*55}')
    print(f'{brand} — {model_type}')
    print(f'{"="*55}')

    brand_id = prod[prod['product_brand_name']==brand]['product_brand_id'].values[0]
    gne_b    = gne[gne['product_brand_name']==brand].copy()

    # ── Brand-level monthly total ──────────────────────────────
    monthly = gne_b.groupby('date_year_month')[TARGET].sum().reset_index()
    monthly = monthly.sort_values('date_year_month')
    monthly['date'] = pd.to_datetime(monthly['date_year_month'].astype(str), format='%Y%m')

    trn = monthly[~monthly['date_year_month'].isin(VAL_MONTHS)]
    val = monthly[monthly['date_year_month'].isin(VAL_MONTHS)]

    ts_trn = TimeSeries.from_dataframe(trn.set_index('date')[[TARGET]], freq='MS')
    ts_val = TimeSeries.from_dataframe(val.set_index('date')[[TARGET]], freq='MS')

    print(f'  Brand total — Train: {len(ts_trn)} months | Val: {len(ts_val)} months')
    print(f'  Train range: {trn["iqvia_sales_qty_eqv"].min():.0f} - {trn["iqvia_sales_qty_eqv"].max():.0f} units/month')

    # ── Fit model ──────────────────────────────────────────────
    try:
        if model_type == 'Prophet':
            model = Prophet(
                seasonality_mode='additive',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
        else:  # ETS
            best_model, best_w = None, 999
            for trend in ['additive', 'multiplicative', None]:
                for seasonal in ['additive', None]:
                    try:
                        kwargs = {'trend': trend, 'seasonal': seasonal}
                        if seasonal: kwargs['seasonal_periods'] = 12
                        m = ExponentialSmoothing(**kwargs)
                        m.fit(ts_trn)
                        p = m.predict(n=VAL)
                        w = (np.sum(np.abs(ts_val.values().flatten() - p.values().flatten())) /
                             (np.sum(np.abs(ts_val.values().flatten())) + 1e-8))
                        if w < best_w:
                            best_w     = w
                            best_model = m
                    except: pass
            model = best_model

        if model_type == 'Prophet':
            model.fit(ts_trn)

        pred_val  = model.predict(n=VAL)
        pred_test = model.predict(n=VAL + len(TEST_MONTHS))

        brand_val_preds  = pred_val.values().flatten().clip(min=0)
        brand_test_preds = pred_test.values().flatten()[-len(TEST_MONTHS):].clip(min=0)

    except Exception as e:
        print(f'  Model failed: {e}')
        continue

    # Brand-level WAPE
    brand_actual = ts_val.values().flatten()
    w_brand = np.sum(np.abs(brand_actual - brand_val_preds)) / (np.sum(np.abs(brand_actual)) + 1e-8)
    w_tide_brand = np.sum(np.abs(brand_actual - diag[diag['product_brand_name']==brand].groupby('date_year_month')['y_pred'].sum().values)) / \
                   (np.sum(np.abs(brand_actual)) + 1e-8)

    print(f'  Brand-level {model_type} WAPE: {w_brand*100:.2f}%')

    # ── Zone shares from training data ─────────────────────────
    gne_trn = gne_b[~gne_b['date_year_month'].isin(VAL_MONTHS)]
    monthly_total = gne_trn.groupby('date_year_month')[TARGET].transform('sum')
    gne_trn = gne_trn.copy()
    gne_trn['zone_share'] = gne_trn[TARGET] / (monthly_total + 1e-8)
    zone_shares = gne_trn.groupby('eco_id' if 'eco_id' in gne_trn.columns else 'ecosystem_id')['zone_share'].mean()
    zone_shares = zone_shares / zone_shares.sum()

    # ── Compute zone-level predictions for validation ──────────
    bd_diag = diag[diag['product_brand_name']==brand].copy()

    zone_preds_val, zone_actuals, zone_ids_val, steps_val = [], [], [], []
    for step_idx, (month, brand_pred) in enumerate(zip(VAL_MONTHS, brand_val_preds)):
        month_diag = bd_diag[bd_diag['date_year_month']==month]
        for _, row in month_diag.iterrows():
            eco_id = row['ecosystem_id']
            share  = zone_shares.get(eco_id, 1/80)
            zone_preds_val.append(brand_pred * share)
            zone_actuals.append(row['y_true'])
            zone_ids_val.append(eco_id)
            steps_val.append(step_idx+1)

    zone_preds_val  = np.array(zone_preds_val).clip(min=0)
    zone_actuals    = np.array(zone_actuals)
    tide_zone_preds = bd_diag.sort_values(['date_year_month','ecosystem_id'])['y_pred'].values

    w_zone = np.sum(np.abs(zone_actuals - zone_preds_val)) / (np.sum(np.abs(zone_actuals)) + 1e-8)
    w_tide = np.sum(np.abs(zone_actuals - tide_zone_preds[:len(zone_actuals)])) / \
             (np.sum(np.abs(zone_actuals)) + 1e-8)

    print(f'  Zone-level {model_type} WAPE : {w_zone*100:.2f}%')
    print(f'  Zone-level TiDE v5 WAPE: {w_tide*100:.2f}%')

    # ── Find optimal blend ────────────────────────────────────
    tide_sorted = bd_diag.sort_values(['date_year_month','ecosystem_id'])['y_pred'].values[:len(zone_actuals)]
    best_alpha, best_wape = 1.0, w_tide
    for alpha in np.arange(0, 1.05, 0.05):
        blended = alpha * tide_sorted + (1-alpha) * zone_preds_val
        w_b = np.sum(np.abs(zone_actuals-blended)) / (np.sum(np.abs(zone_actuals))+1e-8)
        if w_b < best_wape:
            best_wape = w_b
            best_alpha = alpha

    improvement = (TIDE_V5_WAPE[brand] - best_wape) / TIDE_V5_WAPE[brand] * 100
    verdict = '✅ Helps' if best_wape < TIDE_V5_WAPE[brand] else '❌ TiDE v5 better'
    print(f'  Best blend ({best_alpha:.0%} TiDE + {1-best_alpha:.0%} {model_type}): '
          f'{best_wape*100:.2f}%')
    print(f'  vs TiDE v5 ({TIDE_V5_WAPE[brand]*100:.2f}%): {improvement:+.1f}%  {verdict}')

    # ── Test predictions ───────────────────────────────────────
    test_zone_preds = []
    for step_idx, (month, brand_pred) in enumerate(zip(TEST_MONTHS, brand_test_preds)):
        month_test = test_m[(test_m['product_brand_name']==brand) &
                             (test_m['date_year_month']==month)]
        for _, row in month_test.iterrows():
            eco_id = row['ecosystem_id']
            share  = zone_shares.get(eco_id, 1/80)
            tide_pred = tide_sub[(tide_sub['product_brand_name']==brand) &
                                  (tide_sub['ecosystem_id']==eco_id) &
                                  (tide_sub['date_year_month']==month)]['forecast_units_eqv']
            tide_val = tide_pred.values[0] if len(tide_pred) > 0 else brand_pred * share
            blended  = best_alpha * tide_val + (1-best_alpha) * max(brand_pred * share, 0)
            test_zone_preds.append({'row_id': row['row_id'],
                                    'forecast_units_eqv': round(float(blended), 4)})

    all_results[brand] = {
        'model': model_type,
        'wape_model_zone': float(w_zone),
        'wape_tide': float(w_tide),
        'best_alpha': float(best_alpha),
        'best_wape': float(best_wape),
        'test_preds': test_zone_preds
    }

# ── Summary ────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('SUMMARY')
print(f'{"="*60}')
print(f'\n{"Brand":<12} {"Model":>8} {"Model WAPE":>12} {"Best Blend":>12} {"TiDE v5":>9} {"Verdict"}')
print('-'*70)
for brand, res in all_results.items():
    v = '✅ Add to ensemble' if res['best_wape'] < TIDE_V5_WAPE[brand] else '❌ Skip'
    print(f'  {brand:<12} {res["model"]:>8} {res["wape_model_zone"]*100:>10.2f}%  '
          f'{res["best_wape"]*100:>10.2f}%  {TIDE_V5_WAPE[brand]*100:>7.2f}%  {v}')

json.dump({b: {k:v for k,v in r.items() if k!='test_preds'}
           for b,r in all_results.items()},
          open(OUTPUT/'prophet_ets_results.json','w'), indent=2)
print(f'\nResults saved: 04_outputs/ensemble/prophet_ets_results.json')
print('If any model helps, run run_ensemble_final.py to build final submission.')
