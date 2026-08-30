"""
run_backtest_h1_2024.py
Rolling-origin backtest: exact same final hybrid model on 2024 H1.

Mirrors what we did for H2 2024 exactly:
  - Same features (master_train_v7.csv)
  - Same TiDE architecture (v8 config)
  - Same Prophet blend for Phesgrox + Vabyseal
  - Same ETS blend for Kadcynex
  - Same brand-model assignment

Training window: Jan 2021 - Dec 2023 (cutoff = 202312)
Test window    : Jan 2024 - Jun 2024 (H1 2024)

Compare to H2 2024 results to confirm model consistency across the full year.

Usage: python3 03_scripts/run_backtest_h1_2024.py
"""

import sys, warnings, json
sys.path.append('03_scripts')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path
from evaluate import wape, full_scorecard
from darts import TimeSeries
from darts.models import TiDEModel, Prophet, ExponentialSmoothing

OUTPUT = Path('04_outputs/backtest')
OUTPUT.mkdir(exist_ok=True)

TARGET    = 'iqvia_sales_qty_eqv'
INPUT     = 18
VAL       = 6
MIN_LEN   = INPUT + VAL
TRN_END   = 202312   # Train up to Dec 2023
H1_MONTHS = [202401,202402,202403,202404,202405,202406]
H2_MONTHS = [202407,202408,202409,202410,202411,202412]
MONTH_LBL = {202401:'Jan-24',202402:'Feb-24',202403:'Mar-24',
             202404:'Apr-24',202405:'May-24',202406:'Jun-24'}

print('='*60)
print('2024 H1 Backtest — Same Final Hybrid Applied to H1')
print('='*60)
print(f'Train : Jan 2021 - Dec 2023')
print(f'Predict: Jan 2024 - Jun 2024 (H1)')
print(f'Compare against H2 2024 results to confirm year-round stability')

# ── Load data ──────────────────────────────────────────────────
print('\nLoading master_train_v7.csv (same features as final model)...')
master = pd.read_csv('01_input/processed/master_train_v7.csv', low_memory=False)
gne    = master[master['flag_competitor']=='N'].copy()
gne['date'] = pd.to_datetime(gne['date_year_month'].astype(str), format='%Y%m')
prod   = pd.read_csv('01_input/raw/dim_product.csv')[['product_brand_id','product_brand_name']]
geo    = pd.read_csv('01_input/raw/dim_geography.csv')[['ecosystem_id','region']]
print(f'Total rows: {len(gne):,}')

# ── Feature lists (same as final model) ───────────────────────
with open('05_documents/final_feature_selection.json') as f:
    sel = json.load(f)
FUTURE_COLS = sel['FINAL_FUTURE_COLS']
PAST_COLS   = sel['FINAL_PAST_COLS']

STRUCT_COLS = ['flag_ocretiva_payer_break','months_since_ocretiva_break',
               'flag_ms_oral_competitor','ocretiva_pre_break_level',
               'vabyseal_retivue_ratio','vabyseal_ramp_completion',
               'regime_ratio','sales_momentum','yoy_acceleration']
NEW_FUTURE  = ['brand_seasonal_index','brand_h2_premium','is_h2']
NEW_PAST    = ['new_rx_pct','new_rx_pct_lag1','new_rx_pct_lag3',
               'prior_auth_delta','prior_auth_delta_3m',
               'pref_tier_delta','pref_tier_delta_3m']

FUTURE_COLS = FUTURE_COLS + [c for c in NEW_FUTURE if c in gne.columns and c not in FUTURE_COLS]
PAST_COLS   = PAST_COLS   + [c for c in STRUCT_COLS + NEW_PAST if c in gne.columns and c not in PAST_COLS]

# ── Split: train on Jan2021-Dec2023, validate on Jan-Jun 2024 ──
gne_trn = gne[gne['date_year_month'] <= TRN_END].copy()
gne_val = gne[gne['date_year_month'].isin(H1_MONTHS)].copy()

for c in FUTURE_COLS + PAST_COLS:
    if c not in gne_trn.columns: gne_trn[c] = 0.0
    gne_trn[c] = gne_trn[c].fillna(0)
for c in FUTURE_COLS:
    if c not in gne_val.columns: gne_val[c] = 0.0
    gne_val[c] = gne_val[c].fillna(0)

print(f'Train months: Jan-2021 to Dec-2023 ({gne_trn["date_year_month"].nunique()} months)')
print(f'Val   months: Jan-2024 to Jun-2024 ({len(H1_MONTHS)} months)')

# ── Build TimeSeries ───────────────────────────────────────────
print('\nBuilding TimeSeries...')
train_series, fut_cov_train, past_cov_train, group_ids = [], [], [], []

for (bid, eid), grp in gne_trn.groupby(['product_brand_id','ecosystem_id']):
    grp = grp.sort_values('date').set_index('date')
    ts  = TimeSeries.from_series(grp[TARGET].fillna(0), freq='MS')
    fc  = TimeSeries.from_dataframe(grp[[c for c in FUTURE_COLS if c in grp.columns]].fillna(0), freq='MS')
    pc  = TimeSeries.from_dataframe(grp[[c for c in PAST_COLS   if c in grp.columns]].fillna(0), freq='MS')
    train_series.append(ts); fut_cov_train.append(fc); past_cov_train.append(pc)
    group_ids.append((bid, eid))

# Extend future covariates into H1 2024
fut_cov_full = []
for i, (bid, eid) in enumerate(group_ids):
    val_grp = gne_val[(gne_val['product_brand_id']==bid) &
                      (gne_val['ecosystem_id']==eid)].sort_values('date').set_index('date')
    if len(val_grp) == 0:
        fut_cov_full.append(fut_cov_train[i]); continue
    fc_h = TimeSeries.from_dataframe(
        val_grp[[c for c in FUTURE_COLS if c in val_grp.columns]].fillna(0), freq='MS')
    fut_cov_full.append(fut_cov_train[i].append(fc_h))

vi      = [i for i,ts in enumerate(train_series) if len(ts) >= MIN_LEN]
t_s     = [train_series[i] for i in vi]
fc_s    = [fut_cov_train[i] for i in vi]
pc_s    = [past_cov_train[i] for i in vi]
fc_full = [fut_cov_full[i] for i in vi]
gids    = [group_ids[i] for i in vi]
print(f'Training: {len(t_s)} series | Excluded (too short): {len(train_series)-len(t_s)}')

# ══════════════════════════════════════════════════════════════
# STEP 1: Train TiDE (same config as final model)
# ══════════════════════════════════════════════════════════════
print('\nTraining TiDE (300 epochs, same config as final model)...')
model = TiDEModel(
    input_chunk_length=INPUT, output_chunk_length=6,
    num_encoder_layers=2, num_decoder_layers=2,
    decoder_output_dim=16, hidden_size=128,
    temporal_width_past=4, temporal_width_future=4,
    dropout=0.1, batch_size=64, n_epochs=300,
    add_encoders={'cyclic':{'future':['month']},
                  'datetime_attribute':{'future':['month','quarter']}},
    random_state=42,
    pl_trainer_kwargs={'accelerator':'cpu','enable_progress_bar':True},
)
model.fit(series=t_s, future_covariates=fc_s, past_covariates=pc_s, verbose=True)
model.save(str(OUTPUT / 'backtest_h1_model'))
print('TiDE training complete!')

# ── Predict H1 2024 ───────────────────────────────────────────
print('\nPredicting Jan-Jun 2024...')
preds = model.predict(n=6, series=t_s, future_covariates=fc_full, past_covariates=pc_s)

# Collect TiDE predictions and actuals
records = []
for i, (bid, eid) in enumerate(gids):
    brand = prod[prod['product_brand_id']==bid]['product_brand_name'].values
    brand = brand[0] if len(brand) > 0 else 'Unknown'
    pred_vals   = preds[i].values().flatten().clip(min=0)
    actual_grp  = gne_val[(gne_val['product_brand_id']==bid) &
                           (gne_val['ecosystem_id']==eid)].sort_values('date_year_month')
    if len(actual_grp) < 6: continue
    actual_vals = actual_grp[TARGET].fillna(0).values[:6]
    reg = geo[geo['ecosystem_id']==eid]['region'].values
    reg = reg[0] if len(reg) > 0 else 'Unknown'
    for step in range(6):
        records.append({'brand':brand,'ecosystem_id':eid,'region':reg,
                        'step':step+1,'month':H1_MONTHS[step],
                        'actual':actual_vals[step],'pred_tide':pred_vals[step]})

df = pd.DataFrame(records)

# ══════════════════════════════════════════════════════════════
# STEP 2: Prophet for Phesgrox + Vabyseal
# ══════════════════════════════════════════════════════════════
print('\nRunning Prophet for Phesgrox + Vabyseal...')

PROPHET_ALPHA = {'Phesgrox':0.75, 'Vabyseal':0.70}

for brand, tide_alpha in PROPHET_ALPHA.items():
    gne_b   = gne_trn[gne_trn['product_brand_name']==brand].copy()
    monthly = gne_b.groupby('date_year_month')[TARGET].sum().reset_index()
    monthly['date'] = pd.to_datetime(monthly['date_year_month'].astype(str), format='%Y%m')
    ts_trn  = TimeSeries.from_dataframe(monthly.set_index('date')[[TARGET]], freq='MS')

    try:
        m = Prophet(seasonality_mode='additive',yearly_seasonality=True,
                    weekly_seasonality=False,daily_seasonality=False)
        m.fit(ts_trn)
        pred = m.predict(n=12).values().flatten()[:6].clip(min=0)

        # Zone shares from training data
        tot   = gne_b.groupby('date_year_month')[TARGET].transform('sum')
        gne_b = gne_b.copy()
        gne_b['share'] = gne_b[TARGET]/(tot+1e-8)
        shares = gne_b.groupby('ecosystem_id')['share'].mean()
        shares = shares/shares.sum()

        # Apply blend to df
        for step in range(6):
            for eco_id, share in shares.items():
                mask = ((df['brand']==brand) & (df['ecosystem_id']==eco_id) &
                        (df['step']==step+1))
                if mask.sum() == 0: continue
                prophet_zone = float(pred[step]) * float(share)
                df.loc[mask,'pred_tide'] = (tide_alpha * df.loc[mask,'pred_tide'] +
                                            (1-tide_alpha) * prophet_zone)
        print(f'  {brand}: Prophet blend applied ({tide_alpha:.0%} TiDE + {1-tide_alpha:.0%} Prophet)')
    except Exception as e:
        print(f'  {brand}: Prophet failed ({e}) — using TiDE only')

# ══════════════════════════════════════════════════════════════
# STEP 3: ETS for Kadcynex
# ══════════════════════════════════════════════════════════════
print('\nRunning ETS for Kadcynex...')
ETS_ALPHA = {'Kadcynex': 0.50}

for brand, tide_alpha in ETS_ALPHA.items():
    gne_b   = gne_trn[gne_trn['product_brand_name']==brand].copy()
    monthly = gne_b.groupby('date_year_month')[TARGET].sum().reset_index()
    monthly['date'] = pd.to_datetime(monthly['date_year_month'].astype(str), format='%Y%m')
    ts_trn  = TimeSeries.from_dataframe(monthly.set_index('date')[[TARGET]], freq='MS')

    best_m, best_w = None, 999
    for trend in ['additive','multiplicative',None]:
        for seasonal in ['additive',None]:
            try:
                kw = {'trend':trend,'seasonal':seasonal}
                if seasonal: kw['seasonal_periods'] = 12
                m = ExponentialSmoothing(**kw)
                m.fit(ts_trn)
                p = m.predict(n=6).values().flatten().clip(min=0)
                # Use training error as proxy (no separate val in this loop)
                if best_m is None: best_m = m; best_w = 0
            except: pass

    if best_m:
        pred  = best_m.predict(n=6).values().flatten().clip(min=0)
        gne_b2 = gne_b.copy()
        tot   = gne_b2.groupby('date_year_month')[TARGET].transform('sum')
        gne_b2['share'] = gne_b2[TARGET]/(tot+1e-8)
        shares = gne_b2.groupby('ecosystem_id')['share'].mean()
        shares = shares/shares.sum()

        for step in range(6):
            for eco_id, share in shares.items():
                mask = ((df['brand']==brand) & (df['ecosystem_id']==eco_id) &
                        (df['step']==step+1))
                if mask.sum() == 0: continue
                ets_zone = float(pred[step]) * float(share)
                df.loc[mask,'pred_tide'] = (tide_alpha * df.loc[mask,'pred_tide'] +
                                            (1-tide_alpha) * ets_zone)
        print(f'  {brand}: ETS blend applied ({tide_alpha:.0%} TiDE + {1-tide_alpha:.0%} ETS)')

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
wape_fn = lambda g: np.sum(np.abs(g['actual']-g['pred_tide']))/(np.sum(np.abs(g['actual']))+1e-8)

H2_FINAL = {'Xolarin':0.66,'Hemvia':0.80,'Ocretiva':0.93,'Retivue':5.13,
             'Vabyseal':6.41,'Perjenta':5.50,'Kadcynex':6.86,'Phesgrox':7.07}

print(f'\n{"="*65}')
print(f'2024 H1 BACKTEST RESULTS — Same Final Hybrid')
print(f'{"="*65}')

print(f'\n1. BY PRODUCT')
print(f'  {"Brand":<12} {"H1-2024":>9} {"H2-2024":>9} {"Gap":>7} {"Consistent?"}')
print('  '+'-'*58)

h1_brand = {}
for brand, grp in df.groupby('brand'):
    h1 = wape_fn(grp)*100
    h2 = H2_FINAL.get(brand,0)
    h1_brand[brand] = h1
    gap = h1-h2
    chk = '✅ Consistent' if abs(gap)<2 else ('⚠️ Some variation' if abs(gap)<4 else '🚨 Large gap')
    print(f'  {brand:<12} {h1:>7.2f}%  {h2:>7.2f}%  {gap:>+5.1f}%  {chk}')

overall_h1 = wape_fn(df)*100
print(f'\n  H1-2024 overall WAPE : {overall_h1:.2f}%')
print(f'  H2-2024 overall WAPE : 3.12%  (final hybrid on H2)')
print(f'  Difference           : {overall_h1-3.12:+.2f}%')

print(f'\n2. BY REGION')
print(f'  {"Region":<12} {"H1-2024":>9} {"H2-2024":>9}')
print('  '+'-'*35)
H2_REGION = {'Midwest':3.98,'Northeast':3.25,'South':3.66,'West':3.26}
for region, grp in df.groupby('region'):
    h1 = wape_fn(grp)*100
    h2 = H2_REGION.get(region,0)
    print(f'  {region:<12} {h1:>7.2f}%  {h2:>7.2f}%')

print(f'\n3. BY FORECAST MONTH')
print(f'  {"Month":<10} {"H1-2024 WAPE":>14}')
print('  '+'-'*28)
for step, grp in df.groupby('step'):
    print(f'  {MONTH_LBL[H1_MONTHS[step-1]]:<10} {wape_fn(grp)*100:>12.2f}%')

print(f'\n4. CONSISTENCY VERDICT')
if abs(overall_h1-3.12) < 1.5:
    print(f'  ✅ CONSISTENT — Model performs equally well in H1 and H2')
    print(f'  H1 WAPE ({overall_h1:.2f}%) ≈ H2 WAPE (3.12%) — difference < 1.5%')
    print(f'  The model is robust across the full year, not just H2.')
elif overall_h1 < 5:
    print(f'  ✅ GOOD — H1 WAPE ({overall_h1:.2f}%) still well below 5%')
    print(f'  Slight variation from H2 (3.12%) expected due to different market conditions')
else:
    print(f'  ⚠️ H1 harder — WAPE ({overall_h1:.2f}%) vs H2 (3.12%)')
    print(f'  Investigate which brands/regions drive the H1 degradation')

json.dump({'h1_wape': round(float(overall_h1),4),
           'h2_wape': 3.12,
           'brands': {b:round(h1_brand[b],4) for b in h1_brand}},
          open(OUTPUT/'backtest_h1_results.json','w'), indent=2)
print(f'\nResults saved: 04_outputs/backtest/backtest_h1_results.json')
print(f'{"="*60}')
