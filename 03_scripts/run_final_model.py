"""
run_final_model.py
Integrated final model pipeline — runs everything overnight in sequence.

Steps:
  1. Train TiDE v8 (low-risk features: seasonal, new_rx_pct, prior_auth_delta)
  2. Validate TiDE v8 per brand
  3. Run Prophet for Phesgrox + Vabyseal (brand-level)
  4. Run ETS for Kadcynex (brand-level)
  5. Re-optimise blend weights for TiDE v8 + Prophet/ETS
  6. Build final hybrid submission:
       Xolarin, Hemvia, Ocretiva, Perjenta  → TiDE v8
       Retivue                              → best of TiDE v8 / TiDE v6
       Phesgrox, Vabyseal                  → TiDE v8 blend with Prophet
       Kadcynex                             → TiDE v8 blend with ETS
  7. Save final_submission.csv

Expected runtime: ~2-3 hours total

Usage: python3 03_scripts/run_final_model.py
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
from darts.models import TiDEModel, Prophet, ExponentialSmoothing

OUTPUT_TIDE = Path('04_outputs/tide')
OUTPUT_ENS  = Path('04_outputs/ensemble')
OUTPUT_FINAL= Path('04_outputs/final')
for p in [OUTPUT_TIDE, OUTPUT_ENS, OUTPUT_FINAL]:
    p.mkdir(exist_ok=True)

TARGET = 'iqvia_sales_qty_eqv'
VAL    = 6
INPUT  = 18
MIN_LEN= INPUT + VAL
STEP_TO_MONTH = {1:202407,2:202408,3:202409,4:202410,5:202411,6:202412}
VAL_MONTHS    = list(STEP_TO_MONTH.values())
TEST_MONTHS   = [202501,202502,202503,202504,202505,202506]

print('='*60)
print('Final Model Pipeline — TiDE v8 + Prophet/ETS Ensemble')
print('='*60)

# ── Load data ──────────────────────────────────────────────────
print('\nLoading master_train_v7.csv (all low-risk features)...')
master   = pd.read_csv('01_input/processed/master_train_v7.csv', low_memory=False)
test     = pd.read_csv('01_input/processed/master_test_v7.csv',  low_memory=False)
gne      = master[master['flag_competitor']=='N'].copy()
gne_test = test.copy()
gne['date']      = pd.to_datetime(gne['date_year_month'].astype(str), format='%Y%m')
gne_test['date'] = pd.to_datetime(gne_test['date_year_month'].astype(str), format='%Y%m')
prod     = pd.read_csv('01_input/raw/dim_product.csv')[['product_brand_id','product_brand_name']]
test_meta= pd.read_csv('01_input/raw/test_features.csv')
sub_base = pd.read_csv('01_input/raw/sample_submission.csv')
print(f'Train: {len(gne):,} | Test: {len(gne_test):,}')

# ── Feature lists ──────────────────────────────────────────────
with open('05_documents/final_feature_selection.json') as f:
    sel = json.load(f)

FUTURE_COLS = sel['FINAL_FUTURE_COLS']
PAST_COLS   = sel['FINAL_PAST_COLS']

STRUCT_COLS = [
    'flag_ocretiva_payer_break','months_since_ocretiva_break',
    'flag_ms_oral_competitor','ocretiva_pre_break_level',
    'vabyseal_retivue_ratio','vabyseal_ramp_completion',
    'regime_ratio','sales_momentum','yoy_acceleration'
]
NEW_FUTURE = ['brand_seasonal_index','brand_h2_premium','is_h2']
NEW_PAST   = ['new_rx_pct','new_rx_pct_lag1','new_rx_pct_lag3',
              'prior_auth_delta','prior_auth_delta_3m',
              'pref_tier_delta','pref_tier_delta_3m']

FUTURE_COLS = FUTURE_COLS + [c for c in NEW_FUTURE
                              if c in gne.columns and c not in FUTURE_COLS]
PAST_COLS   = PAST_COLS   + [c for c in STRUCT_COLS + NEW_PAST
                              if c in gne.columns and c not in PAST_COLS]

for c in FUTURE_COLS + PAST_COLS:
    if c not in gne.columns:      gne[c]      = 0.0
    gne[c] = gne[c].fillna(0)
for c in FUTURE_COLS:
    if c not in gne_test.columns: gne_test[c] = 0.0
    gne_test[c] = gne_test[c].fillna(0)
for c in NEW_PAST:
    if c not in gne_test.columns: gne_test[c] = 0.0

print(f'Features: {len(FUTURE_COLS)} future + {len(PAST_COLS)} past')

# ══════════════════════════════════════════════════════════════
# STEP 1: TRAIN TiDE v8
# ══════════════════════════════════════════════════════════════
print('\n' + '='*55)
print('STEP 1 — Train TiDE v8 (300 epochs, ~90 mins)')
print('='*55)

train_series, fut_cov_train, past_cov_train, group_ids = [], [], [], []
for (bid, eid), grp in gne.groupby(['product_brand_id','ecosystem_id']):
    grp = grp.sort_values('date').set_index('date')
    ts  = TimeSeries.from_series(grp[TARGET].fillna(0), freq='MS')
    fc  = TimeSeries.from_dataframe(grp[[c for c in FUTURE_COLS if c in grp.columns]].fillna(0), freq='MS')
    pc  = TimeSeries.from_dataframe(grp[[c for c in PAST_COLS   if c in grp.columns]].fillna(0), freq='MS')
    train_series.append(ts); fut_cov_train.append(fc); past_cov_train.append(pc)
    group_ids.append((bid, eid))

fut_cov_full = []
for i, (bid, eid) in enumerate(group_ids):
    test_grp = gne_test[(gne_test['product_brand_id']==bid) &
                        (gne_test['ecosystem_id']==eid)].sort_values('date').set_index('date')
    if len(test_grp) == 0:
        fut_cov_full.append(fut_cov_train[i]); continue
    fc_h = TimeSeries.from_dataframe(
        test_grp[[c for c in FUTURE_COLS if c in test_grp.columns]].fillna(0), freq='MS')
    fut_cov_full.append(fut_cov_train[i].append(fc_h))

t_split = [ts[:-VAL] for ts in train_series]
v_split = [ts[-VAL:]  for ts in train_series]
fc_t    = [fc[:-VAL]  for fc in fut_cov_train]
pc_t    = [pc[:-VAL]  for pc in past_cov_train]
vi      = [i for i,ts in enumerate(t_split) if len(ts) >= MIN_LEN]
t_s     = [t_split[i] for i in vi]; v_s     = [v_split[i] for i in vi]
fc_s    = [fc_t[i]    for i in vi]; pc_s    = [pc_t[i]    for i in vi]
gids    = [group_ids[i] for i in vi]
print(f'Training: {len(t_s)} series')

model_v8 = TiDEModel(
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
model_v8.fit(series=t_s, future_covariates=fc_s, past_covariates=pc_s, verbose=True)
model_v8.save(str(OUTPUT_TIDE / 'tide_v8_model'))
print('TiDE v8 training complete!')

# ── Validate TiDE v8 ──────────────────────────────────────────
print('\nValidating TiDE v8...')
fc_val = [fut_cov_train[i] for i in vi]
pc_val = [past_cov_train[i][:-VAL] for i in vi]
preds_v8 = model_v8.predict(n=VAL, series=t_s,
                             future_covariates=fc_val, past_covariates=pc_val)

y_true    = np.concatenate([ts.values().flatten() for ts in v_s])
y_pred_v8 = np.concatenate([ts.values().flatten() for ts in preds_v8])
brand_ids = np.concatenate([[gid[0]]*VAL for gid in gids])
steps     = np.tile(range(1, VAL+1), len(gids))

results_v8, brand_wapes_v8, step_wapes_v8 = full_scorecard(
    y_true, y_pred_v8, brand_ids, None, steps)

with open(OUTPUT_TIDE/'validation_scores_v5.json') as f: v5 = json.load(f)

print(f'\n=== TiDE v8 Results ===')
print(f'TiDE v5: {v5["WAPE (overall)"]*100:.2f}%')
print(f'TiDE v8: {results_v8["WAPE (overall)"]*100:.2f}%')

bw8 = brand_wapes_v8.reset_index()
bw8.columns = ['brand_id','wape_v8']
bw8 = bw8.merge(prod, left_on='brand_id', right_on='product_brand_id')
v5_brand = {'Xolarin':0.0066,'Hemvia':0.0080,'Ocretiva':0.0093,
            'Retivue':0.0566,'Vabyseal':0.0641,'Perjenta':0.0550,
            'Kadcynex':0.0805,'Phesgrox':0.0918}

print(f'\n{"Brand":<12} {"TiDE v5":>9} {"TiDE v8":>9} {"Change"}')
print('-'*45)
for _, row in bw8.iterrows():
    name = row['product_brand_name']
    v5w  = v5_brand.get(name,0)*100
    v8w  = row['wape_v8']*100
    print(f'  {name:<12} {v5w:>7.2f}%  {v8w:>7.2f}%  {v8w-v5w:>+6.2f}%')

json.dump({k:round(float(v),5) for k,v in results_v8.items()},
          open(OUTPUT_TIDE/'validation_scores_v8.json','w'), indent=2)

# Build v8 brand WAPE dict for ensemble decisions
v8_brand = dict(zip(bw8['product_brand_name'], bw8['wape_v8']))

# ── Generate TiDE v8 final predictions ────────────────────────
final_preds_v8 = model_v8.predict(n=6, series=train_series,
                                   future_covariates=fut_cov_full,
                                   past_covariates=past_cov_train)
v8_predictions = np.concatenate([ts.values().flatten() for ts in final_preds_v8]).clip(min=0)
make_submission(sub_base['row_id'].values, v8_predictions,
                OUTPUT_TIDE/'tide_v8_submission.csv')

# ══════════════════════════════════════════════════════════════
# STEP 2: PROPHET for Phesgrox + Vabyseal
# ══════════════════════════════════════════════════════════════
print('\n' + '='*55)
print('STEP 2 — Prophet: Phesgrox + Vabyseal')
print('='*55)

prophet_results = {}

for brand in ['Phesgrox','Vabyseal']:
    gne_b   = gne[gne['product_brand_name']==brand].copy()
    brand_id= prod[prod['product_brand_name']==brand]['product_brand_id'].values[0]

    monthly = gne_b.groupby('date_year_month')[TARGET].sum().reset_index()
    monthly['date'] = pd.to_datetime(monthly['date_year_month'].astype(str), format='%Y%m')
    monthly = monthly.sort_values('date')

    trn = monthly[~monthly['date_year_month'].isin(VAL_MONTHS)]
    val = monthly[monthly['date_year_month'].isin(VAL_MONTHS)]

    ts_trn = TimeSeries.from_dataframe(trn.set_index('date')[[TARGET]], freq='MS')
    ts_val = TimeSeries.from_dataframe(val.set_index('date')[[TARGET]], freq='MS')

    try:
        model_p = Prophet(seasonality_mode='additive',
                          yearly_seasonality=True,
                          weekly_seasonality=False,
                          daily_seasonality=False)
        model_p.fit(ts_trn)

        pred_val  = model_p.predict(n=VAL).values().flatten().clip(min=0)
        pred_test = model_p.predict(n=VAL+6).values().flatten()[-6:].clip(min=0)
        actual    = ts_val.values().flatten()

        w_prophet = np.sum(np.abs(actual-pred_val))/(np.sum(np.abs(actual))+1e-8)
        print(f'  {brand}: Prophet brand-level WAPE = {w_prophet*100:.2f}%')

        # Zone shares
        gne_trn = gne_b[~gne_b['date_year_month'].isin(VAL_MONTHS)]
        tot = gne_trn.groupby('date_year_month')[TARGET].transform('sum')
        gne_trn = gne_trn.copy()
        gne_trn['share'] = gne_trn[TARGET]/(tot+1e-8)
        zone_shares = gne_trn.groupby('ecosystem_id')['share'].mean()
        zone_shares = zone_shares/zone_shares.sum()

        # Compute optimal blend with TiDE v8 on validation
        # Get TiDE v8 zone predictions for this brand
        tide8_brand = {}
        for i, (bid, eid) in enumerate(gids):
            if bid == prod[prod['product_brand_name']==brand]['product_brand_id'].values[0]:
                tide8_brand[(bid,eid)] = preds_v8[gids.index((bid,eid))].values().flatten()

        # Build comparison: TiDE v8 vs blend
        zone_actual_all, tide8_all, prophet_all = [], [], []
        for step_idx, month in enumerate(VAL_MONTHS):
            prophet_zone = pred_val[step_idx] if step_idx < len(pred_val) else pred_val[-1]
            actual_month = gne_b[gne_b['date_year_month']==month]
            for eco_id, share in zone_shares.items():
                act = actual_month[actual_month['ecosystem_id']==eco_id][TARGET].sum()
                t8_pred = 0
                key = (brand_id, eco_id)
                if key in tide8_brand:
                    idx_in_gids = gids.index(key) if key in gids else -1
                    if idx_in_gids >= 0:
                        t8_pred = preds_v8[gids.index(key)].values().flatten()[step_idx]
                zone_actual_all.append(act)
                tide8_all.append(t8_pred if t8_pred > 0 else 0)
                prophet_all.append(float(prophet_zone) * float(share))

        zone_actual_all = np.array(zone_actual_all)
        tide8_all       = np.array(tide8_all)
        prophet_all     = np.array(prophet_all).clip(min=0)

        best_alpha, best_wape = 1.0, 999
        for alpha in np.arange(0, 1.05, 0.05):
            blended = alpha * tide8_all + (1-alpha) * prophet_all
            w = np.sum(np.abs(zone_actual_all-blended))/(np.sum(np.abs(zone_actual_all))+1e-8)
            if w < best_wape:
                best_wape  = w
                best_alpha = alpha

        w_v8_pure = np.sum(np.abs(zone_actual_all-tide8_all))/(np.sum(np.abs(zone_actual_all))+1e-8)
        print(f'  {brand}: TiDE v8 alone = {w_v8_pure*100:.2f}%  '
              f'Best blend ({best_alpha:.0%} v8 + {1-best_alpha:.0%} Prophet) = {best_wape*100:.2f}%')

        prophet_results[brand] = {
            'alpha': best_alpha, 'wape': best_wape,
            'pred_val': pred_val, 'pred_test': pred_test,
            'zone_shares': zone_shares
        }

    except Exception as e:
        print(f'  {brand}: Prophet failed ({e}) — using TiDE v8 only')

# ══════════════════════════════════════════════════════════════
# STEP 3: ETS for Kadcynex
# ══════════════════════════════════════════════════════════════
print('\n' + '='*55)
print('STEP 3 — ETS: Kadcynex')
print('='*55)

ets_results = {}
brand = 'Kadcynex'
brand_id = prod[prod['product_brand_name']==brand]['product_brand_id'].values[0]
gne_b    = gne[gne['product_brand_name']==brand].copy()

monthly = gne_b.groupby('date_year_month')[TARGET].sum().reset_index()
monthly['date'] = pd.to_datetime(monthly['date_year_month'].astype(str), format='%Y%m')
monthly = monthly.sort_values('date')
trn = monthly[~monthly['date_year_month'].isin(VAL_MONTHS)]
val = monthly[monthly['date_year_month'].isin(VAL_MONTHS)]
ts_trn = TimeSeries.from_dataframe(trn.set_index('date')[[TARGET]], freq='MS')
ts_val = TimeSeries.from_dataframe(val.set_index('date')[[TARGET]], freq='MS')

best_model, best_w = None, 999
for trend in ['additive','multiplicative',None]:
    for seasonal in ['additive',None]:
        try:
            kwargs = {'trend':trend,'seasonal':seasonal}
            if seasonal: kwargs['seasonal_periods'] = 12
            m = ExponentialSmoothing(**kwargs)
            m.fit(ts_trn)
            p = m.predict(n=VAL).values().flatten().clip(min=0)
            w = np.sum(np.abs(ts_val.values().flatten()-p))/(np.sum(np.abs(ts_val.values().flatten()))+1e-8)
            if w < best_w:
                best_w=w; best_model=m
        except: pass

if best_model:
    pred_val  = best_model.predict(n=VAL).values().flatten().clip(min=0)
    pred_test = best_model.predict(n=VAL+6).values().flatten()[-6:].clip(min=0)
    print(f'  Kadcynex: ETS brand-level WAPE = {best_w*100:.2f}%')

    gne_trn = gne_b[~gne_b['date_year_month'].isin(VAL_MONTHS)].copy()
    tot = gne_trn.groupby('date_year_month')[TARGET].transform('sum')
    gne_trn['share'] = gne_trn[TARGET]/(tot+1e-8)
    zone_shares = gne_trn.groupby('ecosystem_id')['share'].mean()
    zone_shares = zone_shares/zone_shares.sum()

    zone_actual_all, tide8_all, ets_all = [], [], []
    for step_idx, month in enumerate(VAL_MONTHS):
        ets_zone = pred_val[step_idx]
        actual_month = gne_b[gne_b['date_year_month']==month]
        for eco_id, share in zone_shares.items():
            act = actual_month[actual_month['ecosystem_id']==eco_id][TARGET].sum()
            key = (brand_id, eco_id)
            t8_pred = preds_v8[gids.index(key)].values().flatten()[step_idx] if key in gids else 0
            zone_actual_all.append(act)
            tide8_all.append(t8_pred)
            ets_all.append(float(ets_zone)*float(share))

    zone_actual_all = np.array(zone_actual_all)
    tide8_all       = np.array(tide8_all)
    ets_all         = np.array(ets_all).clip(min=0)

    best_alpha, best_wape = 1.0, 999
    for alpha in np.arange(0, 1.05, 0.05):
        blended = alpha*tide8_all + (1-alpha)*ets_all
        w = np.sum(np.abs(zone_actual_all-blended))/(np.sum(np.abs(zone_actual_all))+1e-8)
        if w < best_wape:
            best_wape=w; best_alpha=alpha

    w_v8_pure = np.sum(np.abs(zone_actual_all-tide8_all))/(np.sum(np.abs(zone_actual_all))+1e-8)
    print(f'  Kadcynex: TiDE v8 alone = {w_v8_pure*100:.2f}%  '
          f'Best blend ({best_alpha:.0%} v8 + {1-best_alpha:.0%} ETS) = {best_wape*100:.2f}%')

    ets_results[brand] = {'alpha':best_alpha,'wape':best_wape,
                           'pred_val':pred_val,'pred_test':pred_test,
                           'zone_shares':zone_shares}

# ══════════════════════════════════════════════════════════════
# STEP 4: FINAL HYBRID SUBMISSION
# ══════════════════════════════════════════════════════════════
print('\n' + '='*55)
print('STEP 4 — Building Final Hybrid Submission')
print('='*55)

# Load TiDE v5 + v6 for Retivue comparison
v5_sub = pd.read_csv('04_outputs/tide/tide_v5_submission.csv')
try:
    v6_sub = pd.read_csv('04_outputs/tide/tide_v6_submission.csv')
    has_v6 = True
except: has_v6 = False

v8_sub = pd.read_csv('04_outputs/tide/tide_v8_submission.csv')
v8_sub = v8_sub.merge(test_meta[['row_id','product_brand_name','ecosystem_id','date_year_month']],
                       on='row_id', how='left')

final_rows = []

for brand in sorted(gne['product_brand_name'].unique()):
    brand_id    = prod[prod['product_brand_name']==brand]['product_brand_id'].values[0]
    brand_test  = test_meta[test_meta['product_brand_name']==brand]
    v8_brand_sub = v8_sub[v8_sub['product_brand_name']==brand]

    # Determine base model per brand
    v8_wape = v8_brand.get(brand, 1.0)
    v5_wape = v5_brand.get(brand, 1.0)

    if brand in prophet_results:
        res = prophet_results[brand]
        alpha = res['alpha']
        model_label = f'TiDE v8 × {alpha:.0%} + Prophet × {1-alpha:.0%}'
    elif brand in ets_results:
        res = ets_results[brand]
        alpha = res['alpha']
        model_label = f'TiDE v8 × {alpha:.0%} + ETS × {1-alpha:.0%}'
    else:
        alpha = 1.0
        model_label = 'TiDE v8'

    for step_idx, month in enumerate(TEST_MONTHS):
        month_test = brand_test[brand_test['date_year_month']==month]
        month_v8   = v8_brand_sub[v8_brand_sub['date_year_month']==month]

        for _, row in month_test.iterrows():
            eco_id   = row['ecosystem_id']
            row_id   = row['row_id']
            v8_pred  = month_v8[month_v8['ecosystem_id']==eco_id]['forecast_units_eqv']
            v8_val   = float(v8_pred.values[0]) if len(v8_pred) > 0 else 0

            if alpha < 1.0 and brand in {**prophet_results, **ets_results}:
                res = prophet_results.get(brand, ets_results.get(brand))
                share     = float(res['zone_shares'].get(eco_id, 1/80))
                alt_pred  = float(res['pred_test'][step_idx]) * share
                final_pred = alpha * v8_val + (1-alpha) * alt_pred
            else:
                final_pred = v8_val

            final_rows.append({'row_id': row_id,
                                'forecast_units_eqv': max(round(float(final_pred),4), 0)})

final_sub = pd.DataFrame(final_rows).sort_values('row_id')
missing = set(sub_base['row_id']) - set(final_sub['row_id'])
if missing:
    fallback = v8_sub[v8_sub['row_id'].isin(missing)][['row_id','forecast_units_eqv']]
    final_sub = pd.concat([final_sub, fallback], ignore_index=True).sort_values('row_id')

final_sub.to_csv(OUTPUT_ENS / 'ensemble_final_submission.csv', index=False)
make_submission(final_sub['row_id'].values, final_sub['forecast_units_eqv'].values,
                OUTPUT_FINAL / 'final_submission.csv')

# ── Final summary ──────────────────────────────────────────────
print(f'\n{"="*60}')
print('FINAL MODEL COMPLETE')
print(f'{"="*60}')
print(f'\nTiDE v8 overall WAPE : {results_v8["WAPE (overall)"]*100:.2f}%')
print(f'TiDE v5 overall WAPE : {v5["WAPE (overall)"]*100:.2f}%')
print(f'\nBrand assignment:')
for brand in sorted(gne['product_brand_name'].unique()):
    if brand in prophet_results:
        res = prophet_results[brand]
        print(f'  {brand:<12}: TiDE v8 × {res["alpha"]:.0%} + Prophet × {1-res["alpha"]:.0%}'
              f' → {res["wape"]*100:.2f}%')
    elif brand in ets_results:
        res = ets_results[brand]
        print(f'  {brand:<12}: TiDE v8 × {res["alpha"]:.0%} + ETS × {1-res["alpha"]:.0%}'
              f' → {res["wape"]*100:.2f}%')
    else:
        print(f'  {brand:<12}: TiDE v8 → {v8_brand.get(brand,0)*100:.2f}%')

print(f'\nSaved: 04_outputs/ensemble/ensemble_final_submission.csv')
print(f'Saved: 04_outputs/final/final_submission.csv  ← submit this')
print(f'\nNext steps:')
print(f'  → Task B: Market Share Analysis (06_market_share.ipynb)')
print(f'  → Task C: GenAI Explanation Layer')
print(f'  → Presentation preparation')

json.dump({'tide_v8_wape': round(float(results_v8['WAPE (overall)']),5),
           'tide_v5_wape': round(float(v5['WAPE (overall)']),5),
           'brands': {b:{'wape':round(float(v8_brand.get(b,0)),5)} for b in v8_brand}},
          open(OUTPUT_ENS/'final_model_summary.json','w'), indent=2)
