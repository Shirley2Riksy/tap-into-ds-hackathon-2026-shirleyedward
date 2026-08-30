"""
run_tide_v5.py
TiDE v5 — trained on clean data with updated proven features.

What's different from TiDE v3:
  Data   : master_train_v3.csv (DDD normalised, outliers capped, smart lag fill)
  Features: updated selection — promotion now included (0.5% gain proven)
            Fourier seasonality added (0.7% gain, helps Month 6 gap)
  Epochs : 300 (same as v3)
  Lookback: 18 months (same as v3 — v4 proved 24 is worse)

Usage: python3 03_scripts/run_tide_v5.py
"""

import sys, warnings, json
sys.path.append('03_scripts')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from pathlib import Path
from evaluate import wape, macro_wape, full_scorecard
from utils import make_submission
from darts import TimeSeries
from darts.models import TiDEModel

OUTPUT = Path('04_outputs/tide')
FINAL  = Path('04_outputs/final')
OUTPUT.mkdir(exist_ok=True)
FINAL.mkdir(exist_ok=True)

print('=' * 60)
print('TiDE v5 — Clean Data + Updated Features')
print('=' * 60)
print('Data    : master_train_v3.csv (DDD normalised, outliers fixed)')
print('Features: promotion now included, Fourier seasonality added')
print('Epochs  : 300 | Lookback: 18 months')
print('=' * 60)

# ── Load clean data ────────────────────────────────────────────
print('\nLoading clean data...')
master   = pd.read_csv('01_input/processed/master_train_v3.csv', low_memory=False)
test     = pd.read_csv('01_input/processed/master_test_v3.csv',  low_memory=False)
gne      = master[master['flag_competitor']=='N'].copy()
gne_test = test.copy()
gne['date']      = pd.to_datetime(gne['date_year_month'].astype(str),      format='%Y%m')
gne_test['date'] = pd.to_datetime(gne_test['date_year_month'].astype(str), format='%Y%m')
print(f'Train: {len(gne):,} rows | Test: {len(gne_test):,} rows')

# ── Load updated feature selection ─────────────────────────────
print('\nLoading updated feature selection...')
with open('05_documents/final_feature_selection.json') as f:
    sel = json.load(f)
FUTURE_COLS = sel['FINAL_FUTURE_COLS']
PAST_COLS   = sel['FINAL_PAST_COLS']

# Fill missing
for c in FUTURE_COLS + PAST_COLS:
    if c not in gne.columns:      gne[c]      = 0
    gne[c] = gne[c].fillna(0)
for c in FUTURE_COLS:
    if c not in gne_test.columns: gne_test[c] = 0
    gne_test[c] = gne_test[c].fillna(0)

print(f'Future covariates: {len(FUTURE_COLS)} (includes promotion + Fourier)')
print(f'Past covariates  : {len(PAST_COLS)}')

# ── Build TimeSeries ───────────────────────────────────────────
print('\nBuilding TimeSeries...')
train_series, fut_cov_train, past_cov_train, group_ids = [], [], [], []

for (bid, eid), grp in gne.groupby(['product_brand_id','ecosystem_id']):
    grp = grp.sort_values('date').set_index('date')
    ts  = TimeSeries.from_series(grp['iqvia_sales_qty_eqv'].fillna(0), freq='MS')
    fc  = TimeSeries.from_dataframe(grp[[c for c in FUTURE_COLS if c in grp.columns]].fillna(0), freq='MS')
    pc  = TimeSeries.from_dataframe(grp[[c for c in PAST_COLS   if c in grp.columns]].fillna(0), freq='MS')
    train_series.append(ts)
    fut_cov_train.append(fc)
    past_cov_train.append(pc)
    group_ids.append((bid, eid))

print(f'Series: {len(train_series)} | Length: {min(len(t) for t in train_series)}-{max(len(t) for t in train_series)} months')

# Extend future covariates through horizon
fut_cov_full = []
for i, (bid, eid) in enumerate(group_ids):
    test_grp = gne_test[
        (gne_test['product_brand_id']==bid) &
        (gne_test['ecosystem_id']==eid)
    ].sort_values('date').set_index('date')
    if len(test_grp) == 0:
        fut_cov_full.append(fut_cov_train[i])
        continue
    fc_h = TimeSeries.from_dataframe(
        test_grp[[c for c in FUTURE_COLS if c in test_grp.columns]].fillna(0), freq='MS')
    fut_cov_full.append(fut_cov_train[i].append(fc_h))

# ── Split ──────────────────────────────────────────────────────
VAL_MONTHS = 6
INPUT_LEN  = 18
MIN_LEN    = INPUT_LEN + VAL_MONTHS

train_split    = [ts[:-VAL_MONTHS] for ts in train_series]
val_split      = [ts[-VAL_MONTHS:]  for ts in train_series]
fc_train_split = [fc[:-VAL_MONTHS]  for fc in fut_cov_train]
pc_train_split = [pc[:-VAL_MONTHS]  for pc in past_cov_train]

valid_idx      = [i for i,ts in enumerate(train_split) if len(ts) >= MIN_LEN]
train_split    = [train_split[i]    for i in valid_idx]
val_split      = [val_split[i]      for i in valid_idx]
fc_train_split = [fc_train_split[i] for i in valid_idx]
pc_train_split = [pc_train_split[i] for i in valid_idx]
group_ids_v    = [group_ids[i]      for i in valid_idx]

print(f'Training series: {len(train_split)} | Excluded: {640-len(train_split)} (too short)')

# ── Train ──────────────────────────────────────────────────────
print('\nConfiguring TiDE v5...')
model = TiDEModel(
    input_chunk_length=INPUT_LEN,
    output_chunk_length=6,
    num_encoder_layers=2,
    num_decoder_layers=2,
    decoder_output_dim=16,
    hidden_size=128,
    temporal_width_past=4,
    temporal_width_future=4,
    dropout=0.1,
    batch_size=64,
    n_epochs=300,
    add_encoders={
        'cyclic': {'future': ['month']},
        'datetime_attribute': {'future': ['month', 'quarter']},
    },
    random_state=42,
    pl_trainer_kwargs={'accelerator': 'cpu', 'enable_progress_bar': True},
)

print('Starting training (300 epochs, ~40-60 mins)...\n')
model.fit(
    series=train_split,
    future_covariates=fc_train_split,
    past_covariates=pc_train_split,
    verbose=True,
)
print('\nTraining complete!')
model.save(str(OUTPUT / 'tide_v5_model'))
print('Model saved.')

# ── Validate ───────────────────────────────────────────────────
print('\nValidating...')
fc_for_val = [fut_cov_train[i] for i in valid_idx]
pc_for_val = [past_cov_train[i][:-VAL_MONTHS] for i in valid_idx]

val_preds = model.predict(
    n=VAL_MONTHS, series=train_split,
    future_covariates=fc_for_val,
    past_covariates=pc_for_val,
)

y_true    = np.concatenate([ts.values().flatten() for ts in val_split])
y_pred    = np.concatenate([ts.values().flatten() for ts in val_preds])
brand_ids = np.concatenate([[gid[0]] * VAL_MONTHS for gid in group_ids_v])
steps     = np.tile(range(1, VAL_MONTHS+1), len(group_ids_v))

results, brand_wapes, step_wapes = full_scorecard(y_true, y_pred, brand_ids, None, steps)

print(f'\n=== ALL TiDE VERSIONS ===')
print(f'TM1 Baseline : 0.1370  (13.7%)  — old company model')
print(f'TiDE v5      : {results["WAPE (overall)"]:.4f}  ({results["WAPE (overall)"]*100:.1f}%)  — corrected DDD + features  <-- NEW')

print(f'\nHorizon breakdown:')
print(step_wapes.to_string())

with open(OUTPUT / 'validation_scores_v5.json', 'w') as f:
    json.dump({k: round(float(v), 5) for k, v in results.items()}, f, indent=2)
print('Scores saved.')

# ── Final Predictions ──────────────────────────────────────────
print('\nGenerating final predictions...')
final_preds = model.predict(
    n=6, series=train_series,
    future_covariates=fut_cov_full,
    past_covariates=past_cov_train,
)
predictions = np.concatenate([ts.values().flatten() for ts in final_preds])
predictions = np.clip(predictions, 0, None)

sub = pd.read_csv('01_input/raw/sample_submission.csv')
make_submission(sub['row_id'].values, predictions, OUTPUT / 'tide_v5_submission.csv')

print(f'\n{"="*60}')
print(f'TiDE v5 COMPLETE')
print(f'WAPE: {results["WAPE (overall)"]*100:.2f}%')
print(f'vs TM1 baseline: {(0.137 - results["WAPE (overall)"])/0.137*100:.1f}% better')
print(f'{"="*60}')
print('Next: apply MinTrace to TiDE v5 predictions')
