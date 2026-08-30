"""
run_features_v7.py
Builds master_train_v7.csv / master_test_v7.csv with all new features.

New features added:
  1. Brand-specific seasonal index (within-year detrended, strong evidence only)
  2. new_rx_pct = new_rx_count / total_rx_count  (market penetration signal)
  3. prior_auth_delta = month-over-month change in prior_auth_required
  4. prior_auth_delta_3m = 3-month change (captures slower policy shifts)
  5. pseudo_tm1 = simulated naive forecast (training) / actual TM1 (test)
  6. tm1_uncertainty = (TM1_max - TM1_min) / TM1_point  (confidence band)
  7. prevalence_volume + incidence_volume (unused epi columns)

Usage: python3 03_scripts/run_features_v7.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROCESSED = Path('01_input/processed')
print('='*60)
print('Building Feature Set v7')
print('='*60)

# ── Load base data ─────────────────────────────────────────────
master = pd.read_csv(PROCESSED / 'master_train_v4.csv', low_memory=False)
test   = pd.read_csv(PROCESSED / 'master_test_v4.csv',  low_memory=False)
TARGET = 'iqvia_sales_qty_eqv'

gne      = master[master['flag_competitor']=='N'].copy()
gne_test = test.copy()

# ── Load raw files ─────────────────────────────────────────────
sales  = pd.read_csv('01_input/raw/fact_sales_monthly.csv')
payer  = pd.read_csv('01_input/raw/fact_payer_access_monthly.csv')
tm1    = pd.read_csv('01_input/raw/fact_internal_forecast.csv')
epi    = pd.read_csv('01_input/raw/fact_epidemiology_yearly.csv')
prod   = pd.read_csv('01_input/raw/dim_product.csv')

VAL_MONTHS = [202407,202408,202409,202410,202411,202412]
gne_trn    = gne[~gne['date_year_month'].isin(VAL_MONTHS)].copy()

print(f'Train: {len(gne):,} rows | Test: {len(gne_test):,} rows')

# ══════════════════════════════════════════════════════════════
# FEATURE 1: Brand-Specific Seasonal Index (within-year detrended)
# ══════════════════════════════════════════════════════════════
print('\n1. Brand-specific seasonal indices (within-year detrended)...')

STRONG_SEASONAL = {'Retivue', 'Hemvia', 'Xolarin'}
WEAK_SEASONAL   = {'Phesgrox', 'Vabyseal'}
MIXED_BRANDS    = {'Kadcynex', 'Ocretiva', 'Perjenta'}

gne_trn['year']  = gne_trn['date_year_month'] // 100
gne_trn['month'] = gne_trn['date_year_month'] % 100

brand_seasonal = {}
for brand in sorted(gne['product_brand_name'].unique()):
    df_b = gne_trn[gne_trn['product_brand_name']==brand].copy()

    if brand in MIXED_BRANDS:
        brand_seasonal[brand] = {mo: 1.0 for mo in range(1,13)}
        print(f'   {brand:<12}: flat (mixed evidence — no seasonal correction)')
        continue

    monthly_ratios = {mo: [] for mo in range(1,13)}
    for year, grp_y in df_b.groupby('year'):
        if grp_y['month'].nunique() < 10: continue
        yr_avg = grp_y[TARGET].mean()
        if yr_avg < 1e-6: continue
        for mo, grp_mo in grp_y.groupby('month'):
            monthly_ratios[mo].append(grp_mo[TARGET].mean() / yr_avg)

    idx = {mo: (np.mean(v) if v else 1.0) for mo, v in monthly_ratios.items()}
    mean_idx = np.mean(list(idx.values()))
    idx = {mo: v/mean_idx for mo, v in idx.items()}
    brand_seasonal[brand] = idx

    h1 = np.mean([idx[m] for m in range(1,7)])
    h2 = np.mean([idx[m] for m in range(7,13)])
    tag = 'STRONG' if brand in STRONG_SEASONAL else 'WEAK'
    print(f'   {brand:<12} [{tag}] H1={h1:.3f} H2={h2:.3f} '
          f'→ {"H2 higher ↑" if h2>h1 else "H1 higher ↓"}')

# Apply seasonal features
for df, label in [(gne, 'train'), (gne_test, 'test')]:
    df['month_tmp'] = df['date_year_month'] % 100
    df['brand_seasonal_index'] = df.apply(
        lambda r: brand_seasonal.get(r['product_brand_name'], {}).get(r['month_tmp'], 1.0), axis=1)
    h1 = df['month_tmp'].between(1,6)
    df['is_h2'] = (~h1).astype(int)
    df['brand_h2_premium'] = df['product_brand_name'].map({
        b: np.mean([v for m,v in idx.items() if m>6]) /
           np.mean([v for m,v in idx.items() if m<=6])
        for b, idx in brand_seasonal.items()
    })
    df.drop(columns=['month_tmp'], inplace=True)

# ══════════════════════════════════════════════════════════════
# FEATURE 2: New RX Percentage (market penetration signal)
# ══════════════════════════════════════════════════════════════
print('\n2. New RX percentage (market penetration signal)...')

gne_sales = sales[sales['flag_competitor']=='N'].copy()
gne_sales['data_provider_name'] = gne_sales['data_provider_name'].str.upper()
rx_monthly = gne_sales.groupby(['date_year_month','ecosystem_id','product_brand_id'])[
    ['iqvia_sales_new_rx_count','iqvia_sales_tot_rx_count']].sum().reset_index()
rx_monthly['new_rx_pct'] = (rx_monthly['iqvia_sales_new_rx_count'] /
                             (rx_monthly['iqvia_sales_tot_rx_count'] + 1e-8))
rx_monthly['new_rx_pct_lag1'] = rx_monthly.groupby(
    ['ecosystem_id','product_brand_id'])['new_rx_pct'].shift(1).fillna(0)
rx_monthly['new_rx_pct_lag3'] = rx_monthly.groupby(
    ['ecosystem_id','product_brand_id'])['new_rx_pct'].shift(3).fillna(0)

for df in [gne, gne_test]:
    df_merged = df.merge(rx_monthly[['date_year_month','ecosystem_id','product_brand_id',
                                      'new_rx_pct','new_rx_pct_lag1','new_rx_pct_lag3']],
                         on=['date_year_month','ecosystem_id','product_brand_id'], how='left')
    for col in ['new_rx_pct','new_rx_pct_lag1','new_rx_pct_lag3']:
        df[col] = df_merged[col].fillna(df_merged[col].median())
    print(f'   Added new_rx_pct: mean={df["new_rx_pct"].mean():.3f}')

# ══════════════════════════════════════════════════════════════
# FEATURE 3: Prior Auth Delta (rate of change in access restrictions)
# ══════════════════════════════════════════════════════════════
print('\n3. Prior auth delta (month-over-month change in access restrictions)...')

payer_clean = payer.copy()
payer_clean = payer_clean.sort_values(['product_brand_id','ecosystem_id','date_year_month'])
grp_pay = payer_clean.groupby(['product_brand_id','ecosystem_id'])

payer_clean['prior_auth_delta'] = grp_pay['pct_prior_auth_required'].diff().fillna(0)
payer_clean['prior_auth_delta_3m'] = grp_pay['pct_prior_auth_required'].diff(3).fillna(0)
payer_clean['pref_tier_delta'] = grp_pay['pct_preferred'].diff().fillna(0)
payer_clean['pref_tier_delta_3m'] = grp_pay['pct_preferred'].diff(3).fillna(0)

DELTA_COLS = ['prior_auth_delta','prior_auth_delta_3m','pref_tier_delta','pref_tier_delta_3m']

for df in [gne, gne_test]:
    df_merged = df.merge(payer_clean[['date_year_month','ecosystem_id','product_brand_id']+DELTA_COLS],
                         on=['date_year_month','ecosystem_id','product_brand_id'], how='left')
    for col in DELTA_COLS:
        df[col] = df_merged[col].fillna(0)

print(f'   Added: {DELTA_COLS}')

# ══════════════════════════════════════════════════════════════
# FEATURE 4: Pseudo-TM1 Forecast (training) / Actual TM1 (test)
# ══════════════════════════════════════════════════════════════
print('\n4. TM1 forecast features...')

# For TEST: use actual TM1 predictions
tm1_test = tm1.rename(columns={
    'gross_point_estimate': 'tm1_forecast',
    'gross_minimum':        'tm1_min',
    'gross_maximum':        'tm1_max'
})
tm1_test['tm1_uncertainty'] = (
    (tm1_test['tm1_max'] - tm1_test['tm1_min']) /
    (tm1_test['tm1_forecast'].abs() + 1e-8)
)

gne_test = gne_test.merge(
    tm1_test[['date_year_month','ecosystem_id','product_brand_id',
              'tm1_forecast','tm1_min','tm1_max','tm1_uncertainty']],
    on=['date_year_month','ecosystem_id','product_brand_id'], how='left')

for col in ['tm1_forecast','tm1_min','tm1_max','tm1_uncertainty']:
    gne_test[col] = gne_test[col].fillna(0)

# For TRAINING: simulate pseudo-TM1 using 6-month lag + rolling trend
# This mimics a naive model: "last 6 months value × local growth rate"
print('   Building pseudo-TM1 for training data...')
gne_sorted = gne.sort_values(['product_brand_id','ecosystem_id','date_year_month'])
grp_train  = gne_sorted.groupby(['product_brand_id','ecosystem_id'])

gne_sorted['lag_6']         = grp_train[TARGET].shift(6)
gne_sorted['lag_12']        = grp_train[TARGET].shift(12)
gne_sorted['roll_6_growth'] = (grp_train[TARGET].shift(1).rolling(6,min_periods=2).mean() /
                                (grp_train[TARGET].shift(7).rolling(6,min_periods=2).mean() + 1e-8))

# Pseudo-TM1 = lag6 × 6-month growth rate (naive carry-forward with trend)
gne_sorted['tm1_forecast']   = (gne_sorted['lag_6'] * gne_sorted['roll_6_growth']).clip(lower=0).fillna(0)
gne_sorted['tm1_min']        = (gne_sorted['tm1_forecast'] * 0.85).fillna(0)  # -15% range
gne_sorted['tm1_max']        = (gne_sorted['tm1_forecast'] * 1.15).fillna(0)  # +15% range
gne_sorted['tm1_uncertainty'] = 0.30  # constant 30% uncertainty for pseudo-TM1

# Put back into gne
for col in ['tm1_forecast','tm1_min','tm1_max','tm1_uncertainty']:
    gne[col] = gne_sorted[col].values

print(f'   Training pseudo-TM1: mean={gne["tm1_forecast"].mean():.1f} units')
print(f'   Test actual TM1:     mean={gne_test["tm1_forecast"].mean():.1f} units')

# ══════════════════════════════════════════════════════════════
# FEATURE 5: Unused Epidemiology Columns
# ══════════════════════════════════════════════════════════════
print('\n5. Adding unused epidemiology columns...')

epi_extra = epi[['ecosystem_id','year','prevalence_volume','incidence_volume',
                  'county_census_population_count']].copy()
epi_extra['date_year'] = epi_extra['year']

for df in [gne, gne_test]:
    df['date_year_tmp'] = df['date_year_month'] // 100
    df_m = df.merge(epi_extra[['ecosystem_id','date_year','prevalence_volume',
                                 'incidence_volume','county_census_population_count']],
                    left_on=['ecosystem_id','date_year_tmp'],
                    right_on=['ecosystem_id','date_year'], how='left')
    for col in ['prevalence_volume','incidence_volume','county_census_population_count']:
        df[col] = df_m[col].fillna(df_m[col].median())
    df.drop(columns=['date_year_tmp'], inplace=True, errors='ignore')

print(f'   Added: prevalence_volume, incidence_volume, county_census_population_count')

# ══════════════════════════════════════════════════════════════
# SUMMARY & SAVE
# ══════════════════════════════════════════════════════════════
NEW_FEATURES = [
    'brand_seasonal_index','is_h2','brand_h2_premium',
    'new_rx_pct','new_rx_pct_lag1','new_rx_pct_lag3',
    'prior_auth_delta','prior_auth_delta_3m',
    'pref_tier_delta','pref_tier_delta_3m',
    'tm1_forecast','tm1_min','tm1_max','tm1_uncertainty',
    'prevalence_volume','incidence_volume','county_census_population_count'
]

print(f'\n{"="*60}')
print(f'New features added: {len(NEW_FEATURES)}')
for f in NEW_FEATURES:
    train_ok = f in gne.columns
    test_ok  = f in gne_test.columns
    print(f'   {f:<35} train={"✅" if train_ok else "❌"}  test={"✅" if test_ok else "❌"}')

# Final fill
for col in NEW_FEATURES:
    if col in gne.columns:      gne[col]      = gne[col].fillna(0)
    if col in gne_test.columns: gne_test[col] = gne_test[col].fillna(0)

gne.to_csv(PROCESSED / 'master_train_v7.csv', index=False)
gne_test.to_csv(PROCESSED / 'master_test_v7.csv', index=False)

print(f'\n{"="*60}')
print('FEATURE BUILD v7 COMPLETE')
print(f'{"="*60}')
print(f'Saved: master_train_v7.csv ({len(gne):,} rows, {len(gne.columns)} cols)')
print(f'Saved: master_test_v7.csv  ({len(gne_test):,} rows, {len(gne_test.columns)} cols)')
print(f'\nKey features for problem brands:')
print(f'  new_rx_pct       → Phesgrox/Vabyseal still acquiring patients (growth signal)')
print(f'  prior_auth_delta → Captures sudden payer access changes (volatility driver)')
print(f'  tm1_forecast     → Domain knowledge from commercial team (directional anchor)')
print(f'  brand_seasonal   → H2 seasonality for Retivue (3-year strong evidence)')
print(f'\nNext: python3 03_scripts/run_tide_v8.py  (update to use master_train_v7.csv)')
