"""
Task B v3 — GNE Portfolio Market Share
---------------------------------------
Approach:
  Multi-brand markets (ONC: 3 brands, OPH: 2 brands):
    share = brand_forecast / sum(all GNE brands in same market × zone × month)

  Single-brand markets (HEM, MS, RESP — 1 GNE brand each):
    share = GNE brand / (GNE brand + historical ETS competitor forecast)
    using historical fact_sales_monthly to get total market per zone

Output: 04_outputs/market_share/share_submission.csv
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path

ROOT  = Path(__file__).parent.parent
RAW   = ROOT / '01_input/raw'
FINAL = ROOT / '04_outputs/final'
OUT   = ROOT / '04_outputs/market_share'
OUT.mkdir(exist_ok=True)

HORIZON = [202501, 202502, 202503, 202504, 202505, 202506]
MULTI_BRAND_MARKETS = {'ONC', 'OPH'}   # multiple GNE brands compete within market
SINGLE_BRAND_MARKETS = {'HEM', 'MS', 'RESP'}  # 1 GNE brand, use historical share

print('=' * 60)
print('Task B v3 — GNE Portfolio Market Share')
print('=' * 60)

# ── Load data ─────────────────────────────────────────────────
test  = pd.read_csv(RAW / 'test_features.csv')
sales = pd.read_csv(RAW / 'fact_sales_monthly.csv')
sub   = pd.read_csv(FINAL / 'final_submission.csv')

# Join submission with test metadata
sub = sub.merge(
    test[['row_id', 'date_year_month', 'ecosystem_id', 'product_brand_id',
          'product_brand_name', 'market_code']],
    on='row_id', how='left'
)

print(f'\nSubmission rows: {len(sub):,}')
print(f'Brands: {sorted(sub.product_brand_name.unique())}')

# ── Part 1: Multi-brand markets (ONC, OPH) ────────────────────
print('\n[1] Multi-brand markets — GNE basket share')

multi = sub[sub['market_code'].isin(MULTI_BRAND_MARKETS)].copy()

basket = (
    multi.groupby(['date_year_month', 'ecosystem_id', 'market_code'])['forecast_units_eqv']
    .sum()
    .reset_index()
    .rename(columns={'forecast_units_eqv': 'gne_basket'})
)
multi = multi.merge(basket, on=['date_year_month', 'ecosystem_id', 'market_code'])
multi['forecast_share'] = (multi['forecast_units_eqv'] / multi['gne_basket'].clip(lower=1e-6)).clip(0, 1)

print(f'  Rows: {len(multi):,}')
for mkt in MULTI_BRAND_MARKETS:
    brands = sorted(multi[multi.market_code == mkt]['product_brand_name'].unique())
    avg_shares = multi[multi.market_code == mkt].groupby('product_brand_name')['forecast_share'].mean()
    print(f'  {mkt}: {brands}')
    for b, s in avg_shares.items():
        print(f'      {b}: avg share = {s:.1%}')

# ── Part 2: Single-brand markets — hist share projected forward ─
print('\n[2] Single-brand markets — historical share extrapolated')

gne_sales  = sales[sales['flag_competitor'] == 'N'].copy()
all_sales  = sales.copy()

hist_total = (
    all_sales.groupby(['date_year_month', 'ecosystem_id', 'market_code'])['iqvia_sales_qty_eqv']
    .sum()
    .reset_index()
    .rename(columns={'iqvia_sales_qty_eqv': 'total_market'})
)
hist_gne = gne_sales[['date_year_month', 'ecosystem_id', 'market_code',
                       'product_brand_id', 'product_brand_name', 'iqvia_sales_qty_eqv']].copy()
hist_gne = hist_gne.merge(hist_total, on=['date_year_month', 'ecosystem_id', 'market_code'])
hist_gne['hist_share'] = hist_gne['iqvia_sales_qty_eqv'] / (hist_gne['total_market'] + 1e-6)

# Use last 6 months of 2024 as the baseline share, then hold flat for 2025
recent = hist_gne[hist_gne['date_year_month'].between(202407, 202412)]
avg_share = (
    recent.groupby(['ecosystem_id', 'product_brand_id'])['hist_share']
    .mean()
    .reset_index()
    .rename(columns={'hist_share': 'avg_hist_share'})
)

single = sub[sub['market_code'].isin(SINGLE_BRAND_MARKETS)].copy()
single = single.merge(avg_share, on=['ecosystem_id', 'product_brand_id'], how='left')

# Fill any missing with brand-level mean
brand_fallback = avg_share.groupby('product_brand_id')['avg_hist_share'].mean()
single['avg_hist_share'] = single.apply(
    lambda r: r['avg_hist_share'] if pd.notna(r['avg_hist_share'])
    else brand_fallback.get(r['product_brand_id'], 0.25),
    axis=1
)
single['forecast_share'] = single['avg_hist_share'].clip(0, 1)

print(f'  Rows: {len(single):,}')
for brand in sorted(single.product_brand_name.unique()):
    avg_s = single[single.product_brand_name == brand]['forecast_share'].mean()
    mkt   = single[single.product_brand_name == brand]['market_code'].iloc[0]
    print(f'  {mkt} / {brand}: avg forecast share = {avg_s:.1%}')

# ── Combine and save ──────────────────────────────────────────
print('\n[3] Combining and saving...')

result = pd.concat([
    multi[['ecosystem_id', 'product_brand_id', 'date_year_month', 'forecast_share']],
    single[['ecosystem_id', 'product_brand_id', 'date_year_month', 'forecast_share']]
], ignore_index=True)

result = result.sort_values(['ecosystem_id', 'product_brand_id', 'date_year_month']).reset_index(drop=True)

print(f'  Total rows: {len(result):,}')
print(f'  Nulls: {result.isna().sum().sum()}')
print(f'  Share range: {result.forecast_share.min():.4f} – {result.forecast_share.max():.4f}')

result.to_csv(OUT / 'share_submission.csv', index=False)
print(f'  Saved: {OUT}/share_submission.csv')

# ── Summary table ─────────────────────────────────────────────
print('\n=== FINAL SHARE SUMMARY ===')
summary = result.merge(
    test[['ecosystem_id', 'product_brand_id', 'product_brand_name', 'market_code']].drop_duplicates(),
    on=['ecosystem_id', 'product_brand_id']
)
brand_avg = summary.groupby(['market_code', 'product_brand_name'])['forecast_share'].mean()
print(f'\n  {"Market":<6} {"Brand":<12} {"Avg Forecast Share (Jan-Jun 2025)"}')
print(f'  {"-"*50}')
for (mkt, brand), share in brand_avg.items():
    print(f'  {mkt:<6} {brand:<12}  {share:.1%}')

print('\n✅ Task B v3 complete')
print('   Multi-brand markets: GNE basket intra-market share')
print('   Single-brand markets: H2 2024 historical share held forward')
