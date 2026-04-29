#!/usr/bin/env python3
"""
ShelbyFirst: Outcomes Data Table (standalone)
Run:    python shelbyfirst_table_chart.py
Output: outputs/submission_package/figures/charts/shelbyfirst_table_chart.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from project_paths import CHARTS_DIR, PROCESSED_DATA_DIR

# ── Palette ───────────────────────────────────────────────────────────────────
NAV = '#1b2a6b'
TBL = '#2e4296'
WHT = '#ffffff'
EVN = '#eef1fa'
GRY = '#dee3ef'
RED = '#c0392b'
GRN = '#1a7a35'
BLU = '#1a4db5'
DRK = '#1a1a2e'
SUB = '#8898cc'

# ── Model-driven projections ──────────────────────────────────────────────────
CLUSTER_ID = "Tennessee | Shelby County | cluster_77"

_data = PROCESSED_DATA_DIR

_panel  = pd.read_parquet(_data / 'enrichment_tract_year_panel.parquet')
_shelby = _panel[_panel['geoid'].astype(str).str.startswith('47157')].copy()
_shelby['income_k'] = _shelby['median_household_income'] / 1000

_YEARS = [2024, 2025, 2026, 2027]
UPTAKE = [0.0,  0.15, 0.35, 0.60]

_PEER = {
    'labor_force': 66.9, 'poverty':  5.5, 'income': 119.0,
    'insured':    94.0,  'no_veh':   4.2, 'dental':  32.1,
    'obesity':    32.5,  'diabetes': 11.3,
}

EFFECT = {
    'labor_force': 0.22,
    'poverty':     0.15,
    'income':      0.10,
    'insured':     0.40,
    'no_veh':      0.20,
    'dental':      0.35,
    'obesity':     0.22,
    'diabetes':    0.18,
}


def _ridge_trend(df, col, scale=1.0):
    d = df[['year', col]].dropna()
    pipe = Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=1.0))])
    pipe.fit(d[['year']].values, d[col].values)
    return (pipe.predict(np.array([[yr] for yr in _YEARS])) * scale).tolist()


def _gap_boost(baseline, peer, pillar, direction):
    gap = abs(peer - baseline)
    eff = EFFECT[pillar]
    return [baseline + direction * gap * eff * u for u in UPTAKE]


def _econ(df, col, scale, peer, pillar, direction):
    trend = _ridge_trend(df, col, scale)
    gap   = abs(peer - trend[0])
    eff   = EFFECT[pillar]
    result = [trend[0]]
    for i in range(1, 4):
        result.append(trend[i] + direction * gap * eff * UPTAKE[i])
    return result


_lfpr_v = _econ(_shelby, 'lfpr_16p',    100, _PEER['labor_force'], 'labor_force',  1)
_pov_v  = _econ(_shelby, 'poverty_rate', 100, _PEER['poverty'],    'poverty',     -1)
_inc_v  = _econ(_shelby, 'income_k',       1, _PEER['income'],     'income',       1)

_cs   = pd.read_parquet(_data / 'cluster_shortlist_with_healthcare.parquet')
_r77  = _cs[_cs['cluster_id'].astype(str) == CLUSTER_ID]
_ins0 = (1 - float(_r77.iloc[0]['uninsured_rate'])) * 100 \
        if len(_r77) and not pd.isna(_r77.iloc[0]['uninsured_rate']) else 82.0

_ins_v    = _gap_boost(_ins0, _PEER['insured'],  'insured',  1)
_noveh_v  = _gap_boost(13.8,  _PEER['no_veh'],   'no_veh',  -1)
_dental_v = _gap_boost(55.6,  _PEER['dental'],   'dental',  -1)
_obes_v   = _gap_boost(45.2,  _PEER['obesity'],  'obesity', -1)
_diab_v   = _gap_boost(21.6,  _PEER['diabetes'], 'diabetes',-1)

_IGS = [35.8, 36.8, 38.4, 40.5]

METRICS = [
    ('IGS Score',     1,  _IGS,      55.7,
     '#4545c0', '', '{:.1f}'),
    ('Labor Force %', 1,  _lfpr_v,   _PEER['labor_force'],
     '#1a8c5a', '', '{:.1f}%'),
    ('Insured %',     1,  _ins_v,    _PEER['insured'],
     '#c9a000', '', '{:.1f}%'),
    ('Med. Income',   1,  _inc_v,    _PEER['income'],
     '#e06000', '', '${:.1f}k'),
    ('Poverty %',    -1,  _pov_v,    _PEER['poverty'],
     '#8b1515', '', '{:.1f}%'),
    ('No-Veh HH %',  -1,  _noveh_v,  _PEER['no_veh'],
     '#cc1515', '', '{:.1f}%'),
    ('Dental Gap %', -1,  _dental_v, _PEER['dental'],
     '#1a8cd4', '', '{:.1f}%'),
    ('Obesity %',    -1,  _obes_v,   _PEER['obesity'],
     '#8bc820', '', '{:.1f}%'),
    ('Diabetes %',   -1,  _diab_v,   _PEER['diabetes'],
     '#30a040', '', '{:.1f}%'),
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9), dpi=130, facecolor=NAV)

# ── Header ────────────────────────────────────────────────────────────────────
fig.text(0.030, 0.965,
         'ShelbyFirst: Projected Outcomes',
         color=WHT, fontsize=22, fontweight='bold', va='top')
fig.text(0.030, 0.922,
         'ShelbyFirst gap-closing model (Move · Nourish · Connect)  ·  '
         'Uptake: 15% Pilot → 35% Expand → 60% Scale  ·  96 tracts  ·  pop. 273K',
         color=SUB, fontsize=10.5, va='top')

# ── Table ─────────────────────────────────────────────────────────────────────
ax_t = fig.add_axes([0.030, 0.115, 0.945, 0.775])
ax_t.set_xlim(0, 1)
ax_t.set_ylim(0, 1)
ax_t.axis('off')

COL_W = [0.280, 0.112, 0.108, 0.108, 0.108, 0.200]
COL_X = list(np.cumsum([0.0] + COL_W[:-1]))
COL_H = ['Metric', 'Now  (2024)', 'Year 1  2025', 'Year 2  2026', 'Year 3  2027', 'High IGS Shelby County']

ROW_H_HDR  = 0.095
ROW_H_SEC  = 0.070
ROW_H_DATA = 0.155

TABLE_ROWS = [
    ('hdr',  None),
    ('sec',  '── IGS SCORE  ·  Planning target — rises as a consequence of ShelbyFirst programs ─────────────────'),
    ('data', METRICS[0]),
    ('sec',  '── HEALTH METRICS  ·  ShelbyFirst Move (rides → dental access) / Nourish (garden → obesity, diabetes)'),
    ('data', METRICS[6]),
    ('data', METRICS[7]),
    ('data', METRICS[8]),
]

y        = 0.99
data_idx = 0

for rtype, rcontent in TABLE_ROWS:
    rh = {'hdr': ROW_H_HDR, 'sec': ROW_H_SEC, 'data': ROW_H_DATA}[rtype]

    if rtype == 'hdr':
        for i, (cx, cw, ch) in enumerate(zip(COL_X, COL_W, COL_H)):
            ax_t.add_patch(Rectangle((cx, y - rh), cw, rh,
                                     facecolor=NAV, edgecolor='#3a4ea0',
                                     linewidth=0.6, zorder=2))
            xt = cx + 0.015 if i == 0 else cx + cw / 2
            fs = 10.5 if i == len(COL_H) - 1 else 13
            ax_t.text(xt, y - rh / 2, ch,
                      ha='left' if i == 0 else 'center', va='center',
                      fontsize=fs, color=WHT, fontweight='bold')

    elif rtype == 'sec':
        ax_t.add_patch(Rectangle((0, y - rh), 1.0, rh,
                                  facecolor=TBL, edgecolor='none', zorder=2))
        ax_t.text(0.015, y - rh / 2, rcontent,
                  ha='left', va='center', fontsize=9,
                  color='#aac8ff', style='italic')

    else:
        tbl_lbl, direction, vals, peer, color, _, fmt = rcontent
        bg = EVN if data_idx % 2 == 1 else WHT
        data_idx += 1

        ax_t.add_patch(Rectangle((0, y - rh), 1.0, rh,
                                  facecolor=bg, edgecolor=GRY,
                                  linewidth=0.4, zorder=1))

        # Left color accent bar
        ax_t.add_patch(Rectangle((0, y - rh), 0.006, rh,
                                  facecolor=color, edgecolor='none', zorder=3))

        for cx in COL_X[1:]:
            ax_t.plot([cx, cx], [y - rh, y], color=GRY, linewidth=0.5, zorder=2)

        ax_t.text(COL_X[0] + 0.018, y - rh / 2, tbl_lbl,
                  ha='left', va='center', fontsize=14, color=DRK, fontweight='semibold')

        ax_t.text(COL_X[1] + COL_W[1] / 2, y - rh / 2, fmt.format(vals[0]),
                  ha='center', va='center', fontsize=14, color=RED, fontweight='bold')

        ax_t.text(COL_X[2] + COL_W[2] / 2, y - rh / 2, fmt.format(vals[1]),
                  ha='center', va='center', fontsize=14, color=DRK)

        ax_t.text(COL_X[3] + COL_W[3] / 2, y - rh / 2, fmt.format(vals[2]),
                  ha='center', va='center', fontsize=14, color=DRK)

        ax_t.text(COL_X[4] + COL_W[4] / 2, y - rh / 2, fmt.format(vals[3]),
                  ha='center', va='center', fontsize=15, color=GRN, fontweight='bold')

        ax_t.text(COL_X[5] + COL_W[5] / 2, y - rh / 2, fmt.format(peer),
                  ha='center', va='center', fontsize=14, color=BLU, fontweight='bold')

    y -= rh

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(
    0.030, 0.012,
    'Model: improvement = (peer gap) × program effectiveness × uptake  ·  '
    'Econ metrics add Ridge year-trend as counterfactual baseline  ·  '
    'Effectiveness grounded in Move (rides), Nourish (garden), Connect (app/referrals) pillars  ·  PDF slides 6–8',
    color=SUB, fontsize=8, va='bottom',
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = CHARTS_DIR / 'shelbyfirst_table_chart.png'
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=NAV, pad_inches=0.2)
print(f'Saved: {out}')
plt.close()
