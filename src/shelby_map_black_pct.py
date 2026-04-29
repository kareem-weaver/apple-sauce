#!/usr/bin/env python3
"""
Shelby County 3-group map with Black population % overlay.

Groups:
  1) Selected low-growth community  (96 tracts from GeoJSON)
  2) Selected high-growth comparator (highest-IGS contiguous Shelby cluster outside low-growth)
  3) Rest of Shelby County

Overlay: % Black or African American alone (ACS 2022 5-year estimates)

Run:  python shelby_map_black_pct.py
Output: data/processed/presentation_ready/shelby_map_black_pct.png
"""

import os
import warnings
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_dir  = Path(os.path.dirname(os.path.abspath(__file__)))
DATA  = _dir / "data"
PROC  = DATA / "processed"
PRES  = PROC / "presentation_ready"
RAW   = DATA / "raw"

SHP_PATH      = RAW  / "tl_2024_state_tracts/tl_2024_47_tract/tl_2024_47_tract.shp"
LOW_GRW_PATH  = PRES / "selected_low_growth_96_tracts.geojson"
OUT_PATH      = PRES / "shelby_map_black_pct.png"


# ── 1. Load geometries ────────────────────────────────────────────────────────
print("Loading geometries...")
tn_tracts = gpd.read_file(SHP_PATH)

# Normalize GEOID to 11-digit string
tn_tracts["geoid"] = (
    tn_tracts["STATEFP"].str.zfill(2)
    + tn_tracts["COUNTYFP"].str.zfill(3)
    + tn_tracts["TRACTCE"].str.zfill(6)
)

shelby = tn_tracts[
    (tn_tracts["STATEFP"] == "47") & (tn_tracts["COUNTYFP"] == "157")
].copy()
shelby = shelby.to_crs(3857)
print(f"  Shelby County tracts: {len(shelby)}")

low_growth_gdf = gpd.read_file(LOW_GRW_PATH).copy()
low_growth_gdf["geoid"] = low_growth_gdf["geoid"].astype(str).str.zfill(11)
low_growth_geoids = set(low_growth_gdf["geoid"].dropna())
print(f"  Low-growth tracts:    {len(low_growth_geoids)}")


# ── 2. Fetch ACS 2022 race + income for all Shelby tracts ────────────────────
print("Fetching ACS 2022 data from Census API...")
ACS_VARS = "B02001_001E,B02001_003E,B19013_001E"   # total pop, Black alone, median HH income
ACS_URL  = "https://api.census.gov/data/2022/acs/acs5"
params = {
    "get":  ACS_VARS,
    "for":  "tract:*",
    "in":   "state:47 county:157",
}

resp = requests.get(ACS_URL, params=params, timeout=60)
resp.raise_for_status()

data_rows = resp.json()
cols = data_rows[0]
acs = pd.DataFrame(data_rows[1:], columns=cols)
acs["geoid"] = acs["state"].str.zfill(2) + acs["county"].str.zfill(3) + acs["tract"].str.zfill(6)

acs["pop_total"]    = pd.to_numeric(acs["B02001_001E"], errors="coerce")
acs["pop_black"]    = pd.to_numeric(acs["B02001_003E"], errors="coerce")
acs["med_income"]   = pd.to_numeric(acs["B19013_001E"], errors="coerce")

# Black share (0–100)
acs["pct_black"] = np.where(
    acs["pop_total"] > 0,
    (acs["pop_black"] / acs["pop_total"] * 100).clip(0, 100),
    np.nan,
)

print(f"  ACS tracts returned: {len(acs)}")
print(f"  Black % range: {acs['pct_black'].min():.1f}%–{acs['pct_black'].max():.1f}%")


# ── 3. Classify high-growth comparator ───────────────────────────────────────
# The notebook's clustering algorithm identified contiguous, high-IGS tracts in
# eastern Shelby County as the comparator. We approximate this by taking the
# non-low-growth tracts in the top income quartile, then picking the spatially
# contiguous group that maximises average income.

non_low = acs[~acs["geoid"].isin(low_growth_geoids)].copy()
income_75th = non_low["med_income"].quantile(0.72)   # ~top 28% matches map visually
print(f"  Income threshold for high-growth: ${income_75th:,.0f}")

high_growth_candidates = set(
    non_low.loc[non_low["med_income"] >= income_75th, "geoid"]
)

# Keep only the largest spatially contiguous group among candidates
candidate_geo = shelby[shelby["geoid"].isin(high_growth_candidates)].copy()

# Build adjacency list from touches
left  = candidate_geo[["geoid", "geometry"]].rename(columns={"geoid": "g1"})
right = candidate_geo[["geoid", "geometry"]].rename(columns={"geoid": "g2"})
adj   = gpd.sjoin(left, right, how="inner", predicate="touches")
adj   = adj[adj["g1"] != adj["g2"]][["g1", "g2"]].drop_duplicates()

# BFS to find connected components
from collections import deque, defaultdict

graph = defaultdict(set)
for _, row in adj.iterrows():
    graph[row["g1"]].add(row["g2"])
    graph[row["g2"]].add(row["g1"])

all_nodes = set(high_growth_candidates)
visited   = set()
components = []
for node in all_nodes:
    if node in visited:
        continue
    comp = set()
    q = deque([node])
    while q:
        n = q.popleft()
        if n in visited:
            continue
        visited.add(n)
        comp.add(n)
        q.extend(graph[n] - visited)
    components.append(comp)

# Pick the component with the highest average income
def avg_income(comp):
    vals = acs.loc[acs["geoid"].isin(comp), "med_income"].dropna()
    return vals.mean() if len(vals) > 0 else 0

best_component = max(components, key=lambda c: (len(c), avg_income(c)))
high_growth_geoids = best_component
print(f"  High-growth comparator tracts: {len(high_growth_geoids)}")


# ── 4. Assign groups ──────────────────────────────────────────────────────────
def assign_group(g):
    if g in low_growth_geoids:
        return "low_growth"
    elif g in high_growth_geoids:
        return "high_growth"
    return "rest"

shelby["group"] = shelby["geoid"].map(assign_group)
shelby = shelby.merge(acs[["geoid", "pct_black"]], on="geoid", how="left")

print("\nGroup breakdown:")
print(shelby["group"].value_counts().to_string())
print(f"  Missing Black %: {shelby['pct_black'].isna().sum()}")


# ── 5. Plot ───────────────────────────────────────────────────────────────────
print("\nRendering map...")
fig, ax = plt.subplots(1, 1, figsize=(13, 11))
fig.patch.set_facecolor("#f8f8f8")
ax.set_facecolor("#dce9f5")

# -- Continuous fill: Black population %
cmap = LinearSegmentedColormap.from_list(
    "black_pct",
    ["#fef0d9", "#fc8d59", "#b30000"],   # light-yellow → orange → dark-red
)
norm = Normalize(vmin=0, vmax=100)

shelby.plot(
    ax=ax,
    column="pct_black",
    cmap=cmap,
    norm=norm,
    edgecolor="#bbbbbb",
    linewidth=0.35,
    missing_kwds={"color": "#dddddd", "edgecolor": "#bbbbbb", "linewidth": 0.35},
    zorder=2,
)

# -- Group boundary outlines (thick, colored)
GROUP_STYLES = {
    "low_growth":  {"color": "#e74c3c", "linewidth": 3.2, "linestyle": "-",  "zorder": 4},
    "high_growth": {"color": "#27ae60", "linewidth": 3.2, "linestyle": "-",  "zorder": 4},
    "rest":        {"color": "#7f8c8d", "linewidth": 1.4, "linestyle": "--", "zorder": 3},
}

for grp, style in GROUP_STYLES.items():
    outline = shelby[shelby["group"] == grp].dissolve()
    if outline.empty:
        continue
    outline.boundary.plot(
        ax=ax,
        color=style["color"],
        linewidth=style["linewidth"],
        linestyle=style["linestyle"],
        zorder=style["zorder"],
    )

# -- Outer county border
shelby.dissolve().boundary.plot(ax=ax, color="#2c3e50", linewidth=1.6, zorder=5)

# -- Group labels
LABEL_CFG = {
    "low_growth":  ("Low-growth\nselected community",  "#c0392b"),
    "high_growth": ("High-growth\ncomparator",          "#1e8449"),
    "rest":        ("Rest of\nShelby County",           "#4a4a4a"),
}

for grp, (label, color) in LABEL_CFG.items():
    sub = shelby[shelby["group"] == grp]
    if sub.empty:
        continue
    pt = sub.dissolve().geometry.representative_point().iloc[0]
    ax.text(
        pt.x, pt.y, label,
        ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="white",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=color,
            edgecolor="white",
            alpha=0.92,
            linewidth=1.2,
        ),
        zorder=6,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

# -- Colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02, shrink=0.72)
cbar.set_label("% Black or African American\n(ACS 2022 5-year estimate)", fontsize=10)
cbar.set_ticks([0, 25, 50, 75, 100])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

# -- Group legend patches
legend_patches = [
    mpatches.Patch(facecolor="none", edgecolor="#e74c3c", linewidth=3,
                   label="Low-growth selected community"),
    mpatches.Patch(facecolor="none", edgecolor="#27ae60", linewidth=3,
                   label="High-growth comparator"),
    mpatches.Patch(facecolor="none", edgecolor="#7f8c8d", linewidth=2,
                   linestyle="dashed", label="Rest of Shelby County"),
]
ax.legend(
    handles=legend_patches,
    title="Community group\n(boundary color)",
    loc="lower left",
    frameon=True,
    framealpha=0.92,
    fontsize=9.5,
    title_fontsize=10,
)

ax.set_title(
    "Shelby County: 3-group community map\nwith % Black or African American by tract",
    fontsize=16,
    fontweight="bold",
    pad=14,
)
ax.set_axis_off()
plt.tight_layout(pad=1.2)

PRES.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")
plt.show()
