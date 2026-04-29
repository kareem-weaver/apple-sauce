#!/usr/bin/env python3
"""
Shelby County: selected low-growth community and Black population share by tract.

High-growth comparator is identified using the same methodology as final_model2.ipynb:
  - Spatially-constrained Ward agglomerative clustering (N=6 communities)
  - Connectivity matrix built from tract adjacency (touching polygons)
  - Features: ACS economic indicators (income, poverty, unemployment, LFPR)
    as proxies for IGS Economy pillar
  - Best non-low-growth community = highest composite rank across income, poverty,
    unemployment, and LFPR (mirrors notebook Cells 8F/8G percentile-ranking)

Run:    python shelby_map_black_pct.py
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
from scipy.sparse import coo_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_dir         = Path(os.path.dirname(os.path.abspath(__file__)))
SHP_PATH     = _dir / "data/raw/tl_2024_state_tracts/tl_2024_47_tract/tl_2024_47_tract.shp"
LOW_GRW_PATH = _dir / "data/processed/presentation_ready/selected_low_growth_96_tracts.geojson"
OUT_PATH     = _dir / "data/processed/presentation_ready/shelby_map_black_pct.png"
CLUSTER_CACHE = _dir / "data/processed/shelby_full_tract_cluster_map.geojson"


# ─────────────────────────────────────────────────────────────────────────────
# ACS data fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_acs(year: int = 2022) -> pd.DataFrame:
    """
    Fetch ACS 5-year estimates for all Shelby County tracts.
    Variables: race, income, poverty, unemployment, LFPR.
    """
    print(f"  Fetching ACS {year} 5-year estimates...")
    vars_needed = ",".join([
        "B02001_001E",   # total population
        "B02001_003E",   # Black or African American alone
        "B19013_001E",   # median household income
        "B17001_001E",   # population for poverty determination
        "B17001_002E",   # below poverty level
        "B23025_002E",   # civilian noninstitutional population 16+ (LFPR denominator)
        "B23025_003E",   # civilian labor force (LFPR numerator)
        "B23025_005E",   # unemployed
    ])
    r = requests.get(
        f"https://api.census.gov/data/{year}/acs/acs5",
        params={"get": vars_needed, "for": "tract:*", "in": "state:47 county:157"},
        timeout=60,
    )
    r.raise_for_status()
    rows = r.json()
    df   = pd.DataFrame(rows[1:], columns=rows[0])

    df["geoid"]       = df["state"].str.zfill(2) + df["county"].str.zfill(3) + df["tract"].str.zfill(6)
    df["pop_total"]   = pd.to_numeric(df["B02001_001E"], errors="coerce").clip(lower=0)
    df["pop_black"]   = pd.to_numeric(df["B02001_003E"], errors="coerce").clip(lower=0)
    df["med_income"]  = pd.to_numeric(df["B19013_001E"], errors="coerce")
    df["pov_denom"]   = pd.to_numeric(df["B17001_001E"], errors="coerce").clip(lower=0)
    df["pov_below"]   = pd.to_numeric(df["B17001_002E"], errors="coerce").clip(lower=0)
    df["pop_16plus"]  = pd.to_numeric(df["B23025_002E"], errors="coerce").clip(lower=0)
    df["clf"]         = pd.to_numeric(df["B23025_003E"], errors="coerce").clip(lower=0)
    df["unemployed"]  = pd.to_numeric(df["B23025_005E"], errors="coerce").clip(lower=0)

    df["pct_black"]   = np.where(df["pop_total"] > 0,
                                 (df["pop_black"] / df["pop_total"] * 100).clip(0, 100), np.nan)
    df["poverty_rate"] = np.where(df["pov_denom"] > 0,
                                  (df["pov_below"] / df["pov_denom"]).clip(0, 1), np.nan)
    df["unemp_rate"]   = np.where(df["clf"] > 0,
                                  (df["unemployed"] / df["clf"]).clip(0, 1), np.nan)
    df["lfpr"]         = np.where(df["pop_16plus"] > 0,
                                  (df["clf"] / df["pop_16plus"]).clip(0, 1), np.nan)

    print(f"  {len(df)} tracts fetched. Black % range: "
          f"{df['pct_black'].min():.1f}%-{df['pct_black'].max():.1f}%")
    return df[["geoid", "pop_total", "pop_black", "pct_black",
               "med_income", "poverty_rate", "unemp_rate", "lfpr"]]


# ─────────────────────────────────────────────────────────────────────────────
# Spatially-constrained Ward clustering (mirrors final_model2.ipynb Cell 8C)
# ─────────────────────────────────────────────────────────────────────────────

def build_shelby_communities(shelby_gdf: gpd.GeoDataFrame,
                              acs: pd.DataFrame,
                              n_communities: int = 6) -> gpd.GeoDataFrame:
    """
    Partition all Shelby tracts into N spatially contiguous communities using
    Ward agglomerative clustering with an adjacency connectivity constraint.

    Mirrors the notebook's Cell 8C methodology:
      - Features: economic indicators (income, poverty, unemployment)
        as proxies for IGS Economy pillar, signs flipped so higher = more vulnerable
      - Connectivity: tract adjacency matrix built from shapefile touches
      - Linkage: ward, metric: euclidean
    """
    print(f"  Building spatially-constrained Ward clustering ({n_communities} communities)...")

    df = shelby_gdf[["geoid", "geometry"]].merge(acs, on="geoid", how="left").copy()

    # Feature matrix — flip sign so all features encode vulnerability (higher = worse)
    # Mirrors notebook Cell 8C: economic + LFPR proxies for IGS Economy pillar
    features = pd.DataFrame({
        "neg_income":   -pd.to_numeric(df["med_income"],    errors="coerce"),
        "poverty_rate":  pd.to_numeric(df["poverty_rate"],  errors="coerce"),
        "unemp_rate":    pd.to_numeric(df["unemp_rate"],    errors="coerce"),
        "neg_lfpr":     -pd.to_numeric(df["lfpr"],          errors="coerce"),
    }, index=df.index)

    X = SimpleImputer(strategy="median").fit_transform(features)
    X = StandardScaler().fit_transform(X)

    # Adjacency matrix for spatial connectivity constraint
    geoid_list = df["geoid"].tolist()
    idx_map    = {g: i for i, g in enumerate(geoid_list)}

    left  = shelby_gdf[["geoid", "geometry"]].rename(columns={"geoid": "g1"})
    right = shelby_gdf[["geoid", "geometry"]].rename(columns={"geoid": "g2"})
    adj   = gpd.sjoin(left, right, how="inner", predicate="touches")
    adj   = adj[adj["g1"] != adj["g2"]][["g1", "g2"]].drop_duplicates()

    valid = adj["g1"].isin(idx_map) & adj["g2"].isin(idx_map)
    adj   = adj[valid]
    r_idx = adj["g1"].map(idx_map).to_numpy()
    c_idx = adj["g2"].map(idx_map).to_numpy()

    connectivity = coo_matrix(
        (np.ones(len(r_idx), dtype=int), (r_idx, c_idx)),
        shape=(len(geoid_list), len(geoid_list)),
    )
    connectivity = connectivity.maximum(connectivity.T)

    # Ward clustering with spatial constraint
    model  = AgglomerativeClustering(
        n_clusters=n_communities,
        linkage="ward",
        connectivity=connectivity,
    )
    labels = model.fit_predict(X)

    df["shelby_community_id"] = [f"Shelby County community {int(x) + 1}" for x in labels]

    counts = df["shelby_community_id"].value_counts()
    print("  Community tract counts:")
    for name, cnt in counts.items():
        print(f"    {name}: {cnt} tracts")

    return df[["geoid", "geometry", "shelby_community_id"]].copy()


def select_high_growth_community(community_gdf: gpd.GeoDataFrame,
                                  acs: pd.DataFrame,
                                  low_growth_geoids: set) -> set:
    """
    Pick the best comparator community following notebook Cells 8F/8G logic.

    The pre-defined low-growth tracts may span multiple Ward clusters, so we
    cannot require zero-overlap at the cluster level.  Instead:
      1. Identify which Ward cluster contains the MOST low-growth tracts — that
         cluster acts as the low-growth cluster proxy and is excluded.
      2. From the remaining clusters (non-overlapping by Ward construction),
         require >= 5 tracts and pop >= 10,000.
      3. Rank by composite percentile score: income (high=good), poverty
         (low=good), unemployment (low=good), LFPR (high=good).
    """
    # Step 1: find which cluster "is" the low-growth cluster
    overlap_counts = {}
    for community_id, grp in community_gdf.groupby("shelby_community_id"):
        geoids = set(grp["geoid"].dropna())
        overlap_counts[community_id] = len(geoids & low_growth_geoids)

    low_growth_cluster = max(overlap_counts, key=overlap_counts.get)
    print(f"  Low-growth cluster proxy: {low_growth_cluster} "
          f"({overlap_counts[low_growth_cluster]} of 96 low-growth tracts)")

    # Step 2: build eligible candidate list from all other clusters
    rows = []
    for community_id, grp in community_gdf.groupby("shelby_community_id"):
        if community_id == low_growth_cluster:
            continue
        geoids = set(grp["geoid"].dropna())
        if len(geoids) < 5:
            continue
        sub = acs[acs["geoid"].isin(geoids)]
        pop = pd.to_numeric(sub["pop_total"], errors="coerce").fillna(0).sum()
        if pop < 10_000:
            continue
        rows.append({
            "community_id": community_id,
            "geoids":       geoids,
            "income":       sub["med_income"].median(),
            "poverty":      sub["poverty_rate"].median(),
            "unemp":        sub["unemp_rate"].median(),
            "lfpr":         sub["lfpr"].median(),
        })

    if not rows:
        raise ValueError("No eligible high-growth comparator communities found.")

    # Step 3: percentile-rank each dimension, take the mean
    rank_df = pd.DataFrame(rows)
    rank_df["r_income"]  = rank_df["income"].rank(pct=True)
    rank_df["r_poverty"] = rank_df["poverty"].rank(pct=True, ascending=False)
    rank_df["r_unemp"]   = rank_df["unemp"].rank(pct=True, ascending=False)
    rank_df["r_lfpr"]    = rank_df["lfpr"].rank(pct=True)
    rank_df["score"]     = rank_df[["r_income", "r_poverty", "r_unemp", "r_lfpr"]].mean(axis=1)

    best = rank_df.loc[rank_df["score"].idxmax()]
    print(f"  High-growth comparator selected: {best['community_id']} "
          f"({len(best['geoids'])} tracts, score={best['score']:.3f})")
    return best["geoids"]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_and_classify() -> gpd.GeoDataFrame:
    print("Loading Shelby County geometry...")
    tn = gpd.read_file(SHP_PATH)
    tn["geoid"] = (tn["STATEFP"].str.zfill(2)
                   + tn["COUNTYFP"].str.zfill(3)
                   + tn["TRACTCE"].str.zfill(6))
    shelby = tn[(tn["STATEFP"] == "47") & (tn["COUNTYFP"] == "157")].copy()
    shelby = shelby.to_crs(3857)
    print(f"  {len(shelby)} Shelby County tracts")

    lg_gdf = gpd.read_file(LOW_GRW_PATH).copy()
    lg_gdf["geoid"]     = lg_gdf["geoid"].astype(str).str.zfill(11)
    low_growth_geoids   = set(lg_gdf["geoid"].dropna())
    print(f"  Low-growth tracts: {len(low_growth_geoids)}")

    acs = fetch_acs(2022)

    # Use cached cluster map if available, else recompute
    if CLUSTER_CACHE.exists():
        print(f"  Loading cached community map: {CLUSTER_CACHE}")
        community_gdf = gpd.read_file(CLUSTER_CACHE).copy()
        community_gdf["geoid"] = community_gdf["geoid"].astype(str).str.zfill(11)
        # Reproject to match shelby CRS
        community_gdf = community_gdf.set_crs(4326, allow_override=True).to_crs(3857)
    else:
        community_gdf = build_shelby_communities(shelby, acs, n_communities=6)
        # Save for reuse
        save_gdf = community_gdf.copy().to_crs(4326)
        save_gdf.to_file(CLUSTER_CACHE, driver="GeoJSON")
        print(f"  Saved community map: {CLUSTER_CACHE}")

    high_growth_geoids = select_high_growth_community(community_gdf, acs, low_growth_geoids)

    def assign_group(g):
        if g in low_growth_geoids:  return "low_growth"
        if g in high_growth_geoids: return "high_growth"
        return "rest"

    shelby["group"] = shelby["geoid"].map(assign_group)
    shelby = shelby.merge(
        acs[["geoid", "pct_black", "pop_black", "pop_total"]],
        on="geoid", how="left",
    )
    return shelby


# ─────────────────────────────────────────────────────────────────────────────
# Map
# ─────────────────────────────────────────────────────────────────────────────

# Visual config per group
_GROUP_STYLE = {
    "low_growth":  {"fill": "#F4845A", "edge": "#B83300",
                    "label": "Selected low-growth community"},
    "high_growth": {"fill": "#74C476", "edge": "#1A6B1A",
                    "label": "High-growth comparator"},
    "rest":        {"fill": "#D9D9D9", "edge": "#999999",
                    "label": "Rest of Shelby County"},
}


def plot_shelby_communities(gdf: gpd.GeoDataFrame, output_path=None):
    """
    Simple 3-color categorical map: low-growth / high-growth / rest.
    Returns fig, ax. Saves to output_path if provided.
    """
    stats = {}
    for grp in _GROUP_STYLE:
        sub = gdf[gdf["group"] == grp]
        pop_b = pd.to_numeric(sub["pop_black"], errors="coerce").fillna(0)
        pct   = pd.to_numeric(sub["pct_black"],  errors="coerce")
        stats[grp] = {
            "n":           len(sub),
            "total_black": int(pop_b.sum()),
            "avg_pct":     float(pct.mean()),
        }

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Solid fill for each group — draw rest first so highlights sit on top
    for grp in ["rest", "high_growth", "low_growth"]:
        style = _GROUP_STYLE[grp]
        sub   = gdf[gdf["group"] == grp]
        if sub.empty:
            continue
        sub.plot(ax=ax, color=style["fill"],
                 edgecolor="#FFFFFF", linewidth=0.25, zorder=2)

    # Bold dissolved boundary outlines for the two named communities
    for grp in ["high_growth", "low_growth"]:
        style    = _GROUP_STYLE[grp]
        dissolved = gdf[gdf["group"] == grp].dissolve()
        if dissolved.empty:
            continue
        dissolved.boundary.plot(
            ax=ax, color=style["edge"], linewidth=3.5,
            capstyle="round", joinstyle="round", zorder=5,
        )

    # Legend — include tract count and avg % Black in each label
    handles = []
    for grp, style in _GROUP_STYLE.items():
        s = stats[grp]
        label = (f"{style['label']}\n"
                 f"{s['n']} tracts  |  avg {s['avg_pct']:.0f}% Black  |  "
                 f"{s['total_black']:,} Black residents")
        handles.append(
            mpatches.Patch(facecolor=style["fill"],
                           edgecolor=style["edge"], linewidth=1.5,
                           label=label)
        )
    ax.legend(
        handles=handles,
        loc="upper left", fontsize=10.5,
        frameon=True, framealpha=0.95, edgecolor="#AAAAAA",
        handlelength=1.8, handleheight=1.4,
    )

    # Title + subtitle
    ax.set_title(
        "Shelby County: low-growth, high-growth, and the rest",
        fontsize=17, fontweight="bold", pad=14, loc="left",
    )
    fig.text(
        0.03, 0.935,
        "Communities identified by spatially-constrained Ward clustering on ACS economic indicators (2022).",
        fontsize=11, color="#555555", fontstyle="italic",
        transform=fig.transFigure,
    )

    # Story note at bottom
    lg, hg, rs = stats["low_growth"], stats["high_growth"], stats["rest"]
    story = (
        f"The selected low-growth community is {lg['avg_pct']:.0f}% Black, compared with "
        f"{hg['avg_pct']:.0f}% in the high-growth comparator and "
        f"{rs['avg_pct']:.0f}% in the rest of Shelby County."
    )
    fig.text(
        0.03, 0.015, story,
        fontsize=10.5, color="#333333", va="bottom",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7",
                  edgecolor="#BBBBBB", linewidth=0.9),
    )

    ax.set_axis_off()
    plt.subplots_adjust(bottom=0.10, top=0.92, left=0.01, right=0.99)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")

    return fig, ax


def plot_black_population_choropleth(gdf, output_path=None):
    return plot_shelby_communities(gdf, output_path=output_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gdf = load_and_classify()
    print(f"\nGroup counts:\n{gdf['group'].value_counts().to_string()}")
    print(f"Missing pct_black: {gdf['pct_black'].isna().sum()}")
    print("\nRendering map...")
    fig, ax = plot_black_population_choropleth(gdf, output_path=OUT_PATH)
    plt.show()
