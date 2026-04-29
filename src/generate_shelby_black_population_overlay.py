from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from project_paths import MAPS_DIR, PRESENTATION_READY_DIR, PROCESSED_DATA_DIR

CLUSTER_ID = "Tennessee | Shelby County | cluster_77"
SHELBY_COUNTY_FIPS = "47157"
RECOVERY_YEARS = [2022, 2023, 2024]

FULL_TRACT_MAP_PATH = PRESENTATION_READY_DIR / "shelby_full_tract_cluster_map.geojson"
LOW_GROWTH_TRACTS_PATH = PRESENTATION_READY_DIR / "selected_low_growth_96_tracts.geojson"
FULL_COMPARE_PATH = PRESENTATION_READY_DIR / "shelby_full_community_compare.csv"
SHORTLIST_PATH = PROCESSED_DATA_DIR / "cluster_shortlist.parquet"
SEED_COUNTIES_PATH = PROCESSED_DATA_DIR / "seed_counties.parquet"
PANEL_PATH = PROCESSED_DATA_DIR / "eda_panel_clean.parquet"
DEFAULT_OUTPUT_IMAGE_PATH = MAPS_DIR / "shelby_black_share_choropleth.png"

GROUP_COL = "group"

GROUP_ORDER = [
    "Selected low-growth community",
    "Rest of Shelby County (excluding both selected communities)",
    "Selected high-growth comparator",
]

GROUP_LABELS = {
    "Selected low-growth community": "Selected low-growth community",
    "Rest of Shelby County (excluding both selected communities)": "Rest of Shelby County",
    "Selected high-growth comparator": "High-growth comparator",
}

OUTLINE_COLORS = {
    "Selected low-growth community": "#fb6a4a",
    "Selected high-growth comparator": "#74c476",
}

TRACT_EDGE_COLOR = "#e9ecef"
COUNTY_OUTLINE_COLOR = "#4a4a4a"
CHOROPLETH_CMAP = "Blues"


def normalize_geoid(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.extract(r"(\d{11})", expand=False)
        .fillna(series.astype("string"))
        .str.replace(r"\D", "", regex=True)
        .str.zfill(11)
    )


def safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = pd.Series(np.nan, index=num.index, dtype=float)
    valid = num.notna() & den.notna() & (den > 0)
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def pick_high_growth_comparator(compare_df: pd.DataFrame) -> str:
    work = compare_df.copy()
    work["n_tracts"] = pd.to_numeric(work["n_tracts"], errors="coerce")
    work["igs_total_recovery"] = pd.to_numeric(work["igs_total_recovery"], errors="coerce")
    work = work.loc[work["n_tracts"].fillna(0) >= 40].copy()
    work = work.dropna(subset=["igs_total_recovery"]).sort_values("igs_total_recovery", ascending=False)
    if work.empty:
        raise ValueError("Could not identify a high-growth Shelby comparator from the saved community summary.")
    return str(work.iloc[0]["community_id"])


def load_cluster_rank() -> tuple[pd.Series, pd.DataFrame]:
    shortlist = pd.read_parquet(SHORTLIST_PATH).copy()
    shortlist = shortlist.sort_values(["rank", "cluster_shortlist_score"], ascending=[True, False]).reset_index(drop=True)
    row = shortlist.loc[shortlist["cluster_id"].astype(str) == CLUSTER_ID]
    if row.empty:
        raise ValueError(f"{CLUSTER_ID} not found in {SHORTLIST_PATH}.")
    return row.iloc[0], shortlist


def load_seed_county_rank() -> pd.Series | None:
    seed = pd.read_parquet(SEED_COUNTIES_PATH).copy()
    seed = seed.sort_values(
        ["county_seed_score", "n_signal_tracts", "mean_scope_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    seed["rank"] = np.arange(1, len(seed) + 1)
    row = seed.loc[
        (seed["display_state"].astype(str) == "Tennessee")
        & (seed["display_county"].astype(str) == "Shelby County")
    ]
    if row.empty:
        return None
    return row.iloc[0]


def build_shelby_black_share_gdf() -> gpd.GeoDataFrame:
    tracts_gdf = gpd.read_file(FULL_TRACT_MAP_PATH)
    selected_gdf = gpd.read_file(LOW_GROWTH_TRACTS_PATH)
    compare_df = pd.read_csv(FULL_COMPARE_PATH)

    tracts_gdf["geoid"] = normalize_geoid(tracts_gdf["geoid"])
    selected_gdf["geoid"] = normalize_geoid(selected_gdf["geoid"])
    selected_geoids = set(selected_gdf["geoid"].dropna())

    high_growth_cluster_id = pick_high_growth_comparator(compare_df)
    high_growth_mask = (
        tracts_gdf["shelby_full_cluster_id"].astype(str).eq(high_growth_cluster_id)
        & ~tracts_gdf["geoid"].isin(selected_geoids)
    )

    tracts_gdf[GROUP_COL] = "Rest of Shelby County (excluding both selected communities)"
    tracts_gdf.loc[high_growth_mask, GROUP_COL] = "Selected high-growth comparator"
    tracts_gdf.loc[tracts_gdf["geoid"].isin(selected_geoids), GROUP_COL] = "Selected low-growth community"

    panel = pd.read_parquet(
        PANEL_PATH,
        columns=["geoid", "year", "B02001_003E", "share_black", "pop_total"],
    ).copy()
    panel["geoid"] = normalize_geoid(panel["geoid"])
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel = panel.loc[
        panel["geoid"].str[:5].eq(SHELBY_COUNTY_FIPS)
        & panel["year"].isin(RECOVERY_YEARS)
    ].copy()

    panel["black_population"] = pd.to_numeric(panel["B02001_003E"], errors="coerce")
    panel["total_population"] = pd.to_numeric(panel["pop_total"], errors="coerce")
    panel["pct_black_row"] = safe_pct(panel["black_population"], panel["total_population"])
    panel["share_black"] = pd.to_numeric(panel["share_black"], errors="coerce")

    tract_demo = (
        panel.groupby("geoid", as_index=False)
        .agg(
            black_population=("black_population", "mean"),
            total_population=("total_population", "mean"),
            share_black_mean=("share_black", "mean"),
            pct_black_row_mean=("pct_black_row", "mean"),
        )
    )

    tract_demo["pct_black"] = safe_pct(tract_demo["black_population"], tract_demo["total_population"])
    fallback_mask = tract_demo["pct_black"].isna()
    tract_demo.loc[fallback_mask, "pct_black"] = tract_demo.loc[fallback_mask, "share_black_mean"]
    second_fallback = tract_demo["pct_black"].isna()
    tract_demo.loc[second_fallback, "pct_black"] = tract_demo.loc[second_fallback, "pct_black_row_mean"]
    tract_demo["pct_black"] = tract_demo["pct_black"].clip(lower=0, upper=1)

    gdf = tracts_gdf.merge(tract_demo, on="geoid", how="left", validate="1:1")
    gdf["pct_black"] = pd.to_numeric(gdf["pct_black"], errors="coerce").clip(lower=0, upper=1)
    gdf["black_population"] = pd.to_numeric(gdf["black_population"], errors="coerce")
    gdf["total_population"] = pd.to_numeric(gdf["total_population"], errors="coerce")

    if gdf.crs is not None and str(gdf.crs) != "EPSG:3857":
        gdf = gdf.to_crs(3857)

    return gdf


def summarize_groups(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        sub = gdf.loc[gdf[GROUP_COL] == group].copy()
        total_black = float(pd.to_numeric(sub["black_population"], errors="coerce").fillna(0).sum())
        total_pop = float(pd.to_numeric(sub["total_population"], errors="coerce").fillna(0).sum())
        avg_pct_black = (total_black / total_pop) if total_pop > 0 else np.nan
        rows.append(
            {
                "group": group,
                "label": GROUP_LABELS[group],
                "n_tracts": int(sub["geoid"].nunique()),
                "black_population": total_black,
                "total_population": total_pop,
                "avg_pct_black": avg_pct_black,
            }
        )
    return pd.DataFrame(rows)


def build_summary_text(summary_df: pd.DataFrame) -> str:
    ordered = summary_df.set_index("group").loc[GROUP_ORDER].reset_index()
    lines = ["Community snapshot (ACS tract averages, 2022-2024)"]
    for _, row in ordered.iterrows():
        lines.append(
            f"{row['label']}: {row['avg_pct_black']:.0%} Black | "
            f"{row['black_population']:,.0f} est. Black residents | "
            f"{int(row['n_tracts'])} tracts"
        )
    return "\n".join(lines)


def validate_plot_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()

    if "pct_black" not in work.columns:
        required = {"black_population", "total_population"}
        if not required.issubset(work.columns):
            raise ValueError(
                "GeoDataFrame must include either `pct_black` or both "
                "`black_population` and `total_population`."
            )
        work["pct_black"] = safe_pct(work["black_population"], work["total_population"])

    if GROUP_COL not in work.columns:
        raise ValueError(f"GeoDataFrame must include `{GROUP_COL}` for the community grouping.")

    work["pct_black"] = pd.to_numeric(work["pct_black"], errors="coerce").clip(lower=0, upper=1)
    return work


def plot_black_population_choropleth(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    plot_gdf = validate_plot_columns(gdf)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(15.5, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.subplots_adjust(left=0.03, right=0.84, top=0.86, bottom=0.10)

    choropleth = plot_gdf.plot(
        column="pct_black",
        cmap=CHOROPLETH_CMAP,
        vmin=0,
        vmax=1,
        linewidth=0.45,
        edgecolor=TRACT_EDGE_COLOR,
        ax=ax,
        zorder=1,
        missing_kwds={
            "color": "#f3f4f6",
            "edgecolor": "#e5e7eb",
        },
    )

    county_outline = plot_gdf.dissolve()
    county_outline.boundary.plot(
        ax=ax,
        color=COUNTY_OUTLINE_COLOR,
        linewidth=2.4,
        zorder=3,
    )

    outline_gdf = plot_gdf.dissolve(by=GROUP_COL).reset_index()
    outline_gdf = outline_gdf.loc[outline_gdf[GROUP_COL].isin(OUTLINE_COLORS)].copy()

    for _, row in outline_gdf.iterrows():
        boundary = gpd.GeoDataFrame([row], geometry="geometry", crs=plot_gdf.crs)
        boundary.boundary.plot(
            ax=ax,
            color="white",
            linewidth=8.4,
            zorder=4,
        )
        if row[GROUP_COL] == "Selected high-growth comparator":
            boundary.boundary.plot(
                ax=ax,
                color="black",
                linewidth=6.2,
                zorder=5,
            )
        boundary.boundary.plot(
            ax=ax,
            color=OUTLINE_COLORS[row[GROUP_COL]],
            linewidth=5.8,
            zorder=6,
        )

    ax.set_axis_off()

    fig.suptitle(
        "Shelby County Black Share",
        fontsize=24,
        y=0.968,
    )
    fig.text(
        0.5,
        0.925,
        "Tract shading shows percent Black population using ACS tract averages for 2022–2024.",
        ha="center",
        va="center",
        fontsize=14,
        color="#4b5563",
    )

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CHOROPLETH_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.855, 0.29, 0.038, 0.49])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("% Black residents", fontsize=16)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    cbar.ax.tick_params(labelsize=14)

    outline_handles = [
        Line2D(
            [0],
            [0],
            color=OUTLINE_COLORS["Selected low-growth community"],
            linewidth=5.8,
            label="Selected low-growth community outline",
        ),
        Line2D(
            [0],
            [0],
            color=OUTLINE_COLORS["Selected high-growth comparator"],
            linewidth=5.8,
            label="High-growth comparator outline",
        ),
    ]

    outline_legend = ax.legend(
        handles=outline_handles,
        title="Community outlines",
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.95,
        fontsize=12,
        title_fontsize=14,
    )
    ax.add_artist(outline_legend)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=240)

    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Shelby County Black population share choropleth."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_IMAGE_PATH,
        help="Output PNG path.",
    )
    return parser.parse_args()


def main(output_path: str | Path | None = None) -> None:
    cluster_row, shortlist = load_cluster_rank()
    seed_row = load_seed_county_rank()
    shelby_gdf = build_shelby_black_share_gdf()
    summary_df = summarize_groups(shelby_gdf)
    final_output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_IMAGE_PATH
    plot_black_population_choropleth(shelby_gdf, output_path=final_output_path)

    print(f"Saved choropleth map: {final_output_path}")
    print()
    print("Current cluster shortlist confirmation")
    print("------------------------------------")
    print(f"Cluster ID: {cluster_row['cluster_id']}")
    print(f"Rank: #{int(cluster_row['rank'])} of {len(shortlist)} shortlisted clusters")
    print(f"Shortlist score: {float(cluster_row['cluster_shortlist_score']):.6f}")
    print()
    print("Black population share summary")
    print("------------------------------")
    print(summary_df[["label", "avg_pct_black", "black_population", "n_tracts"]].to_string(index=False))
    print()

    if seed_row is not None:
        print("County seed-stage context")
        print("-------------------------")
        print(f"Shelby County seed rank: #{int(seed_row['rank'])}")
        print(f"County seed score: {float(seed_row['county_seed_score']):.6f}")
        print(f"Signal tracts: {int(seed_row['n_signal_tracts'])}")
        print(f"Mean scope score: {float(seed_row['mean_scope_score']):.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(output_path=args.output)
