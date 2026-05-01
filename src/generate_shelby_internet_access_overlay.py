from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from presentation_callouts import add_why_it_matters_callout
from generate_shelby_black_population_overlay import (
    FULL_COMPARE_PATH,
    FULL_TRACT_MAP_PATH,
    LOW_GROWTH_TRACTS_PATH,
    OUTLINE_COLORS,
    TRACT_EDGE_COLOR,
    COUNTY_OUTLINE_COLOR,
    CHOROPLETH_CMAP,
    GROUP_COL,
    normalize_geoid,
    pick_high_growth_comparator,
    load_cluster_rank,
    load_seed_county_rank,
)
from project_paths import MAPS_DIR, PROCESSED_DATA_DIR

DEFAULT_OUTPUT_IMAGE_PATH = MAPS_DIR / "shelby_internet_access_choropleth.png"
INTERNET_DATA_PATH = PROCESSED_DATA_DIR / "shelby_acs_internet_access_tract_2022_2024.parquet"
DEFAULT_YEAR = 2024
FALLBACK_YEARS = [2024, 2023, 2022]


def build_why_it_matters_text(plot_gdf: gpd.GeoDataFrame) -> str:
    low_growth = plot_gdf.loc[plot_gdf[GROUP_COL] == "Selected low-growth community"].copy()
    households = pd.to_numeric(low_growth.get("total_households"), errors="coerce").fillna(0)
    internet_households = pd.to_numeric(
        low_growth.get("internet_subscription_households"),
        errors="coerce",
    ).fillna(0)
    share = internet_households.sum() / households.sum() if households.sum() > 0 else float("nan")
    if pd.notna(share):
        return (
            f"Only {share:.1%} of households in the selected low-growth community report an "
            "internet subscription. That means booking, alerts, and rider support cannot rely "
            "only on digital channels."
        )
    return (
        "Internet access is another place-based barrier in Shelby County, so booking, alerts, "
        "and rider support cannot rely only on digital channels."
    )


def build_internet_subtitle(plot_gdf: gpd.GeoDataFrame) -> str:
    if "internet_source_year" not in plot_gdf.columns:
        return "Tract shading shows household internet subscription share using ACS 5-year tract estimates."
    source_years = sorted(
        {
            int(year)
            for year in pd.to_numeric(plot_gdf["internet_source_year"], errors="coerce").dropna().tolist()
        }
    )
    if not source_years:
        return "Tract shading shows household internet subscription share using ACS 5-year tract estimates."
    if len(source_years) == 1:
        year = source_years[0]
        return (
            f"Tract shading shows household internet subscription share using the "
            f"{year - 4}-{year} ACS 5-year estimates."
        )
    return (
        "Tract shading shows household internet subscription share using the "
        f"{DEFAULT_YEAR - 4}-{DEFAULT_YEAR} ACS 5-year estimates, with tract-level fallback "
        "to earlier ACS releases only where needed."
    )


def build_shelby_internet_access_gdf() -> gpd.GeoDataFrame:
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

    if not INTERNET_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed Shelby ACS internet data: {INTERNET_DATA_PATH}\n"
            "Run `python fetch_acs_internet_access.py` first."
        )

    internet_df = pd.read_parquet(INTERNET_DATA_PATH).copy()
    internet_df["geoid"] = normalize_geoid(internet_df["geoid"])
    internet_df["year"] = pd.to_numeric(internet_df["year"], errors="coerce").astype("Int64")
    internet_df["internet_sub_share"] = pd.to_numeric(internet_df["internet_sub_share"], errors="coerce")
    internet_df["internet_subscription_pct"] = pd.to_numeric(
        internet_df.get("internet_subscription_pct"), errors="coerce"
    )
    internet_df["broadband_share"] = pd.to_numeric(internet_df.get("broadband_share"), errors="coerce")
    internet_df["no_internet_share"] = pd.to_numeric(internet_df.get("no_internet_share"), errors="coerce")

    available_years = sorted({int(year) for year in internet_df["year"].dropna().tolist()}, reverse=True)
    preferred_years = [year for year in FALLBACK_YEARS if year in available_years]
    if not preferred_years:
        preferred_years = available_years
    if not preferred_years:
        raise ValueError(f"No usable ACS internet rows were found in {INTERNET_DATA_PATH}.")

    fallback_df = internet_df.loc[internet_df["year"].isin(preferred_years)].copy()
    fallback_df = fallback_df.sort_values(["geoid", "year"], ascending=[True, False])
    internet_by_tract = (
        fallback_df
        .dropna(subset=["internet_sub_share"])
        .drop_duplicates(subset=["geoid"], keep="first")
        .rename(columns={"year": "internet_source_year"})
    )

    gdf = tracts_gdf.merge(internet_by_tract, on="geoid", how="left", validate="1:1")
    gdf["internet_sub_share"] = pd.to_numeric(gdf["internet_sub_share"], errors="coerce")
    gdf["internet_sub_share"] = gdf["internet_sub_share"].clip(lower=0, upper=1)
    gdf["internet_source_year"] = pd.to_numeric(gdf.get("internet_source_year"), errors="coerce").astype("Int64")
    gdf.attrs["internet_candidate_years"] = preferred_years
    gdf.attrs["internet_source_years_used"] = sorted(
        {int(year) for year in gdf["internet_source_year"].dropna().tolist()}
    )

    if gdf.crs is not None and str(gdf.crs) != "EPSG:3857":
        gdf = gdf.to_crs(3857)

    return gdf


def validate_plot_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    if "internet_sub_share" not in work.columns:
        raise ValueError("GeoDataFrame must include `internet_sub_share` for the map.")
    if GROUP_COL not in work.columns:
        raise ValueError(f"GeoDataFrame must include `{GROUP_COL}` for the community grouping.")
    work["internet_sub_share"] = pd.to_numeric(work["internet_sub_share"], errors="coerce").clip(lower=0, upper=1)
    return work


def plot_internet_access_choropleth(
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

    fig = plt.figure(figsize=(15.5, 11), dpi=240)
    ax = fig.add_axes([0.19, 0.14, 0.63, 0.70])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plot_gdf.plot(
        column="internet_sub_share",
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
        "Shelby County Internet Access",
        fontsize=25,
        y=0.968,
    )
    fig.text(
        0.5,
        0.925,
        build_internet_subtitle(plot_gdf),
        ha="center",
        va="center",
        fontsize=14,
        color="#4b5563",
    )

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CHOROPLETH_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.87, 0.32, 0.028, 0.45])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("% households with internet subscription", fontsize=16)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    cbar.ax.tick_params(labelsize=13.5)

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

    outline_legend = fig.legend(
        handles=outline_handles,
        title="Community outlines",
        loc="upper left",
        bbox_to_anchor=(0.055, 0.87),
        bbox_transform=fig.transFigure,
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.95,
        fontsize=10.7,
        title_fontsize=12,
        handlelength=1.8,
        borderpad=0.45,
        labelspacing=0.5,
    )
    fig.add_artist(outline_legend)

    add_why_it_matters_callout(
        fig,
        bounds=(0.055, 0.48, 0.20, 0.19),
        body=build_why_it_matters_text(plot_gdf),
        wrap_width=30,
        title_fontsize=16,
        body_fontsize=10.1,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=240, facecolor="white")

    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Shelby County internet access choropleth."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_IMAGE_PATH,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after rendering.",
    )
    return parser.parse_args()


def main(output_path: str | Path | None = None, show: bool = False) -> None:
    cluster_row, shortlist = load_cluster_rank()
    seed_row = load_seed_county_rank()
    shelby_gdf = build_shelby_internet_access_gdf()
    final_output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_IMAGE_PATH
    fig, _ = plot_internet_access_choropleth(shelby_gdf, output_path=final_output_path)

    print(f"Saved choropleth map: {final_output_path}")
    print()
    print("Internet access coverage")
    print("------------------------")
    print(f"Preferred year: {DEFAULT_YEAR}")
    source_years_used = shelby_gdf.attrs.get("internet_source_years_used", [])
    print(f"Source years used after fallback: {source_years_used or 'none'}")
    print(
        f"Tracts with values: {int(shelby_gdf['internet_sub_share'].notna().sum())}/"
        f"{len(shelby_gdf)}"
    )
    if "has_households" in shelby_gdf.columns:
        zero_household = int(shelby_gdf["has_households"].fillna(False).eq(False).sum())
        print(f"Zero-household tracts left unshaded by ACS: {zero_household}")
    if "internet_source_year" in shelby_gdf.columns:
        counts = (
            shelby_gdf["internet_source_year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        if not counts.empty:
            print("Tracts by ACS source year:")
            print(counts.to_string())
            print()
    print("Current cluster shortlist confirmation")
    print("------------------------------------")
    print(f"Cluster ID: {cluster_row['cluster_id']}")
    print(f"Rank: #{int(cluster_row['rank'])} of {len(shortlist)} shortlisted clusters")
    print(f"Shortlist score: {float(cluster_row['cluster_shortlist_score']):.6f}")
    print()
    if seed_row is not None:
        print("County seed-stage context")
        print("-------------------------")
        print(f"Shelby County seed rank: #{int(seed_row['rank'])}")
        print(f"County seed score: {float(seed_row['county_seed_score']):.6f}")
        print(f"Signal tracts: {int(seed_row['n_signal_tracts'])}")
        print(f"Mean scope score: {float(seed_row['mean_scope_score']):.6f}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    main(output_path=args.output, show=args.show)
