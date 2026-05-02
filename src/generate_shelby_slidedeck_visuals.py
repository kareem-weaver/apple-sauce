from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from presentation_callouts import add_why_it_matters_callout
from project_paths import (
    PRESENTATION_READY_DIR,
    PRESENTATION_READY_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

RECOVERY_YEARS = [2022, 2023, 2024]

FULL_TRACT_MAP_PATH = PRESENTATION_READY_DIR / "shelby_full_tract_cluster_map.geojson"
LOW_GROWTH_TRACTS_PATH = PRESENTATION_READY_DIR / "selected_low_growth_96_tracts.geojson"
FULL_COMPARE_PATH = PRESENTATION_READY_DIR / "shelby_full_community_compare.csv"
DEMOGRAPHIC_COMPARE_PATH = PRESENTATION_READY_DIR / "shelby_demographic_compare.csv"
THREE_GROUP_SUMMARY_PATH = PRESENTATION_READY_DIR / "three_group_selected_vs_comparator_vs_rest.csv"
PANEL_PATH = PROCESSED_DATA_DIR / "eda_panel_clean.parquet"
INTERNET_ACCESS_RAW_PATH = RAW_DATA_DIR / "external" / "acs_s2801_shelby_tract_2022_2024.parquet"
VEHICLE_ACCESS_RAW_PATH = RAW_DATA_DIR / "external" / "acs_b08201_shelby_tract_2022_2024.parquet"
LEGACY_VEHICLE_ACCESS_RAW_PATH = RAW_DATA_DIR / "external" / "acs_vehicle_tract_2022_2024.parquet"
ZCTA_SHAPEFILE_PATH = RAW_DATA_DIR / "zcta" / "tl_2024_us_zcta520" / "tl_2024_us_zcta520.shp"

LOW_RED = "#f79a9b"
LOW_RED_DARK = "#ef7f84"
REST_GRAY = "#c9c9c9"
HIGH_GREEN = "#a7e0b4"
HIGH_GREEN_DARK = "#76c780"
TN_GREEN = "#247a39"
US_RED = "#a81515"
NAVY_TEXT = "#0f4a92"
MUTED_TEXT = "#5f6b7d"
GRID_COLOR = "#dfe5ee"
BORDER_COLOR = "#d7dde8"
MAP_LOW_OUTLINE = "#ff603a"
MAP_HIGH_OUTLINE = "#77c96f"
COUNTY_OUTLINE = "#4c5157"
TRACT_EDGE = "#d8dee8"

GROUP_LOW = "Selected low-growth community"
GROUP_REST = "Rest of Shelby County (excluding both selected communities)"
GROUP_HIGH = "Selected high-growth comparator"

GROUP_ORDER = [GROUP_LOW, GROUP_REST, GROUP_HIGH]

NO_INTERNET_BENCHMARK_YEAR = 2024
VEHICLE_CHART_YEAR = 2024
TARGET_ZCTAS = ["38109", "38116", "38117"]
TARGET_ZCTA_LABELS = {
    "38109": "38109 (Low-growth)",
    "38116": "38116 (Low-growth)",
    "38117": "38117 (High-growth)",
}

PANEL_COLUMNS = [
    "geoid",
    "year",
    "State",
    "pop_total",
    "share_black",
    "igs_total",
    "poverty_rate",
    "unemp_rate",
    "median_household_income",
    "lfpr_16p",
    "B28002_001E",
    "B28002_002E",
]

PANEL_NUMERIC_COLUMNS = [
    "year",
    "pop_total",
    "share_black",
    "igs_total",
    "poverty_rate",
    "unemp_rate",
    "median_household_income",
    "lfpr_16p",
    "B28002_001E",
    "B28002_002E",
]

VEHICLE_REQUIRED_COLUMNS = [
    "hh_total",
    "hh_no_vehicle",
    "hh_one_vehicle",
    "hh_two_vehicle",
    "hh_three_vehicle",
    "hh_four_plus_vehicle",
]

OUTPUT_FILES = {
    "black-map": "shelby_black_share_choropleth.png",
    "demographic": "shelby_visual_demographic_comparison.png",
    "economic": "shelby_slidedeck_economic_benchmark.png",
    "median-income": "shelby_slidedeck_median_income.png",
    "health": "shelby_slidedeck_health_indicator_comparison.png",
    "vehicle": "shelby_slidedeck_vehicle_access_by_zip.png",
    "no-internet": "shelby_no_internet_bar_chart.png",
}

BLACK_MAP_WHY_IT_MATTERS = (
    "This suggests that Shelby's growth divide is not just economic - it is also "
    "geographic and demographic. That concentration is one reason ShelbyFirst is "
    "designed as a targeted, place-based intervention for marginalized communities."
)

NO_INTERNET_WHY_IT_MATTERS_TEMPLATE = (
    "Our reduced-cost ride service depends on internet access. Since {low_pct:.1f}% "
    "of households in the selected low-growth community lack internet, booking "
    "cannot rely only on an app. We plan to place booking terminals in libraries "
    "and similar community locations."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Shelby slide-deck charts and maps as PNGs."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(OUTPUT_FILES),
        default=list(OUTPUT_FILES),
        help="Subset of visuals to render. Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PRESENTATION_READY_OUTPUT_DIR,
        help="Directory where PNGs will be written. Defaults to src/outputs/presentation_ready/.",
    )
    return parser.parse_args()


def normalize_geoid(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.extract(r"(\d{11})", expand=False)
        .fillna(values.astype("string"))
        .str.replace(r"\D", "", regex=True)
        .str.zfill(11)
    )


def pick_high_growth_comparator(compare_df: pd.DataFrame) -> str:
    work = compare_df.copy()
    work["n_tracts"] = pd.to_numeric(work["n_tracts"], errors="coerce")
    work["igs_total_recovery"] = pd.to_numeric(work["igs_total_recovery"], errors="coerce")
    work = work.loc[work["n_tracts"].fillna(0) >= 40].copy()
    work = work.dropna(subset=["igs_total_recovery"]).sort_values("igs_total_recovery", ascending=False)
    if work.empty:
        raise ValueError("Could not identify a Shelby high-growth comparator from the saved summary table.")
    return str(work.iloc[0]["community_id"])


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def to_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_series = pd.to_numeric(values, errors="coerce")
    weight_series = pd.to_numeric(weights, errors="coerce")
    valid = value_series.notna() & weight_series.notna() & (weight_series > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(value_series.loc[valid], weights=weight_series.loc[valid]))


@lru_cache(maxsize=1)
def load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH, columns=PANEL_COLUMNS).copy()
    panel["geoid"] = normalize_geoid(panel["geoid"])
    return to_numeric_columns(panel, PANEL_NUMERIC_COLUMNS)


@lru_cache(maxsize=1)
def load_three_group_summary() -> pd.DataFrame:
    summary = pd.read_csv(THREE_GROUP_SUMMARY_PATH).copy()
    summary = to_numeric_columns(summary, [column for column in summary.columns if column != "group"])
    return summary


@lru_cache(maxsize=1)
def load_shelby_group_lookup() -> pd.DataFrame:
    full_map = gpd.read_file(FULL_TRACT_MAP_PATH)[["geoid", "shelby_full_cluster_id"]].copy()
    low_growth = gpd.read_file(LOW_GROWTH_TRACTS_PATH)[["geoid"]].copy()
    full_compare = pd.read_csv(FULL_COMPARE_PATH).copy()

    full_map["geoid"] = normalize_geoid(full_map["geoid"])
    low_growth["geoid"] = normalize_geoid(low_growth["geoid"])
    low_geoids = set(low_growth["geoid"].dropna())
    comparator_id = pick_high_growth_comparator(full_compare)

    lookup = full_map.drop_duplicates(subset=["geoid"]).copy()
    lookup["group"] = GROUP_REST
    lookup.loc[lookup["geoid"].isin(low_geoids), "group"] = GROUP_LOW
    lookup.loc[
        lookup["shelby_full_cluster_id"].astype(str).eq(comparator_id) & ~lookup["geoid"].isin(low_geoids),
        "group",
    ] = GROUP_HIGH
    return lookup[["geoid", "group", "shelby_full_cluster_id"]].copy()


def build_population_weighted_benchmarks() -> pd.DataFrame:
    panel = load_panel()
    panel = panel.loc[panel["year"].isin(RECOVERY_YEARS)].copy()
    rows = []
    for label, mask in [("Tennessee", panel["State"].eq("Tennessee")), ("U.S.", panel["State"].notna())]:
        sub = panel.loc[mask].copy()
        weights = sub["pop_total"]
        rows.append(
            {
                "label": label,
                "igs_total": weighted_mean(sub["igs_total"], weights),
                "poverty_rate": weighted_mean(sub["poverty_rate"], weights),
                "unemp_rate": weighted_mean(sub["unemp_rate"], weights),
                "median_household_income": weighted_mean(sub["median_household_income"], weights),
                "lfpr_16p": weighted_mean(sub["lfpr_16p"], weights),
            }
        )
    return pd.DataFrame(rows)


def build_economic_benchmark_data() -> pd.DataFrame:
    summary = load_three_group_summary().set_index("group")
    benchmarks = build_population_weighted_benchmarks().set_index("label")
    return pd.DataFrame(
        {
            "Metric": ["IGS Score", "Poverty (%)", "Unemployment (%)", "LFP (%)"],
            "Low-IGS Shelby": [
                summary.loc[GROUP_LOW, "igs_total_recovery"],
                summary.loc[GROUP_LOW, "poverty_rate_recovery"] * 100.0,
                summary.loc[GROUP_LOW, "unemp_rate_recovery"] * 100.0,
                summary.loc[GROUP_LOW, "lfpr_16p_recovery"] * 100.0,
            ],
            "High-Growth Shelby": [
                summary.loc[GROUP_HIGH, "igs_total_recovery"],
                summary.loc[GROUP_HIGH, "poverty_rate_recovery"] * 100.0,
                summary.loc[GROUP_HIGH, "unemp_rate_recovery"] * 100.0,
                summary.loc[GROUP_HIGH, "lfpr_16p_recovery"] * 100.0,
            ],
            "Tennessee": [
                benchmarks.loc["Tennessee", "igs_total"],
                benchmarks.loc["Tennessee", "poverty_rate"] * 100.0,
                benchmarks.loc["Tennessee", "unemp_rate"] * 100.0,
                benchmarks.loc["Tennessee", "lfpr_16p"] * 100.0,
            ],
            "U.S.": [
                benchmarks.loc["U.S.", "igs_total"],
                benchmarks.loc["U.S.", "poverty_rate"] * 100.0,
                benchmarks.loc["U.S.", "unemp_rate"] * 100.0,
                benchmarks.loc["U.S.", "lfpr_16p"] * 100.0,
            ],
        }
    )


def build_median_income_data() -> pd.DataFrame:
    summary = load_three_group_summary().set_index("group")
    benchmarks = build_population_weighted_benchmarks().set_index("label")
    return pd.DataFrame(
        {
            "Group": ["Low-IGS Shelby", "High-IGS Shelby", "Tennessee", "U.S."],
            "Median Income": [
                summary.loc[GROUP_LOW, "median_household_income_recovery"] / 1000.0,
                summary.loc[GROUP_HIGH, "median_household_income_recovery"] / 1000.0,
                benchmarks.loc["Tennessee", "median_household_income"] / 1000.0,
                benchmarks.loc["U.S.", "median_household_income"] / 1000.0,
            ],
        }
    )


def build_health_comparison_data() -> pd.DataFrame:
    summary = load_three_group_summary().set_index("group")
    return pd.DataFrame(
        {
            "Indicator": [
                "Uninsured Adults (%)",
                "Dental Visit Gap",
                "Obesity",
                "Diabetes",
            ],
            "Low-IGS Shelby": [
                summary.loc[GROUP_LOW, "uninsured_adults_pct"],
                summary.loc[GROUP_LOW, "dental_visit_gap_pct"],
                summary.loc[GROUP_LOW, "obesity_pct"],
                summary.loc[GROUP_LOW, "diabetes_pct"],
            ],
            "Rest of Shelby County": [
                summary.loc[GROUP_REST, "uninsured_adults_pct"],
                summary.loc[GROUP_REST, "dental_visit_gap_pct"],
                summary.loc[GROUP_REST, "obesity_pct"],
                summary.loc[GROUP_REST, "diabetes_pct"],
            ],
            "High-IGS Shelby": [
                summary.loc[GROUP_HIGH, "uninsured_adults_pct"],
                summary.loc[GROUP_HIGH, "dental_visit_gap_pct"],
                summary.loc[GROUP_HIGH, "obesity_pct"],
                summary.loc[GROUP_HIGH, "diabetes_pct"],
            ],
        }
    )


def build_no_internet_benchmarks() -> dict[str, tuple[float, str]]:
    panel = load_panel()
    panel = panel.loc[panel["year"] == NO_INTERNET_BENCHMARK_YEAR].copy()
    benchmarks: dict[str, tuple[float, str]] = {}
    for label, state_name, color in [
        ("Tennessee average", "Tennessee", "#27a327"),
        ("U.S. average", None, "#ff1f1f"),
    ]:
        sub = panel if state_name is None else panel.loc[panel["State"].eq(state_name)].copy()
        total_households = pd.to_numeric(sub["B28002_001E"], errors="coerce").sum()
        connected_households = pd.to_numeric(sub["B28002_002E"], errors="coerce").sum()
        if total_households <= 0:
            share_without_internet = float("nan")
        else:
            share_without_internet = 100.0 * (1.0 - (connected_households / total_households))
        benchmarks[label] = (share_without_internet, color)
    return benchmarks


def build_no_internet_chart_data() -> tuple[pd.DataFrame, dict[str, tuple[float, str]], str]:
    internet = pd.read_parquet(INTERNET_ACCESS_RAW_PATH).copy()
    internet["geoid"] = normalize_geoid(internet["geoid"])
    internet = to_numeric_columns(internet, ["year", "total_households", "no_internet_households"])
    internet = internet.loc[internet["year"] == NO_INTERNET_BENCHMARK_YEAR].copy()

    group_lookup = load_shelby_group_lookup()[["geoid", "group"]].copy()
    internet = internet.merge(group_lookup, on="geoid", how="inner", validate="many_to_one")

    grouped = (
        internet.groupby("group", as_index=False)[["total_households", "no_internet_households"]]
        .sum(min_count=1)
        .copy()
    )
    grouped["Percent"] = 100.0 * grouped["no_internet_households"] / grouped["total_households"]
    grouped["Households"] = grouped["no_internet_households"]
    grouped["Color"] = grouped["group"].map(
        {
            GROUP_LOW: LOW_RED,
            GROUP_REST: REST_GRAY,
            GROUP_HIGH: HIGH_GREEN,
        }
    )
    grouped["Group"] = grouped["group"].map(
        {
            GROUP_LOW: "Selected low-growth community",
            GROUP_REST: "Rest of Shelby County",
            GROUP_HIGH: "High-growth comparator",
        }
    )
    grouped = grouped.set_index("group").loc[GROUP_ORDER].reset_index()

    low_pct = float(grouped.loc[grouped["group"] == GROUP_LOW, "Percent"].iloc[0])
    why_text = NO_INTERNET_WHY_IT_MATTERS_TEMPLATE.format(low_pct=low_pct)
    return grouped[["Group", "Percent", "Households", "Color"]], build_no_internet_benchmarks(), why_text


def load_vehicle_access_panel() -> pd.DataFrame:
    if VEHICLE_ACCESS_RAW_PATH.exists():
        source_path = VEHICLE_ACCESS_RAW_PATH
    elif LEGACY_VEHICLE_ACCESS_RAW_PATH.exists():
        source_path = LEGACY_VEHICLE_ACCESS_RAW_PATH
    else:
        raise FileNotFoundError(
            "Vehicle-access source data was not found. Run `python src/fetch_acs_vehicle_access.py` first."
        )

    vehicle = pd.read_parquet(source_path).copy()
    missing_columns = [column for column in VEHICLE_REQUIRED_COLUMNS if column not in vehicle.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(
            f"{source_path.name} is missing required columns: {missing_list}. "
            "Refresh the Shelby ACS vehicle cache with `python src/fetch_acs_vehicle_access.py`."
        )

    vehicle["geoid"] = normalize_geoid(vehicle["geoid"])
    vehicle = to_numeric_columns(vehicle, ["year"] + VEHICLE_REQUIRED_COLUMNS)
    return vehicle


def build_vehicle_access_data() -> pd.DataFrame:
    vehicle = load_vehicle_access_panel()
    vehicle = vehicle.loc[vehicle["year"] == VEHICLE_CHART_YEAR].copy()
    vehicle["hh_three_plus_vehicle"] = vehicle["hh_three_vehicle"] + vehicle["hh_four_plus_vehicle"]

    tracts = gpd.read_file(FULL_TRACT_MAP_PATH)[["geoid", "geometry"]].copy()
    tracts["geoid"] = normalize_geoid(tracts["geoid"])
    tracts = tracts.to_crs(5070)

    zctas = gpd.read_file(ZCTA_SHAPEFILE_PATH)[["ZCTA5CE20", "geometry"]].copy()
    zctas["ZCTA5CE20"] = zctas["ZCTA5CE20"].astype("string")
    zctas = zctas.loc[zctas["ZCTA5CE20"].isin(TARGET_ZCTAS)].copy().to_crs(5070)

    tract_vehicle = tracts.merge(
        vehicle[
            [
                "geoid",
                "hh_total",
                "hh_no_vehicle",
                "hh_one_vehicle",
                "hh_two_vehicle",
                "hh_three_plus_vehicle",
            ]
        ],
        on="geoid",
        how="inner",
        validate="1:1",
    )
    tract_vehicle["tract_area_sq_m"] = tract_vehicle.geometry.area

    overlay = gpd.overlay(tract_vehicle, zctas, how="intersection", keep_geom_type=False)
    overlay["overlap_share"] = overlay.geometry.area / overlay["tract_area_sq_m"]
    for column in [
        "hh_total",
        "hh_no_vehicle",
        "hh_one_vehicle",
        "hh_two_vehicle",
        "hh_three_plus_vehicle",
    ]:
        overlay[f"{column}_alloc"] = overlay[column] * overlay["overlap_share"]

    zip_summary = (
        overlay.groupby("ZCTA5CE20", as_index=False)[
            [
                "hh_total_alloc",
                "hh_no_vehicle_alloc",
                "hh_one_vehicle_alloc",
                "hh_two_vehicle_alloc",
                "hh_three_plus_vehicle_alloc",
            ]
        ]
        .sum(min_count=1)
        .copy()
    )
    zip_summary["hh_total_alloc"] = zip_summary["hh_total_alloc"].replace(0, np.nan)

    rows = []
    for category, alloc_column in [
        ("No Vehicle", "hh_no_vehicle_alloc"),
        ("1 Vehicle", "hh_one_vehicle_alloc"),
        ("2 Vehicles", "hh_two_vehicle_alloc"),
        ("3+ Vehicles", "hh_three_plus_vehicle_alloc"),
    ]:
        row = {"Vehicles Available": category}
        for zcta in TARGET_ZCTAS:
            sub = zip_summary.loc[zip_summary["ZCTA5CE20"] == zcta].copy()
            numerator = float(sub[alloc_column].iloc[0]) if not sub.empty else float("nan")
            denominator = float(sub["hh_total_alloc"].iloc[0]) if not sub.empty else float("nan")
            row[TARGET_ZCTA_LABELS[zcta]] = 100.0 * numerator / denominator if denominator and denominator > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def blend_with_white(color: str, amount: float) -> tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(base * (1 - amount) + white * amount)


def draw_pill_legend(
    fig: plt.Figure,
    items: list[tuple[str, str]],
    *,
    x0: float,
    y: float,
    height: float = 0.033,
    gap: float = 0.012,
    fontsize: float = 10,
) -> None:
    x = x0
    for label, color in items:
        width = 0.040 + 0.0053 * len(label)
        pill = FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.003,rounding_size=0.015",
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.8,
            clip_on=False,
            zorder=25,
        )
        fig.add_artist(pill)

        swatch = FancyBboxPatch(
            (x + 0.006, y - height * 0.18),
            0.010,
            height * 0.36,
            transform=fig.transFigure,
            boxstyle="round,pad=0.0,rounding_size=0.003",
            facecolor=color,
            edgecolor="none",
            clip_on=False,
            zorder=26,
        )
        fig.add_artist(swatch)

        fig.text(
            x + 0.020,
            y,
            label,
            transform=fig.transFigure,
            ha="left",
            va="center",
            fontsize=fontsize,
            color="#333333",
            zorder=27,
        )
        x += width + gap


def decorate_card_axis(
    ax: plt.Axes,
    *,
    right_y: bool = False,
    ylim: tuple[float, float] | None = None,
    yticks: list[float] | np.ndarray | None = None,
    formatter: FuncFormatter | None = None,
) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linestyle=(0, (1.2, 2.2)), linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=8, labelsize=10, colors=MUTED_TEXT)
    ax.tick_params(axis="y", length=0, labelsize=10, colors=MUTED_TEXT)
    if right_y:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    else:
        ax.yaxis.tick_left()
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if formatter is not None:
        ax.yaxis.set_major_formatter(formatter)
    border = FancyBboxPatch(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        fill=False,
        edgecolor=BORDER_COLOR,
        linewidth=1.0,
        linestyle=(0, (1.4, 2.2)),
        clip_on=False,
        zorder=15,
    )
    ax.add_patch(border)


def set_categorical_xlim(
    ax: plt.Axes,
    *,
    n_categories: int,
    left_padding: float = 0.5,
    right_padding: float = 0.5,
) -> None:
    ax.set_xlim(-left_padding, (n_categories - 1) + right_padding)


def apply_gradient_to_bar(ax: plt.Axes, bar, base_color: str, *, zorder: int = 4) -> None:
    if bar.get_height() <= 0:
        return
    top = blend_with_white(base_color, 0.22)
    bottom = mcolors.to_rgb(base_color)
    cmap = LinearSegmentedColormap.from_list("bar_grad", [top, bottom])
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    x0 = bar.get_x()
    x1 = x0 + bar.get_width()
    y0 = bar.get_y()
    y1 = y0 + bar.get_height()
    image = ax.imshow(
        gradient,
        extent=(x0, x1, y0, y1),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bicubic",
        zorder=zorder,
        clip_on=True,
    )
    image.set_clip_path(bar)
    bar.set_facecolor("none")
    bar.set_edgecolor("none")


def add_bar_value_labels(
    ax: plt.Axes,
    bars,
    values: list[float],
    colors: list[str],
    *,
    fmt: str,
    pad: float,
    fontsize: float = 8.5,
    inside: bool = False,
) -> None:
    for bar, value, color in zip(bars, values, colors):
        if np.isnan(value):
            continue
        x = bar.get_x() + bar.get_width() / 2
        if inside:
            y = value - pad
            va = "top"
        else:
            y = value + pad
            va = "bottom"
        ax.text(
            x,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=fontsize,
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.18,rounding_size=0.12",
                "facecolor": blend_with_white(color, 0.10),
                "edgecolor": "none",
                "alpha": 0.95,
            },
            zorder=8,
        )


def build_black_share_gdf() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    full_map = gpd.read_file(FULL_TRACT_MAP_PATH).copy()
    full_map["geoid"] = normalize_geoid(full_map["geoid"])
    group_lookup = load_shelby_group_lookup()[["geoid", "group"]].copy()

    panel = load_panel().loc[:, ["geoid", "year", "share_black"]].copy()
    panel = panel.loc[panel["geoid"].str.startswith("47157") & panel["year"].isin(RECOVERY_YEARS)].copy()
    share_black = (
        panel.groupby("geoid", as_index=False)["share_black"]
        .mean()
        .rename(columns={"share_black": "pct_black"})
    )
    share_black["pct_black"] = share_black["pct_black"].clip(lower=0, upper=1)

    gdf = full_map.merge(share_black, on="geoid", how="left", validate="1:1")
    gdf = gdf.merge(group_lookup, on="geoid", how="left", validate="1:1")
    gdf["pct_black"] = pd.to_numeric(gdf["pct_black"], errors="coerce").fillna(0).clip(0, 1)
    gdf["group"] = gdf["group"].fillna(GROUP_REST)

    if gdf.crs is not None and str(gdf.crs) != "EPSG:3857":
        gdf = gdf.to_crs(3857)
    low_outline = gdf.loc[gdf["group"] == GROUP_LOW].dissolve()
    high_outline = gdf.loc[gdf["group"] == GROUP_HIGH].dissolve()
    return gdf, low_outline, high_outline


def generate_black_population_map(output_path: Path) -> Path:
    gdf, low_outline, high_outline = build_black_share_gdf()
    fig = plt.figure(figsize=(16, 10), dpi=220, facecolor="white")
    ax = fig.add_axes([0.22, 0.12, 0.64, 0.76])

    gdf.plot(
        ax=ax,
        column="pct_black",
        cmap="Blues",
        vmin=0,
        vmax=1,
        edgecolor=TRACT_EDGE,
        linewidth=0.4,
        missing_kwds={"color": "#f5f7fb", "edgecolor": TRACT_EDGE, "linewidth": 0.4},
        zorder=2,
    )
    gdf.dissolve().boundary.plot(ax=ax, color=COUNTY_OUTLINE, linewidth=2.1, zorder=4)
    low_outline.boundary.plot(ax=ax, color="white", linewidth=6.0, zorder=5)
    low_outline.boundary.plot(ax=ax, color=MAP_LOW_OUTLINE, linewidth=4.4, zorder=6)
    high_outline.boundary.plot(ax=ax, color="white", linewidth=5.6, zorder=5)
    high_outline.boundary.plot(ax=ax, color=MAP_HIGH_OUTLINE, linewidth=4.0, zorder=6)
    ax.set_axis_off()

    fig.suptitle("Shelby County Percent Black Population", fontsize=28, y=0.97)
    fig.text(
        0.5,
        0.93,
        "Tract shading shows percent Black population using ACS tract averages for 2022-2024.",
        ha="center",
        va="center",
        fontsize=15,
        color=MUTED_TEXT,
    )

    handles = [
        Line2D([0], [0], color=MAP_LOW_OUTLINE, linewidth=4.4, label="Selected low-growth community outline"),
        Line2D([0], [0], color=MAP_HIGH_OUTLINE, linewidth=4.0, label="High-growth comparator outline"),
    ]
    legend = ax.legend(
        handles=handles,
        title="Community outlines",
        loc="upper left",
        bbox_to_anchor=(-0.18, 0.96),
        frameon=True,
        facecolor="white",
        edgecolor="#d4dae3",
        framealpha=0.96,
        fontsize=11,
        title_fontsize=12,
    )
    ax.add_artist(legend)

    add_why_it_matters_callout(
        fig,
        bounds=(0.03, 0.53, 0.22, 0.22),
        body=BLACK_MAP_WHY_IT_MATTERS,
        wrap_width=32,
        title_fontsize=17,
        body_fontsize=10.8,
    )

    cax = fig.add_axes([0.88, 0.30, 0.03, 0.46])
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="Blues")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("% Black residents", fontsize=16, labelpad=12)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cbar.ax.tick_params(labelsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def generate_demographic_chart(output_path: Path) -> Path:
    df = pd.read_csv(DEMOGRAPHIC_COMPARE_PATH).copy()
    group_map = {
        GROUP_LOW: "Our Shelby Community",
        GROUP_REST: "Rest of Shelby County",
        GROUP_HIGH: "High-Growth Shelby Community",
    }
    plot_df = (
        df.loc[df["group"].isin(GROUP_ORDER), ["group", "Black", "White", "Asian", "Other / remaining"]]
        .copy()
        .set_index("group")
        .loc[GROUP_ORDER]
        .rename(index=group_map)
    )
    plot_df = (plot_df * 100).round(0)
    plot_df.columns = ["Black", "White", "Asian", "Other"]

    colors = {
        "Our Shelby Community": LOW_RED,
        "Rest of Shelby County": REST_GRAY,
        "High-Growth Shelby Community": HIGH_GREEN,
    }

    fig = plt.figure(figsize=(11, 8), dpi=220, facecolor="white")
    ax = fig.add_axes([0.07, 0.12, 0.86, 0.76])
    categories = plot_df.columns.tolist()
    x = np.arange(len(categories))
    bar_width = 0.22
    order = list(plot_df.index)
    offsets = np.array([-bar_width, 0, bar_width])

    bar_groups = []
    for idx, group in enumerate(order):
        values = plot_df.loc[group].tolist()
        bars = ax.bar(x + offsets[idx], values, width=bar_width * 0.9, color=colors[group], zorder=3)
        for bar in bars:
            apply_gradient_to_bar(ax, bar, colors[group], zorder=4)
        add_bar_value_labels(ax, bars, values, [colors[group]] * len(values), fmt="{:.0f}", pad=1.2, fontsize=8.4)
        bar_groups.append(bars)

    decorate_card_axis(ax, right_y=True, ylim=(0, 80), yticks=np.arange(0, 81, 10))
    set_categorical_xlim(ax, n_categories=len(categories), left_padding=0.55, right_padding=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_xlabel("Demographic", fontsize=10, color=MUTED_TEXT)

    draw_pill_legend(
        fig,
        [
            ("Our Shelby Community", LOW_RED),
            ("Rest of Shelby County", REST_GRAY),
            ("High-Growth Shelby Community", HIGH_GREEN),
        ],
        x0=0.06,
        y=0.95,
        fontsize=9.3,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def generate_economic_benchmark_chart(output_path: Path) -> Path:
    df = build_economic_benchmark_data()
    series_order = ["Low-IGS Shelby", "High-Growth Shelby", "Tennessee", "U.S."]
    colors = {
        "Low-IGS Shelby": LOW_RED,
        "High-Growth Shelby": HIGH_GREEN,
        "Tennessee": TN_GREEN,
        "U.S.": US_RED,
    }
    fig = plt.figure(figsize=(12.8, 6.6), dpi=220, facecolor="white")
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.76])
    x = np.arange(len(df))
    bar_width = 0.18
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(series_order))

    for offset, series in zip(offsets, series_order):
        values = df[series].tolist()
        bars = ax.bar(x + offset, values, width=bar_width * 0.86, color=colors[series], zorder=3)
        for bar in bars:
            apply_gradient_to_bar(ax, bar, colors[series], zorder=4)
        add_bar_value_labels(ax, bars, values, [colors[series]] * len(values), fmt="{:.1f}", pad=0.8, fontsize=7.9)

    decorate_card_axis(ax, right_y=False, ylim=(0, 70), yticks=np.arange(0, 71, 10))
    set_categorical_xlim(ax, n_categories=len(df), left_padding=0.55, right_padding=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Metric"].tolist(), fontsize=10)
    ax.set_xlabel("Metric", fontsize=10, color=MUTED_TEXT)
    ax.text(-0.02, 1.01, "%", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=MUTED_TEXT)

    draw_pill_legend(
        fig,
        [
            ("Low-IGS Shelby", LOW_RED),
            ("High-Growth Shelby", HIGH_GREEN),
            ("Tennessee", TN_GREEN),
            ("U.S.", US_RED),
        ],
        x0=0.03,
        y=0.95,
        fontsize=9.1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def generate_median_income_chart(output_path: Path) -> Path:
    df = build_median_income_data()
    colors = [LOW_RED, HIGH_GREEN, TN_GREEN, US_RED]
    fig = plt.figure(figsize=(10.5, 6.8), dpi=220, facecolor="white")
    ax = fig.add_axes([0.07, 0.12, 0.86, 0.76])
    x = np.arange(len(df))
    bars = ax.bar(x, df["Median Income"].tolist(), width=0.88, color=colors, zorder=3)
    for bar, color in zip(bars, colors):
        apply_gradient_to_bar(ax, bar, color, zorder=4)
    add_bar_value_labels(ax, bars, df["Median Income"].tolist(), colors, fmt="{:.1f}k", pad=2.8, fontsize=8.2)

    decorate_card_axis(ax, right_y=False, ylim=(0, 130), yticks=np.arange(0, 131, 20))
    set_categorical_xlim(ax, n_categories=len(df), left_padding=0.65, right_padding=0.65)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value)}k" if value else "0"))
    ax.set_xticks([1.5])
    ax.set_xticklabels(["Median Income"], fontsize=10)
    ax.set_ylabel("$", rotation=0, labelpad=10, fontsize=10, color=MUTED_TEXT)

    draw_pill_legend(
        fig,
        [
            ("Low-IGS Shelby", LOW_RED),
            ("High-IGS Shelby", HIGH_GREEN),
            ("Tennessee", TN_GREEN),
            ("U.S.", US_RED),
        ],
        x0=0.03,
        y=0.95,
        fontsize=8.7,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def generate_health_chart(output_path: Path) -> Path:
    df = build_health_comparison_data()
    series_order = ["Low-IGS Shelby", "Rest of Shelby County", "High-IGS Shelby"]
    colors = {
        "Low-IGS Shelby": LOW_RED,
        "Rest of Shelby County": REST_GRAY,
        "High-IGS Shelby": HIGH_GREEN,
    }
    fig = plt.figure(figsize=(11, 7.4), dpi=220, facecolor="white")
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.76])
    x = np.arange(len(df))
    bar_width = 0.22
    offsets = np.array([-bar_width, 0, bar_width])

    for offset, series in zip(offsets, series_order):
        values = df[series].tolist()
        bars = ax.bar(x + offset, values, width=bar_width * 0.88, color=colors[series], zorder=3)
        for bar in bars:
            apply_gradient_to_bar(ax, bar, colors[series], zorder=4)
        add_bar_value_labels(ax, bars, values, [colors[series]] * len(values), fmt="{:.1f}", pad=0.8, fontsize=7.8)

    decorate_card_axis(ax, right_y=True, ylim=(0, 60), yticks=np.arange(0, 61, 10))
    set_categorical_xlim(ax, n_categories=len(df), left_padding=0.60, right_padding=0.60)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Indicator"].tolist(), fontsize=10)
    ax.set_xlabel("Indicator", fontsize=10, color=MUTED_TEXT)

    draw_pill_legend(
        fig,
        [
            ("Low-IGS Shelby", LOW_RED),
            ("Rest of Shelby County", REST_GRAY),
            ("High-IGS Shelby", HIGH_GREEN),
        ],
        x0=0.04,
        y=0.95,
        fontsize=8.8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def generate_vehicle_access_chart(output_path: Path) -> Path:
    df = build_vehicle_access_data()
    series_order = ["38109 (Low-growth)", "38116 (Low-growth)", "38117 (High-growth)"]
    colors = {
        "38109 (Low-growth)": LOW_RED,
        "38116 (Low-growth)": LOW_RED_DARK,
        "38117 (High-growth)": HIGH_GREEN,
    }
    fig = plt.figure(figsize=(10, 6.8), dpi=220, facecolor="white")
    ax = fig.add_axes([0.06, 0.15, 0.88, 0.70])
    x = np.arange(len(df))
    bar_width = 0.20
    offsets = np.array([-bar_width, 0, bar_width])

    for offset, series in zip(offsets, series_order):
        values = df[series].tolist()
        bars = ax.bar(x + offset, values, width=bar_width * 0.88, color=colors[series], zorder=3)
        for bar in bars:
            apply_gradient_to_bar(ax, bar, colors[series], zorder=4)

    max_value = float(np.nanmax(df[series_order].to_numpy()))
    upper = max(55, int(np.ceil(max_value / 5.0) * 5))
    decorate_card_axis(ax, right_y=True, ylim=(0, upper), yticks=np.arange(0, upper + 1, 10))
    set_categorical_xlim(ax, n_categories=len(df), left_padding=0.60, right_padding=0.60)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Vehicles Available"].tolist(), fontsize=10)
    ax.set_xlabel("Vehicles Available", fontsize=10, color=MUTED_TEXT)

    fig.suptitle("Vehicle Access by zip-code+", fontsize=14, y=0.93, color="#4c4c4c", fontweight="bold")
    draw_pill_legend(
        fig,
        [
            ("38109 (Low-growth)", LOW_RED),
            ("38116 (Low-growth)", LOW_RED_DARK),
            ("38117 (High-growth)", HIGH_GREEN),
        ],
        x0=0.04,
        y=0.88,
        fontsize=8.3,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def generate_no_internet_chart(output_path: Path) -> Path:
    df, benchmarks, why_text = build_no_internet_chart_data()
    fig = plt.figure(figsize=(14.5, 8.3), dpi=220, facecolor="white")
    ax = fig.add_axes([0.08, 0.16, 0.78, 0.64])
    x = np.arange(len(df))
    colors = df["Color"].tolist()
    bars = ax.bar(x, df["Percent"].tolist(), width=0.55, color=colors, zorder=3)
    for bar, color in zip(bars, colors):
        apply_gradient_to_bar(ax, bar, color, zorder=4)
    add_bar_value_labels(ax, bars, df["Percent"].tolist(), colors, fmt="{:.1f}%", pad=0.25, fontsize=8.7)

    for label, (value, color) in benchmarks.items():
        ax.axhline(value, color=color, linewidth=1.4, linestyle=(0, (6, 6)), zorder=2)
        ax.text(
            2.55,
            value + 0.18,
            f"{label}: {value:.1f}%",
            color=color,
            fontsize=10,
            ha="right",
            va="bottom",
            zorder=8,
        )

    ax.set_xlim(-0.5, 2.6)
    decorate_card_axis(ax, right_y=False, ylim=(0, 25), yticks=np.arange(0, 26, 5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value)}%"))
    ax.set_ylabel("Percent", fontsize=14, color="#111111")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Group"].tolist(), fontsize=11)
    ax.tick_params(axis="y", labelsize=11, colors="#111111")
    ax.set_facecolor("white")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for x_pos, households in zip(x, df["Households"].tolist()):
        ax.text(
            x_pos,
            -0.16,
            f"{households:,.0f} households",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            color="#1d4ed8",
        )

    fig.suptitle("Shelby County Households Without Internet", fontsize=29, y=0.96)
    add_why_it_matters_callout(
        fig,
        bounds=(0.39, 0.69, 0.54, 0.19),
        body=why_text,
        wrap_width=58,
        title_fontsize=21,
        body_fontsize=11.8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def render_visual(name: str, output_dir: Path) -> Path:
    output_path = output_dir / OUTPUT_FILES[name]
    if name == "black-map":
        return generate_black_population_map(output_path)
    if name == "demographic":
        return generate_demographic_chart(output_path)
    if name == "economic":
        return generate_economic_benchmark_chart(output_path)
    if name == "median-income":
        return generate_median_income_chart(output_path)
    if name == "health":
        return generate_health_chart(output_path)
    if name == "vehicle":
        return generate_vehicle_access_chart(output_path)
    if name == "no-internet":
        return generate_no_internet_chart(output_path)
    raise ValueError(f"Unsupported visual name: {name}")


def main(selected: list[str] | None = None, output_dir: Path | None = None) -> list[Path]:
    chosen = selected or list(OUTPUT_FILES)
    final_output_dir = ensure_output_dir(output_dir or PRESENTATION_READY_OUTPUT_DIR)
    written: list[Path] = []
    for name in chosen:
        written.append(render_visual(name, final_output_dir))
    return written


if __name__ == "__main__":
    args = parse_args()
    paths = main(selected=args.only, output_dir=args.output_dir)
    for path in paths:
        print(path)
