#!/usr/bin/env python3
"""
Shelby County internet access comparison charts.

Run:
    python shelby_internet_access_bar_chart.py
    python shelby_internet_access_bar_chart.py --metric no_internet

Outputs:
    outputs/submission_package/figures/charts/shelby_internet_access_bar_chart.png
    outputs/submission_package/figures/charts/shelby_no_internet_bar_chart.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from generate_shelby_black_population_overlay import GROUP_ORDER, OUTLINE_COLORS
from generate_shelby_internet_access_overlay import DEFAULT_YEAR, build_shelby_internet_access_gdf
from presentation_callouts import add_why_it_matters_callout
from project_paths import CHARTS_DIR

DEFAULT_OUTPUT_PATH = CHARTS_DIR / "shelby_internet_access_bar_chart.png"
DEFAULT_NO_INTERNET_OUTPUT_PATH = CHARTS_DIR / "shelby_no_internet_bar_chart.png"

SHORT_LABELS = {
    "Selected low-growth community": "Selected low-growth\ncommunity",
    "Rest of Shelby County (excluding both selected communities)": "Rest of\nShelby County",
    "Selected high-growth comparator": "High-growth\ncomparator",
}

REST_COLOR = "#d1d5db"
GRID_COLOR = "#d9e2ec"
TEXT_MUTED = "#516072"
TEXT_DARK = "#111827"
FIG_BG = "white"
AX_BG = "white"
TN_AVG_COLOR = "#1f7135"
US_AVG_COLOR = "#910D0D"
HOUSEHOLD_COUNT_COLOR = "#1d4ed8"

NO_INTERNET_BENCHMARKS = {
    "Tennessee average": 10.7,
    "U.S. average": 8.9,
}

METRIC_CONFIG = {
    "subscription": {
        "share_col": "internet_share",
        "household_col": "internet_households",
        "title": "Shelby County Internet Subscription Share",
        "subtitle_label": "% Households With Internet Subscription",
        "detail_builder": lambda row: (
            f"{row['internet_households']:,.0f} of {row['total_households']:,.0f} households\n"
            f"Broadband {row['broadband_share'] * 100:.1f}%"
        ),
        "gap_sentence": "Low-growth community trails the comparator by {gap:.1f} percentage points",
        "ymax": 100.0,
        "show_footer_note": True,
        "show_axis_subtitle": True,
    },
    "no_internet": {
        "share_col": "no_internet_share",
        "household_col": "no_internet_households",
        "title": "Shelby County Households Without Internet",
        "subtitle_label": "% Households With No Internet",
        "detail_builder": lambda row: f"{row['no_internet_households']:,.0f} households",
        "gap_sentence": "Low-growth community exceeds the comparator by {gap:.1f} percentage points",
        "ymax": 25.0,
        "show_footer_note": False,
        "show_axis_subtitle": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Shelby County internet access comparison charts."
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_CONFIG),
        default="subscription",
        help="Metric to chart.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults depend on the selected metric.",
    )
    return parser.parse_args()


def default_output_path(metric: str) -> Path:
    if metric == "no_internet":
        return DEFAULT_NO_INTERNET_OUTPUT_PATH
    return DEFAULT_OUTPUT_PATH


def group_color(group: str) -> str:
    if group == "Rest of Shelby County (excluding both selected communities)":
        return REST_COLOR
    return OUTLINE_COLORS.get(group, REST_COLOR)


def summarize_groups(gdf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        sub = gdf.loc[gdf["group"] == group].copy()
        total_households = float(pd.to_numeric(sub["total_households"], errors="coerce").fillna(0).sum())
        internet_households = float(
            pd.to_numeric(sub["internet_subscription_households"], errors="coerce").fillna(0).sum()
        )
        broadband_households = float(pd.to_numeric(sub["broadband_households"], errors="coerce").fillna(0).sum())
        no_internet_households = float(pd.to_numeric(sub["no_internet_households"], errors="coerce").fillna(0).sum())
        source_years = sorted(
            {
                int(year)
                for year in pd.to_numeric(sub["internet_source_year"], errors="coerce").dropna().tolist()
            }
        ) if "internet_source_year" in sub.columns else []
        rows.append(
            {
                "group": group,
                "short_label": SHORT_LABELS.get(group, group),
                "bar_color": group_color(group),
                "tracts_total": int(sub["geoid"].nunique()),
                "tracts_with_data": int(sub["internet_sub_share"].notna().sum()),
                "zero_household_tracts": int(sub["has_households"].fillna(False).eq(False).sum()),
                "total_households": total_households,
                "internet_households": internet_households,
                "broadband_households": broadband_households,
                "no_internet_households": no_internet_households,
                "internet_share": (internet_households / total_households) if total_households > 0 else np.nan,
                "broadband_share": (broadband_households / total_households) if total_households > 0 else np.nan,
                "no_internet_share": (no_internet_households / total_households) if total_households > 0 else np.nan,
                "source_years": source_years,
            }
        )
    return pd.DataFrame(rows)


def format_pct(x: float, _pos: int) -> str:
    return f"{x:.0f}%"


def compute_ymax(metric: str, values_pct: pd.Series) -> float:
    config = METRIC_CONFIG[metric]
    if config["ymax"] is not None:
        return float(config["ymax"])
    max_value = float(values_pct.max()) if len(values_pct) else 0.0
    return max(20.0, min(100.0, np.ceil((max_value + 4.0) / 5.0) * 5.0))


def source_period_text(summary_df: pd.DataFrame) -> str:
    source_years = sorted({year for years in summary_df["source_years"].tolist() for year in years})
    if len(source_years) == 1:
        year = source_years[0]
        return f"{year - 4}-{year} ACS 5-year estimates"
    return f"{DEFAULT_YEAR - 4}-{DEFAULT_YEAR} ACS 5-year estimates with fallback"


def build_no_internet_callout(summary_df: pd.DataFrame) -> str:
    low_growth_value = float(
        summary_df.loc[summary_df["group"] == "Selected low-growth community", "no_internet_share"].iloc[0]
    ) * 100
    return (
        "Our reduced-cost ride service depends on internet access. "
        f"Since {low_growth_value:.1f}% of households in the selected low-growth community lack internet, "
        "booking cannot rely only on an app. We plan to place booking terminals in libraries and similar "
        "community locations."
    )


def plot_chart(
    summary_df: pd.DataFrame,
    metric: str,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    config = METRIC_CONFIG[metric]
    values_pct = summary_df[config["share_col"]] * 100
    ymax = compute_ymax(metric, values_pct)
    x = np.arange(len(summary_df))

    fig, ax = plt.subplots(figsize=(15.0, 8.9), dpi=220)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    bars = ax.bar(
        x,
        values_pct,
        color=summary_df["bar_color"],
        width=0.56 if metric == "no_internet" else 0.62,
        edgecolor="none",
        zorder=3,
    )

    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.yaxis.set_major_formatter(FuncFormatter(format_pct))
    ax.tick_params(axis="x", length=0, pad=10, colors=TEXT_DARK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if config["show_axis_subtitle"]:
        ax.set_xticklabels(summary_df["short_label"], fontsize=14, color=TEXT_DARK, linespacing=1.25)
        ax.tick_params(axis="y", labelsize=13, colors=TEXT_MUTED)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.9, zorder=1)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.set_title(config["subtitle_label"], fontsize=18, color=TEXT_DARK, pad=18)
    else:
        ax.set_xticklabels(summary_df["short_label"], fontsize=15, color=TEXT_DARK, linespacing=1.2)
        ax.grid(False)
        ax.set_axisbelow(False)
        ax.set_yticks(np.arange(0, ymax + 0.1, 5))
        ax.tick_params(axis="y", labelsize=16, colors="black", width=1.2, length=6)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)
        ax.set_ylabel("Percent", fontsize=24, color="black", labelpad=12)

    if metric == "no_internet":
        benchmark_specs = [
            ("Tennessee average", NO_INTERNET_BENCHMARKS["Tennessee average"], TN_AVG_COLOR),
            ("U.S. average", NO_INTERNET_BENCHMARKS["U.S. average"], US_AVG_COLOR),
        ]
        for label, value, color in benchmark_specs:
            ax.axhline(
                value,
                color=color,
                linewidth=1.8,
                linestyle=(0, (6, 6)),
                zorder=2,
                alpha=0.95,
            )
            ax.text(
                1.01,
                value + 0.18,
                f"{label}: {value:.1f}%",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=12,
                color=color,
                bbox={"facecolor": AX_BG, "edgecolor": "none", "pad": 0.2},
                clip_on=False,
            )

    detail_y = -0.14 if metric == "subscription" else -0.145
    for bar, (_, row) in zip(bars, summary_df.iterrows()):
        height = float(bar.get_height())
        center_x = bar.get_x() + bar.get_width() / 2
        ax.text(
            center_x,
            min(height + ymax * 0.02, ymax * 0.98),
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15 if metric == "subscription" else 16,
            fontweight="bold",
            color=TEXT_DARK,
        )
        ax.text(
            center_x,
            detail_y,
            config["detail_builder"](row),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11.5 if metric == "subscription" else 11.2,
            color=TEXT_MUTED if metric == "subscription" else HOUSEHOLD_COUNT_COLOR,
            linespacing=1.2,
        )

    low_growth_value = float(
        summary_df.loc[summary_df["group"] == "Selected low-growth community", config["share_col"]].iloc[0]
    )
    comparator_value = float(
        summary_df.loc[summary_df["group"] == "Selected high-growth comparator", config["share_col"]].iloc[0]
    )
    gap_points = abs(comparator_value - low_growth_value) * 100
    zero_household_total = int(summary_df["zero_household_tracts"].sum())

    fig.suptitle(
        config["title"],
        fontsize=31,
        color=TEXT_DARK,
        y=0.965,
    )
    if metric == "subscription":
        fig.text(
            0.5,
            0.905,
            f"{source_period_text(summary_df)} · household-weighted across the 3 Shelby groups · "
            f"{config['gap_sentence'].format(gap=gap_points)}",
            ha="center",
            va="center",
            fontsize=13.5,
            color=TEXT_MUTED,
        )
    else:
        add_why_it_matters_callout(
            fig,
            bounds=(0.41, 0.69, 0.55, 0.20),
            body=build_no_internet_callout(summary_df),
            wrap_width=72,
            title_fontsize=19,
            body_fontsize=11.1,
        )

    if config["show_footer_note"]:
        fig.text(
            0.5,
            0.055,
            f"Notes: {zero_household_total} Shelby tracts report zero households in ACS and do not affect the group rates. "
            "Orange = selected low-growth community, gray = rest of Shelby County, green = high-growth comparator.",
            ha="center",
            va="center",
            fontsize=10.8,
            color=TEXT_MUTED,
        )

    bottom_margin = 0.24 if config["show_footer_note"] else 0.20
    right_margin = 0.97 if config["show_footer_note"] else 0.88
    fig.subplots_adjust(left=0.10, right=right_margin, top=0.84, bottom=bottom_margin)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, facecolor=FIG_BG)

    return fig, ax


def main(metric: str = "subscription", output_path: str | Path | None = None) -> None:
    gdf = build_shelby_internet_access_gdf()
    summary_df = summarize_groups(gdf)
    final_output_path = Path(output_path) if output_path is not None else default_output_path(metric)
    fig, _ = plot_chart(summary_df, metric=metric, output_path=final_output_path)
    plt.close(fig)

    print(f"Saved internet access chart: {final_output_path}")
    print()
    print(
        summary_df[
            ["group", "internet_share", "broadband_share", "no_internet_share", "total_households"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    args = parse_args()
    main(metric=args.metric, output_path=args.output)
