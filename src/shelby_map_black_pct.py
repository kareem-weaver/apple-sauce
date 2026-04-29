#!/usr/bin/env python3
"""
Shelby County: selected low-growth community and Black population share by tract.

Run:    python shelby_map_black_pct.py
Output: outputs/submission_package/figures/maps/shelby_map_black_pct.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from generate_shelby_black_population_overlay import (
    GROUP_COL as OVERLAY_GROUP_COL,
    build_shelby_black_share_gdf,
)
from project_paths import MAPS_DIR


DEFAULT_OUTPUT_PATH = MAPS_DIR / "shelby_map_black_pct.png"
GROUP_MAP = {
    "Selected low-growth community": "low_growth",
    "Selected high-growth comparator": "high_growth",
    "Rest of Shelby County (excluding both selected communities)": "rest",
}

OUTLINE_STYLE = {
    "low_growth": {"color": "#E8500A", "linewidth": 5.2},
    "high_growth": {"color": "#2CA02C", "linewidth": 5.2},
}


def load_and_classify() -> gpd.GeoDataFrame:
    gdf = build_shelby_black_share_gdf().copy()
    gdf["group"] = gdf[OVERLAY_GROUP_COL].map(GROUP_MAP).fillna("rest")
    gdf["pct_black"] = pd.to_numeric(gdf["pct_black"], errors="coerce").clip(lower=0, upper=1) * 100
    gdf["pop_black"] = pd.to_numeric(gdf["black_population"], errors="coerce")
    gdf["pop_total"] = pd.to_numeric(gdf["total_population"], errors="coerce")
    return gdf


def summarize_group(subset: gpd.GeoDataFrame) -> dict[str, float]:
    total_black = float(pd.to_numeric(subset["pop_black"], errors="coerce").fillna(0).sum())
    total_pop = float(pd.to_numeric(subset["pop_total"], errors="coerce").fillna(0).sum())
    avg_pct = (total_black / total_pop * 100) if total_pop > 0 else np.nan
    return {
        "n": int(len(subset)),
        "total_black": total_black,
        "avg_pct": avg_pct,
    }


def plot_black_population_choropleth(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    stats = {
        group: summarize_group(gdf[gdf["group"] == group].copy())
        for group in ["low_growth", "high_growth", "rest"]
    }

    fig, ax = plt.subplots(figsize=(15, 13))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    norm = Normalize(vmin=0, vmax=100)
    gdf.plot(
        ax=ax,
        column="pct_black",
        cmap="Blues",
        norm=norm,
        edgecolor="#CCCCCC",
        linewidth=0.35,
        missing_kwds={"color": "#EEEEEE", "edgecolor": "#CCCCCC", "linewidth": 0.35},
        zorder=2,
    )

    for group, style in OUTLINE_STYLE.items():
        dissolved = gdf[gdf["group"] == group].dissolve()
        if dissolved.empty:
            continue
        dissolved.boundary.plot(
            ax=ax,
            color=style["color"],
            linewidth=style["linewidth"],
            capstyle="round",
            joinstyle="round",
            zorder=5,
        )

    sm = ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.026, pad=0.012, shrink=0.72, aspect=22)
    cbar.set_label("% Black residents", fontsize=13, labelpad=10)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cbar.ax.tick_params(labelsize=11)

    ax.legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                color=OUTLINE_STYLE["low_growth"]["color"],
                linewidth=3.8,
                label="Selected low-growth community outline",
            ),
            mlines.Line2D(
                [],
                [],
                color=OUTLINE_STYLE["high_growth"]["color"],
                linewidth=3.8,
                label="High-growth comparator outline",
            ),
        ],
        title="Community outlines",
        title_fontsize=12,
        loc="upper left",
        fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor="#AAAAAA",
        handlelength=2.2,
    )

    ax.set_title(
        "Shelby County: selected low-growth community and Black population share by tract",
        fontsize=18,
        fontweight="bold",
        pad=16,
        loc="left",
    )
    fig.text(
        ax.get_position().x0,
        0.905,
        "Tract shading shows percent Black population using ACS tract averages for 2022-2024.",
        fontsize=12,
        color="#555555",
        fontstyle="italic",
        transform=fig.transFigure,
    )

    lg = stats["low_growth"]
    hg = stats["high_growth"]
    rs = stats["rest"]
    summary = "\n".join(
        [
            "Community snapshot (ACS tract averages, 2022-2024)",
            f"Selected low-growth community: {lg['avg_pct']:.0f}% Black | "
            f"{lg['total_black']:,.0f} est. Black residents | {lg['n']} tracts",
            f"Rest of Shelby County: {rs['avg_pct']:.0f}% Black | "
            f"{rs['total_black']:,.0f} est. Black residents | {rs['n']} tracts",
            f"High-growth comparator: {hg['avg_pct']:.0f}% Black | "
            f"{hg['total_black']:,.0f} est. Black residents | {hg['n']} tracts",
            "",
            "Story note",
            (
                "The selected low-growth community overlaps with the darker high-Black-share tract "
                f"corridor: {lg['avg_pct']:.0f}% Black versus {rs['avg_pct']:.0f}% in the rest of "
                f"Shelby County and {hg['avg_pct']:.0f}% in the high-growth comparator."
            ),
        ]
    )
    fig.text(
        0.03,
        0.01,
        summary,
        fontsize=10,
        color="#333333",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="#F7F7F7",
            edgecolor="#BBBBBB",
            linewidth=0.9,
        ),
    )

    ax.set_axis_off()
    plt.subplots_adjust(bottom=0.20, top=0.91, left=0.01, right=0.87)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")

    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Shelby County percent Black map with community outlines."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after rendering.",
    )
    return parser.parse_args()


def main(output_path: str | Path | None = None, show: bool = False) -> None:
    gdf = load_and_classify()
    print(f"\nGroup counts:\n{gdf['group'].value_counts().to_string()}")
    print(f"Missing pct_black: {gdf['pct_black'].isna().sum()}")
    print("\nRendering map...")
    fig, _ = plot_black_population_choropleth(gdf, output_path=output_path or DEFAULT_OUTPUT_PATH)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    main(output_path=args.output, show=args.show)
