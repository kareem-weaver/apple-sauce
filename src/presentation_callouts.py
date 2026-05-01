from __future__ import annotations

import textwrap

from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

CALLOUT_EDGE_COLOR = "#6f8ec7"
CALLOUT_TITLE_COLOR = "#0f4a92"
CALLOUT_ICON_COLOR = "#0f4a92"
CALLOUT_BODY_COLOR = "#1f2937"


def add_why_it_matters_callout(
    fig,
    *,
    bounds: tuple[float, float, float, float],
    body: str,
    title: str = "Why it matters",
    wrap_width: int = 44,
    title_fontsize: float = 18,
    body_fontsize: float = 11.5,
    edge_color: str = CALLOUT_EDGE_COLOR,
    title_color: str = CALLOUT_TITLE_COLOR,
    icon_color: str = CALLOUT_ICON_COLOR,
    body_color: str = CALLOUT_BODY_COLOR,
) -> None:
    x0, y0, width, height = bounds
    callout = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor=edge_color,
        linewidth=1.15,
        clip_on=False,
        zorder=20,
    )
    fig.add_artist(callout)

    icon_radius = min(width, height) * 0.12
    icon_center = (x0 + width * 0.09, y0 + height * 0.85)
    icon_circle = Circle(
        icon_center,
        icon_radius,
        transform=fig.transFigure,
        facecolor=icon_color,
        edgecolor="none",
        clip_on=False,
        zorder=21,
    )
    fig.add_artist(icon_circle)

    bar_width = icon_radius * 0.26
    gap = icon_radius * 0.16
    base_y = icon_center[1] - icon_radius * 0.58
    heights = [icon_radius * 0.62, icon_radius * 0.95, icon_radius * 1.34]
    starts = [
        icon_center[0] - bar_width - gap,
        icon_center[0] - bar_width / 2,
        icon_center[0] + gap,
    ]
    for x_bar, height_bar in zip(starts, heights):
        fig.add_artist(
            Rectangle(
                (x_bar, base_y),
                bar_width,
                height_bar,
                transform=fig.transFigure,
                facecolor="white",
                edgecolor="white",
                linewidth=0.4,
                clip_on=False,
                zorder=22,
            )
        )

    title_x = x0 + width * 0.20
    title_y = y0 + height * 0.86
    body_y = y0 + height * 0.73

    fig.text(
        title_x,
        title_y,
        title,
        transform=fig.transFigure,
        ha="left",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        color=title_color,
        zorder=23,
    )
    fig.text(
        title_x,
        body_y,
        textwrap.fill(body.strip(), width=wrap_width),
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=body_fontsize,
        color=body_color,
        linespacing=1.45,
        zorder=23,
    )
