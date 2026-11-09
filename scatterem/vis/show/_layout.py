"""Figure layout: grid, size, style, panel letters."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any, cast

from scatterem.vis.show._context import FigureContext, Panel
from scatterem.vis.show._figure import ShownFigure
from scatterem.vis.style import figure_params, rc_context

MM = 1 / 25.4
PAPER_WIDTH_MM = {None: 89.0, "single": 89.0, "double": 183.0}
LETTER_SIZE = {"screen": 12, "paper": 8}


def style_rc(ctx: FigureContext) -> dict[Any, Any]:
    """rcParams for the context's style (:func:`~scatterem.vis.style.figure_params`)."""
    return figure_params(ctx.style, ctx.background)


def grid_shape(n: int, ctx: FigureContext) -> tuple[int, int]:
    """Rows and columns for ``n`` panels."""
    if n < 1:
        raise ValueError("nothing to show: the view produced no panels")
    if ctx.style == "screen":
        cols = min(n, 3)
    else:
        cols = min(n, 4 if ctx.width == "double" else 2)
    return math.ceil(n / cols), cols


def figure_size(rows: int, cols: int, ctx: FigureContext) -> tuple[float, float]:
    """Figure size in inches: 4.2 x 3.6 per panel on screen, Nature widths on paper."""
    if ctx.style == "screen":
        return 4.2 * cols, 3.6 * rows
    width = PAPER_WIDTH_MM[ctx.width] * MM
    return width, width / cols * 0.75 * rows


def select(panels: Sequence[Panel], ctx: FigureContext) -> list[Panel]:
    """The panels to draw, in the requested order."""
    if ctx.panels is None:
        return list(panels)
    by_name = {p.name: p for p in panels}
    unknown = [name for name in ctx.panels if name not in by_name]
    if unknown:
        raise ValueError(f"unknown panel(s) {unknown}; this figure has {list(by_name)}")
    return [by_name[name] for name in ctx.panels]


def _new_figure(size: tuple[float, float], gui: bool) -> ShownFigure:
    if gui:
        import matplotlib.pyplot as plt

        return cast(ShownFigure, plt.figure(figsize=size, FigureClass=ShownFigure))
    return ShownFigure(figsize=size)


def _letters(axes, ctx: FigureContext) -> None:
    for i, ax in enumerate(axes):
        ax.text(
            -0.08,
            1.03,
            "abcdefghijklmnopqrstuvwxyz"[i],
            transform=ax.transAxes,
            fontsize=LETTER_SIZE[ctx.style],
            fontweight="bold",
            ha="right",
            va="bottom",
        )


def compose(panels: Sequence[Panel], ctx: FigureContext, *, gui: bool) -> ShownFigure:
    """Draw ``panels`` into one figure with the context's style.

    Parameters
    ----------
    panels : sequence of Panel
        What the view offers.
    ctx : FigureContext
        Style, width, background and panel selection.
    gui : bool
        Create the figure through pyplot (needed for a window).

    Returns
    -------
    ShownFigure
        The finished figure; ``style_rc`` holds the rcParams used to save it.
    """
    chosen = select(panels, ctx)
    rows, cols = grid_shape(len(chosen), ctx)
    rc = style_rc(ctx)
    with rc_context(rc):
        fig = _new_figure(figure_size(rows, cols, ctx), gui)
        fig.style_rc = rc
        axes = fig.subplots(rows, cols, squeeze=False).ravel()
        for ax, panel in zip(axes, chosen, strict=False):
            panel.draw(ax, ctx)
            if ctx.style == "screen" and panel.title and not ax.get_title():
                ax.set_title(panel.title)
        for ax in axes[len(chosen) :]:
            ax.set_axis_off()
        _letters(axes[: len(chosen)], ctx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    return fig
