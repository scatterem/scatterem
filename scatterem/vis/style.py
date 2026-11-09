"""Opt-in matplotlib style: the colour policy (Crameri et al. 2020) and figure sizes.

:func:`style_params` holds the colour policy (colour map, interpolation, line
colours). :func:`figure_params` adds the text and line sizes of a figure style:
``"screen"`` or ``"paper"`` (Nature: 5-7 pt sans-serif, thin lines, PDF text
that stays text). :func:`~scatterem.vis.show.show` and
:func:`~scatterem.vis.show_2d` use :func:`figure_params` for ``style=``; a
script uses :func:`use_style` for its own matplotlib panels.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.rcsetup import cycler  # matplotlib's own, typed; no extra dependency

from scatterem.vis._colour_metrics import srgb_to_lab
from scatterem.vis.colors import _BACKGROUND_RGB, register_colormaps

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

__all__ = [
    "figure_params",
    "line_colours",
    "style_params",
    "use_style",
]

#: Smallest |Delta L*| between a line colour and the background. A thin line is
#: read by its lightness contrast, so chroma does not count here.
LINE_MIN_DL = 30.0

#: The light-background style as a matplotlib style file.
STYLE_PATH = Path(__file__).with_name("scatterem.mplstyle")

#: Text and line sizes of each figure style (``show(style=...)``).
FIGURE_STYLES: dict[str, dict[str, Any]] = {
    "screen": {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    },
    "paper": {
        "font.family": "sans-serif",
        # A tuple, so that an edit of a figure_params() copy cannot change it.
        "font.sans-serif": ("Arial", "Helvetica", "DejaVu Sans"),
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.75,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    },
}


def line_colours(background: str = "light", n: int = 8) -> list[str]:
    """Categorical line colours from batlowS that stand out on ``background``.

    Parameters
    ----------
    background : {"light", "dark"}, default="light"
        The figure background.
    n : int, default=8
        Number of colours.

    Returns
    -------
    list of str
        Hex colours. Each one differs from the background by at least
        :data:`LINE_MIN_DL` in L*. Each next colour is the one farthest (in
        CIELAB) from those already picked.
    """
    register_colormaps()
    table = np.asarray(cast(ListedColormap, mpl.colormaps["batlowS"]).colors)[:, :3]
    lab = srgb_to_lab(table)
    background_l = srgb_to_lab(_BACKGROUND_RGB[background])[0, 0]
    keep = np.abs(lab[:, 0] - background_l) >= LINE_MIN_DL
    pool, rgb = lab[keep], table[keep]
    picked = [0]
    while len(picked) < min(n, len(pool)):
        gaps = np.linalg.norm(pool[:, None, :] - pool[None, picked, :], axis=2)
        picked.append(int(gaps.min(axis=1).argmax()))
    return [to_hex(c) for c in rgb[picked]]


def style_params(background: str = "light") -> dict[RcKeyType, Any]:
    """Return the rcParams of the scatterem style.

    Parameters
    ----------
    background : {"light", "dark"}, default="light"
        The figure background.

    Returns
    -------
    dict
        ``image.cmap`` (batlow), ``image.interpolation`` (none) and an
        ``axes.prop_cycle`` from :func:`line_colours`.
    """
    register_colormaps()
    return {
        "image.cmap": "batlow",
        "image.interpolation": "none",
        "axes.prop_cycle": cycler(color=line_colours(background)),
    }


def figure_params(style: str = "paper", background: str = "light") -> dict[Any, Any]:
    """Return the rcParams of a figure style, as ``show(style=...)`` uses them.

    Parameters
    ----------
    style : {"screen", "paper"}, default="paper"
        The figure style: text and line sizes for a screen or for print
        (Nature: 5-7 pt sans-serif, thin lines, PDF text that stays text).
    background : {"light", "dark"}, default="light"
        The figure background; it sets the line colours.

    Returns
    -------
    dict
        The text and line sizes of the style and ``axes.prop_cycle`` from
        :func:`line_colours`. A new dict on each call; a list value is a
        tuple, so an edit of the dict cannot change the style.

    Raises
    ------
    ValueError
        If ``style`` is not ``"screen"`` or ``"paper"``.
    """
    if style not in FIGURE_STYLES:
        raise ValueError(f"style must be one of {tuple(FIGURE_STYLES)}, not {style!r}")
    rc: dict[Any, Any] = dict(FIGURE_STYLES[style])
    rc["axes.prop_cycle"] = style_params(background)["axes.prop_cycle"]
    return rc


@contextmanager
def use_style(background: str = "light", style: str | None = None) -> Iterator[None]:
    """Use the scatterem style inside a ``with`` block.

    Parameters
    ----------
    background : {"light", "dark"}, default="light"
        The figure background.
    style : {"screen", "paper"}, optional
        Also use the text and line sizes of this figure style
        (:func:`figure_params`), for example for a script's own panels next
        to ``show_2d(..., style="paper")``. Save the figure inside the block:
        ``savefig.dpi`` and the PDF font type are read at save time.

    Yields
    ------
    None
        The previous rcParams come back when the block ends.
    """
    rc: dict[Any, Any] = dict(style_params(background))
    if style is not None:
        rc |= figure_params(style, background)
    with rc_context(rc):
        yield


@contextmanager
def rc_context(rc: dict[Any, Any]) -> Iterator[None]:
    """Use ``mpl.rc_context(rc)``, but keep ``interactive`` as the block sets it.

    The IPython inline backend starts at the first figure of a session and
    sets ``interactive`` to True. If that figure is made inside a plain
    ``rc_context``, the exit sets ``interactive`` back to False, and no later
    figure shows inline.

    Parameters
    ----------
    rc : dict
        The rcParams to use inside the block.

    Yields
    ------
    None
        The previous rcParams, except ``interactive``, come back at the end.
    """
    interactive = mpl.is_interactive()
    try:
        with mpl.rc_context(rc):
            try:
                yield
            finally:
                interactive = mpl.is_interactive()
    finally:
        mpl.interactive(interactive)


@contextmanager
def figure_style(
    style: str | None, background: str = "light", axs: Any = None
) -> Iterator[None]:
    """Use a figure style inside a ``with`` block, also on axes made before it.

    Parameters
    ----------
    style : {"screen", "paper"}, optional
        The figure style (:func:`figure_params`). None changes nothing.
    background : {"light", "dark"}, default="light"
        The figure background; it sets the line colours.
    axs : matplotlib.axes.Axes or array of them, optional
        Axes made before the block. They get the tick, axis-label and spine
        sizes of the style: matplotlib reads these only when it makes axes.

    Yields
    ------
    None
        The previous rcParams come back when the block ends.
    """
    if style is None:
        yield
        return
    with rc_context(figure_params(style, background)):
        rc = mpl.rcParams
        x_ticks = (
            rc["xtick.labelsize"],
            rc["xtick.major.width"],
            rc["xtick.major.size"],
        )
        y_ticks = (
            rc["ytick.labelsize"],
            rc["ytick.major.width"],
            rc["ytick.major.size"],
        )
        for ax in np.asarray(axs if axs is not None else [], dtype=object).ravel():
            for axis, (size, width, length) in (
                (ax.xaxis, x_ticks),
                (ax.yaxis, y_ticks),
            ):
                axis.set_tick_params(
                    which="major", labelsize=size, width=width, length=length
                )
                axis.label.set_size(rc["axes.labelsize"])
            for spine in ax.spines.values():
                spine.set_linewidth(rc["axes.linewidth"])
        yield
