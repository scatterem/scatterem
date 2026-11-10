"""Scale-bar and colorbar helpers shared by the ``scatterem.vis`` plot functions."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from matplotlib import cm, colors, font_manager, rcParams, ticker
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scatterem._warn import warn_outside_package
from scatterem.vis.complex_color import add_phase_colorbar
from scatterem.vis.scale_bar import draw_scale_bar, nice_bar_length


class UncalibratedDataWarning(UserWarning):
    """An image has no physical calibration, so its scale bar reads "1 a.u."."""


#: Passed by grid code for the panels that must not repeat the grid's one bar.
#: It differs from False: a caller's False draws the "1 a.u." bar.
_GRID_NO_BAR: Any = object()


@dataclass
class ScalebarConfig:
    """Where and how the plot functions draw a scale bar.

    Parameters
    ----------
    sampling : float, optional
        Size of one pixel in ``units``. None draws the uncalibrated
        "1 a.u." bar.
    units : str, optional
        Unit text of the bar label. None (with no sampling) shows ``a.u.``.
    length : float, optional
        Bar length in ``units``. None picks the longest 1-2-5 length that
        fits in a third of the image width.
    width_px : float, default=1
        Thickness of the scale bar, in data pixels.
    pad_px : float, default=0.5
        Padding between the bar and the axes corner, in multiples of the label
        font size (not pixels, despite the name -- this is the
        ``AnchoredSizeBar`` convention). Values above a few push the bar out of
        the corner and into the image.
    color : str, default="white"
        Matplotlib colour of the bar and of its label.
    loc : str or int, default="lower right"
        Corner that holds the bar, as a matplotlib legend location: a name
        such as ``"upper left"`` or its integer code.
    fontsize : float, optional
        Label font size in points. ``None`` scales it with the bar thickness.
    """

    sampling: float | None = None
    units: str | None = None
    length: float | None = None
    width_px: float = 1
    pad_px: float = 0.5
    color: str = "white"
    loc: str | int = "lower right"
    fontsize: float | None = None

    def draw_kwargs(self, n_pixels: float) -> dict:
        """Complete keyword arguments for :func:`vis.scale_bar.draw_scale_bar`.

        The plot functions pass the result on as ``**kwargs``, so a new field
        needs a change here only.

        Parameters
        ----------
        n_pixels : float
            Width of the displayed array in pixels, needed to convert
            ``width_px`` into the axes fraction the drawing code wants.

        Returns
        -------
        dict
            Keyword arguments ready to pass to
            :func:`scatterem.vis.scale_bar.draw_scale_bar`.
        """
        size = max(float(n_pixels), 1.0)
        return {
            "n_pixels": n_pixels,
            "pixel_size": self.sampling,
            "unit": self.units,
            "length": self.length,
            "location": self.loc,
            "color": self.color,
            "fontsize": self.fontsize,
            # width_px counts data pixels; the drawing code wants a fraction of
            # the axes height.
            "thickness": max(float(self.width_px), 1.0) / size,
            # pad_px was in multiples of the label font size despite its name;
            # bound it to a sane axes fraction.
            "margin": min(0.12, max(0.01, 0.07 * float(self.pad_px))),
        }


def _resolve_scalebar(cfg: Any, meta=None) -> ScalebarConfig | None:
    """Resolve to a ScalebarConfig. Every spatial image gets a bar.

    cfg: "auto" (default) / True -> a physical bar from meta if calibrated,
    else the "1 a.u." bar and an UncalibratedDataWarning; None/False -> the
    "1 a.u." bar (a scale bar cannot be switched off); dict/ScalebarConfig ->
    explicit, with meta filling unset sampling/units; _GRID_NO_BAR -> None
    (internal: the grid draws its bar on another panel).
    """
    if cfg is _GRID_NO_BAR:
        return None
    if cfg is None or cfg is False:
        return ScalebarConfig()
    if cfg == "auto" or cfg is True:
        if meta is not None and meta.is_calibrated:
            s, u = meta.scalebar_sampling()
            return ScalebarConfig(sampling=s, units=u)
        # The warning skips frames in scatterem, so it points at the user's call.
        warn_outside_package(
            "the image has no calibration, so its scale bar reads '1 a.u.'; pass "
            "sampling=Sampling(pixel_size=..., units=...) to calibrate it",
            UncalibratedDataWarning,
        )
        return ScalebarConfig()
    if isinstance(cfg, dict):
        base = ScalebarConfig(**cfg)
    elif isinstance(cfg, ScalebarConfig):
        base = cfg
    else:
        raise TypeError("scalebar must be 'auto', None, bool, dict, or ScalebarConfig")
    if base.sampling is None and meta is not None and meta.is_calibrated:
        s, u = meta.scalebar_sampling()
        base = replace(base, sampling=s, units=base.units or u)
    return base


def estimate_scalebar_length(n_pixels: float, sampling: float) -> tuple[float, float]:
    """A round bar length for an image ``n_pixels`` wide at ``sampling`` per pixel.

    Thin adapter over :func:`scatterem.vis.scale_bar.nice_bar_length`, kept
    because callers want the length in pixels as well.

    Parameters
    ----------
    n_pixels : float
        Width of the displayed array, in pixels.
    sampling : float
        Physical size of one pixel.

    Returns
    -------
    tuple
        ``(length_in_units, length_in_pixels)``, or ``(0.0, 0.0)`` when the
        geometry is degenerate -- a zero or negative sampling produces no bar
        rather than a divide-by-zero and a NaN one.
    """
    if (
        not np.isfinite(sampling)
        or sampling <= 0
        or not np.isfinite(n_pixels)
        or n_pixels <= 0
    ):
        return 0.0, 0.0
    length = nice_bar_length(float(n_pixels) * float(sampling))
    return length, length / float(sampling)


def add_scalebar_to_ax(
    ax: Axes,
    array_size: float,
    sampling: float | None,
    length_units: float | None,
    units: str | None,
    width_px: float,
    pad_px: float,
    color: str,
    loc: str | int,
) -> None:
    """Positional adapter for :func:`scatterem.vis.scale_bar.draw_scale_bar`.

    The drawing lives in :mod:`scatterem.vis.scale_bar`; this signature exists
    because the figure scripts and :class:`ScalebarConfig` already speak it.

    Two arguments are reinterpreted on the way through, because their names
    describe the old ``AnchoredSizeBar`` implementation rather than the geometry:

    ``width_px``
        Bar thickness. Was a count of data pixels; now converted to the fraction
        of the axes height it corresponds to, so it means the same thing on
        screen for a square image.
    ``pad_px``
        Corner inset. Was in multiples of the label font size despite the name.
        Now read as a fraction of the axes -- ``0.5`` was the old default and
        landed a little inside the corner, so values in that range map to a
        comparable inset, while a caller who passes a pixel count no longer
        pushes the bar into the middle of the image.

    Prefer ``draw_scale_bar`` in new code: it names its arguments after what they
    do and returns the length it drew.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the scale bar into.
    array_size : float
        Width of the displayed array, in pixels.
    sampling : float or None
        Physical units per pixel; None leaves the bar uncalibrated.
    length_units : float or None
        Bar length in ``units``; None picks a round length.
    units : str or None
        Unit label shown next to the bar.
    width_px : float
        Bar thickness, in data pixels.
    pad_px : float
        Corner inset, in multiples of the label font size (legacy
        ``AnchoredSizeBar`` convention).
    color : str
        Matplotlib colour of the bar and of its label.
    loc : str or int
        Corner that holds the bar, as a matplotlib legend location.
    """
    size = max(float(array_size), 1.0)
    thickness = max(float(width_px), 1.0) / size
    # 0.5 (the historical default) -> a 3.5% inset, matching draw_scale_bar's own
    # default; a caller passing pixels gets a bounded inset instead of a bar in
    # the middle of the image.
    margin = min(0.12, max(0.01, 0.07 * float(pad_px)))
    draw_scale_bar(
        ax,
        n_pixels=array_size,
        pixel_size=sampling,
        unit=units,
        length=length_units,
        location=loc,
        color=color,
        thickness=thickness,
        margin=margin,
    )


def add_cbar_to_ax(
    fig: Figure,
    cax: Axes,
    norm: colors.Normalize,
    cmap: colors.Colormap,
    eps: float = 1e-8,
    label: str | None = None,
    ticks: Any = None,
) -> Colorbar:
    """Draw a colorbar for ``norm``/``cmap`` into an existing axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure that gets the bar.
    cax : matplotlib.axes.Axes
        The empty axes that the bar fills, for example from
        ``make_axes_locatable(ax).append_axes``.
    norm : matplotlib.colors.Normalize
        Value mapping of the image. Its ``vmin`` and ``vmax`` must be set;
        they bound the ticks.
    cmap : matplotlib.colors.Colormap
        The colour map of the image.
    eps : float, optional
        Tolerance when discarding ticks that fall outside the limits, so a
        tick sitting exactly on a limit is not lost to rounding.
        Default is 1e-8.
    label : str or None, optional
        Axis label; omitted when None. Default is None.
    ticks : sequence of float or None, optional
        Explicit ticks, written without offset or exponent. Default is None,
        which picks round ticks.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The new bar, for more styling by the caller.
    """
    if norm.vmin is None or norm.vmax is None:
        raise ValueError("norm limits are unresolved")
    lo, hi = float(norm.vmin), float(norm.vmax)
    if ticks is not None:
        formatter = ticker.FuncFormatter(lambda v, _: f"{v:.4g}")
    else:
        candidates = np.asarray(ticker.MaxNLocator(nbins="auto").tick_values(lo, hi))
        # A locator may overshoot the data range to reach a round number; keep
        # only the ticks the colorbar can actually place.
        ticks = candidates[(candidates >= lo - eps) & (candidates <= hi + eps)]
        # Outside about 0.1 to 10, the round ticks share one power of ten,
        # written once at the end of the bar.
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(mappable, cax=cax, ticks=ticks, format=formatter)
    if label is not None:
        bar.set_label(label)
    return bar


def special_ticks(data: Any, role: str) -> np.ndarray | None:
    """Colour-bar ticks for data that round ticks describe badly.

    Constant data gets one tick at its value. Data with the role ``labels`` and
    at most 20 distinct integer values gets one tick per value.

    Parameters
    ----------
    data : array_like
        The displayed values (the amplitude, for complex data).
    role : str
        Colour role of the data.

    Returns
    -------
    numpy.ndarray or None
        The ticks, or None when round ticks are fine.
    """
    values = np.asarray(data)
    if values.size == 0:
        return None
    # Look at the finite values only. No warnings filter is changed here: a change
    # clears the warnings registry and so repeats once-only warnings.
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo, hi = values.min(), values.max()
    if lo == hi:
        return np.array([lo], dtype=float)
    if role != "labels":
        return None  # only labels need the sort below
    unique = np.unique(values[np.isfinite(values)])
    if unique.size <= 20 and np.all(unique == np.round(unique)):
        return unique.astype(float)
    return None


def add_colorbars(
    fig: Figure,
    ax: Axes,
    norm: colors.Normalize,
    cmap: colors.Colormap,
    *,
    value_unit: str | None,
    is_complex: bool,
    ticks: Any = None,
) -> None:
    """Add the value colour bar to ``ax``, plus a phase bar for complex data.

    The phase bar sits far enough to the right for the tick labels of the value
    bar. The value bar says "abs" in a title above it, away from its ticks.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure the colour bars belong to.
    ax : matplotlib.axes.Axes
        Image axes; the bars are appended to its right.
    norm : matplotlib.colors.Normalize
        Normalization of the shown values. Equal limits (constant data) give
        one solid colour.
    cmap : matplotlib.colors.Colormap
        Colour map of the value bar.
    value_unit : str or None
        Unit of the values: the label of the value bar, or part of its "abs"
        title for complex data. None shows no unit.
    is_complex : bool
        Whether the image shows complex data, which adds the phase bar.
    ticks : sequence of float or None, optional
        Explicit ticks of the value bar, as in ``add_cbar_to_ax``. Default
        is None, which picks round ticks.
    """
    if norm.vmin is not None and norm.vmin == norm.vmax:
        # Constant data: one solid colour, the one the image shows.
        solid = colors.ListedColormap([cmap(0.0)], name=cmap.name)
        norm, cmap = colors.Normalize(norm.vmin - 0.5, norm.vmin + 0.5), solid
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="2.5%")
    cb = add_cbar_to_ax(
        fig, cax, norm, cmap, label=None if is_complex else value_unit, ticks=ticks
    )
    if not is_complex:
        return
    size = font_manager.FontProperties(
        size=rcParams["ytick.labelsize"]
    ).get_size_in_points()
    # Room for the tick labels of the value bar.
    pad = 0.06 + 0.04 * size
    add_phase_colorbar(fig, divider.append_axes("right", size="5%", pad=pad))
    cb.ax.set_title(f"abs [{value_unit}]" if value_unit else "abs", fontsize=size)


__all__ = [
    "ScalebarConfig",
    "UncalibratedDataWarning",
    "add_cbar_to_ax",
    "add_colorbars",
    "special_ticks",
    "add_scalebar_to_ax",
    "estimate_scalebar_length",
]
