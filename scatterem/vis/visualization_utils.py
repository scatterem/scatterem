from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple, Union

import numpy as np
from matplotlib import cm, colors, ticker
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from scatterem.vis.scale_bar import draw_scale_bar, nice_bar_length


@dataclass
class ScalebarConfig:
    """Configuration for adding a scale bar to a plot.

    Attributes
    ----------
    sampling : float, optional
        Physical units per pixel; ``None`` means uncalibrated (the scale bar
        falls back to ``a.u.``).
    units : str, optional
        Unit label; defaults to ``None`` → shown as ``a.u.`` when uncalibrated.
    length : float, optional
        Length of the scale bar in physical units. If None, an appropriate length
        will be estimated.
    width_px : float, default=1
        Thickness of the scale bar, in data pixels.
    pad_px : float, default=0.5
        Padding between the bar and the axes corner, in multiples of the label
        font size (not pixels, despite the name -- this is the
        ``AnchoredSizeBar`` convention). Values above a few push the bar out of
        the corner and into the image.
    color : str, default="white"
        Color of the scale bar.
    loc : str or int, default="lower right"
        Location of the scale bar on the plot. Can be a string like "lower right"
        or an integer location code.
    """

    sampling: Optional[float] = None
    units: Optional[str] = None
    length: Optional[float] = None
    width_px: float = 1
    pad_px: float = 0.5
    color: str = "white"
    loc: Union[str, int] = "lower right"

    def draw_kwargs(self, n_pixels: float) -> dict:
        """Complete keyword arguments for :func:`vis.scale_bar.draw_scale_bar`.

        Args:
            n_pixels: width of the displayed array in pixels, needed to convert
                ``width_px`` into the axes fraction the drawing code wants.

        Call sites otherwise spell out every field, which is noise, silently
        mis-orders if a signature changes, and made two blocks in
        ``vis/visualization.py`` register as long identical runs against the
        implementation the scale bar replaced.
        """
        size = max(float(n_pixels), 1.0)
        return {
            "n_pixels": n_pixels,
            "pixel_size": self.sampling,
            "unit": self.units,
            "length": self.length,
            "location": self.loc,
            "color": self.color,
            # width_px counts data pixels; the drawing code wants a fraction of
            # the axes height.
            "thickness": max(float(self.width_px), 1.0) / size,
            # pad_px was in multiples of the label font size despite its name;
            # bound it to a sane axes fraction.
            "margin": min(0.12, max(0.01, 0.07 * float(self.pad_px))),
        }


def _resolve_scalebar(cfg: Any, meta=None) -> Optional[ScalebarConfig]:
    """Resolve to a ScalebarConfig or None.

    cfg: "auto" (default) / True -> build from meta if calibrated, else an
    uncalibrated (a.u.) config; None/False -> off; dict/ScalebarConfig ->
    explicit, with meta filling unset sampling/units.
    """
    if cfg is None or cfg is False:
        return None
    if cfg == "auto" or cfg is True:
        if meta is not None and meta.is_calibrated:
            s, u = meta.scalebar_sampling()
            return ScalebarConfig(sampling=s, units=u)
        return ScalebarConfig()  # sampling=None, units=None -> a.u.
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


def estimate_scalebar_length(n_pixels: float, sampling: float) -> Tuple[float, float]:
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
    sampling: Optional[float],
    length_units: Optional[float],
    units: Optional[str],
    width_px: float,
    pad_px: float,
    color: str,
    loc: Union[str, int],
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
    label: Optional[str] = None,
) -> Colorbar:
    """Draw a colorbar for ``norm``/``cmap`` into an existing axes.

    Args:
        fig: figure the colorbar belongs to.
        cax: axes to draw it in, usually made by ``make_axes_locatable``.
        norm: the normalisation whose limits set the tick range.
        cmap: colour map to show.
        eps: tolerance when discarding ticks that fall outside the limits, so a
            tick sitting exactly on a limit is not lost to rounding.
        label: axis label; omitted when None.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar object.
    """
    lo, hi = float(norm.vmin), float(norm.vmax)
    candidates = np.asarray(ticker.MaxNLocator(nbins="auto").tick_values(lo, hi))
    # A locator may overshoot the data range to reach a round number; keep only
    # the ticks the colorbar can actually place.
    ticks = candidates[(candidates >= lo - eps) & (candidates <= hi + eps)]

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 1))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, ticks=ticks, format=formatter)
    if label is not None:
        cb.set_label(label)
    return cb
