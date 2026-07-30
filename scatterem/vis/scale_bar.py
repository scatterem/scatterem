"""Scale bars, placed in axes coordinates.

Two pieces: pick a round length that fits the field of view, and draw it.

Both are written from the geometry rather than adapted from an existing helper,
and they differ from the usual `AnchoredSizeBar` recipe in two ways that are
deliberate:

*Placement is in axes fractions.* `AnchoredSizeBar` takes its padding in
multiples of the label font size, which reads as pixels and is not, so a caller
who passes a pixel count puts the bar in the middle of the image. Here the
margin is a fraction of the axes, which is what "3% in from the corner" means
and what it does.

*The length ladder is 1-2-5.* A bar exists to be read at a glance, so its length
should be a number a reader can multiply by eye: 1, 2, 5, 10, 20, 50. The
longest such value that fits within :data:`_MAX_SPAN_FRACTION` of the field of
view is chosen, which keeps the bar informative without letting it dominate.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

__all__ = ["nice_bar_length", "draw_scale_bar"]

#: Mantissas a reader can scale by eye.
_LADDER: Tuple[float, ...] = (1.0, 2.0, 5.0)

#: Largest share of the field of view the bar may occupy.
_MAX_SPAN_FRACTION = 1.0 / 3.0

#: Where the bar sits, as (x, y) of its outer corner in axes fractions, and the
#: sign that moves the label away from the nearest horizontal edge.
_CORNERS = {
    "lower right": (1.0, 0.0, +1.0),
    "lower left": (0.0, 0.0, +1.0),
    "upper right": (1.0, 1.0, -1.0),
    "upper left": (0.0, 1.0, -1.0),
}

#: Accepted aliases, including the matplotlib legend integer codes for the four
#: corners, so a caller that already speaks `loc=4` is not turned away.
_ALIASES = {
    "lower right": "lower right",
    "lower left": "lower left",
    "upper right": "upper right",
    "upper left": "upper left",
    "southeast": "lower right",
    "southwest": "lower left",
    "northeast": "upper right",
    "northwest": "upper left",
    1: "upper right",
    2: "upper left",
    3: "lower left",
    4: "lower right",
}


def nice_bar_length(span: float) -> float:
    """Longest 1-2-5 length that fits comfortably across ``span``.

    Args:
        span: extent of the field of view, in physical units.

    Returns:
        The chosen length in the same units, or ``0.0`` if ``span`` is not a
        usable positive number -- a degenerate axis should produce no bar rather
        than a NaN one.
    """
    if not math.isfinite(span) or span <= 0.0:
        return 0.0

    budget = span * _MAX_SPAN_FRACTION
    decade = 10.0 ** math.floor(math.log10(budget))

    # Walk the ladder upwards and keep the last value that still fits. The
    # smallest candidate is a decade below the budget, so one always does.
    chosen = decade * _LADDER[0]
    for mantissa in _LADDER:
        candidate = decade * mantissa
        if candidate <= budget:
            chosen = candidate
    return chosen


def _format_label(length: float, unit: str) -> str:
    """Render the length without trailing noise: ``5 nm``, not ``5.00 nm``."""
    if abs(length - round(length)) < 1e-9:
        return f"{round(length):g} {unit}"
    return f"{length:.4g} {unit}"


def draw_scale_bar(
    ax: Axes,
    n_pixels: float,
    pixel_size: Optional[float] = None,
    unit: Optional[str] = None,
    length: Optional[float] = None,
    location: Union[str, int] = "lower right",
    color: str = "white",
    thickness: float = 0.014,
    margin: float = 0.035,
    label: bool = True,
    fontsize: Optional[float] = None,
    outline: bool = True,
) -> float:
    """Draw a scale bar on ``ax`` and return the length it represents.

    Args:
        ax: axes holding the image. Its data limits are not consulted -- the bar
            is positioned in axes fractions -- but ``n_pixels`` must describe the
            same array that was displayed.
        n_pixels: width of the displayed array, in pixels.
        pixel_size: physical size of one pixel. ``None`` marks the image as
            uncalibrated: the bar is still drawn, labelled ``a.u.``.
        unit: unit of ``pixel_size``, e.g. ``"nm"``. ``None`` is treated as
            uncalibrated, as for ``pixel_size``.
        length: bar length in physical units. ``None`` picks one with
            :func:`nice_bar_length`.
        location: which corner, as ``"lower right"`` and friends, a compass
            name, or a matplotlib legend corner code.
        color: bar and label colour.
        thickness: bar height, as a fraction of the axes height.
        margin: gap between the bar and the axes edges, as a fraction of the
            axes. A true fraction, so ``0.035`` is 3.5% in from the corner.
        label: whether to write the length next to the bar.
        fontsize: label size in points. ``None`` scales it to the bar thickness.
        outline: stroke the bar and label in black, so a white bar stays legible
            over a light region of the image.

    Returns:
        The length drawn, in physical units, or ``0.0`` if the geometry was
        degenerate and nothing was drawn.
    """
    try:
        corner = _ALIASES[location.lower() if isinstance(location, str) else location]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown scale-bar location {location!r}; expected one of "
            f"{sorted(k for k in _ALIASES if isinstance(k, str))}"
        ) from None

    calibrated = pixel_size is not None and unit is not None
    scale = float(pixel_size) if calibrated else 1.0
    unit_label = unit if calibrated else "a.u."

    if (
        not math.isfinite(n_pixels)
        or n_pixels <= 0
        or not math.isfinite(scale)
        or scale <= 0
    ):
        return 0.0

    span = float(n_pixels) * scale
    bar_length = float(length) if length is not None else nice_bar_length(span)
    if not math.isfinite(bar_length) or bar_length <= 0:
        return 0.0

    width_frac = bar_length / span
    if width_frac > 1.0:
        raise ValueError(
            f"scale bar of {bar_length:g} {unit_label} is longer than the "
            f"{span:g} {unit_label} field of view"
        )

    x_edge, y_edge, label_direction = _CORNERS[corner]
    # x_edge is 1.0 at the right, where the bar must extend leftwards.
    x0 = x_edge - margin - width_frac if x_edge else margin
    y0 = y_edge - margin - thickness if y_edge else margin

    effects = (
        [path_effects.withStroke(linewidth=2.5, foreground="black")]
        if outline
        else None
    )

    bar = Rectangle(
        (x0, y0),
        width_frac,
        thickness,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="none",
        zorder=5,
        clip_on=False,
    )
    if effects:
        bar.set_path_effects(effects)
    ax.add_patch(bar)

    if label:
        if fontsize is None:
            # Scale with the bar so a thick bar on a large figure is not labelled
            # in 10pt. Derived from the axes height in points, which is known
            # without a draw, unlike the rendered extent.
            axes_height_pt = (
                ax.figure.get_size_inches()[1] * 72.0 * ax.get_position().height
            )
            fontsize = min(30.0, max(6.0, axes_height_pt * thickness * 1.7))

        # Sit the text just outside the bar, on the side away from the edge.
        gap = thickness * 0.6
        y_text = y0 + thickness + gap if label_direction > 0 else y0 - gap
        text = ax.text(
            x0 + width_frac / 2.0,
            y_text,
            _format_label(bar_length, unit_label),
            transform=ax.transAxes,
            color=color,
            ha="center",
            va="bottom" if label_direction > 0 else "top",
            fontsize=fontsize,
            zorder=6,
            clip_on=False,
        )
        if effects:
            text.set_path_effects(effects)

    return bar_length
