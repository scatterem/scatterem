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
from dataclasses import dataclass

import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

__all__ = ["nice_bar_length", "draw_scale_bar"]

#: Mantissas a reader can scale by eye.
_LADDER: tuple[float, ...] = (1.0, 2.0, 5.0)

#: Largest share of the field of view the bar may occupy.
_MAX_SPAN_FRACTION = 1.0 / 3.0

#: Width of the bar that marks uncalibrated data, as a fraction of the field.
#: The bar defines its own unit: one fifth of the field is "1 a.u.".
_AU_FRACTION = 0.2

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


@dataclass(frozen=True, slots=True)
class _ScaleBarLayout:
    """Resolved geometry for a scale bar, in axes fractions."""

    bar_length: float
    unit_label: str
    x0: float
    y0: float
    width_frac: float
    thickness: float
    label_direction: float


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


#: Metadata spellings of the angstrom units, drawn with the proper symbol.
_UNIT_LABELS = {"A": "Å", "A^-1": "Å⁻¹", "1/A": "1/Å"}


def _format_label(length: float, unit: str) -> str:
    """Render the length without trailing noise: ``5 nm``, not ``5.00 nm``.

    The unit tokens "A", "A^-1" and "1/A" read "Å", "Å⁻¹" and "1/Å".
    """
    unit = _UNIT_LABELS.get(unit, unit)
    if abs(length - round(length)) < 1e-9:
        return f"{round(length):g} {unit}"
    return f"{length:.4g} {unit}"


def _scale_bar_figure_height_inches(ax: Axes) -> float:
    """Read figure height for automatic label sizing; SubFigure lacks this API."""
    figure = ax.figure
    if not isinstance(figure, Figure):
        raise AttributeError("automatic scale-bar font sizing requires a Figure")
    return figure.get_size_inches()[1]


def _resolve_scale_bar_layout(
    n_pixels: float,
    pixel_size: float | None,
    unit: str | None,
    length: float | None,
    location: str | int,
    thickness: float,
    margin: float,
) -> _ScaleBarLayout | None:
    """Validate inputs and compute axes-fraction geometry for the bar.

    Returns ``None`` when the geometry is degenerate (nothing should be drawn).
    """
    try:
        corner = _ALIASES[location.lower() if isinstance(location, str) else location]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown scale-bar location {location!r}; expected one of "
            f"{sorted(k for k in _ALIASES if isinstance(k, str))}"
        ) from None

    if pixel_size is not None and unit is not None:
        scale, unit_label = float(pixel_size), unit
    else:
        # Uncalibrated: the bar is "1 a.u." and one fifth of the field wide,
        # which is exactly as meaningful as the calibration of the data.
        usable = math.isfinite(n_pixels) and n_pixels > 0
        scale = 1.0 / (_AU_FRACTION * float(n_pixels)) if usable else 1.0
        unit_label, length = "a.u.", 1.0

    if (
        not math.isfinite(n_pixels)
        or n_pixels <= 0
        or not math.isfinite(scale)
        or scale <= 0
    ):
        return None

    span = float(n_pixels) * scale
    bar_length = float(length) if length is not None else nice_bar_length(span)
    if not math.isfinite(bar_length) or bar_length <= 0:
        return None

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
    return _ScaleBarLayout(
        bar_length=bar_length,
        unit_label=unit_label,
        x0=x0,
        y0=y0,
        width_frac=width_frac,
        thickness=thickness,
        label_direction=label_direction,
    )


def draw_scale_bar(
    ax: Axes,
    n_pixels: float,
    pixel_size: float | None = None,
    unit: str | None = None,
    length: float | None = None,
    location: str | int = "lower right",
    color: str = "white",
    thickness: float = 0.014,
    margin: float = 0.035,
    label: bool = True,
    fontsize: float | None = None,
    outline: bool = True,
) -> float:
    """Draw a scale bar on ``ax`` and return the length it represents.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes holding the image. Its data limits are not read (the bar sits in
        axes fractions), but ``n_pixels`` must describe the displayed array.
    n_pixels : float
        Width of the displayed array, in pixels.
    pixel_size : float, optional
        Physical size of one pixel. None marks the image as uncalibrated: the
        bar is still drawn, one fifth of the field wide, labelled "1 a.u.".
    unit : str, optional
        Unit of ``pixel_size``, for example ``"nm"``. None is uncalibrated, as
        for ``pixel_size``.
    length : float, optional
        Bar length in physical units. None picks one with
        :func:`nice_bar_length`. Ignored for uncalibrated data.
    location : str or int, default="lower right"
        Which corner: ``"lower right"`` and the like, a compass name, or a
        matplotlib legend corner code.
    color : str, default="white"
        Bar and label colour.
    thickness : float, default=0.014
        Bar height, as a fraction of the axes height.
    margin : float, default=0.035
        Gap between the bar and the axes edges, as a fraction of the axes
        (``0.035`` is 3.5 % in from the corner).
    label : bool, default=True
        Write the length next to the bar.
    fontsize : float, optional
        Label size in points. None scales it to the bar thickness.
    outline : bool, default=True
        Stroke the bar and label in black, so a white bar stays legible over a
        light region of the image.

    Returns
    -------
    float
        The length drawn, in physical units; ``0.0`` if the geometry was
        degenerate and nothing was drawn.
    """
    layout = _resolve_scale_bar_layout(
        n_pixels, pixel_size, unit, length, location, thickness, margin
    )
    if layout is None:
        return 0.0

    if label and fontsize is None:
        # Scale with the bar so a thick bar on a large figure is not labelled
        # in 10pt. Derived from the axes height in points, which is known
        # without a draw, unlike the rendered extent.
        axes_height_pt = (
            _scale_bar_figure_height_inches(ax) * 72.0 * ax.get_position().height
        )
        fontsize = min(30.0, max(6.0, axes_height_pt * layout.thickness * 1.7))

    # The halo follows the font, so small labels do not turn into blobs.
    halo = 2.5 if fontsize is None else min(2.5, max(0.8, 0.25 * fontsize))
    effects: list[path_effects.AbstractPathEffect] | None = (
        [path_effects.withStroke(linewidth=halo, foreground="black")]
        if outline
        else None
    )

    bar = Rectangle(
        (layout.x0, layout.y0),
        layout.width_frac,
        layout.thickness,
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
        # Sit the text just outside the bar, on the side away from the edge.
        gap = layout.thickness * 0.6
        y_text = (
            layout.y0 + layout.thickness + gap
            if layout.label_direction > 0
            else layout.y0 - gap
        )
        text = ax.text(
            layout.x0 + layout.width_frac / 2.0,
            y_text,
            _format_label(layout.bar_length, layout.unit_label),
            transform=ax.transAxes,
            color=color,
            ha="center",
            va="bottom" if layout.label_direction > 0 else "top",
            fontsize=fontsize,
            zorder=6,
            clip_on=False,
        )
        if effects:
            text.set_path_effects(effects)

    return layout.bar_length
