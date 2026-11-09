"""Reusable panels. Each draws into one Axes and follows the colour policy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from scatterem.data.sampling import Sampling
from scatterem.vis._display_input import _to_numpy
from scatterem.vis.colors import colormap_for_role
from scatterem.vis.show._context import FigureContext
from scatterem.vis.style import line_colours
from scatterem.vis.visualization import _show_2d
from scatterem.vis.visualization_utils import ScalebarConfig, _resolve_scalebar


def real_space_sampling(meta: Any) -> Sampling | None:
    """Real-space :class:`Sampling` from a 4D-STEM acquisition metadata object.

    Reads the scan step, ``meta.sampling[:2]`` with ``meta.units[:2]``. Returns
    None when ``meta`` is missing, malformed or uncalibrated.
    """
    try:
        step = np.asarray(_to_numpy(meta.sampling), dtype=float).ravel()
        units = [str(u) for u in meta.units]
        if step.size < 2 or not np.all(np.isfinite(step[:2])) or np.any(step[:2] <= 0):
            return None
        pair = tuple(units[:2]) if len(units) >= 2 else ("Å", "Å")
        return Sampling(pixel_size=(float(step[0]), float(step[1])), units=pair)
    except (AttributeError, TypeError, ValueError):
        return None


def binned_sampling(sampling: Any, factors: Sequence[int]) -> Any:
    """The calibration of an image block-averaged by ``factors``.

    Parameters
    ----------
    sampling : Sampling or None
        The calibration of the unbinned image, ``(dy, dx)``.
    factors : sequence of int
        The ``(row, column)`` block size, as ``central_sections`` returns it.

    Returns
    -------
    Sampling or None
        ``sampling`` with each pixel size times its factor; ``sampling``
        unchanged when it is None or has no pixel size.
    """
    if not isinstance(sampling, Sampling) or sampling.pixel_size is None:
        return sampling
    size = sampling.pixel_size
    scale = tuple(factors)[-len(size) :]
    head = size[: len(size) - len(scale)]
    tail = tuple(p * f for p, f in zip(size[len(head) :], scale, strict=True))
    return replace(sampling, pixel_size=head + tail)


def image(
    ax: Axes,
    array: Any,
    *,
    ctx: FigureContext,
    role: str = "intensity",
    sampling=None,
    title: str | None = None,
    norm=None,
    scalebar: ScalebarConfig | dict | bool | str | None = "auto",
) -> None:
    """An image (or complex field) with colour bar and scale bar.

    The scale-bar label has a fixed size: 8 pt on screen, 6 pt on paper.
    (``draw_scale_bar`` otherwise scales it with the bar thickness.)
    """
    if isinstance(scalebar, str) and scalebar == "auto":
        # Resolve here so an uncalibrated panel still warns; then fix the label size.
        resolved = _resolve_scalebar("auto", sampling)
        size = 6.0 if ctx.style == "paper" else 8.0
        scalebar = replace(resolved, fontsize=size) if resolved else resolved
    _show_2d(
        _to_numpy(array),
        figax=(ax.figure, ax),
        role=role,
        background=ctx.background,
        sampling=sampling,
        scalebar=scalebar,
        norm=norm,
        cbar=True,
        title=title if ctx.style == "screen" else None,
    )


def curve(
    ax: Axes,
    x: Any,
    series: Mapping[str, Any],
    *,
    ctx: FigureContext,
    xlabel: str,
    ylabel: str,
    logy: bool = False,
    title: str | None = None,
    kw: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """One or more curves against a shared x; ``kw`` gives line options by label."""
    xs = _to_numpy(x)
    for label, y in series.items():
        ax.plot(xs, _to_numpy(y), label=label, **(kw or {}).get(label, {}))
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(frameon=False)
    if title and ctx.style == "screen":
        ax.set_title(title)


def bars(
    ax: Axes,
    labels: Sequence[str],
    values: Any,
    *,
    ctx: FigureContext,
    ylabel: str,
    title: str | None = None,
) -> None:
    """A bar chart in the first line colour."""
    ax.bar(list(labels), _to_numpy(values), color=line_colours(ctx.background)[0])
    ax.set_ylabel(ylabel)
    if title and ctx.style == "screen":
        ax.set_title(title)


def positions(
    ax: Axes,
    refined: Any,
    initial: Any = None,
    *,
    ctx: FigureContext,
    title: str | None = None,
) -> None:
    """Scan positions coloured by index; arrows from initial to refined."""
    r = np.asarray(_to_numpy(refined), dtype=float)
    index = np.arange(len(r))
    cmap = colormap_for_role("intensity", background=ctx.background).cmap
    dots = ax.scatter(r[:, 1], r[:, 0], c=index, cmap=cmap, s=6)
    if initial is not None:
        i = np.asarray(_to_numpy(initial), dtype=float)
        d = r - i
        ax.quiver(
            i[:, 1],
            i[:, 0],
            d[:, 1],
            d[:, 0],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    bar = ax.figure.colorbar(dots, ax=ax)
    bar.ax.set_ylabel("scan index")
    if title and ctx.style == "screen":
        ax.set_title(title)


def montage(stack: Any, n_max: int = 9) -> np.ndarray:
    """Up to ``n_max`` evenly spaced frames of ``stack`` tiled with NaN gutters."""
    s = np.asarray(_to_numpy(stack), dtype=float)
    picks = np.unique(
        np.linspace(0, len(s) - 1, min(n_max, len(s))).round().astype(int)
    )
    frames = s[picks]
    cols = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / cols)
    h, w = frames.shape[1:]
    out = np.full((rows * (h + 1) - 1, cols * (w + 1) - 1), np.nan)
    for k, frame in enumerate(frames):
        r, c = divmod(k, cols)
        out[r * (h + 1) : r * (h + 1) + h, c * (w + 1) : c * (w + 1) + w] = frame
    return out


def table(
    ax: Axes,
    rows: Sequence[tuple[str, str]],
    *,
    ctx: FigureContext,
    title: str | None = None,
) -> None:
    """A two-column table (name, value)."""
    ax.set_axis_off()
    cells = ax.table(cellText=[[k, v] for k, v in rows], loc="center", cellLoc="left")
    cells.auto_set_font_size(False)
    cells.set_fontsize(5 if ctx.style == "paper" else 8)
    if title and ctx.style == "screen":
        ax.set_title(title)


def note(ax: Axes, text: str, *, ctx: FigureContext) -> None:
    """A text panel, for example to say which option would add a panel."""
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, transform=ax.transAxes)


def shared_norm(array: Any, low: float = 2.0, high: float = 98.0) -> dict[str, Any]:
    """A manual NormalizationConfig over the finite percentiles of ``array``."""
    values = np.asarray(_to_numpy(array))
    values = np.abs(values) if np.iscomplexobj(values) else values.astype(float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"interval_type": "manual", "vmin": 0.0, "vmax": 1.0}
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return {"interval_type": "manual", "vmin": float(lo), "vmax": float(hi)}


def radial_profile(image: Any) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``image`` in integer-radius rings about its centre."""
    a = np.asarray(_to_numpy(image), dtype=float)
    yy, xx = np.indices(a.shape)
    r = np.hypot(yy - a.shape[0] // 2, xx - a.shape[1] // 2).astype(int).ravel()
    sums = np.bincount(r, weights=np.nan_to_num(a.ravel()))
    counts = np.bincount(r)
    return np.arange(len(sums)), sums / np.maximum(counts, 1)


def radial_frequency_profile(
    image: Any, pixel_size: tuple[float, float] = (1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a centred reciprocal-space ``image`` in rings of physical ``|q|``.

    ``image`` has its zero frequency at ``(ny // 2, nx // 2)``, as after
    ``fftshift``. ``pixel_size`` is the real-space pixel size ``(dy, dx)``
    of the map. The default ``(1, 1)`` gives the normalised frequency in
    1/pixel. Rings are as wide as the finer of the two frequency steps.
    Returns the mean ``|q|`` of each non-empty ring and the mean of ``image``.
    """
    a = np.nan_to_num(np.asarray(_to_numpy(image), dtype=float))
    ny, nx = a.shape
    qy = np.fft.fftshift(np.fft.fftfreq(ny, pixel_size[0]))
    qx = np.fft.fftshift(np.fft.fftfreq(nx, pixel_size[1]))
    q = np.hypot(qy[:, None], qx[None, :]).ravel()
    width = min(1.0 / (ny * pixel_size[0]), 1.0 / (nx * pixel_size[1]))
    ring = (q / width).astype(int)
    counts = np.bincount(ring)
    keep = counts > 0
    mean_q = np.bincount(ring, weights=q)[keep] / counts[keep]
    mean_a = np.bincount(ring, weights=a.ravel())[keep] / counts[keep]
    return mean_q, mean_a
