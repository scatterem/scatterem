"""Figures of the scan-rotation search and of scan maps (CoM, curl, potential).

Both functions return a new open :class:`matplotlib.figure.Figure` and never
call ``plt.show()``. Colours come from :mod:`scatterem.vis.colors`, scale bars
from :mod:`scatterem.vis.scale_bar`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from scatterem.data.calibration._rotation_solver import RotationSearch
from scatterem.vis.colors import ROLES, colormap_for_role, resolve_display
from scatterem.vis.scale_bar import draw_scale_bar

__all__ = ["angle_search_figure", "plot_scan_maps"]

_LABELS = {
    "axes_kept": "as measured",
    "axes_exchanged": "components swapped (reflection)",
}
_STYLES = {"axes_kept": "-", "axes_exchanged": "--"}
_UNITS = {"A": "Å"}


def _extremum_text(angles: np.ndarray, curve: np.ndarray, curve_kind: str) -> str:
    """Return "min 0.0003 at 12.3°" (or "max ...") for the finite values."""
    finite = np.isfinite(curve)
    if not finite.any():
        return "no finite values"
    idx = np.flatnonzero(finite)
    vals = curve[idx]
    k = idx[np.argmin(vals) if curve_kind == "curl" else np.argmax(vals)]
    word = "min" if curve_kind == "curl" else "max"
    return f"{word} {curve[k]:.3g} at {angles[k]:.1f}°"


def angle_search_figure(search: RotationSearch, *, figure_inches=(5.0, 3.2)) -> Figure:
    """Plot the objective curves of a scan-rotation search.

    Layout: one axes, one line per evaluated hypothesis against the trial
    angle (sorted by angle for drawing; the record is not changed). Colours
    are the first two colours of the categorical role ``"labels"``; line
    styles are solid for "as measured" and dashed for "components swapped
    (reflection)", so the hypotheses stay apart without colour vision too.
    Each legend label ends with the extremum of its curve. A dotted dark-grey
    vertical line marks ``search_angle_deg``: that line lies in the plotted
    range, and its label gives the final angle when the 180° test added a
    half-turn. NaN and ±inf values are drawn as gaps; a constant curve is a
    flat line; if no value is finite, the axes say so. No case raises.

    Parameters
    ----------
    search : RotationSearch
        The record from ``find_scan_rotation(..., keep_search=True)``.
    figure_inches : (float, float), optional
        Figure size in inches. Default ``(5.0, 3.2)``.

    Returns
    -------
    matplotlib.figure.Figure
        A new open figure with one axes.

    Raises
    ------
    ValueError
        If ``search`` is not a :class:`RotationSearch`.
    """
    if not isinstance(search, RotationSearch):
        msg = f"search must be a RotationSearch, got {type(search).__name__}"
        raise ValueError(msg)  # noqa: TRY004 - the contract asks for ValueError
    fig, ax = plt.subplots(figsize=figure_inches, layout="constrained")
    cmap = colormap_for_role("labels").cmap
    colours = cmap(np.arange(2))
    angles = np.asarray(search.angle_grid_deg, dtype=np.float64)
    order = np.argsort(angles, kind="stable")
    any_finite = False
    for i, key in enumerate(("axes_kept", "axes_exchanged")):
        if key not in search.hypothesis_curves:
            continue
        curve = np.asarray(search.hypothesis_curves[key], dtype=np.float64)[
            order
        ].copy()
        any_finite |= bool(np.isfinite(curve).any())
        curve[~np.isfinite(curve)] = np.nan
        text = _extremum_text(angles[order], curve, search.curve_kind)
        ax.plot(
            angles[order],
            curve,
            linestyle=_STYLES[key],
            color=colours[i],
            label=f"{_LABELS[key]} ({text})",
        )
    chosen = f"chosen {search.search_angle_deg:.1f}°"
    if search.rotated_180:
        chosen += f" (+180° from sign test → {search.scan_turn_deg:.1f}°)"
    ax.axvline(search.search_angle_deg, linestyle=":", color="0.25", label=chosen)
    if not any_finite:
        ax.text(
            0.5,
            0.5,
            "no finite objective values",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.set_xlabel("trial angle (°)")
    word = "curl" if search.curve_kind == "curl" else "divergence"
    ax.set_ylabel(f"normalised mean-square {word}")
    ax.legend(fontsize="small")
    return fig


def _to_numpy_map(m) -> np.ndarray:
    """Return one map as a 2-D real NumPy array; raise on bad input."""
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    arr = np.asarray(m)
    if np.iscomplexobj(arr):
        raise ValueError("maps must be real, got a complex map")
    if arr.ndim != 2:
        raise ValueError(f"every map must be 2-D, got shape {arr.shape}")
    if arr.dtype.kind not in "f":
        arr = arr.astype(np.float64)
    return arr


def _two_of(value, name: str, kind):
    """Return ``(v0, v1)`` from a scalar or a pair."""
    if isinstance(value, (str, int, float)) or np.ndim(value) == 0:
        return kind(value), kind(value)
    vals = tuple(value)
    if len(vals) != 2:
        raise ValueError(f"{name} must be one value or a pair, got {value!r}")
    return kind(vals[0]), kind(vals[1])


def _per_map(value, n: int, name: str, is_single) -> list:
    """Broadcast one setting to all maps, or check one setting per map."""
    if value is None or is_single(value):
        return [value] * n
    vals = list(value)
    if len(vals) != n:
        raise ValueError(f"{name} needs one entry per map ({n}), got {len(vals)}")
    return vals


def plot_scan_maps(
    scan_images: Sequence,
    *,
    scan_spacing,
    distance_unit,
    map_captions: Sequence[str] | None = None,
    panel_grid: tuple[int, int] | None = None,
    figure_inches: tuple[float, float] | None = None,
    colour_role=None,
    colour_map=None,
    limits=None,
    colour_key: bool = True,
    length_bar: bool = True,
) -> Figure:
    """Show 2-D scan maps in physical scan coordinates.

    Geometry: each map is drawn as stored, rows (axis 0, "x") vertical and
    columns (axis 1, "y") horizontal, ``origin="upper"``. Pixel centres sit at
    ``k · step``, so the axes span ``n · step`` along each axis, with equal
    aspect in data units. Colour: by default a map with any finite value below
    zero takes the diverging role ``"signed"`` with limits symmetric about 0,
    so the neutral colour sits at zero; other maps take ``"intensity"``. NaN
    shows in the NaN colour of the policy. Each panel has a horizontal scale
    bar along axis 1, its title and (by default) a colour bar.

    Parameters
    ----------
    scan_images : sequence of Tensor or numpy.ndarray
        Real 2-D maps; shapes may differ. Tensors may be on any device.
    scan_spacing : float or (float, float)
        Step between scan positions along axis 0 and axis 1. Required: a
        silent default would hide a missing calibration.
    distance_unit : str or (str, str)
        Length unit per axis; ``"A"`` prints as "Å". Required.
    map_captions : sequence of str, optional
        One per map. Default ``"map 0"``, ``"map 1"``, ...
    panel_grid : (int, int), optional
        ``(rows, cols)``. Default ``cols = ceil(sqrt(n))``,
        ``rows = ceil(n / cols)``. Unused panels are removed.
    figure_inches : (float, float), optional
        Override. Default ``(3.4 · cols, 3.0 · rows)`` inches.
    colour_role : str or sequence of str, optional
        Override of the automatic colour role, one for all or one per map.
    colour_map : str or Colormap, optional
        Override with an explicit map (an unscientific map warns).
    limits : (float, float) or sequence of them, optional
        Override of the display limits ``(vmin, vmax)``, one for all or one
        per map.
    colour_key : bool, optional
        Draw a colour bar per panel. Default True.
    length_bar : bool, optional
        Draw a scale bar per panel. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        A new open figure with one image panel per map.

    Raises
    ------
    ValueError
        For an empty map list, a map that is not real 2-D, a bad step or
        unit, a wrong count of captions, roles or limits, or a panel grid that
        does not fit the maps.
    """
    arrays = [_to_numpy_map(m) for m in scan_images]
    n = len(arrays)
    if n == 0:
        raise ValueError("scan_images must not be empty")
    s0, s1 = _two_of(scan_spacing, "scan_spacing", float)
    if not all(math.isfinite(s) and s > 0 for s in (s0, s1)):
        raise ValueError(
            f"scan_spacing must be finite and positive, got {scan_spacing!r}"
        )
    u0, u1 = _two_of(distance_unit, "distance_unit", str)
    captions = (
        [f"map {i}" for i in range(n)] if map_captions is None else list(map_captions)
    )
    if len(captions) != n:
        raise ValueError(f"need {n} map_captions, got {len(captions)}")
    if panel_grid is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    else:
        rows, cols = (int(v) for v in panel_grid)
        if rows < 1 or cols < 1 or rows * cols < n:
            raise ValueError(f"panel_grid {tuple(panel_grid)} does not fit {n} maps")
    roles = _per_map(colour_role, n, "colour_role", lambda v: isinstance(v, str))
    lims = _per_map(limits, n, "limits", lambda v: np.ndim(v[0]) == 0)
    for r in roles:
        if r is not None and r not in ROLES:
            raise ValueError(f"unknown role {r!r}; roles are {sorted(ROLES)}")
    if figure_inches is None:
        figure_inches = (3.4 * cols, 3.0 * rows)
    fig, axes = plt.subplots(
        rows, cols, figsize=figure_inches, layout="constrained", squeeze=False
    )
    flat = axes.ravel()
    for k, (arr, ax) in enumerate(zip(arrays, flat, strict=False)):
        r = roles[k]
        if r is None:
            r = "signed" if bool(np.any(arr[np.isfinite(arr)] < 0)) else "intensity"
        choice, display = resolve_display(colour_map, r, None, data=arr)
        if lims[k] is not None:
            norm = Normalize(float(lims[k][0]), float(lims[k][1]))
        elif isinstance(display, dict) and display.get("interval_type") == "centered":
            half = float(display.get("half_range", 1.0))
            c = float(display["vcenter"])
            norm = Normalize(c - half, c + half)
        elif isinstance(display, dict) and "vmin" in display:
            norm = Normalize(display["vmin"], display["vmax"])
        else:
            norm = None
        m0, m1 = arr.shape
        extent = (-s1 / 2, (m1 - 0.5) * s1, (m0 - 0.5) * s0, -s0 / 2)
        im = ax.imshow(
            arr,
            cmap=choice.cmap,
            norm=norm,
            origin="upper",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(captions[k])
        ax.set_xlabel(f"y ({_UNITS.get(u1, u1)})")
        ax.set_ylabel(f"x ({_UNITS.get(u0, u0)})")
        if length_bar:
            draw_scale_bar(ax, n_pixels=m1, pixel_size=s1, unit=u1)
        if colour_key:
            fig.colorbar(im, ax=ax, shrink=0.85)
    for ax in flat[n:]:
        ax.remove()
    return fig
