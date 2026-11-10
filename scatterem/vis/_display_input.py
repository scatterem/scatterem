"""Turn what ``show_2d`` is given into numpy arrays of a size a display can hold.

``show_2d`` takes arrays, tensors or nested sequences of either, converts each
leaf to numpy and resolves the ``Sampling`` that drives the scale bar. A large
image (a 1524 x 1516 upsampled phase, say) sends every pixel to the figure,
and a notebook then stores megabytes per figure. :func:`bound_display_pixels`
block-averages such images to at most ``max_pixels`` per axis. A block average
does not alias, as a stride ``img[::3, ::3]`` does, and the ``Sampling`` grows
by the same factor, so the scale bar stays correct. :func:`scale_scalebar`
applies the same factor to an explicit scale-bar ``sampling``. The bound is off
by default in ``show_2d``: pixel coordinates (ticks, overlays) change with it.

The last three helpers read the grid inputs of ``show_2d_array``: the axes of
``figax``, the ``title`` of each panel, and the shape of the returned axes.
"""

import contextlib
import dataclasses
import math
import warnings
from collections.abc import Iterator, Sequence
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from scatterem.data.sampling import Sampling
from scatterem.vis.colors import (
    UnscientificColormapWarning,
    colormap_for_role,
    warn_if_unscientific,
)
from scatterem.vis.visualization_utils import ScalebarConfig


def _to_numpy(x: Any) -> NDArray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor (and any subclass)
        return x.detach().cpu().numpy()
    return np.asarray(x)


@contextlib.contextmanager
def one_cmap_warning(cmap: Any) -> Iterator[None]:
    """Warn once for an unscientific ``cmap``, then silence the panels' repeats.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap or None
        The explicit map of a ``show_2d`` call. None gives no warning.

    Yields
    ------
    None
        Inside the block, ``UnscientificColormapWarning`` is ignored.
    """
    if cmap is not None:
        warn_if_unscientific(cmap, stacklevel=5)  # the caller of show_2d_array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnscientificColormapWarning)
        yield


def _clipped_norm(
    array: Any,
    clip_percentile: tuple[float, float],
    role: str = "intensity",
    background: str = "light",
) -> dict[str, Any]:
    """Display norm for ``clip_percentile``, keeping the role's limits rules.

    A centred role gets limits symmetric about its centre. A cyclic role keeps
    one period and warns that the clip is ignored. Other roles get the
    percentile interval.
    """
    choice = colormap_for_role(role, background=background)
    if choice.cls == "cyclic":
        warnings.warn(
            "clip_percentile is ignored for cyclic data: the display keeps one period",
            UserWarning,
            stacklevel=4,
        )
        return choice.display_norm() or {}
    arr = _to_numpy(array)
    vals = np.abs(arr) if np.iscomplexobj(arr) else arr
    lo, hi = np.percentile(vals[np.isfinite(vals)], clip_percentile)
    if choice.centre is not None:
        c = float(choice.centre)
        half = float(max(abs(lo - c), abs(hi - c)))
        if half == 0.0:
            # Both percentiles at the centre: sparse data must still show, so take
            # the unclipped limits (1.0 only when every value is at the centre).
            low, high = choice.limits(vals) or (c - 1.0, c + 1.0)
            half = (high - low) / 2.0
        return {"interval_type": "centered", "vcenter": c, "half_range": half}
    return {"interval_type": "manual", "vmin": float(lo), "vmax": float(hi)}


def _convert_leaves(x: Any) -> Any:
    """Recursively convert leaf arrays/tensors to numpy, preserving list nesting."""
    if isinstance(x, (list, tuple)):
        return [_convert_leaves(e) for e in x]
    return _to_numpy(x)


def _resolve_sampling(array, sampling):
    """Return a Sampling or None. Explicit arg wins, else a sidecar on the object."""
    if sampling is None:
        meta = getattr(array, "sampling_meta", None)
        return meta if isinstance(meta, Sampling) else None
    if isinstance(sampling, Sampling):
        return sampling
    if isinstance(sampling, dict):
        return Sampling(**sampling)
    raise TypeError("sampling must be a Sampling, dict, or None")


def _leaves(x: Any) -> list[NDArray]:
    if isinstance(x, list):
        return [leaf for e in x for leaf in _leaves(e)]
    return [x] if np.ndim(x) >= 2 else []


def _block_mean(array: NDArray, factor: int) -> NDArray:
    """Mean over ``factor x factor`` blocks of the last two axes.

    The last ``H % factor`` rows and ``W % factor`` columns are dropped, so
    every output pixel is the mean of a full block.
    """
    h, w = array.shape[-2:]
    hb, wb = h // factor, w // factor
    trimmed = array[..., : hb * factor, : wb * factor]
    blocks = trimmed.reshape(*array.shape[:-2], hb, factor, wb, factor)
    return blocks.mean(axis=(-3, -1))


def _map_leaves(x: Any, factor: int) -> Any:
    if isinstance(x, list):
        return [_map_leaves(e, factor) for e in x]
    return _block_mean(x, factor) if np.ndim(x) >= 2 else x


def display_factor(arrays: Any, max_pixels: int | None) -> int:
    """Integer block factor that keeps every image axis at ``max_pixels`` or less.

    Parameters
    ----------
    arrays : numpy.ndarray or nested list of numpy.ndarray
        The converted input of ``show_2d``. Each leaf is ``(..., H, W)``.
    max_pixels : int or None
        The most pixels along either image axis; ``None`` gives factor 1.

    Returns
    -------
    int
        ``ceil(largest axis / max_pixels)``, at least 1.

    Raises
    ------
    ValueError
        If ``max_pixels`` is less than 1.
    """
    if max_pixels is None:
        return 1
    if max_pixels < 1:
        raise ValueError(f"max_display_pixels must be >= 1; got {max_pixels}")
    largest = max((max(leaf.shape[-2:]) for leaf in _leaves(arrays)), default=0)
    return max(math.ceil(largest / max_pixels), 1)


def scale_scalebar(scalebar: Any, factor: int) -> Any:
    """Multiply an explicit scale-bar ``sampling`` by the block factor.

    Parameters
    ----------
    scalebar : Any
        The ``scalebar`` argument of ``show_2d``. Only a dict or a
        :class:`ScalebarConfig` with a ``sampling`` changes.
    factor : int
        The block factor of :func:`display_factor`.

    Returns
    -------
    Any
        A copy with ``sampling * factor``, or ``scalebar`` unchanged.
    """
    if factor <= 1:
        return scalebar
    if isinstance(scalebar, ScalebarConfig) and scalebar.sampling is not None:
        return dataclasses.replace(scalebar, sampling=scalebar.sampling * factor)
    if isinstance(scalebar, dict) and scalebar.get("sampling") is not None:
        return {**scalebar, "sampling": scalebar["sampling"] * factor}
    return scalebar


def bound_display_pixels(
    arrays: Any, sampling: Sampling | None, max_pixels: int | None
) -> tuple[Any, Sampling | None]:
    """Block-average ``arrays`` so no image axis exceeds ``max_pixels``.

    One factor ``ceil(largest axis / max_pixels)`` applies to every leaf, so
    the panels keep their relative size and share one ``sampling``.

    Parameters
    ----------
    arrays : numpy.ndarray or nested list of numpy.ndarray
        The converted input of ``show_2d``. Each leaf is ``(..., H, W)``.
    sampling : Sampling or None
        The calibration of one input pixel.
    max_pixels : int or None
        The most pixels along either image axis; ``None`` keeps every pixel.

    Returns
    -------
    arrays : numpy.ndarray or nested list of numpy.ndarray
        The leaves, block-averaged when they were too large, else unchanged.
    sampling : Sampling or None
        ``sampling`` with its ``pixel_size`` multiplied by the factor.

    Raises
    ------
    ValueError
        If ``max_pixels`` is less than 1.
    """
    factor = display_factor(arrays, max_pixels)
    if factor <= 1:
        return arrays, sampling
    if sampling is not None and sampling.pixel_size is not None:
        sampling = dataclasses.replace(
            sampling, pixel_size=tuple(p * factor for p in sampling.pixel_size)
        )
    return _map_leaves(arrays, factor), sampling


def prepare_display(
    array: Any, sampling: Sampling | None, scalebar: Any, max_pixels: int | None
) -> tuple[Any, Sampling | None, Any]:
    """Convert the input of ``show_2d`` to numpy and bound its display size.

    The function converts every leaf with :func:`_convert_leaves`, then
    applies :func:`bound_display_pixels` and :func:`scale_scalebar` with one
    block factor.

    Parameters
    ----------
    array : Any
        An array, a tensor or a (nested) sequence of either.
    sampling : Sampling or None
        The calibration of one input pixel.
    scalebar : Any
        The ``scalebar`` argument of ``show_2d``.
    max_pixels : int or None
        The most pixels along either image axis; ``None`` keeps every pixel.

    Returns
    -------
    arrays, sampling, scalebar
        The bounded numpy arrays, and both samplings scaled by the block factor.
    """
    if isinstance(array, (list, tuple)):
        arrays = _convert_leaves(array)
    else:
        arrays = _to_numpy(array)
    factor = display_factor(arrays, max_pixels)
    arrays, sampling = bound_display_pixels(arrays, sampling, max_pixels)
    return arrays, sampling, scale_scalebar(scalebar, factor)


def _normalize_show_input_to_grid(
    arrays: NDArray | Sequence[NDArray] | Sequence[Sequence[NDArray]],
) -> list[list[NDArray]]:
    """Put whatever the caller passed into one shape: a list of rows.

    ``show_2d`` accepts a single array, a flat sequence, or a sequence of
    sequences; downstream code should not have to care which. A bare array
    becomes a single row of one, a flat sequence becomes a single row, and a
    nested sequence is taken as given.

    Args:
        arrays: one array, a row of them, or rows of them.

    Returns:
        A list of rows, each a list of arrays.
    """
    if isinstance(arrays, np.ndarray):
        if arrays.ndim != 3:
            return [[arrays]]
        # A stack of n slices fills the most nearly square grid with full rows:
        # the row count is the largest divisor of n that is at most sqrt(n).
        n_slices = arrays.shape[0]
        n_rows = max(
            d for d in range(1, int(np.sqrt(n_slices)) + 1) if n_slices % d == 0
        )
        n_cols = n_slices // n_rows
        return [
            arrays[first : first + n_cols].tolist()
            for first in range(0, n_slices, n_cols)
        ]
    if isinstance(arrays, Sequence) and not isinstance(arrays[0], Sequence):
        return [cast(list[NDArray], list(arrays))]
    return [cast(list[NDArray], list(row)) for row in arrays]


def _panel_axes(
    figax: tuple[Any, Any] | None,
    nrows: int,
    ncols: int,
    axsize: tuple[int, int],
) -> tuple[Any, NDArray]:
    """The figure and an ``(nrows, ncols)`` object array of its axes.

    Without ``figax``, a new figure sized ``axsize`` per panel. With it, a
    lone Axes counts as a 1 x 1 grid and a 1D array as one row. Any other
    shape than the panel grid raises ``ValueError``.
    """
    if figax is None:
        width, height = axsize
        return plt.subplots(
            nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False
        )
    fig, given = figax
    if isinstance(given, np.ndarray):
        layout = given[np.newaxis, :] if given.ndim == 1 else given
    else:
        layout = np.empty((1, 1), dtype=object)
        layout[0, 0] = given
    if layout.shape != (nrows, ncols):
        raise ValueError(
            f"figax holds axes of shape {layout.shape}, but the panels need "
            f"{(nrows, ncols)}"
        )
    return fig, layout


def _panel_title(title: Any, r: int, c: int, ncols: int) -> str | None:
    """The title of panel ``(r, c)`` from the ``title`` keyword of a grid.

    ``title`` is None, one string for every panel, a flat list read row by
    row, or a list of rows. A list that is too short gives None.
    """
    if title is None or isinstance(title, str):
        return title
    if isinstance(title[0], str):
        k = r * ncols + c
        return title[k] if k < len(title) else None
    row = title[r] if r < len(title) else ()
    return row[c] if c < len(row) else None


def _drop_unit_axes(axs: NDArray) -> Any:
    """Remove the length-1 dimensions of the axes grid; 1 x 1 gives the Axes."""
    squeezed = np.squeeze(axs)
    return squeezed.item() if squeezed.ndim == 0 else squeezed
