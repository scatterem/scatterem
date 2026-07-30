from collections.abc import Sequence
from typing import Any, List, Optional, Tuple, Union, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from scatterem.utils.data.sampling import Sampling
from scatterem.vis.complex_color import (
    add_phase_colorbar,
    complex_to_rgba,
    overlay_to_rgba,
)
from scatterem.vis.normalization import (
    CustomNormalization,
    NormalizationConfig,
    _resolve_normalization,
    normalization_kwargs,
)
from scatterem.vis.scale_bar import draw_scale_bar
from scatterem.vis.visualization_utils import (
    ScalebarConfig,
    _resolve_scalebar,
    add_cbar_to_ax,
)


def _split_amplitude_phase(array: NDArray):
    """Separate an array into what a display needs: a magnitude and a phase.

    A real array has no phase, so ``None`` is returned for it and the caller
    renders through a colormap instead of a hue.

    Returns
    -------
    tuple
        ``(amplitude, phase)``, where ``phase`` is ``None`` for real input.
    """
    if np.iscomplexobj(array):
        return np.abs(array), np.angle(array)
    return array, None


def _show_2d(
    array: NDArray,
    *,
    norm: Optional[Union[NormalizationConfig, dict, str]] = None,
    scalebar: Optional[Union[ScalebarConfig, dict, bool, str]] = "auto",
    sampling: Optional[Sampling] = None,
    cmap: Union[str, colors.Colormap] = "gray",
    chroma_boost: float = 1.0,
    cbar: bool = False,
    figax: Optional[Tuple[Any, Any]] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
    aspect: Optional[int] = None,
    xticks: Optional[List] = [],
    yticks: Optional[List] = [],
    xticklabels: Optional[List] = [],
    yticklabels: Optional[List] = [],
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Draw a 2D array as an image, optionally with a colorbar and scale bar.

    The array must be real. Normalisation, the colour map, and the scale bar are
    each configurable, and a calibrated dataset supplies the bar's length and
    units on its own.

    Parameters
    ----------
    array : ndarray
        The 2D array to visualize. Can be real or complex.
    norm : NormalizationConfig or dict or str, optional
        Configuration for normalizing the data before visualization.
    scalebar : ScalebarConfig or dict or bool or str, optional
        Configuration for adding a scale bar to the plot.
    sampling : Sampling, optional
        Calibration metadata driving the physical scale bar.
    cmap : str or Colormap, default="gray"
        Colormap to use for real data or amplitude of complex data.
    chroma_boost : float, default=1.0
        Factor to boost color saturation when displaying complex data.
    cbar : bool, default=False
        Whether to add a colorbar to the plot.
    figax : tuple, optional
        (fig, ax) tuple to use for plotting. If None, a new figure and axes are created.
    figsize : tuple, default=(8, 8)
        Figure size in inches, used only if figax is None.
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig : Figure
        The figure that was drawn on, whether it was created here or passed in.
    ax : Axes
        The axes holding the image, for further annotation by the caller.
    """
    amplitude, angle = _split_amplitude_phase(array)
    is_complex = angle is not None

    norm_config = _resolve_normalization(norm)

    norm_obj = CustomNormalization(**normalization_kwargs(norm_config), data=amplitude)

    scaled_amplitude = norm_obj(amplitude)
    rgba = complex_to_rgba(scaled_amplitude, angle, cmap=cmap)

    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    ax.imshow(rgba, aspect=aspect)
    ax.set(
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    if cbar:
        divider = make_axes_locatable(ax)
        ax_cb_abs = divider.append_axes("right", size="5%", pad="2.5%")
        # Convert cmap to Colormap if it's a string
        cmap_obj = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
        value_unit = sampling.value_unit if sampling is not None else None
        cb_abs = add_cbar_to_ax(fig, ax_cb_abs, norm_obj, cmap_obj, label=value_unit)

        if is_complex:
            ax_cb_angle = divider.append_axes("right", size="5%", pad="10%")
            add_phase_colorbar(fig, ax_cb_angle)
            abs_label = f"abs [{value_unit}]" if value_unit is not None else "abs"
            cb_abs.set_label(abs_label, rotation=0, ha="center", va="bottom")
            cb_abs.ax.yaxis.set_label_coords(0.5, -0.05)

    scalebar_config = _resolve_scalebar(scalebar, sampling)
    if scalebar_config is not None:
        if aspect not in (None, 1, 1.0, "equal"):
            import warnings

            warnings.warn("scalebar skipped: non-unit aspect mis-scales the bar")
        else:
            draw_scale_bar(ax, **scalebar_config.draw_kwargs(rgba.shape[1]))

    return fig, ax


def _show_2d_combined(
    list_of_arrays: Sequence[NDArray],
    *,
    norm: Optional[Union[NormalizationConfig, dict, str]] = None,
    scalebar: Optional[Union[ScalebarConfig, dict, bool, str]] = "auto",
    sampling: Optional[Sampling] = None,
    cmap: Union[str, colors.Colormap] = "gray",
    chroma_boost: float = 1.0,
    cbar: bool = False,
    figax: Optional[Tuple[Any, Any]] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Overlay several 2D arrays in one image, one hue each.

    Each array is given its own hue and contributes to a pixel in proportion to
    its normalised amplitude there, so agreement between arrays reads as a
    desaturated blend and disagreement as a single dominant colour. Useful for
    comparing reconstructions of the same field of view.

    Parameters
    ----------
    list_of_arrays : sequence of ndarray
        The arrays to overlay. All must share a shape.
    norm : NormalizationConfig or dict or str, optional
        How to normalise each array before it is coloured.
    scalebar : ScalebarConfig or dict or bool or str, optional
        Scale bar to draw, or ``False`` for none.
    sampling : Sampling, optional
        Calibration metadata driving the physical scale bar.
    cmap : str or Colormap, default="gray"
        Ignored for the overlay itself, which assigns hues, but kept so the
        signature matches the single-array renderer.
    chroma_boost : float, default=1.0
        Saturation multiplier, above 1 for more vivid hues.
    cbar : bool, default=False
        Not implemented for an overlay -- there is no single value axis to label.
    figax : tuple, optional
        An existing ``(fig, ax)`` to draw into. A new figure is made if omitted.
    figsize : tuple, default=(8, 8)
        Size in inches of the figure created when ``figax`` is omitted.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : Figure
        The figure that was drawn on, whether it was created here or passed in.
    ax : Axes
        The axes holding the image, for further annotation by the caller.

    Raises
    ------
    NotImplementedError
        If cbar is True (colorbar for combined visualization not yet implemented).
    """
    norm_config = _resolve_normalization(norm)
    scalebar_config = _resolve_scalebar(scalebar, sampling)

    norm_obj = CustomNormalization(**normalization_kwargs(norm_config))

    rgba = overlay_to_rgba(list(list_of_arrays), norm=norm_obj)

    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    ax.imshow(rgba)
    ax.set(xticks=[], yticks=[], title=title)

    if cbar:
        raise NotImplementedError()

    if scalebar_config is not None:
        draw_scale_bar(ax, **scalebar_config.draw_kwargs(rgba.shape[1]))

    return fig, ax


def _normalize_show_input_to_grid(
    arrays: Union[NDArray, Sequence[NDArray], Sequence[Sequence[NDArray]]],
) -> List[List[NDArray]]:
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
        if arrays.ndim == 3:
            n_slices = arrays.shape[0]

            # Find the best divisor close to target_dim
            best_rows = 1
            best_cols = n_slices
            min_diff = abs(best_rows - best_cols)

            for i in range(1, int(np.sqrt(n_slices)) + 1):
                if n_slices % i == 0:
                    rows, cols = i, n_slices // i
                    diff = abs(rows - cols)
                    if diff < min_diff:
                        min_diff = diff
                        best_rows, best_cols = rows, cols

            # Reshape the array into the best grid
            return [
                arrays[i : i + best_cols].tolist()
                for i in range(0, n_slices, best_cols)
            ]
        else:
            return [[arrays]]
    if isinstance(arrays, Sequence) and not isinstance(arrays[0], Sequence):
        # Convert sequence to list and ensure each element is an NDArray
        return [[cast(NDArray, arr) for arr in arrays]]
    # Convert outer and inner sequences to lists, ensuring proper types
    return [[cast(NDArray, arr) for arr in row] for row in arrays]


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor (and any subclass)
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _convert_leaves(x):
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


def show_2d(
    array,
    *,
    sampling=None,
    scalebar="auto",
    figax=None,
    axsize=(4, 4),
    tight_layout=True,
    combine_images=False,
    clip_percentile: Optional[Tuple[float, float]] = None,
    **kwargs,
):
    """Display one or more 2D arrays in a grid.

    Accepts a numpy array, a torch tensor, or a (nested) sequence of either.
    ``sampling`` (a :class:`Sampling` or a dict of its fields) drives the
    physical scale bar; if omitted, the object's ``.sampling_meta`` is used if
    present, otherwise the bar falls back to ``a.u.``.

    ``scalebar``: ``"auto"`` (default) draws a bar; ``False``/``None`` disables
    it; a dict/``ScalebarConfig`` configures it explicitly.

    ``clip_percentile``: optional ``(low, high)`` percentiles (0-100). When
    given, the display vmin/vmax are computed from those percentiles of each
    array's finite values, absorbing the common
    ``p02 = quantile(...); img -= p02`` boilerplate.
    """
    meta = _resolve_sampling(array, sampling)
    if isinstance(array, (list, tuple)):
        array_np = _convert_leaves(array)
    else:
        array_np = _to_numpy(array)
    return show_2d_array(
        array_np,
        sampling=meta,
        scalebar=scalebar,
        figax=figax,
        axsize=axsize,
        tight_layout=tight_layout,
        combine_images=combine_images,
        clip_percentile=clip_percentile,
        **kwargs,
    )


def show_2d_array(
    arrays: Union[NDArray, Sequence[NDArray], Sequence[Sequence[NDArray]]],
    *,
    figax: Optional[Tuple[Any, Any]] = None,
    axsize: Tuple[int, int] = (4, 4),
    tight_layout: bool = True,
    combine_images: bool = False,
    scalebar: Any = "auto",
    sampling: Optional[Sampling] = None,
    scalebar_panel: Any = "first",
    clip_percentile: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Display one or more 2D arrays in a grid layout.

    This is the main visualization function that can display a single array,
    a list of arrays, or a grid of arrays. It supports both individual and
    combined visualization modes.

    Parameters
    ----------
    arrays : ndarray or sequence of ndarray or sequence of sequences of ndarray
        The arrays to visualize. Can be a single array, a sequence of arrays,
        or a nested sequence representing a grid of arrays.
    figax : tuple, optional
        (fig, axs) tuple to use for plotting. If None, a new figure and axes are created.
    axsize : tuple, default=(4, 4)
        Size of each subplot in inches.
    tight_layout : bool, default=True
        Whether to apply tight_layout to the figure.
    combine_images : bool, default=False
        If True and arrays is a sequence, combine all arrays into a single visualization
        using color encoding. Only works for a single row of arrays.
    scalebar : "auto", bool, dict, or ScalebarConfig, default="auto"
        Configuration for adding a scale bar to the plot.
    sampling : Sampling, optional
        Calibration metadata driving the physical scale bar.
    scalebar_panel : "first", "all", or int, default="first"
        Which panel(s) to draw the scale bar on.
    clip_percentile : tuple of (float, float), optional
        ``(low, high)`` percentiles (0-100). When given, each panel's display
        vmin/vmax are computed from those percentiles of its own finite
        values (amplitude, for complex data) and used as a manual-interval
        normalization, overriding any ``norm`` kwarg for that panel.
    **kwargs : dict
        Additional keyword arguments passed to _show_2d or _show_2d_combined.

    Returns:
        The figure, and the axes -- a single Axes for one array, otherwise a 2D
        array of them matching the grid.

    Raises:
        ValueError: if ``figax`` is given and its axes do not match the grid
            shape.
        NotImplementedError: if ``combine_images`` is requested, which this
            release does not provide.
    """
    grid = _normalize_show_input_to_grid(arrays)
    nrows = len(grid)
    ncols = max(len(row) for row in grid)

    title = kwargs.pop("title", None)

    if combine_images:
        if nrows > 1:
            raise ValueError("combine_images requires a single row of arrays")
        supported = {"norm", "cmap", "chroma_boost", "cbar", "figsize"}
        bad = set(kwargs) - supported
        if bad:
            raise ValueError(
                f"combine_images does not support per-image kwargs: {sorted(bad)}"
            )
        return _show_2d_combined(
            grid[0],
            figax=figax,
            title=title,
            scalebar=scalebar,
            sampling=sampling,
            **kwargs,
        )

    if figax is not None:
        fig, axs = figax
        if not isinstance(axs, np.ndarray):
            axs = np.array([[axs]])
        elif axs.ndim == 1:
            axs = axs.reshape(1, -1)
        if axs.shape != (nrows, ncols):
            raise ValueError()
    else:
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(axsize[0] * ncols, axsize[1] * nrows), squeeze=False
        )

    flat_index = 0
    for i, row in enumerate(grid):
        for j, array in enumerate(row):
            figax_cell = (fig, axs[i][j])
            if title is None:
                t = None
            elif isinstance(title, str):
                t = title
            elif isinstance(title[0], str):
                # Flat list of titles
                t = title[i * ncols + j] if i * ncols + j < len(title) else None
            else:
                # Grid of titles
                t = title[i][j] if i < len(title) and j < len(title[i]) else None

            draw_here = (
                scalebar_panel == "all"
                or (scalebar_panel == "first" and flat_index == 0)
                or (isinstance(scalebar_panel, int) and flat_index == scalebar_panel)
            )
            cell_kwargs = kwargs
            if clip_percentile is not None:
                arr_np = _to_numpy(array)
                vals = np.abs(arr_np) if np.iscomplexobj(arr_np) else arr_np
                finite_vals = vals[np.isfinite(vals)]
                lo, hi = np.percentile(finite_vals, clip_percentile)
                cell_kwargs = dict(kwargs)
                cell_kwargs["norm"] = {
                    "interval_type": "manual",
                    "vmin": float(lo),
                    "vmax": float(hi),
                }
            _show_2d(
                array,
                figax=figax_cell,
                title=t,
                scalebar=(scalebar if draw_here else False),
                sampling=sampling,
                **cell_kwargs,
            )
            flat_index += 1

    # Hide unused axes in incomplete rows
    for i, row in enumerate(grid):
        for j in range(len(row), ncols):
            axs[i][j].axis("off")  # type: ignore

    if tight_layout:
        fig.tight_layout()

    # Squeeze the axes to the expected shape
    if axs.shape == (1, 1):
        axs = axs[0, 0]
    elif axs.shape[0] == 1:
        axs = axs[0]
    elif axs.shape[1] == 1:
        axs = axs[:, 0]
    return fig, axs
