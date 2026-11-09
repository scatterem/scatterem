"""2D array/complex-field display helpers built on matplotlib."""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from numpy.typing import NDArray

from scatterem.data.sampling import Sampling
from scatterem.vis._display_input import (
    _clipped_norm,
    _normalize_show_input_to_grid,
    _resolve_sampling,
    one_cmap_warning,
    prepare_display,
)
from scatterem.vis.colors import resolve_display
from scatterem.vis.complex_color import (
    _amplitude_colormap,
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
from scatterem.vis.style import figure_style
from scatterem.vis.visualization_utils import (
    _GRID_NO_BAR,
    ScalebarConfig,
    _resolve_scalebar,
    add_colorbars,
    special_ticks,
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
    norm: NormalizationConfig | dict | str | None = None,
    scalebar: ScalebarConfig | dict | bool | str | None = "auto",
    sampling: Sampling | None = None,
    cmap: str | colors.Colormap | None = None,
    role: str = "intensity",
    background: str = "light",
    chroma_boost: float = 1.0,
    cbar: bool = True,
    figax: tuple[Any, Any] | None = None,
    figsize: tuple[int, int] = (8, 8),
    title: str | None = None,
    aspect: int | None = None,
    xticks: Sequence[Any] = (),
    yticks: Sequence[Any] = (),
    xticklabels: Sequence[Any] = (),
    yticklabels: Sequence[Any] = (),
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> tuple[Any, Any]:
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
    cmap : str or Colormap, optional
        Explicit colour map; overrides ``role``. Unscientific maps warn.
    role : str, default="intensity"
        What the data is: a key of :data:`scatterem.vis.colors.ROLES`.
    background : {"light", "dark"}, default="light"
        Figure background, used by Fig. 6 of Crameri et al. (2020).
    chroma_boost : float, default=1.0
        Factor to boost color saturation when displaying complex data.
    cbar : bool, default=True
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
    choice, norm = resolve_display(
        cmap, role, norm, data=amplitude, background=background
    )
    if not is_complex and choice.cls == "cyclic" and norm == choice.display_norm():
        # The role's limits are one period: wrap the values, do not clip them.
        amplitude = choice.wrap(amplitude)
    norm_config = _resolve_normalization(norm)

    norm_obj = CustomNormalization(**normalization_kwargs(norm_config), data=amplitude)

    scaled_amplitude = norm_obj(amplitude)
    rgba = complex_to_rgba(scaled_amplitude, angle, cmap=choice.cmap)

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
        value_unit = sampling.value_unit if sampling is not None else None
        # A complex image shows the amplitude as lightness, not with the map.
        bar_cmap = _amplitude_colormap() if is_complex else choice.cmap
        add_colorbars(
            fig,
            ax,
            norm_obj,
            bar_cmap,
            value_unit=value_unit,
            is_complex=is_complex,
            ticks=special_ticks(amplitude, role),
        )

    # The bar is placed in axes fractions from the column count, so it is
    # calibrated in x for any aspect.
    scalebar_config = _resolve_scalebar(scalebar, sampling)
    if scalebar_config is not None:
        draw_scale_bar(ax, **scalebar_config.draw_kwargs(rgba.shape[1]))

    return fig, ax


def _show_2d_combined(
    list_of_arrays: Sequence[NDArray],
    *,
    norm: NormalizationConfig | dict | str | None = None,
    scalebar: ScalebarConfig | dict | bool | str | None = "auto",
    sampling: Sampling | None = None,
    cmap: str | colors.Colormap | None = None,
    chroma_boost: float = 1.0,
    cbar: bool = False,
    figax: tuple[Any, Any] | None = None,
    figsize: tuple[int, int] = (8, 8),
    title: str | None = None,
) -> tuple[Any, Any]:
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
        Scale bar to draw; ``False``/``None`` draw the "1 a.u." bar.
    sampling : Sampling, optional
        Calibration metadata driving the physical scale bar.
    cmap : str or Colormap, optional
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


def show_2d(
    array,
    *,
    sampling=None,
    scalebar="auto",
    figax=None,
    axsize=(4, 4),
    tight_layout=True,
    combine_images=False,
    clip_percentile: tuple[float, float] | None = None,
    max_display_pixels: int | None = None,
    style: str | None = None,
    **kwargs,
):
    """Display one or more 2D arrays in a grid.

    Accepts a numpy array, a torch tensor, or a (nested) sequence of either.
    ``sampling`` (a :class:`Sampling` or a dict of its fields) drives the
    physical scale bar; if omitted, the object's ``.sampling_meta`` is used if
    present, otherwise the bar falls back to ``a.u.``.

    Parameters
    ----------
    array : numpy.ndarray or torch.Tensor or sequence
        Array, tensor, or (nested) sequence of either to display.
    sampling : Sampling or dict, optional
        Calibration metadata driving the physical scale bar; falls back to
        ``array.sampling_meta`` if present, else uncalibrated. Default is None.
    scalebar : "auto", bool, dict, or ScalebarConfig, optional
        Scale-bar configuration; "auto" draws one, False/None draw "1 a.u.".
        Default is "auto".
    figax : tuple, optional
        (fig, axs) tuple to use for plotting. Default is None (new figure).
    axsize : tuple of int, optional
        Size of each subplot, in inches. Default is (4, 4).
    tight_layout : bool, optional
        Whether to apply ``tight_layout`` to the figure. Default is True.
    combine_images : bool, optional
        Whether to combine a sequence of arrays into one color-encoded
        visualization. Default is False.
    clip_percentile : tuple of float, optional
        ``(low, high)`` percentiles (0-100) used to set each array's display
        vmin/vmax. A centred role keeps limits symmetric about its centre.
        A cyclic role ignores it and warns. Default is None.
    max_display_pixels : int or None, optional
        The most pixels shown along either image axis. Larger images are
        block-averaged by one integer factor (no aliasing); every scale-bar
        sampling follows, and ticks and overlays use display pixels. ``None``
        (the default) shows every pixel.
    style : {"screen", "paper"}, optional
        Draw with the text and line sizes of ``show(style=...)``
        (:func:`~scatterem.vis.style.figure_params`). The axes of ``figax``
        get the tick, axis-label and spine sizes too. Save the figure inside
        ``use_style(style=...)``: the dpi, the PDF font type and the font are
        read when the figure is saved or drawn.
    **kwargs
        Forwarded to :func:`show_2d_array`. Common keys: ``role`` (kind of
        data, e.g. "intensity" (default), "signed", "angle"; sets colour map
        and limits), ``background`` ("light" or "dark"), ``cmap`` (overrides
        the role's map) and ``cbar`` (colour bar; default True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axs : numpy.ndarray of matplotlib.axes.Axes
        The matplotlib axes object(s).
    """
    meta = _resolve_sampling(array, sampling)
    arrays, meta, scalebar = prepare_display(array, meta, scalebar, max_display_pixels)
    axes = None if figax is None else figax[1]
    with figure_style(style, kwargs.get("background", "light"), axes):
        return show_2d_array(
            arrays,
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
    arrays: NDArray | Sequence[NDArray] | Sequence[Sequence[NDArray]],
    *,
    figax: tuple[Any, Any] | None = None,
    axsize: tuple[int, int] = (4, 4),
    tight_layout: bool = True,
    combine_images: bool = False,
    scalebar: Any = "auto",
    sampling: Sampling | None = None,
    scalebar_panel: Any = "first",
    clip_percentile: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
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
            elif isinstance(title[0], str):  # a flat list of titles
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
                cell_kwargs = dict(kwargs)
                cell_kwargs["norm"] = _clipped_norm(
                    array,
                    clip_percentile,
                    kwargs.get("role", "intensity"),
                    kwargs.get("background", "light"),
                )
            with one_cmap_warning(kwargs.get("cmap") if flat_index == 0 else None):
                _show_2d(
                    array,
                    figax=figax_cell,
                    title=t,
                    scalebar=(scalebar if draw_here else _GRID_NO_BAR),
                    sampling=sampling,
                    **cell_kwargs,
                )
            flat_index += 1

    # Hide unused axes in incomplete rows
    for i, row in enumerate(grid):
        for j in range(len(row), ncols):
            axs[i][j].axis("off")

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


__all__ = [
    "show_2d",
    "show_2d_array",
]
