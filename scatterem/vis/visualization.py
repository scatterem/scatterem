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
    _drop_unit_axes,
    _normalize_show_input_to_grid,
    _panel_axes,
    _panel_title,
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
    """Draw one panel: a real image through a colour map, a complex one as hue.

    A complex input is split into its modulus, which sets the lightness, and its
    angle, which sets the hue. A real input goes through the colour map that
    ``cmap`` or ``role`` selects. The modulus (or the real values) is mapped to
    ``[0, 1]`` by :class:`CustomNormalization` before it is coloured.

    Parameters
    ----------
    array : ndarray
        One 2D panel, real or complex.
    norm : NormalizationConfig or dict or str, optional
        Interval and stretch of the value mapping. A string names an entry of
        :data:`~scatterem.vis.normalization.NORMALIZATION_PRESETS`. None takes
        the limits of ``role``.
    scalebar : ScalebarConfig or dict or bool or str, optional
        What :func:`~scatterem.vis.visualization_utils._resolve_scalebar`
        accepts. ``"auto"`` draws a calibrated bar when ``sampling`` allows it.
    sampling : Sampling, optional
        Pixel size and units; they set the bar length and the colour-bar unit.
    cmap : str or Colormap, optional
        Explicit colour map; overrides ``role``. Unscientific maps warn.
    role : str, default="intensity"
        What the data is: a key of :data:`scatterem.vis.colors.ROLES`.
    background : {"light", "dark"}, default="light"
        Figure background, used by Fig. 6 of Crameri et al. (2020).
    chroma_boost : float, default=1.0
        Accepted for call compatibility. The hue colouring of complex data does
        not read it.
    cbar : bool, default=True
        Attach a value bar on the right (and a phase bar for complex data).
    figax : tuple, optional
        ``(fig, ax)`` to draw into. Without it, one new figure of ``figsize``.
    figsize : tuple, default=(8, 8)
        Width and height in inches of the new figure; ignored with ``figax``.
    title : str, optional
        Panel title.
    aspect : int, optional
        Passed to :meth:`matplotlib.axes.Axes.imshow`.
    xticks, yticks, xticklabels, yticklabels : sequence, optional
        Tick positions and labels. The defaults are empty: no ticks.
    xlabel, ylabel : str, optional
        Axis labels.

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

    fig, ax = plt.subplots(figsize=figsize) if figax is None else figax
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
        Must stay False: an overlay has no single value axis for a bar.
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
        For ``cbar=True``. The image is drawn first, so the axes hold it.
    """
    norm_config = _resolve_normalization(norm)
    scalebar_config = _resolve_scalebar(scalebar, sampling)
    norm_obj = CustomNormalization(**normalization_kwargs(norm_config))
    rgba = overlay_to_rgba(list(list_of_arrays), norm=norm_obj)

    fig, ax = plt.subplots(figsize=figsize) if figax is None else figax
    ax.imshow(rgba)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    if cbar:
        raise NotImplementedError("an overlay of several arrays has no colour bar")

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
    """Show 2D data as one panel or as a grid of panels.

    The input can be a numpy array, a torch tensor, or a list (or a list of
    lists) of them. A tensor goes to the CPU and becomes a numpy array. A 3D
    array is a stack: each slice gets a panel. The scale bar uses ``sampling``;
    without it, the ``.sampling_meta`` of the input if it has one; else the
    bar reads ``1 a.u.``.

    Parameters
    ----------
    array : numpy.ndarray or torch.Tensor or sequence
        The data. A list gives one row of panels; a list of lists gives rows.
    sampling : Sampling or dict, optional
        Pixel size and units (a :class:`Sampling` or a dict of its fields).
        Default is None: use ``array.sampling_meta`` if it exists.
    scalebar : "auto", bool, dict, or ScalebarConfig, optional
        ``"auto"`` draws a calibrated bar when the sampling allows it.
        False or None still draw a bar, the "1 a.u." bar. Default is "auto".
    figax : tuple, optional
        ``(fig, axes)`` to draw into; ``axes`` must match the panel grid.
        Default is None, which makes a new figure.
    axsize : tuple of int, optional
        Width and height in inches of one panel of a new figure.
        Default is (4, 4).
    tight_layout : bool, optional
        Call ``fig.tight_layout()`` at the end. Default is True.
    combine_images : bool, optional
        Blend one row of arrays into a single image, one hue per array.
        Default is False.
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
        The figure that holds the panels.
    axs : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
        One Axes for one panel. A 1D array for a single row or column of
        panels, and a 2D array for a grid.
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
    """Draw panels into a grid of axes and return the figure and the axes.

    :func:`show_2d` converts and bounds its input, then calls this function.
    Each panel is drawn by ``_show_2d``. With ``combine_images``, the one row
    of panels is blended into one image by ``_show_2d_combined`` instead.

    Parameters
    ----------
    arrays : ndarray or sequence of ndarray or sequence of sequences of ndarray
        One 2D array, a 3D stack (one panel per slice), a list (one row), or
        a list of lists (one row per inner list; rows can be short).
    figax : tuple, optional
        ``(fig, axes)`` to draw into. ``axes`` is one Axes or a numpy array of
        them, with the shape of the panel grid (a 1D array counts as one row).
        Default is None, which makes a new figure.
    axsize : tuple, default=(4, 4)
        Width and height in inches of one panel of a new figure.
    tight_layout : bool, default=True
        Call ``fig.tight_layout()`` after drawing.
    combine_images : bool, default=False
        Blend a single row of arrays into one image, one hue per array. Only
        ``norm``, ``cmap``, ``chroma_boost``, ``cbar`` and ``figsize`` are
        accepted in ``kwargs`` then.
    scalebar : "auto", bool, dict, or ScalebarConfig, default="auto"
        The scale bar of the panels that ``scalebar_panel`` selects.
    sampling : Sampling, optional
        Pixel size and units for the scale bar and the colour-bar unit.
    scalebar_panel : "first", "all", or int, default="first"
        Panels with a bar: the first, all, or the one at this flat index
        (row by row).
    clip_percentile : tuple of (float, float), optional
        ``(low, high)`` percentiles (0-100). Each panel then gets the limits
        of these percentiles of its own finite values (the modulus, for
        complex data), as a manual interval. It replaces ``norm``.
    **kwargs : dict
        Passed to every panel, for example ``norm``, ``cmap``, ``role`` or
        ``cbar``. ``title`` is special: one string for all panels, a flat
        list (row by row), or a list of lists.

    Returns
    -------
    fig : Figure
        The figure that holds the panels.
    axs : Axes or ndarray of Axes
        One Axes for a 1 x 1 grid, a 1D array for one row or one column, and
        a 2D array otherwise.

    Raises
    ------
    ValueError
        If the axes of ``figax`` do not have the grid shape, or if
        ``combine_images`` gets several rows or a per-panel keyword.
    NotImplementedError
        If ``combine_images`` is used with ``cbar=True``.
    """
    grid = _normalize_show_input_to_grid(arrays)
    nrows = len(grid)
    ncols = max(map(len, grid))

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

    fig, axs = _panel_axes(figax, nrows, ncols, axsize)

    # Panels are numbered row by row; a short row leaves its cells empty.
    cells = [(r, c) for r in range(nrows) for c in range(ncols)]
    drawn = 0
    for r, c in cells:
        if c >= len(grid[r]):
            axs[r, c].axis("off")
            continue
        panel = grid[r][c]
        if scalebar_panel == "all":
            bar_here = True
        elif scalebar_panel == "first":
            bar_here = drawn == 0
        else:
            bar_here = isinstance(scalebar_panel, int) and drawn == scalebar_panel
        panel_kwargs = kwargs
        if clip_percentile is not None:
            panel_kwargs = {
                **kwargs,
                "norm": _clipped_norm(
                    panel,
                    clip_percentile,
                    kwargs.get("role", "intensity"),
                    kwargs.get("background", "light"),
                ),
            }
        with one_cmap_warning(None if drawn else kwargs.get("cmap")):
            _show_2d(
                panel,
                figax=(fig, axs[r, c]),
                title=_panel_title(title, r, c, ncols),
                scalebar=scalebar if bar_here else _GRID_NO_BAR,
                sampling=sampling,
                **panel_kwargs,
            )
        drawn += 1

    if tight_layout:
        fig.tight_layout()
    return fig, _drop_unit_axes(axs)


__all__ = [
    "show_2d",
    "show_2d_array",
]
