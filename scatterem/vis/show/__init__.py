"""``show(obj)``: a good default figure for any common scatterem output.

The view comes from the type of ``obj`` (registry or the ``__scatterem_figure__``
protocol), and the environment decides where the figure goes (marimo, Jupyter,
a GUI window, a headless file). See the user guide page ``user_guide/show``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scatterem.vis.show import _data as _data  # noqa: F401 -- registers the data views
from scatterem.vis.show._animate import Animated, FrameSource, animate
from scatterem.vis.show._context import FigureContext, Panel
from scatterem.vis.show._env import _checked, announce, default_path, detect, route
from scatterem.vis.show._figure import ShownFigure
from scatterem.vis.show._layout import compose
from scatterem.vis.show._registry import register, resolve

__all__ = [
    "Animated",
    "FigureContext",
    "FrameSource",
    "Panel",
    "ShownFigure",
    "register",
    "save_animated",
    "show",
    "show_animated",
]


def show(
    obj: Any,
    *,
    panels=None,
    save=None,
    style: str = "screen",
    width: str | None = None,
    background: str = "light",
    env: str | None = None,
    block: bool = True,
    **opts: Any,
) -> ShownFigure:
    """Show ``obj`` as a good default figure.

    Parameters
    ----------
    obj : Any
        An array, a dataset, or a scatterem result.
    panels : sequence of str, optional
        Panel names to draw, in order. None draws the default figure.
    save : str or pathlib.Path, optional
        Also write the figure here (.png, .pdf or .svg).
    style : {"screen", "paper"}, default="screen"
        Paper is Nature style (89 or 183 mm, 5-7 pt, panel letters).
    width : {None, "single", "double"}, optional
        Paper width.
    background : {"light", "dark"}, default="light"
        Figure background for the colour policy.
    env : str, optional
        "marimo", "jupyter", "gui", "headless" or "none"; default: detected.
    block : bool, default=True
        Passed to ``plt.show`` in a GUI.
    **opts
        View-specific options, for example ``point_group="m-3m"``.

    Returns
    -------
    ShownFigure
        The figure. ``saved_path`` is set when a file was written.
    """
    ctx = FigureContext(
        style=style,
        width=width,
        background=background,
        panels=tuple(panels) if panels is not None else None,
        opts=opts,
    )
    view = resolve(obj)
    environment = env if env is not None else detect()
    fig = compose(view.panels(obj, ctx), ctx, gui=environment == "gui")
    return route(fig, environment, save=save, name=view.name, block=block)


def show_animated(
    obj: Any,
    *,
    axis: str | None = None,
    fps: float = 10,
    max_frames: int = 300,
    env: str | None = None,
    style: str = "screen",
    background: str = "light",
    **opts: Any,
) -> Animated:
    """Show ``obj`` as an animation (inline in notebooks, a file when headless).

    Parameters
    ----------
    obj : Any
        An object whose view has frames (multislice, volume, tilt series, ...).
    axis : str, optional
        Which axis to step through; the view's default when None. Volumes take
        "z", "y" or "x"; a view raises
        ``ValueError`` for a value it does not support.
    fps : float, default=10
        Frames per second.
    max_frames : int, default=300
        Longer sequences are subsampled evenly, with a warning.
    env : str, optional
        "marimo", "jupyter", "gui", "headless" or "none"; default: detected.
    style, background : str
        As for :func:`show`.
    **opts
        View-specific options.

    Returns
    -------
    Animated
        MP4 (or GIF without the ``anim`` extra) with an inline HTML form.
    """
    if env is not None:
        _checked(env, "env")
    ctx = FigureContext(style=style, background=background, opts=opts)
    anim, name = animate(obj, axis=axis, fps=fps, max_frames=max_frames, ctx=ctx)
    environment = env if env is not None else detect()
    suffix = ".mp4" if anim.mime == "video/mp4" else ".gif"
    if environment == "jupyter":
        anim.display_in_jupyter()
    elif environment in ("headless", "gui"):
        anim.path = default_path(name, suffix)
        anim.path.parent.mkdir(parents=True, exist_ok=True)
        anim.path.write_bytes(anim.data)
        announce(anim.path)
    return anim


def save_animated(
    obj: Any,
    path,
    *,
    axis: str | None = None,
    fps: float = 10,
    max_frames: int = 300,
    style: str = "screen",
    background: str = "light",
    **opts: Any,
) -> Path:
    """Write ``obj``'s animation to ``path`` (.mp4 or .gif).

    Parameters
    ----------
    obj : Any
        An object whose view has frames.
    path : str or pathlib.Path
        Destination; .mp4 needs the ``anim`` extra, else a .gif is written instead.
    axis, fps, max_frames, style, background, **opts
        As for :func:`show_animated`.

    Returns
    -------
    pathlib.Path
        The written file (the .gif when MP4 was not possible).

    Raises
    ------
    ValueError
        If the suffix is not .mp4 or .gif.
    """
    target = Path(path)
    if target.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError(f"save_animated writes .mp4 or .gif, not {target.name!r}")
    ctx = FigureContext(style=style, background=background, opts=opts)
    anim, _ = animate(
        obj,
        axis=axis,
        fps=fps,
        max_frames=max_frames,
        ctx=ctx,
        fmt=target.suffix[1:].lower(),
    )
    if anim.mime == "image/gif":
        target = target.with_suffix(".gif")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(anim.data)
    return target
