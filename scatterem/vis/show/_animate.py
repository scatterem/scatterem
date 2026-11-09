"""Animated views: frames -> MP4 / GIF / inline HTML5 video."""

from __future__ import annotations

import base64
import importlib.util
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from scatterem.vis._display_input import _to_numpy
from scatterem.vis.colors import colormap_for_role
from scatterem.vis.show._context import FigureContext
from scatterem.vis.show._layout import style_rc
from scatterem.vis.show._panels import image, shared_norm

INSTALL_HINT = (
    "MP4 needs imageio-ffmpeg (pip install imageio-ffmpeg); writing GIF instead"
)


@dataclass(frozen=True)
class FrameSource:
    """The frames of an animated view.

    Parameters
    ----------
    frames : sequence
        2D arrays (values), or (H, W, 3) RGB arrays in [0, 1] when ``role`` is None.
    axis : str
        What the frames step through ("slice", "tilt", "energy", ...).
    role : str or None
        Colour role for value frames; None for RGB frames.
    sampling : Sampling, optional
        Calibration for the scale bar.
    labels : sequence of str, optional
        One label per frame, shown as the title.
    norm : dict, optional
        Shared NormalizationConfig; default: 2-98 % over all frames.
    """

    frames: Sequence[Any]
    axis: str
    role: str | None = "intensity"
    sampling: Any = None
    labels: Sequence[str] | None = None
    norm: dict[str, Any] | None = None


@dataclass
class Animated:
    """An encoded animation that notebooks show inline (once in Jupyter).

    Parameters
    ----------
    data : bytes
        The encoded file.
    mime : str
        "video/mp4" or "image/gif".
    path : pathlib.Path, optional
        Where the file was written, if it was.
    shown : bool, default=False
        True once Jupyter has displayed it, so it appears only once.
    """

    data: bytes
    mime: str
    path: Path | None = None
    shown: bool = field(default=False)

    def _repr_html_(self) -> str:
        b64 = base64.b64encode(self.data).decode("ascii")
        if self.mime == "video/mp4":
            return f'<video autoplay loop muted playsinline controls src="data:video/mp4;base64,{b64}"></video>'
        return f'<img src="data:image/gif;base64,{b64}"/>'

    def display_in_jupyter(self) -> None:
        """Display inline in Jupyter and mark shown."""
        from IPython.display import HTML, display

        display(HTML(self._repr_html_()))
        self.shown = True

    def _ipython_display_(self) -> None:
        if not self.shown:
            self.display_in_jupyter()

    def _display_(self) -> Any:
        import marimo

        return marimo.Html(self._repr_html_())


def _has_ffmpeg() -> bool:
    return importlib.util.find_spec("imageio_ffmpeg") is not None


def _subsample(frames: Sequence[Any], labels, max_frames: int):
    n = len(frames)
    if n <= max_frames:
        return list(frames), list(labels) if labels else None
    warnings.warn(
        f"{n} frames subsampled evenly to max_frames={max_frames}",
        UserWarning,
        stacklevel=5,
    )
    picks = np.unique(np.linspace(0, n - 1, max_frames).round().astype(int))
    return [frames[i] for i in picks], [labels[i] for i in picks] if labels else None


def _even(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    return np.pad(rgb, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")


def render_frames(
    source: FrameSource, ctx: FigureContext, max_frames: int
) -> list[np.ndarray]:
    """Render every frame with colour bar, scale bar and label as RGB uint8."""
    frames, labels = _subsample(source.frames, source.labels, max_frames)
    with rc_context(style_rc(ctx)):
        fig = Figure(figsize=(5.0, 4.2))
        FigureCanvasAgg(fig)
        return _draw_all(fig, source, ctx, frames, labels)


def _draw_all(
    fig, source: FrameSource, ctx: FigureContext, frames, labels
) -> list[np.ndarray]:
    ax = fig.add_subplot()
    if source.role is None:
        artist = ax.imshow(np.asarray(frames[0]))
        ax.set_axis_off()
        to_rgba = lambda f: np.asarray(f)  # noqa: E731
    else:
        norm = source.norm or shared_norm(np.stack([_to_numpy(f) for f in frames]))
        image(
            ax,
            frames[0],
            ctx=ctx,
            role=source.role,
            sampling=source.sampling,
            norm=norm,
        )
        artist = ax.images[0]
        cmap = colormap_for_role(source.role, background=ctx.background).cmap
        lo, hi = norm["vmin"], norm["vmax"]
        to_rgba = lambda f: cmap(np.clip((_to_numpy(f) - lo) / (hi - lo), 0, 1))  # noqa: E731
    out = []
    for k, frame in enumerate(frames):
        artist.set_data(to_rgba(frame))
        if labels:
            ax.set_title(labels[k])
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        out.append(_even(rgb))
    return out


def encode(frames: Sequence[np.ndarray], fmt: str, fps: float) -> bytes:
    """Encode RGB uint8 frames as "mp4" (H.264) or "gif" bytes."""
    import imageio.v3 as iio

    if fmt == "gif":
        return iio.imwrite(
            "<bytes>", np.stack(frames), extension=".gif", duration=1000 / fps, loop=0
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.mp4"
        iio.imwrite(
            path,
            np.stack(frames),
            plugin="FFMPEG",
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        )
        return path.read_bytes()


def check_axis(axis: str | None, allowed: Sequence[str]) -> str:
    """The axis a view steps through: ``axis``, or its default when None.

    Parameters
    ----------
    axis : str or None
        The ``axis=`` that the caller gave.
    allowed : sequence of str
        The values the view supports; the first one is its default.

    Returns
    -------
    str
        The checked axis name.

    Raises
    ------
    ValueError
        If the view does not support ``axis``.
    """
    if axis is None:
        return allowed[0]
    if axis not in allowed:
        raise ValueError(
            f"axis={axis!r} is not supported for this view; use one of {list(allowed)}"
        )
    return axis


def _source(obj: Any, axis: str | None, ctx: FigureContext):
    from scatterem.vis.show._registry import resolve

    view = resolve(obj)
    if view.frames is None:
        raise TypeError(
            f"{type(obj).__name__} has a figure view but no frames to animate"
        )
    return view, view.frames(obj, axis, ctx)


def animate(
    obj: Any, *, axis, fps, max_frames, ctx: FigureContext, fmt: str | None = None
):
    """Render and encode ``obj``'s frames; return (Animated, view name)."""
    view, source = _source(obj, axis, ctx)
    frames = render_frames(source, ctx, max_frames)
    fmt = fmt or ("mp4" if _has_ffmpeg() else "gif")
    if fmt == "mp4" and not _has_ffmpeg():
        warnings.warn(INSTALL_HINT, UserWarning, stacklevel=3)
        fmt = "gif"
    mime = "video/mp4" if fmt == "mp4" else "image/gif"
    return Animated(encode(frames, fmt, fps), mime), view.name
