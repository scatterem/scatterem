"""Registered views for layer-0..2 types: arrays, tensors, datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from scatterem.data.sampling import Sampling
from scatterem.vis._display_input import _to_numpy
from scatterem.vis.show import _panels as P
from scatterem.vis.show._animate import FrameSource, check_axis
from scatterem.vis.show._context import FigureContext, Panel
from scatterem.vis.show._registry import register
from scatterem.vis.volume_frames import central_sections

# Registered by qualified name: importing the dataset stack here would add ~1.5 s
# to ``import scatterem.vis``.
DATASET_4DSTEM = "scatterem.data.datasets.dataset_4dstem.Dataset4DStem"


#: The axes of a ``(Z, Y, X)`` volume that ``show_animated(axis=...)`` steps through.
VOLUME_AXES = ("z", "y", "x")


def _array_frames(obj: Any, axis: str | None, ctx: FigureContext) -> FrameSource:
    name = check_axis(axis, VOLUME_AXES)
    a = _to_numpy(obj)
    if a.ndim != 3:
        raise TypeError("only 3-D (Z, Y, X) arrays animate")
    if np.iscomplexobj(a):
        a = np.abs(a)  # the sections and their shared norm show the magnitude too
    a = np.moveaxis(a, VOLUME_AXES.index(name), 0)
    return FrameSource(
        list(a),
        axis=name,
        sampling=ctx.opts.get("sampling"),
        labels=[f"{name} {k}/{len(a)}" for k in range(len(a))],
    )


def _array_panels(obj: Any, ctx: FigureContext) -> list[Panel]:
    a = _to_numpy(obj)
    sampling = ctx.opts.get("sampling")
    if a.ndim == 2:
        name = "complex" if np.iscomplexobj(a) else "image"
        return [Panel(name, lambda ax, c: P.image(ax, a, ctx=c, sampling=sampling), "")]
    if a.ndim == 3:
        sections = central_sections(
            torch.as_tensor(np.abs(a) if np.iscomplexobj(a) else a)
        )
        norm = P.shared_norm(a)
        # A binned section has larger pixels: scale the calibration with it.
        return [
            Panel(
                k,
                lambda ax, c, s=s: P.image(
                    ax,
                    s[0],
                    ctx=c,
                    norm=norm,
                    sampling=P.binned_sampling(sampling, s[1]),
                ),
                k,
            )
            for k, s in sections.items()
        ]
    hint = (
        "; for 4D-STEM data use a Dataset4DStem"
        if a.ndim == 4
        else f", not a {a.ndim}-D array"
    )
    raise TypeError(f"show() draws 2-D and 3-D arrays{hint}")


register(np.ndarray, name="array", frames=_array_frames)(_array_panels)
register(torch.Tensor, name="array", frames=_array_frames)(_array_panels)


def _real_sampling(ds: Any) -> Sampling:
    s, u = list(ds.sampling), list(ds.units)
    return Sampling(pixel_size=(float(s[0]), float(s[1])), units=(u[0], u[1]))


def _reciprocal_sampling(ds: Any) -> Sampling:
    s, u = list(ds.sampling), list(ds.units)
    return Sampling(pixel_size=(float(s[2]), float(s[3])), units=(u[2], u[3]))


#: Largest block (in elements) that the reductions below hold as float64.
_BLOCK_ELEMENTS = 250_000


def _row_blocks(cube: Any):
    """Slices of whole scan rows, each at most ``_BLOCK_ELEMENTS`` elements."""
    per_row = max(1, int(np.prod(cube.shape[1:])))
    step = max(1, _BLOCK_ELEMENTS // per_row)
    for start in range(0, cube.shape[0], step):
        yield slice(start, start + step)


def _float_block(cube: Any, rows: slice) -> np.ndarray:
    """Scan rows ``rows`` of a numpy or torch cube, as float64 on the host.

    Only this block leaves the device, so a CUDA cube is never copied whole.
    The cast to float64 happens on the host (MPS has no float64).
    """
    return _to_numpy(cube[rows]).astype(float, copy=False)


def _mean_pattern(cube: Any) -> np.ndarray:
    """Mean diffraction pattern, accumulated in float64 over blocks of scan rows."""
    total = np.zeros(tuple(cube.shape[2:]))
    for rows in _row_blocks(cube):
        total += _float_block(cube, rows).sum(axis=(0, 1))
    return total / max(1, cube.shape[0] * cube.shape[1])


def _com(cube: Any) -> tuple[np.ndarray, np.ndarray]:
    """CoM magnitude and direction, by reductions over blocks of scan rows."""
    ky = np.arange(cube.shape[2], dtype=float)
    kx = np.arange(cube.shape[3], dtype=float)
    cy = np.empty(tuple(cube.shape[:2]))
    cx = np.empty(tuple(cube.shape[:2]))
    for rows in _row_blocks(cube):
        block = _float_block(cube, rows)
        total = block.sum(axis=(2, 3)) + 1e-12
        cy[rows] = np.einsum("ijkl,k->ij", block, ky) / total
        cx[rows] = np.einsum("ijkl,l->ij", block, kx) / total
    cy, cx = cy - cy.mean(), cx - cx.mean()
    return np.hypot(cy, cx), np.arctan2(cy, cx)


def _lazy(compute):
    """Call ``compute`` at the first use only, so unselected panels cost nothing."""
    cache: list = []

    def get():
        if not cache:
            cache.append(compute())
        return cache[0]

    return get


def _dataset_panels(ds: Any, ctx: FigureContext) -> list[Panel]:
    real, recip = _real_sampling(ds), _reciprocal_sampling(ds)
    # The cube stays where it is (numpy or a device tensor); the reductions
    # read it in blocks of scan rows, and only when their panel is drawn.
    cube = ds.array
    bf = _lazy(lambda: _to_numpy(ds.virtual_bright_field()))
    adf = _lazy(lambda: _to_numpy(ds.virtual_annular(1.5, 4.0, units="bf_radius")))
    mean = _lazy(lambda: np.log1p(np.clip(_mean_pattern(cube), 0, None)))
    com = _lazy(lambda: _com(cube))
    return [
        Panel("bf", lambda ax, c: P.image(ax, bf(), ctx=c, sampling=real), "virtual BF"),
        Panel("adf", lambda ax, c: P.image(ax, adf(), ctx=c, sampling=real), "virtual ADF"),
        Panel("diffraction", lambda ax, c: P.image(ax, mean(), ctx=c, role="diffraction", sampling=recip), "mean diffraction (log)"),
        Panel("com", lambda ax, c: P.image(ax, com()[0], ctx=c, sampling=real), "CoM magnitude"),
        Panel("com_angle", lambda ax, c: P.image(ax, com()[1], ctx=c, role="angle", sampling=real), "CoM direction"),
    ]  # fmt: skip


def _dataset_frames(ds: Any, axis: str | None, ctx: FigureContext) -> FrameSource:
    name = check_axis(axis, ("scan x",))
    row = ds.array.shape[0] // 2
    patterns = _to_numpy(ds.array[row])  # one scan row leaves the device
    frames = [np.log1p(np.clip(p.astype(float), 0, None)) for p in patterns]
    return FrameSource(frames, axis=name, role="diffraction", sampling=_reciprocal_sampling(ds),
                       labels=[f"scan ({row}, {k})" for k in range(len(frames))])  # fmt: skip


register(DATASET_4DSTEM, name="dataset4dstem", frames=_dataset_frames)(_dataset_panels)








