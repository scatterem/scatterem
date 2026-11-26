"""Linear interpolation at array coordinates and its exact adjoint.

:func:`sample_image` reads an image or volume at fractional pixel coordinates
(bi/trilinear); :func:`insert_into_image` is its transpose, splatting values
onto the grid with the same weights. Both are built from one list of corner
indices and weights, so ``<sample(x), y> == <x, insert(y)>`` holds to rounding
for every padding mode -- the pair a Fourier-slice projector and backprojector
needs.

Conventions: coordinates are in pixel index units in array order
(``(..., z, y, x)``), pixel ``i`` is at coordinate ``i``. This differs on
purpose from ``torch.nn.functional.grid_sample``, which takes
normalised ``[-1, 1]`` coordinates in ``(x, y, z)`` order.

Written for scatterem from the definition of multilinear interpolation; it
comes from no other implementation.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["insert_into_image", "sample_image"]

_PADDINGS = ("zeros", "periodic", "border")


def _corners(
    coords: Tensor, spatial: Sequence[int], padding: str, weight_dtype: torch.dtype
) -> list[tuple[Tensor, Tensor]]:
    """``(linear_index, weight)`` per corner of the interpolation cell.

    ``coords`` is ``(*Q, D)``; each index and weight is ``(*Q,)``. A corner that
    falls outside the grid under ``"zeros"`` gets weight 0 and a clamped index.
    """
    if padding not in _PADDINGS:
        raise ValueError(f"padding must be one of {_PADDINGS}, got {padding!r}")
    ndim = len(spatial)
    if coords.shape[-1] != ndim:
        raise ValueError(
            f"coords has {coords.shape[-1]} components for {ndim} spatial axes"
        )
    axes = []
    for axis, n in enumerate(spatial):
        c = coords[..., axis]
        if padding == "border":
            c = c.clamp(0, n - 1)
        lo = torch.floor(c)
        frac = (c - lo).to(weight_dtype)
        lo = lo.long()
        taps = []
        for offset, w in ((0, 1 - frac), (1, frac)):
            i = lo + offset
            if padding == "periodic":
                i = i % n
            elif padding == "zeros":
                w = w * ((i >= 0) & (i < n)).to(weight_dtype)
            taps.append((i.clamp(0, n - 1), w))
        axes.append(taps)
    strides = [1] * ndim
    for axis in range(ndim - 2, -1, -1):
        strides[axis] = strides[axis + 1] * int(spatial[axis + 1])
    corners = []
    for combo in itertools.product(*axes):
        index = sum(i * s for (i, _), s in zip(combo, strides, strict=True))
        weight = combo[0][1]
        for _, w in combo[1:]:
            weight = weight * w
        corners.append((index, weight))
    return corners


def _real_dtype(x: Tensor, name: str) -> torch.dtype:
    """Real dtype of the interpolation weights for ``x``; floating or complex only.

    An integer weight dtype would truncate every fractional weight to 0 and
    silently floor-sample, so a non-floating, non-complex ``x`` is refused.
    """
    if x.is_complex():
        return x.real.dtype
    if not x.is_floating_point():
        raise TypeError(
            f"{name} must be a floating or complex tensor, got {x.dtype}; "
            f"convert it first, e.g. {name}.float()"
        )
    return x.dtype


def sample_image(x: Tensor, coords: Tensor, *, padding: str = "zeros") -> Tensor:
    """Bi/trilinear samples of ``x`` at fractional pixel coordinates.

    Coordinates are shared across every leading axis of ``x``: ``x`` is
    ``(*B, *S)``, ``coords`` is ``(*Q, D)`` and the result is ``(*B, *Q)``.
    Per-projection or per-view batches of coordinates therefore go in ``Q``,
    not in ``B``.

    Parameters
    ----------
    x : Tensor
        Real or complex image stack ``(*B, *S)`` with ``D = coords.shape[-1]``
        spatial axes ``S`` (``D`` in 1, 2, 3); the leading ``B`` axes share the
        coordinates.
    coords : Tensor
        Floating ``(*Q, D)`` coordinates in pixels, array order.
    padding : {"zeros", "periodic", "border"}, optional
        Value outside the grid: 0, the periodic continuation, or the nearest
        edge pixel. Default ``"zeros"``. With ``"zeros"`` an out-of-grid
        corner is read from the clamped edge pixel and weighted by 0, so a
        NaN or inf in an edge pixel propagates into out-of-grid samples
        (``0 * nan`` is ``nan``).

    Returns
    -------
    Tensor
        ``(*B, *Q)`` samples, dtype of ``x``. Differentiable with respect to
        ``x`` and ``coords``.

    Raises
    ------
    TypeError
        If ``x`` is neither floating nor complex (an integer image would be
        floor-sampled); convert it with ``x.float()``.
    ValueError
        If ``padding`` is unknown, or ``D = coords.shape[-1]`` is not 1, 2 or
        3 or exceeds ``x.ndim``.
    """
    real = _real_dtype(x, "x")
    ndim = coords.shape[-1]
    if not 1 <= ndim <= min(3, x.ndim):
        raise ValueError(f"coords.shape[-1]={ndim} needs 1 <= D <= min(3, x.ndim)")
    spatial = x.shape[-ndim:]
    batch = x.shape[:-ndim]
    query = coords.shape[:-1]
    flat = x.reshape(*batch, -1)
    (index, weight), *rest = _corners(coords, spatial, padding, real)
    out = flat[..., index.reshape(-1)] * weight.reshape(-1)
    for index, weight in rest:
        out = out + flat[..., index.reshape(-1)] * weight.reshape(-1)
    return out.reshape(*batch, *query)


def insert_into_image(
    values: Tensor,
    coords: Tensor,
    shape: Sequence[int],
    *,
    padding: str = "zeros",
) -> tuple[Tensor, Tensor]:
    """Splat ``values`` onto a grid with bi/trilinear weights; adjoint of sampling.

    For the same ``coords`` and ``padding``, this is the exact transpose of
    :func:`sample_image`: ``<sample_image(x, c), y> == <x, insert_into_image(y,
    c, x.shape[-D:])[0]>``. As in :func:`sample_image`, the coordinates are
    shared across the leading axes of ``values``: ``values`` is ``(*B, *Q)``,
    ``coords`` is ``(*Q, D)`` and the image is ``(*B, *S)``.

    Parameters
    ----------
    values : Tensor
        Real or complex ``(*B, *Q)`` values; ``Q = coords.shape[:-1]``.
    coords : Tensor
        Floating ``(*Q, D)`` coordinates in pixels, array order, ``D`` in 1-3.
    shape : sequence of int
        Spatial grid shape ``S`` (``D`` entries).
    padding : {"zeros", "periodic", "border"}, optional
        As in :func:`sample_image`. Default ``"zeros"``. With ``"zeros"`` an
        out-of-grid corner is added to the clamped edge pixel with weight 0, so
        a NaN or inf value inserted outside the grid poisons that edge pixel.

    Returns
    -------
    image : Tensor
        ``(*B, *S)`` accumulated values, dtype of ``values``. Differentiable
        with respect to ``values``.
    weights : Tensor
        ``(*S)`` accumulated weights (the insertion of ones), real; divide by
        it for a normalised gridding.

    Raises
    ------
    TypeError
        If ``values`` is neither floating nor complex (integer values would
        be inserted at the floored coordinates); convert it with
        ``values.float()``.
    ValueError
        If ``padding`` is unknown or the shapes disagree.
    """
    real = _real_dtype(values, "values")
    ndim = coords.shape[-1]
    if len(shape) != ndim or not 1 <= ndim <= 3:
        raise ValueError(f"shape {tuple(shape)} does not match D={ndim} (1 <= D <= 3)")
    query = coords.shape[:-1]
    if values.shape[values.ndim - len(query) :] != query:
        raise ValueError(
            f"values shape {tuple(values.shape)} does not end in coords' {tuple(query)}"
        )
    batch = values.shape[: values.ndim - len(query)]
    n = 1
    for s in shape:
        n *= int(s)
    flat_values = values.reshape(*batch, -1)
    image = values.new_zeros((*batch, n))
    weights = torch.zeros(n, dtype=real, device=values.device)
    for index, weight in _corners(coords, shape, padding, real):
        idx, w = index.reshape(-1), weight.reshape(-1)
        image = image.index_add(-1, idx, flat_values * w)
        weights = weights.index_add(0, idx, w)
    return image.reshape(*batch, *shape), weights.reshape(*shape)
