"""Random streams keyed by global indices, independent of the decomposition.

A draw that belongs to a global object -- a scan tile, a frozen-phonon
configuration -- is made from a generator seeded by a hash of the run seed and
that object's global keys. Whichever rank owns the object draws the same values,
so results do not depend on how the work was split (on one device type; CPU and
CUDA generators differ).

Keys are signed 64-bit integers, hashed exactly into a 63-bit seed. A key outside
that range raises ``ValueError`` instead of wrapping, so wrapping never makes two
different keys equal, and a non-integer key (a float coordinate, say) raises
``TypeError`` instead of being truncated.

Seed space of the generators: torch's CPU generator uses only the low 32 bits of
its seed, so on the CPU two distinct keys can still draw the same stream
(probability about ``n**2 / 2**33`` among ``n`` keys). CUDA generators (Philox)
use all 64 bits. Draw per tile or per position on CUDA; the CPU is fine for a
handful of keys. See :func:`keyed_generator`.
"""

from __future__ import annotations

import hashlib
import operator
import struct
from typing import SupportsIndex

import torch
from torch import Tensor

__all__ = ["keyed_generator", "keyed_seed", "tile_index"]

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _as_int64(value: SupportsIndex, name: str) -> int:
    """``value`` as a Python int, rejecting non-integers and values beyond int64."""
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__} ({value!r}); "
            "round it explicitly if truncation is intended"
        ) from None
    if not _INT64_MIN <= number <= _INT64_MAX:
        raise ValueError(
            f"{name} = {number} does not fit in int64 "
            f"[{_INT64_MIN}, {_INT64_MAX}]; keys are hashed as signed 64-bit "
            "integers and are never wrapped"
        )
    return number


def _positive_int(value: SupportsIndex, name: str) -> int:
    number = _as_int64(value, name)
    if number < 1:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def keyed_seed(seed: int, *keys: int) -> int:
    """Hash a run seed and global keys into one 63-bit seed.

    The hash is BLAKE2b over the signed 64-bit encoding of ``seed`` and every
    key, in order, so ``(5,)``, ``(5, 0)`` and ``(5, 0, 0)`` are different
    objects and the order of the keys matters.

    Parameters
    ----------
    seed : int
        The run's seed.
    *keys : int
        Global indices (configuration, tile, ...), in a fixed order.

    Returns
    -------
    int
        A seed in ``[0, 2**63)``. A CPU ``torch.Generator`` keeps only its low
        32 bits, see :func:`keyed_generator`.

    Raises
    ------
    TypeError
        If ``seed`` or a key is not an integer: anything with ``__index__``
        (NumPy and torch integers) is accepted, floats are not.
    ValueError
        If ``seed`` or a key does not fit in a signed 64-bit integer. The
        message names the offending argument.
    """
    h = hashlib.blake2b(digest_size=8, person=b"scatterem-rng")
    h.update(struct.pack("<q", _as_int64(seed, "seed")))
    for i, key in enumerate(keys):
        h.update(struct.pack("<q", _as_int64(key, f"keys[{i}]")))
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


def keyed_generator(
    seed: int, *keys: int, device: str | torch.device = "cpu"
) -> torch.Generator:
    """A ``torch.Generator`` seeded by :func:`keyed_seed`.

    Every call returns a fresh generator at the start of its stream. Streams are
    reproducible for one device type only: a CPU and a CUDA generator with the
    same seed draw different values, so every rank of a run must draw on the
    same kind of device.

    Parameters
    ----------
    seed : int
        The run's seed.
    *keys : int
        Global indices.
    device : str or torch.device, optional
        Generator device. Default ``"cpu"``.

    Returns
    -------
    torch.Generator
        The seeded generator.

    Raises
    ------
    TypeError, ValueError
        As :func:`keyed_seed`.

    Notes
    -----
    The seed from :func:`keyed_seed` has 63 bits, but torch's CPU generator uses
    only the low 32 of them. Distinct keys can therefore share a CPU stream, with
    probability about ``n**2 / 2**33`` among ``n`` keys: roughly 1 % at ``10**4``
    keys and roughly 70 % at ``10**5``. For example, ``(9, 3, 90579)`` and
    ``(9, 3, 113685)`` draw identical ``torch.randn`` values on the CPU. CUDA
    generators (Philox) use the whole 64-bit seed and do not collide this way.

    Use ``device="cuda"`` when many keys are drawn (per-tile or per-position
    noise). CPU generators are fine for a handful of keys, such as one per
    frozen-phonon configuration.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(keyed_seed(seed, *keys))
    return gen


def tile_index(
    rows: Tensor, cols: Tensor, tile: tuple[int, int], n_cols: int
) -> Tensor:
    """Id of the fixed global tile containing each scan position.

    The scan is cut into ``tile``-shaped blocks anchored at the origin. The last
    tile row and column are partial when the scan is not a multiple of the tile.
    The coordinates must lie inside the scan (``0 <= col < n_cols``): a column
    past ``n_cols`` would take the id of a tile in the next tile row. This is not
    checked, to avoid a device synchronisation per call.

    Parameters
    ----------
    rows, cols : Tensor
        Integer scan coordinates, broadcast against each other.
    tile : tuple of int
        Tile shape ``(rows, cols)``, both positive.
    n_cols : int
        Columns of the whole scan, positive.

    Returns
    -------
    Tensor
        Tile ids (int64), row-major over the tile grid, on the device of the
        inputs.

    Raises
    ------
    TypeError
        If ``rows`` or ``cols`` is a floating-point tensor, or ``tile`` or
        ``n_cols`` holds a non-integer.
    ValueError
        If ``tile`` is not a pair or an entry of ``tile`` or ``n_cols`` is not
        positive.
    """
    if len(tile) != 2:
        raise ValueError(f"tile must be a pair (rows, cols), got {tuple(tile)!r}")
    tile_h = _positive_int(tile[0], "tile[0]")
    tile_w = _positive_int(tile[1], "tile[1]")
    scan_cols = _positive_int(n_cols, "n_cols")
    for name, coord in (("rows", rows), ("cols", cols)):
        if coord.is_floating_point() or coord.is_complex():
            raise TypeError(
                f"{name} must be an integer tensor, got {coord.dtype}; "
                "floor or round scan coordinates first"
            )
    tile_cols = -(-scan_cols // tile_w)
    return (rows.to(torch.int64) // tile_h) * tile_cols + cols.to(torch.int64) // tile_w
