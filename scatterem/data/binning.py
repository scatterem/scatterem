"""Binning: block means over the trailing axes of a tensor."""

import math
from collections.abc import Sequence

from torch import Tensor








def binning_factors(shape: Sequence[int], max_size: int) -> tuple[int, ...]:
    """Smallest integer factor per axis that brings every axis to ``max_size``.

    Parameters
    ----------
    shape : sequence of int
        Array shape.
    max_size : int
        Largest allowed axis length after binning.

    Returns
    -------
    tuple of int
        ``ceil(n / max_size)`` per axis; 1 where the axis already fits.

    Raises
    ------
    ValueError
        If ``max_size`` is smaller than 1.
    """
    if max_size < 1:
        raise ValueError(f"max_size must be >= 1, got {max_size}")
    return tuple(max(1, math.ceil(n / max_size)) for n in shape)


def block_mean(array: Tensor, factors: Sequence[int]) -> Tensor:
    """Average non-overlapping blocks over the trailing axes (area binning).

    The ``n % f`` trailing elements of an axis that do not fill a whole block
    are dropped, so every output element is the mean of exactly
    ``prod(factors)`` inputs: nothing is padded and nothing is decimated.

    Parameters
    ----------
    array : torch.Tensor
        Real tensor; its last ``len(factors)`` axes are binned.
    factors : sequence of int
        Block length per binned axis; 1 leaves an axis unchanged.

    Returns
    -------
    torch.Tensor
        The binned tensor; binned axis ``a`` has length ``n_a // f_a``.

    Raises
    ------
    ValueError
        If a factor is smaller than 1 or larger than its axis.
    """
    k = len(factors)
    lead = tuple(array.shape[: array.ndim - k])
    dims = tuple(array.shape[array.ndim - k :])
    if len(dims) != k or any(
        f < 1 or f > n for f, n in zip(factors, dims, strict=True)
    ):
        raise ValueError(f"factors {tuple(factors)} invalid for axes {dims}")
    if all(f == 1 for f in factors):
        return array
    keep = tuple(slice(0, n // f * f) for n, f in zip(dims, factors, strict=True))
    shape: list[int] = list(lead)
    for n, f in zip(dims, factors, strict=True):
        shape += [n // f, f]
    reduce = tuple(len(lead) + 2 * i + 1 for i in range(k))
    return array[(..., *keep)].reshape(shape).mean(dim=reduce)




