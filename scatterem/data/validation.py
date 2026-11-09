"""Input coercion for the dataset containers.

Independent helpers that turn caller input into the representation each
container works with, and refuse clearly when that is impossible. They replace
the former ``utils.validators`` copies of the same three names.

Behaviour that callers depend on is kept -- in particular a scalar
``sampling`` still broadcasts across all axes, which is what makes
``Dataset4DStem(sampling=0.2)`` mean the same pitch on every axis, and feeds
``dk``. Two deliberate differences from the replaced helpers:

* padding a low-dimensional array up to ``ndim`` **raises** instead of warning
  and silently prepending axes. Silently reshaping measured data hides a real
  mismatch between what the caller has and what they think they have.
* error messages name the offending value, not only its type.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

# Shared with the staged H2D path used by public dataset loaders; importing it
# here means a later switch of container imports cannot drop the pipeline.
from scatterem.utils._device import resolve_device
from scatterem.utils.validators import to_device_staged


def h5py_dataset_to_tensor(array: Any) -> torch.Tensor | None:
    """Read an open ``h5py.Dataset`` into a CPU tensor, or return ``None``.

    ``torch.as_tensor`` does not use ``__array__``. It walks an ``h5py.Dataset`` as
    a nested Python sequence, about 110 ns per element: a 64 MiB float32 cube took
    3.1 s, against 0.13 s through one ``np.asarray`` read.
    ``Dataset.__array__`` reads into a fresh, owning, writeable array, so the
    result aliases nothing the caller holds, the same as the ``as_tensor`` result.

    Returns ``None`` for anything that is not an ``h5py.Dataset`` (h5py is looked
    up in ``sys.modules``: if it was never imported, the input cannot be one), and
    for the datasets where the two conversions do not agree, so that the caller's
    ``torch.as_tensor`` path keeps its behaviour for those: a scalar or empty
    dataset, ``uint64`` in either byte order (``as_tensor`` refuses it), and any
    dtype that is not boolean, integer, float or complex.

    A dataset that cannot be read (a closed file, a missing compression filter)
    raises the h5py error here; :func:`ensure_valid_array` turns it into its
    ``TypeError``.

    Args:
        array: the caller's input.

    Returns:
        The dataset's values as a CPU tensor, or ``None`` (not converted).
    """
    h5py = sys.modules.get("h5py")
    if (
        h5py is None
        or not isinstance(array, h5py.Dataset)
        or array.ndim == 0
        or array.size == 0
        or array.dtype.kind not in "biufc"
        # dtype equality includes the byte order, so test kind and size: a
        # big-endian uint64 must be declined as well.
        or (array.dtype.kind == "u" and array.dtype.itemsize == 8)
    ):
        return None
    arr = np.asarray(array)
    # torch.from_numpy refuses a non-native byte order; torch.as_tensor accepts it.
    if not arr.dtype.isnative:
        arr = arr.astype(arr.dtype.newbyteorder("="))
    try:
        return torch.from_numpy(arr)
    except TypeError:
        return None


def ensure_valid_array(
    array: Any,
    dtype: torch.dtype | None = None,
    ndim: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``array`` as a detached tensor on ``device``.

    Args:
        array: numpy array, torch tensor, or anything ``torch.as_tensor`` accepts.
        dtype: cast to this dtype if given.
        ndim: require exactly this many dimensions.
        device: destination. ``None`` keeps a torch tensor's device; numpy /
            lists / scalars become CPU tensors. An explicit value moves the
            result (via staged H2D when applicable).

    Raises:
        TypeError: the input cannot become a tensor.
        ValueError: ``ndim`` is given and does not match.
    """
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        # Zero-copy, deliberately: a 4D-STEM cube runs to several GB (the Gd2O3
        # dataset is 6.6), and copying on construction would double peak memory
        # for no benefit.
        #
        # The consequence is that the container's in-place work -- normalisation,
        # negative clipping -- is visible in the caller's array. That is
        # pre-existing behaviour of these containers, kept rather than changed
        # here; ``test_shares_memory_with_the_caller_by_design`` pins it so the
        # aliasing is a documented property rather than a surprise.
        tensor = torch.from_numpy(array)
    else:
        try:
            tensor = h5py_dataset_to_tensor(array)
            if tensor is None:
                tensor = torch.as_tensor(array)
        except Exception as exc:
            raise TypeError(
                f"could not convert {type(array).__name__} to a tensor: {exc}"
            ) from exc

    tensor = to_device_staged(tensor, torch.device(resolve_device(device, tensor)))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    tensor.requires_grad_(False)

    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(
            f"expected a {ndim}-dimensional array; got shape {tuple(tensor.shape)} "
            f"({tensor.ndim} dimensions). Reshape it explicitly rather than "
            f"relying on axes being inserted for you."
        )
    return tensor


def ensure_valid_array_owned(
    array: Any, device: torch.device | str | None = None
) -> tuple[torch.Tensor, bool]:
    """:func:`ensure_valid_array`, and whether no caller holds the result's buffer.

    An open ``h5py.Dataset`` is read into a fresh buffer
    (:func:`h5py_dataset_to_tensor`), so a container's protective clone of it
    protects nothing. Skipping that clone saves one full pass over the cube:
    1.41x / 1.25x on the medium / large ``from_array`` read.
    Any other input reports ``False``: a tensor or ndarray may be the
    caller's.

    Args:
        array: the caller's input, as for :func:`ensure_valid_array`.
        device: as for :func:`ensure_valid_array`.

    Returns:
        ``(tensor, owned)``: the tensor, and ``True`` only for an h5py read.

    Raises:
        TypeError: the input cannot become a tensor, as for
            :func:`ensure_valid_array`.
    """
    try:
        fresh = h5py_dataset_to_tensor(array)
    except Exception as exc:  # one read: the same TypeError as ensure_valid_array
        raise TypeError(
            f"could not convert {type(array).__name__} to a tensor: {exc}"
        ) from exc
    if fresh is None:
        return ensure_valid_array(array, device=device), False
    return ensure_valid_array(fresh, device=device), True


def validate_ndinfo(
    value: NDArray | Sequence[float] | float | int,
    ndim: int,
    name: str,
    dtype: Any = None,
) -> NDArray:
    """Return a per-axis numeric array of length ``ndim``.

    A scalar broadcasts to every axis, so ``sampling=0.2`` means 0.2 on all of
    them. A sequence must already have one entry per axis.

    Args:
        value: scalar, or one value per axis.
        ndim: number of axes.
        name: what is being validated, for the error message.
        dtype: dtype of the result.
    """
    if np.isscalar(value):
        out = np.full(ndim, value, dtype=dtype)
    elif isinstance(value, (np.ndarray, tuple, list)):
        try:
            out = np.asarray(value, dtype=dtype).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"could not read {name} as numbers: {exc}") from exc
        if out.size != ndim:
            raise ValueError(
                f"{name} has {out.size} entries but the data has {ndim} axes; "
                f"give one value per axis, or a single value for all of them"
            )
    else:
        raise TypeError(
            f"{name} must be a scalar, list, tuple or array; got {type(value).__name__}"
        )

    if not np.issubdtype(out.dtype, np.number):
        raise ValueError(f"{name} must be numeric; got dtype {out.dtype}")
    return out


def coerce_ndinfo(
    value: NDArray | Sequence[float] | float | int,
    ndim: int,
    dtype: Any = None,
) -> NDArray:
    """Convert origin/sampling to a 1-D array; scalars broadcast to ``ndim``.

    Unlike :func:`validate_ndinfo`, sequence length need not equal ``ndim``
    (virtual bright-field keeps the parent's 4-vector on a 3-D array).
    """
    if np.isscalar(value):
        out = np.full(ndim, value, dtype=dtype)
    elif isinstance(value, (np.ndarray, tuple, list)):
        try:
            out = np.asarray(value, dtype=dtype).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"could not read value as numbers: {exc}") from exc
    else:
        raise TypeError(
            f"must be a scalar, list, tuple or array; got {type(value).__name__}"
        )
    if not np.issubdtype(out.dtype, np.number):
        raise ValueError(f"must be numeric; got dtype {out.dtype}")
    return out


def validate_units(value: str | Sequence[str], ndim: int) -> list[str]:
    """Return one unit string per axis.

    A single string applies to every axis.
    """
    if isinstance(value, str):
        return [value] * ndim
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"units must be a string or a sequence of strings; got "
            f"{type(value).__name__}"
        )
    if len(value) != ndim:
        raise ValueError(f"units has {len(value)} entries but the data has {ndim} axes")
    return [str(unit) for unit in value]
