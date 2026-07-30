"""Input coercion for the dataset containers.

Replaces the three helpers ``utils/data/datasets.py`` uses from
``utils/validators.py``, a module derived from quantem (MIT) with the copyright
notice stripped. These are written from the behaviour they need to provide:
coerce whatever a caller passes into the one representation the container works
with, and refuse clearly when that is impossible.

Behaviour is preserved where anything downstream depends on it -- in particular a
scalar ``sampling`` still broadcasts across all axes, which is what makes
``Dataset4dstem(sampling=0.2)`` mean the same pitch on every axis, and feeds
``dk``. Two things are deliberately different:

* padding a low-dimensional array up to ``ndim`` **raises** instead of warning
  and silently prepending axes. Silently reshaping measured data hides a real
  mismatch between what the caller has and what they think they have; the
  replaced version emitted a warning and continued.
* error messages name the offending value, not only its type.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Union

import numpy as np
import torch
from numpy.typing import NDArray


def ensure_valid_array(
    array: Any,
    dtype: torch.dtype | None = None,
    ndim: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return ``array`` as a detached tensor on ``device``.

    Args:
        array: numpy array, torch tensor, or anything ``torch.as_tensor`` accepts.
        dtype: cast to this dtype if given.
        ndim: require exactly this many dimensions.
        device: device for the result.

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
            tensor = torch.as_tensor(array)
        except Exception as exc:
            raise TypeError(
                f"could not convert {type(array).__name__} to a tensor: {exc}"
            ) from exc

    tensor = tensor.to(device=device)
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


def validate_ndinfo(
    value: Union[NDArray, Sequence[float], float, int],
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
            f"{name} must be a scalar, list, tuple or array; got "
            f"{type(value).__name__}"
        )

    if not np.issubdtype(out.dtype, np.number):
        raise ValueError(f"{name} must be numeric; got dtype {out.dtype}")
    return out


def validate_units(value: Union[str, Sequence[str]], ndim: int) -> List[str]:
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
        raise ValueError(
            f"units has {len(value)} entries but the data has {ndim} axes"
        )
    return [str(unit) for unit in value]
