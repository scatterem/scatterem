"""The float64 total of a cube that ``Dataset4DStem`` records as its dose.

``dataset_4dstem.py`` is frozen size debt, so the sum lives here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = ["array_total", "float64_slice_total", "host_total"]


def float64_slice_total(array: Tensor) -> float:
    """Sum ``array`` in float64, one partial sum per outer slice, on the host.

    Each outer slice is summed in float64 on its device, matching the
    ``total_intensity`` property's own precision strategy. The partial sums
    cross to the host in ONE ``tolist()``, then add in slice order. A
    ``float()`` per slice was one host sync per scan row: 306 on a large
    padded cube, 6 ms of a 90 ms ``pad_and_taper_realspace``.
    Same kernels, same partial sums, same host order, so the total is
    bit-identical to the per-slice loop. The ``+=`` loop is deliberate:
    Python 3.12's ``sum()`` adds floats with compensation, a different result.

    Parameters
    ----------
    array : Tensor
        The cube; its first axis is the outer slice.

    Returns
    -------
    float
        The total, ``0.0`` for an empty first axis.
    """
    partial = [array[i].to(torch.float64).sum() for i in range(array.shape[0])]
    return host_total(torch.stack(partial)) if partial else 0.0


def host_total(partial: Tensor) -> float:
    """Add the float64 partial sums on the host, in order, after ONE read.

    Parameters
    ----------
    partial : Tensor
        The ``(n,)`` float64 partial sums, on any device.

    Returns
    -------
    float
        Their total, added left to right with ``+=``.
    """
    total = 0.0
    for value in partial.tolist():
        total += value
    return total


def array_total(dataset: Any) -> float:
    """The constructor's sum of the stored array, before its division by ``c``.

    The callers that turn the array back into counts use it:
    ``c = array_total(dataset) / sum(dataset.array)``. It is ``total_intensity``,
    except on a crop. A crop keeps the parent's dose and total, but its
    array holds only the cropped counts. The crop keeps its own sum in
    ``_array_total`` (``DetectorOps._carry_record``), in the units of the
    parent's total, also when the parent was normalised. A frame view
    of a source-backed dataset sums its own frames on first use.

    Parameters
    ----------
    dataset : Dataset4DStem
        The dataset.

    Returns
    -------
    float
        The sum, in the units of the array given to ``from_array``.
    """
    own = getattr(dataset, "_array_total", None)
    return float(dataset.total_intensity if own is None else own)
